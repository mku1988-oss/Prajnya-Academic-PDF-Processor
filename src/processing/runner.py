#!/usr/bin/env python3
"""
The shared work path — "process a document", written once.

For biology researchers: this module is what actually runs a paper through the
pipeline. The command line calls it; the graphical interface will call it; a
separate worker process will call it. All three do the same work because it is
the *same code*, not three copies that happen to agree today.

**Why this exists.** All of this used to live inside `run_prajnya.py`, the
command-line entrypoint, which had grown to 727 lines. That was fine while the
command line was the only way in. It stops being fine the moment a second front
end appears: either the GUI imports the entrypoint (which drags argparse and
terminal prompts into a web server), or it re-implements the work (and the two
quietly diverge). Neither is acceptable, so the work moved down here and the
entrypoint became a thin wrapper around it.

**What stayed in the entrypoint:** argparse, logging setup, exit codes, and the
terminal prompts. Nothing in this module reads from a terminal or writes to
one. Asking a question is a *capability the caller supplies* — the CLI passes a
terminal prompt, the GUI will pass a dialog, an unattended run passes nothing
and the cheap default is taken.

**Where the work happens, once processing moves into child processes.** The
parent always owns the pre-flight pass, the book decisions, the batch loop and
**every ledger write**; a child owns `process_one` and nothing else. The ledger
is single-writer deliberately: `CsvLedgerBackend` loads the whole file and
rewrites it on commit, so two writers is a lost-update race.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .batch import BookDecision, PreflightEntry
from .citations.level import resolve_level
from .citations.services import reference_finder_for
from .env_settings import env_setting
from .cli_support import already_processed
from .config.loader import ConfigLoader
from .document_json import enum_value, write_result
from .document_status import FAILED, SUCCESS, failed_page_numbers
from .events import emit
from .ledger.semantic_ledger import LedgerRecord
from .models import DocumentProfile, DocumentType
from .paths import DEFAULT_LEDGER_PATH
from .pdf import get_backend
from .run_provenance import run_provenance

# The cheap CPU inspection half lives next door — deciding is separate from
# doing. Re-exported here so a caller has ONE import site for "run a document",
# which is the whole point of this seam.
from .profiling import (             # noqa: F401  (re-exported on purpose)
    build_document_profile,
    decide_for_single,
    describe_book,
    resolve_decisions,
    run_preflight,
    settings_for,
)
from .review import (
    assess_metadata_completeness,
    build_ledger_record,
    describe_rule,
    regenerate_document_json,
    save_record,
)
from .run_request import DocumentOutcome, RunRequest
from .strategies.router import build_strategy
from .title_page_store import SavedTitlePage, TitlePageStore

#: Documents per chunk when neither `--chunk-size` nor the config says (T39).
DEFAULT_CHUNK_SIZE = 25

LOG = logging.getLogger(__name__)

#: How one document gets processed. The default runs it in this process; a
#: later step will supply one that runs it in a child process instead, so a
#: document that hangs can be killed without losing the batch. Substituting
#: this is the whole reason `process_batch` takes it as a parameter.
Execute = Callable[[PreflightEntry, RunRequest], DocumentOutcome]


def process_one(entry: PreflightEntry, request: RunRequest, backend=None,
                logger: Optional[logging.Logger] = None,
                profile: Optional[DocumentProfile] = None,
                title_pages: Optional[TitlePageStore] = None) -> DocumentOutcome:
    """
    Process ONE document, end to end. The unit of work.

    For biology researchers: this is everything that happens to a single paper
    — read its identifiers, decide what kind of document it is, route it to the
    right pipeline, run it, and write the result as JSON.

    It touches no ledger. That is deliberate: this function is what will run in
    a child process, and the bookkeeping file must have exactly one writer.

    Args:
        entry: the pre-flighted document, carrying its book decision. A single
            PDF run builds a one-entry list, so both shapes use this same path.
        request: the run's order form.
        backend: an open PDF backend, or None to build the requested one.
        logger: optional logger.
        profile: an already-built `DocumentProfile`, when the caller had to
            build one to decide what the document was. Passing it in avoids
            re-opening the PDF and re-scanning its front matter for an answer
            already known. None means build it here — which is what a worker
            in a separate process must do, since a profile does not travel.
        title_pages: the batch's saved title pages (T39), or None. When it
            holds a reading that matches this PDF and these settings, the
            title model is not called again; otherwise the title page is read
            here, as a single document always was.

    Returns:
        A `DocumentOutcome`. A document that raises comes back with `error`
        set rather than propagating — one bad PDF must never end a batch.
    """
    log = logger or LOG
    backend = backend if backend is not None else get_backend(request.backend)
    started = time.time()

    try:
        if profile is None:
            profile = build_document_profile(entry.path, backend,
                                             request.front_matter_pages)
        _log_profile(profile, log)

        decision = entry.decision or BookDecision.safe_default()
        strategy = build_strategy(profile.document_type, request.config_root,
                                  pdf_backend=backend,
                                  chapters=decision.chapters,
                                  max_body_pages=decision.max_pages,
                                  refresh_metadata=request.force,
                                  output_root=request.out_root,
                                  citation_level=request.citations)
        log.info("Routing to %s", type(strategy).__name__)
        saved = _saved_title_page(title_pages, entry.path, strategy)
        if saved is not None:
            profile = strategy.process(profile, test_pages_override=request.pages,
                                       title_page=saved.page)
        else:
            profile = strategy.process(profile, test_pages_override=request.pages)
        _stamp_citation_level(profile, request, log)

        # A batch's title phase is part of this document's cost.
        elapsed = time.time() - started + (saved.elapsed if saved else 0.0)
        out_path = request.output_for(entry.path)
        write_result(profile, out_path, elapsed,
                     provenance=run_provenance(request.config_root,
                                                request.config_overrides()))  # contract 1.7
        emit(log, "document_done", stage="write",
             message=f"Done in {elapsed:.1f}s -> {out_path}",
             path=str(entry.path), elapsed=elapsed, output=str(out_path))

        return _outcome_from(entry.path, profile, out_path, elapsed)

    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        log.error("  failed: %s", message, exc_info=request.verbose)
        emit(log, "document_failed", level=logging.ERROR,
             message=f"  failed: {message}",
             path=str(entry.path), elapsed=round(time.time() - started, 1),
             error=message)
        return DocumentOutcome(path=entry.path, error=message,
                               elapsed=round(time.time() - started, 1))


def process_batch(entries: Sequence[PreflightEntry], request: RunRequest,
                  ledger, execute: Optional[Execute] = None,
                  logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Run every pre-flighted document. One failure never ends the batch.

    For biology researchers: this is the loop. It skips documents the ledger
    already holds (unless `--force`), records each result, and keeps going past
    a document that fails — because losing twenty-nine good results to one bad
    PDF is the worse outcome.

    It never prompts. Anything needing a human is flagged as it goes and asked
    about once, at the end, by the caller — a run that stops halfway to ask a
    question is a run nobody can leave alone.

    Args:
        entries: the pre-flight entries, with book decisions already settled.
        request: the run's order form.
        ledger: the ledger, or None. Written here and only here.
        execute: how to process one document; defaults to doing it in this
            process. See `Execute`.
        logger: optional logger.

    Returns:
        One entry per document keyed by its path as a string, carrying either
        `error`, `skipped`, or the elapsed time, output path and review status.
    """
    log = logger or LOG
    backend = get_backend(request.backend)
    results: Dict[str, Any] = {}
    run_level = _run_citation_level(request, log)

    if execute is not None:
        # Prajnya Studio: one child process per document, so a Stop can kill
        # the one running. Not batched by model (T39 is CLI only; Studio
        # becomes the CLI's front-end in v7).
        _process_each(entries, request, ledger, execute, results, run_level, log)
    else:
        _process_by_model(entries, request, ledger, backend, results, run_level, log)

    _warn_about_other_levels(results, run_level, log)
    emit(log, "batch_done",
         message=f"Batch done: {len(entries)} document(s)",
         total=len(entries))
    return results


def read_title_pages(entries: Sequence[PreflightEntry], request: RunRequest,
                     backend, store: TitlePageStore,
                     log: Optional[logging.Logger] = None) -> None:
    """
    Phase 1 of a batch by model (T39): read the title page of every document.

    For biology researchers: the title model (olmocr, ~5.5 GB) and the body
    model (GLM, ~3 GB) cannot share the 8 GB GPU, and each swap costs ~14 s
    (median of 50 benchmark papers, eval/batch_by_model/GATE.md). Reading
    every title page of a chunk first, then every body, swaps twice per
    chunk instead of twice per paper.

    Each reading goes to disk (`TitlePageStore`), not memory, so a large
    corpus costs no extra memory, and a crash loses no reading already paid
    for. A document already read in an interrupted earlier run is not read
    again.

    Nothing here can fail a document. A title page that cannot be read in
    this phase is simply read again in phase 2, the ordinary way.

    Args:
        entries: the documents to process now (errors and ledger skips
            already removed).
        request: the run's order form.
        backend: an open PDF backend.
        store: where the readings go.
        log: optional logger.
    """
    log = log or LOG
    reader = None
    log.info("Title phase: %d document(s); the title model is loaded once",
             len(entries))
    for position, entry in enumerate(entries, start=1):
        started = time.time()
        try:
            profile = build_document_profile(entry.path, backend,
                                             request.front_matter_pages)
            if profile.document_type is DocumentType.BOOK:
                continue           # books keep their own per-document route
            reader = reader or _title_reader(request, backend, log)
            key = reader.title_settings_key()
            if store.load(entry.path, key) is not None:
                log.info("[%d/%d] title page of %s: already read (saved)",
                         position, len(entries), entry.path.name)
                continue
            log.info("[%d/%d] title page of %s", position, len(entries),
                     entry.path.name)
            page = reader.read_title_page(profile)
            if page is not None:   # None: the cloud path, which is not batched
                store.save(entry.path, key, page, time.time() - started)
        except Exception as error:
            log.warning("Title phase could not read %s (%s: %s); it is read "
                        "with its body instead.", entry.path.name,
                        type(error).__name__, error)


def record_outcome(pdf_path: Path, metadata: Dict[str, Any], document_type: Any,
                   out_path: Optional[Path], ledger,
                   log: Optional[logging.Logger] = None) -> LedgerRecord:
    """
    Assess a processed document, store it in the ledger, and say if it needs a human.

    For biology researchers: this is the step that turns a run into a library
    entry. It checks the fields a citation needs — for a paper, title, authors,
    DOI and journal — writes the record under the file's content hash, and
    flags it if anything is missing.

    ⚠️ `metadata` is a DERIVED value, and this project has been burned three
    times by a derived value that was stored and then outlived the code that
    derived it (the Zotero record cache, and the ledger row `--force` could not
    bypass). The standing rule is **cache facts, derive on read**. Passing
    metadata in is safe here only because it is produced once, at the end of
    the work, and never stored or replayed. If a resume, a retry or a cache is
    ever added between the work and this call, re-derive from the document JSON
    instead of trusting what was handed over.

    Args:
        pdf_path: the source PDF.
        metadata: the merged bibliographic record, as a plain dict.
        document_type: what the document was classified as.
        out_path: the JSON that was written for it, or None.
        ledger: the ledger, or None.
        log: logger.

    Returns:
        The stored `LedgerRecord`. Its `requires_review` says whether a human
        is needed; the caller decides *when* to ask.
    """
    log = log or LOG

    # Judged by what the document IS: a paper needs a journal, a book needs
    # editors and a publisher. See `COMPLETENESS_RULES`.
    assessment = assess_metadata_completeness(metadata, document_type)
    record = build_ledger_record(pdf_path, metadata, assessment,
                                 document_type=document_type)
    save_record(ledger, record, log)

    if assessment.requires_review:
        log.warning("%s needs metadata review: %s (as a %s, which needs %s)",
                    pdf_path.name, assessment.describe(),
                    enum_value(document_type) or "document",
                    describe_rule(document_type))
    else:
        log.info("%s: metadata complete.", pdf_path.name)

    # The output JSON carries the same verdict, so a downstream reader can see
    # that a record is provisional without consulting the ledger.
    if out_path is not None:
        regenerate_document_json(out_path, record, log)
    return record


def record_document(pdf_path: Path, profile: DocumentProfile, out_path: Path,
                    ledger, log: logging.Logger) -> LedgerRecord:
    """
    Record a document from its processed profile. See `record_outcome`.

    Kept because a profile is what an in-process caller has to hand. It reads
    the metadata off the profile and delegates — there is one implementation of
    "what does the ledger say about this document", not two.
    """
    result = profile.processing_result
    metadata = result.metadata.model_dump() if result and result.metadata else {}
    return record_outcome(pdf_path, metadata, profile.document_type, out_path,
                          ledger, log)


def open_ledger(log: logging.Logger, create: bool = True,
                ledger_path: Optional[Path] = None):
    """
    The ledger — the library's permanent record book.

    It is both read (has this file been processed?) and written (here is what
    we found, and whether it needs a human). Opening it creates the file when
    it does not exist, because a first run must still be able to record what
    it did.

    Args:
        log: logger.
        create: when False, an absent ledger is reported as None rather than
            created — what `--review` wants, since reviewing a ledger we just
            invented is meaningless.
        ledger_path: an explicit path, or None for `PRAJNYA_LEDGER_PATH` and then
            the default.

    Returns:
        A `CsvLedgerBackend`, or None on any failure. Failing to open the
        bookkeeping file must never end a run that is otherwise fine.
    """
    try:
        from .ledger.csv_backend import CsvLedgerBackend

        path = Path(ledger_path or env_setting("LEDGER_PATH", DEFAULT_LEDGER_PATH))
        if not create and not path.is_file():
            return None
        return CsvLedgerBackend(file_path=str(path.resolve()))
    except Exception as error:
        log.debug("No ledger available (%s); continuing without it.", error)
        return None


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _process_each(entries: Sequence[PreflightEntry], request: RunRequest, ledger,
                  execute: Execute, results: Dict[str, Any],
                  run_level: Optional[str], log: logging.Logger) -> None:
    """One document after another, each through `execute` (Prajnya Studio)."""
    for position, entry in enumerate(entries, start=1):
        if not _needs_processing(entry, request, ledger, results, run_level, log):
            continue
        _announce(entry, position, len(entries), log)
        outcome = execute(entry, request)
        results[str(entry.path)] = _result_for(outcome, ledger, log)


def _process_by_model(entries: Sequence[PreflightEntry], request: RunRequest,
                      ledger, backend, results: Dict[str, Any],
                      run_level: Optional[str], log: logging.Logger) -> None:
    """
    The command line's batch (T39): chunk by chunk, title phase then body phase.

    The saved title pages are removed only after the WHOLE run, so they can be
    inspected while it runs, and survive a crash for the re-run to reuse.
    """
    store = TitlePageStore(request.out_root, logger=log)
    size = _chunk_size(request, log)
    numbered = list(enumerate(entries, start=1))

    for first in range(0, len(numbered), size):
        chunk = numbered[first:first + size]
        to_run = [(position, entry) for position, entry in chunk
                  if _needs_processing(entry, request, ledger, results, run_level, log)]
        if not to_run:
            continue
        log.info("Chunk of %d document(s): title phase, then body phase", len(to_run))
        read_title_pages([entry for _, entry in to_run], request, backend, store, log)

        for position, entry in to_run:
            _announce(entry, position, len(entries), log)
            outcome = process_one(entry, request, backend=backend, logger=log,
                                  title_pages=store)
            results[str(entry.path)] = _result_for(outcome, ledger, log)

    store.clear()


def _needs_processing(entry: PreflightEntry, request: RunRequest, ledger,
                      results: Dict[str, Any], run_level: Optional[str],
                      log: logging.Logger) -> bool:
    """
    False for a pre-flight error or a ledger skip, whose line is written now.

    A document to process gets a placeholder line at once, so the results
    keep the order of the pre-flight list whatever order the work runs in.
    """
    key = str(entry.path)
    if entry.error:
        results[key] = {"error": entry.error}
        return False
    if not request.force and already_processed(entry.path, ledger, log):
        results[key] = _skipped_result(entry.path, ledger, run_level, log)
        return False
    results[key] = {}
    return True


def _announce(entry: PreflightEntry, position: int, total: int,
              log: logging.Logger) -> None:
    """The `document_start` event a front-end follows progress by."""
    emit(log, "document_start",
         message=f"[{position}/{total}] {entry.path.name}",
         path=str(entry.path), position=position, total=total)


def _chunk_size(request: RunRequest, log: logging.Logger) -> int:
    """
    Documents per chunk: `--chunk-size`, else `batching.chunk_size`, else 25.

    A bigger chunk swaps models less often; a smaller one writes its first
    results sooner. 1 is title-then-body per document, as before T39.
    """
    size = request.chunk_size
    if size is None:
        try:
            config = ConfigLoader(request.config_root, logger=log).strategy()
            size = (config.get("batching") or {}).get("chunk_size")
        except Exception:
            size = None
    size = max(1, int(size or DEFAULT_CHUNK_SIZE))
    log.info("Batch by model: chunks of %d document(s)", size)
    return size


def _title_reader(request: RunRequest, backend, log: logging.Logger):
    """
    The strategy that reads title pages in phase 1.

    One per chunk, built as `process_one` builds an article's: a book
    chapter reads its title page with this same article reader, and a
    saved reading is checked against the document's own strategy's
    settings in phase 2, so a mismatch is read again rather than trusted.
    """
    return build_strategy(DocumentType.RESEARCH_ARTICLE, request.config_root,
                          pdf_backend=backend,
                          refresh_metadata=request.force,
                          output_root=request.out_root,
                          citation_level=request.citations)


def _saved_title_page(store: Optional[TitlePageStore], pdf_path: Path,
                      strategy) -> Optional[SavedTitlePage]:
    """The batch's reading of this title page, when there is one to trust."""
    if store is None or not hasattr(strategy, "title_settings_key"):
        return None
    return store.load(pdf_path, strategy.title_settings_key())


def _log_profile(profile: DocumentProfile, log: logging.Logger) -> None:
    """The cheap facts, announced before any GPU work is paid for."""
    log.info("Profile: %d pages, DOI=%s (confidence %.2f), ISBN=%s, ISSN=%s, found on %s",
             profile.page_count, profile.doi, profile.doi_confidence,
             profile.isbn, profile.issn, profile.identifier_pages or "page 1 only")
    log.info("Classified as %s (confidence %.2f)",
             enum_value(profile.document_type),
             profile.classification_confidence or 0.0)


def _run_citation_level(request: RunRequest, log: logging.Logger) -> Optional[str]:
    """The citation level this run uses, or None when it cannot be read."""
    try:
        config = ConfigLoader(request.config_root, logger=log).strategy()
        return resolve_level(config, request.citations)[0]
    except Exception:
        return None


def _skipped_result(pdf_path: Path, ledger, run_level: Optional[str],
                    log: logging.Logger) -> Dict[str, Any]:
    """
    A skipped document's line in the results, saying so when its level differs.

    For biology researchers: a document already in the ledger is not
    processed again. But if it was processed at ANOTHER citation level than
    this run asks for, skipping it silently would leave you believing your
    whole library is now at the new level. So the difference is reported,
    with the remedy: `--recite` changes the level without re-processing.
    """
    stored = _stored_citation_level(pdf_path, ledger)
    if run_level is None or stored == run_level:
        return {"skipped": "already in the ledger"}

    was = f"citation level {stored}" if stored else "a citation level that was not recorded"
    log.info("  %s was processed at %s; this run uses %s. --recite changes the "
             "level without re-processing.", pdf_path.name, was, run_level)
    return {"skipped": f"already in the ledger, at {was} (this run: {run_level})",
            "citation_level_differs": True}


def _stored_citation_level(pdf_path: Path, ledger) -> Optional[str]:
    """The level on this document's ledger row, or None (also on any failure)."""
    try:
        from .ledger.semantic_ledger import generate_hash

        record = ledger.get_record(generate_hash(str(pdf_path)))
        return getattr(record, "citation_level", None)
    except Exception:
        return None


def _warn_about_other_levels(results: Dict[str, Any], run_level: Optional[str],
                             log: logging.Logger) -> None:
    """One summary line when skipped documents sit at another citation level."""
    count = sum(1 for value in results.values() if value.get("citation_level_differs"))
    if count:
        log.warning("%d document(s) were skipped but were processed at another citation "
                    "level than this run's (%s). Run again with --recite to bring them to "
                    "it: no PDF is read and no model is called.", count, run_level)


def _stamp_citation_level(profile: DocumentProfile, request: RunRequest,
                          log: logging.Logger) -> None:
    """
    Record on the result which citation level this run used, and who chose it.

    For biology researchers: the level says how the paper's bibliographic
    record was established (Prajnya alone, your Zotero library, or online
    registries). Every output file states it, so a program reading the file
    knows how far to trust the record and whether a reference list could be
    present at all.

    It is stamped HERE, once, rather than inside each strategy, so that
    articles, chapters and books all get it from the same reading of the
    same config. A config that cannot be read, or that names a level that
    does not exist, is logged and the level is left unset (written as the
    default): the document's extracted content is already paid for and must
    still be written.
    """
    result = profile.processing_result
    if result is None:
        return
    try:
        config = ConfigLoader(request.config_root, logger=log).strategy()
        level, set_by = resolve_level(config, request.citations)
    except Exception as error:
        log.error("Could not read the citation level (%s); the document is "
                  "written with the default level.", error)
        return
    result.citation_level = level
    result.citation_level_set_by = set_by
    # The ledger row is built from the metadata, so the level travels there
    # too. An article's enrichment has already set it; a book's has not.
    if result.metadata is not None and not result.metadata.citation_level:
        result.metadata.citation_level = level
    log.info("Citation level: %s (set by %s)", level, set_by)
    _attach_reference_list(profile, config, level, log)


def _attach_reference_list(profile: DocumentProfile, config: Dict[str, Any],
                           level: str, log: logging.Logger) -> None:
    """
    At level `online`, fetch the paper's own reference list and keep it on the result.

    For biology researchers: the list is what lets a program follow a
    paper's citations. It is looked up from the paper's DOI (Crossref, then
    OpenAlex) and cross-checked against the list read from the PDF itself.

    BOOKS ARE SKIPPED. A book has one reference list per chapter, and the
    finder reads the PDF's last list, so it would hand back one chapter's
    list as if it were the whole book's.

    A reference list must never cost a document its output: any failure is
    logged as an error and the document is written without a list.

    Args:
        profile: the processed document (its result receives the list).
        config: the loaded strategy config (`citations.online` settings).
        level: the citation level in force.
        log: the run's logger.
    """
    result = profile.processing_result
    if result is None or result.metadata is None or level != "online":
        return
    if enum_value(profile.document_type) == "book":
        return
    try:
        finder = reference_finder_for(level, config, logger=log)
        if finder is None:
            return
        result.citation_references = finder.find(result.metadata.doi, profile.pdf_path)
    except Exception as error:
        log.error("Could not fetch the reference list (%s); the document is "
                  "written without one.", error)
        result.citation_references = None


def _outcome_from(pdf_path: Path, profile: DocumentProfile, out_path: Path,
                  elapsed: float) -> DocumentOutcome:
    """The finished document as plain data — see `DocumentOutcome`."""
    result = profile.processing_result
    metadata = result.metadata.model_dump() if result and result.metadata else {}
    pages = list(result.pages) if result and result.pages else []
    return DocumentOutcome(
        path=pdf_path,
        output=out_path,
        elapsed=elapsed,
        metadata=metadata,
        document_type=profile.document_type,
        errors=list(result.errors) if result and result.errors else [],
        failed_pages=failed_page_numbers(pages),
        page_count=len(pages),
    )


def _result_for(outcome: DocumentOutcome, ledger,
                log: logging.Logger) -> Dict[str, Any]:
    """
    One document's line in the batch results, and its ledger row.

    A failed document gets no ledger record, deliberately: a half-processed
    document must not be marked "already processed" and skipped next run.
    That includes a document that was written but is not a "success" (audit
    D1): one whose strategy recorded an error, or that lost pages. Its line
    says so ("status", "failed_pages") and the next run tries it again.

    The bookkeeping is inside the same error isolation as the work. A ledger
    that cannot be written is worth one failed line in the summary, never the
    remaining documents of a batch.
    """
    if outcome.error:
        return {"error": outcome.error, "elapsed": round(outcome.elapsed, 1),
                "status": FAILED}
    if outcome.status != SUCCESS:
        return _unfinished_result(outcome, log)

    try:
        record = record_outcome(outcome.path, outcome.metadata,
                                outcome.document_type, outcome.output, ledger, log)
    except Exception as error:                                # pragma: no cover
        message = f"{type(error).__name__}: {error}"
        log.error("  failed to record %s: %s", outcome.path.name, message)
        return {"error": message, "elapsed": round(outcome.elapsed, 1),
                "output": str(outcome.output)}

    return {"elapsed": round(outcome.elapsed, 1),
            "output": str(outcome.output),
            "status": SUCCESS,
            "review_status": record.review_status,
            "requires_review": record.requires_review}


def _unfinished_result(outcome: DocumentOutcome, log: logging.Logger) -> Dict[str, Any]:
    """
    The batch line of a document that was written but did not fully succeed.

    No ledger record is written (see `_result_for`), and the log says why, so
    the retry on the next run is never a surprise. The output file is kept:
    it is the best reading there is, and its `document_info.status` says it
    is incomplete.
    """
    if outcome.failed_pages:
        pages = ", ".join(str(number) for number in outcome.failed_pages)
        reason = f"page(s) {pages} failed"
    else:
        reason = "; ".join(outcome.errors) or "no page came out"
    log.warning("  %s is %s (%s): not recorded in the ledger, so the next run "
                "tries it again.", outcome.path.name, outcome.status, reason)
    return {"elapsed": round(outcome.elapsed, 1),
            "output": str(outcome.output) if outcome.output else None,
            "status": outcome.status,
            "failed_pages": list(outcome.failed_pages),
            "errors": list(outcome.errors)}
