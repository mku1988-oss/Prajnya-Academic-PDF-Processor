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
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .batch import BookDecision, PreflightEntry
from .cli_support import already_processed
from .document_json import enum_value, write_result
from .events import emit
from .ledger.semantic_ledger import LedgerRecord
from .models import DocumentProfile
from .paths import DEFAULT_LEDGER_PATH
from .pdf import get_backend

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

LOG = logging.getLogger(__name__)

#: How one document gets processed. The default runs it in this process; a
#: later step will supply one that runs it in a child process instead, so a
#: document that hangs can be killed without losing the batch. Substituting
#: this is the whole reason `process_batch` takes it as a parameter.
Execute = Callable[[PreflightEntry, RunRequest], DocumentOutcome]


def process_one(entry: PreflightEntry, request: RunRequest, backend=None,
                logger: Optional[logging.Logger] = None,
                profile: Optional[DocumentProfile] = None) -> DocumentOutcome:
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
                                  output_root=request.out_root)
        log.info("Routing to %s", type(strategy).__name__)
        profile = strategy.process(profile, test_pages_override=request.pages)

        elapsed = time.time() - started
        out_path = request.output_for(entry.path)
        write_result(profile, out_path, elapsed)
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
    run = execute or (lambda item, req: process_one(item, req, backend=backend,
                                                    logger=log))
    results: Dict[str, Any] = {}

    for position, entry in enumerate(entries, start=1):
        if entry.error:
            results[str(entry.path)] = {"error": entry.error}
            continue

        if not request.force and already_processed(entry.path, ledger, log):
            results[str(entry.path)] = {"skipped": "already in the ledger"}
            continue

        emit(log, "document_start",
             message=f"[{position}/{len(entries)}] {entry.path.name}",
             path=str(entry.path), position=position, total=len(entries))
        outcome = run(entry, request)
        results[str(entry.path)] = _result_for(outcome, ledger, log)

    emit(log, "batch_done",
         message=f"Batch done: {len(entries)} document(s)",
         total=len(entries))
    return results


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
        ledger_path: an explicit path, or None for `SARA_LEDGER_PATH` and then
            the default.

    Returns:
        A `CsvLedgerBackend`, or None on any failure. Failing to open the
        bookkeeping file must never end a run that is otherwise fine.
    """
    try:
        from .ledger.csv_backend import CsvLedgerBackend

        path = Path(ledger_path or os.getenv("SARA_LEDGER_PATH",
                                             DEFAULT_LEDGER_PATH))
        if not create and not path.is_file():
            return None
        return CsvLedgerBackend(file_path=str(path.resolve()))
    except Exception as error:
        log.debug("No ledger available (%s); continuing without it.", error)
        return None


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _log_profile(profile: DocumentProfile, log: logging.Logger) -> None:
    """The cheap facts, announced before any GPU work is paid for."""
    log.info("Profile: %d pages, DOI=%s (confidence %.2f), ISBN=%s, ISSN=%s, found on %s",
             profile.page_count, profile.doi, profile.doi_confidence,
             profile.isbn, profile.issn, profile.identifier_pages or "page 1 only")
    log.info("Classified as %s (confidence %.2f)",
             enum_value(profile.document_type),
             profile.classification_confidence or 0.0)


def _outcome_from(pdf_path: Path, profile: DocumentProfile, out_path: Path,
                  elapsed: float) -> DocumentOutcome:
    """The finished document as plain data — see `DocumentOutcome`."""
    result = profile.processing_result
    metadata = result.metadata.model_dump() if result and result.metadata else {}
    return DocumentOutcome(
        path=pdf_path,
        output=out_path,
        elapsed=elapsed,
        metadata=metadata,
        document_type=profile.document_type,
        errors=list(result.errors) if result and result.errors else [],
    )


def _result_for(outcome: DocumentOutcome, ledger,
                log: logging.Logger) -> Dict[str, Any]:
    """
    One document's line in the batch results, and its ledger row.

    A failed document gets no ledger record, deliberately: a half-processed
    document must not be marked "already processed" and skipped next run.

    The bookkeeping is inside the same error isolation as the work. A ledger
    that cannot be written is worth one failed line in the summary, never the
    remaining documents of a batch.
    """
    if outcome.error:
        return {"error": outcome.error, "elapsed": round(outcome.elapsed, 1)}

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
            "review_status": record.review_status,
            "requires_review": record.requires_review}
