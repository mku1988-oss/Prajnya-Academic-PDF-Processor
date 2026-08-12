#!/usr/bin/env python3
"""
The cheap first look — what a document says about itself, before any GPU work.

For biology researchers: everything in this module runs on the CPU in about a
second per PDF and never calls a model. It opens a paper, finds its DOI, works
out whether it is an article or a book, and — for a book — reads its title and
chapter count off the contents page.

**Why cheapness is the point.** The difference between a paper and a 345-page
volume is minutes versus hours of GPU time. Knowing which one you have *before*
committing to it is what makes an unattended run safe: the pre-flight pass can
tell you document 17 of 30 is a whole book while it still costs nothing to
decide what to do about it. Finding that out afterwards, having spent two hours
on it, is the failure this design exists to prevent.

Split out of `runner.py` so that deciding is separate from doing: nothing here
writes a file, touches the ledger, or loads a model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .batch import BookDecision, PreflightEntry, decision_for, preflight, resolve_book_decisions
from .classification import classify_document
from .extractors.front_matter import DEFAULT_FRONT_MATTER_PAGES, scan_front_matter
from .models import DocumentProfile, DocumentType
from .pdf import get_backend
from .policy import BookSettings, resolve_book_settings
from .run_request import RunRequest

LOG = logging.getLogger(__name__)


def build_document_profile(pdf_path: Path, backend,
                           front_matter_pages: int = DEFAULT_FRONT_MATTER_PAGES
                           ) -> DocumentProfile:
    """
    First-pass analysis: the cheap facts a document declares about itself.

    Opens the PDF once and scans the front matter — the first
    `front_matter_pages` pages — for DOI, ISBN and ISSN with confidence
    scoring (0.98 labeled / 0.95 URL / 0.85 bare DOI). Page 1 keeps
    precedence, so an article behaves exactly as before and
    `prefer_text_layer_doi` still has its page-1 DOI to trust; the deeper
    pages only matter for documents that print their identifiers on a
    copyright page, which is where books put them (see
    `extractors/front_matter.py`).

    Args:
        pdf_path: the PDF to profile
        backend: a PdfBackend (the licence seam — never `fitz` directly)
        front_matter_pages: how many leading pages to scan for identifiers

    The pages the scan read are then handed to the classifier, so deciding
    the document type costs no extra reads. The type chooses the pipeline —
    see `strategies/router.py`.

    Returns:
        A `DocumentProfile` carrying page 1's text, the identifiers found, the
        page each one came from, and the classification with its evidence.
    """
    with backend.open(pdf_path) as doc:
        page_count = doc.page_count
        page_1_text = doc.page(0).text() or ""
        identifiers = scan_front_matter(doc, max_pages=front_matter_pages,
                                        page_1_text=page_1_text)

    # Classification reuses the pages the scan already read — no second pass.
    outcome = classify_document(identifiers.page_texts, page_count, identifiers)

    return DocumentProfile(
        pdf_path=pdf_path,
        page_count=page_count,
        file_size=pdf_path.stat().st_size,
        page_1_text=page_1_text,
        doi=identifiers.doi,
        doi_confidence=identifiers.doi_confidence,
        isbn=identifiers.isbn,
        isbn_confidence=identifiers.isbn_confidence,
        issn=identifiers.issn,
        identifier_pages=identifiers.identifier_pages(),
        front_matter_pages_scanned=identifiers.pages_scanned,
        document_type=outcome.document_type,
        classification_confidence=outcome.confidence,
        classification_signals=outcome.signals,
    )


def describe_book(pdf_path: Path, backend,
                  front_matter_pages: int = DEFAULT_FRONT_MATTER_PAGES
                  ) -> Dict[str, Any]:
    """
    The title and chapter count, so a prompt can show a book not a filename.

    Reads only the front matter (CPU, milliseconds) — enough to name the thing
    the user is being asked about, and nowhere near enough to be expensive.
    """
    from .strategies.book.front_matter_record import build_book_record
    from .strategies.book.toc import parse_toc_entries

    with backend.open(pdf_path) as doc:
        identifiers = scan_front_matter(doc, max_pages=front_matter_pages)

    record = build_book_record(identifiers.page_texts,
                               identifiers.identifier_pages().get("isbn"))
    chapters = sum(1 for text in identifiers.page_texts.values()
                   for entry in parse_toc_entries(text) if entry.number is not None)

    return {"title": record.title or pdf_path.stem, "chapter_count": chapters}


def run_preflight(request: RunRequest, backend=None,
                  logger: Optional[logging.Logger] = None) -> List[PreflightEntry]:
    """
    Classify every PDF on the CPU, before any of them costs GPU time.

    For biology researchers: this is the cheap look-first pass. It opens each
    PDF, works out what it is, and (for a book) reads its title and chapter
    count — about a second each, no model involved.

    Args:
        request: the run's order form; `request.paths` are the PDFs.
        backend: an open PDF backend, or None to build the requested one.
        logger: optional logger.

    Returns:
        One `PreflightEntry` per path, in order. A PDF that cannot be opened
        gets an entry with `error` set rather than aborting the pass.
    """
    log = logger or LOG
    backend = backend if backend is not None else get_backend(request.backend)

    return preflight(
        request.paths,
        lambda path: build_document_profile(path, backend,
                                            request.front_matter_pages),
        book_describer=lambda path: describe_book(path, backend,
                                                  request.front_matter_pages),
        logger=log)


def resolve_decisions(entries: Sequence[PreflightEntry], request: RunRequest,
                      prompt=None,
                      logger: Optional[logging.Logger] = None) -> BookSettings:
    """
    Settle what happens to every book-like document, before the run starts.

    The policy says what to do; `prompt` says whether there is a way to ask.
    Those are separate on purpose: a policy of `ask` with no prompt handler —
    an unattended run, or a GUI that has not supplied its dialog — resolves to
    the cheap default rather than hanging or overspending.

    Args:
        entries: the pre-flight entries; mutated in place with decisions.
        request: the run's order form.
        prompt: asks the user about one document, or None for no way to ask.
            The CLI passes a terminal prompt, a GUI passes a dialog.
        logger: optional logger.

    Returns:
        The resolved settings, so the caller can report what decided them.
    """
    log = logger or LOG
    settings = settings_for(request)
    resolve_book_decisions(entries, settings, prompt=prompt, logger=log)
    return settings


def decide_for_single(profile: DocumentProfile, pdf_path: Path,
                      request: RunRequest, prompt=None, backend=None,
                      logger: Optional[logging.Logger] = None) -> BookDecision:
    """
    Settle what to do with one document, by the same policy a batch uses.

    Single-file and folder runs must not diverge here — a book should behave
    the same whether it arrived alone or as document 17 of 30.
    """
    log = logger or LOG
    if profile.document_type not in (DocumentType.BOOK, DocumentType.BOOK_CHAPTER):
        return BookDecision.safe_default()

    settings = settings_for(request)
    entry = PreflightEntry(path=pdf_path, document_type=profile.document_type,
                           confidence=profile.classification_confidence or 0.0,
                           page_count=profile.page_count)

    if entry.is_implausible_chapter(settings.implausible_chapter_pages):
        log.warning("%s was classified book_chapter but has %d pages — likely a "
                    "misclassified book. Not reading its bodies.",
                    pdf_path.name, profile.page_count)
        return BookDecision.safe_default()

    if settings.wants_prompt() and prompt is not None:
        entry.title, entry.chapter_count = _book_details(pdf_path, request,
                                                         backend, log)
        return prompt(entry)

    return decision_for(settings)


def settings_for(request: RunRequest) -> BookSettings:
    """The book policy this run will use, from config plus the run's overrides."""
    return resolve_book_settings(request.config_root,
                                 cli_chapters=request.chapters,
                                 cli_max_pages=request.max_pages,
                                 noninteractive=request.noninteractive,
                                 overrides=request.book_overrides)


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _book_details(pdf_path: Path, request: RunRequest, backend,
                  log: logging.Logger) -> tuple:
    """Title and chapter count for the prompt; cosmetic, so failure is quiet."""
    try:
        backend = backend if backend is not None else get_backend(request.backend)
        details = describe_book(pdf_path, backend, request.front_matter_pages)
        return details.get("title"), details.get("chapter_count") or 0
    except Exception as error:                                # pragma: no cover
        log.debug("Could not describe %s: %s", pdf_path.name, error)
        return None, 0
