#!/usr/bin/env python3
"""
Batch processing — a folder of PDFs, with the expensive decisions taken first.

For biology researchers: this runs a whole folder rather than one paper at a
time. The order it does things in matters more than it looks.

**The pre-flight pass.** Before any GPU work, every PDF is opened and
classified on the CPU (~1s each). That is cheap, and it means the run can tell
you *up front* that document 17 of 30 is a 345-page book — while it still costs
nothing to decide what to do about it. Finding that out afterwards, having
spent two hours on it, is the failure this design exists to prevent.

**Decisions before work, then no interruptions.** All the questions are asked
in the pre-flight pass, together. The processing loop that follows never
prompts, so a batch can be started and left. This is V5's rule — its
`run_folder` forced `noninteractive=True` for every document and deferred
prompting to the end — with one change: a book's cost has to be decided
*before* it is read, not after, so the decision moves to the front instead.

**One failure does not end the run.** A document that raises is recorded in the
summary and the batch continues, because the alternative is losing twenty-nine
good results to one bad PDF.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .models import DocumentProfile, DocumentType
from .policy import BookSettings

LOG = logging.getLogger(__name__)

#: A "chapter" longer than this is almost certainly a misclassified book. See
#: `BookDecision.for_profile` — the guard exists because the book path's
#: opt-in protection does not cover a document classified as a chapter.
IMPLAUSIBLE_CHAPTER_PAGES = 80


@dataclass
class BookDecision:
    """
    What to do with one book-like document, decided before it is read.

    Attributes:
        chapters: chapter numbers to read; None reads no bodies (the safe
            default), an empty-but-present list means "every chapter".
        max_pages: ceiling on body pages read.
        reason: why this decision was taken, for the run summary.
    """
    chapters: Optional[List[int]] = None
    max_pages: Optional[int] = None
    reason: str = "default: front matter and chapter map only"

    @classmethod
    def safe_default(cls) -> "BookDecision":
        """Read no bodies. What `--noninteractive` always takes."""
        return cls()


@dataclass
class PreflightEntry:
    """One PDF as the pre-flight pass sees it: what it is, and what it will cost."""
    path: Path
    document_type: Optional[DocumentType] = None
    confidence: float = 0.0
    page_count: int = 0
    title: Optional[str] = None
    chapter_count: int = 0
    decision: Optional[BookDecision] = None
    error: Optional[str] = None

    @property
    def is_book_like(self) -> bool:
        return self.document_type in (DocumentType.BOOK, DocumentType.BOOK_CHAPTER)

    def is_implausible_chapter(self, threshold: int = IMPLAUSIBLE_CHAPTER_PAGES) -> bool:
        """A 'chapter' of hundreds of pages is a book that scored wrongly."""
        return (self.document_type is DocumentType.BOOK_CHAPTER
                and self.page_count > threshold)

    @property
    def looks_misclassified(self) -> bool:
        """`is_implausible_chapter` at the built-in threshold, for display."""
        return self.is_implausible_chapter()

    def describe(self) -> str:
        """One line for the pre-flight table."""
        kind = self.document_type.value if self.document_type else "unreadable"
        detail = f"{self.page_count}p"
        if self.chapter_count:
            detail += f", {self.chapter_count} chapters"
        return f"{self.path.name[:52]:54s} {kind:16s} {detail}"


def find_pdfs(folder: Path, recursive: bool = False) -> List[Path]:
    """Every PDF in a folder, sorted so runs are reproducible."""
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def preflight(paths: Sequence[Path], profile_builder: Callable[[Path], DocumentProfile],
              book_describer: Optional[Callable[[Path], Dict[str, Any]]] = None,
              logger: Optional[logging.Logger] = None) -> List[PreflightEntry]:
    """
    Classify every PDF before any of them is read.

    Args:
        paths: the PDFs to inspect.
        profile_builder: builds a `DocumentProfile` (the CPU first pass).
        book_describer: optional; for a book-like document, returns `title`
            and `chapter_count` so the prompt can show a book rather than a
            filename. Injected rather than imported so this module never
            touches a PDF and stays testable without one.
        logger: optional logger.

    Returns:
        One `PreflightEntry` per path, in order. A PDF that cannot be opened
        gets an entry with `error` set rather than aborting the pass.
    """
    log = logger or LOG
    entries: List[PreflightEntry] = []

    for path in paths:
        entry = PreflightEntry(path=path)
        try:
            profile = profile_builder(path)
            entry.document_type = profile.document_type
            entry.confidence = profile.classification_confidence or 0.0
            entry.page_count = profile.page_count
            if entry.is_book_like and book_describer is not None:
                # Cosmetic only — the decision works from the page count
                # alone, so a failure here must not lose the entry.
                try:
                    details = book_describer(path)
                    entry.title = details.get("title")
                    entry.chapter_count = details.get("chapter_count") or 0
                except Exception as error:            # pragma: no cover
                    log.debug("Could not describe %s: %s", path.name, error)
        except Exception as error:
            entry.error = f"{type(error).__name__}: {error}"
            log.error("Pre-flight failed for %s: %s", path.name, entry.error)
        entries.append(entry)

    log.info("Pre-flight: %d PDFs — %s", len(entries), _type_summary(entries))
    return entries


def resolve_book_decisions(entries: Sequence[PreflightEntry],
                           settings: BookSettings,
                           prompt: Optional[Callable[[PreflightEntry], BookDecision]] = None,
                           logger: Optional[logging.Logger] = None) -> None:
    """
    Settle what happens to every book-like document, before the run starts.

    The policy says what to do; `prompt` says whether there is a way to ask.
    Those are separate on purpose — see `policy.py`. A policy of `ask` with no
    prompt handler (an unattended run, or a GUI that has not supplied its
    dialog) resolves to the cheap default rather than hanging or overspending.

    Args:
        entries: the pre-flight entries; mutated in place with decisions.
        settings: the resolved run policy.
        prompt: asks the user about one document. The CLI passes a terminal
            prompt, a GUI passes a dialog, an unattended run passes nothing.
        logger: optional logger.
    """
    log = logger or LOG

    for entry in entries:
        if entry.error or not entry.is_book_like:
            continue

        if entry.is_implausible_chapter(settings.implausible_chapter_pages):
            entry.decision = BookDecision(
                reason=f"refused: {entry.page_count} pages is implausible for a "
                       f"chapter — likely a misclassified book, so bodies were "
                       f"NOT read")
            log.warning("%s classified book_chapter but has %d pages. Reading its "
                        "front matter only; check the classification.",
                        entry.path.name, entry.page_count)
            continue

        if entry.document_type is DocumentType.BOOK_CHAPTER:
            # A PLAUSIBLE chapter is not asked about, because there is nothing
            # to ask. The question's three answers — front matter only, every
            # chapter, these specific chapters — are all phrased about a
            # volume that CONTAINS chapters. This document IS one chapter, and
            # it is article-shaped: `router.build_strategy` hands
            # BookChapterStrategy neither `chapters` nor `max_body_pages`
            # (unlike the BOOK branch), so it reads its pages in full whatever
            # the answer is.
            #
            # Asking anyway was worse than merely useless. The prompt offered
            # "front matter only (fast, no GPU)" and then read all 36 pages of
            # the sample corpus's chapter on the GPU regardless — a promise the
            # pipeline had no way to keep. Found by driving the GUI
            # interactively; the CLI has the identical bug and never showed it,
            # because the corpus runs use --noninteractive, which supplies no
            # prompt and falls to the same default this now takes silently.
            #
            # The oversized-chapter case above still fires FIRST and still
            # warns: a "chapter" of hundreds of pages is a misclassified book,
            # and that is worth telling someone about.
            continue

        entry.decision = _decide(entry, settings, prompt, log)
        log.info("%s -> %s", entry.path.name, entry.decision.reason)


def decision_for(settings: BookSettings) -> BookDecision:
    """
    The decision a policy expresses without asking anybody.

    Used for the non-ASK policies, and as the fallback whenever asking is not
    possible. `ask` is never resolved here — that needs a handler.
    """
    from .policy import BookPolicy

    if settings.policy is BookPolicy.ALL:
        return BookDecision(chapters=[], max_pages=settings.max_pages,
                            reason=f"policy '{settings.policy.value}': read every chapter")
    if settings.policy is BookPolicy.CHAPTERS:
        return BookDecision(chapters=list(settings.chapters),
                            max_pages=settings.max_pages,
                            reason=f"policy 'chapters': {settings.chapters}")
    return BookDecision(max_pages=settings.max_pages,
                        reason=f"policy '{settings.policy.value}': "
                               f"front matter and chapter map only")


def write_batch_summary(entries: Sequence[PreflightEntry], results: Dict[str, Any],
                        output_dir: Path, elapsed: float) -> Path:
    """
    Write the run summary — what was processed, what failed, what was skipped.

    Two files, mirroring V5: a timestamped record, and a stable
    `batch_summary.json` for tooling that wants the latest run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": {
            "finished_utc": datetime.utcnow().isoformat() + "Z",
            "elapsed_seconds": round(elapsed, 1),
            "documents": len(entries),
        },
        "documents": [_entry_summary(entry, results) for entry in entries],
        "failed": [entry.path.name for entry in entries
                   if results.get(str(entry.path), {}).get("error") or entry.error],
    }

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for name in (f"batch_summary_{stamp}.json", "batch_summary.json"):
        with open(output_dir / name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    return output_dir / "batch_summary.json"


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _type_summary(entries: Sequence[PreflightEntry]) -> str:
    """"3 research_article, 1 book" — the headline of the pre-flight pass."""
    counts: Dict[str, int] = {}
    for entry in entries:
        key = entry.document_type.value if entry.document_type else "unreadable"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))


def _entry_summary(entry: PreflightEntry, results: Dict[str, Any]) -> Dict[str, Any]:
    """One document's line in the summary file."""
    result = results.get(str(entry.path), {})
    summary = {
        "file": entry.path.name,
        "path": str(entry.path),
        "document_type": entry.document_type.value if entry.document_type else None,
        "classification_confidence": entry.confidence,
        "page_count": entry.page_count,
        "processing_time": result.get("elapsed"),
        "output": result.get("output"),
        "error": result.get("error") or entry.error,
    }
    if entry.decision is not None:
        summary["book_decision"] = {
            "chapters": entry.decision.chapters,
            "max_pages": entry.decision.max_pages,
            "reason": entry.decision.reason,
        }
    return summary


def _decide(entry: PreflightEntry, settings: BookSettings,
            prompt: Optional[Callable[[PreflightEntry], BookDecision]],
            log: logging.Logger) -> BookDecision:
    """One document's decision: ask if the policy says so and asking is possible."""
    if not settings.wants_prompt():
        return decision_for(settings)

    if prompt is None:
        log.info("Policy is 'ask' but no way to ask was supplied; "
                 "taking the safe default for %s.", entry.path.name)
        return BookDecision(max_pages=settings.max_pages,
                            reason="policy 'ask' with no prompt available: "
                                   "front matter and chapter map only")

    decision = prompt(entry)
    if decision.max_pages is None:
        decision.max_pages = settings.max_pages
    return decision
