#!/usr/bin/env python3
"""
Asking the user what to do with a book — the terminal front end.

For biology researchers: when the system finds a book it stops and asks, rather
than either skipping it silently or spending two hours reading all of it. It
shows you what it found — the title, how many pages, how many chapters and who
wrote them — and you choose.

This lives apart from `batch.py` on purpose. The *decision logic* (what counts
as a book, what the safe default is, what happens when nobody is watching) is
there and is tested without a terminal; this file is only the conversation, and
is the piece a GUI replaces with a dialog. Keeping them apart is what makes the
GUI revamp a substitution rather than a rewrite — see plan.md.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from .batch import BookDecision, PreflightEntry

LOG = logging.getLogger(__name__)


def ask_about_book(entry: PreflightEntry,
                   input_fn=input, output_fn=print) -> BookDecision:
    """
    Show what was found and ask what to do with it.

    Args:
        entry: the pre-flight entry for a book-like document.
        input_fn / output_fn: injected so the conversation can be tested.

    Returns:
        The chosen `BookDecision`. Anything unrecognised — including EOF, a
        closed pipe, or simply pressing enter — takes the SAFE default. A
        misread answer must never start hours of GPU work.
    """
    _show_book(entry, output_fn)

    try:
        answer = input_fn("  Your choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt, OSError):
        output_fn("  No answer available — taking the safe default.")
        return BookDecision.safe_default()

    if answer in ("", "1"):
        return BookDecision.safe_default()

    if answer == "2":
        return BookDecision(chapters=[], reason="user chose: read every chapter")

    if answer == "3":
        return _ask_which_chapters(entry, input_fn, output_fn)

    output_fn(f"  '{answer}' is not one of the options — taking the safe default.")
    return BookDecision.safe_default()


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def _show_book(entry: PreflightEntry, output_fn) -> None:
    """Print what was found, and what each option would cost."""
    from .strategies.book.reading import SECONDS_PER_PAGE_CEILING

    kind = entry.document_type.value if entry.document_type else "unknown"
    worst_case_minutes = entry.page_count * SECONDS_PER_PAGE_CEILING / 60

    output_fn("")
    output_fn("=" * 72)
    output_fn(f"  This looks like a {kind.upper()} "
              f"(confidence {entry.confidence:.2f})")
    output_fn(f"  {entry.path.name}")
    if entry.title:
        output_fn(f"  Title:    {entry.title}")
    output_fn(f"  Pages:    {entry.page_count}")
    if entry.chapter_count:
        output_fn(f"  Chapters: {entry.chapter_count}")
    output_fn("-" * 72)
    output_fn("  1) Front matter and chapter map only — seconds, no GPU  [default]")
    output_fn(f"  2) Read every chapter — up to ~{worst_case_minutes:.0f} minutes of GPU")
    output_fn("  3) Read named chapters")
    output_fn("=" * 72)


def _ask_which_chapters(entry: PreflightEntry, input_fn, output_fn) -> BookDecision:
    """Ask which chapters, accepting '3', '3 4 5' or '3-5'."""
    try:
        answer = input_fn("  Which chapters? (e.g. 3, or 3-5): ").strip()
    except (EOFError, KeyboardInterrupt, OSError):
        return BookDecision.safe_default()

    chapters = parse_chapter_selection(answer)
    if not chapters:
        output_fn("  No chapters recognised — taking the safe default.")
        return BookDecision.safe_default()

    return BookDecision(chapters=chapters,
                        reason=f"user chose: read chapters {chapters}")


def parse_chapter_selection(answer: str) -> List[int]:
    """
    Turn what someone typed into chapter numbers.

    Accepts "3", "3 4 5", "3,4,5" and "3-5", in any combination. Anything
    unparseable is skipped rather than guessed at.

    Example:
        >>> parse_chapter_selection("1, 3-5")
        [1, 3, 4, 5]
    """
    chapters: List[int] = []

    for token in answer.replace(",", " ").split():
        if "-" in token[1:]:                      # not a leading minus sign
            start, _, end = token.partition("-")
            if start.strip().isdigit() and end.strip().isdigit():
                chapters.extend(range(int(start), int(end) + 1))
            continue
        if token.isdigit():
            chapters.append(int(token))

    return sorted(set(chapters))


def show_preflight(entries: Sequence[PreflightEntry], output_fn=print) -> None:
    """
    Print the whole folder before any of it is processed.

    This is the moment a batch run becomes predictable: everything that is
    about to happen, including any book hiding in the middle of it, is on
    screen while it still costs nothing to change your mind.
    """
    output_fn("")
    output_fn(f"Pre-flight: {len(entries)} PDF(s)")
    output_fn("-" * 72)
    for entry in entries:
        marker = "  !" if (entry.error or entry.looks_misclassified) else "   "
        output_fn(f"{marker} {entry.describe()}")
    output_fn("-" * 72)
