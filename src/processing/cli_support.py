#!/usr/bin/env python3
"""
CLI conveniences carried over from V5's `run_Prajnya.py`.

For biology researchers: two small things that matter a lot in day-to-day use.

**Page ranges.** V5 accepted `--pages 2-5`. V6 only understood
`--pages 2 3 4 5`, which is fine until you want pages 40-60. Both forms work
now, so a habit learned on V5 keeps working.

**Not re-reading what you already read.** Re-running a folder after adding two
new papers should cost two papers, not thirty. The semantic ledger already
records every document by content hash, so a document it knows about is
skipped unless `--force` says otherwise. The hash is of the file's *contents*,
so renaming a PDF does not fool it and editing one correctly does not skip it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

LOG = logging.getLogger(__name__)


def parse_pages(values: Optional[List[str]]) -> Optional[List[int]]:
    """
    Turn `--pages` into 1-based page numbers, accepting V5's range syntax.

    Args:
        values: what argparse collected — e.g. `["2-5"]`, `["1", "4", "7"]`,
            or `["1", "3-5"]`. None means the flag was not given.

    Returns:
        Sorted, deduplicated page numbers, or None when the flag was absent.
        An unparseable token is skipped with a warning rather than guessed at
        — processing the wrong pages silently is worse than processing fewer.

    Example:
        >>> parse_pages(["1", "3-5"])
        [1, 3, 4, 5]
    """
    if values is None:
        return None

    pages: List[int] = []
    for token in _split_tokens(values):
        if "-" in token[1:]:
            start, _, end = token.partition("-")
            if start.strip().isdigit() and end.strip().isdigit():
                first, last = int(start), int(end)
                if first <= last:
                    pages.extend(range(first, last + 1))
                    continue
            LOG.warning("Ignoring page range %r — expected something like 2-5", token)
            continue

        if token.isdigit():
            pages.append(int(token))
        else:
            LOG.warning("Ignoring page %r — not a number or a range", token)

    return sorted(set(pages))


def parse_chapters(values: Optional[List[str]],
                   logger: Optional[logging.Logger] = None) -> Optional[List[int]]:
    """
    Turn `--chapters` into chapter numbers, accepting ranges like `3-5`.

    For biology researchers: `--chapters 3-5` and `--chapters 3 4 5` mean the
    same thing, and `--chapters` with nothing after it means *every* chapter.

    Args:
        values: what argparse collected. **None means the flag was absent**,
            which reads no chapter bodies at all; an empty list means the flag
            was given with no numbers, which reads every chapter.
        logger: optional logger.

    Returns:
        None (not requested), an empty list (every chapter), or the numbers.

        Tokens that parse to nothing — a typo like `--chapters abc` — return
        **None**, not "everything". The two answers differ by hours of GPU
        time, and a misread flag must never be the expensive one.

    Example:
        >>> parse_chapters(["1", "3-5"])
        [1, 3, 4, 5]
        >>> parse_chapters([])          # the flag, with no numbers
        []
        >>> parse_chapters(None) is None
        True
    """
    if values is None:
        return None
    if not values:
        return []                       # `--chapters` alone: every chapter

    from .book_prompt import parse_chapter_selection

    chapters = parse_chapter_selection(" ".join(str(value) for value in values))
    if not chapters:
        (logger or LOG).warning(
            "Could not read any chapter number from %s — reading the front "
            "matter and chapter map only. Use --chapters 3, 3-5, or --chapters "
            "with nothing after it to read the whole book.", " ".join(values))
        return None

    return chapters


def already_processed(pdf_path: Path, ledger, logger: Optional[logging.Logger] = None
                      ) -> bool:
    """
    True when the ledger already holds a record for this exact file.

    Identity is the file's SHA-256, so a renamed PDF is still recognised and an
    edited one is correctly treated as new.

    Args:
        pdf_path: the PDF to check.
        ledger: a `SemanticLedger`, or None (nothing is ever skipped).
        logger: optional logger.

    Returns:
        Whether this document can be skipped. Any failure answers **False** —
        re-processing a document costs time, while wrongly skipping it loses
        it from the library entirely.
    """
    log = logger or LOG
    if ledger is None:
        return False

    try:
        from .ledger.semantic_ledger import generate_hash

        record = ledger.get_record(generate_hash(str(pdf_path)))
    except Exception as error:
        log.debug("Ledger check failed for %s (%s); processing it.",
                  pdf_path.name, error)
        return False

    if record is None:
        return False

    log.info("%s is already in the ledger (%s) — skipping. Use --force to redo it.",
             pdf_path.name, record.review_status)
    return True


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _split_tokens(values: List[str]) -> List[str]:
    """Flatten what argparse gave us, so '2-5,7' and '2-5 7' both work."""
    tokens: List[str] = []
    for value in values:
        tokens.extend(part for part in str(value).replace(",", " ").split() if part)
    return tokens
