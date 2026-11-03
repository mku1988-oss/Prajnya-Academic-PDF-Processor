#!/usr/bin/env python
"""Measures how much of a page is a reference list rather than prose.

For biology researchers: the bibliography at the back of a paper is text, but
it is not *content* — nobody's question is answered by a list of other
people's paper titles. Telling the two apart matters more than it sounds,
because a reference list is unusually rich in exactly the words a search is
looking for.

WHY THIS EXISTS — TWO MEASURED HARMS, ONE CAUSE

The retrieval layer hit the same trap twice on the 9-document corpus:

1. Its evaluation set was built by finding pages carrying every distinctive
   term of an answer at once. For "which enzyme deficiency causes Bruck
   Syndrome", the page that qualified was a BIBLIOGRAPHY — because reference
   57's title is "Identification of PLOD2 as telopeptide lysyl hydroxylase".
   A cited paper's title carries every term of an answer while stating
   nothing. The label was wrong and had to be corrected by hand.

2. Once sections were assembled, the largest "section" in the entire corpus
   was `References` (527 paragraphs), followed by `CrossRef` (196). The
   biggest structural units in a scientific corpus were bibliographies.

Both come from the same cause: a reference list is a term-density trap.

WHAT THIS EMITS, AND WHAT IT DELIBERATELY LEAVES ALONE

Two facts per page, and no verdict:

    reference_density      share of substantial lines that look like citations
    has_reference_heading  a "References" / "Bibliography" heading is present

**Where the bibliography STARTS and ENDS is a document-level question**, and
this pass is page-local by design. A reference list runs across pages, its
heading sits on the first one, and a consumer that assembles whole documents
can join those facts up far better than a page can. Measured on the corpus:
body pages score 0.00–0.20 and reference pages 0.50–1.00, so the separation
is there — but pages dense with inline author-year citations also reach 0.5,
which is exactly why this reports a measurement instead of a boolean.

Nothing here parses references. That was tried in the v3 era with GROBID and
failed, and it remains out of scope: this only measures density.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

# Lines shorter than this are headers, page numbers and column furniture.
# They carry no citation signal either way, so they are not counted at all —
# including them would dilute the density of a genuine reference page.
MIN_SUBSTANTIAL_CHARS = 25

# A page needs a few real lines before a ratio means anything.
MIN_LINES_FOR_A_SCORE = 4

# How many independent citation signals a line needs before it counts. Two,
# not one: a bare year appears constantly in running prose, but a year AND a
# page range, or initials AND "et al.", is a citation.
SIGNALS_FOR_A_CITATION = 2

# Each is a cheap, independent mark of bibliographic formatting.
_YEAR = re.compile(r"\((?:19|20)\d{2}[a-z]?\)|\b(?:19|20)\d{2}[a-z]?\b")
_ET_AL = re.compile(r"\bet\s+al\b", re.I)
_PAGE_RANGE = re.compile(r"\b\d{1,4}\s*[-–—]\s*\d{1,4}\b")
_DOI = re.compile(r"\bdoi\b|10\.\d{4,}/", re.I)
_INITIALS = re.compile(r"\b[A-Z]\.\s*(?:[A-Z]\.\s*)?")
_VOLUME_ISSUE = re.compile(r"\b\d+\s*\(\s*\d+\s*\)\s*[:,]")

_CITATION_SIGNALS = (_YEAR, _ET_AL, _PAGE_RANGE, _DOI, _INITIALS, _VOLUME_ISSUE)

# A heading that opens a bibliography, with or without a section number.
_REFERENCE_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s*)?"
    r"(references?|bibliography|literature\s+cited|works\s+cited)"
    r"\s*:?\s*$",
    re.I | re.M,
)


def reference_signals(text: str) -> Dict[str, Any]:
    """Measure one page's bibliographic character.

    Args:
        text: the page's harvested text.

    Returns:
        `reference_density` (0.0–1.0) and `has_reference_heading` (bool).
        Density is 0.0 for a page too short to judge, which is a statement
        that it could not be measured rather than that it is prose.
    """
    return {
        "reference_density": reference_density(text),
        "has_reference_heading": has_reference_heading(text),
    }


def reference_density(text: str) -> float:
    """Share of a page's substantial lines that look like citations.

    Returns:
        0.0 to 1.0, rounded. 0.0 when the page has too few substantial lines
        to judge.
    """
    lines = substantial_lines(text)
    if len(lines) < MIN_LINES_FOR_A_SCORE:
        return 0.0
    citations = sum(1 for line in lines if looks_like_a_citation(line))
    return round(citations / len(lines), 3)


def has_reference_heading(text: str) -> bool:
    """Whether a "References" or "Bibliography" heading appears on the page.

    Matched on a line of its own, so the phrase "see the references below"
    inside a paragraph does not count.
    """
    return bool(_REFERENCE_HEADING.search(text or ""))


def looks_like_a_citation(line: str) -> bool:
    """Whether one line carries enough bibliographic marks to be a reference.

    Two independent signals are required. One is not enough: a year alone is
    ordinary in scientific prose ("since 1998 the model has..."), and so are
    initials. A year AND a page range, or initials AND "et al.", is a
    citation.
    """
    stripped = line.strip()
    if len(stripped) < MIN_SUBSTANTIAL_CHARS:
        return False
    hits = sum(1 for signal in _CITATION_SIGNALS if signal.search(stripped))
    return hits >= SIGNALS_FOR_A_CITATION


def substantial_lines(text: str) -> List[str]:
    """The page's lines long enough to carry a citation."""
    if not text:
        return []
    return [
        line for line in text.split("\n")
        if len(line.strip()) >= MIN_SUBSTANTIAL_CHARS
    ]
