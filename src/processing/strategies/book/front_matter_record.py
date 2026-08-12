#!/usr/bin/env python3
"""
Reading a book's bibliographic record off its front matter — CPU only.

For biology researchers: a book states who wrote it, who published it and when
on two specific pages, and it does so in plain text. The **copyright page** —
the one carrying the ISBN — holds the publisher, the year, the series and the
identifiers. The page immediately before it is the **title page**, holding the
title and the editors. Nothing here needs a model.

This is deliberately more than V5's `BookStrategy`, which set
`title = pdf_path.stem` and extracted no identifiers at all. It is also
deliberately *less* than the eventual book pipeline: it reads the front matter
and stops. Chapter-level reading is the later work this file exists to make
room for, and it can grow here without touching the research-article path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...utils.author_utils import strip_affiliation_markers

# "© Springer Nature Switzerland AG 2021" — publisher and year in one line.
COPYRIGHT_LINE = re.compile(
    r'(?:©|\(c\)|Copyright)\s*(?P<publisher>[A-Z][^\n0-9]{2,80}?)\s*(?P<year>(?:19|20)[0-9]{2})',
    re.IGNORECASE)

# A bare four-digit year, used only as a fallback on the copyright page.
YEAR = re.compile(r'\b(?P<year>(?:19|20)[0-9]{2})\b')

# The role marker under (or over) the editors' names on a title page.
EDITOR_MARKER = re.compile(r'^\s*(?:Editors?|Edited\s+by)\s*$', re.IGNORECASE)

# Lines a title page carries that are not the title.
EDITION_LINE = re.compile(r'^\s*(?:\w+\s+)?Edition\s*$', re.IGNORECASE)
VOLUME_LINE = re.compile(r'^\s*(?:Volume|Vol\.?)\s+[0-9]+\s*$', re.IGNORECASE)
ISSN_OR_ISBN_LINE = re.compile(r'\b(?:ISSN|ISBN)\b', re.IGNORECASE)

# Springer prints the editor's name and role on one line, joined by the
# narrow no-break spaces the text layer preserves: "Jaroslava Halper   Editor".
INLINE_EDITOR = re.compile(r'^(?P<names>.+?)[\s  - ]+Editors?\s*$',
                           re.IGNORECASE)


@dataclass
class BookRecord:
    """The bibliographic facts a book's front matter states about itself."""
    title: Optional[str] = None
    editors: List[str] = field(default_factory=list)
    publisher: Optional[str] = None
    year: Optional[int] = None
    series: Optional[str] = None
    title_page: Optional[int] = None
    copyright_page: Optional[int] = None

    def as_metadata(self) -> Dict:
        """The subset that maps onto `ExtractedMetadata` fields."""
        record = {"title": self.title, "editors": self.editors,
                  "publisher": self.publisher, "year": self.year}
        return {key: value for key, value in record.items() if value}


def build_book_record(page_texts: Dict[int, str],
                      copyright_page: Optional[int] = None) -> BookRecord:
    """
    Read a book's title, editors, publisher, year and series from its front matter.

    Args:
        page_texts: text layer keyed by 1-based page number, as collected by
            `scan_front_matter`.
        copyright_page: the page an identifier was found on, if known — that
            page IS the copyright page. When omitted it is located by looking
            for an ISBN line.

    Returns:
        A `BookRecord`. Every field is optional: a book that does not state
        something gets None rather than a guess.

    Example:
        >>> record = build_book_record(texts, copyright_page=5)   # doctest: +SKIP
        >>> record.title
        'Progress in Heritable Soft Connective Tissue Diseases'
    """
    record = BookRecord()
    record.copyright_page = copyright_page or _find_copyright_page(page_texts)

    if record.copyright_page:
        copyright_text = page_texts.get(record.copyright_page, "")
        record.publisher, record.year = _read_publisher_and_year(copyright_text)
        record.series = _read_series(copyright_text)

    record.title_page = _find_title_page(page_texts, record.copyright_page)
    if record.title_page:
        title_text = page_texts.get(record.title_page, "")
        record.editors = _read_editors(title_text)
        record.title = _read_title(title_text, record.editors, record.series)

    return record


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def _find_copyright_page(page_texts: Dict[int, str]) -> Optional[int]:
    """The copyright page is the one printing the ISBN."""
    for page_number, text in sorted(page_texts.items()):
        if text and re.search(r'\bISBN\b', text, re.IGNORECASE):
            return page_number
    return None


def _find_title_page(page_texts: Dict[int, str],
                     copyright_page: Optional[int]) -> Optional[int]:
    """
    The title page is the page immediately before the copyright page.

    That ordering is a publishing convention, not a guess: half-title, series
    page, title page, copyright page. When there is no copyright page to
    anchor on, fall back to page 1.
    """
    if copyright_page and copyright_page > 1:
        candidate = copyright_page - 1
        if page_texts.get(candidate):
            return candidate
    return 1 if page_texts.get(1) else None


def _read_publisher_and_year(text: str) -> tuple[Optional[str], Optional[int]]:
    """Publisher and year from the copyright line, e.g. '© Springer … 2021'."""
    match = COPYRIGHT_LINE.search(text)
    if match:
        publisher = re.sub(r'\s+', " ", match.group("publisher")).strip(" ,.")
        return publisher or None, int(match.group("year"))

    fallback = YEAR.search(text)
    return None, int(fallback.group("year")) if fallback else None


def _read_series(text: str) -> Optional[str]:
    """
    The series title, printed directly beneath the ISSN line.

    A book series is why an ISSN appears on a book at all — the same fact that
    made V5's classifier call this file a research article.
    """
    lines = [line.strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        if not re.search(r'\bISSN\b', line, re.IGNORECASE):
            continue
        for following in lines[index + 1:index + 3]:
            if following and not ISSN_OR_ISBN_LINE.search(following):
                return following
    return None


def _read_editors(text: str) -> List[str]:
    """
    The editors, from the line beside the 'Editor' / 'Editors' marker.

    Two layouts occur, and both appear in the sample corpus: the marker on its
    own line under the names, or the name and marker on one line separated by
    typographic spaces ("Jaroslava Halper   Editor").
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for index, line in enumerate(lines):
        if EDITOR_MARKER.match(line):
            names = lines[index - 1] if index else ""
            return _split_names(names)

        inline = INLINE_EDITOR.match(line)
        if inline:
            return _split_names(inline.group("names"))

    return []


def split_person_names(names: str) -> List[str]:
    """
    Split a printed name list into individual names.

    Shared by the editors on a title page and the chapter authors on a
    contents page: both are typeset the same way ("A, B, and C") and both
    feed the same downstream author matching. Affiliation markers are removed
    with the same function the article path uses, so a chapter author and a
    Zotero record produce the same key.

    Args:
        names: one printed line, or several joined, e.g.
            "Michael J. Mienaltowski, Nicole L. Gonzales, and Monica Y. Pechanec".

    Returns:
        The individual names, in printed order. Order is data — it encodes
        contribution.

    Example:
        >>> split_person_names("Ann Lee, Bo Chen and Cara Diaz")
        ['Ann Lee', 'Bo Chen', 'Cara Diaz']
    """
    if not names:
        return []
    # Normalise the typographic spaces Springer uses inside and around names.
    names = re.sub(r'[  - ]+', " ", names)

    # "A, B, and C" — the Oxford comma leaves an "and " glued to the last
    # name unless the split consumes it too.
    split: List[str] = []
    for part in re.split(r',\s*(?:and\s+)?|\s+and\s+', names):
        cleaned = strip_affiliation_markers(re.sub(r'\s+', " ", part).strip())
        if cleaned:
            split.append(cleaned)
    return split


def _split_names(names: str) -> List[str]:
    """Alias kept for readability at the two call sites in this module."""
    return split_person_names(names)


def _read_title(text: str, editors: List[str], series: Optional[str]) -> Optional[str]:
    """
    The title: what remains of the title page once the furniture is removed.

    A title page carries the title, the editors and their role marker, the
    edition, and often the series and its volume. Everything but the title is
    recognisable, so the title is what is left.
    """
    editor_names = {name.lower() for name in editors}
    series_name = (series or "").lower()

    kept = []
    for raw in text.splitlines():
        line = re.sub(r'[  - ]+', " ", raw).strip()
        if not line or _is_title_page_furniture(line, editor_names, series_name):
            continue
        kept.append(line)

    title = re.sub(r'\s+', " ", " ".join(kept)).strip()
    return title or None


def _is_title_page_furniture(line: str, editor_names: set, series_name: str) -> bool:
    """True for a title-page line that is anything except the title."""
    lowered = line.lower()
    if lowered in editor_names or (series_name and series_name in lowered):
        return True
    if EDITOR_MARKER.match(line) or INLINE_EDITOR.match(line):
        return True
    if EDITION_LINE.match(line) or VOLUME_LINE.match(line):
        return True
    return bool(ISSN_OR_ISBN_LINE.search(line))
