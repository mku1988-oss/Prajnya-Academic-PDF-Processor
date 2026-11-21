#!/usr/bin/env python
"""Page furniture: running headers, footers and folios, labelled in page text.

For biology researchers: every page of a journal article carries lines that
are not part of the article's text: "Frontiers in Genetics 03 frontiersin.org",
"Cell Reports 22, 1054-1066, January 23, 2018 1055", "Yamamori et al. eLife
2023;12:RP87720 ... 3 of 39". Left in the page text they cut a sentence that
runs over a page in two: a chunk made there reads "...the backbone dihedral
phi and psi Frontiers in Genetics 03 frontiersin.org Liang et al. angles using
the RAMPAGE program...", which is worse for search, reranking and quoting.

They are also useful: they carry the journal, year, volume and DOI, and
Prajnya's metadata reading uses them. So they are LABELLED, never deleted:
the page `content` keeps every line; `metadata.furniture` records where each
furniture line is; `body` (`page_body.py`) leaves those lines out, with the
other parts of a page that are not running text.

A line is furniture when it stands among the first or last few lines of the
page's text and either

1. **repeats**: the same line (spaces collapsed) stands at the edge of many
   pages ("Liang et al. 10.3389/fgene.2023.1219297", "OPEN ACCESS"), or
2. **counts pages**: lines of one shape whose number rises with the page
   ("Frontiers in Genetics 02 ..." on page 2, "... 03 ..." on page 3; Cell
   Reports' folio 1055 on page 2, 1057 on page 4).

A line that is only a page's number in that same series is furniture too.
What it refuses: a line that recurs on only a few pages (eLife's "Appendix 3"
headings, a repeated table header, PLOS's "(TIF)"), and anything inside
Prajnya's own placeholders, comments or tables.

Model-read pages mostly leave the furniture out already; the rule finds
whatever is there, whichever route read the page. Derived on every write from
the stored page text, never cached.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

#: How many prose lines at each end of a page may be furniture. Text-layer
#: pages write a header and a footer at the end, a logo line or two besides.
EDGE_LINES = 5

#: A repeated line is furniture when it stands at the edge of at least this
#: share of the pages that carry furniture at all ...
MIN_REPEAT_SHARE = 0.4

#: ... and of at least this many pages.
MIN_REPEAT_PAGES = 3

#: A line repeated on fewer pages than that share still counts in a book,
#: where each chapter has its own running header: it must then stand on this
#: many pages, densely (`MIN_REPEAT_SHARE` of the pages between its first and
#: last appearance).
MIN_LOCAL_REPEAT_PAGES = 5

#: A page-counting shape must be seen on this many pages ...
MIN_FOLIO_PAGES = 2


#: In an article, bare number lines are page numbers when this share of
#: them agree on one offset (the rest are stray numbers: a table cell).
MIN_BARE_NUMBER_SHARE = 0.6

#: Furniture is short. A longer line is a sentence.
MAX_FURNITURE_CHARACTERS = 160

#: Lines that are Prajnya's own markup, not page text: object placeholders,
#: HTML comments, Markdown or HTML tables, LaTeX display blocks.
_MARKUP_LINE = re.compile(r"^\s*(?:\[\[(?:FIGURE|TABLE) |<!--|\||<table|</?t[rdh]\b|\$\$)",
                          re.IGNORECASE)

#: A line that starts with an object's label is a heading or a legend, not
#: furniture, however often its shape recurs ("FIGURE 1" on page 4 and
#: "FIGURE 6" on page 9 step with the pages by chance).
_OBJECT_LABEL = re.compile(
    r"^\W*(?:fig(?:ure)?s?\b|table|appendix|supplement|box\b|scheme|movie|video)",
    re.IGNORECASE)

#: A number in a line: the candidates for a page number.
_NUMBER = re.compile(r"(?<![\d.])\d{1,5}(?![\d.])")


def label_page_furniture(pages: Sequence[Mapping[str, Any]]
                         ) -> Dict[int, List[Dict[str, Any]]]:
    """
    The furniture lines of every page, as character spans of its `content`.

    Args:
        pages: the document's pages as written (`page_number`, `content`).

    Returns:
        {page_number: [{"kind", "start", "end", "text"}, ...]} for pages with
        furniture. `kind` is "running_line" (a header or footer) or
        "page_number" (a line that is only the page's printed number).
        `start`/`end` index the page's `content`; `end` excludes the line
        break.
    """
    edges = {page["page_number"]: _edge_lines(page["content"]) for page in _readable(pages)}
    repeated = _repeated_lines(edges)
    folio_offsets = _folio_offsets(edges)
    labelled: Dict[int, List[Dict[str, Any]]] = {}
    for page_number, lines in edges.items():
        found = []
        for start, end, text in lines:
            kind = _kind_of(text, page_number, repeated, folio_offsets)
            if kind:
                found.append({"kind": kind, "start": start, "end": end, "text": text.strip()})
        if found:
            labelled[page_number] = found
    return labelled


# --- helpers (after the main functions, per CLAUDE.md) ---------------------

def _readable(pages: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """The pages with a page number and text."""
    return [page for page in pages if isinstance(page, Mapping)
            and isinstance(page.get("page_number"), int)
            and isinstance(page.get("content"), str)]


def _edge_lines(content: str) -> List[Tuple[int, int, str]]:
    """(start, end, text) of the first and last `EDGE_LINES` prose lines."""
    lines, position = [], 0
    for raw in content.splitlines(keepends=True):
        text = raw.rstrip("\r\n")
        if text.strip() and not _MARKUP_LINE.match(text):
            lines.append((position, position + len(text), text))
        position += len(raw)
    if len(lines) <= 2 * EDGE_LINES:
        return lines
    return lines[:EDGE_LINES] + lines[-EDGE_LINES:]


def _key(text: str) -> str:
    """A line compared as written, spaces collapsed."""
    return " ".join(text.split())


def _shape(text: str) -> str:
    """A line with its numbers blanked, to compare folio lines across pages."""
    return _NUMBER.sub("#", _key(text))


def _repeated_lines(edges: Mapping[int, List[Tuple[int, int, str]]]) -> set:
    """The lines that stand, unchanged, at the edge of many pages."""
    pages_with = defaultdict(set)
    for page_number, lines in edges.items():
        for _, _, text in lines:
            if _could_be_furniture(text):
                pages_with[_key(text)].add(page_number)
    seen_pages = {n for pages in pages_with.values() if len(pages) >= 2 for n in pages}
    needed = max(MIN_REPEAT_PAGES, MIN_REPEAT_SHARE * len(seen_pages))
    return {key for key, pages in pages_with.items()
            if len(pages) >= needed or _dense_in_its_stretch(pages)}


def _dense_in_its_stretch(pages: set) -> bool:
    """A book chapter's running header: many pages, close together.

    "M. J. Mienaltowski et al." heads every other page of one chapter and
    none of the others, so it is rare in the whole book but dense in its
    chapter. "Abstract" opens every chapter: as frequent, but spread thin.
    """
    if len(pages) < MIN_LOCAL_REPEAT_PAGES:
        return False
    stretch = max(pages) - min(pages) + 1
    return len(pages) >= MIN_REPEAT_SHARE * stretch


def _folio_offsets(edges: Mapping[int, List[Tuple[int, int, str]]]) -> Dict[str, set]:
    """{line shape: {printed number minus page number}} for page-counting lines.

    Each shape's lines are tried number by number: the shape counts pages when
    one of its numbers, minus the page number, is the same on every page the
    shape is on ("Cell Reports 22, 1054-1066, ... 1055" on page 2 and "...
    1057" on page 4 share 1053). Shapes with no varying number are left to
    `_repeated_lines`.

    A line that is only a number (a book's folio) is the exception: a book
    stitched from chapters shifts its numbering from chapter to chapter, and
    a stray number line (a table cell) shares the shape. There each offset
    that a dense stretch of pages shares counts (`_bare_number_offsets`).
    """
    by_shape = defaultdict(list)
    for page_number, lines in edges.items():
        for _, _, text in lines:
            if _could_be_furniture(text) and _NUMBER.search(text):
                by_shape[_shape(text)].append((page_number, text))
    offsets = {}
    for shape, seen in by_shape.items():
        found = _bare_number_offsets(seen) if shape == "#" else _common_offset(seen)
        if found:
            offsets[shape] = found
    return offsets


def _common_offset(seen: List[Tuple[int, str]]) -> set:
    """{the offset} that one number slot shares on every page, or empty."""
    pages = {page_number for page_number, _ in seen}
    if len(pages) < MIN_FOLIO_PAGES or len(pages) != len(seen):
        return set()
    numbers = [[int(n) for n in _NUMBER.findall(text)] for _, text in seen]
    for slot in range(min(len(row) for row in numbers)):
        found = {row[slot] - page_number for row, (page_number, _) in zip(numbers, seen)}
        if len(found) == 1:
            return found
    return set()


def _bare_number_offsets(seen: List[Tuple[int, str]]) -> set:
    """The offsets of bare page numbers: one most lines share (an article),
    or each that a dense stretch of pages shares (a book's chapters)."""
    pages_by_offset = defaultdict(set)
    for page_number, text in seen:
        pages_by_offset[int(_key(text)) - page_number].add(page_number)
    return {offset for offset, pages in pages_by_offset.items()
            if _dense_in_its_stretch(pages)
            or (len(pages) >= MIN_FOLIO_PAGES and len(pages) >= MIN_BARE_NUMBER_SHARE * len(seen))}


def _kind_of(text: str, page_number: int, repeated: set,
             folio_offsets: Mapping[str, set]) -> str:
    """'running_line', 'page_number' or '' for one edge line."""
    if not _could_be_furniture(text):
        return ""
    key = _key(text)
    if key.isdigit():
        # A bare number is a page number when it fits any folio series of
        # the document (Silberbush prints "page 3"-style lines AND numbers).
        series = set().union(*folio_offsets.values()) if folio_offsets else set()
        return "page_number" if int(key) - page_number in series else ""
    if key in repeated:
        return "running_line"
    numbers = {int(n) for n in _NUMBER.findall(text)}
    if any(offset + page_number in numbers for offset in folio_offsets.get(_shape(text), ())):
        return "running_line"
    return ""


def _could_be_furniture(text: str) -> bool:
    """Short enough, and not an object's label line."""
    key = _key(text)
    return 0 < len(key) <= MAX_FURNITURE_CHARACTERS and not _OBJECT_LABEL.match(key)
