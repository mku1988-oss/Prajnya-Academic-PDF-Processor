#!/usr/bin/env python3
"""
Table-of-contents parsing for books.

For biology researchers: a book's contents page is a small structured index —
chapter number, chapter title, and the page it starts on — and it is printed
in plain text, so we can read it without any model at all. Having it means the
system knows a 345-page volume contains 15 chapters and where each begins,
which is what any later per-chapter processing will need.

The page is typeset with **dot leaders**:

    3 Tendon Extracellular Matrix Assembly,
      Maintenance and Dysregulation Throughout Life . . . . . . .  45
      Seyed Mohammad Siadat, Danae E. Zamboulis, ...

Three things make this harder than one regex: a title may wrap over several
lines before the leaders arrive; the *authors* of a chapter are printed after
its entry and must not be swallowed into the next one; and some entries
("Index", "Correction to: …") carry no chapter number at all.

So each entry is found from its leader line and read **backwards**, stopping
at whichever comes first: a line that opens with a chapter number, a line that
looks like an author list, or a small line budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ...utils.regex_patterns import DOT_LEADER_RUN, TOC_PAGE_NUMBER
from ...utils.author_utils import strip_affiliation_markers
from .front_matter_record import split_person_names

# The leader line, split into the title before the dots and the page after
# them. The run of dots itself is the shared pattern, so this parser and the
# classifier's contents-page detection can never drift apart.
LEADER_LINE = re.compile(
    rf'^(?P<title>.*?){DOT_LEADER_RUN}[ \t]*(?P<page>{TOC_PAGE_NUMBER})[ \t]*$')

# An entry that opens with its chapter number: "3 Tendon Extracellular…".
NUMBERED_START = re.compile(r'^\s*(?P<number>[0-9]{1,3})\s+(?P<rest>\S.*)$')

# An author list under a contents entry: capitalised names joined by commas or
# by a trailing "and". Only used for entries with no chapter number to anchor
# on — see `_title_for_entry`.
AUTHOR_LINE = re.compile(r'^\s*[A-Z][A-Za-z.\'’-]+(?:\s+[A-Z][A-Za-z.\'’-]+)*\s*,')
TRAILING_AND = re.compile(r'\band\s+(?:[A-Z][A-Za-z.\'’-]+\s*)+$')

# How many lines back an entry's title may begin. Springer wraps a long
# chapter title over four lines before the leaders arrive.
MAX_TITLE_LINES = 6

# How many lines an author list may occupy. Five authors with affiliations
# wrap over three lines in the sample; four is headroom, and the cap stops a
# malformed page from swallowing the rest of the column as "authors".
MAX_AUTHOR_LINES = 4

# Page furniture that ends an author list: the running "Contents" heading, or
# the roman folio a contents page is headed with ("v", "vi").
PAGE_FURNITURE = re.compile(r'^\s*(?:contents|[ivxlc]{1,7})\s*$', re.IGNORECASE)

# Lowercase words that legitimately appear inside a printed name list — the
# conjunction, and the nobiliary particles that are part of a surname.
NAME_PARTICLES = {
    "and", "de", "del", "della", "der", "den", "di", "da", "dos", "du",
    "la", "le", "van", "von", "y", "bin", "ibn", "ter", "ten",
}


@dataclass
class TocEntry:
    """
    One contents line: which chapter, called what, by whom, starting where.

    `authors` is the point of carrying this structure at all. In an edited
    scientific volume **each chapter has different authors**, so a retrieval
    system quoting from chapter 3 must cite Siadat et al., not the book's
    editor. The contents page states that mapping outright, and this is the
    only place it is cheap to read — by the time the body pages are being
    processed, the front matter is long out of scope.
    """
    number: Optional[int]
    title: str
    start_page: Optional[int]
    raw_page: str
    authors: List[str] = field(default_factory=list)

    def as_text(self) -> str:
        """The flat form stored in `ProcessingResult.toc`."""
        if self.number is not None:
            return f"{self.number}. {self.title}"
        return self.title


def parse_toc_entries(text: str) -> List[TocEntry]:
    """
    Read every contents entry on one contents page.

    Args:
        text: the text layer of a contents page.

    Returns:
        The entries in printed order. An empty list when the page holds no
        dot-leader lines at all — this parser never guesses at prose.

    Example:
        >>> entries = parse_toc_entries("1 Introduction . . . . . 1\\n")
        >>> entries[0].number, entries[0].title, entries[0].start_page
        (1, 'Introduction', 1)
    """
    lines = text.splitlines()
    entries: List[TocEntry] = []

    for index, line in enumerate(lines):
        match = LEADER_LINE.match(line)
        if not match:
            continue
        title = _title_for_entry(lines, index, match.group("title"))
        if not title:
            continue
        entry = _build_entry(title, match.group("page"))
        entry.authors = _authors_for_entry(lines, index)
        entries.append(entry)

    return entries


def toc_entries_to_text(entries: List[TocEntry]) -> List[str]:
    """Flatten entries for `ProcessingResult.toc`, which stores plain strings."""
    return [entry.as_text() for entry in entries]


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _title_for_entry(lines: List[str], leader_index: int, leader_title: str) -> str:
    """
    Rebuild an entry's title by walking backwards from its leader line.

    The leader line usually holds only the *last* line of a wrapped title. The
    lines above it are a mixture: the rest of this title, and above that the
    author list belonging to the PREVIOUS entry.

    The chapter number is the reliable anchor, because an author line never
    begins with a bare number. So we look back for the numbered line that
    opens this entry and take everything from there — which excludes the
    previous entry's authors without having to recognise them at all. That
    matters: "Laminin, Elastin, Fibrillins, Fibulins, Matrilins," is a wrapped
    *title* line that looks exactly like an author list.

    Only for entries with no chapter number ("Index", "Correction to: …") do
    we fall back to trimming by what looks like an author list.
    """
    first_line = leader_title.strip()
    if NUMBERED_START.match(first_line):
        return _join_title_lines([first_line])

    window = _preceding_lines(lines, leader_index)

    for offset, line in enumerate(window):
        if NUMBERED_START.match(line):
            return _join_title_lines(window[offset:] + [first_line])

    return _join_title_lines(_trim_to_title(window) + [first_line])


def _authors_for_entry(lines: List[str], leader_index: int) -> List[str]:
    """
    The chapter's authors: the lines printed directly *under* its entry.

    The structure is what makes this reliable. An entry's title runs from its
    chapter number down to its leader line; everything between that leader
    line and the *next* entry's start is the author list. So the same boundary
    that keeps authors out of titles hands us the authors, for free.

    Collection stops at the next entry (a numbered line or another leader
    line), at page furniture — the running "Contents" heading, or the roman
    folio a contents page is headed with — or at the first line that is not a
    list of names. That last test is the one doing the real work: it is also
    what rejects a numbered continuation line, so the explicit next-entry
    checks below are deliberate belt-and-braces rather than load-bearing.
    """
    collected: List[str] = []

    for line in lines[leader_index + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if NUMBERED_START.match(stripped) or LEADER_LINE.match(stripped):
            break
        if PAGE_FURNITURE.match(stripped) or not _looks_like_a_name_list(stripped):
            break
        collected.append(stripped)
        if len(collected) >= MAX_AUTHOR_LINES:
            break

    return split_person_names(" ".join(collected))


def _looks_like_a_name_list(line: str) -> bool:
    """
    True when a line consists only of people's names.

    Needed because an unnumbered entry can follow an author list directly:

        Jennifer Hope Roberts and Jaroslava Halper      <- chapter 15's authors
        Correction to: Clinical and Molecular …         <- the NEXT entry
        Laxa Syndromes: … . . . . . C1

    Nothing structural separates those two lines, so the author collection has
    to recognise where the names stop. Every token must look like part of a
    name — capitalised, an initial, or a lowercase particle ("van", "de",
    "and"). A digit or a colon is decisive evidence of something else.
    """
    if not line or ":" in line:
        return False

    tokens = line.replace(",", " ").split()
    if not tokens:
        return False

    for token in tokens:
        if token.lower() in NAME_PARTICLES:
            continue
        # A name may carry an affiliation marker ("Bella*1"), so digits are
        # judged after those are removed — a token that is ONLY digits is a
        # volume or a page, never a name.
        bare = strip_affiliation_markers(token)
        if not bare or not bare[0].isupper() or any(ch.isdigit() for ch in bare):
            return False

    return True


def _preceding_lines(lines: List[str], leader_index: int) -> List[str]:
    """The candidate title lines above a leader line, in printed order."""
    start = max(0, leader_index - MAX_TITLE_LINES)
    window = []
    for line in lines[start:leader_index]:
        stripped = line.strip()
        # A blank line or another entry's leader line ends the candidates.
        if not stripped or LEADER_LINE.match(stripped):
            window = []
            continue
        window.append(stripped)
    return window


def _trim_to_title(window: List[str]) -> List[str]:
    """Drop the previous entry's author lines from an unnumbered entry."""
    kept: List[str] = []
    for line in reversed(window):
        if _looks_like_authors(line):
            break
        kept.insert(0, line)
    return kept


def _join_title_lines(collected: List[str]) -> str:
    """Join wrapped title lines, tidying the hyphen and comma seams."""
    joined = " ".join(part for part in collected if part)
    joined = re.sub(r'\s+', " ", joined).strip()
    return joined.rstrip(". ").strip()


def _looks_like_authors(line: str) -> bool:
    """True for the author list printed beneath a contents entry."""
    if NUMBERED_START.match(line):
        return False
    return bool(AUTHOR_LINE.match(line) or TRAILING_AND.search(line))


def _build_entry(title: str, raw_page: str) -> TocEntry:
    """Split a leading chapter number off the title and parse the page."""
    number: Optional[int] = None
    numbered = NUMBERED_START.match(title)
    if numbered:
        number = int(numbered.group("number"))
        title = numbered.group("rest").strip()

    start_page = int(raw_page) if raw_page.isdigit() else None
    return TocEntry(number=number, title=title.strip(" .,"),
                    start_page=start_page, raw_page=raw_page)
