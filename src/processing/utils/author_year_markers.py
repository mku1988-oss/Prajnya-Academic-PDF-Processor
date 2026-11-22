#!/usr/bin/env python
"""Finds author-year citation markers in a paper's text and links them to its list (S3b).

For biology researchers: many journals cite by name and year, not by number:
"... was shown (Doi et al., 2005; Smith and Jones, 1999a)" or "Doi et al.
(2005) showed ...". This module finds each such citation in the PDF's text
layer and asks the paper's reference list which entry it means. The list
decides: a citation links only when EXACTLY ONE entry of the list has that
first author and that year. Two candidates, or none: no link. A wrong link
sends a reader to the wrong paper, so every rule here prefers to say nothing.

HOW THE TEXT IS READ

- Only the body lines before the printed reference list are read (the list
  itself is full of "Name (Year)" patterns that cite nothing).
- The body is read as ONE stream across lines and pages, so a parenthesis that
  opens on one page and closes on the next is read whole. A line that ends in
  a hyphen before a lowercase word is a broken word ("assem-" "bly") and is
  joined without the hyphen; a hyphen before a capital ("Pucci-" "Minafra")
  is a real one and stays. Each character remembers its page; a citation's
  page is the page its YEAR is printed on.

WHAT IS A CITATION

- Parenthetical: inside "( ... )" every "Authors, Year" group, whether the
  groups are separated by ";" or only by commas. Prose in the parenthesis
  ("for review, see", "e.g.,") is skipped because it has no year after a name.
- Narrative: "Surname (Year)", "Surname and Surname (Year)", "Surname &
  Surname (Year)", "Surname et al. (Year, Year)".
- An author group may be "A", "A and B", "A & B", "A et al.", or an APA list
  ("A, B, & C"). It may carry several years ("Bella et al., 1994, 1995"). A
  year with no author in the stream (a bare "2014;") is no citation.
- The first surname may carry its author's initials ("Gong S et al., 2015",
  "Li X. H. et al., 2022"): Frontiers and others print them when two first
  authors share a surname and a year.

HOW A CITATION LINKS

1. Candidates are the entries whose first author (accents and case folded)
   and year equal the citation's.
2. A year with a letter ("2001a") keeps only an entry whose printed text
   carries that very year and letter.
3. Initials printed after the first surname keep only the entries whose
   first author has those initials ("Gong S" keeps "Gong, S., He, B., ..."
   and drops "Gong, Z., ..."); an entry whose initials cannot be read stays.
4. With more than one candidate left, the second author ("A and B") narrows
   them (the second surname must stand in the entry's author part, the text
   before its year); "et al." keeps entries with three or more authors and a
   single-author citation keeps entries with one author, each only when the
   printed text lets that be told.
5. Exactly one left: linked (to its `position`). None: "no_candidate".
   Letter not printed: "letter". Two or more: "ambiguous".

One marker is made per author group; `numbers` holds the linked positions in
printed order. A group none of whose years linked is no marker.
"""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .citation_markers import CONTEXT_CHARS, CitationMarker
from .reference_list import ReferenceList

KIND_AUTHOR_YEAR = "author_year"

#: A parenthesis longer than this is not read (an unmatched "(" in the text).
MAX_PAREN_CHARS = 400

#: An author part that says "et al." counts as at least this many authors.
ET_AL_AUTHORS = 99

# --- the shapes a citation can take -------------------------------------------------------

# A capitalised surname: "Doi", "Münder", "Pucci-Minafra", "O'Brien".
_NAME = r"[A-ZÀ-ÖØ-Þ][^\W\d_]*(?:['’\-‐‑][^\W\d_]+)*"
_PARTICLES = {"van", "von", "de", "der", "den", "del", "della", "di", "da", "dos", "du", "la", "le",
              "ter", "ten", "el", "al", "bin", "ibn"}
_PARTICLE = r"(?:" + "|".join(sorted(_PARTICLES)) + r")"
# A surname with its particles: "van der Flier".
_SURNAME = r"(?:" + _PARTICLE + r"\s+)*" + _NAME
_SURNAME_RE = re.compile(_SURNAME)
# Initials after a surname: "S", "XH", "X. H.", "J-P." (at most three letters).
_INITIALS = r"(?:[A-Z]\.?-?\s?){0,2}[A-Z]\.?"
# The authors of one citation: the first surname, its initials when printed,
# then nothing, "et al.", or "and B" / "& B" / ", B, & C" (an APA list).
_AUTHORS = (r"(?P<first>" + _SURNAME + r")"
            r"(?:\s+(?P<initials>" + _INITIALS + r")(?=[\s,(]))?"
            r"(?P<rest>\s+et\s+al\.?|(?:\s*,\s*" + _SURNAME + r")*\s*,?\s*(?:and|&)\s+" + _SURNAME + r")?")
# A year 1800-2099 with an optional lower-case letter, not part of a longer number or word.
_YEAR = r"(?<!\d)(?:1[89]|20)\d\d[a-z]?(?![A-Za-z0-9])"
_YEARS = _YEAR + r"(?:\s*,\s*" + _YEAR + r")*"
_YEAR_RE = re.compile(_YEAR)
_NOT_NAME_BEFORE = r"(?<![\w'’\-])"

_PAREN_GROUP_RE = re.compile(_NOT_NAME_BEFORE + _AUTHORS + r"\s*,?\s*(?P<years>" + _YEARS + r")")
_NARRATIVE_RE = re.compile(_NOT_NAME_BEFORE + _AUTHORS + r"\s*\(\s*(?P<years>" + _YEARS + r")\s*\)")
_PAREN_RE = re.compile(r"\(([^()]{1,%d})\)" % MAX_PAREN_CHARS)

# The initials an entry prints after its first surname: "Gong, S., He, B." -> "S.",
# "Li, X. H., ..." -> "X. H.", "Doi M, Nishi Y" -> "M".
_ENTRY_INITIALS_RE = re.compile(r"\s*" + _SURNAME + r"\s*,?\s*(?P<initials>(?:[A-Z]\.?-?\s?){1,4})(?=[,;.\s]|$)")

# A token that is a run of initials: "J.", "J.A.,", "JA", "J-P.".
_INITIALS_RE = re.compile(r"^(?:[A-Z]\.?-?){1,3}[,;.]*$")


@dataclass
class AuthorYearSearch:
    """What `search_author_year_markers` found, and what it threw away.

    Attributes:
        markers: the markers kept, in reading order (`CitationMarker`, kind "author_year").
        dropped: how many (author group, year) links were refused, by reason:
            "no_candidate" (no entry has that author and year), "ambiguous"
            (two or more do) and "letter" (the 2001a / 2001b letter is not
            printed on any candidate).
    """

    markers: List[CitationMarker]
    dropped: Dict[str, int]


@dataclass
class _Stream:
    """The body as one string, with the page of every character."""

    text: str
    starts: List[int]      # where each line's text starts in `text`
    pages: List[int]       # the page of each line

    def page_of(self, position: int) -> int:
        """The page the character at `position` is printed on."""
        index = max(0, bisect_right(self.starts, position) - 1)
        return self.pages[index]


@dataclass
class _Group:
    """One author group as found in the stream, before linking."""

    start: int
    end: int
    first: str
    second: Optional[str]
    et_al: bool
    years: List[Tuple[str, int]]       # (year as printed, its position in the stream)
    initials: Optional[str] = None     # the first author's initials as letters ("XH"), when printed


@dataclass
class _Reference:
    """One entry of the merged list, prepared for matching."""

    position: int
    keys: Set[str]
    printed: str
    author_part: Optional[str]
    authors: Optional[int] = field(default=None)
    initials: Optional[str] = None     # the first author's initials as letters ("XH"), when readable


def search_author_year_markers(lines: Sequence[Any], reference_list: Optional[ReferenceList],
                               references: Sequence[Mapping[str, Any]]) -> AuthorYearSearch:
    """
    Every author-year citation of the body that links to exactly one reference.

    Args:
        lines: the document's lines in reading order (anything with `.page`
            and `.text`, as `LayoutLine`).
        reference_list: what the PDF's own list was read into; only its style
            and `first_line` are used.
        references: the merged list's entries (`position`, `first_author`,
            `year`, `printed_text`).

    Returns:
        An `AuthorYearSearch`; empty for a numbered or missing list.
    """
    dropped = {"no_candidate": 0, "ambiguous": 0, "letter": 0}
    if reference_list is None or getattr(reference_list, "style", None) != "author_year":
        return AuthorYearSearch([], dropped)
    stream = body_stream(text_outside_the_list(lines, reference_list))
    index = reference_index(references)
    markers: List[CitationMarker] = []
    for group in find_groups(stream.text):
        marker = link_group(group, stream, index, dropped)
        if marker is not None:
            markers.append(marker)
    return AuthorYearSearch(markers, dropped)


# --- helpers (after the main function, per CLAUDE.md) --------------------------------------

def text_outside_the_list(lines: Sequence[Any], reference_list: ReferenceList) -> List[Any]:
    """The lines before the reference list, and those after it when the
    paper goes on (Cell Press's STAR Methods cite references too)."""
    body = list(lines[:reference_list.first_line])
    end_line = getattr(reference_list, "end_line", None)
    if end_line is not None:
        body.extend(lines[end_line:])
    return body


def body_stream(body: Sequence[Any]) -> _Stream:
    """The body lines joined into one stream; broken words are mended, pages remembered."""
    pieces: List[str] = []
    starts: List[int] = []
    pages: List[int] = []
    length = 0
    for line in body:
        text = (line.text or "").strip()
        if not text:
            continue
        joiner = " " if pieces else ""
        if pieces and _ends_in_word_hyphen(pieces[-1]) and text[:1].isalpha():
            joiner = ""                                # the word goes on without a space
            if text[0].islower():
                pieces[-1] = pieces[-1][:-1]           # a broken word: the hyphen was only typesetting
                length -= 1
        pieces.append(joiner + text)
        starts.append(length + len(joiner))
        pages.append(line.page)
        length += len(joiner) + len(text)
    return _Stream("".join(pieces), starts, pages)


def _ends_in_word_hyphen(previous: str) -> bool:
    """Whether the previous line ends in a letter and a hyphen ("assem-", "Pucci-")."""
    return len(previous) >= 2 and previous.endswith("-") and previous[-2].isalpha()


def find_groups(stream: str) -> List[_Group]:
    """Every author group, parenthetical and narrative, in reading order, none overlapping."""
    groups: List[_Group] = []
    for paren in _PAREN_RE.finditer(stream):
        for match in _PAREN_GROUP_RE.finditer(stream, paren.start(1), paren.end(1)):
            groups.append(_group_of(match, stream))
    for match in _NARRATIVE_RE.finditer(stream):
        groups.append(_group_of(match, stream))
    groups.sort(key=lambda g: (g.start, -g.end))
    kept: List[_Group] = []
    for group in groups:
        if not kept or group.start >= kept[-1].end:
            kept.append(group)
    return kept


def _group_of(match: "re.Match", stream: str) -> _Group:
    """The author group a regular-expression match stands for."""
    rest = match.group("rest")
    et_al = bool(rest) and re.match(r"\s+et\s+al", rest) is not None
    second = None
    if rest and not et_al:
        names = _SURNAME_RE.findall(rest)
        second = names[0] if names else None
    years = [(y.group(0), y.start()) for y in
             _YEAR_RE.finditer(stream, match.start("years"), match.end("years"))]
    return _Group(match.start(), match.end(), match.group("first"), second, et_al, years,
                  initials=initial_letters(match.group("initials")))


def link_group(group: _Group, stream: _Stream, index: Dict[int, List[_Reference]],
               dropped: Dict[str, int]) -> Optional[CitationMarker]:
    """The marker for one author group, or None when none of its years links."""
    positions: List[int] = []
    page: Optional[int] = None
    for year_text, where in group.years:
        found = link_year(group, year_text, index, dropped)
        if found is None:
            continue
        if found not in positions:
            positions.append(found)
        if page is None:
            page = stream.page_of(where)
    if not positions or page is None:
        return None
    context = stream.text[max(0, group.start - CONTEXT_CHARS):group.start].strip()
    return CitationMarker(page=page, text=stream.text[group.start:group.end], kind=KIND_AUTHOR_YEAR,
                          numbers=positions, context=context)


def link_year(group: _Group, year_text: str, index: Dict[int, List[_Reference]],
              dropped: Dict[str, int]) -> Optional[int]:
    """The position of the one reference this author group + year names, or None (and why)."""
    keys = citation_keys(group.first)
    candidates = [r for r in index.get(int(year_text[:4]), []) if r.keys & keys]
    if not candidates:
        dropped["no_candidate"] += 1
        return None
    if year_text[4:]:
        candidates = [r for r in candidates if _prints_year(r.printed, year_text)]
        if not candidates:
            dropped["letter"] += 1
            return None
    if group.initials:
        candidates = [r for r in candidates if _has_initials(r, group.initials)]
    if len(candidates) > 1:
        candidates = narrow_candidates(candidates, group)
    if len(candidates) == 1:
        return candidates[0].position
    dropped["no_candidate" if not candidates else "ambiguous"] += 1
    return None


def narrow_candidates(candidates: List[_Reference], group: _Group) -> List[_Reference]:
    """Candidates left after the second author, "et al." or "single author" is taken into account."""
    if group.second:
        candidates = [r for r in candidates if _names_second_author(r, group.second)]
    if group.et_al:
        candidates = [r for r in candidates if r.authors is None or r.authors >= 3]
    elif not group.second:
        candidates = [r for r in candidates if r.authors is None or r.authors == 1]
    return candidates


def _has_initials(reference: _Reference, initials: str) -> bool:
    """Whether the entry's first author has the cited initials ("X" fits "XH"; unreadable: yes)."""
    return reference.initials is None or reference.initials.startswith(initials)


def initial_letters(initials: Optional[str]) -> Optional[str]:
    """ "X. H." -> "XH"; None or no letter -> None."""
    letters = "".join(c for c in initials or "" if c.isalpha())
    return letters or None


def _entry_initials(printed: str) -> Optional[str]:
    """The initials an entry prints after its first surname, as letters, or None."""
    match = _ENTRY_INITIALS_RE.match(printed)
    return initial_letters(match.group("initials")) if match else None


def _names_second_author(reference: _Reference, second: str) -> bool:
    """Whether the second surname stands in the entry's author part (unknown author part: yes)."""
    if reference.author_part is None:
        return True
    wanted = fold_words(strip_particles(second))
    return bool(wanted) and f" {wanted} " in f" {fold_words(reference.author_part)} "


def _prints_year(printed: str, year_text: str) -> bool:
    """Whether the entry's printed text carries exactly this year and letter ("2001a")."""
    return re.search(r"(?<!\d)" + re.escape(year_text) + r"(?![A-Za-z0-9])", printed) is not None


def reference_index(references: Sequence[Mapping[str, Any]]) -> Dict[int, List[_Reference]]:
    """The merged list's entries grouped by year, each prepared for matching."""
    index: Dict[int, List[_Reference]] = {}
    for reference in references or []:
        year = _year_of(reference.get("year"))
        position = reference.get("position")
        if year is None or not isinstance(position, int):
            continue
        printed = reference.get("printed_text") or ""
        author = reference.get("first_author") or _leading_surname(printed)
        keys = citation_keys(author) if author else set()
        if not keys:
            continue
        author_part = _author_part(printed, year)
        index.setdefault(year, []).append(_Reference(
            position=position, keys=keys, printed=printed, author_part=author_part,
            authors=count_authors(author_part) if author_part is not None else None,
            initials=_entry_initials(printed)))
    return index


def _year_of(value: Any) -> Optional[int]:
    """The four-digit year of a reference's `year` field, or None."""
    match = re.match(r"\s*(\d{4})", str(value)) if value is not None else None
    return int(match.group(1)) if match else None


def _leading_surname(printed: str) -> Optional[str]:
    """The first surname printed at the start of an entry, for a reference with no `first_author`."""
    match = re.match(r"\s*(" + _SURNAME + ")", printed)
    return match.group(1) if match else None


def _author_part(printed: str, year: int) -> Optional[str]:
    """The entry's text before its year (where the authors are), or None when the year is not found."""
    match = re.search(r"(?<!\d)" + str(year), printed)
    return printed[:match.start()] if match else None


def count_authors(author_part: str) -> Optional[int]:
    """How many authors an author part names, when its initials make that clear; else None."""
    if re.search(r"\bet al\b", author_part):
        return ET_AL_AUTHORS
    runs, in_run = 0, False
    for token in author_part.split():
        is_initials = _INITIALS_RE.match(token) is not None
        if is_initials and not in_run:
            runs += 1
        in_run = is_initials
    return runs or None


def citation_keys(surname: str) -> Set[str]:
    """The folded spellings a surname may be matched under (with and without its particles)."""
    keys = {fold_letters(surname), fold_letters(strip_particles(surname))}
    keys.discard("")
    return keys


def strip_particles(surname: str) -> str:
    """"van der Flier" -> "Flier"."""
    words = surname.split()
    while len(words) > 1 and words[0].islower() and words[0] in _PARTICLES:
        words.pop(0)
    return " ".join(words)


def _fold(text: str) -> str:
    """Accents removed, case folded."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return plain.casefold()


def fold_letters(text: str) -> str:
    """Only the letters and digits, accents removed, case folded ("Pucci-Minafra" -> "pucciminafra")."""
    return "".join(c for c in _fold(text) if c.isalnum())


def fold_words(text: str) -> str:
    """Words of letters and digits, accents removed, case folded, single-spaced."""
    return " ".join("".join(c if c.isalnum() else " " for c in _fold(text)).split())
