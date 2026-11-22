#!/usr/bin/env python
"""Finds a paper's reference list and splits it into entries (Milestone A2).

For biology researchers: the bibliography at the back of a paper is a list
of entries, and where each entry STARTS is printed in one of two ways:

- **numbered** ("245 Doi, M., Nishi, Y. ... (2005) ..."): an entry starts
  where a line starts with the next number. The list is the longest run of
  entry numbers 1, 2, 3 ... N found anywhere in the document, which also says
  where the list begins and ends, with no need for a "References" heading
  (Silberbush prints its heading as "■ REFERENCES", and the page-level
  `reference_density` misses four of the corpus's lists).
  MDPI prints the number on a line of its own, either before the entry's
  first line ("1." then the entry) or, centred on a two-line entry, after
  it; both count when the neighbouring line starts like an author.
- **author-year** ("Bella, J., & Hulmes, D. J. (2017). ..."): the list
  starts after its "References" heading, and an entry starts at the column's
  left edge while its other lines are indented (a hanging indent) or, in
  styles with no indent (Cell Press), after a wider gap between lines.

Each entry also gets its first author's surname and its year, which is
enough to name it ("Doi et al. 2005") and to check it against the answer key.
Its other fields (title, journal, volume, pages) are Milestone C.

Text layer only: no model, no network. Measured against the hand-checked key
in `eval/reference_lists/` (GATE_A.md).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .layout_lines import LayoutLine

# A list needs at least this many entries to count as one.
MIN_ENTRIES = 5

# At least this share of a numbered list's entries must start like an author
# ("Fan, P.", "Edzes HT,"): numbered section headings do not.
MIN_AUTHOR_SHARE = 0.6

# At least this share of a numbered list's entries must print a year: a run
# of numbered SECTION headings ("1. Introduction", "2. Methods") prints none.
MIN_YEAR_SHARE = 0.6

# The last entry has no next entry to end it: it takes at most this many
# lines, and stops at a heading that follows a list.
LAST_ENTRY_MAX_LINES = 6

# Hanging-indent style: a column whose lines start this far right of its left
# edge (points) are continuation lines...
INDENT_POINTS = 2.0
# ...and a column is in that style when at least this share of its lines are.
INDENTED_SHARE = 0.2
# A column's left edge is measured on lines of at least this many words.
# Shorter lines can be fragments of the NEIGHBOURING column that the layout
# put in this one: on PMC8333197 (Cell Reports) p12, "their", "s. We",
# "red in" sit 29 points left of the right column's real edge, and taking
# the edge from them made every real line look indented.
EDGE_MIN_WORDS = 3

# Block style (no indent): a gap above a line this many times the usual line
# pitch starts an entry.
BLOCK_GAP_SHARE = 1.15

# Text size of an author-year list (Frontiers sets back-matter notes inside
# it, in a larger size). The size is the median line height of the first
# SIZE_SAMPLE_LINES author-like lines within SIZE_SEARCH_LINES of the
# heading; a line within SIZE_TOLERANCE of it (share) is list-sized.
# Measured on PMC10977656: entries 9.44, notes 12.14 (+29%), headings 14.88.
# A scanned list's heights wobble: lib10's "Sternfeld" entry reads 10.98
# against 9.91 (+11%), so 10% dropped it; 20% keeps it and still parts
# Frontiers' notes from its entries.
SIZE_SAMPLE_LINES = 10
SIZE_SEARCH_LINES = 60
SIZE_TOLERANCE = 0.2
# A heading opens a note inside the list (not the list's end) when, of the
# next RESUME_LOOKAHEAD_LINES list-sized lines on its page, at least
# RESUME_MIN_ENTRIES start like an author, with as many "(YYYY)" years.
RESUME_LOOKAHEAD_LINES = 12
RESUME_MIN_ENTRIES = 2

# "245 Doi", "[12] Smith", "(16) Lee", "40. Maki", "38. ten Cate".
_NUMBER_START = re.compile(r"^[\[(]?(\d{1,4})[\]).]?\s+(?=[^\W\d_]|[\"“‘'(])")

# A number printed alone on its line WITH its punctuation: "1.", "[1]", "(1)".
# A bare "3" (a page number) does not match.
_BARE_NUMBER = re.compile(r"^\s*(?:\[(\d{1,4})\]|\((\d{1,4})\)|(\d{1,4})\.)\s*$")

_REFERENCE_HEADING = re.compile(
    r"^\W*(?:\d+\.?\s*)?(references?(?:\s+and\s+notes)?|reference\s+list|bibliography|"
    r"literature\s+cited|works\s+cited)\s*:?\s*$", re.I)

# Headings that follow a reference list and so end it. Cell Press prints its
# STAR Methods AFTER the list ("STAR★METHODS", read as "STAR+METHODS"),
# opening with the KEY RESOURCES TABLE.
_AFTER_LIST_HEADING = re.compile(
    r"^\W*(star\W{0,3}methods|key resources table|acknowledge?ments?|supplementary|supporting information|appendix|figure legends?|"
    r"further (related )?reading|author contributions|competing (financial )?interests?|"
    r"conflicts? of interest|abbreviations|associated content|author information|notes?)\b", re.I)

_PARTICLES = r"(?:(?:van|von|de|der|den|di|du|da|del|della|la|le|ten|ter|dos|das)\s+)*"
_NAME_WORD = r"[A-ZÀ-ɏ][\wÀ-ɏ'’\-]*"
_SURNAME_FIRST = (r"^" + _PARTICLES + _NAME_WORD + r"(?:\s+" + _PARTICLES + _NAME_WORD + r"){0,2},?\s+"
                  r"(?:[A-ZÀ-ɏ][.\-]|[A-Z]{1,3}\b|&|and\b|et\s+al)")
# Initials first, as Science prints: "P. W. Reddien, A. Sánchez Alvarado,".
_INITIALS = r"(?:[A-ZÀ-ɏ]\.(?:\s?-?[A-Z]\.)*\s+)"
_INITIALS_FIRST = (r"^" + _INITIALS + _PARTICLES + _NAME_WORD + r"(?:\s+" + _PARTICLES + _NAME_WORD
                   + r"){0,2}(?:\s+(?:Jr|Sr|II|III|IV|2nd|3rd|4th)\.?)?(?:,|\s+et\s+al|\s+and\b)")
_AUTHOR_START = re.compile(_SURNAME_FIRST + "|" + _INITIALS_FIRST)
_INITIALS_FIRST_NAME = re.compile(_INITIALS_FIRST)

# Generational suffixes, not part of a surname: "Kaelin Jr WG", "Kidd 3rd, A.".
_SUFFIX = re.compile(r"\s+(?:Jr|Sr|II|III|IV|2nd|3rd|4th)\.?$")

_YEAR = re.compile(r"(?<![\d/])(?:18|19|20)\d{2}[a-z]?(?![\d/])")
_YEAR_IN_PARENTHESES = re.compile(r"\(((?:18|19|20)\d{2}[a-z]?)\)")


@dataclass
class ReferenceEntry:
    """One entry of a reference list.

    Attributes:
        number: the printed number, or None in an author-year list.
        page: the 1-based page the entry STARTS on.
        text: the entry as printed (its lines joined, number removed).
        first_author: the first author's surname as printed ("Doi", "van der Flier").
        year: the publication year as printed ("2005", "1981a"), or None.
    """

    number: Optional[int]
    page: int
    text: str
    first_author: str
    year: Optional[str]


@dataclass
class ReferenceList:
    """A document's reference list.

    Attributes:
        style: "numbered" or "author_year".
        entries: the entries in printed order.
        first_line: the position of the list's first line in the document's
            lines, so a caller can tell the body (before it) from the list.
        end_line: the position just after the list's last line, when text
            follows the list (Cell Press prints its Methods there); None when
            the list runs to the end or its end is not known.
    """

    style: str
    entries: List[ReferenceEntry]
    first_line: int = 0
    end_line: Optional[int] = None


def find_reference_list(lines: Sequence[LayoutLine]) -> Optional[ReferenceList]:
    """The document's reference list, or None if it has none this can find.

    Args:
        lines: every page's lines in reading order (`document_layout_lines`).
    """
    numbered = numbered_list(lines)
    if numbered is not None:
        return numbered
    return author_year_list(lines)


def numbered_list(lines: Sequence[LayoutLine]) -> Optional[ReferenceList]:
    """The longest run of entry numbers 1..N at line starts, as a list, if it looks like one."""
    starts = longest_number_run(lines)
    if len(starts) < MIN_ENTRIES:
        return None
    entries = []
    for position, (index, number) in enumerate(starts):
        following = starts[position + 1][0] if position + 1 < len(starts) else -1
        end = following if following > index else last_entry_end(lines, index)
        entries.append(make_entry(number, lines[index].page, numbered_entry_text(lines, index, end, number)))
    if sum(1 for e in entries if e.year) < MIN_YEAR_SHARE * len(entries):
        return None
    if sum(1 for e in entries if _AUTHOR_START.match(e.text)) < MIN_AUTHOR_SHARE * len(entries):
        return None
    return ReferenceList(style="numbered", entries=entries, first_line=starts[0][0])


def numbered_entry_text(lines: Sequence[LayoutLine], start: int, end: int, number: int) -> str:
    """An entry's text: its lines from `start` up to `end`, without its printed number.

    A number printed alone on a line (MDPI) is dropped wherever it falls in
    the entry, since it may sit after the entry's first line, but ONLY when it
    equals the entry's own `number`. Any other bare number is part of the
    entry (the last line of "... Nat. Chem. 2014, 6," is just "596.", a page
    number) and is kept. A number that begins the first line ("12 Smith, ...")
    is stripped from it.
    """
    first = _NUMBER_START.sub("", lines[start].text, count=1)
    rest = [line.text for line in lines[start + 1:end]]
    kept = [text for text in [first] + rest if not is_own_number_line(text, number)]
    return " ".join(kept).strip()


def is_own_number_line(text: str, number: int) -> bool:
    """True when `text` is only the entry number `number` with its punctuation ("8.", "[8]", "(8)")."""
    bare = _BARE_NUMBER.match(text)
    if not bare:
        return False
    return int(next(g for g in bare.groups() if g)) == number


def entry_start_at(lines: Sequence[LayoutLine], index: int) -> Optional[Tuple[int, int]]:
    """(index of the line where an entry STARTS, its number) for the number at line `index`, or None.

    A line carries an entry's number when it begins with a number and text
    ("12 Smith,": the entry starts right there), or when it is only a number
    with its punctuation ("1.", "[1]", "(1)"). MDPI sets that number on its
    own line, either BEFORE the entry's first line (the NEXT line starts like
    an author: the entry starts at the number) or, centred on a two-line
    entry, AFTER its first line (the PREVIOUS line starts like an author and
    is not itself a number: the entry starts one line earlier).
    """
    text = lines[index].text
    match = _NUMBER_START.match(text)
    if match:
        return index, int(match.group(1))
    bare = _BARE_NUMBER.match(text)
    if not bare:
        return None
    number = int(next(g for g in bare.groups() if g))
    if index + 1 < len(lines) and _AUTHOR_START.match(lines[index + 1].text):
        return index, number
    if index >= 1 and _AUTHOR_START.match(lines[index - 1].text) \
            and not _BARE_NUMBER.match(lines[index - 1].text):
        return index - 1, number
    return None


def longest_number_run(lines: Sequence[LayoutLine]) -> List[Tuple[int, int]]:
    """(start line index, number) of the longest chain 1, 2, ... N, each start after the previous.

    Dynamic programming over every line that carries an entry number: an entry
    numbered n extends the best chain ending at n-1 that started before it.
    Body text that happens to start with a number ("2 h at room temperature")
    cannot hijack the list, because the list's own chain is far longer.
    """
    best_ending: Dict[int, Tuple[int, int]] = {}   # number -> (chain length, start index)
    previous: Dict[int, Optional[int]] = {}        # start index -> start index of number-1
    numbers: Dict[int, int] = {}
    for index in range(len(lines)):
        found = entry_start_at(lines, index)
        if found is None:
            continue
        start, number = found
        if number == 1:
            length, link = 1, None
        elif number - 1 in best_ending and best_ending[number - 1][1] < start:
            length, link = best_ending[number - 1][0] + 1, best_ending[number - 1][1]
        else:
            continue
        # ">=": of two equally long chains the LATER wins, so a "1" in the
        # body ("1 Introduction") gives way to the list's own entry 1.
        if length >= best_ending.get(number, (0, -1))[0]:
            best_ending[number] = (length, start)
            previous[start], numbers[start] = link, number
    if not best_ending:
        return []
    _, start = max(best_ending.values())
    chain: List[Tuple[int, int]] = []
    current: Optional[int] = start
    while current is not None:
        chain.append((current, numbers[current]))
        current = previous[current]
    return extend_on_last_page(lines, chain[::-1])


def extend_on_last_page(lines: Sequence[LayoutLine], chain: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Add entries N+1, N+2 ... printed on the chain's last page but read before entry N.

    On a list's last page a column only a few lines tall can be read out of
    order (its gutter is too short to find), so entry 83 comes before 82. A
    number that continues the chain on the SAME page is still the next entry.
    Entries are identified by their START line.
    """
    if not chain:
        return chain
    last_page = lines[chain[-1][0]].page
    used = {start for start, _ in chain}
    while True:
        wanted = chain[-1][1] + 1
        found = None
        for index, line in enumerate(lines):
            if line.page != last_page:
                continue
            candidate = entry_start_at(lines, index)
            if candidate and candidate[1] == wanted and candidate[0] not in used:
                found = candidate[0]
                break
        if found is None:
            return chain
        chain.append((found, wanted))
        used.add(found)


def last_entry_end(lines: Sequence[LayoutLine], start: int) -> int:
    """Where the last entry ends: a heading that follows lists, or a few lines on."""
    end = min(len(lines), start + LAST_ENTRY_MAX_LINES)
    for index in range(start + 1, end):
        if _AFTER_LIST_HEADING.match(lines[index].text):
            return index
    return end


def author_year_list(lines: Sequence[LayoutLine]) -> Optional[ReferenceList]:
    """The list after the last "References" heading, split by hanging indent or by gaps."""
    region, end_line = list_region(lines)
    if not region:
        return None
    first_line = list(lines).index(region[0])
    starts = [index for index in range(len(region)) if starts_author_entry(region, index)]
    entries = []
    for position, index in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(region)
        text = " ".join(line.text for line in region[index:end])
        entries.append(make_entry(None, region[index].page, text))
    if len(entries) < MIN_ENTRIES:
        return None
    return ReferenceList(style="author_year", entries=entries, first_line=first_line,
                         end_line=end_line)


def list_region(lines: Sequence[LayoutLine]) -> Tuple[List[LayoutLine], Optional[int]]:
    """The list's lines after the last reference heading, and where the list ends.

    The list ends at a heading that follows lists ("Conflict of interest"),
    unless that heading only opens a back-matter note set inside the list:
    Frontiers (2023 on) starts its list in the left column while the right
    column first finishes the back matter ("Funding", "Conflict of
    interest", "Publisher's note") and then carries on with the list. Such a
    note is printed in a different text size from the entries, so lines of
    another size are left out of the list, and the list goes on after them.

    Returns:
        (the list's lines, the position just after the list in `lines`, or
        None when the list runs to the end of the document).
    """
    headings = [index for index, line in enumerate(lines) if _REFERENCE_HEADING.match(line.text)]
    if not headings:
        return [], None
    start = headings[-1] + 1
    size = list_text_size(lines, start)
    region: List[LayoutLine] = []
    for index in range(start, len(lines)):
        line = lines[index]
        if _AFTER_LIST_HEADING.match(line.text) and not list_resumes_after(lines, index, size):
            return region, index
        if is_list_sized(line, size):
            region.append(line)
    return region, None


def list_text_size(lines: Sequence[LayoutLine], start: int) -> Optional[float]:
    """The text size (line height) of the list's entries, or None if it cannot be told.

    Taken as the median height of the first `SIZE_SAMPLE_LINES` lines after
    the heading that start like an author ("Abbas, Y., Yun, S., ...").
    """
    sample: List[float] = []
    for line in lines[start:start + SIZE_SEARCH_LINES]:
        if _AUTHOR_START.match(line.text) and line.height > 0:
            sample.append(line.height)
            if len(sample) == SIZE_SAMPLE_LINES:
                break
    return statistics.median(sample) if sample else None


def is_list_sized(line: LayoutLine, size: Optional[float]) -> bool:
    """Whether a line is printed in the list's text size (always True when the size is unknown)."""
    if size is None:
        return True
    return abs(line.height - size) <= SIZE_TOLERANCE * size


def list_resumes_after(lines: Sequence[LayoutLine], heading: int, size: Optional[float]) -> bool:
    """Whether the list carries on after the back-matter note that opens at line `heading`.

    It does when the heading is NOT in the list's text size (so the note can
    be told from the list at all), nor is the line under it (the note's
    text; its lines are skipped), and the list-sized lines that follow on the same
    page read as reference entries: at least `RESUME_MIN_ENTRIES` of the next
    `RESUME_LOOKAHEAD_LINES` start like an author, with as many years in
    parentheses. A Methods section or a table after the list (Cell Press)
    does not read like that, so it still ends the list.
    """
    if size is None or is_list_sized(lines[heading], size):
        return False
    # The note itself: its first line must be in another size too. A box
    # set in the list's size right under its heading ("Further Related
    # Reading" in Int. J. Dev. Biol., lib02) is not a note; it ends the list.
    if heading + 1 >= len(lines) or is_list_sized(lines[heading + 1], size):
        return False
    page = lines[heading].page
    following = [line for line in lines[heading + 1:]
                 if line.page == page and is_list_sized(line, size)][:RESUME_LOOKAHEAD_LINES]
    if any(_AFTER_LIST_HEADING.match(line.text) for line in following):
        return False
    authors = sum(1 for line in following if _AUTHOR_START.match(line.text))
    years = len(_YEAR_IN_PARENTHESES.findall(" ".join(line.text for line in following)))
    return authors >= RESUME_MIN_ENTRIES and years >= RESUME_MIN_ENTRIES


def starts_author_entry(region: Sequence[LayoutLine], index: int) -> bool:
    """Whether line `index` of the list starts an entry (author-year styles)."""
    line = region[index]
    if not _AUTHOR_START.match(line.text):
        return False
    column = [other for other in region if (other.page, other.band, other.column)
              == (line.page, line.band, line.column)]
    left = column_left_edge(column)
    indented = sum(1 for other in column if other.x0 > left + INDENT_POINTS)
    if indented >= INDENTED_SHARE * len(column):
        at_edge = line.x0 <= left + INDENT_POINTS
        if starts_are_indented(column, left):
            return not at_edge                          # first-line indent (Frontiers 2023 on)
        return at_edge                                  # hanging indent: at the left edge
    position = column.index(line)
    if position == 0:
        return True                                     # block style: top of a column
    pitches = [b.baseline - a.baseline for a, b in zip(column, column[1:])]
    usual = statistics.median(pitches) if pitches else 0.0
    return line.baseline - column[position - 1].baseline > BLOCK_GAP_SHARE * usual


def column_left_edge(column: Sequence[LayoutLine]) -> float:
    """Where a column's lines start: the leftmost line of EDGE_MIN_WORDS words or more.

    A one- or two-word line can be a fragment of the neighbouring column
    (see EDGE_MIN_WORDS), so it is not trusted to set the edge. A column
    with no line that long (a short tail on the last page) falls back to
    its leftmost line, which is what this used to be for every column.
    """
    full_lines = [line.x0 for line in column if len(line.text.split()) >= EDGE_MIN_WORDS]
    return min(full_lines) if full_lines else min(line.x0 for line in column)


def starts_are_indented(column: Sequence[LayoutLine], left: float) -> bool:
    """Whether a column indents each entry's FIRST line instead of the others.

    Most lists hang their entries (first line at the left edge, the rest
    indented). Frontiers (2023 on) does the reverse: "Abbas, Y., Yun, S.
    ... (2021). Recent" sits 6 points in, its next lines at the edge. Told
    apart by where the author-like lines sit: in a first-line-indent list
    most of them are indented, in a hanging list almost none are (a
    continuation line rarely starts like an author).
    """
    authors = [line for line in column if _AUTHOR_START.match(line.text)]
    indented = sum(1 for line in authors if line.x0 > left + INDENT_POINTS)
    return bool(authors) and indented > len(authors) / 2


def make_entry(number: Optional[int], page: int, text: str) -> ReferenceEntry:
    """A ReferenceEntry with its first author and year read off the text."""
    text = re.sub(r"\s+", " ", text).strip()
    return ReferenceEntry(number=number, page=page, text=text,
                          first_author=first_author(text), year=entry_year(text))


def first_author(text: str) -> str:
    """The first author's surname: up to the first comma, full stop, bracket or initials.

    "Fan, P., Li, M.H." -> "Fan"; "Edzes HT, Samulski ET." -> "Edzes";
    "AMANO S (2009)" -> "AMANO"; "van der Flier, L.G." -> "van der Flier".
    """
    text = text.lstrip(" .,;:")
    if _INITIALS_FIRST_NAME.match(text):
        text = re.sub("^" + _INITIALS, "", text)      # "P. W. Reddien, ..." -> "Reddien, ..."
    surname = re.split(r"[,;(]|\.\s| (?:[A-ZÀ-ɏ]\.?){1,3}(?=[,\s.(]|$)| et al\b", text, maxsplit=1)[0]
    return _SUFFIX.sub("", surname.strip(" .")).strip(" .")


def entry_year(text: str) -> Optional[str]:
    """The year: in parentheses first ("(2009)", "(1981a)"), else the first that is not a page.

    A bare four-digit number is often a page ("1973-1995"), so it is the last resort.
    """
    in_parentheses = _YEAR_IN_PARENTHESES.search(text)
    if in_parentheses:
        return in_parentheses.group(1)
    for match in _YEAR.finditer(text):
        after = text[match.end():match.end() + 3]
        before = text[max(0, match.start() - 2):match.start()]
        if not re.match(r"\s*[-–—−]\s*\d", after) and not re.search(r"[-–—−]\s*$", before):
            return match.group(0)
    return None
