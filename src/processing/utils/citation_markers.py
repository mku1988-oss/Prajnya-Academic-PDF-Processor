#!/usr/bin/env python
"""Finds numeric citation markers in a paper's text and links them to its entries (A3).

For biology researchers: "a dramatic decrease in thermal stability [245]"
cites entry 245 of the paper's own reference list. Journals print that
number in one of two ways:

- **in brackets** ("[245]", "[3–7]", "[94,160]");
- **in parentheses** ("(12)", "(24, 25)", "(26–28)"), as Science and the
  J. Biol. Chem. print them;
- **raised** above the line ("stability²⁴⁵"). The PDF's text layer stores a
  raised number glued to its word ("stability245"), so it is recovered from
  where each character SITS: a run of digits set above the line's baseline
  and smaller than its letters (`PrintedChar.baseline` and box height).

ONE STYLE PER DOCUMENT. A paper cites in brackets, parentheses or
superscripts, not a mix. A document with bracket markers links only
brackets, so the raised digits of its chemistry ("Hyp⁴S", "R²") are never
read as citations. Parentheses are chosen only when they outnumber raised
markers AND cite a good share of the list: methods text is full of numbers
in parentheses ("(15 mM)", "(1)" for an equation) that cite nothing. In that
style a parenthesised number beyond the list is simply not a marker (a
quantity), rather than evidence that the list is incomplete.

NOT EVERY RAISED NUMBER CITES. Left out: exponents (a run that starts with a
minus, or follows one, "cm⁻¹"), charges ("Zn²⁺", a run followed by a raised
"+"), isotopes ("¹H", a run followed directly by a capital letter), powers
after a unit, a number or a lone capital ("μm²", "10³", "−10³", "R²"), anything
inside a formula (an "=" in the word it is attached to), and the affiliation
marks of an author line ("Haocheng Quan ¹": set after a space, behind a
capitalised name).

A WRONG LINK IS WORSE THAN NONE (the same rule as figure labels). Every
number of a marker must name an entry 1..N. What a number beyond N means
depends on whether the list is KNOWN to be whole:

- NOT CONFIRMED (the default): if ANY bracket or raised marker names a number
  beyond N, the list may not have been found whole, so the document gets no
  links at all.
- CONFIRMED (`list_confirmed=True`: the list was read twice and both readings
  agree on its length): a number beyond N is not evidence of a short list, it
  is simply not a citation. That one marker is dropped and the others stay.

THE ORDER CHECK. A numbered paper cites 1, 2, 3 ... in order of first
appearance. The affiliation marks of an author line ("Jean-Philippe Lambert¹,¹³,
Gordana Ivosev²,¹³,") are raised numbers glued to surnames and followed by a
comma, so they look like citations and are all small enough to pass the
self-check above. A run of such name-attached raised numbers is dropped when
one of its numbers shows up for the first time out of order (13 when only 1 has
been cited). A run is also dropped when it is redundant: it holds the first
appearance of 1, and the body goes on to cite those same numbers for the first
time, in order, exactly as if the run were not there ("Luo¹,², Chen¹,², Deng²,³"
followed by a body that cites 1, 2, 3). A real opening "Collins¹, Smith²" stays,
because without it the body's first new number would be 3. Affiliation marks
come as a block, so once one such run on a page is dropped, every such run on
that page is dropped with it. A run that keeps
the order ("Collins¹⁸, Smith¹⁹" after 17) stays. This applies to raised
markers only, never to brackets or parentheses.

AUTHOR LINES ARE NEVER CITATIONS. On the first 3 pages, a line whose words are
names (at least 2 capitalised, at most 1 lowercase; connectives and surname
particles skipped; raised characters left out) is an author or affiliation
line, whatever its marks are glued to ("Huong Le, MD¹; Rebecca Kleinerman,
MD¹;", "Lambert¹,¹³, Ivosev²,¹³,"). Every raised number on it is dropped. Body
prose always has lowercase words. The order check above stays as a second guard.

ENUMERATIONS ARE NOT CITATIONS. A figure legend numbers its gel lanes "ClaI (2),
NruI (3), MluI (4), NarI (5), PvuI (6), SmaI (7), or undigested (1)". By then the
paper has already cited 1 to 7, and the lane numbers look exactly like
citations. A run of at least four single-number brackets or parentheses, close
together, that count consecutively from 1 or 2 and cite nothing new is an
enumeration and is dropped. A run that is new to the paper ("acetylcholinesterase
[11], adiponectin [12], ...") or does not start at 1 or 2 is a real citation list.

Numbered lists only; author-year citations are Milestone B. Text layer only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from ..pdf.base import PrintedChar
from .layout_lines import LayoutLine
from .reference_list import ReferenceList

# A document with at least this many bracket markers cites in brackets.
MIN_BRACKET_MARKERS = 3

# Parenthesised numbers are the style only when at least this many are found
# and the distinct entries they cite cover this share of the list.
MIN_PAREN_MARKERS = 3
MIN_PAREN_COVERAGE = 0.3

# A parenthesised number right after one of these words is a figure, an
# equation or a step, not a citation ("Eq. (3)", "lane (2)").
_NOT_CITING_BEFORE = re.compile(r"\b(?:eqs?|equations?|figs?|figures?|tables?|steps?|lanes?|panels?|"
                                r"refs?|reactions?|schemes?|compounds?|n)\.?\s*$", re.I)

# A raised character sits at least this share of its line's height above the
# line's baseline, and is at most this share of the line's height tall.
MIN_RISE_SHARE = 0.15
MAX_RAISED_HEIGHT_SHARE = 0.85

# Range joiners and list separators inside a marker. Used inside a regular
# expression the dash must be escaped: written bare, "‒-−" is a RANGE of
# characters from U+2012 to U+2212 that left the plain keyboard hyphen out, so
# "(5-7)" and "[3-12]" were never read as ranges.
_JOINERS = ",–—‒-−"
_DASHES = "–—‒-−"
_JOINER_CLASS = re.escape(_JOINERS)
_DASH_CLASS = re.escape(_DASHES)

# How many lines ahead the closing half of a split parenthesis is looked for,
# and what it looks like: the last numbers and the closing parenthesis.
MAX_PAREN_LOOKAHEAD = 3
_PAREN_CLOSING_HEAD = re.compile(r"\s*\d{1,3}(?:\s*[" + _JOINER_CLASS + r"]\s*\d{1,3})*\s*\)")

# An enumeration run (gel lanes, panels) needs this many markers; its markers sit
# at most this many characters apart.
MIN_ENUMERATION_MARKERS = 4
MAX_ENUMERATION_GAP = 40
_SENTENCE_END = re.compile(r"\.\s+[A-Z]")

# Author lines: only the first pages hold them; a line of names has at least
# this many capitalised words and at most this many lowercase ones.
MAX_AUTHOR_LINE_PAGE = 3
MIN_AUTHOR_LINE_NAMES = 2
MAX_AUTHOR_LINE_LOWERCASE = 1
# Connectives and surname particles that are not names and not prose.
_AUTHOR_LINE_SKIP = {"and", "van", "von", "der", "den", "de", "del", "della", "di", "da", "dos", "du",
                     "la", "le", "ten", "ter", "y", "e"}
_WORD = re.compile(r"[^\W\d_][^\W\d_'’\-\.]*")

# How much text before a marker is kept to find it again.
CONTEXT_CHARS = 40

# Words after which a raised number is a power, not a citation.
_UNITS = {"m", "cm", "mm", "μm", "µm", "nm", "km", "s", "ms", "g", "mg", "kg", "l", "ml", "μl", "µl",
          "k", "hz", "khz", "mhz", "j", "w", "v", "pa", "kpa", "mpa", "gpa", "mol", "n", "h"}

_BRACKET = re.compile(r"\[\s*(\d{1,4}(?:\s*[" + _JOINER_CLASS + r"]\s*\d{1,4})*)\s*\]")
# A capitalised name glued to a raised number ("Lambert", "Gingras", "SWATH").
_NAME_LIKE = re.compile(r"[A-Z][A-Za-z\u00C0-\u024F'\u2019\-]+")
# What may follow a raised affiliation mark: a comma, "&", "and", or nothing.
_AFTER_NAME_MARK = re.compile(r"(?:[,&]|and\b)|$")
# Corresponding-author and footnote signs, possibly after a comma ("Ruberti1,2,*").
_AFTER_SIGN_MARK = re.compile(r"[,\s]*[*†‡§¶#]")
# The next author's name, or "and" / "&", after a mark set behind a comma ("Olszewski,1 Siadat").
_NEXT_AUTHOR = re.compile(r"[A-Z]|and\b|&")

_PAREN = re.compile(r"(?<![\w)\]])\(\s*(\d{1,3}(?:\s*[" + _JOINER_CLASS + r"]\s*\d{1,3})*)\s*\)")
# A parenthesis opened at the end of a line that still needs its closing half
# on the next line: "(15," or "(15, 18, 22,".
_PAREN_OPEN_TAIL = re.compile(r"(?<![\w)\]])\(\s*\d{1,3}(?:\s*[" + _JOINER_CLASS + r"]\s*\d{1,3})*\s*["
                              + _JOINER_CLASS + r"]\s*$")


@dataclass
class CitationMarker:
    """One in-text citation marker, linked to the entries it cites.

    Attributes:
        page: 1-based page it is printed on.
        text: the marker as printed ("[3–7]", "12,14").
        kind: "bracket", "paren" or "superscript".
        numbers: the entry numbers it cites, ranges expanded (3–7 -> 3, 4, 5, 6, 7).
        context: the text just before it, to find it again.
        start, end: where it sits in its page's text (the page's lines joined by
            single spaces); -1 when unknown. Used to measure the gap to its
            neighbours; not part of what makes two markers equal.
    """

    page: int
    text: str
    kind: str
    numbers: List[int] = field(default_factory=list)
    context: str = ""
    start: int = field(default=-1, compare=False, repr=False)
    end: int = field(default=-1, compare=False, repr=False)


@dataclass
class MarkerSearch:
    """What `search_numeric_markers` found, and what it threw away.

    Attributes:
        markers: the markers kept, in reading order.
        dropped: how many markers each rule removed:
            "beyond_list" (a number beyond a confirmed list's end),
            "author_line" (raised numbers on an author or affiliation line),
            "order_check" (affiliation marks that break the citation order) and
            "enumeration" (lane or panel numbers counted in a legend).
    """

    markers: List[CitationMarker]
    dropped: Dict[str, int]


def find_numeric_markers(lines: Sequence[LayoutLine], reference_list: Optional[ReferenceList],
                         last: Optional[int] = None, list_confirmed: bool = False) -> List[CitationMarker]:
    """Every numeric citation marker in the body, linked to the numbered list.

    Args:
        lines: the document's lines in reading order (`document_layout_lines`).
        reference_list: what `find_reference_list` found in the same lines.
        last: the largest entry number of the list (default: the list's own maximum).
        list_confirmed: the list's length was confirmed by a second reading.

    Returns:
        Markers in reading order; [] for an author-year or missing list, and
        [] for a document whose markers fail the self-check.
    """
    return search_numeric_markers(lines, reference_list, last, list_confirmed).markers


def search_numeric_markers(lines: Sequence[LayoutLine], reference_list: Optional[ReferenceList],
                           last: Optional[int] = None, list_confirmed: bool = False) -> MarkerSearch:
    """Like `find_numeric_markers`, but also reports how many markers each rule dropped."""
    dropped = {"beyond_list": 0, "order_check": 0, "enumeration": 0, "author_line": 0}
    if reference_list is None or reference_list.style != "numbered":
        return MarkerSearch([], dropped)
    body = list(lines[:reference_list.first_line])
    if last is None:
        last = max(entry.number for entry in reference_list.entries)
    brackets = bracket_markers(body)
    beyond = 0
    if list_confirmed:
        brackets, beyond = split_beyond_list(brackets, last)
    if len(brackets) >= MIN_BRACKET_MARKERS:
        dropped["beyond_list"] = beyond
        if not passes_self_check(brackets, last):
            return MarkerSearch([], dropped)
        brackets, dropped["enumeration"] = drop_enumerations(brackets, body)
        return MarkerSearch(brackets, dropped)
    candidates, dropped["author_line"] = drop_author_line_candidates(raised_candidates(body), body)
    raised, dropped["order_check"] = drop_out_of_order_name_runs(candidates)
    beyond = 0
    if list_confirmed:
        raised, beyond = split_beyond_list(raised, last)
    in_parentheses = parenthesised_markers(body, last)
    if cites_in_parentheses(in_parentheses, raised, last):
        in_parentheses, dropped["enumeration"] = drop_enumerations(in_parentheses, body)
        return MarkerSearch(in_parentheses, dropped)
    dropped["beyond_list"] = beyond
    return MarkerSearch(raised if passes_self_check(raised, last) else [], dropped)


def split_beyond_list(markers: Sequence[CitationMarker], last: int) -> tuple:
    """(markers whose numbers all lie in 1..last, how many markers were left out)."""
    kept = [m for m in markers if all(1 <= number <= last for number in m.numbers)]
    return kept, len(markers) - len(kept)


def cites_in_parentheses(in_parentheses: Sequence[CitationMarker], raised: Sequence[CitationMarker],
                         last: int) -> bool:
    """Whether this document's citation style is numbers in parentheses."""
    cited = {number for marker in in_parentheses for number in marker.numbers}
    return (len(in_parentheses) >= MIN_PAREN_MARKERS and len(in_parentheses) > len(raised)
            and len(cited) >= MIN_PAREN_COVERAGE * last)


def parenthesised_markers(lines: Sequence[LayoutLine], last: int) -> List[CitationMarker]:
    """"(12)", "(24, 25)", "(26–28)" that name entries 1..last, line by line.

    Left out: a parenthesis that is the whole line or ends a line holding
    "=" (an equation number), one after "Fig.", "Eq.", "lane" and the like,
    and one naming any number outside the list (a quantity, not a citation).
    A parenthesis that opens at the end of a line ("(15,") and closes on the
    next one ("19, 24)") is read as one marker.
    """
    markers: List[CitationMarker] = []
    offsets = page_line_offsets(lines)
    for index, line in enumerate(lines):
        text = line.text
        for match in _PAREN.finditer(text):
            marker = paren_marker(text, match, line.page, offsets[index], last)
            if marker is not None:
                markers.append(marker)
        split = split_paren_across_lines(lines, index)
        if split is not None:
            joined, match, adjacent = split
            marker = paren_marker(joined, match, line.page, offsets[index], last)
            if marker is not None:
                markers.append(marker if adjacent else without_end(marker))
    return markers


def paren_marker(text: str, match: "re.Match", page: int, line_offset: int, last: int) -> Optional[CitationMarker]:
    """The marker for one parenthesis found in `text`, or None when a guard rejects it."""
    before = text[:match.start()]
    alone = not before.strip() and not text[match.end():].strip(" .,;")
    if alone or (match.end() >= len(text.rstrip(" .,;")) and "=" in text):
        return None
    if _NOT_CITING_BEFORE.search(before):
        return None
    numbers = expand(match.group(1))
    if not numbers or not all(1 <= number <= last for number in numbers):
        return None
    return CitationMarker(page=page, text=match.group(0), kind="paren", numbers=numbers,
                          context=before[-CONTEXT_CHARS:].strip(),
                          start=line_offset + match.start(), end=line_offset + match.end())


def split_paren_across_lines(lines: Sequence[LayoutLine], index: int) -> Optional[tuple]:
    """(line joined to its continuation, the parenthesis match, whether the continuation is the next line).

    Used when line `index` ends inside a parenthesis ("(15, 18, 22,"). The
    continuation is the next line on the same page, or one of the next few
    when a figure legend was read in between (two-column pages); it must start
    with the rest of the numbers and the closing parenthesis ("23) species").
    The match must run past the end of the first line, so a parenthesis that
    closes on its own line is not read twice.
    """
    line = lines[index]
    tail = _PAREN_OPEN_TAIL.search(line.text)
    if tail is None:
        return None
    for ahead in range(1, MAX_PAREN_LOOKAHEAD + 1):
        if index + ahead >= len(lines) or lines[index + ahead].page != line.page:
            return None
        if not _PAREN_CLOSING_HEAD.match(lines[index + ahead].text):
            continue
        joined = line.text + " " + lines[index + ahead].text
        match = _PAREN.match(joined, tail.start())
        if match is not None and match.end() > len(line.text):
            return joined, match, ahead == 1
        return None
    return None


def without_end(marker: CitationMarker) -> CitationMarker:
    """The marker with its end position forgotten (its text is not contiguous on the page)."""
    marker.end = -1
    return marker


def page_line_offsets(lines: Sequence[LayoutLine]) -> List[int]:
    """Where each line starts in its page's text (the page's lines joined by single spaces)."""
    offsets: List[int] = []
    running: Dict[int, int] = {}
    for line in lines:
        offset = running.get(line.page, 0)
        offsets.append(offset)
        running[line.page] = offset + len(line.text) + 1
    return offsets


def drop_enumerations(markers: Sequence[CitationMarker], lines: Sequence[LayoutLine]) -> tuple:
    """(markers kept, how many were dropped) after removing enumerations such as gel lane numbers.

    Walks the markers in reading order, remembering which numbers kept markers
    have cited. A run of close, single-number markers is judged by
    `is_enumeration_run` when it ends; a run that is not an enumeration is kept.
    """
    page_texts = {page: " ".join(line.text for line in lines if line.page == page)
                  for page in {line.page for line in lines}}
    kept: List[CitationMarker] = []
    seen: Set[int] = set()
    run: List[CitationMarker] = []
    dropped = 0
    for marker in markers:
        if len(marker.numbers) == 1 and run and continues_run(run[-1], marker, page_texts):
            run.append(marker)
            continue
        dropped += settle_run(run, seen, kept)
        run = [marker] if len(marker.numbers) == 1 else []
        if not run:
            kept.append(marker)
            seen.update(marker.numbers)
    dropped += settle_run(run, seen, kept)
    return kept, dropped


def settle_run(run: Sequence[CitationMarker], seen: Set[int], kept: List[CitationMarker]) -> int:
    """Keep or drop one finished run; returns how many markers were dropped."""
    if is_enumeration_run(run, seen):
        return len(run)
    kept.extend(run)
    for marker in run:
        seen.update(marker.numbers)
    return 0


def continues_run(previous: CitationMarker, marker: CitationMarker, page_texts: Dict[int, str]) -> bool:
    """Whether `marker` follows `previous` closely: same page, a short gap, no sentence end between."""
    if previous.page != marker.page or previous.end < 0 or marker.start < previous.end:
        return False
    gap = page_texts.get(marker.page, "")[previous.end:marker.start]
    return len(gap) <= MAX_ENUMERATION_GAP and _SENTENCE_END.search(gap) is None


def is_enumeration_run(run: Sequence[CitationMarker], seen: Set[int]) -> bool:
    """Whether a run of single-number markers counts items (lanes, panels) instead of citing.

    Needs at least four markers whose numbers are distinct and consecutive
    integers, the smallest being 1 or 2 (items are counted from one), and no
    number that is a first appearance (all were cited earlier by kept markers).
    """
    if len(run) < MIN_ENUMERATION_MARKERS:
        return False
    numbers = [marker.numbers[0] for marker in run]
    distinct = set(numbers)
    consecutive = len(distinct) == len(numbers) and max(numbers) - min(numbers) + 1 == len(numbers)
    return consecutive and min(numbers) in (1, 2) and distinct <= seen


def passes_self_check(markers: Sequence[CitationMarker], last: int) -> bool:
    """Whether every marker names entries 1..last (a larger number means the list is incomplete)."""
    return all(1 <= number <= last for marker in markers for number in marker.numbers)


def bracket_markers(lines: Sequence[LayoutLine]) -> List[CitationMarker]:
    """"[245]", "[3–7]", "[94,160]", read page by page so a marker may break across lines."""
    markers: List[CitationMarker] = []
    for page in sorted({line.page for line in lines}):
        text = " ".join(line.text for line in lines if line.page == page)
        for match in _BRACKET.finditer(text):
            numbers = expand(match.group(1))
            if numbers:
                markers.append(CitationMarker(page=page, text=match.group(0), kind="bracket", numbers=numbers,
                                              context=text[max(0, match.start() - CONTEXT_CHARS):match.start()].strip(),
                                              start=match.start(), end=match.end()))
    return markers


def superscript_markers(lines: Sequence[LayoutLine]) -> List[CitationMarker]:
    """Runs of raised digits that cite, line by line (before the order check)."""
    return [candidate.marker for candidate in raised_candidates(lines)]


@dataclass
class RaisedCandidate:
    """A raised marker with what the order check needs to know about it."""

    marker: CitationMarker
    line_index: int          # which line of the document it sits on
    name_attached: bool      # glued to a capitalised name and set like an author's affiliation mark


def raised_candidates(lines: Sequence[LayoutLine]) -> List[RaisedCandidate]:
    """Every raised run that cites, in reading order, with its line and name-attached flag."""
    candidates: List[RaisedCandidate] = []
    for line_index, line in enumerate(lines):
        for start, end in raised_digit_runs(line):
            marker = superscript_marker(line, start, end)
            if marker is not None:
                candidates.append(RaisedCandidate(marker, line_index, is_name_attached(line, start, end)))
    return candidates


def drop_author_line_candidates(candidates: Sequence[RaisedCandidate], lines: Sequence[LayoutLine]) -> tuple:
    """(candidates kept, how many were dropped) after removing every one that sits on an author line.

    `lines` must be the same lines `raised_candidates` was given, because each
    candidate points into them by `line_index`.
    """
    author_lines: Dict[int, bool] = {}
    kept: List[RaisedCandidate] = []
    for candidate in candidates:
        index = candidate.line_index
        if index not in author_lines:
            author_lines[index] = is_author_line(lines[index])
        if not author_lines[index]:
            kept.append(candidate)
    return kept, len(candidates) - len(kept)


def is_author_line(line: LayoutLine) -> bool:
    """Whether a line on the first pages is made of names (an author or affiliation line).

    Its words, with raised characters left out and connectives skipped, must
    include at least 2 capitalised words and at most 1 lowercase word.
    """
    if line.page > MAX_AUTHOR_LINE_PAGE:
        return False
    words = [w for w in line_words(line) if w.lower() not in _AUTHOR_LINE_SKIP]
    capitalised = sum(1 for w in words if w[0].isupper())
    lowercase = sum(1 for w in words if w[0].islower())
    return capitalised >= MIN_AUTHOR_LINE_NAMES and lowercase <= MAX_AUTHOR_LINE_LOWERCASE


def line_words(line: LayoutLine) -> List[str]:
    """The words of a line as printed, raised characters (marks) left out."""
    text = "".join((" " + c.text) if c.space_before else c.text
                   for c in line.chars if not is_raised(c, line))
    return _WORD.findall(text)


def is_name_attached(line: LayoutLine, start: int, end: int) -> bool:
    """Whether a raised run is an affiliation-style mark: "Lambert¹,¹³,", "Tate² &", "Ruberti¹,²,*".

    True when the word it is glued to looks like a capitalised name AND the run
    is set like an author's mark: followed by a comma, "&", "and" or the end of
    the line; followed by a corresponding-author sign ("*", "†" ...); or, when
    the comma was printed BEFORE the number ("Olszewski,¹ Siadat"), followed
    by the next author's name.
    """
    glued = attached_word(line.chars, start)
    if not _NAME_LIKE.fullmatch(glued.rstrip(".,;:)")):
        return False
    run = "".join(c.text for c in line.chars[start:end])
    after = printed_after(line, end)
    if run.endswith(",") or _AFTER_NAME_MARK.match(after) or _AFTER_SIGN_MARK.match(after):
        return True                                      # a raised comma belongs to the run
    return glued.endswith(",") and _NEXT_AUTHOR.match(after) is not None


def printed_after(line: LayoutLine, end: int) -> str:
    """The text printed after position `end` of a line, spaces kept where the PDF has them."""
    return "".join((" " + c.text) if c.space_before else c.text for c in line.chars[end:]).lstrip()


def drop_out_of_order_name_runs(candidates: Sequence[RaisedCandidate]) -> tuple:
    """(markers kept, how many were dropped) after the order check on name-attached runs.

    Walks the markers in reading order; a name run that cites a number for the
    first time out of order is dropped, and with it every name run on its page.
    Dropping a page changes what was "seen", so the walk restarts until no new
    page is condemned.
    """
    bad_pages: Set[int] = set()
    while True:
        kept, new_bad_page = walk_for_order(candidates, bad_pages)
        if new_bad_page is None:
            return [c.marker for c in kept], len(candidates) - len(kept)
        bad_pages.add(new_bad_page)


def walk_for_order(candidates: Sequence[RaisedCandidate], bad_pages: Set[int]) -> tuple:
    """One walk: (candidates kept, the first page newly found to hold a bad name run, or None).

    A name run is bad when it cites a number for the first time out of order,
    or when it is redundant (the paper cites the same numbers in order without it).
    """
    kept: List[RaisedCandidate] = []
    seen: Set[int] = set()
    runs = group_name_runs(candidates)
    for position, run in enumerate(runs):
        if run[0].name_attached and run[0].marker.page in bad_pages:
            continue
        numbers = run_numbers(run)
        if run[0].name_attached and (breaks_order(numbers, seen)
                                     or is_redundant_run(runs, position, seen, bad_pages)):
            return kept, run[0].marker.page
        kept.extend(run)
        seen.update(numbers)
    return kept, None


def run_numbers(run: Sequence[RaisedCandidate]) -> List[int]:
    """Every entry number cited by a run of markers, in printed order."""
    return [n for candidate in run for n in candidate.marker.numbers]


def is_redundant_run(runs: Sequence[Sequence[RaisedCandidate]], position: int, seen: Set[int],
                     bad_pages: Set[int]) -> bool:
    """Whether the name run at `position` is not needed: the body cites its numbers in order without it.

    Only a run that makes the FIRST appearance of number 1 can be redundant
    (affiliations always include 1; a real citation of entry 1 opens the paper's
    citations). The walk goes on through the later markers, leaving out name
    runs on this run's page and on condemned pages, with `seen` as it was
    before the run. If every first appearance of the run turns up in order
    there, the run was only affiliation marks.
    """
    run = runs[position]
    if 1 in seen or 1 not in run_numbers(run):
        return False
    needed = set(run_numbers(run)) - seen
    ahead, page = set(seen), run[0].marker.page
    for later in runs[position + 1:]:
        if later[0].name_attached and (later[0].marker.page == page or later[0].marker.page in bad_pages):
            continue
        later_numbers = run_numbers(later)
        if breaks_order(later_numbers, ahead):
            return False
        ahead.update(later_numbers)
        if needed <= ahead:
            return True
    return False


def group_name_runs(candidates: Sequence[RaisedCandidate]) -> List[List[RaisedCandidate]]:
    """Consecutive name-attached markers of one line form a run; every other marker stands alone."""
    groups: List[List[RaisedCandidate]] = []
    for candidate in candidates:
        previous = groups[-1] if groups else None
        if (previous and candidate.name_attached and previous[-1].name_attached
                and previous[-1].line_index == candidate.line_index):
            previous.append(candidate)
        else:
            groups.append([candidate])
    return groups


def breaks_order(numbers: Sequence[int], seen: Set[int]) -> bool:
    """Whether a first appearance in `numbers` jumps more than one past the highest number cited so far."""
    running_top = max(seen) if seen else 0
    met: Set[int] = set()
    for number in numbers:
        if number in seen or number in met:
            continue
        met.add(number)
        if number > running_top + 1:
            return True
        running_top = max(running_top, number)
    return False


def raised_digit_runs(line: LayoutLine) -> List[tuple]:
    """(start, end) positions in `line.chars` of each run of raised digits and joiners.

    A range's dash is often set at its own height ("26−28" with a full-size
    minus between raised digits), so a joiner BETWEEN two raised digits
    belongs to the run whatever its height.
    """
    runs, start = [], None
    chars = line.chars
    for index, char in enumerate(chars):
        raised_part = is_raised(char, line) and (char.text.isdigit() or char.text in _JOINERS)
        bridging = (start is not None and char.text in _JOINERS and index + 1 < len(chars)
                    and chars[index + 1].text.isdigit() and is_raised(chars[index + 1], line))
        if raised_part or bridging:
            start = index if start is None else start
            continue
        if start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(line.chars)))
    return [run for run in runs if any(c.text.isdigit() for c in line.chars[run[0]:run[1]])]


def is_raised(char: PrintedChar, line: LayoutLine) -> bool:
    """Whether a character is set above its line and smaller than its letters."""
    return (char.baseline < line.baseline - MIN_RISE_SHARE * line.height
            and char.rect.height <= MAX_RAISED_HEIGHT_SHARE * line.height)


def superscript_marker(line: LayoutLine, start: int, end: int) -> Optional[CitationMarker]:
    """The marker for one raised run, or None when the run is a power, charge or isotope."""
    chars = line.chars
    run = "".join(c.text for c in chars[start:end])
    if run[0] in _DASHES:
        return None                                    # "cm⁻¹": an exponent
    following = chars[end] if end < len(chars) else None
    if following is not None and following.text in "+⁺" and is_raised(following, line):
        return None                                    # "Zn²⁺": a charge
    if following is not None and following.text.isupper() and following.rect.x0 - chars[end - 1].rect.x1 < 1.0:
        return None                                    # "¹H", "¹⁵N": an isotope
    word = attached_word(chars, start)
    bare = word.strip(_DASHES)
    if word.rstrip(_DASHES) != word or "=" in word:
        return None                                    # "cm−¹" with a full-size minus; a formula
    if bare.lower() in _UNITS or re.fullmatch(r"[A-Z][a-z]?|\d+", bare):
        return None                                    # "μm²", "R²", "Zn²", "10³": a power
    if chars[start].space_before and re.fullmatch(r"[A-Z][a-z\u00C0-\u024F'’\-]+", word.strip()):
        return None                                    # "Haocheng Quan ¹": an affiliation mark
    numbers = expand(run.strip(_JOINERS))
    if not numbers:
        return None
    before = "".join(c.text if not c.space_before else " " + c.text for c in chars[max(0, start - CONTEXT_CHARS):start])
    return CitationMarker(page=line.page, text=run.strip(_JOINERS), kind="superscript",
                          numbers=numbers, context=before.strip())


def attached_word(chars: Sequence[PrintedChar], start: int) -> str:
    """The letters printed directly before position `start`, back to the last space."""
    letters: List[str] = []
    index = start - 1
    while index >= 0:
        char = chars[index]
        letters.append(char.text)
        if char.space_before:
            break
        index -= 1
    return "".join(reversed(letters))


def expand(spec: str) -> List[int]:
    """"3,5–7" -> [3, 5, 6, 7]. A malformed piece, or a range running backwards, gives []."""
    numbers: List[int] = []
    for piece in re.split(r"\s*,\s*", spec.strip()):
        bounds = [b for b in re.split(r"\s*[" + _DASH_CLASS + r"]\s*", piece) if b]
        if len(bounds) == 1 and bounds[0].isdigit():
            numbers.append(int(bounds[0]))
        elif len(bounds) == 2 and all(b.isdigit() for b in bounds) and int(bounds[0]) < int(bounds[1]):
            numbers.extend(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            return []
    return numbers
