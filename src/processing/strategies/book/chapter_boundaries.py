#!/usr/bin/env python3
"""
Which chapter is each page in? — the boundary problem.

For biology researchers: this is what lets the system cite a book correctly.
In an edited volume every chapter has different authors, so a passage
retrieved from PDF page 60 must be attributed to *that chapter's* authors. The
contents page already tells us chapter 3 is by Siadat et al.
(`toc.py`); what is missing is the fact that PDF page 60 is *in* chapter 3.

That sounds like arithmetic and is not. The contents page gives start pages in
the book's **printed** numbering, which does not line up with PDF page indices
— the front matter offsets them, and the offset **drifts** as unnumbered
plates and part-title pages are inserted. Measured on the sample volume: the
offset is 7 at chapter 3 and 6 at chapter 4. A single global offset is wrong.

So we do not compute the mapping — we **read** it, from what the publisher
already prints on every page:

    idx 51  last line: '2 Basic Structure, Physiology, and Biochemistry…'   <- recto
    idx 52  last line: '3'                                                  <- opening page
    idx 53  last line: 'S. M. Siadat et al.'                                <- verso
    idx 54  last line: '3 Tendon Extracellular Matrix Assembly…'            <- recto

A **ladder** of three signals, strongest first — each rung only fills what the
rung above left unresolved:

1. **Recto running header** — the chapter number, printed outright. 140/345
   pages on the sample.
2. **Verso running header** — the chapter's authors in short form
   ("S. M. Siadat et al."), matched against the contents page's author lists.
   A further 111 pages. This is why capturing TOC authors first was worth it.
3. **Gap assignment and interpolation** — everything between two resolved
   pages of the same chapter belongs to that chapter; a gap *between* two
   chapters belongs to the **later** one, because a chapter's opening page
   carries no running header. A chapter with no resolved page at all (chapter
   1 on the sample, which is short) is placed by printed-page arithmetic
   anchored on its nearest confirmed neighbour — locally, never globally.

**Deliberately not used: the printed folio.** Reading the bare page number and
inferring an offset measures badly — on the sample, candidate offsets of 5, 7,
3, 6 and 8 all polled strongly, because bare numbers also appear in table
cells, and a chapter opening page's trailing bare number is the *chapter*
number, not the folio. It is a plausible-looking signal that is wrong often
enough to poison attribution, which is worse than admitting ignorance.

**Scope.** Every measurement here is from one publisher (Springer). The
structure of the ladder is general — recto number, verso author, interpolate —
but the regexes are not guaranteed to be. `detect_chapter_spans` reports its
own coverage so a caller can tell a confident mapping from a guess, and
`ChapterSpan.confidence` says which rung placed each span.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .toc import TocEntry

LOG = logging.getLogger(__name__)

# A recto running header: a chapter number, then that chapter's title. Both
# halves are checked against the contents page — the number must BE a chapter
# and the text must match ITS title. That is what separates a running header
# from a numbered list item ("3 mL of buffer was added"), and it is stricter
# and more general than the obvious alternative of demanding a long title:
# chapter 1 of the sample volume is called "Introduction", twelve characters,
# and any length threshold that excludes list items also excludes it.
RECTO_HEADER = re.compile(r'^(?P<number>\d{1,3})\s+(?P<title>\S.*)$')

# A chapter's opening page carries no running header — it carries the
# publisher's copyright line and the bare chapter number. Accepted only when
# corroborated (see `_apply_opening_pages`), because a bare number is also
# what a folio looks like.
BARE_NUMBER = re.compile(r'^(?P<number>\d{1,3})$')

# How much of a running header's title must match the contents entry. Running
# headers are routinely truncated to fit the page width, so only a prefix can
# be compared.
#
# Known limitation, verified rather than assumed: the text layer drops
# ligatures ("fbrils" for "fibrils"), and a ligature inside the first twelve
# characters WILL defeat this comparison. It costs one page's rung-1 match,
# which rung 2 or the gap logic then covers, so it is a coverage cost and not
# a correctness one. Loosening the comparison to absorb it would start
# matching the wrong chapters, which is the failure that actually matters.
TITLE_PREFIX_CHARS = 12

# A verso running header in short author form: "S. M. Siadat et al.",
# "J. Halper", "Van Laer and Loeys".
VERSO_HEADER = re.compile(
    r'^(?:[A-Z]\.\s*)*(?P<surname>[A-Z][\w\'’-]+)(?:\s+et\s*\.?\s*al\.?)?$')

# Below this fraction of body pages resolved by rungs 1-2, the mapping is
# reported as low confidence: too much of it rests on interpolation.
MIN_DIRECT_COVERAGE = 0.40


@dataclass
class ChapterSpan:
    """
    One chapter's extent in the PDF, and how surely we know it.

    Attributes:
        number: chapter number from the contents page.
        title / authors: carried through from the contents entry, so this is
            the whole citation record for any page in the span.
        start_index / end_index: 0-based PDF page indices, inclusive.
        confidence: "header" when a running header placed both ends,
            "interpolated" when a gap did, "arithmetic" when the chapter had
            no resolved page at all and printed-page maths placed it.
    """
    number: int
    title: str
    authors: List[str] = field(default_factory=list)
    start_index: int = 0
    end_index: int = 0
    confidence: str = "header"

    def contains(self, page_index: int) -> bool:
        """True when this 0-based PDF page falls inside the chapter."""
        return self.start_index <= page_index <= self.end_index


@dataclass
class ChapterMap:
    """Every chapter's span, plus how much of it was read rather than inferred."""
    spans: List[ChapterSpan] = field(default_factory=list)
    pages_resolved_directly: int = 0
    body_pages: int = 0

    @property
    def direct_coverage(self) -> float:
        """Fraction of body pages a running header identified outright."""
        if not self.body_pages:
            return 0.0
        return self.pages_resolved_directly / self.body_pages

    @property
    def is_confident(self) -> bool:
        """False when too much of the mapping rests on interpolation."""
        return bool(self.spans) and self.direct_coverage >= MIN_DIRECT_COVERAGE

    def chapter_for_page(self, page_index: int) -> Optional[ChapterSpan]:
        """
        The chapter a 0-based PDF page belongs to, or None for front matter.

        This is the call a retrieval index makes to attribute a passage.
        """
        for span in self.spans:
            if span.contains(page_index):
                return span
        return None


def detect_chapter_spans(page_texts: Dict[int, str], entries: Sequence[TocEntry],
                         first_body_index: int = 0) -> ChapterMap:
    """
    Work out which pages belong to which chapter.

    Args:
        page_texts: text layer of the whole book, keyed by **0-based** PDF page
            index. Pages may be missing; they are treated as unresolved.
        entries: the contents-page entries from `parse_toc_entries`, which
            supply chapter numbers, titles and authors.
        first_body_index: the first page after the front matter. Pages before
            it are never assigned to a chapter.

    Returns:
        A `ChapterMap`. Empty spans when the contents page yielded no numbered
        chapters — this function never invents a chapter the book did not list.

    Example:
        >>> chapter_map = detect_chapter_spans(texts, entries)   # doctest: +SKIP
        >>> chapter_map.chapter_for_page(60).authors[0]
        'Seyed Mohammad Siadat'
    """
    chapters = [entry for entry in entries if entry.number is not None]
    if not chapters:
        LOG.info("No numbered chapters in the contents; no boundaries to detect.")
        return ChapterMap()

    surname_index = _index_authors_by_surname(chapters)
    title_index = _index_titles_by_number(chapters)
    resolved = _resolve_pages(page_texts, surname_index, title_index, first_body_index)
    direct_pages = len(resolved)
    _apply_opening_pages(page_texts, resolved, title_index, first_body_index)

    last_index = max(page_texts) if page_texts else first_body_index
    spans = _spans_from_resolved(chapters, resolved, first_body_index, last_index)

    body_pages = sum(1 for index in page_texts if index >= first_body_index)
    chapter_map = ChapterMap(spans=spans, body_pages=body_pages,
                             pages_resolved_directly=direct_pages)

    LOG.info("Chapter boundaries: %d chapters, %.0f%% of body pages identified "
             "by a running header (%s)", len(spans), 100 * chapter_map.direct_coverage,
             "confident" if chapter_map.is_confident else "LOW CONFIDENCE")
    return chapter_map


# --- the ladder (helpers, after the main function per CLAUDE.md) ------------

def _resolve_pages(page_texts: Dict[int, str], surname_index: Dict[str, int],
                   title_index: Dict[int, str],
                   first_body_index: int) -> Dict[int, int]:
    """
    Rungs 1 and 2: the pages a running header identifies outright.

    Returns:
        {0-based page index: chapter number} for those pages only. Everything
        else is left for the gap logic, which is honest about being inference.
    """
    resolved: Dict[int, int] = {}

    for index in sorted(page_texts):
        if index < first_body_index:
            continue
        header_lines = _header_lines(page_texts[index])
        number = _chapter_from_recto(header_lines, title_index)
        if number is None:
            number = _chapter_from_verso(header_lines, surname_index)
        if number is not None:
            resolved[index] = number

    return resolved


def _apply_opening_pages(page_texts: Dict[int, str], resolved: Dict[int, int],
                         title_index: Dict[int, str], first_body_index: int) -> None:
    """
    Rung 3: a chapter's opening page, which carries a bare chapter number.

    The opening page has no running header — it has the publisher's copyright
    line and the chapter number alone. That bare number is worth reading
    because it marks the boundary *exactly*, but on its own it is
    indistinguishable from a page folio, so it is accepted only when the next
    already-resolved page belongs to the same chapter. Self-corroborating:
    a folio would have to coincide with its own chapter's number to slip
    through, and it would then be assigning the page to the chapter it is
    already in.

    Mutates `resolved` in place. Deliberately not counted toward direct
    coverage — it is a boundary marker, not an identification of content.
    """
    for index in sorted(page_texts):
        if index < first_body_index or index in resolved:
            continue

        number = _bare_chapter_number(_header_lines(page_texts[index]), title_index)
        if number is None:
            continue
        if _next_resolved_chapter(resolved, index) == number:
            resolved[index] = number


def _bare_chapter_number(header_lines: Sequence[str],
                         title_index: Dict[int, str]) -> Optional[int]:
    """A header line that is nothing but a number naming a real chapter."""
    for line in header_lines:
        match = BARE_NUMBER.match(line)
        if match and int(match.group("number")) in title_index:
            return int(match.group("number"))
    return None


def _next_resolved_chapter(resolved: Dict[int, int], index: int) -> Optional[int]:
    """The chapter of the nearest resolved page after `index`."""
    later = [page for page in resolved if page > index]
    return resolved[min(later)] if later else None


def _header_lines(text: str) -> List[str]:
    """
    The lines a running header could be on: the first and last of the page.

    Which one it is depends on the publisher and on whether the page is a
    recto or a verso, so both are offered and the patterns decide.
    """
    lines = [line.strip().replace("\xa0", " ")
             for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return []
    return [lines[-1], lines[0]] if len(lines) > 1 else [lines[0]]


def _chapter_from_recto(header_lines: Sequence[str],
                        title_index: Dict[int, str]) -> Optional[int]:
    """
    Rung 1: the chapter number printed at the head of its own title.

    Both halves must agree with the contents page — a number that names a real
    chapter, followed by text matching *that* chapter's title. Checking the
    title is what makes this safe without a length heuristic.
    """
    for line in header_lines:
        match = RECTO_HEADER.match(line)
        if not match:
            continue
        number = int(match.group("number"))
        expected = title_index.get(number)
        if expected and _titles_agree(match.group("title"), expected):
            return number
    return None


def _index_titles_by_number(chapters: Sequence[TocEntry]) -> Dict[int, str]:
    """Chapter number -> its normalised contents-page title."""
    return {entry.number: _normalize_title(entry.title) for entry in chapters}


def _normalize_title(title: str) -> str:
    """
    Fold a title for comparison: lowercase, letters and digits only.

    The text layer mangles ligatures — "fibrils" comes out as "fbrils" — and
    running headers drop punctuation and non-breaking spaces the contents page
    keeps. Comparing only alphanumerics sidesteps all of it.
    """
    return re.sub(r'[^a-z0-9]+', "", (title or "").lower())


def _titles_agree(header_title: str, expected: str) -> bool:
    """
    True when a header's title is the start of the chapter's real title.

    Prefix comparison, because running headers are routinely truncated to fit
    the page width.
    """
    seen = _normalize_title(header_title)
    if not seen or not expected:
        return False
    length = min(TITLE_PREFIX_CHARS, len(seen), len(expected))
    return seen[:length] == expected[:length]


def _chapter_from_verso(header_lines: Sequence[str],
                        surname_index: Dict[str, int]) -> Optional[int]:
    """
    Rung 2: the chapter's authors in short form, matched to the contents page.

    Only an **unambiguous** surname counts. An editor who also wrote several
    chapters (Halper wrote four of the sample's fifteen) appears under more
    than one, and guessing between them would attribute a passage to the wrong
    chapter — the exact failure this module exists to prevent.
    """
    for line in header_lines:
        match = VERSO_HEADER.match(line)
        if not match:
            continue
        number = surname_index.get(match.group("surname").lower())
        if number is not None:
            return number
    return None


def _index_authors_by_surname(chapters: Sequence[TocEntry]) -> Dict[str, int]:
    """
    Surname -> chapter number, keeping only surnames unique to one chapter.

    A surname appearing in two chapters is dropped rather than resolved
    arbitrarily; rung 3 will place those pages from their neighbours.
    """
    seen: Dict[str, set] = {}
    for entry in chapters:
        for author in entry.authors:
            parts = author.split()
            if parts:
                seen.setdefault(parts[-1].lower(), set()).add(entry.number)

    return {surname: next(iter(numbers))
            for surname, numbers in seen.items() if len(numbers) == 1}


def _spans_from_resolved(chapters: Sequence[TocEntry], resolved: Dict[int, int],
                         first_body_index: int, last_index: int) -> List[ChapterSpan]:
    """
    Rung 3: turn resolved pages into contiguous spans covering the whole body.

    A chapter starts where the previous one stopped being seen — the gap goes
    to the LATER chapter, because a chapter's opening page carries no running
    header (it carries the copyright line and the bare chapter number instead).
    """
    ordered = sorted(chapters, key=lambda entry: entry.number)
    starts = _chapter_start_indices(ordered, resolved, first_body_index)

    spans: List[ChapterSpan] = []
    for position, entry in enumerate(ordered):
        start = starts.get(entry.number)
        if start is None:
            continue
        end = _chapter_end_index(ordered, starts, position, last_index)
        if end < start:
            continue
        spans.append(ChapterSpan(
            number=entry.number, title=entry.title, authors=list(entry.authors),
            start_index=start, end_index=end,
            confidence=("header" if entry.number in set(resolved.values())
                        else "arithmetic"),
        ))

    return spans


def _chapter_start_indices(ordered: Sequence[TocEntry], resolved: Dict[int, int],
                           first_body_index: int) -> Dict[int, int]:
    """Where each chapter begins, by observation where possible."""
    first_seen: Dict[int, int] = {}
    last_seen: Dict[int, int] = {}
    for index, number in sorted(resolved.items()):
        first_seen.setdefault(number, index)
        last_seen[number] = index

    # Pass 1 — chapters a header actually named. A chapter begins where the
    # previous one stopped being seen, so the gap between them falls to the
    # later chapter (its opening pages carry no running header). The pull-back
    # only applies when the PREVIOUS chapter was also observed; otherwise
    # there is no evidence the gap is not the previous chapter's body, and
    # claiming it would swallow a whole short chapter.
    starts: Dict[int, int] = {}
    for position, entry in enumerate(ordered):
        if entry.number not in first_seen:
            continue
        previous = ordered[position - 1].number if position else None
        if previous in last_seen:
            starts[entry.number] = max(first_body_index,
                                       min(first_seen[entry.number],
                                           last_seen[previous] + 1))
        else:
            starts[entry.number] = max(first_body_index, first_seen[entry.number])

    # Pass 2 — chapters no header ever named. Deferred to here because the
    # arithmetic anchors on a neighbour's start, which pass 1 has to have
    # computed first.
    for position, entry in enumerate(ordered):
        if entry.number in starts:
            continue
        placed = _place_by_printed_arithmetic(ordered, position, starts)
        # NOT clamped into range, and refused outright if it collides with a
        # chapter a running header already placed. Arithmetic landing before
        # the body began means the chapter is not in this PDF (an extract
        # starting mid-book, or a bad anchor); landing on top of an observed
        # neighbour means the anchor drifted. Either way the honest answer is
        # "not placed" — observation must outrank extrapolation, and a page
        # with no attribution is recoverable where a misattributed one is not.
        if placed is not None and _fits_between_neighbours(
                ordered, position, starts, placed, first_body_index):
            starts[entry.number] = placed

    return _drop_inconsistent_starts(ordered, starts)


def _fits_between_neighbours(ordered: Sequence[TocEntry], position: int,
                             starts: Dict[int, int], candidate: int,
                             first_body_index: int) -> bool:
    """
    True when an arithmetic placement does not collide with a placed chapter.

    A chapter must begin after the body starts, after every earlier chapter
    already placed, and before every later one.
    """
    if candidate < first_body_index:
        return False

    for earlier in ordered[:position]:
        start = starts.get(earlier.number)
        if start is not None and candidate <= start:
            return False

    for later in ordered[position + 1:]:
        start = starts.get(later.number)
        if start is not None and candidate >= start:
            return False

    return True


def _drop_inconsistent_starts(ordered: Sequence[TocEntry],
                              starts: Dict[int, int]) -> Dict[int, int]:
    """
    Keep only starts that increase with the chapter number.

    Chapters do not overlap and do not run backwards, so a start that is not
    strictly greater than the previous chapter's is evidence the placement is
    wrong — most often the arithmetic fallback extrapolating off the front of
    a short document. Such a chapter is DROPPED rather than overlapped onto
    its neighbour: a page with no attribution is recoverable, a page
    attributed to the wrong authors is not.
    """
    consistent: Dict[int, int] = {}
    previous_start: Optional[int] = None

    for entry in ordered:
        start = starts.get(entry.number)
        if start is None:
            continue
        if previous_start is not None and start <= previous_start:
            LOG.debug("Dropping chapter %s: start %d does not follow %d",
                      entry.number, start, previous_start)
            continue
        consistent[entry.number] = start
        previous_start = start

    return consistent


def _place_by_printed_arithmetic(ordered: Sequence[TocEntry], position: int,
                                 starts: Dict[int, int]) -> Optional[int]:
    """
    Last resort: place a chapter no header ever named, from its neighbour.

    The printed offset drifts across a book (measured 7 at chapter 3, 6 at
    chapter 4), so this is anchored on the NEAREST located chapter and never
    on a global offset. It applies to short chapters that carry no running
    header at all — chapter 1 of the sample volume, four pages long.
    """
    entry = ordered[position]
    if entry.start_page is None:
        return None

    for neighbour in _nearest_located_neighbours(ordered, position, starts):
        if neighbour.start_page is None:
            continue
        return starts[neighbour.number] + (entry.start_page - neighbour.start_page)

    return None


def _nearest_located_neighbours(ordered: Sequence[TocEntry], position: int,
                                starts: Dict[int, int]) -> List[TocEntry]:
    """Located chapters, nearest first, for the arithmetic fallback to anchor on."""
    candidates = []
    for distance in range(1, len(ordered)):
        for neighbour_position in (position - distance, position + distance):
            if 0 <= neighbour_position < len(ordered):
                neighbour = ordered[neighbour_position]
                if neighbour.number in starts:
                    candidates.append(neighbour)
    return candidates


def _chapter_end_index(ordered: Sequence[TocEntry], starts: Dict[int, int],
                       position: int, last_index: int) -> int:
    """A chapter runs up to the page before the next chapter starts."""
    for next_entry in ordered[position + 1:]:
        next_start = starts.get(next_entry.number)
        if next_start is not None:
            return next_start - 1
    return last_index
