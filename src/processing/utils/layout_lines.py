#!/usr/bin/env python
"""Printed lines in reading order, rebuilt from a page's characters.

For biology researchers: a reference list is read line by line, and where a
line STARTS matters as much as what it says. A new entry begins with its
number ("245 Doi, M., ...") or, in an author-year list, at the column's left
edge while the entry's other lines are indented. So the lines must be exactly
as printed: one per column, in reading order, with their left edge.

WHY NOT THE EXISTING LINE READERS

`styled_lines()` groups characters by the bottom of their ink, so the tails
of "g", "p" and "y" become lines of their own, and it joins neighbouring
columns into one line. The PDF's own text order is not reliable either: some
publishers write a page row by row across both columns. So lines are rebuilt
here from the characters' BASELINES (the line each letter sits on), cut at
column gutters, and put in reading order.

COLUMNS ARE LOCAL

A page can change layout halfway down (a Nature review's last page: two
columns of prose above a three-column reference list). So a gutter is found
for each ROW from its neighbouring rows: a vertical strip that none of them
puts a character in. Rows with the same gutters form a BAND; reading order is
band by band, then column by column, then top to bottom.

Pure geometry on `PrintedChar`s: no model, no network.
"""
from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..pdf.base import PrintedChar

# A band of rows with fewer gutters is joined to its neighbour only when its
# text stays out of the neighbour's extra gutters. When more than this share
# of its rows put ink in such a gutter, it is full-width text (a Cell Reports
# SUMMARY above a two-column INTRODUCTION), a different layout.
FULL_WIDTH_SHARE = 0.25

# ... and never on fewer rows than this. One row across a gutter is weak
# evidence: in a scan's text layer, the boxes of one column line can run
# across the gutter (the 1960 Saturated Solutions scan, p6), and taking that
# row for full-width text cut its column in two.
MIN_FULL_WIDTH_ROWS = 2

# ... and only when a band WITH those gutters is a real column layout, at
# least this many rows tall. A few rows of text wrapping round a small inset
# (Frontiers) show a local extra gutter that the column above runs across.
MIN_LAYOUT_ROWS = 12

# How far left of a gutter's position (where the next column starts) ink is
# looked for: inside the empty strip of a real gutter, inside the text of a
# full-width row.
GUTTER_PROBE_POINTS = (4.0, 1.0)

# Two characters share a line when their baselines differ by less than this
# share of the line's character height. Superscript digits rise ~0.3-0.4 of
# it, so they stay on their line; the next line is a full line height away.
SAME_LINE_SHARE = 0.5

# A horizontal gap wider than this many character heights is never a space
# between words, even where no gutter was found.
GUTTER_HEIGHTS = 1.6

# Characters may overlap their left neighbour by up to this many character
# heights and still continue the line: a ligature ("fl" in "Knoepfler") is
# one wide glyph that the text layer reports as two overlapping letters.
OVERLAP_HEIGHTS = 1.0

# A gap wider than this share of the character height is a word space even
# when the PDF records no space character.
SPACE_SHARE = 0.18

# A gutter: an empty vertical strip at least this wide (an ACS page's
# justified columns are only ~3.5 pt apart)...
MIN_GUTTER_POINTS = 2.5
# ...measured in strips of this many points...
STRIP_POINTS = 0.5
# ...looked for over this many rows above and below a row (a short window
# of justified prose can leave a narrow empty strip by chance)...
ROW_WINDOW = 10
# ...with columns on both sides at least this share of the page's text width
# (so the narrow column of entry NUMBERS beside a list is not a column).
MIN_COLUMN_SHARE = 0.15

# A gutter is recorded as the x where the next column's text begins: lists
# are ragged on the right, so the empty strip's middle drifts from row to row,
# while left-aligned text starts at the same x on every row. Gutter sets that
# differ by more than this share of the text width start a new band; less
# is the column of entry numbers ("100" starts left of its entry's text).
BAND_SHIFT_SHARE = 0.06

# A gutter needs text on BOTH sides, within `GUTTER_SIDE_POINTS` of it, in at
# least this share of the window's rows: an empty strip inside a column that
# most rows leave blank (a short last column) is not a gutter. A quarter,
# because neighbouring columns rarely share baselines, so each side of a real
# gutter holds only about half of the rows.
GUTTER_SUPPORT_SHARE = 0.25
# ...but never more than this many rows: a short last column (three entries
# on a list's final page) is still a column.
MIN_SUPPORT_ROWS = 3
GUTTER_SIDE_POINTS = 40.0

# A band's columns are the gutters at least this share of its rows found.
BAND_CONSENSUS_SHARE = 0.4

# Running headers and footers live in these top and bottom shares of a page.
FURNITURE_BAND = 0.10

# A recurring line in that band is furniture only if it holds this many letters.
MIN_FURNITURE_LETTERS = 3

# Spacing accents that some PDFs print as separate glyphs after their letter
# ("Ma¨ki"), mapped to the combining mark that joins them ("Mäki").
_SPACING_ACCENTS = {
    "´": "́", "¨": "̈", "`": "̀", "ˆ": "̂",
    "˜": "̃", "¸": "̧", "ˇ": "̌", "˚": "̊",
}


@dataclass
class LayoutLine:
    """One printed line of one column.

    Attributes:
        page: 1-based page number.
        band: 0 for the top layout band of its page (see the module note).
        column: 0 for the leftmost column of its band.
        text: the line as printed, words separated by single spaces.
        x0, x1: its left and right edges (points).
        baseline: the y its letters sit on (top-left coordinates).
        height: the median loose height of its characters (about one em).
        chars: its characters, left to right.
    """

    page: int
    band: int
    column: int
    text: str
    x0: float
    x1: float
    baseline: float
    height: float
    chars: List[PrintedChar] = field(repr=False)


def page_layout_lines(characters: Sequence[PrintedChar], page_number: int,
                      page_height: float = 0.0) -> List[LayoutLine]:
    """A page's printed lines, one per column, in reading order.

    Args:
        characters: the page's `PdfPage.characters()`.
        page_number: 1-based, stored on each line.
        page_height: the page's height in points; when given, the running
            header and footer rows take no part in finding the columns.

    Returns:
        Lines sorted by band, column, then top to bottom. Turned characters
        (margin stamps) and lines with no letter or digit are left out.
    """
    upright = [c for c in characters if not c.turned]
    if not upright:
        return []
    layout = RowLayout(upright, page_height)
    segments = merge_neighbours(split_into_segments(upright, layout), layout)
    lines = [build_line(segment, page_number) for segment in segments]
    lines = [line for line in lines if any(ch.isalnum() for ch in line.text)]
    for line in lines:
        line.band, line.column = layout.place(line.baseline, line.x0)
    split_columns_left_unfound(lines)
    return sorted(lines, key=lambda line: (line.band, line.column, line.baseline, line.x0))


def split_columns_left_unfound(lines: List[LayoutLine]) -> None:
    """Give a column of its own to lines that start right of where their column's lines end.

    A column only a few lines tall (the last three entries of a list, on its
    final page) leaves too few rows to show a gutter. But a line cannot start
    to the right of where most lines of its own column END: such lines belong
    to the next column, and so do any that start further right still.
    """
    groups: Dict[Tuple[int, int], List[LayoutLine]] = {}
    for line in lines:
        groups.setdefault((line.band, line.column), []).append(line)
    for group in groups.values():
        if len(group) < 3:
            continue
        usual_end = statistics.median(line.x1 for line in group)
        for line in group:
            if line.x0 >= usual_end:
                line.column += 1


def document_layout_lines(pages_characters: Iterable[Sequence[PrintedChar]],
                          page_heights: Sequence[float]) -> List[LayoutLine]:
    """Every page's lines in reading order, running headers and footers removed."""
    lines: List[LayoutLine] = []
    for index, characters in enumerate(pages_characters):
        lines.extend(page_layout_lines(characters, index + 1, page_heights[index]))
    return drop_running_furniture(lines, page_heights)


class RowLayout:
    """The gutters and band of every row of one page.

    A row is the characters sharing a rounded baseline across the whole page.
    """

    def __init__(self, characters: Sequence[PrintedChar], page_height: float = 0.0) -> None:
        rows: Dict[int, List[PrintedChar]] = {}
        for char in characters:
            rows.setdefault(round(char.baseline), []).append(char)
        self.keys = sorted(rows)
        text_width = max(c.rect.x1 for c in characters) - min(c.rect.x0 for c in characters)
        tolerance = BAND_SHIFT_SHARE * text_width
        row_gutters = body_rows_gutters(rows, self.keys, page_height)
        first_bands = bands_of(row_gutters, tolerance)
        spans = FullWidthTest([rows[key] for key in self.keys], first_bands)
        self.bands = merge_compatible_bands(first_bands, band_consensus(first_bands, row_gutters, tolerance),
                                            tolerance, spans)
        consensus = band_consensus(self.bands, row_gutters, tolerance)
        self.gutters = [consensus[band] for band in self.bands]

    def row_index(self, baseline: float) -> int:
        """The row nearest a baseline."""
        target = round(baseline)
        return min(range(len(self.keys)), key=lambda i: abs(self.keys[i] - target))

    def gutters_at(self, baseline: float) -> Tuple[float, ...]:
        return self.gutters[self.row_index(baseline)]

    def place(self, baseline: float, x0: float) -> Tuple[int, int]:
        """(band, column) of a line starting at x0 on this baseline."""
        index = self.row_index(baseline)
        return self.bands[index], sum(1 for gutter in self.gutters[index] if x0 >= gutter - 0.5)


def band_consensus(bands: List[int], gutters: List[Tuple[float, ...]],
                   tolerance: float) -> Dict[int, Tuple[float, ...]]:
    """Each band's gutters: those found by at least `BAND_CONSENSUS_SHARE` of its rows.

    Gutters within `tolerance` of each other are one gutter, placed at its
    EARLIEST start (where a column's entry numbers begin). A single row
    cannot add a column, and rows with fewer gutters (a running header, the
    rows below a column that ended early) cannot remove one.
    """
    rows_by_band: Dict[int, List[Tuple[float, ...]]] = {}
    for band, row in zip(bands, gutters):
        rows_by_band.setdefault(band, []).append(row)
    consensus: Dict[int, Tuple[float, ...]] = {}
    for band, rows in rows_by_band.items():
        clusters: List[List[float]] = []
        for row in rows:
            for gutter in row:
                home = next((c for c in clusters if abs(c[0] - gutter) <= tolerance), None)
                (home.append(gutter) if home is not None else clusters.append([gutter]))
        needed = BAND_CONSENSUS_SHARE * len(rows)
        consensus[band] = tuple(sorted(min(c) for c in clusters if len(c) >= needed))
    return consensus


def body_rows_gutters(rows: Dict[int, List[PrintedChar]], keys: List[int],
                      page_height: float) -> List[Tuple[float, ...]]:
    """Gutters of every row, found from the BODY rows only.

    Running header and footer rows (the top and bottom `FURNITURE_BAND` of
    the page) are often laid out differently (a page number left, a DOI
    right) and would split the bottom of a column into a band of its own.
    They take the gutters of the nearest body row instead.
    """
    def is_body(key: int) -> bool:
        return not page_height or FURNITURE_BAND * page_height <= key <= (1 - FURNITURE_BAND) * page_height

    body_keys = [key for key in keys if is_body(key)] or keys
    body_chars = [char for key in body_keys for char in rows[key]]
    body_gutters = dict(zip(body_keys, rows_gutters([rows[key] for key in body_keys], body_chars)))
    return [body_gutters[key] if key in body_gutters
            else body_gutters[min(body_keys, key=lambda b: abs(b - key))] for key in keys]


def rows_gutters(rows: List[List[PrintedChar]], characters: Sequence[PrintedChar]) -> List[Tuple[float, ...]]:
    """Each row's gutters, from the window above it, below it, or around it: whichever finds most.

    Taking the best of three windows matters where a page changes layout: a
    window straddling the change sees both layouts and so finds no gutter,
    while the window on the row's own side finds its columns.
    """
    width = int(max(c.rect.x1 for c in characters) / STRIP_POINTS) + 2
    covered = np.zeros((len(rows) + 1, width), dtype=np.int32)
    for index, row in enumerate(rows):
        for char in row:
            first = max(0, int(np.floor(char.rect.x0 / STRIP_POINTS)))
            last = int(np.ceil(char.rect.x1 / STRIP_POINTS))
            covered[index + 1, first:last] = 1
    running = covered.cumsum(axis=0)
    text_width = max(c.rect.x1 for c in characters) - min(c.rect.x0 for c in characters)
    last = len(rows) - 1
    result = []
    for index in range(len(rows)):
        windows = [(index - ROW_WINDOW, index), (index, index + ROW_WINDOW),
                   (index - ROW_WINDOW // 2, index + ROW_WINDOW // 2)]
        windows = [(max(0, lo), min(last, hi)) for lo, hi in windows]
        full = [(lo, hi) for lo, hi in windows if hi - lo >= min(ROW_WINDOW // 2, last)]
        found = [window_gutters(running, lo, hi, text_width) for lo, hi in full or [(0, last)]]
        result.append(max(found, key=len))
    return result


def window_gutters(running: np.ndarray, low: int, high: int, text_width: float) -> Tuple[float, ...]:
    """Gutters of rows low..high: empty strips with wide enough columns on both sides."""
    counts = running[high + 1] - running[low]
    busy = np.nonzero(counts)[0]
    if high - low < 2 or len(busy) == 0:
        return ()
    start, end = int(busy[0]), int(busy[-1])
    strips, run_start = [], None
    for x in range(start, end + 1):
        if counts[x] == 0:
            run_start = x if run_start is None else run_start
            continue
        if (run_start is not None and (x - run_start) * STRIP_POINTS >= MIN_GUTTER_POINTS
                and supported_on_both_sides(counts, run_start, x, high - low + 1)):
            strips.append(x * STRIP_POINTS)   # where the next column's text BEGINS
        run_start = None
    return keep_wide_columns(strips, start * STRIP_POINTS, end * STRIP_POINTS,
                             MIN_COLUMN_SHARE * text_width)


def supported_on_both_sides(counts: np.ndarray, gap_start: int, gap_end: int, rows: int) -> bool:
    """Whether most rows of the window have text just left AND just right of an empty strip."""
    reach = int(GUTTER_SIDE_POINTS / STRIP_POINTS)
    left = counts[max(0, gap_start - reach):gap_start]
    right = counts[gap_end:gap_end + reach]
    needed = min(GUTTER_SUPPORT_SHARE * rows, MIN_SUPPORT_ROWS)
    return len(left) > 0 and len(right) > 0 and left.max() >= needed and right.max() >= needed


def keep_wide_columns(strips: List[float], start: float, end: float, min_width: float) -> Tuple[float, ...]:
    """Drop strips that would leave a column narrower than `min_width` on either side."""
    kept: List[float] = []
    left_edge = start
    for strip in strips:
        if strip - left_edge >= min_width and end - strip >= min_width:
            kept.append(strip)
            left_edge = strip
    return tuple(kept)


def bands_of(gutters: List[Tuple[float, ...]], tolerance: float) -> List[int]:
    """Band number of each row: a new band wherever the gutters change."""
    bands, band, previous = [], 0, None
    for current in gutters:
        if previous is not None and not same_gutters(previous, current, tolerance):
            band += 1
        bands.append(band)
        previous = current
    return bands


def merge_compatible_bands(bands: List[int], band_gutters: Dict[int, Tuple[float, ...]],
                           tolerance: float,
                           spans: Optional["FullWidthTest"] = None) -> List[int]:
    """Join neighbouring bands when one's gutters are a subset of the other's.

    A column that ends early (the right column of a list's last page) leaves
    the rows below it with fewer gutters. That is the same layout, not a new
    one: its lines continue the left column. Only bands whose gutters
    disagree (two columns of prose above three of references) stay apart.

    `spans` (optional, a `FullWidthTest`) says whether a band's text runs
    across gutters. Such a band is full-width text, not a column that ended
    early, and is never joined to a band of `MIN_LAYOUT_ROWS` or more rows
    that has those gutters.
    """
    renumber: Dict[int, int] = {}
    group, group_gutters = 0, None
    group_bands: List[int] = []
    for band in sorted(band_gutters):
        current = band_gutters[band]
        if group_gutters is not None and _compatible(band, group_bands, group_gutters,
                                                     band_gutters, tolerance, spans):
            group_gutters = current if len(current) > len(group_gutters) else group_gutters
            group_bands.append(band)
        elif group_gutters is not None:
            group, group_gutters, group_bands = group + 1, current, [band]
        else:
            group_gutters, group_bands = current, [band]
        renumber[band] = group
    return [renumber[band] for band in bands]


def _compatible(band: int, group_bands: List[int], group_gutters: Tuple[float, ...],
                band_gutters: Dict[int, Tuple[float, ...]], tolerance: float,
                spans: Optional["FullWidthTest"]) -> bool:
    """Whether `band` may join the group of bands before it (see `merge_compatible_bands`)."""
    gutters = band_gutters[band]
    if subset_of(gutters, group_gutters, tolerance):
        fewer, more = [band], group_bands
        extra = tuple(g for g in group_gutters if not subset_of((g,), gutters, tolerance))
    elif subset_of(group_gutters, gutters, tolerance):
        fewer, more = group_bands, [band]
        extra = tuple(g for g in gutters if not subset_of((g,), group_gutters, tolerance))
    else:
        return False
    if not (spans and extra):
        return True
    # A layout is ONE band with those gutters, tall enough: three 2-row bands
    # of text wrapping round an inset are not one.
    tallest = max((spans.rows([b]) for b in more
                   if subset_of(extra, band_gutters[b], tolerance)), default=0)
    return not (tallest >= MIN_LAYOUT_ROWS and spans.crosses(fewer, extra))


class FullWidthTest:
    """Which first-pass bands are full-width text, for `merge_compatible_bands`."""

    def __init__(self, rows: List[List[PrintedChar]], bands: List[int]) -> None:
        self._rows = rows
        self._bands = bands

    def rows(self, bands: List[int]) -> int:
        """How many rows those bands hold."""
        return sum(1 for b in self._bands if b in bands)

    def crosses(self, bands: List[int], gutters: Tuple[float, ...]) -> bool:
        """Whether more than `FULL_WIDTH_SHARE` of those bands' rows, and at
        least `MIN_FULL_WIDTH_ROWS` of them, put ink just left of any of the
        gutters (where a real gutter is empty)."""
        band_rows = [row for row, b in zip(self._rows, self._bands) if b in bands]
        if not band_rows:
            return False
        far, near = GUTTER_PROBE_POINTS
        inked = sum(1 for row in band_rows
                    if any(c.rect.x0 < g - near and c.rect.x1 > g - far
                           for c in row for g in gutters))
        return inked >= MIN_FULL_WIDTH_ROWS and inked > FULL_WIDTH_SHARE * len(band_rows)


def subset_of(small: Tuple[float, ...], large: Tuple[float, ...], tolerance: float) -> bool:
    """Whether every gutter of `small` has a gutter of `large` within `tolerance`."""
    return all(any(abs(a - b) <= tolerance for b in large) for a in small)


def same_gutters(first: Tuple[float, ...], second: Tuple[float, ...], tolerance: float) -> bool:
    """Whether two rows have the same columns: as many gutters, each within `tolerance`."""
    return len(first) == len(second) and all(abs(a - b) <= tolerance for a, b in zip(first, second))


def split_into_segments(characters: Sequence[PrintedChar], layout: RowLayout) -> List[List[PrintedChar]]:
    """Runs of characters on one baseline with no gutter between them, in text order."""
    segments: List[List[PrintedChar]] = []
    current: List[PrintedChar] = []
    for char in characters:
        if current and not continues_segment(current, char, layout):
            segments.append(current)
            current = []
        current.append(char)
    if current:
        segments.append(current)
    return segments


def continues_segment(segment: List[PrintedChar], char: PrintedChar, layout: RowLayout) -> bool:
    """Whether `char` carries on the segment: same line, to the right, no gutter between."""
    if char.text in _SPACING_ACCENTS:
        return True                       # an accent always belongs to its letter
    height = segment_height(segment)
    last = segment[-1]
    same_line = abs(char.baseline - segment_baseline(segment)) < SAME_LINE_SHARE * height
    gap = char.rect.x0 - last.rect.x1
    return (same_line and -OVERLAP_HEIGHTS * height < gap < GUTTER_HEIGHTS * height
            and not crosses_gutter(last.rect.x1, char.rect.x0, layout.gutters_at(last.baseline)))


def crosses_gutter(left: float, right: float, gutters: Sequence[float]) -> bool:
    """Whether a gap from `left` to `right` reaches a column start (so it is a gutter)."""
    return any(left < gutter <= right + 0.5 for gutter in gutters)


def merge_neighbours(segments: List[List[PrintedChar]], layout: RowLayout) -> List[List[PrintedChar]]:
    """Join segments of one line that the text order split (an italic run written later)."""
    merged: List[List[PrintedChar]] = []
    for segment in sorted(segments, key=lambda s: (round(segment_baseline(s)), s[0].rect.x0)):
        if merged and same_printed_line(merged[-1], segment, layout):
            merged[-1] = merged[-1] + segment
        else:
            merged.append(segment)
    return merged


def same_printed_line(left: List[PrintedChar], right: List[PrintedChar], layout: RowLayout) -> bool:
    """Whether `right` continues `left` on the same line with no gutter between."""
    height = max(segment_height(left), segment_height(right))
    gap = right[0].rect.x0 - left[-1].rect.x1
    return (abs(segment_baseline(left) - segment_baseline(right)) < SAME_LINE_SHARE * height
            and -OVERLAP_HEIGHTS * height < gap < GUTTER_HEIGHTS * height
            and not crosses_gutter(left[-1].rect.x1, right[0].rect.x0,
                                   layout.gutters_at(segment_baseline(left))))


def build_line(segment: List[PrintedChar], page_number: int) -> LayoutLine:
    """A LayoutLine from one segment: text with spaces and accents restored."""
    chars = sorted(segment, key=lambda c: c.rect.x0) if is_jumbled(segment) else segment
    height = segment_height(chars)
    pieces: List[str] = []
    letter, after_accent = None, False
    for char in chars:
        if char.text in _SPACING_ACCENTS and pieces:
            pieces[-1] = unicodedata.normalize("NFC", pieces[-1] + _SPACING_ACCENTS[char.text])
            after_accent = True
            continue
        if letter is not None:
            # After a separately printed accent, PDFium often adds a space of
            # its own ("Ne´ methy"); only the gap from the LETTER counts there.
            gap = char.rect.x0 - letter.rect.x1
            spaced = gap > SPACE_SHARE * height or (char.space_before and not after_accent)
            if spaced and pieces[-1] != " ":
                pieces.append(" ")
        pieces.append(char.text)
        letter, after_accent = char, False
    text = re.sub(r"\s+", " ", "".join(pieces)).strip()
    return LayoutLine(page=page_number, band=0, column=0, text=text,
                      x0=min(c.rect.x0 for c in chars), x1=max(c.rect.x1 for c in chars),
                      baseline=segment_baseline(chars), height=height, chars=list(chars))


def drop_running_furniture(lines: List[LayoutLine], page_heights: Sequence[float]) -> List[LayoutLine]:
    """Remove running headers and footers: lines in the top or bottom band that recur.

    A line counts as furniture when it sits in the band AND, with its digits
    removed, the same text sits in the band on another page ("Collagen triple
    helical structure 1023") or it is nothing but a page number. The repeated
    text must hold some letters: the tail of a reference ("2010;330:341-6.")
    leaves only punctuation, and many tails share that shape.
    """
    def in_band(line: LayoutLine) -> bool:
        height = page_heights[line.page - 1]
        return line.baseline < FURNITURE_BAND * height or line.baseline > (1 - FURNITURE_BAND) * height

    def shape(line: LayoutLine) -> str:
        return re.sub(r"[\d\s]+", "", line.text).lower()

    pages_by_shape: Dict[str, set] = {}
    for line in lines:
        if in_band(line):
            pages_by_shape.setdefault(shape(line), set()).add(line.page)
    def is_furniture(line: LayoutLine) -> bool:
        text = shape(line)
        if not text:
            return True                                    # a bare page number
        letters = sum(1 for ch in text if ch.isalpha())
        return letters >= MIN_FURNITURE_LETTERS and len(pages_by_shape[text]) > 1

    return [line for line in lines if not (in_band(line) and is_furniture(line))]


def segment_baseline(segment: Sequence[PrintedChar]) -> float:
    """The baseline most of a segment's characters sit on (the median)."""
    return statistics.median(c.baseline for c in segment)


def segment_height(segment: Sequence[PrintedChar]) -> float:
    """The median loose character height of a segment, at least 1 point."""
    return max(1.0, statistics.median(c.rect.height for c in segment))


def is_jumbled(segment: Sequence[PrintedChar]) -> bool:
    """Whether the text order runs backwards anywhere (so the line must be re-sorted by x)."""
    return any(b.rect.x0 < a.rect.x0 - 0.5 * a.rect.height for a, b in zip(segment, segment[1:]))
