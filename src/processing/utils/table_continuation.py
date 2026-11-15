#!/usr/bin/env python3
"""
Table continuation — give a header-less continuation page its column headers.

For biology researchers: a long table printed across two pages often prints its
column names (genes, conditions, samples) only on the FIRST page. On the next
page the reader sees a grid of "+" and "-" signs with no idea which column is
which gene, and on the Springer book's Table 13.1 it dropped one of eleven
columns, so every later value sat under the wrong gene. The fix is what a human
reader does: look back at the header. The header band is cut from the previous
page's table image and pasted above the continuation before it is read; the
pasted row is then removed from the result.

Measured on the corpus (ledger T22, `eval/header_carry/GATE.md`): t28 went from
0 to 3 of 3 named checks (cell F1 0.957 -> 1.000), t31 from 2 to 4 of 5, and
continuations that already print their own header were unharmed.

**When it fires.** Every one of these must hold, otherwise the table is read
exactly as before:

1. The previous table was read on the IMMEDIATELY preceding page, and it was
   that page's LAST table; the candidate is THIS page's FIRST table.
2. The previous table has a header band: two full-width ruled lines at its top.
3. The two tables' vertical column rules line up: the same number of them, at
   the same spacing, give or take a constant shift (the two pages' crops start
   at different x). This is the test that two crops are one table, and it is
   done on pixels, with no model.
4. After the read, the model's FIRST row is the pasted header. If it is not,
   the headed read is thrown away and the table is read plain.

VLM-AGNOSTIC: this module names no model and holds no prompt; the caller passes
the same crop reader every other table read uses.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

#: Reads one table image (the extraction model with its table prompt).
CropReader = Callable[["Image.Image"], str]

#: A pixel darker than this is ink or a ruled line. Printed table rules are thin
#: and anti-aliased to mid grey (~149 at 300 DPI on the Springer book), so the
#: usual "black" threshold of ~128 misses them entirely.
DARK = 190

#: A FULL-width horizontal rule (or full-height vertical one): at least this
#: share of the row (column) is dark. A rule drawn under only some columns
#: (t27's at 55% of the width) must not count as the header's closing rule.
RULE_FRACTION = 0.80

#: Column rules of the two tables may sit this many pixels off their common
#: shift and still be "the same column". Measured spread on the corpus pairs:
#: 0 to 2 px at 300 DPI.
COLUMN_TOLERANCE_PX = 4

#: A table needs at least this many vertical rules for the column test to mean
#: anything (two rules = one interior column boundary plus an edge, or two
#: boundaries). Fewer: no evidence, so no carry.
MIN_COLUMN_RULES = 2

#: Share of the header's non-empty cells that the read's first row must contain
#: to count as the pasted header (the corpus pairs matched fully).
HEADER_MATCH = 0.60


@dataclass
class CarriedHeader:
    """
    The header band of the last table read, offered to the next page.

    Attributes:
        strip: the header band image (top rule to the rule under the header).
        columns: x positions of the band's vertical rules, in band pixels.
        cells: the header row's cell texts, normalised, from that table's grid.
        page_number: the 1-indexed page it was cut from.
    """

    strip: "Image.Image"
    columns: List[float]
    cells: List[str]
    page_number: int


@dataclass
class ContinuationTracker:
    """
    Remembers one header across pages, and applies it where it belongs.

    One tracker lives for one document's body pass. The caller announces each
    page (`begin_page`), asks for a headed read of that page's first table
    (`headed_read`), and records every table it read (`remember`), in order.

    Attributes:
        logger: where decisions are logged.
        carried: the header on offer, or None.
        page_number: the page currently being processed.
        tables_seen_on_page: how many tables of this page were read so far.
        headed_tables: (page_number, index) of every table read with a carried
            header, so a run can be audited.
    """

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("TableContinuation"))
    carried: Optional[CarriedHeader] = None
    page_number: int = 0
    tables_seen_on_page: int = 0
    headed_tables: List[tuple] = field(default_factory=list)

    def begin_page(self, page_number: int) -> None:
        """Start a page; a header from anywhere but the page before is dropped."""
        if self.carried is not None and self.carried.page_number != page_number - 1:
            self.carried = None
        self.page_number = page_number
        self.tables_seen_on_page = 0

    def headed_read(self, table_image: "Image.Image", read: CropReader,
                    finish: Callable[["Image.Image", str], str]) -> Optional[str]:
        """
        The table read with the carried header pasted above, or None.

        Args:
            table_image: the upright table image production reads.
            read: reads one image.
            finish: production's post-read step (the section check), applied to
                the headed image and its read BEFORE the pasted row is dropped,
                exactly as the gate measured it.

        Returns:
            The grid without the pasted row, or None when the carry does not
            apply or the model did not return the header as its first row; the
            caller then reads the table plain.
        """
        if self.carried is None or self.tables_seen_on_page != 0:
            return None
        shift = column_shift(self.carried.columns, column_rules(table_image))
        if shift is None:
            self.logger.debug("Page %d: first table's columns do not line up with "
                              "page %d's header; read plain", self.page_number,
                              self.carried.page_number)
            return None
        headed_image = paste_header(self.carried.strip, table_image, shift)
        grid = finish(headed_image, (read(headed_image) or "").strip())
        trimmed = drop_pasted_header(grid, self.carried.cells)
        if trimmed is None:
            self.logger.info("Page %d: the headed read did not return the pasted "
                             "header first; read plain", self.page_number)
            return None
        self.headed_tables.append((self.page_number, 1))
        self.logger.info("Page %d: continuation table read with page %d's header "
                         "(columns shifted %d px)", self.page_number,
                         self.carried.page_number, shift)
        return trimmed

    def remember(self, table_image: "Image.Image", grid: Optional[str],
                 was_headed: bool) -> None:
        """
        Record a table just read; the page's LAST one is what the next page sees.

        A table that was itself read with a carried header keeps passing the
        ORIGINAL header on: its own top band is a data row, not a header.
        """
        self.tables_seen_on_page += 1
        if was_headed and self.carried is not None:
            self.carried = CarriedHeader(self.carried.strip, self.carried.columns,
                                         self.carried.cells, self.page_number)
            return
        self.carried = carried_header_for(table_image, grid, self.page_number)


# --- helpers (after the main class, per CLAUDE.md) --------------------------

def carried_header_for(table_image: "Image.Image", grid: Optional[str],
                       page_number: int) -> Optional[CarriedHeader]:
    """A table's header band, its column rules and header cells, or None."""
    strip = header_band(table_image)
    if strip is None or not grid:
        return None
    columns = column_rules(strip)
    cells = first_row_cells(grid)
    if len(columns) < MIN_COLUMN_RULES or not any(cells):
        return None
    return CarriedHeader(strip=strip, columns=columns, cells=cells,
                         page_number=page_number)


def header_band(image: "Image.Image") -> Optional["Image.Image"]:
    """
    The band from a table's top rule down to the rule under its header.

    The FIRST full-width rule is the table's top edge (a caption may sit above
    it inside the crop, so the band starts at the rule, not at the image top);
    the SECOND closes the header. None for a table with fewer than two such
    rules: an unruled table gives no reliable cut, and no cut is guessed.
    """
    rules = _rule_runs(_dark(image).mean(axis=1) > RULE_FRACTION)
    if len(rules) < 2:
        return None
    top, closing = rules[0], rules[1]
    return image.crop((0, top[0], image.width, closing[1] + 1))


def column_rules(image: "Image.Image") -> List[float]:
    """x centres of the full-height vertical rules (column boundaries)."""
    runs = _rule_runs(_dark(image).mean(axis=0) > RULE_FRACTION)
    return [(start + end) / 2 for start, end in runs]


def column_shift(header_columns: List[float], table_columns: List[float]) -> Optional[int]:
    """
    How far right to move the header so its columns sit over the table's.

    None unless both have the same number of rules (at least MIN_COLUMN_RULES)
    and every pair is the same distance apart, within COLUMN_TOLERANCE_PX.
    """
    if (len(header_columns) < MIN_COLUMN_RULES
            or len(header_columns) != len(table_columns)):
        return None
    offsets = [t - h for h, t in zip(header_columns, table_columns)]
    middle = float(np.median(offsets))
    if any(abs(offset - middle) > COLUMN_TOLERANCE_PX for offset in offsets):
        return None
    return int(round(middle))


def paste_header(strip: "Image.Image", table_image: "Image.Image",
                 shift: int) -> "Image.Image":
    """The header band stacked on the table, moved `shift` px to line up."""
    canvas = Image.new("RGB", (table_image.width, strip.height + table_image.height),
                       "white")
    canvas.paste(strip.convert("RGB"), (shift, 0))
    canvas.paste(table_image.convert("RGB"), (0, strip.height))
    return canvas


def drop_pasted_header(grid: str, header_cells: List[str]) -> Optional[str]:
    """The grid without its first row, if that row is the pasted header; else None."""
    rows = list(re.finditer(r"<tr\b[^>]*>.*?</tr>", grid or "", re.S))
    wanted = {cell for cell in header_cells if cell}
    if not rows or not wanted:
        return None
    first = set(_row_cells(rows[0].group(0)))
    if len(wanted & first) / len(wanted) < HEADER_MATCH:
        return None
    return grid[:rows[0].start()] + grid[rows[0].end():]


def first_row_cells(grid: str) -> List[str]:
    """Normalised cell texts of a grid's first row ([] when it has none)."""
    match = re.search(r"<tr\b[^>]*>.*?</tr>", grid or "", re.S)
    return _row_cells(match.group(0)) if match else []


def _row_cells(row_html: str) -> List[str]:
    """Normalised cell texts of one <tr> element."""
    cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, re.S)
    return [_normalise(cell) for cell in cells]


def _normalise(cell_html: str) -> str:
    """Tags out, case folded, every run of whitespace removed."""
    text = re.sub(r"<[^>]+>", "", cell_html)
    return re.sub(r"\s+", "", text).lower()


def _dark(image: "Image.Image") -> np.ndarray:
    """True where a pixel is ink or a rule."""
    return np.asarray(image.convert("L")) < DARK


def _rule_runs(flags) -> List[tuple]:
    """(start, end) of each run of consecutive True values, end inclusive."""
    runs, start = [], None
    for position, flag in enumerate(flags):
        if flag and start is None:
            start = position
        if not flag and start is not None:
            runs.append((start, position - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs
