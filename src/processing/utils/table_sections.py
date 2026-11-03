"""
Table sections -- one detected table box that holds more than one table.

For biology researchers: an older paper may print two sub-tables under one
heading ("At 20 °C" and "At 25 °C") so close together that the layout detector
draws ONE box round both. The reading model (GLM-OCR) reads the first sub-table,
closes it, and stops: everything below is silently lost. Measured on the corpus,
"Table II" of the humidity paper lost 64 of its 82 values this way.

This module finds the places where such a crop could be cut (wide blank bands
running across its whole width), groups the pieces into blocks big enough to be
a table, and merges what a separate reading of each block adds. It never calls
a model itself: the caller passes the reader in.

See eval/TABLE_SECTIONS_GATE.md for the measured rule.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from statistics import median
from typing import Callable, List, Optional, Tuple

import numpy as np

#: A band at least this many times the crop's median blank band is a section gap.
GAP_FACTOR = 3.0
#: ...and at least this many pixels tall (at the 300 DPI extraction render).
MIN_GAP_PX = 20
#: A block needs this many text lines to stand alone; fewer joins a neighbour.
MIN_BLOCK_LINES = 4
#: A pixel darker than this counts as ink.
INK_THRESHOLD = 128
#: A pixel row with at most this much ink still counts as blank (scan speckle).
BLANK_ROW_MAX_INK = 2
#: A block's rows are added only when fewer than this share of its cells are
#: already in the whole-crop grid.
MAX_SHARED_CELLS = 0.5

CropReader = Callable[["Image.Image"], str]


def read_table_with_sections(crop_image, whole_grid: str, read_crop: CropReader,
                             logger: Optional[logging.Logger] = None) -> str:
    """
    The whole-crop grid, plus the rows of any later section it missed.

    Args:
        crop_image: the table crop that `whole_grid` was read from.
        whole_grid: the reading of the whole crop.
        read_crop: reads one cropped image, returning HTML.
        logger: optional logger.

    Returns:
        `whole_grid` unchanged when the crop holds one section, or when every
        later section only repeats it; otherwise one table with the new rows
        appended after the whole-crop rows.
    """
    log = logger or logging.getLogger("TableSections")
    blocks = find_section_blocks(crop_image)
    if len(blocks) < 2:
        return whole_grid

    added_rows: List[str] = []
    known_cells = set(_cell_texts(whole_grid))
    for top, bottom in blocks[1:]:
        section = read_crop(crop_image.crop((0, top, crop_image.width, bottom))) or ""
        new_rows = _rows_not_already_read(section, known_cells)
        if new_rows:
            added_rows.extend(new_rows)
            known_cells.update(_cell_texts("".join(new_rows)))
            log.info("Table section %d-%dpx added %d row(s) the whole read missed",
                     top, bottom, len(new_rows))
    if not added_rows:
        return whole_grid
    return _append_rows(whole_grid, added_rows)


def find_section_blocks(crop_image) -> List[Tuple[int, int]]:
    """
    Vertical (top, bottom) pixel spans of the crop's table-sized blocks.

    One span covering the whole crop means there is nothing to split.
    """
    ink = np.asarray(crop_image.convert("L")) < INK_THRESHOLD
    blank = ink.sum(axis=1) <= BLANK_ROW_MAX_INK
    bands = _blank_bands(blank)
    if not bands:
        return [(0, ink.shape[0])]
    typical = median(end - start for start, end in bands)
    cut_at = [(start + end) // 2 for start, end in bands
              if end - start >= max(GAP_FACTOR * typical, MIN_GAP_PX)]
    edges = [0] + cut_at + [ink.shape[0]]
    pieces = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return _merge_small_pieces(pieces, blank)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _blank_bands(blank: np.ndarray) -> List[Tuple[int, int]]:
    """Runs of blank pixel rows strictly inside the crop (not its margins)."""
    bands, start = [], None
    for y, is_blank in enumerate(blank):
        if is_blank and start is None:
            start = y
        elif not is_blank and start is not None:
            if start > 0:
                bands.append((start, y))
            start = None
    return bands


def _line_count(blank: np.ndarray, top: int, bottom: int) -> int:
    """Text lines in rows [top, bottom): runs of non-blank pixel rows."""
    segment = ~blank[top:bottom]
    return int(np.count_nonzero(segment[1:] & ~segment[:-1]) + (1 if segment.size and segment[0] else 0))


def _merge_small_pieces(pieces: List[Tuple[int, int]], blank: np.ndarray) -> List[Tuple[int, int]]:
    """Join every piece with too few lines to its neighbour, until none is left."""
    pieces = list(pieces)
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        for i, (top, bottom) in enumerate(pieces):
            if _line_count(blank, top, bottom) >= MIN_BLOCK_LINES:
                continue
            j = i + 1 if i + 1 < len(pieces) else i - 1
            low, high = sorted((i, j))
            pieces[low:high + 1] = [(pieces[low][0], pieces[high][1])]
            changed = True
            break
    return pieces


def _cell_texts(html: str) -> List[str]:
    """Non-empty cell texts in a comparable form, for the overlap check.

    The two readings of one table can write the same cell differently -- the
    whole read gave "BaI₂·2H₂O", the section read "$BaI_{2}·2H_{2}O" -- and a
    raw comparison took the second for new rows, so the first version of this
    duplicated t12's whole 20 °C block (found by driving the GUI, 2026-11-04).
    """
    found = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", html, flags=re.S)
    return [text for text in (_comparable(cell) for cell in found) if text]


def _comparable(cell: str) -> str:
    """Tags, LaTeX markers, Unicode sub/superscripts, dashes and spaces folded away."""
    text = re.sub(r"<[^>]+>", "", cell)
    text = re.sub(r"\\[a-zA-Z]+", "", text)                 # \mathrm, \text, ...
    text = unicodedata.normalize("NFKC", text)              # ₂ -> 2, ² -> 2
    text = re.sub(r"[$\\{}_^]", "", text)
    text = re.sub(r"[−–—‐-]", "-", text)
    return re.sub(r"\s+", "", text).lower()


def _rows_not_already_read(section_html: str, known_cells: set) -> List[str]:
    """The section's rows, or [] when most of its cells are already known."""
    cells = _cell_texts(section_html)
    if not cells:
        return []
    shared = sum(1 for cell in cells if cell in known_cells) / len(cells)
    if shared >= MAX_SHARED_CELLS:
        return []
    return re.findall(r"<tr[^>]*>.*?</tr>", section_html, flags=re.S)


def _append_rows(whole_grid: str, rows: List[str]) -> str:
    """Insert rows before the whole grid's closing tags (inside its table)."""
    closing = re.search(r"(</tbody>\s*)?</table>\s*$", whole_grid)
    if closing is None:
        return whole_grid + "<table>" + "".join(rows) + "</table>"
    return whole_grid[:closing.start()] + "".join(rows) + whole_grid[closing.start():]
