#!/usr/bin/env python3
"""
Side-by-side figure split: one crop that holds two printed figures becomes two.

For biology researchers: journals sometimes print two figures next to each
other, each with its own short tag underneath ("Fig. 3"   "Fig. 4"), and the
layout detector draws ONE box round both. The second figure then has no crop
at all: a reader asking for "Figure 4" finds nothing, and the crop labelled
"Fig. 3" also shows Figure 4. Seen on "Intercalary regeneration in
amphibians" p5 (ledger T27, `eval/side_by_side/GATE.md`).

The page itself says where to cut. When two or more DIFFERENT figure labels sit
on one row directly beneath a crop, each inside its width, the crop holds that
many figures, and the cut lies somewhere between neighbouring tags. Where
exactly is read from the pixels, in this order:

1. **A white gutter**: a run of columns that are nearly all white, top to
   bottom. Cut in its middle.
2. **A seam**: a thin column brighter than the columns a few points to either
   side, along (nearly) the whole height. Two photographs butted together
   show one, as on the Intercalary page.

No gutter and no seam: the split is DECLINED and the crop is left exactly as
detected. A wrong split cuts a real figure in two, which is worse than the
defect this fixes, so declining is the default.

Model-free and VLM-agnostic: the text layer and pixel brightness only.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.processing.layout.base import LABEL_IMAGE, Detection
from src.processing.utils.figure_crop_fit import PixelMapper
from src.processing.utils.figure_label_pass import (
    CAPTION_OVERLAP_TOLERANCE, MAX_GAP_FRACTION, LabelLine, build_text_lines,
    find_label_lines, label_words)

PixelBox = Tuple[int, int, int, int]

#: Two tags are on the same row when their tops differ by less than this share
#: of a tag's own height. Tags on one row share a baseline in print.
SAME_ROW_SHARE = 0.5

#: A gray level at or above this counts as white paper (0-255 scale).
WHITE_LEVEL = 245

#: Share of a column's pixels that must be white for it to be a gutter column.
#: Not 100%: a hairline or a stray speck may cross a real gutter.
GUTTER_WHITE_SHARE = 0.98

#: The narrowest white run, in points, that counts as a gutter.
MIN_GUTTER_PT = 1.0

#: How far either side of a candidate seam its neighbours are read, in points.
#: The Intercalary seam is about 2 pt wide, so 3 pt reaches past it.
SEAM_NEIGHBOUR_PT = 3.0

#: How much brighter (gray levels) the seam must be than BOTH neighbours, in a row.
SEAM_CONTRAST = 5

#: Share of rows in which the seam must be brighter than both neighbours. A
#: bright object inside a photograph does not run the full height; a seam does.
SEAM_ROW_SHARE = 0.9

#: How far the search for a cut may reach past a tag's edge, in points, but
#: never past the tag's centre. A tag set flush with its figure's edge puts the
#: seam right at that edge.
EDGE_REACH_PT = 2.0


def split_side_by_side_figures(
    page,
    page_image,
    detections: Sequence[Detection],
    logger: Optional[logging.Logger] = None,
) -> List[Detection]:
    """
    Split every figure box that holds several printed figures side by side.

    Args:
        page: the PDF page, for its text layer and size.
        page_image: the padded render the detections were made against.
        detections: this page's layout detections.
        logger: optional logger.

    Returns:
        The detections with each such figure box replaced by one box per
        figure, left to right. Every other detection, and every box where the
        split was declined, is returned unchanged and in place.
    """
    log = logger or logging.getLogger("FigureSideSplit")
    detections = list(detections)
    if not any(d.label == LABEL_IMAGE for d in detections):
        return detections

    # Step 1: the figure tags on this page, in pixels.
    mapper = PixelMapper(page, page_image)
    tags = _label_lines_in_pixels(page, mapper)
    if len(tags) < 2:
        return detections

    # Step 2: one grayscale copy of the page for the gutter and seam tests.
    gray = np.asarray(page_image.convert("L"), dtype=np.int16)
    reach = int(round(page.size[1] * MAX_GAP_FRACTION * mapper.scale))
    inside = int(round(page.size[1] * CAPTION_OVERLAP_TOLERANCE * mapper.scale))

    # Step 3: each figure box, split where the evidence allows.
    result: List[Detection] = []
    for detection in detections:
        pieces = None
        if detection.label == LABEL_IMAGE:
            row = _tag_row_beneath(detection.box_px, tags, reach, inside)
            if row:
                pieces = _split_box(detection.box_px, row, gray, mapper.scale)
        if pieces:
            log.info("Page %d: one figure box %s holds %s side by side; split "
                     "into %d.", page.number + 1, list(detection.box_px),
                     " + ".join(line.printed for line, _ in row), len(pieces))
            result.extend(Detection(detection.label, box, detection.score)
                          for box in pieces)
        else:
            result.append(detection)
    return result


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def _label_lines_in_pixels(page, mapper: PixelMapper
                           ) -> List[Tuple[LabelLine, PixelBox]]:
    """Every caption-opening line on the page, with its box in pixels."""
    width, _ = page.size
    lines = find_label_lines(build_text_lines(label_words(page), width))
    return [(line, mapper.to_pixels(line.rect)) for line in lines]


def _tag_row_beneath(box: PixelBox,
                     tags: Sequence[Tuple[LabelLine, PixelBox]],
                     reach: int,
                     inside: int) -> List[Tuple[LabelLine, PixelBox]]:
    """
    The row of two or more different figure tags printed directly under a box.

    A tag belongs to the box when its top lies between slightly inside the
    box's bottom and `reach` below it, and its horizontal centre lies within
    the box. Of those, the topmost row is taken. It counts only when it holds
    at least two different labels that do not overlap each other sideways.

    Returns:
        The row's tags, left to right, or [] when there is no such row.
    """
    x0, _, x1, y1 = box
    beneath = [(line, rect) for line, rect in tags
               if -inside <= rect[1] - y1 <= reach
               and x0 <= (rect[0] + rect[2]) / 2 <= x1]
    if len(beneath) < 2:
        return []

    beneath.sort(key=lambda pair: pair[1][1])
    first_top = beneath[0][1][1]
    tolerance = (beneath[0][1][3] - first_top) * SAME_ROW_SHARE
    row = sorted((pair for pair in beneath if pair[1][1] - first_top <= tolerance),
                 key=lambda pair: pair[1][0])

    keys = [line.key for line, _ in row]
    if len(row) < 2 or len(set(keys)) != len(keys):
        return []
    if any(left[1][2] > right[1][0] for left, right in zip(row, row[1:])):
        return []
    return row


def _split_box(box: PixelBox,
               row: Sequence[Tuple[LabelLine, PixelBox]],
               gray: np.ndarray,
               scale: float) -> Optional[List[PixelBox]]:
    """
    Cut a box once between each pair of neighbouring tags.

    Returns:
        One box per tag, left to right, or None when any one cut finds
        neither a gutter nor a seam (the whole split is then declined).
    """
    x0, y0, x1, y1 = box
    region = gray[y0:y1, :]
    edges: List[Tuple[int, int]] = []
    for (_, left), (_, right) in zip(row, row[1:]):
        low, high = _search_span(left, right, scale)
        cut = _find_gutter(region, low, high, scale) or _find_seam(region, low, high, scale)
        if cut is None:
            return None
        edges.append(cut)

    pieces: List[PixelBox] = []
    start = x0
    for gap_start, gap_end in edges:
        pieces.append((start, y0, gap_start, y1))
        start = gap_end
    pieces.append((start, y0, x1, y1))
    if any(piece[2] - piece[0] < 2 for piece in pieces):
        return None
    return pieces


def _search_span(left: PixelBox, right: PixelBox, scale: float) -> Tuple[int, int]:
    """The columns where the cut between two tags may lie."""
    reach = int(round(EDGE_REACH_PT * scale))
    left_centre = (left[0] + left[2]) // 2
    right_centre = (right[0] + right[2]) // 2
    return max(left[2] - reach, left_centre + 1), min(right[0] + reach, right_centre - 1)


def _find_gutter(region: np.ndarray, low: int, high: int,
                 scale: float) -> Optional[Tuple[int, int]]:
    """
    The widest run of white columns between `low` and `high`.

    Returns:
        `(first, last + 1)` of the run, so the pieces leave the gutter out,
        or None when no run is at least `MIN_GUTTER_PT` wide.
    """
    if high <= low:
        return None
    white = (region[:, low:high] >= WHITE_LEVEL).mean(axis=0) >= GUTTER_WHITE_SHARE
    best: Optional[Tuple[int, int]] = None
    run_start = None
    for offset, is_white in enumerate(list(white) + [False]):
        if is_white and run_start is None:
            run_start = offset
        elif not is_white and run_start is not None:
            if best is None or offset - run_start > best[1] - best[0]:
                best = (run_start, offset)
            run_start = None
    if best is None or best[1] - best[0] < MIN_GUTTER_PT * scale:
        return None
    return low + best[0], low + best[1]


def _find_seam(region: np.ndarray, low: int, high: int,
               scale: float) -> Optional[Tuple[int, int]]:
    """
    A thin full-height column brighter than its neighbours on both sides.

    Returns:
        `(column, column + 1)` of the strongest seam, or None when no column
        is brighter than both neighbours in at least `SEAM_ROW_SHARE` of rows.
    """
    offset = max(1, int(round(SEAM_NEIGHBOUR_PT * scale)))
    first, last = max(low, offset), min(high, region.shape[1] - offset)
    if last <= first:
        return None
    centre = region[:, first:last]
    left = region[:, first - offset:last - offset]
    right = region[:, first + offset:last + offset]
    brighter = ((centre - left >= SEAM_CONTRAST)
                & (centre - right >= SEAM_CONTRAST)).mean(axis=0)
    best = int(np.argmax(brighter))
    if brighter[best] < SEAM_ROW_SHARE:
        return None
    column = first + best
    return column, column + 1
