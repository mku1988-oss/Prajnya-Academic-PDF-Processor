#!/usr/bin/env python3
"""
Crop-safe regions — turning detector boxes into boxes it is safe to cut out.

For biology researchers: a layout detector tells you roughly where a table or
figure sits. Cropping exactly on that box loses content, in three ways this
module fixes:

1. **Tight edges slice glyphs.** A box drawn on the table's ink can cut a digit
   in half. On one page a raw crop turned "52.0" into "52.(" — every number in
   that column would have been silently truncated. Fixed by growing each edge
   outward until it sits in clear whitespace.
2. **A table's caption and footnote are separate boxes.** Cropping the bare
   table discards the footnote that defines its symbols. Fixed by folding a
   table's own satellites into its crop.
3. **Some content no detector boxes at all.** On one scanned page the footnotes
   ("Light sensitive", "Good only below 29.6 deg") were in neither the layout
   pass nor the OCR read. That is chemistry, not decoration. Fixed by finding
   ink that no region claims and attaching it to the table above it.

Figures are handled differently from tables on purpose: a table's caption is
folded *into* the crop so the whole thing gets re-read as one unit, while a
figure keeps its caption box *separate* — the picture is saved for multimodal
RAG and the caption is read as text from the PDF's own text layer.

## What changed from V5

V5's equivalent took the layout VLM's raw grounding string plus a
`GroundingSpec` and parsed it here. V6 takes `List[Detection]` — the parse
belongs to the detector. This module is now pure geometry: it operates on
pixels and knows nothing about any model, which is what lets the CPU and VLM
detectors feed it interchangeably.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from src.processing.layout.base import (
    LABEL_IMAGE,
    LABEL_IMAGE_CAPTION,
    LABEL_TABLE,
    LABEL_TABLE_CAPTION,
    LABEL_TABLE_FOOTNOTE,
    Detection,
    detections_with_label,
)

# --- crop-safety constants, all measured on real pages ----------------------
INK_THRESHOLD = 160          # 0-255 grey level below which a pixel counts as ink
WHITESPACE_MARGIN_PX = 6     # this many blank pixel-lines means the glyph is cleared
MAX_EDGE_GROWTH_PX = 90      # never grow one edge further than this
ORPHAN_ATTACH_GAP_PX = 140   # unclaimed ink this close beneath a table belongs to it
MIN_ORPHAN_INK_PIXELS = 300  # below this, it is scan speckle
PAGE_MARGIN_FRACTION = 0.08  # top/bottom bands holding running heads and footers


@dataclass
class TableRegion:
    """
    One table on the page, as a pixel box ready to crop.

    Attributes:
        box: (x0, y0, x1, y1) in pixels of the rendered page image.
        has_caption: whether a caption was folded into the box.
        has_footnote: whether a footnote was folded into the box.
        grew_for_orphan_ink: pixels the box grew downward to rescue unclaimed ink.
    """

    box: Tuple[int, int, int, int]
    has_caption: bool = False
    has_footnote: bool = False
    grew_for_orphan_ink: int = 0


@dataclass
class FigureRegion:
    """
    One figure: the image box to crop, plus its caption box to read separately.

    Unlike `TableRegion`, whose caption is folded into the crop, a figure keeps
    its caption box apart — the image is saved as a picture and the caption is
    read as text.

    Attributes:
        image_box: (x0, y0, x1, y1) in pixels — the figure crop.
        caption_box: the caption's box, or None when the detector boxed none.
    """

    image_box: Tuple[int, int, int, int]
    caption_box: Optional[Tuple[int, int, int, int]] = None


# ---------------------------------------------------------------------------
# The two entry points
# ---------------------------------------------------------------------------

def locate_table_regions(
    detections: Sequence[Detection],
    page_image: "Image.Image",
    logger: Optional[logging.Logger] = None,
) -> List[TableRegion]:
    """
    Turn detector boxes into crop-safe table regions.

    Four steps, each closing a way content gets lost:
      1. Take every table box, and every caption/footnote box.
      2. Fold each table's own caption and footnote into its box.
      3. Snap every edge out to whitespace, so no glyph is sliced.
      4. Attach unclaimed ink beneath the table — the footnotes nothing boxed.

    Args:
        detections: this page's detections, from any `LayoutDetector`.
        page_image: the rendered page the detections were made against.
        logger: optional logger.

    Returns:
        One `TableRegion` per table, ready to crop. Empty when the page has
        no table — never None.
    """
    log = logger or logging.getLogger("TableRegions")

    table_boxes = [list(d.box_px) for d in detections_with_label(detections, LABEL_TABLE)]
    if not table_boxes:
        log.info("no table detected on this page")
        return []

    satellites = [
        (list(d.box_px), d.label == LABEL_TABLE_FOOTNOTE)
        for d in detections
        if d.label in (LABEL_TABLE_CAPTION, LABEL_TABLE_FOOTNOTE)
    ]

    ink = build_ink_mask(page_image)
    page_height = ink.shape[0]
    orphan_blobs = find_unclaimed_ink(ink, [list(d.box_px) for d in detections])

    regions: List[TableRegion] = []
    for table_box in table_boxes:
        with_satellites, has_caption, has_footnote = _attach_satellites(
            table_box, satellites
        )
        snapped = snap_box_to_whitespace(with_satellites, ink)
        with_orphans, grew = attach_orphan_ink(snapped, orphan_blobs, page_height)
        final_box = snap_box_to_whitespace(with_orphans, ink)

        regions.append(TableRegion(
            box=tuple(final_box),
            has_caption=has_caption,
            has_footnote=has_footnote or grew > 0,
            grew_for_orphan_ink=grew,
        ))
        log.debug("table region %s caption=%s footnote=%s orphan_growth=%dpx",
                  final_box, has_caption, has_footnote, grew)

    return regions


def locate_figure_regions(
    detections: Sequence[Detection],
    page_image: "Image.Image",
    logger: Optional[logging.Logger] = None,
) -> List[FigureRegion]:
    """
    Turn detector boxes into figure regions (image crop + separate caption box).

    Args:
        detections: this page's detections, from any `LayoutDetector`.
        page_image: the rendered page the detections were made against.
        logger: optional logger.

    Returns:
        One `FigureRegion` per figure, in top-to-bottom page order. Empty when
        the page has no figure.
    """
    log = logger or logging.getLogger("FigureRegions")

    image_boxes = [list(d.box_px) for d in detections_with_label(detections, LABEL_IMAGE)]
    if not image_boxes:
        return []

    caption_boxes = [
        list(d.box_px) for d in detections_with_label(detections, LABEL_IMAGE_CAPTION)
    ]
    ink = build_ink_mask(page_image)

    regions: List[FigureRegion] = []
    for image_box in sorted(image_boxes, key=lambda b: b[1]):  # top-to-bottom
        snapped = snap_box_to_whitespace(image_box, ink)
        caption_box = _pair_caption_to_figure(snapped, caption_boxes)
        regions.append(FigureRegion(
            image_box=tuple(snapped),
            caption_box=tuple(caption_box) if caption_box is not None else None,
        ))
        log.debug("figure region %s caption=%s", snapped, caption_box)

    return regions


# ---------------------------------------------------------------------------
# Geometry helpers (after the main functions, per CLAUDE.md)
# ---------------------------------------------------------------------------

def build_ink_mask(page_image: "Image.Image") -> np.ndarray:
    """Boolean array, True wherever the page has ink. Basis of every crop-safety step."""
    greyscale = np.asarray(page_image.convert("L"))
    return greyscale < INK_THRESHOLD


def snap_box_to_whitespace(box: Sequence[int], ink: np.ndarray) -> List[int]:
    """
    Grow a box outward until each edge sits in clear whitespace.

    Detector boxes are tight, and an edge cutting through a character still has
    ink beside it — glyphs are contiguous runs of ink. So each edge steps
    outward until a clear margin of `WHITESPACE_MARGIN_PX` blank lines appears,
    capped at `MAX_EDGE_GROWTH_PX` so a dense page cannot grow a box forever.

    Args:
        box: [x0, y0, x1, y1] in pixels.
        ink: boolean ink mask of the whole page.

    Returns:
        The widened box, clamped to the page.
    """
    height, width = ink.shape
    x0, y0, x1, y1 = box

    def line_has_ink(axis: str, index: int) -> bool:
        if axis == "column":
            return 0 <= index < width and bool(ink[y0:y1, index].any())
        return 0 <= index < height and bool(ink[index, x0:x1].any())

    def grow_edge(start: int, step: int, axis: str, limit: int) -> int:
        position = start
        blank_run = 0
        for _ in range(MAX_EDGE_GROWTH_PX):
            candidate = position + step
            if not (0 <= candidate < limit):
                break
            if line_has_ink(axis, candidate):
                blank_run = 0
            else:
                blank_run += 1
                if blank_run >= WHITESPACE_MARGIN_PX:
                    break
            position = candidate
        return position

    x0 = grow_edge(x0, -1, "column", width)
    x1 = grow_edge(x1, +1, "column", width)
    y0 = grow_edge(y0, -1, "row", height)
    y1 = grow_edge(y1, +1, "row", height)
    return [max(0, x0), max(0, y0), min(width - 1, x1), min(height - 1, y1)]


def find_unclaimed_ink(ink: np.ndarray,
                       claimed_boxes: Sequence[Sequence[int]]) -> List[List[int]]:
    """
    Bounding boxes of ink that no detected region covers.

    This catches content the detector missed entirely — most often table
    footnotes, which carry real scientific meaning. Ink is dilated first so the
    separate characters of one footnote line merge into a single blob, then
    blobs smaller than `MIN_ORPHAN_INK_PIXELS` are discarded as scan speckle.

    Args:
        ink: boolean ink mask of the page.
        claimed_boxes: every detected region, in pixels.

    Returns:
        Bounding boxes of leftover ink blobs, as **plain Python ints**.

        That coercion is load-bearing, not tidiness. OpenCV's `stats` rows are
        a numpy array, so unpacking them yields `numpy.intc` — which travels
        happily through box arithmetic, lands in a table's page metadata, and
        then fails at the very last step of the document with
        `PydanticSerializationError: Unable to serialize unknown type`. Found
        the expensive way: a 25-page paper finished every GPU call and lost all
        of it while writing the JSON.
    """
    import cv2  # local import: only the escalation path needs OpenCV

    covered = np.zeros_like(ink, dtype=bool)
    for x0, y0, x1, y1 in claimed_boxes:
        covered[y0:y1, x0:x1] = True

    leftover = (ink & ~covered).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    merged = cv2.dilate(leftover, kernel, iterations=2)
    label_count, _, stats, _ = cv2.connectedComponentsWithStats(merged, 8)

    blobs: List[List[int]] = []
    for index in range(1, label_count):  # 0 is the background component
        x, y, w, h, _ = (int(value) for value in stats[index][:5])
        ink_pixels = int(leftover[y:y + h, x:x + w].sum() // 255)
        if ink_pixels >= MIN_ORPHAN_INK_PIXELS:
            blobs.append([x, y, x + w, y + h])
    return blobs


def attach_orphan_ink(table_box: Sequence[int], orphan_blobs: Sequence[Sequence[int]],
                      page_height: int) -> Tuple[List[int], int]:
    """
    Grow a table box to swallow unclaimed ink sitting directly beneath it.

    Qualifying ink must overlap the table horizontally and begin within
    `ORPHAN_ATTACH_GAP_PX` of its bottom edge. Page furniture — running heads
    and download stamps — is excluded by position, since it is not table
    content.

    Returns:
        (grown box, pixels grown downward).
    """
    x0, y0, x1, y1 = table_box
    merged = [x0, y0, x1, y1]

    for bx0, by0, bx1, by1 in orphan_blobs:
        in_page_margin = (
            by0 < page_height * PAGE_MARGIN_FRACTION
            or by1 > page_height * (1 - PAGE_MARGIN_FRACTION - 0.02)
        )
        if in_page_margin:
            continue

        overlaps_horizontally = min(bx1, x1) - max(bx0, x0) > 0
        sits_just_below = 0 <= (by0 - y1) <= ORPHAN_ATTACH_GAP_PX
        if overlaps_horizontally and sits_just_below:
            merged = [min(merged[0], bx0), min(merged[1], by0),
                      max(merged[2], bx1), max(merged[3], by1)]

    return merged, merged[3] - y1


def _boxes_overlap_horizontally(first: Sequence[int], second: Sequence[int]) -> bool:
    """Do two pixel boxes share any horizontal extent?"""
    return min(first[2], second[2]) - max(first[0], second[0]) > 0


def _attach_satellites(
    table_box: Sequence[int],
    satellites: Sequence[Tuple[Sequence[int], bool]],
) -> Tuple[List[int], bool, bool]:
    """
    Fold a table's own caption and footnote boxes into its crop box.

    A satellite belongs to this table when it overlaps horizontally AND sits
    within the attach gap of the table's top or bottom edge — which is what
    stops a stacked second table's caption from being swallowed.

    Args:
        table_box: the table's own box, in pixels.
        satellites: (box, is_footnote) pairs.

    Returns:
        (merged box, whether a caption was attached, whether a footnote was).
    """
    merged = list(table_box)
    attached_caption = False
    attached_footnote = False

    for satellite_box, is_footnote in satellites:
        if not _boxes_overlap_horizontally(satellite_box, table_box):
            continue

        sits_above = 0 <= (table_box[1] - satellite_box[3]) <= ORPHAN_ATTACH_GAP_PX
        sits_below = 0 <= (satellite_box[1] - table_box[3]) <= ORPHAN_ATTACH_GAP_PX
        if not (sits_above or sits_below):
            continue

        merged = [
            min(merged[0], satellite_box[0]), min(merged[1], satellite_box[1]),
            max(merged[2], satellite_box[2]), max(merged[3], satellite_box[3]),
        ]
        if is_footnote:
            attached_footnote = True
        else:
            attached_caption = True

    return merged, attached_caption, attached_footnote


def _pair_caption_to_figure(image_box: Sequence[int],
                            caption_boxes: Sequence[Sequence[int]]) -> Optional[List[int]]:
    """
    Pick the caption belonging to a figure.

    It must overlap horizontally and sit closest to the figure's bottom
    (preferred) or top edge, within `ORPHAN_ATTACH_GAP_PX`. Returns None when
    nothing qualifies.
    """
    best: Optional[Sequence[int]] = None
    best_gap = ORPHAN_ATTACH_GAP_PX + 1
    for caption_box in caption_boxes:
        if not _boxes_overlap_horizontally(caption_box, image_box):
            continue
        gap_below = caption_box[1] - image_box[3]   # caption starts below figure
        gap_above = image_box[1] - caption_box[3]   # caption ends above figure
        gap = gap_below if 0 <= gap_below <= ORPHAN_ATTACH_GAP_PX else (
            gap_above if 0 <= gap_above <= ORPHAN_ATTACH_GAP_PX else None
        )
        if gap is not None and gap < best_gap:
            best_gap = gap
            best = caption_box
    return list(best) if best is not None else None
