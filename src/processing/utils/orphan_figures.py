#!/usr/bin/env python3
"""
Orphan figures — find a figure the layout detector missed, from its caption.

For biology researchers: the layout detector gives every box a confidence
score and keeps those above 0.5. Some real figures score lower (three 3D
renders of a triple helix at 0.39-0.48; a row of confocal time-lapse panels at
0.22) and are never cut out. Lowering the cut-off for every page was measured
and rejected: it adds duplicate boxes elsewhere. But the page itself says when
something is missing: a printed caption ("Figure 11 Interstrand hydrogen
bonding ...") that no cropped figure on the page claims. Only then, and only in
the space that caption labels, are the detector's weaker boxes trusted.

**The rule** (ledger T24, `eval/orphan_captions/GATE.md`):

1. ORPHAN: a text-layer caption line (it starts a block; tables are not figure
   labels) whose key none of this page's figure crops was given.
2. ITS BAND: the area above the caption, across the caption block's width (the
   whole page when the block is wider than half the page), up to the nearest
   crop, table, other caption, or full line of prose above it.
3. Re-detect at `RESCUE_THRESHOLD`; keep image boxes whose centre is in the
   band and that overlap no existing crop or table by more than
   `MAX_OVERLAP_SHARE` of their area.
4. Their UNION is one new figure box. Nothing in the band: nothing is added
   (a caption printed under its figure's facing page is left alone).

VLM-AGNOSTIC and model-free: the detector passed in is the same ONNX layout
model, only with a lower cut-off.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from src.processing.layout.base import LABEL_IMAGE, LABEL_TABLE, Detection
from src.processing.utils.figure_label_pass import (
    LabelLine, build_text_lines, find_label_lines, harvest_caption_block, label_words)

Box = Tuple[float, float, float, float]

#: The cut-off used INSIDE an orphan caption's band only. JRuberti Figure 5's
#: panels score 0.220 (`scripts/diagnose_missing_figures.py`).
RESCUE_THRESHOLD = 0.2

#: A rescued box may overlap an existing crop or table by at most this share of
#: its own area; more means it is a duplicate of something already kept.
MAX_OVERLAP_SHARE = 0.10

#: A text line at least this share of the band's width is prose, and the band
#: stops below it. Panel letters, axis labels and time stamps inside a figure
#: are short and do not stop it.
PROSE_LINE_SHARE = 0.5

#: A caption block wider than this share of the page belongs to a full-width
#: figure, so the band spans the whole page.
FULL_WIDTH_SHARE = 0.5


def rescue_orphan_figures(
    page,
    page_image,
    figure_assets: Sequence,
    detections: Sequence[Detection],
    low_detector,
    logger: Optional[logging.Logger] = None,
) -> List[Detection]:
    """
    New image detections for figures whose captions no crop on this page claims.

    Args:
        page: the PDF page (text layer, size in points).
        page_image: the render all boxes are in, in pixels.
        figure_assets: this page's figure crops, already through the label pass
            (`label_key`, `bbox` in pixels).
        detections: this page's detections at the normal cut-off.
        low_detector: the same layout model at `RESCUE_THRESHOLD`; called only
            when there is an orphan caption.
        logger: optional logger.

    Returns:
        One `image` Detection per rescued figure (pixels), [] when none.
    """
    log = logger or logging.getLogger("OrphanFigures")
    width, height = page.size
    scale = width / page_image.width
    text_lines = build_text_lines(label_words(page), width)
    orphans = orphan_caption_lines(text_lines, figure_assets)
    if not orphans:
        return []
    occupied = occupied_boxes(figure_assets, detections, scale)
    candidates = [scaled(d.box_px, scale) for d in low_detector.detect(page_image)
                  if d.label == LABEL_IMAGE]
    rescued = []
    for orphan in orphans:
        band = caption_band(orphan, text_lines, occupied, width)
        kept = boxes_in_band(candidates, band, occupied)
        if not kept:
            log.info("Page %d: caption %r has no crop and nothing in its band",
                     page.number + 1, orphan.printed)
            continue
        box = union(kept)
        occupied.append(box)
        rescued.append(Detection(label=LABEL_IMAGE,
                                 box_px=tuple(int(round(v / scale)) for v in box)))
        log.info("Page %d: rescued %s from %d low-score box(es) in its caption's band",
                 page.number + 1, orphan.printed, len(kept))
    return rescued


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def orphan_caption_lines(text_lines, figure_assets: Sequence) -> List[LabelLine]:
    """Caption-starting label lines whose key no crop on the page carries."""
    claimed = {getattr(a, "label_key", None) for a in figure_assets}
    return [line for line in find_label_lines(text_lines)
            if line.starts_a_block and line.key not in claimed]


def occupied_boxes(figure_assets: Sequence, detections: Sequence[Detection],
                   scale: float) -> List[Box]:
    """Existing figure crops and tables, in points: a rescue must not duplicate them."""
    boxes = [scaled(a.bbox, scale) for a in figure_assets]
    boxes += [scaled(d.box_px, scale) for d in detections if d.label == LABEL_TABLE]
    return boxes


def caption_band(orphan: LabelLine, text_lines, occupied: List[Box],
                 page_width: float) -> Box:
    """The space above the caption that its figure must occupy (points)."""
    left, right = caption_extent(orphan, text_lines)
    if right - left > FULL_WIDTH_SHARE * page_width:
        left, right = 0.0, page_width
    bottom = orphan.rect[1]
    top = 0.0
    for box in occupied:
        if box[3] <= bottom and overlaps_horizontally(box, (left, right)):
            top = max(top, box[3])
    for text, x0, y0, x1, y1 in text_lines:
        line_box = (x0, y0, x1, y1)
        if (y1 <= bottom and (x1 - x0) >= PROSE_LINE_SHARE * (right - left)
                and overlaps_horizontally(line_box, (left, right))):
            top = max(top, y1)
    return (left, top, right, bottom)


def caption_extent(orphan: LabelLine, text_lines) -> Tuple[float, float]:
    """Left and right edges of the whole caption block (its first line may be short)."""
    block = [orphan.rect]
    for text, x0, y0, x1, y1 in text_lines:
        if y0 >= orphan.rect[1] and abs(x0 - orphan.rect[0]) <= 12.0:
            block.append((x0, y0, x1, y1))
    caption = harvest_caption_block(orphan, text_lines)
    lefts = [b[0] for b in block]
    rights = [b[2] for b in block if _line_text_in(b, text_lines, caption)]
    return min(lefts), max(rights or [orphan.rect[2]])


def boxes_in_band(candidates: List[Box], band: Box, occupied: List[Box]) -> List[Box]:
    """Candidate boxes centred in the band and not duplicating anything kept."""
    kept = []
    for box in candidates:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not (band[0] <= cx <= band[2] and band[1] <= cy <= band[3]):
            continue
        if any(overlap_share(box, other) > MAX_OVERLAP_SHARE for other in occupied):
            continue
        kept.append(box)
    return kept


def union(boxes: List[Box]) -> Box:
    """The smallest box holding all of them."""
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def overlap_share(box: Box, other: Box) -> float:
    """Share of `box`'s area inside `other`."""
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 0:
        return 0.0
    w = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
    h = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
    return w * h / area


def overlaps_horizontally(box: Box, span: Tuple[float, float]) -> bool:
    """Whether a box shares any x-range with the span."""
    return box[0] < span[1] and box[2] > span[0]


def scaled(box, scale: float) -> Box:
    """A pixel box in points."""
    return tuple(float(v) * scale for v in box)


def _line_text_in(line_box: Box, text_lines, caption: str) -> bool:
    """Whether the text line at `line_box` is part of the harvested caption."""
    for text, x0, y0, x1, y1 in text_lines:
        if (x0, y0, x1, y1) == tuple(line_box):
            return text.strip()[:20] in caption
    return False
