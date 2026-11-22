#!/usr/bin/env python3
"""
Figure crop fit — make a figure crop match the space its printed caption labels.

For biology researchers: the layout detector draws a box around each figure,
and two things go wrong with that box often enough to matter:

1. **The caption ends up inside the picture.** Frontiers draws a frame round a
   figure AND its legend, and the box follows the frame, so the crop carries
   the first lines of its own caption (PMC5880930 Figures 1 and 3).
2. **Panels are missing.** The detector finds one panel and misses the others
   (PMC5880930 Figure 7: panel C only, the western blot in panel A is lost;
   Figure 6: panel B's photographs are lost), or stops short of the labels
   printed beside a photograph (Figure 3's red strain names).

The page itself says where the figure is: a printed caption ("FIGURE 7 | ...")
labels the space directly above it. So, for each caption that exactly one crop
claims (ledger T25, `eval/crop_fit/GATE.md`):

* **Trim**: the crop's bottom may not pass the caption's first line.
* **Fit**: the crop grows to hold all the ink in the caption's BAND, the same
  band the orphan-figure rescue uses (T24, `orphan_figures.caption_band`):
  above the caption, across the caption's width, up to the nearest crop,
  table, prose line, caption or detected text block above it. Frame lines (long
  thin rules, nearly empty boxes) do not count as ink. The crop only ever grows
  here; it is never made smaller than the detector's box.

Untouched: a caption with no crop (T24's job), a caption shared by several
panel crops, a caption above or beside its figure, a scanned page (no text
layer, so no caption line to go by).

Model-free and VLM-agnostic: page geometry, the text layer and pixel ink only.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.processing.layout.base import LABEL_IMAGE, LABEL_TABLE, Detection
from src.processing.layout.regions import build_ink_mask
from src.processing.utils.figure_label_pass import (
    LabelLine, build_text_lines, find_label_lines, label_words)
from src.processing.utils.orphan_figures import (
    caption_band, caption_extent, overlap_share, overlaps_horizontally)

Box = Tuple[float, float, float, float]

#: White border `rendering.render_page` puts round every page image, in pixels.
#: Repeated here (not imported) so this utility does not depend on a strategy.
RENDER_PADDING_PX = 25

#: Clear pixels left between a trimmed crop's bottom and its caption's top.
CAPTION_GAP_PX = 4

#: A detected text block or heading counts as the band's ceiling unless at
#: least this share of it already lies inside the crop (text inside a figure).
INSIDE_CROP_SHARE = 0.5

#: A grown crop may overlap another crop or a table by at most this share of
#: that other box; more means the band reached into a neighbour.
MAX_NEIGHBOUR_OVERLAP = 0.10

#: Text lines starting in this top share of the page are running heads: the
#: band never reaches above them.
TOP_MARGIN_SHARE = 0.08

#: An ink component this small (pixels) is anti-aliasing or scan speckle.
MIN_COMPONENT_PIXELS = 20

#: A FRAME: an ink component spanning at least this share of the band's width
#: or height while inking under `FRAME_MAX_FILL` of its own bounding box. Frontiers'
#: figure frame is a thin rectangle round figure and legend; inside the band it
#: is a U of three lines, huge and empty.
FRAME_MIN_SPAN = 0.8
FRAME_MAX_FILL = 0.05

#: A RULE: ink spanning `FRAME_MIN_SPAN` of the band with at most this many
#: pixels of ink per pixel of length (a line about that thick). A frame's top
#: edge often renders as its own one-pixel line, apart from its sides; the
#: bottom edge of the frame above enters the band as a line with two short
#: stubs (PMC5880930 p8: 1018 px long, 1.0 px of ink per column).
RULE_MAX_THICKNESS = 4

#: Margin of white kept round the ink, as the whitespace snap does.
INK_MARGIN_PX = 6

#: Band ink reaching no further than this beyond the crop is the crop's own
#: ink: the whitespace snap leaves its edge a pixel or two from where this
#: module's margin would put it, and that difference is not a missing panel.
NEGLIGIBLE_GROWTH_PX = 10

#: Ink joins the crop only if it lies within this many POINTS of the crop as
#: grown so far, so a figure grows panel by panel and label by label but never
#: jumps a wide white gap. Measured (eval/crop_fit/GATE.md): missing panels
#: sit 5-36 pt from the crop. Page furniture sits 38-49 pt away, too close to
#: separate by distance alone, so it is kept out of the band instead
#: (`FURNITURE_TOP_SHARE`, `TEXT_BLOCK_MARGIN_PT`).
MAX_PANEL_GAP_PT = 40.0

#: Ink lying ENTIRELY in this top share of the page is where journal logos,
#: header rules and running heads live (every one measured ends at or above
#: 8.2% of the page height: PLOS Biology, Cell Press). Such ink joins a figure
#: only from `TOP_ZONE_GAP_PT` away: a figure printed high on the page has its
#: panel letters there, a few points from their panels (PMC8440575 Fig. 4
#: "a", "b", "d"; Collagen structure Fig. 1 "A", "B"), while the furniture
#: sits 38-49 pt away.
FURNITURE_TOP_SHARE = 0.09
TOP_ZONE_GAP_PT = 10.0

#: The band stays within the page's text block, widened by this many points:
#: edge tabs (Nature Reviews) and rotated "Downloaded from" stamps sit in the
#: side margins, beside the text, never inside it.
TEXT_BLOCK_MARGIN_PT = 10.0

#: A text line at least this share of the page width is running text; the
#: text block is the span of those lines.
TEXT_BLOCK_LINE_SHARE = 0.25


def fit_crops_to_captions(
    page,
    page_image,
    assets: Sequence,
    detections: Sequence[Detection],
    logger: Optional[logging.Logger] = None,
) -> List[Tuple[object, Tuple[int, int, int, int]]]:
    """
    Trim and grow each figure crop to the band its printed caption labels.

    Nothing is changed here: the caller sets each new box and saves the image
    again, so a failure part way leaves every crop as the detector gave it.
    Every box is decided against the detector's boxes, not against one fitted
    a moment earlier.

    Args:
        page: the PDF page (text layer; size in points).
        page_image: the padded render every box is in (pixels).
        assets: this page's figure crops, after the label pass (`label_key`).
        detections: this page's layout detections (pixels).
        logger: optional logger.

    Returns:
        (asset, new box) for every crop whose box should change; [] when none.
    """
    log = logger or logging.getLogger("FigureCropFit")
    if not assets:
        return []
    mapper = PixelMapper(page, page_image)
    text_lines = build_text_lines(label_words(page), page.size[0])
    label_lines = [line for line in find_label_lines(text_lines) if line.starts_a_block]
    ink = build_ink_mask(page_image)

    changed = []
    for asset in assets:
        line = own_caption_line(asset, assets, label_lines, mapper)
        if line is None:
            continue
        new_box = fitted_box(asset, line, assets, detections, text_lines,
                             label_lines, ink, mapper)
        if new_box is None or new_box == tuple(asset.bbox):
            continue
        changed.append((asset, new_box))
        log.info("Page %d: %s crop fitted to its caption's band %s -> %s",
                 asset.page, asset.printed_label, list(asset.bbox), list(new_box))
    return changed


# --- helpers (after the main function, per CLAUDE.md) -----------------------

class PixelMapper:
    """Points <-> pixels for a page rendered with `RENDER_PADDING_PX` of border."""

    def __init__(self, page, page_image):
        self.scale = (page_image.width - 2 * RENDER_PADDING_PX) / page.size[0]
        self.width = page_image.width
        self.height = page_image.height
        self.page_width, self.page_height = page.size

    def to_points(self, box) -> Box:
        return tuple((float(v) - RENDER_PADDING_PX) / self.scale for v in box)

    def to_pixels(self, box: Box) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = (int(round(v * self.scale + RENDER_PADDING_PX)) for v in box)
        return (max(0, x0), max(0, y0), min(self.width - 1, x1), min(self.height - 1, y1))


def own_caption_line(asset, assets: Sequence, label_lines: Sequence[LabelLine],
                     mapper: PixelMapper) -> Optional[LabelLine]:
    """
    The caption line printed under this crop, when the crop is its figure's only one.

    Returns None when the crop has no label, shares it with other crops
    (panels detected separately), or no line with its label sits below it and
    across it.
    """
    key = getattr(asset, "label_key", None)
    if not key or sum(1 for other in assets if other.label_key == key) != 1:
        return None
    crop = mapper.to_points(asset.bbox)
    middle = (crop[1] + crop[3]) / 2
    below = [line for line in label_lines
             if line.key == key and line.rect[1] > middle
             and overlaps_horizontally(line.rect, (crop[0], crop[2]))]
    return min(below, key=lambda line: line.rect[1]) if below else None


def fitted_box(asset, line: LabelLine, assets: Sequence, detections: Sequence[Detection],
               text_lines, label_lines: Sequence[LabelLine], ink: np.ndarray,
               mapper: PixelMapper) -> Optional[Tuple[int, int, int, int]]:
    """The crop trimmed above its caption, then grown over the ink of its band."""
    caption_top = mapper.to_pixels(line.rect)[1]
    trimmed = trim_above_caption(tuple(asset.bbox), caption_top)
    neighbours = neighbour_boxes(asset, assets, detections)
    band = band_in_pixels(line, text_lines, label_lines, neighbours, detections,
                          trimmed, mapper)
    if band is None:
        return trimmed
    pieces = ink_pieces_in_band(ink, band, caption_width(line, text_lines, mapper))
    limits = [int((TOP_ZONE_GAP_PT if above_the_furniture_line(piece, mapper)
                   else MAX_PANEL_GAP_PT) * mapper.scale) for piece in pieces]
    ink_box = chained_ink_box(trimmed, pieces, limits, band)
    if ink_box is None or contains(trimmed, ink_box, NEGLIGIBLE_GROWTH_PX):
        return trimmed
    grown = union_box(trimmed, ink_box)
    grown = (grown[0], grown[1], grown[2], min(grown[3], caption_top - CAPTION_GAP_PX))
    if any(overlap_share(other, grown) > MAX_NEIGHBOUR_OVERLAP for other in neighbours):
        return trimmed
    return tuple(int(v) for v in grown)


def trim_above_caption(crop: Tuple[int, int, int, int], caption_top: int
                       ) -> Tuple[int, int, int, int]:
    """The crop with its bottom moved above the caption's first line, if it ran into it."""
    limit = caption_top - CAPTION_GAP_PX
    if crop[3] <= limit or crop[1] >= limit:
        return crop
    return (crop[0], crop[1], crop[2], limit)


def neighbour_boxes(asset, assets: Sequence, detections: Sequence[Detection]
                    ) -> List[Tuple[int, int, int, int]]:
    """Every other crop and every table on the page (pixels): the band must not take them."""
    boxes = [tuple(other.bbox) for other in assets if other is not asset]
    boxes += [tuple(d.box_px) for d in detections if d.label == LABEL_TABLE]
    return boxes


def band_in_pixels(line: LabelLine, text_lines, label_lines: Sequence[LabelLine],
                   neighbours, detections: Sequence[Detection], crop,
                   mapper: PixelMapper) -> Optional[Tuple[int, int, int, int]]:
    """
    The caption's band (T24's definition), lowered under anything else above it.

    Extra ceilings beyond T24's crops, tables and prose lines: other captions,
    the detector's text blocks, headings, running heads (unless mostly inside
    this crop: text printed in a figure), and text lines in the top margin.
    """
    occupied = [mapper.to_points(box) for box in neighbours]
    left, top, right, bottom = caption_band(line, text_lines, occupied, mapper.page_width)
    ceilings = [other.rect for other in label_lines if other is not line]
    ceilings += [mapper.to_points(d.box_px) for d in detections
                 if d.label != LABEL_IMAGE
                 and overlap_share(mapper.to_points(d.box_px), mapper.to_points(crop))
                 < INSIDE_CROP_SHARE]
    ceilings += [(x0, y0, x1, y1) for _, x0, y0, x1, y1 in text_lines
                 if y0 < TOP_MARGIN_SHARE * mapper.page_height]
    for box in ceilings:
        if box[3] <= bottom and overlaps_horizontally(box, (left, right)):
            top = max(top, box[3])
    left, right = within_the_text_block(left, right, text_lines, mapper)
    if bottom - top <= 0 or right - left <= 0:
        return None
    return mapper.to_pixels((left, top, right, bottom))


def within_the_text_block(left: float, right: float, text_lines,
                          mapper: PixelMapper) -> Tuple[float, float]:
    """
    The band's left and right kept within the page's text block (points).

    For biology researchers: logos, header rules, edge tabs and download
    stamps are printed OUTSIDE the block of running text. Measured on the dev
    set and the corpus, they sit 38-49 pt from a figure, about as far as a
    figure's own panels can be, so distance cannot tell them apart; where they
    sit can. Edge tabs and stamps sit in the side margins, so the band keeps
    to the text block's width (plus `TEXT_BLOCK_MARGIN_PT`); logos and rules
    sit at the page top (`above_the_furniture_line`). A page with no line of
    running text keeps its band as it is.
    """
    running = [(x0, x1) for _, x0, _, x1, _ in text_lines
               if x1 - x0 >= TEXT_BLOCK_LINE_SHARE * mapper.page_width]
    if running:
        left = max(left, min(x0 for x0, _ in running) - TEXT_BLOCK_MARGIN_PT)
        right = min(right, max(x1 for _, x1 in running) + TEXT_BLOCK_MARGIN_PT)
    return left, right


def above_the_furniture_line(piece, mapper: PixelMapper) -> bool:
    """Whether a piece of ink (pixels) lies wholly in the page's top furniture zone."""
    line = mapper.to_pixels((0.0, FURNITURE_TOP_SHARE * mapper.page_height, 0.0, 0.0))[1]
    return piece[3] < line


def caption_width(line: LabelLine, text_lines, mapper: PixelMapper) -> int:
    """Width of the caption block in pixels (frames are about this wide)."""
    left, right = caption_extent(line, text_lines)
    return int((right - left) * mapper.scale)


def ink_pieces_in_band(ink: np.ndarray, band: Tuple[int, int, int, int],
                       frame_width: int) -> List[Tuple[int, int, int, int]]:
    """
    The boxes (page pixels) of every piece of figure ink in the band, frames
    and speckle left out.

    Args:
        ink: the page's ink mask.
        band: the band, in pixels.
        frame_width: the caption's width in pixels; a frame round figure and
            caption is at least about this wide.

    Returns:
        One box per connected piece of ink; [] when the band holds none.
    """
    import cv2  # local import, as in layout/regions.py

    x0, y0, x1, y1 = band
    region = ink[y0:y1, x0:x1].astype(np.uint8)
    if not region.any():
        return []
    count, _, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    band_width, band_height = x1 - x0, y1 - y0
    span_width = min(band_width, frame_width)
    kept = []
    for index in range(1, count):          # 0 is the background
        left, top, width, height, pixels = (int(v) for v in stats[index][:5])
        if pixels < MIN_COMPONENT_PIXELS:
            continue
        if is_frame(width, height, pixels, span_width, band_height):
            continue
        kept.append((x0 + left, y0 + top, x0 + left + width, y0 + top + height))
    return kept


def chained_ink_box(crop, pieces, max_gaps, band
                    ) -> Optional[Tuple[int, int, int, int]]:
    """
    The box round the ink reachable from the crop, step by step.

    Starts from the pieces touching the crop and keeps adding any piece within
    its own limit (`max_gaps[i]` pixels, one per piece) of the box grown so
    far. Returns it with `INK_MARGIN_PX` of margin, clipped to the band; None
    when no ink is reachable.
    """
    grown, reached = tuple(crop), []
    remaining = list(zip(pieces, max_gaps))
    changed = True
    while changed:
        changed = False
        for entry in list(remaining):
            piece, max_gap = entry
            if gap_between(grown, piece) <= max_gap:
                reached.append(piece)
                remaining.remove(entry)
                grown = union_box(grown, piece)
                changed = True
    if not reached:
        return None
    box = union_box(*reached)
    return (max(band[0], box[0] - INK_MARGIN_PX), max(band[1], box[1] - INK_MARGIN_PX),
            min(band[2], box[2] + INK_MARGIN_PX), min(band[3], box[3] + INK_MARGIN_PX))


def gap_between(first, second) -> int:
    """White distance between two boxes (0 when they touch or overlap)."""
    horizontal = max(0, max(first[0], second[0]) - min(first[2], second[2]))
    vertical = max(0, max(first[1], second[1]) - min(first[3], second[3]))
    return max(horizontal, vertical)


def is_frame(width: int, height: int, pixels: int, span_width: int, band_height: int) -> bool:
    """A long, nearly empty ink outline or a long thin line: a frame or rule, not figure content."""
    wide = width >= FRAME_MIN_SPAN * span_width
    tall = height >= FRAME_MIN_SPAN * band_height
    if (wide and pixels <= RULE_MAX_THICKNESS * width) or             (tall and pixels <= RULE_MAX_THICKNESS * height):
        return True
    return (wide or tall) and pixels < FRAME_MAX_FILL * width * height


def contains(outer, inner, tolerance: int) -> bool:
    """Whether `inner` lies inside `outer` widened by `tolerance` on every side."""
    return (inner[0] >= outer[0] - tolerance and inner[1] >= outer[1] - tolerance
            and inner[2] <= outer[2] + tolerance and inner[3] <= outer[3] + tolerance)


def union_box(*boxes) -> Tuple[int, int, int, int]:
    """The smallest box holding all of them."""
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
