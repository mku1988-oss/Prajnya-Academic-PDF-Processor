"""
Table Region Detector — find the tables a recognition model loses, so we can re-read them.

For biology researchers: when the content-drop check tells us a page lost content, something on
that page — almost always a table — was skipped by the OCR engine. This module finds WHERE the
table sits, so the engine can be handed a clean crop of it. GLM-OCR reads a cropped table at
96.7-100% numeric recall; it is only whole mixed pages that defeat it.

Why a vision model and not PyMuPDF geometry:

    PyMuPDF's `find_tables()` reports ZERO tables on both of our known table pages. Most
    journals set tables in "booktabs" style — horizontal rules only, no vertical rules — and
    the default strategy needs vertical lines. Forcing `vertical_strategy="text"` does find
    them, but it merges two stacked tables plus the prose between them into one box, and clips
    the right edge mid-column.

    A grounding-capable layout VLM emits labelled layout boxes (`table`, `table_caption`,
    `table_footnote`) and found all four tables across our test pages, with the right edge
    landing on the true text margin. The current such model is DeepSeek-OCR-2 (via its
    `<|grounding|>` mode); it costs no model swap, being co-resident with GLM on an 8 GB card.

    We use the layout VLM ONLY as a detector. Its table CONTENT fabricates on dense tables —
    garbled values, invented rowspans — and must never be trusted. The boxes are all we take.

THIS MODULE IS VLM-AGNOSTIC. It carries no knowledge of which layout model produced the
grounding output. Everything model-specific — the box surface syntax (regex), the coordinate
normalization scale, and the label vocabulary — arrives in a `GroundingSpec`, which the caller
builds from the layout model's profile config (e.g. deepseek-ocr-v2.yaml -> `grounding_format`).
Swapping in a different layout detector is therefore a config change, not a code change. Only
the geometry below (ink masking, snap-to-whitespace, orphan-ink and satellite attachment) is
intrinsic to the module, because it operates on pixels and knows nothing about any model.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class GroundingSpec:
    """
    Everything model-specific about reading a layout VLM's grounding output.

    Sourced from the layout model's profile config (the `grounding_format` section of, e.g.,
    deepseek-ocr-v2.yaml) so that a future detector swap never touches this module.

    Attributes:
        box_pattern: regex with named groups `label` and `coords` that matches one labelled
            box. It must accept whatever surface form the model emits. DeepSeek emits two,
            depending on how it is served:
                llama-mtmd-cli:  <|ref|>table<|/ref|><|det|>[[85, 101, 915, 786]]<|/det|>
                llama-server:    table[[85, 101, 915, 786]]
            so its pattern accepts an optional <|ref|> prefix and a line-start bare label. A
            pattern written for one form silently matches nothing against the other — which
            looks exactly like a total detection failure — so both must be covered.
        coordinate_scale: the grid the model normalizes boxes to (DeepSeek: 1000), independent
            of render DPI.
        table_label: the label marking a table body.
        caption_label: the label marking a table caption.
        footnote_label: the label marking a table footnote.
        image_label: the label marking a figure/image body. Optional: a detector without figure
            grounding leaves it empty, which disables figure extraction (falsy -> off).
        image_caption_label: the label marking a figure caption. Optional, as above.
        prompt: the grounding prompt to send the model (kept here so the caller need not
            hardcode it either); unused by this module, which only PARSES output.
    """
    box_pattern: str
    coordinate_scale: float
    table_label: str
    caption_label: str
    footnote_label: str
    image_label: str = ""
    image_caption_label: str = ""
    prompt: str = ""
    _compiled: "re.Pattern" = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self._compiled = re.compile(self.box_pattern, re.MULTILINE)

    @classmethod
    def from_config(cls, grounding_format: dict) -> "GroundingSpec":
        """Build a spec from a profile's `grounding_format` config dict."""
        return cls(
            box_pattern=grounding_format["box_pattern"],
            coordinate_scale=float(grounding_format["coordinate_scale"]),
            table_label=grounding_format["table_label"],
            caption_label=grounding_format["caption_label"],
            footnote_label=grounding_format["footnote_label"],
            image_label=grounding_format.get("image_label", ""),
            image_caption_label=grounding_format.get("image_caption_label", ""),
            prompt=grounding_format.get("prompt", ""),
        )

    @property
    def satellite_labels(self) -> Tuple[str, str]:
        """Caption and footnote labels, the boxes folded into a table's crop."""
        return (self.caption_label, self.footnote_label)

    @property
    def supports_figures(self) -> bool:
        """True when this detector declares a figure label, so figure extraction can run."""
        return bool(self.image_label)


INK_THRESHOLD = 160        # 0-255 grey level below which a pixel counts as ink
WHITESPACE_MARGIN_PX = 6   # this many blank pixel-lines means we have cleared the glyph
MAX_EDGE_GROWTH_PX = 90    # never grow one edge further than this
ORPHAN_ATTACH_GAP_PX = 140 # unclaimed ink this close beneath a table belongs to it
MIN_ORPHAN_INK_PIXELS = 300  # below this, it is scan speckle
PAGE_MARGIN_FRACTION = 0.08  # top/bottom bands holding running heads and footers


@dataclass
class TableRegion:
    """
    One table on the page, as a pixel box ready to crop.

    Attributes:
        box: (x0, y0, x1, y1) in pixels of the rendered page image
        has_caption: whether a caption was folded into the box
        has_footnote: whether a footnote was folded into the box
        grew_for_orphan_ink: pixels the box grew downward to rescue unclaimed ink
    """
    box: Tuple[int, int, int, int]
    has_caption: bool = False
    has_footnote: bool = False
    grew_for_orphan_ink: int = 0


@dataclass
class Detection:
    """A single labelled box from the layout VLM, still in its normalized coordinates."""
    label: str
    box_normalized: Tuple[int, int, int, int]


def parse_grounding_output(raw_text: str, spec: GroundingSpec) -> List[Detection]:
    """
    Pull every labelled layout box out of the layout VLM's raw grounding response.

    Args:
        raw_text: the model's response text
        spec: the model's grounding format (regex + coordinate scale + labels), from config

    Returns:
        Every detection found, in the order emitted. Labels are whatever the model emits,
        e.g. table, table_caption, table_footnote, text, image, image_caption, title.
    """
    detections = []
    for match in spec._compiled.finditer(raw_text):
        numbers = [int(value) for value in match.group("coords").split(",")]
        if len(numbers) != 4:
            continue
        detections.append(Detection(label=match.group("label"), box_normalized=tuple(numbers)))
    return detections


def normalized_box_to_pixels(
    box_normalized: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    coordinate_scale: float,
) -> List[int]:
    """
    Map a normalized grounding box onto pixel coordinates of the rendered page.

    Because the model saw the whole page, the mapping is a straight proportional scale and does
    not depend on the DPI we rendered at.

    Args:
        box_normalized: (x0, y0, x1, y1) on the model's normalized grid
        image_size: (width, height) of the rendered page in pixels
        coordinate_scale: the model's grid size (e.g. 1000 for DeepSeek), from the spec

    Returns:
        [x0, y0, x1, y1] in pixels.
    """
    width, height = image_size
    x0, y0, x1, y1 = box_normalized
    return [
        int(x0 / coordinate_scale * width),
        int(y0 / coordinate_scale * height),
        int(x1 / coordinate_scale * width),
        int(y1 / coordinate_scale * height),
    ]


def build_ink_mask(page_image: Image.Image) -> np.ndarray:
    """Boolean array, True wherever the page has ink. Used for every crop-safety step."""
    greyscale = np.asarray(page_image.convert("L"))
    return greyscale < INK_THRESHOLD


def snap_box_to_whitespace(box: List[int], ink: np.ndarray) -> List[int]:
    """
    Grow a box outward until each edge sits in clear whitespace.

    DeepSeek's boxes are TIGHT. On one page the raw box sliced the right-hand column of values
    mid-digit — "52.0" came back as "52.(" — which would have truncated every number in that
    column invisibly. Glyphs are contiguous runs of ink, so an edge cutting a character still
    has ink beside it; we step outward until a clear margin appears.

    Args:
        box: [x0, y0, x1, y1] in pixels
        ink: boolean ink mask of the whole page

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


def find_unclaimed_ink(ink: np.ndarray, claimed_boxes: List[List[int]]) -> List[List[int]]:
    """
    Bounding boxes of ink that no detected region covers.

    This is how we catch content BOTH models forgot. On one scanned page DeepSeek emitted no
    `table_footnote` at all, and the OCR engine had dropped the table along with its footnotes —
    so footnotes 21/22/23 ("Light sensitive", "Good only below 29.6 deg") existed in neither
    pass. Footnote 22 says which salts degrade in light; that is chemistry, not decoration.

    Args:
        ink: boolean ink mask of the page
        claimed_boxes: every region DeepSeek labelled, in pixels

    Returns:
        Bounding boxes of leftover ink blobs, speckle excluded.
    """
    import cv2  # local import: OpenCV is only needed on the escalation path

    covered = np.zeros_like(ink, dtype=bool)
    for x0, y0, x1, y1 in claimed_boxes:
        covered[y0:y1, x0:x1] = True

    leftover = (ink & ~covered).astype(np.uint8) * 255

    # Dilate so the characters of one footnote line merge into a single blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    merged = cv2.dilate(leftover, kernel, iterations=2)
    label_count, _, stats, _ = cv2.connectedComponentsWithStats(merged, 8)

    blobs = []
    for index in range(1, label_count):  # 0 is background
        x, y, w, h, _ = stats[index]
        ink_pixels = int(leftover[y:y + h, x:x + w].sum() // 255)
        if ink_pixels >= MIN_ORPHAN_INK_PIXELS:
            blobs.append([x, y, x + w, y + h])
    return blobs


def attach_orphan_ink(
    table_box: List[int], orphan_blobs: List[List[int]], page_height: int
) -> Tuple[List[int], int]:
    """
    Grow a table box to swallow unclaimed ink sitting directly beneath it.

    Qualifying ink must overlap the table horizontally and begin within
    ORPHAN_ATTACH_GAP_PX of its bottom edge. Page furniture — running heads at the top,
    download stamps at the bottom — is excluded by position, since it is not table content.

    Args:
        table_box: the snapped table box, in pixels
        orphan_blobs: unclaimed ink boxes from find_unclaimed_ink()
        page_height: page height in pixels, for the margin bands

    Returns:
        (grown box, pixels grown downward)
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


def _boxes_overlap_horizontally(first: List[int], second: List[int]) -> bool:
    """Do two pixel boxes share any horizontal extent?"""
    return min(first[2], second[2]) - max(first[0], second[0]) > 0


def _attach_satellites(
    table_box: List[int], satellites: List[List[int]], page_height: int
) -> Tuple[List[int], bool, bool]:
    """
    Fold a table's own caption and footnote boxes into its crop box.

    DeepSeek reports them separately. Cropping the bare table would discard the footnote that
    defines the table's symbols. A satellite belongs to this table when it overlaps horizontally
    and sits within the attach gap of the table's top or bottom edge — which stops a stacked
    second table's caption from being swallowed.

    Args:
        table_box: the table's own box, in pixels
        satellites: caption/footnote boxes in pixels, tagged (box, is_footnote)
        page_height: unused today, kept for symmetry with attach_orphan_ink

    Returns:
        (merged box, whether a caption was attached, whether a footnote was attached)
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


def locate_table_regions(
    grounding_output: str,
    page_image: Image.Image,
    spec: GroundingSpec,
    logger: Optional[logging.Logger] = None,
) -> List[TableRegion]:
    """
    Turn a layout VLM's raw grounding response into crop-safe table regions.

    Four steps, each closing a way content gets lost:

      1. Parse the labelled boxes (surface form and labels come from `spec`).
      2. Fold each table's own caption and footnote into its box.
      3. Snap every edge out to whitespace, so no glyph is sliced.
      4. Attach unclaimed ink beneath the table — the footnotes neither model boxed.

    Args:
        grounding_output: raw text from the layout VLM's grounding call
        page_image: the rendered page, same image the model saw
        spec: the model's grounding format (regex + coordinate scale + labels), from config
        logger: optional logger

    Returns:
        One TableRegion per table found, ready to crop. Empty when the page has no table.

    Example:
        >>> regions = locate_table_regions(raw, page_image, spec)
        >>> for region in regions:
        ...     crop = page_image.crop(region.box)
    """
    log = logger or logging.getLogger("TableRegionDetector")

    detections = parse_grounding_output(grounding_output, spec)
    if not detections:
        log.warning("Layout VLM returned no layout boxes - cannot locate tables")
        return []

    scale = spec.coordinate_scale
    all_boxes = [normalized_box_to_pixels(d.box_normalized, page_image.size, scale) for d in detections]
    table_boxes = [
        normalized_box_to_pixels(d.box_normalized, page_image.size, scale)
        for d in detections if d.label == spec.table_label
    ]
    if not table_boxes:
        log.info("Layout VLM found no table on this page")
        return []

    satellites = [
        (normalized_box_to_pixels(d.box_normalized, page_image.size, scale),
         d.label == spec.footnote_label)
        for d in detections if d.label in spec.satellite_labels
    ]

    ink = build_ink_mask(page_image)
    orphan_blobs = find_unclaimed_ink(ink, all_boxes)
    page_height = ink.shape[0]

    regions = []
    for table_box in table_boxes:
        with_satellites, has_caption, has_footnote = _attach_satellites(
            table_box, satellites, page_height
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
        log.debug(
            f"table region {final_box} caption={has_caption} "
            f"footnote={has_footnote} orphan_growth={grew}px"
        )

    return regions


@dataclass
class FigureRegion:
    """
    One figure on the page: the image box to crop, plus its caption box to read (if any).

    Unlike a TableRegion (whose caption is folded INTO the crop so the whole thing is re-read),
    a figure keeps its caption box SEPARATE: the image is saved as a picture for the multimodal
    RAG, and the caption text is read from the digital text layer and tagged alongside it.

    Attributes:
        image_box: (x0, y0, x1, y1) in pixels of the rendered page image — the figure crop.
        caption_box: (x0, y0, x1, y1) in pixels of the caption, or None if the model boxed none.
    """
    image_box: Tuple[int, int, int, int]
    caption_box: Optional[Tuple[int, int, int, int]] = None


def locate_figure_regions(
    grounding_output: str,
    page_image: Image.Image,
    spec: GroundingSpec,
    logger: Optional[logging.Logger] = None,
) -> List[FigureRegion]:
    """
    Turn a layout VLM's grounding response into figure regions (image crop + caption box).

    Model-agnostic: the figure/caption label vocabulary comes from `spec`. A detector that
    declares no `image_label` (spec.supports_figures is False) yields an empty list, so figure
    extraction disables itself without any code branch on model identity.

    Steps:
      1. Parse the labelled boxes.
      2. Collect every `image` box (the figure bodies) and every `image_caption` box.
      3. Snap each image box out to whitespace so the crop keeps the whole figure.
      4. Pair each figure with its nearest caption (horizontal overlap, within the attach gap
         above or below — captions usually sit just below the figure).

    Args:
        grounding_output: raw text from the layout VLM's grounding call.
        page_image: the rendered page, same image the model saw.
        spec: the model's grounding format (regex + scale + labels), from config.
        logger: optional logger.

    Returns:
        One FigureRegion per figure found, in page (top-to-bottom) order. Empty when the page
        has no figure or the detector lacks figure support.
    """
    log = logger or logging.getLogger("FigureRegionDetector")

    if not spec.supports_figures:
        return []

    detections = parse_grounding_output(grounding_output, spec)
    if not detections:
        return []

    scale = spec.coordinate_scale
    image_boxes = [
        normalized_box_to_pixels(d.box_normalized, page_image.size, scale)
        for d in detections if d.label == spec.image_label
    ]
    if not image_boxes:
        return []

    caption_boxes = [
        normalized_box_to_pixels(d.box_normalized, page_image.size, scale)
        for d in detections if d.label == spec.image_caption_label
    ]

    ink = build_ink_mask(page_image)

    regions = []
    for image_box in sorted(image_boxes, key=lambda b: b[1]):  # top-to-bottom
        snapped = snap_box_to_whitespace(image_box, ink)
        caption_box = _pair_caption_to_figure(snapped, caption_boxes)
        regions.append(FigureRegion(
            image_box=tuple(snapped),
            caption_box=tuple(caption_box) if caption_box is not None else None,
        ))
        log.debug(f"figure region {snapped} caption={caption_box}")

    return regions


def _pair_caption_to_figure(
    image_box: List[int], caption_boxes: List[List[int]]
) -> Optional[List[int]]:
    """
    Pick the caption box that belongs to a figure: horizontally overlapping and closest to the
    figure's bottom (preferred) or top edge, within ORPHAN_ATTACH_GAP_PX. Returns None if none
    qualifies.
    """
    best = None
    best_gap = ORPHAN_ATTACH_GAP_PX + 1
    for caption_box in caption_boxes:
        if not _boxes_overlap_horizontally(caption_box, image_box):
            continue
        gap_below = caption_box[1] - image_box[3]   # caption starts below figure bottom
        gap_above = image_box[1] - caption_box[3]   # caption ends above figure top
        gap = gap_below if 0 <= gap_below <= ORPHAN_ATTACH_GAP_PX else (
            gap_above if 0 <= gap_above <= ORPHAN_ATTACH_GAP_PX else None
        )
        if gap is not None and gap < best_gap:
            best_gap = gap
            best = caption_box
    return best
