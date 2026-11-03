#!/usr/bin/env python3
"""
The layout-detection interface — one seam for "where are the tables and figures?"

For biology researchers: before reading a page, the pipeline needs to know what
is ON the page — where the tables sit, where the figures are, which regions are
plain prose. That question can be answered two very different ways:

1. **A small CPU model** (ONNX, ~15 ms per page) — essentially free, so it can
   run on every page up front.
2. **A grounding-capable vision model** (a VLM asked to draw labelled boxes) —
   accurate but costs ~17 s of GPU time per page.

This interface makes the two interchangeable. Every consumer receives the same
`List[Detection]` regardless of which detector produced it, so choosing between
them is a config change (`profile_registry.layout_detector`), not a code change.

V5 got this wrong: consumers took the VLM's *raw text output* plus a regex spec,
which forced any alternative detector to fake a specific model's text syntax.
Here the parse belongs to the detector; boxes come out already structured.

## Coordinate convention

`Detection.box_px` is **pixels of the rendered page image the detector was
given**, top-left origin, y increasing downward — the same orientation as
`PdfBackend` rectangles (which are in PDF points; divide by the render scale to
convert). Consumers crop `page_image` with these boxes directly.

## Label vocabulary

Labels are the canonical strings in `CANONICAL_LABELS`. Each implementation is
responsible for mapping its model's native class names onto these — consumers
never see model-specific vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

try:  # Pillow is what detectors take pages in as; soft import for type-checking.
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from src.processing.pdf.base import PdfPage, Rect


# ---------------------------------------------------------------------------
# The shared vocabulary
# ---------------------------------------------------------------------------

# Canonical region labels, as consumers know them. Detectors map their model's
# native class names onto these; anything that has no sensible mapping keeps its
# native name (consumers ignore labels they do not know).
LABEL_TABLE = "table"
LABEL_TABLE_CAPTION = "table_caption"
LABEL_TABLE_FOOTNOTE = "table_footnote"
LABEL_IMAGE = "image"
LABEL_IMAGE_CAPTION = "image_caption"

LABEL_TEXT = "text"
LABEL_TITLE = "title"

CANONICAL_LABELS = frozenset({
    LABEL_TABLE, LABEL_TABLE_CAPTION, LABEL_TABLE_FOOTNOTE,
    LABEL_IMAGE, LABEL_IMAGE_CAPTION, LABEL_TEXT, LABEL_TITLE,
})

#: A caption is assigned to a panel group only when that group is an
#: UNAMBIGUOUS match: it must contain at least this fraction of the caption's
#: width, and at least `CAPTION_OWNER_MARGIN` times as much as any rival.
#:
#: An absolute threshold cannot do this job, and that was measured rather
#: than assumed. On Silberbush-2019 p5 a column caption sits at 0.53 against
#: a rival's 0.00; on Collagen structure p18 a single full-width caption
#: under one two-column figure sits at 0.39 against 0.38. No cutoff
#: separates 0.53 from 0.39 -- but "one clear winner" separates them
#: cleanly, which is exactly the question being asked.
#: How much of a panel's width must sit inside a caption's width before the
#: panel is judged to belong to that caption. On a two-column page a panel
#: and its own column's caption overlap almost completely, while a panel in
#: the other column overlaps it not at all, so the test is not delicate.
PANEL_CAPTION_OVERLAP = 0.5


@dataclass(frozen=True)
class Detection:
    """
    One labelled region on a page.

    Attributes:
        label: canonical region type (see module docstring). Compare against the
            LABEL_* constants, not string literals.
        box_px: (x0, y0, x1, y1) in pixels of the rendered page image,
            top-left origin. Ready to crop with `page_image.crop(box_px)`.
        score: detector confidence in [0, 1]. Detectors that do not report
            confidence (the grounding VLM) use 1.0, so downstream thresholds
            never accidentally drop their boxes.
    """

    label: str
    box_px: Tuple[int, int, int, int]
    score: float = 1.0

    def rect(self) -> Rect:
        """The box as a `Rect`, for IoU checks against other geometry."""
        return Rect(*self.box_px)


# ---------------------------------------------------------------------------
# The detector protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LayoutDetector(Protocol):
    """
    Something that can find labelled regions on a rendered page.

    Implementations in this package:
    - `OnnxLayoutDetector` — PP-DocLayout via onnxruntime; CPU, ~15 ms/page.
    - `VlmGroundingDetector` — wraps a grounding-capable VLM call; GPU, slow,
      kept as a config-selectable fallback.
    """

    @property
    def name(self) -> str:
        """Short identifier used in config and logs, e.g. "onnx" or "vlm"."""
        ...

    @property
    def supports_figures(self) -> bool:
        """
        Whether this detector emits figure boxes (`image` / `image_caption`).

        Figure extraction is skipped entirely when False, rather than running
        against labels that will never appear.
        """
        ...

    def detect(self, page_image: "Image.Image",
               page: Optional[PdfPage] = None) -> List[Detection]:
        """
        Find every labelled region on the rendered page.

        Args:
            page_image: the rendered page, the coordinate space of the result.
            page: the PDF page behind the image, when available. Detectors that
                can exploit the text layer (e.g. to snap loose boxes to text
                margins) may use it; pure vision detectors ignore it.

        Returns:
            All detections, in the order the model emitted them. Empty list
            when the page has no recognizable regions — never None.
        """
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def detections_with_label(detections: Sequence[Detection], label: str) -> List[Detection]:
    """The subset of detections carrying `label`, order preserved."""
    return [d for d in detections if d.label == label]


def group_figure_panels(
    detections: Sequence[Detection],
    page_size: Tuple[int, int],
    *,
    label: str = LABEL_IMAGE,
    max_gap_fraction: float = 0.05,
    caption_label=None,
) -> List[Detection]:
    """
    Merge the separate panels of a multi-panel figure into one box.

    For biology researchers: a figure like "Fig. 2 (a) ... (b) ... (c) ..." is
    one figure with one caption, but a layout model sees three pictures and
    boxes each separately. Cropping them individually would scatter one figure
    across three files and orphan the caption. This merges panels that sit
    close enough together to belong to the same figure.

    Two panels merge when the gap between their rectangles is under
    `max_gap_fraction` of the page height *and* no caption sits between them —
    a caption in the gap is the signal that these are two different figures
    stacked on the page, not two panels of one.

    Measured on the benchmark paper: without this, pages with multi-panel
    figures score IoU 0.36–0.47 against the known-good boxes; with it, all
    five figures land at 0.90–0.95.

    Args:
        detections: all detections for the page.
        page_size: (width, height) of the rendered page image, in pixels.
        label: which label to group. Defaults to figures.
        max_gap_fraction: largest gap that still counts as "same figure",
            as a fraction of page height.

    Returns:
        A new list: merged boxes replace their panels, everything else is
        passed through unchanged and in order.
    """
    panels = [d for d in detections if d.label == label]
    if len(panels) < 2:
        return list(detections)

    max_gap = page_size[1] * max_gap_fraction
    captions = [d for d in detections if d.label.endswith("_caption")]
    groups = _merge_panel_groups(panels, captions, max_gap, caption_label)

    merged = [_union_detection(group) for group in groups]
    result: List[Detection] = []
    emitted = False
    for det in detections:
        if det.label != label:
            result.append(det)
        elif not emitted:  # put all merged figures at the first panel's place
            result.extend(merged)
            emitted = True
    return result


def _merge_panel_groups(panels: List[Detection], captions: List[Detection],
                        max_gap: float, caption_label=None
                        ) -> List[List[Detection]]:
    """
    Cluster panels into figures by repeatedly joining mergeable pairs.

    A simple fixed-point loop rather than union-find: pages carry a handful of
    panels, so clarity is worth more here than asymptotics.
    """
    # Which printed figure each panel belongs to, decided ONCE from the
    # panels and the captions rather than from the partly-built groups.
    # Order-independence is the whole point: an earlier version asked
    # "do these two GROUPS own different captions?", and because merging is
    # greedy, a panel of the right-hand figure could join the left-hand group
    # before it owned any caption -- putting the boundary one panel too far
    # right. Measured on Silberbush-2019 p5 at the pipeline's real render
    # size, which split at x=1943 where the columns part at x~1350.
    belongs_to = _partition_panels_by_caption(panels, captions, caption_label)

    groups = [[index] for index in range(len(panels))]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _groups_belong_together(
                        [panels[k] for k in groups[i]],
                        [panels[k] for k in groups[j]],
                        captions, max_gap,
                        groups[i], groups[j], belongs_to):
                    groups[i] = groups[i] + groups[j]
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    return [[panels[k] for k in group] for group in groups]


def _partition_panels_by_caption(panels: Sequence[Detection],
                                 captions: Sequence[Detection],
                                 caption_label=None) -> dict:
    """Map each panel's index to the index of the caption it belongs to.

    A panel belongs to the labelled caption that sits below it and overlaps
    it most horizontally -- which on a two-column page is the caption in its
    own column. Panels belonging to different captions are different figures
    and must never merge, whatever order the merge loop visits them in.

    Only captions that START with a printed figure label count, when the
    caption text is readable. A block without one is the CONTINUATION of a
    caption set across two columns, not a caption of its own; treating it as
    one splits a single figure. With fewer than two labelled captions there
    is nothing to separate, so the partition is empty and merging proceeds on
    geometry alone exactly as before.
    """
    labelled = [
        (index, caption) for index, caption in enumerate(captions)
        if caption.label == LABEL_IMAGE_CAPTION
        and (caption_label is None or caption_label(caption.box_px))
    ]
    if len(labelled) < 2:
        return {}

    belongs_to = {}
    for panel_index, panel in enumerate(panels):
        best_caption, best_overlap = None, 0.0
        for caption_index, caption in labelled:
            if caption.box_px[1] < panel.box_px[3]:
                continue  # this caption sits above the panel, so is not its
            overlap = _fraction_inside(
                panel.box_px[0], panel.box_px[2],
                caption.box_px[0], caption.box_px[2])
            if overlap > best_overlap:
                best_caption, best_overlap = caption_index, overlap
        if best_caption is not None and best_overlap >= PANEL_CAPTION_OVERLAP:
            belongs_to[panel_index] = best_caption
    return belongs_to


def _groups_belong_together(a: List[Detection], b: List[Detection],
                            captions: List[Detection], max_gap: float,
                            indices_a: Sequence[int], indices_b: Sequence[int],
                            belongs_to: dict) -> bool:
    """True when any panel of `a` is close to any panel of `b`, uninterrupted."""
    # Two DIFFERENT figures printed side by side in separate columns are close
    # enough to merge on distance alone, and `_caption_between` cannot part
    # them: it looks only at a VERTICAL gap, and side-by-side panels overlap
    # in y, so no caption ever sits in a gap between them. A two-column
    # paper's gutter is routinely narrower than `max_gap`, so without this
    # check Figure 3 and Figure 4 of a two-column page become one figure and
    # one of them stops existing. Measured on Silberbush-2019 p5.
    if _belong_to_different_figures(indices_a, indices_b, belongs_to):
        return False

    # Panels KNOWN to belong to the same printed figure merge on distance
    # alone. The caption-in-the-gap veto exists to separate two stacked
    # figures, and it misfires here: a layout model boxes the panel letters
    # "A", "B", "C" printed inside a figure as captions, and one sitting
    # between two panel rows would split a figure the partition has already
    # identified as one. Measured on Collagen structure p19, where three
    # ~50px "caption" boxes are panel letters of Figure 13.
    if _belong_to_same_figure(indices_a, indices_b, belongs_to):
        return any(_rect_gap(p.rect(), q.rect()) <= max_gap
                   for p in a for q in b)

    return any(
        _rect_gap(p.rect(), q.rect()) <= max_gap
        and not _caption_between(p, q, captions)
        for p in a for q in b
    )


def _belong_to_same_figure(indices_a: Sequence[int], indices_b: Sequence[int],
                           belongs_to: dict) -> bool:
    """Whether both groups' panels are known to be of ONE printed figure."""
    captions_a = {belongs_to[i] for i in indices_a if i in belongs_to}
    captions_b = {belongs_to[i] for i in indices_b if i in belongs_to}
    return bool(captions_a) and captions_a == captions_b


def _belong_to_different_figures(indices_a: Sequence[int],
                                 indices_b: Sequence[int],
                                 belongs_to: dict) -> bool:
    """Whether these two groups hold panels of two DIFFERENT printed figures."""
    captions_a = {belongs_to[i] for i in indices_a if i in belongs_to}
    captions_b = {belongs_to[i] for i in indices_b if i in belongs_to}
    return bool(captions_a and captions_b and captions_a != captions_b)


def _fraction_inside(a0: float, a1: float, b0: float, b1: float) -> float:
    """How much of span `a` lies within span `b`, as a fraction of `a`."""
    width = a1 - a0
    if width <= 0:
        return 0.0
    return max(0.0, min(a1, b1) - max(a0, b0)) / width


def _group_box(group: List[Detection]) -> Tuple[int, int, int, int]:
    """The bounding box of every panel in the group."""
    boxes = [d.box_px for d in group]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _rect_gap(a: Rect, b: Rect) -> float:
    """
    Shortest distance between two rectangles; 0 when they touch or overlap.

    Measured per axis and combined, so panels that sit side by side and panels
    that sit stacked are both handled by one number.
    """
    dx = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return (dx ** 2 + dy ** 2) ** 0.5


def _caption_between(a: Detection, b: Detection,
                     captions: Sequence[Detection]) -> bool:
    """
    Whether a caption's centre falls in the vertical gap separating two panels.

    That is the mark of two distinct stacked figures: each carries its own
    caption, so the caption interrupts what would otherwise look like one
    tall figure.
    """
    top, bottom = sorted((a.rect(), b.rect()), key=lambda r: r.y0)
    gap_start, gap_end = top.y1, bottom.y0
    if gap_end <= gap_start:
        return False  # the panels overlap vertically; no gap to interrupt
    return any(
        gap_start < (c.box_px[1] + c.box_px[3]) / 2 < gap_end
        for c in captions
    )


def _union_detection(group: List[Detection]) -> Detection:
    """One detection spanning every panel in the group."""
    if len(group) == 1:
        return group[0]
    return Detection(
        label=group[0].label,
        box_px=(
            min(d.box_px[0] for d in group), min(d.box_px[1] for d in group),
            max(d.box_px[2] for d in group), max(d.box_px[3] for d in group),
        ),
        score=min(d.score for d in group),
    )


def scale_normalized_box(
    box_normalized: Sequence[float],
    image_size: Tuple[int, int],
    coordinate_scale: float,
) -> Tuple[int, int, int, int]:
    """
    Map a box from a model's normalized grid onto pixels of the rendered page.

    Grounding VLMs emit coordinates on a fixed grid (DeepSeek: 0–1000)
    regardless of the image's actual size; ONNX detectors work at their input
    resolution. Both reduce to a proportional scale, clamped to the image so a
    slightly-out-of-range box never produces a negative or oversized crop.
    """
    width, height = image_size
    x0, y0, x1, y1 = (float(v) / coordinate_scale for v in box_normalized)
    return (
        max(0, min(width, round(x0 * width))),
        max(0, min(height, round(y0 * height))),
        max(0, min(width, round(x1 * width))),
        max(0, min(height, round(y1 * height))),
    )
