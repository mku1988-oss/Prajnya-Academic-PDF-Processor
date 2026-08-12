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
    groups = _merge_panel_groups(panels, captions, max_gap)

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
                        max_gap: float) -> List[List[Detection]]:
    """
    Cluster panels into figures by repeatedly joining mergeable pairs.

    A simple fixed-point loop rather than union-find: pages carry a handful of
    panels, so clarity is worth more here than asymptotics.
    """
    groups = [[p] for p in panels]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _groups_belong_together(groups[i], groups[j], captions, max_gap):
                    groups[i] = groups[i] + groups[j]
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    return groups


def _groups_belong_together(a: List[Detection], b: List[Detection],
                            captions: List[Detection], max_gap: float) -> bool:
    """True when any panel of `a` is close to any panel of `b`, uninterrupted."""
    return any(
        _rect_gap(p.rect(), q.rect()) <= max_gap
        and not _caption_between(p, q, captions)
        for p in a for q in b
    )


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
