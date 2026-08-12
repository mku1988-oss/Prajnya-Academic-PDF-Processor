#!/usr/bin/env python3
"""
Layout detection on the CPU — the detector that makes the GPU optional.

For biology researchers: this runs PP-DocLayoutV3, a small document-layout
model, through onnxruntime on the CPU. It answers "where are the tables,
figures, and captions on this page?" in a fraction of a second, where the VLM
grounding fallback needs ~17 s of GPU time for the same boxes. Because it is
cheap it runs on *every* page up front, which is what lets the pipeline decide
per page whether the GPU is needed at all (see plan.md, Stage 2).

Model file: `models/layout/pp-doclayoutv3.onnx` — Apache-2.0, fetched per
`models/README.md`, deliberately not tracked in git.

## What the model emits and how it becomes `Detection`s

The model predicts 25 native classes (see `PP_DOCLAYOUTV3_LABELS`) plus a
reading-order index per box. Three things happen on the way out:

1. Native labels are mapped to the canonical vocabulary in `base.py`
   (`table` stays `table`, `doc_title` becomes `title`, `chart` counts as an
   `image`, ...). Unmapped native labels pass through unchanged.
2. The model has ONE generic caption class (`figure_title`) covering table,
   figure, and chart captions alike. Each caption is re-labelled
   `table_caption` or `image_caption` by whichever table/image body sits
   nearest — the same adjacency the human eye uses. Likewise
   `vision_footnote` becomes `table_footnote` when a table is its neighbour.
3. Detections are sorted by the model's reading-order index, so downstream
   consumers receive them in reading order for free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    LABEL_TEXT,
    LABEL_TITLE,
    Detection,
    group_figure_panels,
)
from src.processing.pdf.base import PdfPage

# The 25 native classes, index == position. Verbatim from the official
# PaddlePaddle/PP-DocLayoutV3 inference.yml label_list — do not reorder.
PP_DOCLAYOUTV3_LABELS = (
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer", "footer_image",
    "footnote", "formula_number", "header", "header_image", "image",
    "inline_formula", "number", "paragraph_title", "reference",
    "reference_content", "seal", "table", "text", "vertical_text",
    "vision_footnote",
)

# Native -> canonical, for the labels that map directly. `figure_title` and
# `vision_footnote` are handled by adjacency instead (see module docstring).
_DIRECT_LABEL_MAP: Dict[str, str] = {
    "table": LABEL_TABLE,
    "image": LABEL_IMAGE,
    "chart": LABEL_IMAGE,        # a chart is cropped and captioned like a figure
    "doc_title": LABEL_TITLE,
    "text": LABEL_TEXT,
}

_GENERIC_CAPTION = "figure_title"     # one native class for ALL caption kinds
_GENERIC_FOOTNOTE = "vision_footnote"  # footnote attached to a table or figure

#: Fixed network input resolution (the model resizes without preserving
#: aspect ratio; boxes come back already scaled to the original image).
_INPUT_SIZE = 800

#: Default confidence cutoff, from the model's own inference config.
DEFAULT_SCORE_THRESHOLD = 0.5


class OnnxLayoutDetector:
    """
    `LayoutDetector` implementation backed by PP-DocLayoutV3 via onnxruntime.

    Construction loads the model once (~131 MB); keep one instance per run.
    Inference is CPU-only by design — the whole point is not touching the GPU.
    """

    def __init__(self, model_path: Path | str,
                 score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                 group_panels: bool = True):
        import onnxruntime  # local: only this class needs it

        self._group_panels = group_panels
        self._model_path = Path(model_path)
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"layout model not found at {self._model_path} — "
                "fetch it per models/README.md"
            )
        self._score_threshold = score_threshold
        self._session = onnxruntime.InferenceSession(
            str(self._model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._session.get_inputs()]

    @property
    def name(self) -> str:
        return "onnx"

    @property
    def supports_figures(self) -> bool:
        return True

    def detect(self, page_image: "Image.Image",
               page: Optional[PdfPage] = None) -> List[Detection]:
        """
        Find every labelled region on the rendered page.

        Returns detections in the model's reading order, boxes in pixels of
        `page_image`. The text layer (`page`) is not needed by this detector.

        Multi-panel figures are merged into one box unless the detector was
        built with `group_panels=False` — see `group_figure_panels`.
        """
        raw = self._run_model(page_image)
        detections = self._raw_to_detections(raw, page_image.size)
        detections = _assign_satellite_labels(detections)
        if self._group_panels:
            detections = group_figure_panels(detections, page_image.size)
        return detections

    # --- model I/O (helpers after the main methods, per CLAUDE.md) ---------

    def _run_model(self, page_image: "Image.Image") -> np.ndarray:
        """Preprocess, run the network, and return the raw (N, 7+) array."""
        blob, scale_h, scale_w = _preprocess(page_image)
        feed = self._build_feed(blob, scale_h, scale_w)
        outputs = self._session.run(None, feed)
        # The detection tensor is the one shaped (N, 7): one row per box.
        for out in outputs:
            if out.ndim == 2 and out.shape[1] >= 6:
                return out
        raise RuntimeError(
            f"no detection tensor in model output "
            f"(shapes: {[o.shape for o in outputs]})"
        )

    def _build_feed(self, blob: np.ndarray, scale_h: float,
                    scale_w: float) -> Dict[str, np.ndarray]:
        """
        Route the three inputs to the names this export uses.

        Paddle detection exports take the image plus two bookkeeping tensors
        (network input shape and the resize factors); names vary between
        exports, so match on the conventional names with a shape fallback.
        """
        im_shape = np.array([[_INPUT_SIZE, _INPUT_SIZE]], dtype=np.float32)
        scale_factor = np.array([[scale_h, scale_w]], dtype=np.float32)
        feed: Dict[str, np.ndarray] = {}
        for name in self._input_names:
            lowered = name.lower()
            if "image" in lowered or "x" == lowered:
                feed[name] = blob
            elif "shape" in lowered:
                feed[name] = im_shape
            elif "scale" in lowered:
                feed[name] = scale_factor
            else:
                raise RuntimeError(f"unrecognized model input {name!r}")
        return feed

    def _raw_to_detections(self, raw: np.ndarray,
                           image_size: Tuple[int, int]) -> List[Detection]:
        """
        Rows above the score threshold, as canonical-labelled `Detection`s.

        Row layout: [class_id, score, x0, y0, x1, y1, (read_order)]. Boxes
        arrive already in original-image pixels (the export applies
        scale_factor internally); clamp defensively and sort by read order.
        """
        width, height = image_size
        keep: List[Tuple[float, Detection]] = []
        for row in raw:
            class_id, score = int(row[0]), float(row[1])
            if score < self._score_threshold or class_id < 0:
                continue
            native = (PP_DOCLAYOUTV3_LABELS[class_id]
                      if class_id < len(PP_DOCLAYOUTV3_LABELS) else f"class_{class_id}")
            box = (
                max(0, min(width, round(float(row[2])))),
                max(0, min(height, round(float(row[3])))),
                max(0, min(width, round(float(row[4])))),
                max(0, min(height, round(float(row[5])))),
            )
            read_order = float(row[6]) if raw.shape[1] > 6 else float(len(keep))
            keep.append((read_order, Detection(
                label=_DIRECT_LABEL_MAP.get(native, native),
                box_px=box,
                score=score,
            )))
        keep.sort(key=lambda item: item[0])
        return [d for _, d in keep]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _preprocess(page_image: "Image.Image") -> Tuple[np.ndarray, float, float]:
    """
    Page image -> the (1, 3, 800, 800) float blob the network expects.

    Resize does NOT preserve aspect ratio (the model was trained that way);
    the returned scale factors let the network map boxes back to the original
    pixels. Normalization is ImageNet mean/std, matching the conversion's
    reference implementation.
    """
    orig_w, orig_h = page_image.size
    resized = page_image.convert("RGB").resize(
        (_INPUT_SIZE, _INPUT_SIZE), Image.BILINEAR
    )
    blob = np.asarray(resized, dtype=np.float32) / 255.0
    blob = (blob - np.array([0.485, 0.456, 0.406], dtype=np.float32)) \
        / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    return blob, _INPUT_SIZE / orig_h, _INPUT_SIZE / orig_w


def _assign_satellite_labels(detections: List[Detection]) -> List[Detection]:
    """
    Turn generic captions/footnotes into table- or figure-specific ones.

    PP-DocLayoutV3 emits one caption class for tables, figures, and charts
    alike. Each caption (and vision footnote) is claimed by the nearest table
    or image body — measured centre-to-centre, the way captions actually sit
    on a page. A caption with no body anywhere near it stays generic, which
    consumers ignore.
    """
    tables = [d for d in detections if d.label == LABEL_TABLE]
    images = [d for d in detections if d.label == LABEL_IMAGE]
    if not tables and not images:
        return detections

    relabelled: List[Detection] = []
    for det in detections:
        if det.label == _GENERIC_CAPTION:
            near_table = _nearest_distance(det, tables)
            near_image = _nearest_distance(det, images)
            new_label = (LABEL_TABLE_CAPTION if near_table <= near_image
                         else LABEL_IMAGE_CAPTION)
            relabelled.append(Detection(new_label, det.box_px, det.score))
        elif det.label == _GENERIC_FOOTNOTE and tables:
            near_table = _nearest_distance(det, tables)
            near_image = _nearest_distance(det, images)
            if near_table <= near_image:
                relabelled.append(
                    Detection(LABEL_TABLE_FOOTNOTE, det.box_px, det.score))
            else:
                relabelled.append(det)
        else:
            relabelled.append(det)
    return relabelled


def _nearest_distance(det: Detection, bodies: Sequence[Detection]) -> float:
    """Centre-to-centre distance to the closest body; inf when none exist."""
    if not bodies:
        return float("inf")
    cx = (det.box_px[0] + det.box_px[2]) / 2
    cy = (det.box_px[1] + det.box_px[3]) / 2
    best = float("inf")
    for body in bodies:
        bx = (body.box_px[0] + body.box_px[2]) / 2
        by = (body.box_px[1] + body.box_px[3]) / 2
        best = min(best, ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5)
    return best
