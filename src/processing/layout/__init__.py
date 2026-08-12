#!/usr/bin/env python3
"""
Layout detection layer — find tables, figures, and captions on a page.

    from src.processing.layout import Detection, LayoutDetector

    detections = detector.detect(page_image)
    tables = detections_with_label(detections, LABEL_TABLE)

Two implementations exist behind the same interface (see `base.py`):
`OnnxLayoutDetector` (CPU, ~15 ms/page, the default) and
`VlmGroundingDetector` (GPU grounding VLM, the proven fallback). Unlike the
PDF backends, detectors need construction arguments (model path, grounding
spec, a provider callable), so they are built by the profile registry rather
than a name-only factory here.
"""

from __future__ import annotations

from .base import (
    CANONICAL_LABELS,
    LABEL_IMAGE,
    LABEL_IMAGE_CAPTION,
    LABEL_TABLE,
    LABEL_TABLE_CAPTION,
    LABEL_TABLE_FOOTNOTE,
    LABEL_TEXT,
    LABEL_TITLE,
    Detection,
    LayoutDetector,
    detections_with_label,
    group_figure_panels,
    scale_normalized_box,
)
from .vlm_detector import GroundingSpec, VlmGroundingDetector

__all__ = [
    "CANONICAL_LABELS",
    "LABEL_IMAGE", "LABEL_IMAGE_CAPTION", "LABEL_TABLE", "LABEL_TABLE_CAPTION",
    "LABEL_TABLE_FOOTNOTE", "LABEL_TEXT", "LABEL_TITLE",
    "Detection", "LayoutDetector", "GroundingSpec", "VlmGroundingDetector",
    "detections_with_label", "group_figure_panels", "scale_normalized_box",
]
