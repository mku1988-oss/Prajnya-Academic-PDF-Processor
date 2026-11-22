#!/usr/bin/env python3
"""
Tests for the layout-detection seam (`src/processing/layout/`).

For biology researchers: these tests prove the layout interface behaves the
same regardless of which detector sits behind it. The `VlmGroundingDetector`
is exercised against DeepSeek-OCR-2's *actual* grounding syntax (copied from
the V5 profile YAML) in both surface forms the model is known to emit — a
regex written for one form silently matches nothing on the other, which looks
exactly like a total detection failure.

No GPU, no model files: the VLM call is a canned string.
"""

from PIL import Image

from src.processing.layout import (
    LABEL_IMAGE,
    LABEL_IMAGE_CAPTION,
    LABEL_TABLE,
    LABEL_TABLE_CAPTION,
    Detection,
    GroundingSpec,
    LayoutDetector,
    VlmGroundingDetector,
    detections_with_label,
    scale_normalized_box,
)

# The production grounding format for DeepSeek-OCR-2, verbatim from V5's
# deepseek-ocr-v2.yaml. If V5 changes its config this copy does not follow —
# that is fine, the test pins the format this parser must keep accepting.
DEEPSEEK_GROUNDING_FORMAT = {
    "box_pattern": r"(?:^|<\|ref\|>)\s*(?P<label>[a-z_]+)(?:<\|/ref\|>\s*<\|det\|>)?\s*\[\[(?P<coords>[\d,\s]+)\]\]",
    "coordinate_scale": 1000,
    "table_label": "table",
    "caption_label": "table_caption",
    "footnote_label": "table_footnote",
    "image_label": "image",
    "image_caption_label": "image_caption",
    "prompt": "<|grounding|>Locate all layout elements.",
}

# A 1000x1000 page image makes normalized 0-1000 coords equal pixel coords,
# so expected boxes can be read straight off the grounding string.
PAGE_1000 = Image.new("RGB", (1000, 1000), "white")


def make_detector(response: str) -> VlmGroundingDetector:
    """A detector whose "model" always answers with the canned string."""
    spec = GroundingSpec.from_config(DEEPSEEK_GROUNDING_FORMAT)
    return VlmGroundingDetector(spec, run_grounding=lambda image: response)


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------

def test_detection_rect_roundtrip():
    detection = Detection(label=LABEL_TABLE, box_px=(10, 20, 110, 220))
    rect = detection.rect()
    assert (rect.x0, rect.y0, rect.x1, rect.y1) == (10, 20, 110, 220)
    assert rect.width == 100 and rect.height == 200


def test_detections_with_label_filters_and_preserves_order():
    detections = [
        Detection(LABEL_TABLE, (0, 0, 1, 1)),
        Detection(LABEL_IMAGE, (0, 0, 2, 2)),
        Detection(LABEL_TABLE, (0, 0, 3, 3)),
    ]
    tables = detections_with_label(detections, LABEL_TABLE)
    assert [d.box_px[2] for d in tables] == [1, 3]


def test_scale_normalized_box_scales_proportionally():
    # A box at half-grid on a 2000x1000 image lands at half the image.
    box = scale_normalized_box([0, 0, 500, 500], (2000, 1000), 1000)
    assert box == (0, 0, 1000, 500)


def test_scale_normalized_box_clamps_out_of_range():
    # Models occasionally emit coords slightly past the grid; the crop box
    # must never leave the image.
    box = scale_normalized_box([-10, 0, 1020, 990], (1000, 800), 1000)
    assert box == (0, 0, 1000, 792)


# ---------------------------------------------------------------------------
# GroundingSpec
# ---------------------------------------------------------------------------

def test_spec_reports_figure_support_from_config():
    assert GroundingSpec.from_config(DEEPSEEK_GROUNDING_FORMAT).supports_figures

    without_figures = {**DEEPSEEK_GROUNDING_FORMAT}
    without_figures.pop("image_label")
    without_figures.pop("image_caption_label")
    assert not GroundingSpec.from_config(without_figures).supports_figures


# ---------------------------------------------------------------------------
# VlmGroundingDetector — both DeepSeek surface forms
# ---------------------------------------------------------------------------

def test_parses_llama_server_bare_label_form():
    detector = make_detector(
        "table[[85, 101, 915, 786]]\ntable_caption[[85, 60, 915, 95]]"
    )
    detections = detector.detect(PAGE_1000)
    assert [d.label for d in detections] == [LABEL_TABLE, LABEL_TABLE_CAPTION]
    assert detections[0].box_px == (85, 101, 915, 786)
    # The VLM reports no confidence, so score must default high, never 0.
    assert detections[0].score == 1.0


def test_parses_mtmd_cli_ref_det_form():
    detector = make_detector(
        "<|ref|>image<|/ref|><|det|>[[100, 200, 400, 500]]<|/det|>"
        "<|ref|>image_caption<|/ref|><|det|>[[100, 510, 400, 560]]<|/det|>"
    )
    detections = detector.detect(PAGE_1000)
    assert [d.label for d in detections] == [LABEL_IMAGE, LABEL_IMAGE_CAPTION]
    assert detections[1].box_px == (100, 510, 400, 560)


def test_boxes_scale_with_image_size():
    # Same grounding string, half-size render -> half-size pixel boxes.
    detector = make_detector("table[[100, 200, 300, 400]]")
    detections = detector.detect(Image.new("RGB", (500, 500), "white"))
    assert detections[0].box_px == (50, 100, 150, 200)


def test_empty_and_malformed_responses_yield_no_detections():
    assert make_detector("").detect(PAGE_1000) == []
    assert make_detector("The page contains a table.").detect(PAGE_1000) == []
    # A box with the wrong number of coordinates is skipped, not guessed at.
    assert make_detector("table[[1, 2, 3]]").detect(PAGE_1000) == []


def test_unknown_native_labels_pass_through():
    # Consumers ignore labels they do not know; the parser must not drop them.
    detector = make_detector("formula[[10, 10, 90, 90]]")
    detections = detector.detect(PAGE_1000)
    assert detections[0].label == "formula"


def test_satisfies_layout_detector_protocol():
    detector = make_detector("")
    assert isinstance(detector, LayoutDetector)
    assert detector.name == "vlm"
    assert detector.supports_figures
