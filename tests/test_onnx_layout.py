#!/usr/bin/env python3
"""
Tests for the CPU layout detector (`OnnxLayoutDetector`) — the Stage 2 gate.

For biology researchers: these check that the cheap CPU detector finds the same
regions on the benchmark paper that the expensive GPU pipeline found, because
that agreement is the whole justification for replacing it (plan.md, Stage 2).

The gate from plan.md is **IoU >= 0.7** against the known-good figure boxes.
Measured worst case is 0.90, so the assertions have real headroom.

Two groups of tests:

- **Pure-geometry tests** run everywhere: panel grouping and label mapping use
  hand-built detections, no model file and no PDF.
- **Model tests** are skipped unless `models/layout/pp-doclayoutv3.onnx` and
  the benchmark PDF are both present. The weights are gitignored (131 MB), so
  a clean checkout skips them rather than failing.
"""

import json
from pathlib import Path

import pytest

from src.processing.layout import (
    LABEL_IMAGE,
    LABEL_IMAGE_CAPTION,
    LABEL_TABLE,
    LABEL_TABLE_CAPTION,
    LABEL_TABLE_FOOTNOTE,
    Detection,
    LayoutDetector,
    detections_with_label,
    group_figure_panels,
)
from src.processing.pdf import Rect, get_backend
from tests.local_samples import BENCHMARK_PDF as LOCAL_BENCHMARK_PDF

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "layout" / "pp-doclayoutv3.onnx"
BASELINE = REPO_ROOT / "tests" / "baselines" / "nmr_known_good_document.json"
BENCHMARK_PDF = LOCAL_BENCHMARK_PDF

#: The baseline's figure boxes were recorded from a 300 dpi render; IoU is only
#: meaningful when the comparison render matches.
BASELINE_DPI = 300

#: plan.md's Stage 2 acceptance threshold for box agreement.
IOU_GATE = 0.7

needs_model = pytest.mark.skipif(
    not (MODEL_PATH.is_file() and BENCHMARK_PDF.is_file()),
    reason="layout model or benchmark PDF unavailable (see models/README.md)",
)


# ---------------------------------------------------------------------------
# Panel grouping — pure geometry, no model needed
# ---------------------------------------------------------------------------

PAGE = (2480, 3309)  # the benchmark's 300 dpi page size


def test_stacked_panels_merge_into_one_figure():
    """Three panels of one figure, small gaps between them -> one box."""
    detections = [
        Detection(LABEL_IMAGE, (252, 285, 1058, 880)),
        Detection(LABEL_IMAGE, (252, 948, 1039, 1566)),
        Detection(LABEL_IMAGE, (251, 1609, 1173, 2363)),
        Detection(LABEL_IMAGE_CAPTION, (155, 2392, 1239, 3113)),
    ]
    figures = detections_with_label(group_figure_panels(detections, PAGE), LABEL_IMAGE)
    assert len(figures) == 1
    assert figures[0].box_px == (251, 285, 1173, 2363)


def test_side_by_side_panels_merge():
    """Panels laid out horizontally are one figure too."""
    detections = [
        Detection(LABEL_IMAGE, (407, 304, 1197, 904)),
        Detection(LABEL_IMAGE, (1234, 298, 2026, 899)),
    ]
    figures = detections_with_label(group_figure_panels(detections, PAGE), LABEL_IMAGE)
    assert len(figures) == 1
    assert figures[0].box_px == (407, 298, 2026, 904)


def test_caption_between_panels_keeps_them_separate():
    """
    Two stacked figures, each with its own caption, must NOT merge.

    This is the safeguard that stops grouping from swallowing a whole page:
    the caption sitting in the gap is what marks a figure boundary.
    """
    detections = [
        Detection(LABEL_IMAGE, (300, 200, 1200, 900)),
        Detection(LABEL_IMAGE_CAPTION, (300, 930, 1200, 1000)),
        Detection(LABEL_IMAGE, (300, 1030, 1200, 1700)),
        Detection(LABEL_IMAGE_CAPTION, (300, 1730, 1200, 1800)),
    ]
    figures = detections_with_label(group_figure_panels(detections, PAGE), LABEL_IMAGE)
    assert len(figures) == 2


def test_two_column_figures_each_with_a_caption_do_not_merge():
    """Two DIFFERENT figures side by side in two columns must not merge.

    A two-column paper's gutter is narrower than the merge threshold, and the
    panels overlap vertically -- so distance says "merge" and the vertical
    caption test cannot fire, because there is no vertical gap for a caption
    to sit in. Each side owning its own caption is what marks the boundary.

    Measured on Silberbush-2019 p5, where Figure 3 (left, six panels) and
    Figure 4 (right, two panels) were merged into one figure and Figure 4
    stopped existing: eight panels detected at 0.88-0.95, and one crop saved.
    """
    detections = [
        # Left column: Figure 3's panels, with its caption beneath them.
        Detection(LABEL_IMAGE, (200, 200, 1150, 1300)),
        Detection(LABEL_IMAGE, (200, 1310, 1150, 1900)),
        Detection(LABEL_IMAGE_CAPTION, (200, 1960, 1150, 2400)),
        # Right column: Figure 4's panels, with ITS caption beneath them.
        # The gutter is 50px -- well inside the merge threshold.
        Detection(LABEL_IMAGE, (1200, 700, 2100, 1500)),
        Detection(LABEL_IMAGE_CAPTION, (1200, 1560, 2100, 2000)),
    ]
    figures = detections_with_label(group_figure_panels(detections, PAGE), LABEL_IMAGE)
    assert len(figures) == 2, "two captioned figures in two columns are two figures"


def test_two_column_panels_sharing_one_caption_still_merge():
    """One figure whose panels straddle both columns stays ONE figure.

    The guard above must not fire here: only one side owns the caption, so
    these are panels of a single figure, not two figures. Measured on
    Collagen structure p18, where Figure 11's panel A sits in the left column
    and panels B and C in the right, under one caption.
    """
    detections = [
        Detection(LABEL_IMAGE, (200, 200, 1150, 1800)),
        Detection(LABEL_IMAGE, (1200, 210, 2100, 900)),
        Detection(LABEL_IMAGE, (1200, 910, 2100, 1800)),
        Detection(LABEL_IMAGE_CAPTION, (200, 1860, 1150, 2000)),
    ]
    figures = detections_with_label(group_figure_panels(detections, PAGE), LABEL_IMAGE)
    assert len(figures) == 1, "panels sharing one caption are one figure"


def test_a_table_caption_does_not_make_a_panel_own_a_caption():
    """A table's caption in the same column must not split a figure.

    Caught while fixing the two-column case: the caption list carries
    `table_caption` too, and on a page holding a figure and a table the
    table's caption often sits under one of the figure's panels. Counting it
    split Collagen structure p18's Figure 11 in two.
    """
    detections = [
        Detection(LABEL_IMAGE, (200, 200, 1150, 1800)),
        Detection(LABEL_IMAGE, (1200, 210, 2100, 1800)),
        Detection(LABEL_IMAGE_CAPTION, (200, 1860, 1150, 2000)),
        # The table's caption, sitting squarely under the right-hand panel.
        Detection(LABEL_TABLE_CAPTION, (1200, 2100, 2100, 2300)),
    ]
    figures = detections_with_label(group_figure_panels(detections, PAGE), LABEL_IMAGE)
    assert len(figures) == 1, "a table caption is not a figure's caption"


def test_a_caption_running_across_two_columns_does_not_split_its_figure():
    """One caption set across two columns must not look like two captions.

    The two-column guard cannot be geometric alone, and that was measured.
    A figure whose panels sit side by side under a single caption typesets
    that caption as TWO column-shaped blocks -- geometrically identical to
    two figures each with its own caption. Only the text separates them:
    the second block is a CONTINUATION and carries no printed label.

    Measured on Tendon assembly p20, p332 and p334, which the geometric
    version of this guard split in two apiece.
    """
    detections = [
        Detection(LABEL_IMAGE, (200, 200, 1150, 1300)),
        Detection(LABEL_IMAGE, (1200, 200, 2100, 1300)),
        Detection(LABEL_IMAGE_CAPTION, (200, 1360, 1150, 1900)),
        Detection(LABEL_IMAGE_CAPTION, (1200, 1360, 2100, 1900)),
    ]
    # Only the first block names a figure; the second continues it.
    labels = {(200, 1360, 1150, 1900): "figure 2.2"}
    figures = detections_with_label(
        group_figure_panels(detections, PAGE,
                            caption_label=lambda box: labels.get(box)),
        LABEL_IMAGE)
    assert len(figures) == 1, "a continuation block is not a second caption"


def test_two_captions_naming_different_figures_do_split():
    """The same geometry, but both blocks name a figure -- so two figures.

    This is Silberbush-2019 p5, where the blocks read "Figure 3." and
    "Figure 4." and the right-hand figure was being lost.
    """
    detections = [
        Detection(LABEL_IMAGE, (200, 200, 1150, 1300)),
        Detection(LABEL_IMAGE, (1200, 200, 2100, 1300)),
        Detection(LABEL_IMAGE_CAPTION, (200, 1360, 1150, 1900)),
        Detection(LABEL_IMAGE_CAPTION, (1200, 1360, 2100, 1900)),
    ]
    labels = {(200, 1360, 1150, 1900): "figure 3",
              (1200, 1360, 2100, 1900): "figure 4"}
    figures = detections_with_label(
        group_figure_panels(detections, PAGE,
                            caption_label=lambda box: labels.get(box)),
        LABEL_IMAGE)
    assert len(figures) == 2, "two captions naming two figures are two figures"


def test_panels_of_one_figure_merge_across_a_panel_letter_box():
    """A panel LETTER boxed as a caption must not split its own figure.

    A layout model boxes the "A", "B", "C" printed inside a multi-panel
    figure as image_captions. One of those sitting between two panel rows
    trips the caption-in-the-gap veto -- which exists to separate two STACKED
    figures -- and splits a figure the caption partition has already
    identified as one.

    This is Collagen structure p19 as measured: Figure 12 in the left column,
    Figure 13 as two stacked panels in the right, and three ~50px boxes that
    are Figure 13's panel letters. The page must yield TWO figures, not three.
    """
    detections = [
        # Figure 12, left column, with its caption below.
        Detection(LABEL_IMAGE, (300, 300, 1150, 965)),
        Detection(LABEL_IMAGE_CAPTION, (191, 1994, 1229, 2454)),
        # Figure 13, right column, two stacked panels under one caption.
        Detection(LABEL_IMAGE, (1300, 310, 2316, 816)),
        Detection(LABEL_IMAGE, (1320, 920, 2318, 1840)),
        Detection(LABEL_IMAGE_CAPTION, (1286, 1949, 2326, 2483)),
        # A panel letter, boxed as a caption, sitting in Figure 13's gap.
        Detection(LABEL_IMAGE_CAPTION, (1300, 852, 1350, 908)),
    ]
    labels = {(191, 1994, 1229, 2454): "figure 12",
              (1286, 1949, 2326, 2483): "figure 13"}
    figures = detections_with_label(
        group_figure_panels(detections, PAGE,
                            caption_label=lambda box: labels.get(box)),
        LABEL_IMAGE)
    assert len(figures) == 2, "a panel letter is not a figure boundary"


def test_two_stacked_figures_still_split_on_a_real_caption():
    """The veto above must survive: a REAL caption between two figures parts them.

    Same shape as the panel-letter case, except the box in the gap is a
    genuine caption naming its own figure. Without this the relaxation would
    merge every stacked pair on the page.
    """
    detections = [
        Detection(LABEL_IMAGE, (300, 200, 1200, 900)),
        Detection(LABEL_IMAGE_CAPTION, (300, 930, 1200, 1000)),
        Detection(LABEL_IMAGE, (300, 1030, 1200, 1700)),
        Detection(LABEL_IMAGE_CAPTION, (300, 1730, 1200, 1800)),
    ]
    labels = {(300, 930, 1200, 1000): "figure 1",
              (300, 1730, 1200, 1800): "figure 2"}
    figures = detections_with_label(
        group_figure_panels(detections, PAGE,
                            caption_label=lambda box: labels.get(box)),
        LABEL_IMAGE)
    assert len(figures) == 2, "two captioned figures stacked are two figures"


def test_distant_figures_do_not_merge():
    """A gap far larger than the threshold means two separate figures."""
    detections = [
        Detection(LABEL_IMAGE, (300, 100, 1200, 600)),
        Detection(LABEL_IMAGE, (300, 2600, 1200, 3100)),
    ]
    figures = detections_with_label(group_figure_panels(detections, PAGE), LABEL_IMAGE)
    assert len(figures) == 2


def test_grouping_preserves_other_detections():
    """Tables and captions pass through grouping untouched."""
    detections = [
        Detection(LABEL_TABLE, (600, 200, 1100, 400)),
        Detection(LABEL_IMAGE, (252, 285, 1058, 880)),
        Detection(LABEL_IMAGE, (252, 948, 1039, 1566)),
        Detection(LABEL_TABLE_CAPTION, (600, 150, 1100, 190)),
    ]
    grouped = group_figure_panels(detections, PAGE)
    assert len(detections_with_label(grouped, LABEL_TABLE)) == 1
    assert len(detections_with_label(grouped, LABEL_TABLE_CAPTION)) == 1
    assert len(detections_with_label(grouped, LABEL_IMAGE)) == 1


def test_single_panel_page_is_unchanged():
    detections = [Detection(LABEL_IMAGE, (300, 100, 1200, 600))]
    assert group_figure_panels(detections, PAGE) == detections


# ---------------------------------------------------------------------------
# The model itself, against the known-good baseline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    from src.processing.layout.onnx_detector import OnnxLayoutDetector
    return OnnxLayoutDetector(MODEL_PATH)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def page_detections(detector, baseline) -> dict:
    """Detections for every page of the benchmark, keyed by page index."""
    found = {}
    with get_backend().open(BENCHMARK_PDF) as doc:
        for index in range(len(baseline["pages"])):
            image = doc.page(index).render(dpi=BASELINE_DPI)
            found[index] = detector.detect(image)
    return found


@needs_model
def test_detector_satisfies_the_protocol(detector):
    assert isinstance(detector, LayoutDetector)
    assert detector.name == "onnx"
    assert detector.supports_figures


@needs_model
def test_every_known_figure_is_found_within_the_iou_gate(page_detections, baseline):
    """
    The Stage 2 gate: each known-good figure box has a detection over IoU 0.7.

    Failing this means the CPU detector is not a safe replacement for the VLM
    grounding pass, and Stage 2 must not proceed.
    """
    checked = 0
    for index, page in enumerate(baseline["pages"]):
        for figure in page.get("metadata", {}).get("figures", []):
            reference = Rect(*figure["bbox"])
            candidates = detections_with_label(page_detections[index], LABEL_IMAGE)
            assert candidates, f"no figure detected on page {index + 1}"
            best = max(reference.iou(d.rect()) for d in candidates)
            assert best >= IOU_GATE, (
                f"{figure['figure_id']}: best IoU {best:.3f} < {IOU_GATE}"
            )
            checked += 1
    assert checked == 5, f"baseline should carry 5 figures, found {checked}"


@needs_model
def test_figure_count_per_page_matches_the_baseline(page_detections, baseline):
    """
    One detected figure per known figure — panel grouping's real job.

    Without grouping the multi-panel pages report 2–3 figures each, which would
    scatter one figure across several crops and orphan its caption.
    """
    for index, page in enumerate(baseline["pages"]):
        expected = page.get("metadata", {}).get("figure_count", 0)
        if not expected:
            continue
        found = len(detections_with_label(page_detections[index], LABEL_IMAGE))
        assert found == expected, f"page {index + 1}: {found} figures, expected {expected}"


@needs_model
def test_both_tables_on_page_four_are_detected(page_detections):
    """
    Page 4 carries the two tables the parity harness checks.

    Table detection is what drives the Stage 3 decision to make a GLM crop read
    instead of a full-page read, so missing one costs content.
    """
    tables = detections_with_label(page_detections[3], LABEL_TABLE)
    assert len(tables) == 2
    assert all(d.score >= 0.5 for d in tables)


@needs_model
def test_table_satellites_are_labelled_by_adjacency(page_detections):
    """
    The model emits one generic caption class; page 4's must resolve to a
    table caption and a table footnote, not figure captions.
    """
    page4 = page_detections[3]
    assert detections_with_label(page4, LABEL_TABLE_CAPTION)
    assert detections_with_label(page4, LABEL_TABLE_FOOTNOTE)


@needs_model
def test_detection_boxes_stay_inside_the_page(page_detections):
    """A box outside the image would crash or silently mis-crop."""
    width, height = PAGE
    for index, detections in page_detections.items():
        for d in detections:
            x0, y0, x1, y1 = d.box_px
            assert 0 <= x0 <= x1 <= width, f"page {index + 1}: bad x range {d.box_px}"
            assert 0 <= y0 <= y1 <= height, f"page {index + 1}: bad y range {d.box_px}"
