#!/usr/bin/env python3
"""
Tests for the consumers of the layout seam: `table_repair` and `figure_extractor`.

For biology researchers: these prove the whole chain works end to end —
detector finds the boxes, `regions.py` makes them safe to cut out, and the two
consumers crop and use them. The point of the V6 rewrite is that both consumers
take `List[Detection]` and never see a model-specific grounding string, so the
CPU and VLM detectors are interchangeable here.

Two groups:

- **Logic tests** run everywhere: splicing, anchors, and crop-safety geometry
  on hand-built detections and synthetic images. No model, no PDF.
- **End-to-end tests** are skipped unless the ONNX weights and the benchmark
  PDF are present, and check the real detector against the parity baseline.
"""

import json
import logging
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from src.processing.layout import (
    LABEL_IMAGE,
    LABEL_IMAGE_CAPTION,
    LABEL_TABLE,
    LABEL_TABLE_CAPTION,
    LABEL_TABLE_FOOTNOTE,
    Detection,
)
from src.processing.layout.regions import (
    ORPHAN_ATTACH_GAP_PX,
    build_ink_mask,
    locate_figure_regions,
    locate_table_regions,
    snap_box_to_whitespace,
)
from src.processing.pdf import Rect, get_backend
from src.processing.utils.figure_extractor import (
    FigureAsset,
    append_figure_anchors,
    extract_and_save_figures,
    safe_doc_id,
)
from src.processing.utils.table_repair import (
    RECOVERED_TABLE_MARKER,
    repair_page_tables,
)
from tests.local_samples import BENCHMARK_PDF as LOCAL_BENCHMARK_PDF

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "layout" / "pp-doclayoutv3.onnx"
BASELINE = REPO_ROOT / "tests" / "baselines" / "nmr_known_good_document.json"
BENCHMARK_PDF = LOCAL_BENCHMARK_PDF
BASELINE_DPI = 300

#: plan.md's Stage 2 box-agreement threshold, same gate as test_onnx_layout.py.
IOU_GATE = 0.7

needs_model = pytest.mark.skipif(
    not (MODEL_PATH.is_file() and BENCHMARK_PDF.is_file()),
    reason="layout model or benchmark PDF unavailable (see models/README.md)",
)

QUIET = logging.getLogger("test")
QUIET.addHandler(logging.NullHandler())


def blank_page(width: int = 800, height: int = 1000) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def page_with_marks(boxes) -> Image.Image:
    """A white page with solid black rectangles — synthetic 'ink' to snap to."""
    page = blank_page()
    draw = ImageDraw.Draw(page)
    for box in boxes:
        draw.rectangle(box, fill="black")
    return page


# ---------------------------------------------------------------------------
# table_repair — splicing logic
# ---------------------------------------------------------------------------

def test_repair_appends_recovered_table_behind_a_marker():
    page = page_with_marks([(100, 300, 700, 500)])
    detections = [Detection(LABEL_TABLE, (100, 300, 700, 500))]

    result = repair_page_tables(
        page, "Prose that survived the read.", detections,
        read_table_crop=lambda crop: "<table><tr><td>7.4</td></tr></table>",
        logger=QUIET,
    )

    assert result.repaired
    assert result.tables_recovered == 1
    assert RECOVERED_TABLE_MARKER in result.text
    assert "7.4" in result.text
    # The prose the reader got right must survive untouched.
    assert "Prose that survived the read." in result.text


def test_repair_is_a_no_op_when_no_table_is_detected():
    """Safe to call unconditionally — that is why it returns the text unchanged."""
    page = blank_page()
    detections = [Detection(LABEL_IMAGE, (100, 100, 400, 400))]

    result = repair_page_tables(page, "Original text.", detections,
                                read_table_crop=lambda crop: "should never be called",
                                logger=QUIET)

    assert not result.repaired
    assert result.text == "Original text."
    assert result.region_boxes == []


def test_repair_reports_boxes_when_the_reread_returns_nothing():
    """
    A failed re-read must not silently look like 'no table here'.

    The boxes are still reported so the failure is visible in the ledger.
    """
    page = page_with_marks([(100, 300, 700, 500)])
    detections = [Detection(LABEL_TABLE, (100, 300, 700, 500))]

    result = repair_page_tables(page, "Text.", detections,
                                read_table_crop=lambda crop: "   ",
                                logger=QUIET)

    assert not result.repaired
    assert result.text == "Text."
    assert len(result.region_boxes) == 1


def test_repair_splices_every_table_on_a_multi_table_page():
    page = page_with_marks([(100, 200, 700, 350), (100, 600, 700, 750)])
    detections = [
        Detection(LABEL_TABLE, (100, 200, 700, 350)),
        Detection(LABEL_TABLE, (100, 600, 700, 750)),
    ]
    grids = iter(["FIRST GRID", "SECOND GRID"])

    result = repair_page_tables(page, "Prose.", detections,
                                read_table_crop=lambda crop: next(grids),
                                logger=QUIET)

    assert result.tables_recovered == 2
    assert "FIRST GRID" in result.text and "SECOND GRID" in result.text


# ---------------------------------------------------------------------------
# regions — crop safety
# ---------------------------------------------------------------------------

def test_snapping_grows_a_tight_box_off_the_glyphs():
    """
    The bug this prevents: a tight box slices a digit and '52.0' reads '52.('.

    A box drawn inside the ink must grow outward until its edges are clear.
    """
    page = page_with_marks([(200, 400, 600, 500)])
    ink = build_ink_mask(page)

    tight = [250, 420, 550, 480]  # strictly inside the ink block
    snapped = snap_box_to_whitespace(tight, ink)

    assert snapped[0] < tight[0] and snapped[1] < tight[1]
    assert snapped[2] > tight[2] and snapped[3] > tight[3]
    # And it must clear the ink entirely, not merely move.
    assert snapped[0] <= 200 and snapped[2] >= 600


def test_table_caption_and_footnote_are_folded_into_the_crop():
    """
    Cropping the bare table would discard the footnote defining its symbols.
    """
    page = page_with_marks([(100, 260, 700, 280), (100, 300, 700, 500),
                            (100, 520, 700, 545)])
    detections = [
        Detection(LABEL_TABLE_CAPTION, (100, 260, 700, 280)),
        Detection(LABEL_TABLE, (100, 300, 700, 500)),
        Detection(LABEL_TABLE_FOOTNOTE, (100, 520, 700, 545)),
    ]

    regions = locate_table_regions(detections, page, logger=QUIET)

    assert len(regions) == 1
    region = regions[0]
    assert region.has_caption and region.has_footnote
    # The crop must span from above the caption to below the footnote.
    assert region.box[1] <= 260 and region.box[3] >= 545


def test_a_distant_caption_is_not_swallowed():
    """
    The attach gap is what stops a stacked second table's caption being taken.
    """
    far_above = 300 - ORPHAN_ATTACH_GAP_PX - 60
    page = page_with_marks([(100, far_above, 700, far_above + 20),
                            (100, 300, 700, 500)])
    detections = [
        Detection(LABEL_TABLE_CAPTION, (100, far_above, 700, far_above + 20)),
        Detection(LABEL_TABLE, (100, 300, 700, 500)),
    ]

    regions = locate_table_regions(detections, page, logger=QUIET)
    assert not regions[0].has_caption


def test_figure_caption_is_kept_separate_from_the_crop():
    """
    A figure's caption is NOT folded in: the picture is saved, the caption read.
    """
    page = page_with_marks([(150, 200, 650, 500), (150, 530, 650, 560)])
    detections = [
        Detection(LABEL_IMAGE, (150, 200, 650, 500)),
        Detection(LABEL_IMAGE_CAPTION, (150, 530, 650, 560)),
    ]

    regions = locate_figure_regions(detections, page, logger=QUIET)

    assert len(regions) == 1
    assert regions[0].caption_box is not None
    # The image crop must stop short of the caption.
    assert regions[0].image_box[3] < 530


def test_figure_regions_come_back_in_reading_order():
    page = page_with_marks([(150, 600, 650, 800), (150, 100, 650, 300)])
    detections = [
        Detection(LABEL_IMAGE, (150, 600, 650, 800)),
        Detection(LABEL_IMAGE, (150, 100, 650, 300)),
    ]

    regions = locate_figure_regions(detections, page, logger=QUIET)
    assert regions[0].image_box[1] < regions[1].image_box[1]


# ---------------------------------------------------------------------------
# figure_extractor — saving and anchoring
# ---------------------------------------------------------------------------

def test_figures_are_saved_with_captions_and_portable_paths(tmp_path):
    page = page_with_marks([(150, 200, 650, 500), (150, 530, 650, 560)])
    detections = [
        Detection(LABEL_IMAGE, (150, 200, 650, 500)),
        Detection(LABEL_IMAGE_CAPTION, (150, 530, 650, 560)),
    ]
    out_dir = tmp_path / "figures" / "doc"

    assets = extract_and_save_figures(
        page, detections, out_dir, page_number=4,
        read_caption=lambda box: "Fig. 1. A caption.",
        path_root=tmp_path, logger=QUIET,
    )

    assert len(assets) == 1
    asset = assets[0]
    assert asset.figure_id == "p4_fig1"
    assert asset.caption == "Fig. 1. A caption."
    assert asset.page == 4
    # Portable, forward-slashed, relative to the run root — not an absolute path.
    assert asset.image_path == "figures/doc/p4_fig1.png"
    assert (out_dir / "p4_fig1.png").is_file()


def test_a_failing_caption_reader_still_saves_the_image(tmp_path):
    """A missing caption degrades the RAG entry; it must not lose the figure."""
    page = page_with_marks([(150, 200, 650, 500)])
    detections = [Detection(LABEL_IMAGE, (150, 200, 650, 500))]

    def broken_reader(box):
        raise RuntimeError("text layer unavailable")

    assets = extract_and_save_figures(page, detections, tmp_path, 2,
                                      read_caption=broken_reader, logger=QUIET)

    assert len(assets) == 1
    assert assets[0].caption == ""
    assert (tmp_path / "p2_fig1.png").is_file()


def test_anchors_link_image_and_caption_into_the_page_text():
    assets = [FigureAsset("p2_fig1", "figures/doc/p2_fig1.png",
                          "Fig. 1.  Multi-line\ncaption.", 2, (0, 0, 10, 10))]

    text = append_figure_anchors("Page prose.", assets)

    assert "Page prose." in text
    assert "[[FIGURE p2_fig1 |" in text
    # Caption whitespace is collapsed so the anchor stays on one line.
    assert "Fig. 1. Multi-line caption." in text


def test_anchors_are_a_no_op_without_figures():
    assert append_figure_anchors("Just prose.", []) == "Just prose."


def test_safe_doc_id_sanitises_titles():
    assert safe_doc_id("An NMR method: water/compartments") == "An_NMR_method_water_compartments"
    assert safe_doc_id("???") == "document"


# ---------------------------------------------------------------------------
# End to end, against the real detector and the parity baseline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    from src.processing.layout.onnx_detector import OnnxLayoutDetector
    return OnnxLayoutDetector(MODEL_PATH)


@needs_model
def test_real_detections_drive_figure_extraction(detector, tmp_path):
    """
    The full chain on the benchmark's page 2: detect, locate, crop, save.

    The saved crop must match the baseline's known-good figure box, which is
    what proves the ported consumers still agree with V5's output.
    """
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected_bbox = baseline["pages"][1]["metadata"]["figures"][0]["bbox"]

    with get_backend().open(BENCHMARK_PDF) as doc:
        page_image = doc.page(1).render(dpi=BASELINE_DPI)

    assets = extract_and_save_figures(
        page_image, detector.detect(page_image), tmp_path, page_number=2,
        read_caption=lambda box: "caption", path_root=tmp_path, logger=QUIET,
    )

    assert len(assets) == 1, "page 2 carries exactly one figure in the baseline"
    saved = Image.open(tmp_path / "p2_fig1.png")
    assert saved.width > 100 and saved.height > 100, "crop must not be degenerate"

    box = assets[0].bbox
    assert Rect(*expected_bbox).iou(Rect(*box)) >= IOU_GATE

    # Agreement is not exact and does not need to be: the two boxes differ only
    # in blank margin. What must hold is that no figure INK falls outside our
    # crop — that would be silently lost picture. (Measured: the baseline's box
    # runs ~20px lower than ours, and that band is entirely white.)
    ink = build_ink_mask(page_image)
    for y0, y1 in [(expected_bbox[1], box[1]), (box[3], expected_bbox[3])]:
        if y1 > y0:
            outside = ink[y0:y1, expected_bbox[0]:expected_bbox[2]]
            assert int(outside.sum()) == 0, f"figure ink lost in band y={y0}-{y1}"


@needs_model
def test_real_detections_drive_table_repair(detector):
    """
    Page 4 carries the two tables the parity harness checks.

    Both must be located and re-read, since a missed table is lost content.
    """
    with get_backend().open(BENCHMARK_PDF) as doc:
        page_image = doc.page(3).render(dpi=BASELINE_DPI)

    crops_read = []

    def fake_read(crop):
        crops_read.append(crop.size)
        return f"<table>grid {len(crops_read)}</table>"

    result = repair_page_tables(page_image, "Prose from the full-page read.",
                                detector.detect(page_image), fake_read, logger=QUIET)

    assert result.tables_recovered == 2
    assert len(result.region_boxes) == 2
    assert all(w > 50 and h > 50 for w, h in crops_read), "crops must be real regions"
    assert "Prose from the full-page read." in result.text
