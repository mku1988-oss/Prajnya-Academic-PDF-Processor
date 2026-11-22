import json, logging
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from src.processing.layout import Detection
from src.processing.layout.base import LABEL_TABLE, LABEL_TABLE_CAPTION, LABEL_IMAGE
from src.processing.strategies.research_article.tables import (
    TABLE_ANCHOR_MARKER,
    TableAsset,
    append_table_anchors,
    extract_page_tables,
    write_table_manifest,
)
from src.processing.utils.table_repair import RecoveredTable, repair_page_tables
from tests.fakes import FakePage

QUIET = logging.getLogger("test")
QUIET.addHandler(logging.NullHandler())


def page_with_marks(boxes, width=800, height=1000):
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    for box in boxes:
        draw.rectangle(box, fill="black")
    return page


class TestRecoveredTablePairing:
    """
    The repair path already reads each table crop. Pairing that transcription
    with its box is what lets a table become a retrievable object without
    paying for a second model call.
    """

    def test_recovered_tables_pair_box_with_grid(self):
        page = page_with_marks([(100, 300, 700, 500)])
        result = repair_page_tables(
            page, "Prose.", [Detection(LABEL_TABLE, (100, 300, 700, 500))],
            read_table_crop=lambda crop: "| Zone | Rate |",
            logger=QUIET)
        assert len(result.tables) == 1
        assert result.tables[0].grid == "| Zone | Rate |"
        assert len(result.tables[0].box) == 4
        assert result.tables_recovered == len(result.tables)

    def test_empty_reread_yields_no_recovered_table(self):
        page = page_with_marks([(100, 300, 700, 500)])
        result = repair_page_tables(
            page, "Prose.", [Detection(LABEL_TABLE, (100, 300, 700, 500))],
            read_table_crop=lambda crop: "   ", logger=QUIET)
        assert result.tables == []
        assert result.tables_recovered == 0


class TestTableTagging:
    """Tables become retrievable objects: id, crop, caption, page, bbox."""

    def test_table_is_cropped_saved_and_tagged(self, tmp_path):
        page_image = page_with_marks([(100, 300, 700, 500)])
        assets = extract_page_tables(
            FakePage(), page_image, [Detection(LABEL_TABLE, (100, 300, 700, 500))],
            pdf_path=tmp_path / "paper.pdf", tables_base_dir=tmp_path / "tables",
            page_idx=3, logger=QUIET)

        assert len(assets) == 1
        asset = assets[0]
        assert asset.table_id == "p4_tab1"      # 1-indexed page
        assert asset.page == 4
        assert len(asset.bbox) == 4
        saved = tmp_path / "tables" / "paper" / "p4_tab1.png"
        assert saved.is_file(), "the crop must actually be written"

    def test_grid_is_carried_through_from_repair(self, tmp_path):
        page_image = page_with_marks([(100, 300, 700, 500)])
        detections = [Detection(LABEL_TABLE, (100, 300, 700, 500))]
        # Use the boxes repair itself would produce, so the lookup pairs up.
        repaired = repair_page_tables(
            page_image, "Prose.", detections,
            read_table_crop=lambda crop: "| Zone | Rate |", logger=QUIET)

        assets = extract_page_tables(
            FakePage(), page_image, detections, pdf_path=tmp_path / "p.pdf",
            tables_base_dir=tmp_path / "tables", page_idx=3,
            recovered=repaired.tables, logger=QUIET)
        assert assets[0].grid == "| Zone | Rate |"

    def test_a_table_repair_never_read_is_read_from_its_crop(self, tmp_path):
        """REPLACES test_grid_is_none_when_repair_never_ran (2026-09-11).

        That test pinned "no second model call just to fill a field the page
        text already has". Measured downstream, the field was the ONLY place
        retrieval reads a table's content from: 14 of 31 corpus tables reached
        no index. So a table the repair path did not transcribe is read from
        its own crop.
        """
        page_image = page_with_marks([(100, 300, 700, 500)])
        crops_read = []

        def reader(crop):
            crops_read.append(crop.size)
            return "<table><tr><td>Cardiac-valvular</td><td>AR</td></tr></table>"

        assets = extract_page_tables(
            FakePage(), page_image, [Detection(LABEL_TABLE, (100, 300, 700, 500))],
            pdf_path=tmp_path / "p.pdf", tables_base_dir=tmp_path / "tables",
            page_idx=0, recovered=None, read_table_crop=reader, logger=QUIET)

        assert assets[0].grid.startswith("<table>")
        assert len(crops_read) == 1

    def test_a_repair_transcription_is_not_read_twice(self, tmp_path):
        page_image = page_with_marks([(100, 300, 700, 500)])
        detections = [Detection(LABEL_TABLE, (100, 300, 700, 500))]
        repaired = repair_page_tables(
            page_image, "Prose.", detections,
            read_table_crop=lambda crop: "| Zone | Rate |", logger=QUIET)

        assets = extract_page_tables(
            FakePage(), page_image, detections, pdf_path=tmp_path / "p.pdf",
            tables_base_dir=tmp_path / "tables", page_idx=3,
            recovered=repaired.tables, read_table_crop=_must_not_be_called,
            logger=QUIET)

        assert assets[0].grid == "| Zone | Rate |"

    def test_a_second_printed_section_is_read_even_on_the_repair_path(self, tmp_path):
        """t12's grid came from the repair path, and lost its second sub-table."""
        lines = [(120, y, 680, y + 12) for y in range(220, 460, 30)]      # section A, 8 lines
        lines += [(120, y, 680, y + 12) for y in range(540, 840, 30)]     # section B, 10 lines
        page_image = page_with_marks(lines)
        detections = [Detection(LABEL_TABLE, (100, 200, 700, 860))]
        repaired = repair_page_tables(
            page_image, "Prose.", detections,
            read_table_crop=lambda crop: "<table><tr><td>A1</td></tr></table>", logger=QUIET)

        assets = extract_page_tables(
            FakePage(), page_image, detections, pdf_path=tmp_path / "p.pdf",
            tables_base_dir=tmp_path / "tables", page_idx=3, recovered=repaired.tables,
            read_table_crop=lambda crop: "<table><tr><td>B1</td></tr></table>", logger=QUIET)

        assert assets[0].grid == "<table><tr><td>A1</td></tr><tr><td>B1</td></tr></table>"

    def test_without_a_reader_the_grid_stays_none(self, tmp_path):
        """A caller passing no reader keeps the previous behaviour."""
        page_image = page_with_marks([(100, 300, 700, 500)])
        assets = extract_page_tables(
            FakePage(), page_image, [Detection(LABEL_TABLE, (100, 300, 700, 500))],
            pdf_path=tmp_path / "p.pdf", tables_base_dir=tmp_path / "tables",
            page_idx=0, recovered=None, logger=QUIET)
        assert assets[0].grid is None

    @pytest.mark.parametrize("reader", [lambda crop: "   ", "failing"])
    def test_a_failed_or_empty_read_keeps_the_table(self, tmp_path, reader):
        """Losing a transcription must never lose the table itself."""
        if reader == "failing":
            reader = _failing_reader
        page_image = page_with_marks([(100, 300, 700, 500)])
        assets = extract_page_tables(
            FakePage(), page_image, [Detection(LABEL_TABLE, (100, 300, 700, 500))],
            pdf_path=tmp_path / "p.pdf", tables_base_dir=tmp_path / "tables",
            page_idx=0, read_table_crop=reader, logger=QUIET)
        assert len(assets) == 1
        assert assets[0].grid is None

    def test_no_detections_yields_no_tables(self, tmp_path):
        assets = extract_page_tables(
            FakePage(), page_with_marks([]), None, pdf_path=tmp_path / "p.pdf",
            tables_base_dir=tmp_path / "tables", page_idx=0, logger=QUIET)
        assert assets == []

    def test_a_page_of_figures_yields_no_tables(self, tmp_path):
        assets = extract_page_tables(
            FakePage(), page_with_marks([(100, 100, 400, 400)]),
            [Detection(LABEL_IMAGE, (100, 100, 400, 400))],
            pdf_path=tmp_path / "p.pdf", tables_base_dir=tmp_path / "tables",
            page_idx=0, logger=QUIET)
        assert assets == []

    def test_ids_are_unique_across_multiple_tables(self, tmp_path):
        page_image = page_with_marks([(100, 200, 700, 350), (100, 600, 700, 750)])
        assets = extract_page_tables(
            FakePage(), page_image,
            [Detection(LABEL_TABLE, (100, 200, 700, 350)),
             Detection(LABEL_TABLE, (100, 600, 700, 750))],
            pdf_path=tmp_path / "p.pdf", tables_base_dir=tmp_path / "tables",
            page_idx=3, logger=QUIET)
        assert [a.table_id for a in assets] == ["p4_tab1", "p4_tab2"]


class TestTableAnchorsAndManifest:
    def test_anchor_links_crop_back_into_the_page_text(self):
        asset = TableAsset("p4_tab1", "tables/doc/p4_tab1.png", "Table 1. Rates",
                           4, (1, 2, 3, 4))
        text = append_table_anchors("Page prose.", [asset])
        assert TABLE_ANCHOR_MARKER in text
        assert "[[TABLE p4_tab1 | Table 1. Rates | tables/doc/p4_tab1.png]]" in text
        assert "Page prose." in text

    def test_no_tables_leaves_text_untouched(self):
        assert append_table_anchors("Page prose.", []) == "Page prose."

    def test_manifest_mirrors_the_figure_manifest_shape(self, tmp_path):
        asset = TableAsset("p4_tab1", "tables/doc/p4_tab1.png", "Table 1", 4,
                           (1, 2, 3, 4), grid="| a |")
        path = write_table_manifest(tmp_path / "doc.pdf", tmp_path / "tables",
                                    [asset.to_dict()], logger=QUIET)
        assert path is not None
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["table_count"] == 1
        assert data["tables"][0]["table_id"] == "p4_tab1"
        assert data["tables"][0]["grid"] == "| a |"

    def test_no_manifest_when_document_has_no_tables(self, tmp_path):
        assert write_table_manifest(tmp_path / "d.pdf", tmp_path / "t", [],
                                    logger=QUIET) is None


# --- helpers (after the tests, per CLAUDE.md) --------------------------------

def _must_not_be_called(crop):
    raise AssertionError("the repair path already read this table")


def _failing_reader(crop):
    raise RuntimeError("llama-server is not answering")
