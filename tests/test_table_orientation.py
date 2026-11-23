#!/usr/bin/env python3
"""
Tests for the tight, upright table read (`utils/table_orientation.py`).

For biology researchers: the table reader is given the table body alone, the
right way up, because that is what fixed 4 of the 7 seriously wrong corpus
tables (eval/second_reader/GATE.md, "Finding 0"). These tests pin the three
decisions that make that happen:

  1. **Tight**: the model reads the table body, not the saved crop with its
     caption, while the SAVED crop still carries the caption.
  2. **Sideways**: a table is turned only when most of its words run
     vertically, and never on a page with no text layer.
  3. **Which way**: text reading up the page is turned clockwise, text reading
     down is turned anticlockwise.

The last group runs the real PDF backend on the real sideways table (t26),
because a fake can only echo the direction a test declares.
"""

import logging
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from src.processing.layout import Detection
from src.processing.layout.base import LABEL_TABLE, LABEL_TABLE_CAPTION
from src.processing.layout.regions import locate_table_regions
from src.processing.pdf import Rect, get_backend
from src.processing.pdf.base import vertical_direction_from_steps
from src.processing.strategies.research_article.tables import extract_page_tables
from src.processing.utils.table_orientation import (
    table_read_image,
    table_turn_degrees,
)
from src.processing.utils.table_repair import repair_page_tables
from tests.fakes import FakePage, rotated_word, word_row
from tests.local_samples import CORPUS as LOCAL_CORPUS

QUIET = logging.getLogger("test")
QUIET.addHandler(logging.NullHandler())

#: The test page image is 800 x 1000 px; FakePage is 612 x 792 pt.
IMAGE_SIZE = (800, 1000)
TABLE_PX = (100, 300, 700, 500)
CAPTION_PX = (100, 240, 700, 270)

CORPUS_DIR = LOCAL_CORPUS
TENDON_PDF = CORPUS_DIR / "Tendon assembly 2021.pdf"
NMR_PDF = CORPUS_DIR / ("An NMR method to characterize multiple water "
                        "compartments on mammalian collagen.pdf")


def page_image_with(boxes):
    """A white page with black rectangles: synthetic ink for the snapping."""
    image = Image.new("RGB", IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, fill="black")
    return image


def table_rect_in_points() -> Rect:
    """TABLE_PX on the FakePage, in PDF points."""
    sx, sy = 612 / IMAGE_SIZE[0], 792 / IMAGE_SIZE[1]
    x0, y0, x1, y1 = TABLE_PX
    return Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def sideways_page(count: int = 10, direction: str = "") -> FakePage:
    """A page whose table holds `count` words all printed vertically."""
    rect = table_rect_in_points()
    words = [rotated_word("Collagen", x=rect.x0 + 10 + 12 * i, y=rect.y0 + 10)
             for i in range(count)]
    return FakePage(words=words, vertical_direction=direction)


def upright_page() -> FakePage:
    """A page whose table holds ordinary horizontal words."""
    rect = table_rect_in_points()
    words = []
    for row in range(5):
        words += word_row(["Tendon", "Fibril", "Diameter"], x=rect.x0 + 10,
                          y=rect.y0 + 10 + 20 * row)
    return FakePage(words=words)


# ---------------------------------------------------------------------------
# 1. Tight: the model reads the body, the saved crop keeps the caption
# ---------------------------------------------------------------------------

class TestTightReadBox:

    def test_the_read_box_leaves_out_the_caption_the_saved_box_keeps(self):
        image = page_image_with([TABLE_PX, CAPTION_PX])
        regions = locate_table_regions(
            [Detection(LABEL_TABLE, TABLE_PX), Detection(LABEL_TABLE_CAPTION, CAPTION_PX)],
            image, logger=QUIET)
        region = regions[0]
        assert region.box[1] <= CAPTION_PX[1], "saved crop must still hold the caption"
        assert region.reading_box[1] > CAPTION_PX[3], "the read must start below the caption"

    def test_a_hand_built_region_reads_its_own_box(self):
        from src.processing.layout.regions import TableRegion
        region = TableRegion(box=(1, 2, 3, 4))
        assert region.reading_box == (1, 2, 3, 4)

    def test_the_tagging_path_reads_the_tight_body(self, tmp_path):
        image = page_image_with([TABLE_PX, CAPTION_PX])
        sizes = []

        def reader(crop):
            sizes.append(crop.size)
            return "<table><tr><td>1</td></tr></table>"

        assets = extract_page_tables(
            FakePage(), image,
            [Detection(LABEL_TABLE, TABLE_PX), Detection(LABEL_TABLE_CAPTION, CAPTION_PX)],
            pdf_path=tmp_path / "p.pdf", tables_base_dir=tmp_path / "tables",
            page_idx=0, read_table_crop=reader, logger=QUIET)

        saved_height = assets[0].bbox[3] - assets[0].bbox[1]
        assert sizes[0][1] < saved_height, "the model must not be shown the caption"
        saved = Image.open(tmp_path / "tables" / "p" / "p1_tab1.png")
        assert saved.size[1] == saved_height, "the saved PNG is unchanged"

    def test_the_repair_path_reads_the_tight_body_and_keys_by_the_saved_box(self):
        image = page_image_with([TABLE_PX, CAPTION_PX])
        sizes = []
        detections = [Detection(LABEL_TABLE, TABLE_PX),
                      Detection(LABEL_TABLE_CAPTION, CAPTION_PX)]
        result = repair_page_tables(
            image, "Prose.", detections,
            read_table_crop=lambda crop: sizes.append(crop.size) or "| a |",
            logger=QUIET, page=FakePage())
        region = locate_table_regions(detections, image, logger=QUIET)[0]
        assert result.tables[0].box == region.box
        assert sizes[0][1] == region.reading_box[3] - region.reading_box[1]


# ---------------------------------------------------------------------------
# 2. Sideways: turned only on clear evidence
# ---------------------------------------------------------------------------

class TestSidewaysDecision:

    def test_a_table_of_vertical_words_is_turned_and_its_crop_swaps_sides(self):
        image = page_image_with([TABLE_PX])
        crop = table_read_image(image, TABLE_PX, sideways_page())
        width, height = TABLE_PX[2] - TABLE_PX[0], TABLE_PX[3] - TABLE_PX[1]
        assert crop.size == (height, width)

    def test_an_ordinary_table_is_not_turned(self):
        assert table_turn_degrees(upright_page(), TABLE_PX, IMAGE_SIZE) == 0

    def test_a_few_rotated_headers_do_not_turn_a_table(self):
        page = sideways_page(count=3)
        page._words += upright_page().words()
        assert table_turn_degrees(page, TABLE_PX, IMAGE_SIZE) == 0

    def test_too_few_words_to_judge_leaves_it_as_printed(self):
        assert table_turn_degrees(sideways_page(count=4), TABLE_PX, IMAGE_SIZE) == 0

    def test_a_scan_with_no_text_layer_is_never_turned(self):
        assert table_turn_degrees(FakePage(), TABLE_PX, IMAGE_SIZE) == 0

    def test_without_a_page_it_is_read_as_printed(self):
        assert table_turn_degrees(None, TABLE_PX, IMAGE_SIZE) == 0

    def test_the_tagging_path_turns_a_sideways_table(self, tmp_path):
        sizes = []
        extract_page_tables(
            sideways_page(), page_image_with([TABLE_PX]),
            [Detection(LABEL_TABLE, TABLE_PX)],
            pdf_path=tmp_path / "p.pdf", tables_base_dir=tmp_path / "tables",
            page_idx=0, read_table_crop=lambda c: sizes.append(c.size) or "<table/>",
            logger=QUIET)
        assert sizes[0][0] < sizes[0][1], "a wide table, read upright, comes out tall"


# ---------------------------------------------------------------------------
# 3. Which way
# ---------------------------------------------------------------------------

class TestTurnDirection:

    def _marked_corner_after_turn(self, direction: str) -> str:
        """Where the table's top-left corner ends up after the turn."""
        image = page_image_with([])
        ImageDraw.Draw(image).rectangle((100, 300, 130, 330), fill="black")
        crop = table_read_image(image, TABLE_PX, sideways_page(direction=direction))
        w, h = crop.size
        corners = {"top-left": (5, 5), "top-right": (w - 6, 5),
                   "bottom-left": (5, h - 6), "bottom-right": (w - 6, h - 6)}
        return next(name for name, xy in corners.items() if crop.getpixel(xy) == (0, 0, 0))

    def test_text_reading_up_is_turned_clockwise(self):
        assert self._marked_corner_after_turn("up") == "top-right"

    def test_text_reading_down_is_turned_anticlockwise(self):
        assert self._marked_corner_after_turn("down") == "bottom-left"

    def test_an_unknown_direction_is_taken_as_up(self):
        assert self._marked_corner_after_turn("") == "top-right"

    @pytest.mark.parametrize("steps, expected", [
        ([(0.1, -6.0)] * 5, "up"),
        ([(0.1, 6.0)] * 5, "down"),
        ([(6.0, 0.1)] * 5, ""),          # ordinary horizontal text
        ([(0.0, -6.0), (0.0, 6.0)], ""),  # a tie decides nothing
        ([], ""),
    ])
    def test_the_vote_between_letters(self, steps, expected):
        assert vertical_direction_from_steps(steps) == expected


# ---------------------------------------------------------------------------
# 4. The real backend on the real tables
# ---------------------------------------------------------------------------

def _direction_on(pdf: Path, page_number: int, box_px_at_300dpi) -> str:
    with get_backend().open(pdf) as document:
        page = document.page(page_number - 1)
        return page.vertical_text_direction(Rect(*(v * 72 / 300 for v in box_px_at_300dpi)))


@pytest.mark.skipif(not TENDON_PDF.is_file(), reason="corpus PDF unavailable")
def test_the_sideways_corpus_table_t26_reads_up():
    """t26 was turned clockwise by hand in the eval, and read right that way."""
    assert _direction_on(TENDON_PDF, 271, (363, 275, 1845, 2781)) == "up"


@pytest.mark.skipif(not NMR_PDF.is_file(), reason="benchmark PDF unavailable")
def test_an_upright_benchmark_table_has_no_vertical_text():
    assert _direction_on(NMR_PDF, 4, (1303, 436, 2376, 815)) == ""
