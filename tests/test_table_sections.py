"""
Table sections: a table crop that holds two printed sub-tables.

Measured defect (eval/TABLE_SECTIONS_GATE.md): t12's crop holds "At 20 °C" and
"At 25 °C"; the reader stopped after the first and 64 of 82 values were lost.
These tests build synthetic crops, so they need no model and no PDF.
"""
from PIL import Image, ImageDraw

from src.processing.utils.table_sections import (
    find_section_blocks,
    read_table_with_sections,
)

LINE = 30      # pixel pitch of one text line
INK = 12       # ink height of one text line


def _crop(*blocks, gap=60, width=400):
    """A white crop with text-like ink bars: `blocks` = line counts, separated by `gap`."""
    height = sum(n * LINE for n in blocks) + gap * (len(blocks) - 1) + 20
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = 10
    for i, lines in enumerate(blocks):
        for _ in range(lines):
            draw.rectangle([20, y, width - 20, y + INK], fill="black")
            y += LINE
        if i < len(blocks) - 1:
            y += gap
    return image


def _table(*cells):
    return "<table>" + "".join(f"<tr><td>{c}</td></tr>" for c in cells) + "</table>"


class TestFindSectionBlocks:
    def test_one_block_is_not_split(self):
        assert len(find_section_blocks(_crop(12))) == 1

    def test_two_table_sized_blocks_are_found(self):
        assert len(find_section_blocks(_crop(8, 10))) == 2

    def test_a_short_caption_above_joins_the_table(self):
        # a 2-line caption, a gap, then the table: one block, nothing to read twice
        assert len(find_section_blocks(_crop(2, 10))) == 1

    def test_a_short_footnote_below_joins_the_table(self):
        assert len(find_section_blocks(_crop(10, 2))) == 1

    def test_a_heading_joins_the_block_below_it(self):
        # caption, sub-table A, heading, sub-table B -> two blocks
        blocks = find_section_blocks(_crop(2, 8, 1, 10))
        assert len(blocks) == 2

    def test_ordinary_row_spacing_never_splits(self):
        assert len(find_section_blocks(_crop(6, 6, gap=LINE - INK + 2))) == 1


class TestReadTableWithSections:
    def test_rows_the_whole_read_missed_are_appended(self):
        image = _crop(8, 10)
        whole = _table("a1", "a2")
        merged = read_table_with_sections(image, whole, lambda crop: _table("b1", "b2", "b3"))
        assert merged == "<table>" + "".join(
            f"<tr><td>{c}</td></tr>" for c in ("a1", "a2", "b1", "b2", "b3")) + "</table>"

    def test_a_section_that_repeats_the_whole_read_adds_nothing(self):
        image = _crop(8, 10)
        whole = _table("a1", "b1", "b2", "b3")
        assert read_table_with_sections(image, whole, lambda crop: _table("b1", "b2", "x")) == whole

    def test_the_same_rows_written_differently_are_not_added_again(self):
        """t12, first GUI run: the 20 C block came back in LaTeX and was duplicated."""
        image = _crop(8, 10)
        whole = _table("BaI₂·2H₂O", "44.0", "CaSO₄·2H₂O", "98.0")
        section = _table("$BaI_{2}·2H_{2}O", "$44.0$", "CaSO_{4}·2H_{2}O", "98.0")
        assert read_table_with_sections(image, whole, lambda crop: section) == whole

    def test_a_single_section_crop_is_never_read_again(self):
        calls = []
        whole = _table("a1")
        result = read_table_with_sections(_crop(12), whole, lambda crop: calls.append(1) or "")
        assert result == whole and calls == []

    def test_an_empty_section_read_changes_nothing(self):
        whole = _table("a1", "a2")
        assert read_table_with_sections(_crop(8, 10), whole, lambda crop: "") == whole

    def test_rows_land_inside_tbody(self):
        whole = "<table><tbody><tr><td>a1</td></tr></tbody></table>"
        merged = read_table_with_sections(_crop(8, 10), whole, lambda crop: _table("b1"))
        assert merged == "<table><tbody><tr><td>a1</td></tr><tr><td>b1</td></tr></tbody></table>"
