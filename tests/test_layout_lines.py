"""Tests for utils/layout_lines.py: printed lines rebuilt from characters.

For biology researchers: each test prints a tiny page from literal
characters and checks the lines come back as a reader would see them: one per
column, in reading order, with words, accents and ligatures intact.
"""
from typing import List

from src.processing.pdf.base import PrintedChar, Rect
from src.processing.utils.layout_lines import document_layout_lines, page_layout_lines

CHAR_WIDTH = 5.0
HEIGHT = 10.0

_WORDS = ["collagen", "a", "fibril", "of", "triple", "helix", "the", "tendon", "is", "hydrated", "by", "water"]


def prose(n: int, words: int = 5) -> str:
    """A row of varied words, so word spaces do not line up from row to row (as in real print)."""
    return " ".join(_WORDS[(n * 5 + k * 7) % len(_WORDS)] for k in range(words))


def printed(text: str, x: float, baseline: float, *, width: float = CHAR_WIDTH,
            height: float = HEIGHT) -> List[PrintedChar]:
    """Characters for `text` set left to right from x, a space marking each word boundary."""
    chars, space = [], False
    for character in text:
        if character == " ":
            space, x = True, x + width
            continue
        rect = Rect(x, baseline - 0.8 * height, x + width, baseline + 0.2 * height)
        chars.append(PrintedChar(text=character, rect=rect, baseline=baseline, space_before=space))
        space, x = False, x + width
    return chars


def column(x: float, first_baseline: float, rows: List[str], pitch: float = 12.0) -> List[PrintedChar]:
    chars: List[PrintedChar] = []
    for index, row in enumerate(rows):
        chars += printed(row, x, first_baseline + index * pitch)
    return chars


ROWS = [prose(n) for n in range(12)]


class TestColumns:
    def test_two_columns_are_read_one_after_the_other(self):
        # The right column's baselines are offset by 3 pt, as in real journals.
        left, right = [prose(n) for n in range(12)], [prose(n + 3) for n in range(12)]
        lines = page_layout_lines(column(40, 100, left) + column(320, 103, right), 1)
        assert [line.text for line in lines] == left + right
        assert {line.column for line in lines[:12]} == {0} and {line.column for line in lines[12:]} == {1}

    def test_rows_written_across_both_columns_are_still_cut_at_the_gutter(self):
        # Some publishers write a page row by row: left row, right row, next left row...
        chars: List[PrintedChar] = []
        for n in range(12):
            chars += printed(prose(n, 4), 40, 100 + 12 * n)
            chars += printed(prose(n + 5, 4), 230, 100 + 12 * n)
        lines = page_layout_lines(chars, 1)
        assert lines[0].text == prose(0, 4)
        assert lines[12].text == prose(5, 4)

    def test_a_column_of_entry_numbers_is_not_a_column(self):
        chars: List[PrintedChar] = []
        for n in range(12):
            chars += printed(str(75 + n), 40, 100 + 12 * n)
            chars += printed(prose(n, 6), 58, 100 + 12 * n)
        assert page_layout_lines(chars, 1)[0].text == "75 " + prose(0, 6)


class TestCharacters:
    def test_a_descender_stays_on_its_line(self):
        chars = printed("ab", 40, 100) + [PrintedChar(
            text="g", rect=Rect(50, 93, 55, 104), baseline=100)] + printed("c", 55, 100)
        assert [line.text for line in page_layout_lines(chars, 1)] == ["abgc"]

    def test_a_separately_printed_accent_joins_its_letter(self):
        chars = printed("Ne", 40, 100)
        chars.append(PrintedChar(text="´", rect=Rect(46, 92, 49, 95), baseline=100))
        rest = printed("methy", 51, 100)
        rest[0] = PrintedChar(text="m", rect=rest[0].rect, baseline=100, space_before=True)
        assert page_layout_lines(chars + rest, 1)[0].text == "Némethy"

    def test_a_ligature_overlap_does_not_break_the_line(self):
        chars = printed("Knoepf", 40, 100) + printed("ler", 40 + 6 * CHAR_WIDTH - 3.6, 100)
        assert page_layout_lines(chars, 1)[0].text == "Knoepfler"

    def test_turned_characters_are_left_out(self):
        chars = printed("text", 40, 100) + [PrintedChar(
            text="D", rect=Rect(580, 50, 590, 55), baseline=55, turned=True)]
        assert [line.text for line in page_layout_lines(chars, 1)] == ["text"]


class TestFurniture:
    def test_a_running_header_on_every_page_is_removed(self):
        pages = [printed(f"Collagen triple helical structure {1000 + n}", 300, 40)
                 + column(40, 100, ROWS) for n in range(3)]
        lines = document_layout_lines(pages, [842.0] * 3)
        assert not any("Collagen triple" in line.text for line in lines)
        assert len(lines) == 3 * len(ROWS)

    def test_a_line_in_the_band_on_one_page_only_is_kept(self):
        pages = [printed("A title near the top", 300, 40) + column(40, 100, ROWS), column(40, 100, ROWS)]
        lines = document_layout_lines(pages, [842.0] * 2)
        assert any(line.text == "A title near the top" for line in lines)


def test_a_reference_tail_in_the_top_band_is_not_a_running_header():
    pages = [printed("2010;330:341\u20136.", 40, 40) + column(40, 100, ROWS),
             printed("2011;30:600\u201310.", 40, 40) + column(40, 100, ROWS)]
    lines = document_layout_lines(pages, [842.0] * 2)
    assert any(line.text.startswith("2010;330") for line in lines)


class TestFullWidthTextAboveColumns:
    """Cell Reports page 2: a full-width SUMMARY above a two-column
    INTRODUCTION. The two are different layouts (item 000a, fix 5)."""

    def test_full_width_rows_do_not_hide_the_gutter_below(self):
        wide = [" ".join([prose(n, 5), prose(n + 1, 5), prose(n + 2, 3)]) for n in range(16)]
        left, right = [prose(n, 4) for n in range(14)], [prose(n + 3, 4) for n in range(14)]
        chars = column(40, 100, wide) + column(40, 320, left) + column(320, 320, right)
        texts = [line.text for line in page_layout_lines(chars, 1)]
        assert texts == wide + left + right

    def test_a_few_rows_round_an_inset_do_not_split_the_column_above(self):
        """Frontiers: three short rows beside a small inset show a local
        third gutter; the two columns above and below stay one layout."""
        left = [prose(n) for n in range(24)]
        right_top = [prose(n + 3) for n in range(10)]
        right_wrap = [prose(n, 2) for n in range(2)]
        right_bottom = [prose(n + 5) for n in range(12)]
        chars = (column(40, 100, left) + column(320, 100, right_top)
                 + column(320, 220, right_wrap) + column(440, 220, right_wrap)
                 + column(320, 244, right_bottom))
        lines = page_layout_lines(chars, 1)
        assert [line.text for line in lines][:24] == left

    def test_one_line_whose_boxes_cross_the_gutter_does_not_split_the_column(self):
        """The 1960 Saturated Solutions scan, p6: the text layer's boxes of one
        left-column line run across the gutter. One row is not full-width
        text: the column is still read top to bottom, then the next."""
        left_top, left_bottom = [prose(n, 4) for n in range(14)], [prose(n + 2, 4) for n in range(14)]
        right = [prose(n + 4, 4) for n in range(29)]
        crossing = "x" * 60                                   # 300 points wide: 40 to 340
        chars = (column(40, 100, left_top) + printed(crossing, 40, 100 + 14 * 12)
                 + column(40, 100 + 15 * 12, left_bottom) + column(320, 100, right))
        texts = [line.text for line in page_layout_lines(chars, 1)]
        # The left column whole first (its crossing row in place), then the right.
        assert texts[:14] == left_top and texts[15:29] == left_bottom
