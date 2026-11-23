"""
Tests for pairing a crop with a legend printed BESIDE it
(figure_label_pass._side_caption, ledger T28).

For biology researchers: a hand-built page (tests/fakes.py) shaped like a
J Cell Biol page: the figure on the right, its legend in a column on the
left, the legend's first line level with the figure's top.
"""

from src.processing.pdf.base import Rect, Word
from src.processing.utils.figure_label_pass import (
    assign_labels_on_page, build_text_lines, find_label_lines, label_words)
from tests.fakes import FakePage

SIZE = (612.0, 792.0)

#: The figure, right of the legend column, top at y=60.
FIGURE = (273.0, 60.0, 562.0, 244.0)


def word(text, x0, y0, x1, y1):
    return Word(text=text, rect=Rect(x0, y0, x1, y1), rotated=False, baseline=y1 - 2)


def line(text, x0, y0, width_per_word=30.0, height=8.0):
    """A printed line, one word box per word, left to right."""
    out, x = [], x0
    for token in text.split():
        out.append(word(token, x, y0, x + width_per_word - 4, y0 + height))
        x += width_per_word
    return out


def legend(x0=64, top=62, label="Figure 1."):
    return (line(f"{label} Different distributions of", x0, top)
            + line("14-3-3 and its ligands in cells", x0, top + 10)
            + line("stained for the protein here", x0, top + 20))


def assign(words, boxes=(FIGURE,), **kwargs):
    page = FakePage(words=words, size=SIZE)
    lines = build_text_lines(label_words(page), SIZE[0])
    return assign_labels_on_page(list(boxes), find_label_lines(lines), SIZE[1],
                                 text_lines=lines, **kwargs)


class TestSideCaption:
    def test_legend_left_of_the_figure_pairs(self):
        result = assign(legend())
        assert result[0].key == "figure 1" and result[0].beside
        assert result[0].caption.startswith("Figure 1. Different")
        assert "stained for the protein" in result[0].caption

    def test_legend_right_of_the_figure_pairs(self):
        figure = (48.0, 60.0, 383.0, 310.0)
        result = assign(legend(x0=404), boxes=(figure,))
        assert result[0].key == "figure 1" and result[0].beside

    def test_switched_off_leaves_the_crop_unlabelled(self):
        assert assign(legend(), read_side_captions=False)[0].key is None

    def test_a_legend_not_level_with_the_top_is_not_taken(self):
        # 100 pt below the figure's top: beyond CAPTION_OVERLAP_TOLERANCE.
        # (The older last-resort orphan adoption may still take it; the side
        # rule must not, and must not change that outcome.)
        words = legend(top=160)
        result = assign(words)[0]
        assert not result.beside
        assert result == assign(words, read_side_captions=False)[0]

    def test_text_between_legend_and_figure_blocks_the_pairing(self):
        between = line("prose", 230, 62, width_per_word=30.0)
        assert assign(legend(x0=20) + between)[0].key is None

    def test_a_caption_below_still_wins(self):
        below = line("Figure 2. Below the figure here", 280, 250)
        result = assign(legend() + below)
        assert result[0].key == "figure 2" and not result[0].beside

    def test_a_mid_paragraph_mention_is_not_a_legend(self):
        # A body line runs into the "Figure 1 ..." line from above at the same
        # margin, so it does not begin a block.
        words = (line("cells were imaged as described in Methods", 64, 52)
                 + line("Figure 1 shows the distribution of", 64, 62))
        assert assign(words)[0].key is None
