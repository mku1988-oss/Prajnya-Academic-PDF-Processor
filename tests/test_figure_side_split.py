"""
Tests for splitting one crop that holds two side-by-side figures
(utils/figure_side_split.py, ledger T27), and for reading a bare tag's
legend in the label pass.

For biology researchers: a hand-built page (tests/fakes.py) with two figure
stand-ins side by side, a short tag under each ("Fig. 3", "Fig. 4") and the
two legends below. The image is the page plus the renderer's 25-pixel white
border, at one pixel per point, so a point box is the pixel box shifted by 25.
"""

from PIL import Image, ImageDraw

from src.processing.layout.base import LABEL_IMAGE, LABEL_TABLE, Detection
from src.processing.pdf.base import Rect, Word
from src.processing.utils.figure_crop_fit import RENDER_PADDING_PX
from src.processing.utils.figure_label_pass import (
    assign_labels_on_page, build_text_lines, find_label_lines, label_words)
from src.processing.utils.figure_side_split import split_side_by_side_figures
from tests.fakes import FakePage

SIZE = (612.0, 792.0)
PAD = RENDER_PADDING_PX

#: Two figures side by side, 20 pt of white between them.
LEFT = (70, 240, 280, 470)
RIGHT = (300, 240, 480, 470)
BOTH = (70, 240, 480, 470)


def word(text, x0, y0, x1, y1):
    return Word(text=text, rect=Rect(x0, y0, x1, y1), rotated=False, baseline=y1 - 2)


def line(text, x0, y0, width_per_word=40.0, height=8.0):
    """A printed line, one word box per word, left to right."""
    out, x = [], x0
    for token in text.split():
        out.append(word(token, x, y0, x + width_per_word - 6, y0 + height))
        x += width_per_word
    return out


def px(box):
    """A point box as pixels in the padded render."""
    return tuple(int(v + PAD) for v in box)


TAGS = line("Fig. 3", 150, 476) + line("Fig. 4", 360, 476)
LEGENDS = (line("Fig. 3. An irradiated white distal blastema", 60, 500)
           + line("Fig. 4. A Y-shaped cartilage", 60, 530))


def blank():
    return Image.new("RGB", (int(SIZE[0]) + 2 * PAD, int(SIZE[1]) + 2 * PAD), "white")


def two_photos_with_gutter():
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.rectangle(px(LEFT), fill="black")
    draw.rectangle(px(RIGHT), fill="black")
    return image


def two_photos_with_seam():
    """Two gray plates butted together, a 2 pt light seam between them."""
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.rectangle(px(BOTH), fill=(200, 200, 200))
    draw.rectangle(px((289, 240, 290, 470)), fill=(240, 240, 240))
    return image


def one_solid_block():
    image = blank()
    ImageDraw.Draw(image).rectangle(px(BOTH), fill="black")
    return image


def text_lines(words):
    """Page lines as the label pass builds them from a page's words."""
    return build_text_lines(label_words(FakePage(words=words, size=SIZE)), SIZE[0])


def split(image, words=TAGS + LEGENDS, detections=None):
    page = FakePage(words=words, size=SIZE, number=4)
    detections = detections or [Detection(LABEL_IMAGE, px(BOTH), 0.8)]
    return split_side_by_side_figures(page, image, detections)


class TestSplit:
    def test_white_gutter_splits_into_two_figures(self):
        result = split(two_photos_with_gutter())
        assert len(result) == 2
        left, right = (d.box_px for d in result)
        assert left[0] == px(BOTH)[0] and right[2] == px(BOTH)[2]
        # The cut leaves the gutter out: left ends, right begins, inside it.
        assert px(LEFT)[2] < left[2] <= right[0] <= px(RIGHT)[0]

    def test_light_seam_between_butted_photos_splits(self):
        result = split(two_photos_with_seam())
        assert len(result) == 2
        cut = result[0].box_px[2]
        assert abs(cut - px((289, 0, 0, 0))[0]) <= 2

    def test_no_gutter_and_no_seam_declines(self):
        result = split(one_solid_block())
        assert [d.box_px for d in result] == [px(BOTH)]

    def test_a_single_tag_never_splits(self):
        result = split(two_photos_with_gutter(), words=line("Fig. 3", 150, 476) + LEGENDS)
        assert len(result) == 1

    def test_the_same_label_twice_never_splits(self):
        words = line("Fig. 3", 150, 476) + line("Fig. 3", 360, 476)
        assert len(split(two_photos_with_gutter(), words=words)) == 1

    def test_tags_on_different_rows_never_split(self):
        words = line("Fig. 3", 150, 476) + line("Fig. 4", 360, 520)
        assert len(split(two_photos_with_gutter(), words=words)) == 1

    def test_tags_far_below_the_crop_never_split(self):
        words = line("Fig. 3", 150, 700) + line("Fig. 4", 360, 700)
        assert len(split(two_photos_with_gutter(), words=words)) == 1

    def test_other_detections_pass_through_in_place(self):
        table = Detection(LABEL_TABLE, px((60, 600, 500, 700)), 0.9)
        result = split(two_photos_with_gutter(),
                       detections=[table, Detection(LABEL_IMAGE, px(BOTH), 0.8)])
        assert result[0] == table
        assert [d.label for d in result[1:]] == [LABEL_IMAGE, LABEL_IMAGE]


class TestBareTagLegend:
    def labels(self):
        lines = text_lines(TAGS + LEGENDS)
        return lines, find_label_lines(lines)

    def test_bare_tag_reads_its_own_legend(self):
        lines, label_lines = self.labels()
        result = assign_labels_on_page([LEFT, RIGHT], label_lines, SIZE[1],
                                       text_lines=lines)
        assert [a.key for a in result] == ["figure 3", "figure 4"]
        assert all(a.bare_tag for a in result)
        assert result[0].caption.startswith("Fig. 3. An irradiated")
        assert result[1].caption.startswith("Fig. 4. A Y-shaped")

    def test_switched_off_keeps_the_tag_alone(self):
        lines, label_lines = self.labels()
        result = assign_labels_on_page([LEFT, RIGHT], label_lines, SIZE[1],
                                       text_lines=lines, read_bare_tags=False)
        assert [a.caption for a in result] == ["Fig. 3", "Fig. 4"]
        assert not any(a.bare_tag for a in result)

    def test_a_tag_with_no_legend_keeps_the_tag(self):
        lines = text_lines(TAGS)
        result = assign_labels_on_page([LEFT, RIGHT], find_label_lines(lines), SIZE[1],
                                       text_lines=lines)
        assert [a.caption for a in result] == ["Fig. 3", "Fig. 4"]
        assert all(a.bare_tag for a in result)

    def test_a_caption_line_is_not_a_bare_tag(self):
        lines = text_lines(LEGENDS)
        result = assign_labels_on_page([(60, 300, 400, 495)], find_label_lines(lines),
                                       SIZE[1], text_lines=lines)
        assert result[0].key == "figure 3" and not result[0].bare_tag
