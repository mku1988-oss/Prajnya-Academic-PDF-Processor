"""
Tests for fitting a figure crop to its caption's band (utils/figure_crop_fit.py,
ledger T25).

For biology researchers: a hand-built page (tests/fakes.py) with a printed
caption, a line of prose above it, and black rectangles standing in for figure
panels. The image is the page plus the renderer's 25-pixel white border, at one
pixel per point, so a point box (x0, y0, x1, y1) is the pixel box shifted by 25.
"""

from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw

from src.processing.layout.base import LABEL_IMAGE, LABEL_TABLE, LABEL_TEXT, Detection
from src.processing.pdf.base import Rect, Word
from src.processing.utils.figure_crop_fit import RENDER_PADDING_PX, fit_crops_to_captions
from tests.fakes import FakePage

SIZE = (612.0, 792.0)
PAD = RENDER_PADDING_PX


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


def snapped(box):
    """A panel's crop as the whitespace snap leaves it: 6 clear pixels round the ink."""
    x0, y0, x1, y1 = px(box)
    return (x0 - 6, y0 - 6, x1 + 6, y1 + 6)


PROSE_ABOVE = line("The pulling event creates a thin filament between two cells here", 50, 200)
CAPTION = (line("Figure 5. Colocalization and stretching of collagen and FN", 50, 500)
           + line("Shown is a thin membrane extension colocalized with collagen", 50, 511))
LEFT_PANEL = (70, 240, 280, 470)
RIGHT_PANEL = (300, 240, 480, 470)


@dataclass
class Crop:
    bbox: tuple
    label_key: Optional[str]
    printed_label: Optional[str] = None
    page: int = 9
    figure_id: str = "p9_fig1"


def page_image(*panels, frame=None):
    """A white padded render with solid panels and an optional thin frame (points)."""
    image = Image.new("RGB", (int(SIZE[0]) + 2 * PAD, int(SIZE[1]) + 2 * PAD), "white")
    draw = ImageDraw.Draw(image)
    for panel in panels:
        draw.rectangle(px(panel), fill="black")
    if frame is not None:
        draw.rectangle(px(frame), outline="black", width=1)
    return image


def fit(image, crops, detections=(), words=PROSE_ABOVE + CAPTION):
    page = FakePage(words=words, size=SIZE, number=8)
    return {id(c): box for c, box in fit_crops_to_captions(page, image, crops, list(detections))}


def test_a_crop_running_into_its_caption_stops_above_it():
    """Frontiers: the box follows the frame and holds the legend's first line."""
    crop = Crop(px((60, 230, 550, 506)), "figure 5")
    image = page_image((60, 230, 550, 480))
    new = fit(image, [crop])[id(crop)]
    assert new[3] < px((0, 500, 0, 500))[1]
    assert new[:3] == crop.bbox[:3]


def test_a_missed_panel_in_the_caption_band_joins_the_crop():
    """PMC5880930 Figure 7: the detector found one panel of three."""
    crop = Crop(snapped(LEFT_PANEL), "figure 5")
    image = page_image(LEFT_PANEL, RIGHT_PANEL)
    new = fit(image, [crop])[id(crop)]
    assert new[0] <= px(LEFT_PANEL)[0] and new[2] >= px(RIGHT_PANEL)[2]
    assert new[3] < px((0, 500, 0, 500))[1]


def test_a_frame_round_figure_and_caption_is_not_figure_ink():
    crop = Crop(snapped(LEFT_PANEL), "figure 5")
    image = page_image(LEFT_PANEL, RIGHT_PANEL, frame=(45, 225, 560, 530))
    new = fit(image, [crop])[id(crop)]
    assert new[0] > px((45, 0, 0, 0))[0] and new[2] < px((0, 0, 560, 0))[2]
    assert new[1] > px((0, 225, 0, 0))[1]


def test_ink_above_the_prose_line_is_not_taken():
    """The band stops at the nearest full line of prose above the caption."""
    crop = Crop(snapped(LEFT_PANEL), "figure 5")
    image = page_image(LEFT_PANEL, (70, 120, 540, 180))
    assert fit(image, [crop]) == {}


def test_a_detected_text_block_above_is_a_ceiling():
    crop = Crop(snapped(LEFT_PANEL), "figure 5")
    image = page_image(LEFT_PANEL, (300, 240, 540, 300))
    block = Detection(label=LABEL_TEXT, box_px=px((300, 230, 560, 320)))
    assert fit(image, [crop], detections=[block]) == {}


def test_a_caption_shared_by_two_panel_crops_is_left_alone():
    left, right = Crop(snapped(LEFT_PANEL), "figure 5"), Crop(snapped(RIGHT_PANEL), "figure 5")
    image = page_image(LEFT_PANEL, RIGHT_PANEL, (70, 480, 540, 495))
    assert fit(image, [left, right]) == {}


def test_growth_into_a_neighbouring_table_is_refused():
    crop = Crop(snapped(LEFT_PANEL), "figure 5")
    image = page_image(LEFT_PANEL, RIGHT_PANEL)
    table = Detection(label=LABEL_TABLE, box_px=px(RIGHT_PANEL))
    assert fit(image, [crop], detections=[table]) == {}


def test_a_crop_is_never_made_smaller_than_the_detector_box_above_the_caption():
    crop = Crop(px((40, 220, 570, 480)), "figure 5")
    image = page_image(LEFT_PANEL)
    assert fit(image, [crop]) == {}


def test_an_unlabelled_crop_is_left_alone():
    crop = Crop(snapped(LEFT_PANEL), None)
    image = page_image(LEFT_PANEL, RIGHT_PANEL)
    assert fit(image, [crop]) == {}


def test_a_figure_detection_is_not_a_ceiling():
    """Only non-figure detections (text, headings, heads) stop the band."""
    crop = Crop(snapped(LEFT_PANEL), "figure 5")
    image = page_image(LEFT_PANEL, RIGHT_PANEL)
    other_figure_box = Detection(label=LABEL_IMAGE, box_px=px(LEFT_PANEL))
    new = fit(image, [crop], detections=[other_figure_box])[id(crop)]
    assert new[2] >= px(RIGHT_PANEL)[2]


def test_ink_across_a_wide_white_gap_is_not_taken():
    """PMC12416696: a journal logo at the page top, far above the figure."""
    words = CAPTION                       # no prose line above: the gap is the only stop
    tall_panel = (70, 300, 280, 470)
    crop = Crop(snapped(tall_panel), "figure 5")
    image = page_image(tall_panel, (70, 40, 200, 80))
    assert fit(image, [crop], words=words) == {}


def test_an_edge_tab_beside_the_text_block_is_not_taken():
    """Nature Reviews prints a coloured tab in the side margin, close to a figure."""
    crop = Crop(snapped(RIGHT_PANEL), "figure 5")
    image = page_image(RIGHT_PANEL, (560, 300, 600, 360))      # text block ends at 484
    assert fit(image, [crop]) == {}


def test_a_logo_in_the_top_of_the_page_is_not_taken_however_close():
    """Cell Press: a header rule 38 pt above the figure, nearer than a far panel."""
    words = line("Running text spans the column from the left margin to the right one", 50, 700)
    words += CAPTION
    panel = (70, 100, 280, 470)
    crop = Crop(snapped(panel), "figure 5")
    image = page_image(panel, (70, 60, 280, 64))               # top 9% of 792 pt = 71 pt
    assert fit(image, [crop], words=words) == {}


def test_a_panel_36_points_away_still_joins():
    """PMC9772451 Fig. 4: heatmap C starts 34-36 pt below panels A and B."""
    crop = Crop(snapped(LEFT_PANEL), "figure 5")
    image = page_image(LEFT_PANEL, (316, 240, 480, 470))       # 36 pt to its right
    new = fit(image, [crop])[id(crop)]
    assert new[2] >= px((0, 0, 480, 0))[2]


def test_a_panel_letter_crossing_the_top_zone_line_is_kept():
    """Collagen structure Fig. 1: a figure printed high on the page; its "A"
    crosses the 9% line (71 pt here) and must come in whole."""
    words = line("Running text spans the column from the left margin to the right one", 50, 700)
    words += CAPTION
    panel = (70, 84, 280, 470)
    crop = Crop(snapped(panel), "figure 5")
    image = page_image(panel, (70, 66, 80, 78))
    new = fit(image, [crop], words=words)[id(crop)]
    assert new[1] <= px((0, 66, 0, 0))[1]


def test_a_panel_letter_wholly_in_the_top_zone_joins_from_close_by():
    """PMC8440575 Fig. 4: "a" sits wholly above the 9% line, 4 pt over its panel."""
    words = line("Running text spans the column from the left margin to the right one", 50, 700)
    words += CAPTION
    panel = (70, 74, 280, 470)
    crop = Crop(snapped(panel), "figure 5")
    image = page_image(panel, (70, 58, 80, 70))
    new = fit(image, [crop], words=words)[id(crop)]
    assert new[1] <= px((0, 58, 0, 0))[1]
