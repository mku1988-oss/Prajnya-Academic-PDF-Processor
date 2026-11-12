#!/usr/bin/env python3
"""
Table orientation — the tight, upright image a table is READ from.

For biology researchers: the table reader (GLM-OCR) reads a table more
faithfully when it is given just the table body, the right way up. Measured on
the 31 corpus tables (eval/second_reader/GATE.md, "Finding 0"): reading the
tight, upright crop fixed 4 of the 7 tables Prajnya had got seriously wrong
(t09, t14, t29, t30), and made none of them worse. Five of the corpus tables
are printed SIDEWAYS on a portrait page; in that experiment they were turned
by hand. This module is the production version of both halves:

  1. **Tight.** The read uses the table's own box (snapped to whitespace so no
     glyph is cut), not the larger saved crop that also carries the caption
     and footnote. Nothing is lost by this: across all 62 stored grids, GLM's
     table reader never wrote any text outside the table itself, and the
     caption is read separately from the PDF's text layer.
  2. **Upright.** A table counts as sideways when most of its words run
     vertically in the PDF's text layer (the rule fixed in advance in
     eval/second_reader/confirm/CONFIRM_GATE.md). Which way to turn it comes
     from the order of its characters: text reading up the page is turned
     clockwise, text reading down is turned anticlockwise.

The SAVED table PNG is untouched: people and the RAG still see the table as
printed, caption and all. Only what the model reads changes.

A scanned page has no text layer, so it can never be judged sideways here; its
table is still read from the tight box, as printed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..pdf.base import PdfPage, Rect, words_in_rect

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

#: A word needs at least this many characters to vote: a single letter or
#: digit is near-square, so its box says nothing about direction.
MIN_VOTING_WORD_LENGTH = 2

#: Fewer voting words than this and the page is not trusted to say "sideways".
#: Measured: the sideways corpus tables carry 20 to 264 words each, ALL of them
#: vertical; ordinary tables carry 0 to 11 vertical words (rotated column
#: headers), never a majority.
MIN_VOTING_WORDS = 5

#: PIL's `rotate` turns ANTICLOCKWISE for positive angles. Text reading up the
#: page needs a clockwise quarter turn to stand upright, hence -90.
TURN_FOR_DIRECTION = {"up": -90, "down": 90}


def table_read_image(
    page_image: "Image.Image",
    read_box: Sequence[int],
    page: Optional[PdfPage] = None,
) -> "Image.Image":
    """
    The image a table should be READ from: its tight box, turned upright.

    Args:
        page_image: the rendered page the box was measured on.
        read_box: the table's tight box, (x0, y0, x1, y1) in page-image pixels.
        page: the PDF page, for its text layer. None (or a page with no
            text) leaves the crop as printed.

    Returns:
        The crop, turned a quarter when the table is printed sideways.
    """
    crop = page_image.crop(tuple(int(v) for v in read_box))
    turn = table_turn_degrees(page, read_box, page_image.size)
    return crop.rotate(turn, expand=True) if turn else crop


def table_turn_degrees(
    page: Optional[PdfPage],
    read_box: Sequence[int],
    image_size: Tuple[int, int],
) -> int:
    """
    How far to turn a table's crop (PIL degrees): 0, -90 (clockwise) or 90.

    Two questions, answered from the text layer:
      1. Is it sideways? Most of its words run vertically.
      2. Which way? The order its letters come in (see
         `PdfPage.vertical_text_direction`). When the page cannot say, "up"
         is assumed: it is the publishing convention, and all five sideways
         corpus tables read up.
    """
    if page is None:
        return 0
    rect = _box_to_pdf_rect(read_box, page, image_size)
    if not _is_sideways(page, rect):
        return 0
    direction = page.vertical_text_direction(rect) or "up"
    return TURN_FOR_DIRECTION[direction]


# --- helpers (after the main functions, per CLAUDE.md) ---------------------

def _is_sideways(page: PdfPage, rect: Rect) -> bool:
    """True when most of the multi-letter words in `rect` run vertically."""
    words = [w for w in words_in_rect(page.words(), rect)
             if len(w.text.strip()) >= MIN_VOTING_WORD_LENGTH]
    if len(words) < MIN_VOTING_WORDS:
        return False
    vertical = sum(1 for w in words if w.rect.height > w.rect.width)
    return vertical > len(words) / 2


def _box_to_pdf_rect(
    box: Sequence[int],
    page: PdfPage,
    image_size: Tuple[int, int],
) -> Rect:
    """Page-image pixels to PDF points, by the render's own scale on each axis."""
    page_width, page_height = page.size
    scale_x = page_width / image_size[0]
    scale_y = page_height / image_size[1]
    x0, y0, x1, y1 = box
    return Rect(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
