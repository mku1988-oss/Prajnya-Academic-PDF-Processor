#!/usr/bin/env python3
"""
pypdfium2 (Google's PDFium) implementation of the PDF interface.

**This is the permissively-licensed path**: pypdfium2 is Apache-2.0 OR
BSD-3-Clause and PDFium itself is BSD-3-Clause, so unlike PyMuPDF it carries no
copyleft obligation. V5 already depended on it for rendering.

## The coordinate flip

PDFium uses a **bottom-left** origin with y increasing upward; this interface
promises a **top-left** origin with y increasing downward. Every rectangle
leaving this module therefore passes through `flip_y`. Getting this wrong would
crop figures from the mirrored part of the page — which is exactly the class of
bug the shared interface exists to prevent, so the cross-backend agreement test
in `tests/test_pdf_backend.py` checks it explicitly.
"""

from __future__ import annotations

from pathlib import Path
import ctypes
import math
import statistics
from typing import Iterator, List, Optional

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image

from .base import (
    DrawingBlock,
    ImageBlock,
    Rect,
    group_characters_into_lines,
    styled_line_from_characters,
    StyledLine,
    Word,
    flip_y,
    render_scale,
    vertical_direction_from_steps,
    words_in_rect,
)

#: PDFium page-object type constants, named for readability at the call site.
_OBJ_IMAGE = pdfium_raw.FPDF_PAGEOBJ_IMAGE
_OBJ_PATH = pdfium_raw.FPDF_PAGEOBJ_PATH


class PyPdfiumPage:
    """One page, backed by a PDFium page handle."""

    def __init__(self, page: "pdfium.PdfPage", index: int) -> None:
        self._page = page
        self._index = index
        self._textpage: Optional["pdfium.PdfTextPage"] = None
        self._words: Optional[List[Word]] = None

    @property
    def number(self) -> int:
        return self._index

    @property
    def size(self) -> tuple[float, float]:
        return self._page.get_size()

    def _text_page(self) -> "pdfium.PdfTextPage":
        """Lazily build the text page — it is expensive and not always needed."""
        if self._textpage is None:
            self._textpage = self._page.get_textpage()
        return self._textpage

    def text(self) -> str:
        return self._text_page().get_text_bounded()

    def words(self) -> List[Word]:
        """
        Word boxes, built from PDFium's individual character boxes.

        For biology researchers: a word's box says where on the page it sits.
        Gutters between table columns, the gap between a journal page's two
        columns, and which words fall inside a figure's caption box are all
        read from these boxes, so they must be where the ink is.

        PDFium has no "give me words" call. Walking its characters in text
        order, a word is a run of characters that are not whitespace (the
        same rule as `str.split()`), and its box is the union of its
        characters' LOOSE boxes: font ascent to descent, glyph origin to
        advance, the same kind of box PyMuPDF reports.

        **Until v6.4.0 this split each PDFium text rectangle (often several
        words, sometimes a whole table row) by character count, which erased
        every gap between words.** That made gutter-based column counting
        report one column (ledger T18) and defeated the column-gap rule in
        `group_words_into_lines`. Measured against the old boxes and PyMuPDF
        in `eval/word_boxes/GATE.md` (ledger T19).
        """
        if self._words is not None:
            return self._words

        text_page = self._text_page()
        try:
            count = pdfium_raw.FPDFText_CountChars(text_page)
        except Exception:                                   # pragma: no cover
            count = 0

        words: List[Word] = []
        letters: List[str] = []
        boxes: List[tuple] = []
        turns: List[bool] = []
        for index in range(count):
            character = _character_at(text_page, index)
            if character is None:
                continue
            if character.isspace():
                _close_word(words, letters, boxes, turns, self.size[1])
                continue
            box = _character_box(text_page, index)
            # PDFium joins a word hyphenated at a line end ("pro\x02posed")
            # with no whitespace, so a line break must also end the word, or
            # its box would span two lines.
            if box is not None and boxes and not _on_one_line(boxes[-1], box):
                _close_word(words, letters, boxes, turns, self.size[1])
            letters.append(character)
            if box is not None:
                boxes.append(box)
            turned = _character_is_turned(text_page, index)
            if turned is not None:
                turns.append(turned)
        _close_word(words, letters, boxes, turns, self.size[1])

        self._words = words
        return words


    def styled_lines(self) -> List[StyledLine]:
        """
        Typeset lines with cap-height, weight and case, from a character walk.

        PDFium exposes font data per CHARACTER, not per word, so this walks
        the text page once and groups characters into lines by baseline.

        **`FPDFText_GetFontSize` is not usable here and that is not a bug in
        this code.** PDFium reports the size in text space, which is 1.0 for
        every character in every file we process because the real size lives
        in the text matrix. The rendered glyph box is the honest measurement,
        so cap-height is taken from `FPDFText_GetCharBox`.
        """
        try:
            text_page = self._text_page()
            count = pdfium_raw.FPDFText_CountChars(text_page)
        except Exception:                                   # pragma: no cover
            return []
        if not count:
            return []

        height = self.size[1]
        chars = []
        for index in range(count):
            try:
                code = pdfium_raw.FPDFText_GetUnicode(text_page, index)
                left, right, bottom, top = (ctypes.c_double() for _ in range(4))
                pdfium_raw.FPDFText_GetCharBox(
                    text_page, index, ctypes.byref(left), ctypes.byref(right),
                    ctypes.byref(bottom), ctypes.byref(top))
                weight = pdfium_raw.FPDFText_GetFontWeight(text_page, index)
            except Exception:                               # pragma: no cover
                continue
            character = chr(code) if code else ""
            # Spaces are kept: they are the word boundaries of the line's
            # text. They are excluded from the MEASUREMENTS further down,
            # where a zero-height box would drag a median.
            if not character or not character.isprintable():
                continue
            chars.append({
                "ch": character,
                "x0": left.value, "x1": right.value,
                "y0": bottom.value, "y1": top.value,
                "weight": float(weight) if weight and weight > 0 else 0.0,
            })

        lines = []
        for group in group_characters_into_lines(chars):
            line = styled_line_from_characters(group, height)
            if line is not None:
                lines.append(line)
        return lines

    def text_in(self, rect: Rect) -> str:
        return " ".join(w.text for w in words_in_rect(self.words(), rect))

    def vertical_text_direction(self, rect: Rect) -> str:
        """
        "up", "down" or "" for the vertical text inside `rect`.

        Walks PDFium's characters in their stored (reading) order and records
        the step from each letter to the next, for every pair of NEIGHBOURING
        letters whose centres both lie inside `rect`. The vote itself is
        `vertical_direction_from_steps`, shared with the other backend.
        """
        try:
            text_page = self._text_page()
            count = pdfium_raw.FPDFText_CountChars(text_page)
        except Exception:                                   # pragma: no cover
            return ""
        height = self.size[1]
        steps, previous = [], None
        for index in range(count):
            centre = self._letter_centre(text_page, index, height)
            if centre is None or not (rect.x0 <= centre[0] <= rect.x1
                                      and rect.y0 <= centre[1] <= rect.y1):
                previous = None          # a gap breaks the chain of neighbours
                continue
            if previous is not None:
                steps.append((centre[0] - previous[0], centre[1] - previous[1]))
            previous = centre
        return vertical_direction_from_steps(steps)

    def _letter_centre(self, text_page, index: int, height: float):
        """Top-left-origin centre of character `index`, or None if not a letter/digit."""
        try:
            code = pdfium_raw.FPDFText_GetUnicode(text_page, index)
            if not code or not chr(code).isalnum():
                return None
            left, right, bottom, top = (ctypes.c_double() for _ in range(4))
            pdfium_raw.FPDFText_GetCharBox(
                text_page, index, ctypes.byref(left), ctypes.byref(right),
                ctypes.byref(bottom), ctypes.byref(top))
        except Exception:                                   # pragma: no cover
            return None
        return ((left.value + right.value) / 2,
                height - (bottom.value + top.value) / 2)

    def _object_rects(self, object_type: int) -> List[Rect]:
        """Bounding boxes of every page object of the given PDFium type."""
        height = self.size[1]
        rects: List[Rect] = []
        for obj in self._page.get_objects():
            if obj.type != object_type:
                continue
            left, bottom, right, top = _object_bounds(obj)
            rects.append(flip_y(Rect(left, bottom, right, top), height).normalized())
        return rects

    def images(self) -> List[ImageBlock]:
        return [ImageBlock(rect=r) for r in self._object_rects(_OBJ_IMAGE)]

    def drawings(self) -> List[DrawingBlock]:
        return [DrawingBlock(rect=r) for r in self._object_rects(_OBJ_PATH)]

    def render(self, *, dpi: Optional[int] = None, max_edge: Optional[int] = None) -> Image.Image:
        scale = render_scale(self.size, dpi=dpi, max_edge=max_edge)
        bitmap = self._page.render(scale=scale)
        return bitmap.to_pil().convert("RGB")


def _character_at(text_page, index: int) -> Optional[str]:
    """The character at `index`, or None for an empty code (PDFium's 0)."""
    try:
        code = pdfium_raw.FPDFText_GetUnicode(text_page, index)
    except Exception:                                       # pragma: no cover
        return None
    return chr(code) if code else None


def _character_box(text_page, index: int) -> Optional[tuple]:
    """(left, bottom, right, top) of one character in PDF space, or None.

    The LOOSE box first (font ascent to descent, origin to advance), so every
    letter on a line has the same height and a word box does not shrink to
    its ink. A loose box with no area falls back to the tight glyph box; a
    character with neither (a generated or invisible one) has no box.
    """
    loose = pdfium_raw.FS_RECTF()
    try:
        if pdfium_raw.FPDFText_GetLooseCharBox(text_page, index, ctypes.byref(loose)):
            box = (min(loose.left, loose.right), min(loose.bottom, loose.top),
                   max(loose.left, loose.right), max(loose.bottom, loose.top))
            if box[2] > box[0] and box[3] > box[1]:
                return box
        left, right, bottom, top = (ctypes.c_double() for _ in range(4))
        if pdfium_raw.FPDFText_GetCharBox(text_page, index, ctypes.byref(left), ctypes.byref(right),
                                          ctypes.byref(bottom), ctypes.byref(top)):
            box = (left.value, bottom.value, right.value, top.value)
            if box[2] > box[0] or box[3] > box[1]:
                return box
    except Exception:                                       # pragma: no cover
        pass
    return None


def _on_one_line(previous: tuple, current: tuple) -> bool:
    """Whether two character boxes sit on one printed line, horizontal or sideways.

    On one horizontal line two letters share their height; on one sideways
    line they share their width. Sharing more than half of the smaller extent
    on EITHER axis counts as one line (eval/word_boxes/GATE.md, Addendum 1).
    """
    def shared(low_a, high_a, low_b, high_b) -> bool:
        overlap = min(high_a, high_b) - max(low_a, low_b)
        return overlap > 0.5 * min(high_a - low_a, high_b - low_b)

    return (shared(previous[1], previous[3], current[1], current[3])
            or shared(previous[0], previous[2], current[0], current[2]))


def _character_is_turned(text_page, index: int) -> Optional[bool]:
    """Whether a character is set more than 45 degrees from upright, or None if PDFium cannot say.

    PDFium reports the angle in radians: 0 for ordinary text, about 1.57 (a
    quarter turn) for the "Downloaded from ..." stamp down a margin. It
    returns -1 when it has no answer, which must not be read as an angle.
    """
    try:
        angle = pdfium_raw.FPDFText_GetCharAngle(text_page, index)
    except Exception:                                       # pragma: no cover
        return None
    if angle is None or angle < 0:
        return None
    return abs(math.sin(angle)) > math.sin(math.radians(45))


def _close_word(words: List[Word], letters: List[str], boxes: List[tuple],
                turns: List[bool], page_height: float) -> None:
    """Turn the letters gathered so far into a Word (box = union of its letters), then reset.

    A word none of whose letters has a box keeps its text, placed as a
    zero-size box where the previous word ended, so text is never lost. The
    word is `rotated` when most of its letters are turned; None when PDFium
    gave no angle for any of them.
    """
    rotated = (sum(turns) > len(turns) / 2) if turns else None
    if letters:
        if boxes:
            union = Rect(min(b[0] for b in boxes), min(b[1] for b in boxes),
                         max(b[2] for b in boxes), max(b[3] for b in boxes))
            rect = flip_y(union, page_height).normalized()
        elif words:
            last = words[-1].rect
            rect = Rect(last.x1, last.y0, last.x1, last.y1)
        else:
            rect = Rect(0.0, 0.0, 0.0, 0.0)
        words.append(Word(text="".join(letters), rect=rect, rotated=rotated))
    letters.clear()
    turns.clear()
    boxes.clear()


def _object_bounds(obj) -> tuple:
    """(left, bottom, right, top) of a page object, on pypdfium2 4.x and 5.x alike.

    pypdfium2 5.0 renamed `PdfObject.get_pos()` to `get_bounds()` with no
    fallback. `requirements.txt` allowed any version >= 4.24, so a fresh
    install of v6.3.0 got 5.x and the page classifier crashed on EVERY page
    (`'PdfImage' object has no attribute 'get_pos'`). Found 2026-11-13.
    """
    if hasattr(obj, "get_bounds"):
        return obj.get_bounds()
    return obj.get_pos()


class PyPdfiumDocument:
    """An open PDF held by a single PDFium document handle."""

    def __init__(self, path: Path | str) -> None:
        self._doc = pdfium.PdfDocument(str(path))
        self._path = Path(path)

    @property
    def page_count(self) -> int:
        return len(self._doc)

    def page(self, index: int) -> PyPdfiumPage:
        return PyPdfiumPage(self._doc[index], index)

    def pages(self) -> Iterator[PyPdfiumPage]:
        for index in range(self.page_count):
            yield self.page(index)

    def close(self) -> None:
        self._doc.close()

    def __enter__(self) -> "PyPdfiumDocument":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"PyPdfiumDocument({self._path.name!r}, pages={self.page_count})"


class PyPdfiumBackend:
    """The permissively-licensed backend. See the module docstring."""

    @property
    def name(self) -> str:
        return "pypdfium2"

    @property
    def licence(self) -> str:
        return "Apache-2.0 OR BSD-3-Clause"

    def open(self, path: Path | str) -> PyPdfiumDocument:
        return PyPdfiumDocument(path)
