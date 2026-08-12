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
from typing import Iterator, List, Optional

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image

from .base import (
    DrawingBlock,
    ImageBlock,
    Rect,
    Word,
    flip_y,
    render_scale,
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
        Word boxes, assembled from PDFium's character rectangles.

        PDFium has no "give me words" call — it reports contiguous text *rects*,
        each of which may hold several words. We split each rect's text on
        whitespace and apportion the rect's width across the resulting words in
        proportion to their character counts.

        That apportioning is an approximation: it assumes roughly even character
        widths within a rect. It is good enough for the two things word boxes are
        used for — the word-coverage ratio (which only counts tokens) and picking
        up caption text inside a region — but do not rely on these boxes for
        pixel-accurate cropping. PyMuPDF's are exact if that is ever needed.
        """
        if self._words is not None:
            return self._words

        text_page = self._text_page()
        height = self.size[1]
        words: List[Word] = []

        # count_rects() must be called before get_rect(); PDFium builds its
        # internal rect list as a side effect.
        for rect_index in range(text_page.count_rects()):
            left, bottom, right, top = text_page.get_rect(rect_index)
            content = text_page.get_text_bounded(left, bottom, right, top)
            tokens = content.split()
            if not tokens:
                continue

            span = right - left
            total_chars = sum(len(t) for t in tokens)
            if total_chars == 0:
                continue

            cursor = left
            for token in tokens:
                token_width = span * (len(token) / total_chars)
                box = Rect(cursor, bottom, cursor + token_width, top)
                words.append(Word(text=token, rect=flip_y(box, height).normalized()))
                cursor += token_width

        self._words = words
        return words

    def text_in(self, rect: Rect) -> str:
        return " ".join(w.text for w in words_in_rect(self.words(), rect))

    def _object_rects(self, object_type: int) -> List[Rect]:
        """Bounding boxes of every page object of the given PDFium type."""
        height = self.size[1]
        rects: List[Rect] = []
        for obj in self._page.get_objects():
            if obj.type != object_type:
                continue
            left, bottom, right, top = obj.get_pos()
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
