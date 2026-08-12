#!/usr/bin/env python3
"""
PyMuPDF (`fitz`) implementation of the PDF interface.

**Licence warning: PyMuPDF is AGPL-3.0** (or a paid Artifex licence). It is the
fastest option and is what V5 used, so it remains the default for local work —
but it cannot ship in a closed-source or hosted product without resolving that.
See `pypdfium_backend.py` for the permissive alternative and CLAUDE.md for the
project rule.

PyMuPDF already uses a top-left origin, so no coordinate flipping is needed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

from PIL import Image

from .base import (
    DrawingBlock,
    ImageBlock,
    Rect,
    Word,
    render_scale,
    words_in_rect,
)

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyMuPDF is not installed. It is AGPL-3.0 and deliberately not listed in "
        "requirements.txt — install it explicitly, or use the pypdfium2 backend."
    ) from exc


class PyMuPdfPage:
    """One page, backed by a `fitz.Page`."""

    def __init__(self, page: "fitz.Page", index: int) -> None:
        self._page = page
        self._index = index
        self._words: Optional[List[Word]] = None

    @property
    def number(self) -> int:
        return self._index

    @property
    def size(self) -> tuple[float, float]:
        rect = self._page.rect
        return (rect.width, rect.height)

    def text(self) -> str:
        return self._page.get_text("text")

    def words(self) -> List[Word]:
        """
        Word boxes from PyMuPDF's `get_text("words")`.

        Cached because the coverage check and caption extraction both want them,
        and re-parsing the page for each is pure waste.
        """
        if self._words is None:
            self._words = [
                Word(text=item[4], rect=Rect(item[0], item[1], item[2], item[3]))
                for item in self._page.get_text("words")
            ]
        return self._words

    def text_in(self, rect: Rect) -> str:
        return " ".join(w.text for w in words_in_rect(self.words(), rect))

    def images(self) -> List[ImageBlock]:
        """
        Embedded raster images, located by their placement on the page.

        `get_image_info` gives the on-page bbox directly, which is what we want —
        `get_images` would give the resource entries without placement.
        """
        blocks: List[ImageBlock] = []
        for info in self._page.get_image_info():
            bbox = info.get("bbox")
            if bbox:
                blocks.append(ImageBlock(rect=Rect(*bbox).normalized()))
        return blocks

    def drawings(self) -> List[DrawingBlock]:
        blocks: List[DrawingBlock] = []
        for drawing in self._page.get_drawings():
            bbox = drawing.get("rect")
            if bbox is not None:
                blocks.append(DrawingBlock(rect=Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1).normalized()))
        return blocks

    def render(self, *, dpi: Optional[int] = None, max_edge: Optional[int] = None) -> Image.Image:
        scale = render_scale(self.size, dpi=dpi, max_edge=max_edge)
        pixmap = self._page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


class PyMuPdfDocument:
    """An open PDF held by a single `fitz.Document` handle."""

    def __init__(self, path: Path | str) -> None:
        self._doc = fitz.open(str(path))
        self._path = Path(path)

    @property
    def page_count(self) -> int:
        return self._doc.page_count

    def page(self, index: int) -> PyMuPdfPage:
        return PyMuPdfPage(self._doc[index], index)

    def pages(self) -> Iterator[PyMuPdfPage]:
        for index in range(self.page_count):
            yield self.page(index)

    def close(self) -> None:
        self._doc.close()

    def __enter__(self) -> "PyMuPdfDocument":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"PyMuPdfDocument({self._path.name!r}, pages={self.page_count})"


class PyMuPdfBackend:
    """Fast, feature-rich, and AGPL-3.0. See the module docstring."""

    @property
    def name(self) -> str:
        return "pymupdf"

    @property
    def licence(self) -> str:
        return "AGPL-3.0-or-later"

    def open(self, path: Path | str) -> PyMuPdfDocument:
        return PyMuPdfDocument(path)
