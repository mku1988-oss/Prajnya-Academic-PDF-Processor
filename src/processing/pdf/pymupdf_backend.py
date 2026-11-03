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

from dataclasses import replace
from pathlib import Path
from typing import Iterator, List, Optional

from PIL import Image

from .base import (
    DrawingBlock,
    ImageBlock,
    Rect,
    group_characters_into_lines,
    styled_line_from_characters,
    StyledLine,
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


# Bit 4 of a PyMuPDF span's `flags` marks a bold face. Named because a
# bare `1 << 4` in the middle of a character walk is unreadable.
_BOLD_FLAG = 1 << 4

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


    def styled_lines(self) -> List[StyledLine]:
        """
        Typeset lines with cap-height, weight and case, from `rawdict` spans.

        **`cap_height` is deliberately left at 0.0 ("unknown") on this
        backend, and that is a library limitation rather than an omission.**
        Cap-height means the ink height of a capital glyph, which is what
        makes an ALL-CAPS heading comparable with mixed-case prose. PyMuPDF
        reports every character's box as the whole LINE box -- `rawdict` and
        `get_texttrace()` both give 'C' and 'o' the identical height -- so
        the measurement simply is not available here. pypdfium2's
        `FPDFText_GetCharBox` does give ink boxes, and pypdfium2 is the
        shipping backend.

        Emitting `span["size"]` in its place was rejected: it is the true
        point size, which is a DIFFERENT quantity, and a consumer comparing
        it against pypdfium2's cap heights would be comparing two things that
        happen to share a field name. `weight` and `caps_fraction` are
        computed identically on both backends and do agree.
        """
        try:
            raw = self._page.get_text("rawdict")
        except Exception:                                   # pragma: no cover
            return []

        height = self.size[1]
        lines: List[StyledLine] = []
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                chars = []
                for span in line.get("spans", []):
                    weight = 700.0 if span.get("flags", 0) & _BOLD_FLAG else 400.0
                    for char in span.get("chars", []):
                        character = char.get("c") or ""
                        if not character or not character.isprintable():
                            continue
                        x0, y0, x1, y1 = char.get("bbox", (0, 0, 0, 0))
                        chars.append({
                            "ch": character,
                            "x0": x0, "x1": x1,
                            # rawdict is already top-left origin, and the
                            # helper expects a bottom-up box, so flip here.
                            "y0": height - y1, "y1": height - y0,
                            "weight": weight,
                        })
                built = styled_line_from_characters(chars, height)
                if built is not None:
                    # See the docstring: this backend cannot measure ink
                    # height, so it must not claim to.
                    lines.append(replace(built, cap_height=0.0))
        return lines

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
