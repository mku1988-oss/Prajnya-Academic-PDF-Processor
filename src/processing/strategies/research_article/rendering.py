#!/usr/bin/env python3
"""
Page rendering and the document cache.

For biology researchers: before any model can "look" at a PDF page, the page
must be turned into an image, the same way you would photograph a printed
sheet before showing it to someone. This module does that, in exactly one
place, so every consumer sees pages rendered the same way.

Ported from V5's `_render_page_pypdfium` / `_render_extraction_page`
(research_article_strategy.py L2643 / L1682), with two deliberate changes:

1. **All access goes through `PdfBackend`.** V5 opened the PDF with pypdfium2
   directly, once per render — an 8-page document with a title render, body
   renders, and repair re-renders opened the same file from disk many times.
2. **`DocumentCache` (new, per plan.md Stage 0)** holds one open document and
   remembers rendered images by (page, size), so a page rendered for reading
   is not rendered again for repair. It also owns the handle-lifetime problem
   plan.md flags: pages are rendered eagerly inside the cache, so no caller
   ever holds a page proxy whose document has been closed.

The 25px white border (`RENDER_PADDING_PX`) is part of the rendering contract:
V5's known-good GLM/OlmOCR results were produced against padded images, and
the caption box-to-PDF-point mapping in `figures.py` subtracts the same
constant. Change one and the other silently mismaps — that is why the constant
lives here and is imported, never re-declared.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image

from ...pdf.base import PdfBackend, PdfDocument, PdfPage, TextStyle
from ...utils.text_style import document_body_style

#: White border, in pixels, added to every rendered page (matches V5 /
#: OlmOCR Hybrid v2, which the archived known-good output was produced with).
RENDER_PADDING_PX = 25


def render_page(page: PdfPage, target_size: int,
                padding: int = RENDER_PADDING_PX) -> Image.Image:
    """
    Render one page so its longest edge is `target_size` pixels, plus padding.

    Args:
        page: the page to render (any PdfBackend implementation).
        target_size: pixel length of the longest edge, before padding.
        padding: white border to add on every side.

    Returns:
        An RGB PIL image, `2 * padding` larger than the render on each axis.

    Raises:
        ValueError: if the page has zero dimensions (a corrupt PDF).
    """
    width_pts, height_pts = page.size
    if max(width_pts, height_pts) == 0:
        raise ValueError(f"Page {page.number + 1} has zero dimensions")

    image = page.render(max_edge=target_size)

    new_size = (image.width + 2 * padding, image.height + 2 * padding)
    padded = Image.new("RGB", new_size, (255, 255, 255))
    padded.paste(image, (padding, padding))
    return padded


def render_extraction_page(page: PdfPage, extraction_dpi: int) -> Image.Image:
    """
    Render a page at the extraction model's DPI rather than a fixed pixel size.

    The extraction model (GLM) was benchmarked at a given DPI, so its target
    size scales with the physical page: a longer page gets a taller image
    instead of being squeezed into a fixed box. (72 points = 1 inch, so
    points / 72 * dpi = pixels.)
    """
    width_pts, height_pts = page.size
    longest_pts = max(width_pts, height_pts)
    target_size = int(longest_pts / 72.0 * extraction_dpi)
    return render_page(page, target_size)


class DocumentCache:
    """
    One open document plus a memo of every page image rendered so far.

    Usage:

        with DocumentCache(backend, pdf_path) as cache:
            image = cache.page_image(3, target_size=1024)   # renders
            again = cache.page_image(3, target_size=1024)   # cache hit
            page = cache.page(3)                            # text-layer access

    The cache never hands out page proxies past `close()` — callers that need
    text access take `page()` while the cache is open. Rendered images are
    plain PIL objects and stay valid after close.
    """

    def __init__(self, backend: PdfBackend, pdf_path: Path | str,
                 logger: Optional[logging.Logger] = None) -> None:
        self.pdf_path = Path(pdf_path)
        self.logger = logger or logging.getLogger("DocumentCache")
        self._doc: Optional[PdfDocument] = backend.open(self.pdf_path)
        self._images: Dict[Tuple[int, int], Image.Image] = {}
        self._body_style: Optional[TextStyle] = None
        self._body_style_read = False

    @property
    def page_count(self) -> int:
        return self._require_open().page_count

    def page(self, index: int) -> PdfPage:
        """The live page object — valid only while the cache is open."""
        return self._require_open().page(index)

    def page_image(self, index: int, target_size: int) -> Image.Image:
        """Render a page (longest edge = `target_size` + padding), memoized."""
        key = (index, target_size)
        if key not in self._images:
            self._images[key] = render_page(self.page(index), target_size)
            self.logger.debug("Rendered page %d at %dpx", index + 1, target_size)
        return self._images[key]

    def extraction_image(self, index: int, extraction_dpi: int) -> Image.Image:
        """Render a page at the extraction model's DPI, memoized."""
        page = self.page(index)
        longest_pts = max(page.size)
        return self.page_image(index, int(longest_pts / 72.0 * extraction_dpi))

    def body_style(self) -> Optional[TextStyle]:
        """
        The document's running-text typeface, measured once and memoized.

        Used to end figure captions where the body text resumes (see
        `utils/text_style.py`). None for a document with no text layer.
        """
        if not self._body_style_read:
            doc = self._require_open()
            pages = [doc.page(index) for index in range(doc.page_count)]
            self._body_style = document_body_style(pages)
            self._body_style_read = True
        return self._body_style

    def close(self) -> None:
        """Close the document handle. Cached images remain usable."""
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def __enter__(self) -> "DocumentCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- helpers (after the main methods, per CLAUDE.md) ------------------

    def _require_open(self) -> PdfDocument:
        if self._doc is None:
            raise RuntimeError(
                f"DocumentCache for {self.pdf_path.name} is closed; "
                "page access is only valid while it is open."
            )
        return self._doc
