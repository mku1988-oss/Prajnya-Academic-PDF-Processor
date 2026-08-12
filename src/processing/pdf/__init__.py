#!/usr/bin/env python3
"""
PDF access layer — pick a backend by name, use it through one interface.

    from src.processing.pdf import get_backend

    backend = get_backend("pypdfium2")
    with backend.open(pdf_path) as doc:
        page = doc.page(1)
        print(page.text())

Backends are imported lazily so that having only one of the two libraries
installed is not an error. That matters here: PyMuPDF is AGPL-3.0 and is
deliberately absent from `requirements.txt`, so a clean install has the
permissive backend only.
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import (
    DrawingBlock,
    ImageBlock,
    PdfBackend,
    PdfDocument,
    PdfPage,
    Rect,
    Word,
    coverage,
    flip_y,
    render_scale,
    words_in_rect,
)

__all__ = [
    "DrawingBlock", "ImageBlock", "PdfBackend", "PdfDocument", "PdfPage",
    "Rect", "Word", "coverage", "flip_y", "render_scale", "words_in_rect",
    "get_backend", "available_backends", "DEFAULT_BACKEND",
]

#: The permissive backend is the default. PyMuPDF is faster and more precise but
#: is AGPL-3.0, so choosing it must be a deliberate, visible act.
DEFAULT_BACKEND = "pypdfium2"


def _load_pypdfium() -> PdfBackend:
    from .pypdfium_backend import PyPdfiumBackend
    return PyPdfiumBackend()


def _load_pymupdf() -> PdfBackend:
    from .pymupdf_backend import PyMuPdfBackend
    return PyMuPdfBackend()


_LOADERS: Dict[str, Callable[[], PdfBackend]] = {
    "pypdfium2": _load_pypdfium,
    "pymupdf": _load_pymupdf,
}


def get_backend(name: str | None = None) -> PdfBackend:
    """
    Build a PDF backend by name.

    Args:
        name: "pypdfium2" (Apache-2.0/BSD-3, the default) or "pymupdf"
              (AGPL-3.0, faster). None selects DEFAULT_BACKEND.

    Raises:
        ValueError: unknown backend name.
        ImportError: the named library is not installed.
    """
    key = (name or DEFAULT_BACKEND).lower()
    loader = _LOADERS.get(key)
    if loader is None:
        raise ValueError(f"unknown PDF backend {name!r}; available: {sorted(_LOADERS)}")
    return loader()


def available_backends() -> Dict[str, str]:
    """
    Which backends can actually be constructed here, mapped to their licences.

    Useful in diagnostics and at startup, so a run that silently fell back to an
    AGPL library is visible rather than assumed.
    """
    found: Dict[str, str] = {}
    for key, loader in _LOADERS.items():
        try:
            found[key] = loader().licence
        except ImportError:
            continue
    return found
