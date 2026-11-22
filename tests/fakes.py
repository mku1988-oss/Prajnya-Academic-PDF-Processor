#!/usr/bin/env python3
"""
A hand-built `PdfPage` for tests — no PDF file, no backend, no I/O.

For biology researchers: the modules under test only ask a page simple
questions (its size, its words and their boxes, its images and drawings).
Because the `PdfBackend` seam is a Protocol — a description of behaviour, not
a base class — any object answering those questions is a valid page. This one
is built directly from the values a test wants, which makes each test's
geometry explicit and keeps the suite instant.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from src.processing.pdf.base import (
    DrawingBlock,
    ImageBlock,
    Rect,
    Word,
    words_in_rect,
)


class FakePage:
    """A `PdfPage` assembled from literal test values."""

    def __init__(
        self,
        *,
        words: Sequence[Word] = (),
        images: Sequence[ImageBlock] = (),
        drawings: Sequence[DrawingBlock] = (),
        size: tuple = (612.0, 792.0),  # US Letter in PDF points
        text: Optional[str] = None,
        number: int = 0,
        vertical_direction: str = "",
        leaders: Sequence[Rect] = (),
        glyphs: Sequence = (),
        characters: Sequence = (),
    ) -> None:
        self._characters = list(characters)
        self._leaders = list(leaders)
        self._glyphs = list(glyphs)
        self._vertical_direction = vertical_direction
        self._words = list(words)
        self._images = list(images)
        self._drawings = list(drawings)
        self._size = size
        self._text = text
        self._number = number

    @property
    def number(self) -> int:
        return self._number

    @property
    def size(self) -> tuple:
        return self._size

    def text(self) -> str:
        """Explicit text when given, else the words joined in list order."""
        if self._text is not None:
            return self._text
        return " ".join(w.text for w in self._words)

    def words(self) -> List[Word]:
        return list(self._words)

    def text_in(self, rect: Rect) -> str:
        return " ".join(w.text for w in words_in_rect(self._words, rect))

    def vertical_text_direction(self, rect: Rect) -> str:
        """The direction the test declared; a fake has no characters to walk."""
        return self._vertical_direction

    def images(self) -> List[ImageBlock]:
        return list(self._images)

    def drawings(self, limit: Optional[int] = None) -> List[DrawingBlock]:
        drawings = list(self._drawings)
        return drawings if limit is None else drawings[:limit + 1]

    def leaders(self) -> List[Rect]:
        """The dot-leader boxes the test declared (none by default)."""
        return list(self._leaders)

    def glyph_styles(self) -> list:
        """The characters the test declared (none by default: no typography)."""
        return list(self._glyphs)

    def characters(self) -> list:
        """The printed characters the test declared (none by default)."""
        return list(self._characters)

    def render(self, *, dpi=None, max_edge=None):  # pragma: no cover
        raise NotImplementedError("FakePage does not render")


def word_row(
    texts: Sequence[str],
    *,
    y: float,
    x: float = 72.0,
    height: float = 10.0,
    char_width: float = 5.0,
    gap: float = 4.0,
) -> List[Word]:
    """
    Lay a sequence of words out as one horizontal line.

    Each word's box width follows its character count, so the geometry looks
    like real typesetting: word gaps (~0.4× the height) stay far below the
    column-gap threshold used by line reconstruction.
    """
    words: List[Word] = []
    cursor = x
    for text in texts:
        width = max(char_width, char_width * len(text))
        words.append(Word(text=text, rect=Rect(cursor, y, cursor + width, y + height)))
        cursor += width + gap
    return words


def rotated_word(text: str, *, x: float, y: float, height_per_char: float = 5.0,
                 width: float = 9.0) -> Word:
    """A word rotated 90° — its box is taller than it is wide."""
    return Word(text=text, rect=Rect(x, y, x + width, y + height_per_char * len(text)))
