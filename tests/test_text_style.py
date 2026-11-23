#!/usr/bin/env python3
"""
Tests for `utils/text_style.py`: telling a caption's type from the body's.

For biology researchers: journals set figure captions in another font, or the
same font smaller. These tests pin the two measurements built on that (the
document's body style, and "is this line body text?") on hand-built
characters, then on the real corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.processing.pdf.base import GlyphStyle, TextStyle
from src.processing.utils.text_style import (
    PageTypography,
    document_body_style,
    dominant_style,
)
from tests.fakes import FakePage
from tests.local_samples import CORPUS as LOCAL_CORPUS

CORPUS = LOCAL_CORPUS


def _glyphs(font, x_height, count, y=100.0):
    """`count` x-height letters in one font, along one line."""
    return [GlyphStyle(x=10.0 + i, y=y, font=font, x_height=x_height) for i in range(count)]


class TestTheBodyStyle:
    def test_the_commonest_font_is_the_body_font(self):
        page = FakePage(glyphs=_glyphs("Times-Roman", 4.4, 300) + _glyphs("Helvetica", 4.3, 80))
        assert document_body_style([page]).font == "Times-Roman"

    def test_the_body_size_is_the_largest_size_used_at_length(self):
        """
        The NMR paper's shape: one font, MORE small print than body text.
        The median and the commonest size both pick the small print.
        """
        page = FakePage(glyphs=_glyphs("F", 3.73, 7758) + _glyphs("F", 4.66, 7071)
                        + _glyphs("F", 6.0, 50))            # a heading: large, rare
        assert document_body_style([page]).x_height == pytest.approx(4.66)

    def test_a_document_with_no_text_layer_has_no_body_style(self):
        assert document_body_style([FakePage(), FakePage()]) is None


class TestIsBodyText:
    HELVETICA = TextStyle("Helvetica-Condensed-Light", 4.33)
    TIMES = TextStyle("Times-Roman", 4.44)

    def test_caption_in_its_own_font_ends_at_the_body_font(self):
        typography = PageTypography(glyphs=[], body=self.TIMES)
        assert typography.is_body_text(self.TIMES, self.HELVETICA)
        assert not typography.is_body_text(self.HELVETICA, self.HELVETICA)

    def test_italics_inside_a_caption_do_not_end_it(self):
        """An italic species name is a third font, never the body's."""
        typography = PageTypography(glyphs=[], body=self.TIMES)
        italic = TextStyle("Helvetica-Condensed-LightItalic", 4.33)
        assert not typography.is_body_text(italic, self.HELVETICA)

    def test_same_font_captions_end_at_the_body_size(self):
        """Silberbush (ACS): one font, captions at x-height 3.78, body 4.20."""
        body = TextStyle("AdvOT2e364b11", 4.20)
        caption = TextStyle("AdvOT2e364b11", 3.78)
        typography = PageTypography(glyphs=[], body=body)
        assert typography.is_body_text(TextStyle("AdvOT2e364b11", 4.20), caption)
        assert not typography.is_body_text(TextStyle("AdvOT2e364b11", 3.77), caption)

    def test_a_legend_in_the_body_font_but_smaller_is_not_body_text(self):
        """Cell Reports (pilot PMC9396529): bold title line, legend in the
        body's own font at x-height 3.76, body at 4.48. The legend is caption."""
        body = TextStyle("AdvPSA183", 4.48)
        title = TextStyle("AdvPSA189", 4.06)
        typography = PageTypography(glyphs=[], body=body)
        assert not typography.is_body_text(TextStyle("AdvPSA183", 3.76), title)
        assert typography.is_body_text(TextStyle("AdvPSA183", 4.56), title)
        assert typography.is_body_text(TextStyle("AdvPSA183", 0.0), title)  # size unknown

    def test_unknown_styles_never_end_a_caption(self):
        typography = PageTypography(glyphs=[], body=self.TIMES)
        assert not typography.is_body_text(None, self.HELVETICA)
        assert not typography.is_body_text(self.TIMES, None)


class TestStyleIn:
    def test_the_dominant_font_inside_a_rect(self):
        glyphs = (_glyphs("Times-Roman", 4.4, 10, y=100.0)
                  + _glyphs("Helvetica", 4.3, 30, y=200.0))
        typography = PageTypography(glyphs=glyphs, body=TextStyle("Times-Roman", 4.4))
        assert typography.style_in((0.0, 195.0, 100.0, 205.0)).font == "Helvetica"
        assert typography.style_in((0.0, 300.0, 100.0, 310.0)) is None

    def test_dominant_style_of_nothing_is_none(self):
        assert dominant_style([]) is None


needs_corpus = pytest.mark.skipif(not CORPUS.is_dir(), reason="sample corpus not on this machine")


@needs_corpus
class TestTheCorpus:
    """Measured 2026-11-13; identical under pypdfium2 4.30 and 5.15.2."""

    @pytest.mark.parametrize("stem, font, x_height", [
        ("An NMR", "AdvPSA88A", 4.66),
        ("Collagen structure", "Times-Roman", 4.44),
        ("Hydration-induced", "MinionPro-Regular", 4.23),
        ("JRuberti", "AdvPSA5B8", 4.15),
        ("Silberbush", "AdvOT2e364b11", 4.20),
    ])
    def test_the_body_style_of_each_journal(self, stem, font, x_height):
        from src.processing.pdf import get_backend
        pdf = next(CORPUS.glob(stem + "*.pdf"))
        with get_backend().open(pdf) as doc:
            body = document_body_style([doc.page(i) for i in range(doc.page_count)])
        assert body.font.endswith(font)
        assert body.x_height == pytest.approx(x_height, abs=0.06)
