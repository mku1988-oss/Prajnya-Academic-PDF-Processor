#!/usr/bin/env python3
"""
Tests for the pypdfium2 backend's word baselines and glyph styles.

For biology researchers: a word's box top depends on its font (a degree sign
or a symbol-font minus starts higher), but every word on a printed line sits
on the same baseline. The figure-caption reader groups lines by baseline, and
tells captions from body text by font. These tests pin both on a real page.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.local_samples import CORPUS as LOCAL_CORPUS

CORPUS = LOCAL_CORPUS
COLLAGEN = CORPUS / "Collagen structure- new tricks from a very old dog.pdf"

pytestmark = pytest.mark.skipif(not COLLAGEN.is_file(),
                                reason="sample corpus not on this machine")


def _words_ending(page_index: int, last: str, following: str, count: int):
    """The `count` words up to and including `last`, where `following` comes next."""
    from src.processing.pdf import get_backend
    with get_backend("pypdfium2").open(COLLAGEN) as doc:
        words = doc.page(page_index).words()
    end = next(i for i, w in enumerate(words[:-1])
               if w.text == last and words[i + 1].text == following)
    return words[end - count + 1:end + 1]


class TestBaselines:
    def test_a_line_shares_one_baseline_whatever_its_box_tops(self):
        """p4: "unit twist κ =− 106◦.": tops spread over 2 pt, one baseline."""
        line = _words_ending(3, "106◦.", "This", 4)
        assert len({round(w.baseline, 2) for w in line}) == 1
        assert max(w.rect.y0 for w in line) - min(w.rect.y0 for w in line) > 2.0

    def test_a_subscripted_word_sits_on_its_first_letters_baseline(self):
        """NMR p6 "Rbp": two subscript letters of three must not move the word."""
        from src.processing.pdf import get_backend
        pdf = next(CORPUS.glob("An NMR*.pdf"))
        with get_backend("pypdfium2").open(pdf) as doc:
            words = doc.page(5).words()
        index = next(i for i, w in enumerate(words) if w.text == "Rbp")
        assert words[index].baseline == pytest.approx(words[index + 1].baseline)


class TestGlyphStyles:
    def test_caption_and_body_fonts_are_read(self):
        """p2: Figure 1's caption is Helvetica-Condensed; the article is Times."""
        from src.processing.pdf import get_backend
        with get_backend("pypdfium2").open(COLLAGEN) as doc:
            fonts = {g.font for g in doc.page(1).glyph_styles()}
        assert any(f.startswith("Helvetica-Condensed") for f in fonts)
        assert "Times-Roman" in fonts

    def test_x_height_is_measured_only_on_x_height_letters(self):
        from src.processing.pdf import get_backend
        with get_backend("pypdfium2").open(COLLAGEN) as doc:
            glyphs = doc.page(1).glyph_styles()
        measured = [g.x_height for g in glyphs if g.x_height]
        assert measured and 3.0 < sorted(measured)[len(measured) // 2] < 5.5
        assert any(g.x_height is None for g in glyphs)
