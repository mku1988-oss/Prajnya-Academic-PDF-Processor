#!/usr/bin/env python
"""Tests for the typographic heading pass.

The pass exists so a consumer can tell a top-level heading from a sub-heading
without the fonts, which never leave this repo. Its contract is that it emits
MEASUREMENTS and never a level, so these tests pin the measurements and the
"unknown means 0.0, never small" rule.
"""
from pathlib import Path

import pytest

from src.processing.pdf import get_backend
from src.processing.pdf.base import Rect, StyledLine
from src.processing.utils.heading_pass import (
    body_cap_height,
    body_font_weight,
    heading_candidates_for_page,
    is_heading_shaped,
)
from tests.local_samples import CORPUS as LOCAL_CORPUS

SAMPLES = LOCAL_CORPUS
TRICKS = SAMPLES / "Collagen structure- new tricks from a very old dog.pdf"
ASSEMBLY = SAMPLES / "CollagenFibrilAssemblyandFunction.pdf"


def _line(text, cap=6.0, weight=400.0, caps=0.0):
    return StyledLine(text=text, rect=Rect(0, 0, 100, 10),
                      cap_height=cap, weight=weight, caps_fraction=caps)


class TestWhatCountsAsHeadingShaped:
    def test_a_short_titled_line_qualifies(self):
        assert is_heading_shaped("Fibril diameter measurement")

    def test_running_prose_does_not(self):
        assert not is_heading_shaped(
            "The fibrils were measured. Their diameters varied widely."
        )

    def test_a_trailing_colon_is_still_a_heading(self):
        assert is_heading_shaped("HYDROXYPROLINE AND COLLAGEN STABILITY:")

    def test_a_stray_glyph_does_not_qualify(self):
        """Super/subscripts sit on their own baseline and form their own
        lines. Measured on the corpus: '4R', '(yJ', a lone 'H' -- each 100%
        caps, each would otherwise pass as a heading."""
        for fragment in ("4R", "(yJ", "H", "3.2", ""):
            assert not is_heading_shaped(fragment)

    def test_a_whole_paragraph_does_not_qualify(self):
        assert not is_heading_shaped("word " * 40)


class TestBodyMeasurements:
    def test_body_size_is_the_median_not_the_mean(self):
        """One display heading must not drag the body size upwards."""
        lines = [_line("a", cap=6.0)] * 9 + [_line("BIG", cap=40.0)]
        assert body_cap_height(lines) == 6.0

    def test_unmeasurable_lines_are_ignored_not_counted_as_zero(self):
        lines = [_line("a", cap=0.0), _line("b", cap=6.0), _line("c", cap=6.0)]
        assert body_cap_height(lines) == 6.0

    def test_a_page_with_no_measurements_reports_zero(self):
        assert body_cap_height([_line("a", cap=0.0)]) == 0.0
        assert body_font_weight([_line("a", weight=0.0)]) == 0.0


class TestAgainstRealPages:
    """Fixtures prove the code runs; only real PDFs prove it is wired up."""

    @pytest.fixture(scope="class")
    def backend(self):
        return get_backend("pypdfium2")

    def _candidates(self, backend, pdf, page_number):
        if not pdf.is_file():
            pytest.skip(f"sample PDF not available: {pdf}")
        with backend.open(pdf) as document:
            return heading_candidates_for_page(document.page(page_number - 1))

    def test_a_bold_heading_is_found_by_its_weight(self, backend):
        """Measured: 1.07x cap height but weight 650 against body 420. Size
        alone would miss it, which is why weight is emitted."""
        found = self._candidates(backend, TRICKS, 9)
        hit = [c for c in found if "HYDROXYPROLINE" in c["text"]]
        assert hit, "the bold section heading was not recorded"
        assert hit[0]["is_bold"]
        assert hit[0]["caps_fraction"] == 1.0

    def test_a_large_heading_is_found_by_its_size(self, backend):
        """Measured: 1.36x cap height with NO weight recorded at all. Weight
        alone would miss it, which is why size is emitted."""
        found = self._candidates(backend, ASSEMBLY, 29)
        hit = [c for c in found if c["text"].strip() == "REFERENCES"]
        assert hit, "the large section heading was not recorded"
        assert hit[0]["size_ratio"] >= 1.15
        assert hit[0]["weight_ratio"] == 0.0  # the file records no weights

    def test_body_prose_is_not_recorded_as_a_heading(self, backend):
        """The failure mode of the attempt this replaced: ~30 body lines per
        page recorded as candidates."""
        found = self._candidates(backend, TRICKS, 9)
        assert len(found) <= 6, [c["text"] for c in found]

    def test_a_zero_ratio_means_unknown_never_small(self, backend):
        """A consumer that reads 0.0 as 'tiny' would rank real headings last."""
        found = self._candidates(backend, ASSEMBLY, 29)
        assert all(c["weight_ratio"] == 0.0 for c in found)
        assert all(c["size_ratio"] > 0 for c in found)
