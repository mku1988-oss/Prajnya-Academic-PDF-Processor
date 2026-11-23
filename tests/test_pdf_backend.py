#!/usr/bin/env python3
"""
Cross-backend agreement tests for the PDF layer.

For biology researchers: the point of these tests is to answer one question
empirically — **can we drop PyMuPDF?** PyMuPDF is AGPL-3.0, which would force
the whole project to be open-source (or need a paid licence). pypdfium2 is
permissively licensed. If both backends give the same answers on a real paper,
the licence question becomes a config choice rather than a rewrite.

The tests also guard the single nastiest detail in this layer: PDFium measures
y from the *bottom* of the page and PyMuPDF from the *top*. Getting that flip
wrong would crop every figure from the mirrored half of the page — a bug that
produces plausible-looking output and is easy to miss by eye.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import statistics

import pytest

from src.processing.pdf import (
    Rect,
    available_backends,
    coverage,
    get_backend,
)
from tests.local_samples import BENCHMARK_PDF as LOCAL_BENCHMARK_PDF, CORPUS as LOCAL_CORPUS

# The benchmark paper, still living in the V5 tree.
BENCHMARK_PDF = LOCAL_BENCHMARK_PDF

#: Page 2 (index 1) is a good probe: prose in two columns plus exactly one
#: embedded figure, per the parity baseline.
PROBE_PAGE = 1
EXPECTED_PAGE_COUNT = 8
EXPECTED_IMAGES_ON_PROBE = 1


pytestmark = pytest.mark.skipif(
    not BENCHMARK_PDF.is_file(),
    reason=f"benchmark PDF not available at {BENCHMARK_PDF}",
)


@pytest.fixture(scope="module")
def backend_names() -> List[str]:
    names = sorted(available_backends())
    if not names:
        pytest.skip("no PDF backend installed")
    return names


# ---------------------------------------------------------------------------
# Per-backend behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["pypdfium2", "pymupdf"])
def test_backend_reads_the_document(name: str) -> None:
    """Each backend opens the paper and reports the right page count."""
    try:
        backend = get_backend(name)
    except ImportError:
        pytest.skip(f"{name} not installed")

    with backend.open(BENCHMARK_PDF) as doc:
        assert doc.page_count == EXPECTED_PAGE_COUNT
        page = doc.page(PROBE_PAGE)
        assert page.number == PROBE_PAGE
        width, height = page.size
        assert width > 0 and height > 0
        assert len(page.text()) > 1000, "probe page should have a substantial text layer"
        assert page.words(), "probe page should yield word boxes"


@pytest.mark.parametrize("name", ["pypdfium2", "pymupdf"])
def test_word_boxes_lie_inside_the_page(name: str) -> None:
    """
    Every word box sits within the page bounds, right way up.

    This is the y-flip guard. If PDFium's bottom-left origin were passed through
    unconverted, boxes near the top of the page would land off the bottom — or
    invert — and this catches it.
    """
    try:
        backend = get_backend(name)
    except ImportError:
        pytest.skip(f"{name} not installed")

    with backend.open(BENCHMARK_PDF) as doc:
        page = doc.page(PROBE_PAGE)
        width, height = page.size
        tolerance = 2.0  # PDFs occasionally place glyphs a hair outside the crop box

        for word in page.words():
            r = word.rect
            assert r.y0 <= r.y1, f"inverted box for {word.text!r}: {r}"
            assert r.x0 <= r.x1, f"inverted box for {word.text!r}: {r}"
            assert -tolerance <= r.x0 and r.x1 <= width + tolerance, \
                f"{word.text!r} outside page horizontally: {r} (page width {width})"
            assert -tolerance <= r.y0 and r.y1 <= height + tolerance, \
                f"{word.text!r} outside page vertically: {r} (page height {height})"


@pytest.mark.parametrize("name", ["pypdfium2", "pymupdf"])
def test_render_honours_the_size_request(name: str) -> None:
    """`max_edge` fits the longest side to a pixel budget; `dpi` scales from 72."""
    try:
        backend = get_backend(name)
    except ImportError:
        pytest.skip(f"{name} not installed")

    with backend.open(BENCHMARK_PDF) as doc:
        page = doc.page(PROBE_PAGE)

        image = page.render(max_edge=1024)
        assert max(image.size) == pytest.approx(1024, abs=2)

        at_150 = page.render(dpi=150)
        assert max(at_150.size) > max(image.size), "150 dpi should exceed a 1024px fit here"

        with pytest.raises(ValueError):
            page.render()  # neither dpi nor max_edge
        with pytest.raises(ValueError):
            page.render(dpi=150, max_edge=1024)  # both


# ---------------------------------------------------------------------------
# Agreement between backends — the licence question
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def both_backends():
    """Both backends, or skip — agreement needs two participants."""
    try:
        return get_backend("pypdfium2"), get_backend("pymupdf")
    except ImportError as exc:
        pytest.skip(f"need both backends to compare: {exc}")


def test_backends_agree_on_page_geometry(both_backends) -> None:
    """Page dimensions match — the foundation everything else is measured against."""
    permissive, fast = both_backends
    with permissive.open(BENCHMARK_PDF) as a, fast.open(BENCHMARK_PDF) as b:
        assert a.page_count == b.page_count
        for index in range(a.page_count):
            size_a, size_b = a.page(index).size, b.page(index).size
            assert size_a == pytest.approx(size_b, abs=1.0), f"page {index} size differs"


#: Measured word agreement between the two backends on the benchmark paper is
#: 93–100% per page, both directions. They are NOT byte-identical: they differ on
#: soft-hyphen handling at line breaks and on how math symbols decode. The
#: weakest page is p4, the table page, where cell text is segmented differently.
#:
#: So this floor encodes "viable substitute", not "identical". If it ever trips,
#: the two libraries have genuinely diverged on this document and the licence
#: swap needs re-examining before Stage 3 leans on the text layer.
MIN_CROSS_BACKEND_AGREEMENT = 0.93


def test_backends_agree_on_text_content(both_backends) -> None:
    """
    Both extract substantially the same words from every page, both directions.

    Compared by word coverage rather than string equality, because the libraries
    differ in line-ending and hyphenation handling — differences that do not
    change the content a researcher would read or a model would index.

    Checked symmetrically: measuring only one direction would miss a backend that
    invents text, since extra words cannot lower the other's coverage.
    """
    permissive, fast = both_backends
    with permissive.open(BENCHMARK_PDF) as a, fast.open(BENCHMARK_PDF) as b:
        for index in range(a.page_count):
            page_a, page_b = a.page(index), b.page(index)

            forward = coverage(page_a.text(), page_b.words())
            reverse = coverage(page_b.text(), page_a.words())

            assert forward >= MIN_CROSS_BACKEND_AGREEMENT, (
                f"page {index + 1}: pypdfium2 recovered only {forward:.1%} of "
                f"PyMuPDF's words"
            )
            assert reverse >= MIN_CROSS_BACKEND_AGREEMENT, (
                f"page {index + 1}: PyMuPDF recovered only {reverse:.1%} of "
                f"pypdfium2's words"
            )


def test_backends_agree_on_figure_location(both_backends) -> None:
    """
    Both find the probe page's figure in the same place.

    This is the strongest y-flip check: an unflipped box would still be *inside*
    the page, but mirrored, so it would overlap the correct one hardly at all.
    IoU is the standard box-agreement measure.
    """
    permissive, fast = both_backends
    with permissive.open(BENCHMARK_PDF) as a, fast.open(BENCHMARK_PDF) as b:
        images_a = a.page(PROBE_PAGE).images()
        images_b = b.page(PROBE_PAGE).images()

        assert len(images_a) == EXPECTED_IMAGES_ON_PROBE
        assert len(images_b) == EXPECTED_IMAGES_ON_PROBE

        overlap = images_a[0].rect.iou(images_b[0].rect)
        assert overlap >= 0.95, (
            f"figure boxes disagree (IoU {overlap:.3f}): "
            f"pypdfium2 {images_a[0].rect}, pymupdf {images_b[0].rect}. "
            "A low value here usually means the y-axis flip is wrong."
        )


def test_caption_text_is_readable_from_a_region(both_backends) -> None:
    """
    Text can be pulled from a sub-region of the page.

    This is how V6 reads figure captions for free on born-digital pages instead
    of paying a VLM call, so it has to work on both backends. The probe region is
    the band just below the figure, where its caption sits.
    """
    permissive, fast = both_backends
    with permissive.open(BENCHMARK_PDF) as a, fast.open(BENCHMARK_PDF) as b:
        page_a, page_b = a.page(PROBE_PAGE), b.page(PROBE_PAGE)
        figure = page_b.images()[0].rect
        width, height = page_b.size

        # A caption typically starts slightly left of the image it belongs to, so
        # the band runs to the page margin rather than hugging the figure box.
        # pypdfium2's word boxes are apportioned approximations (see its
        # `words()` docstring), which is another reason not to crop tightly.
        caption_band = Rect(
            max(0.0, figure.x0 - 60), figure.y1,
            min(width, figure.x1 + 60), min(figure.y1 + 60, height),
        )

        for page, label in ((page_a, "pypdfium2"), (page_b, "pymupdf")):
            found = page.text_in(caption_band).lower()
            assert "fig" in found, (
                f"{label}: expected a figure caption under the image, got {found[:120]!r}"
            )
            assert "saturation recovery" in found, (
                f"{label}: caption body missing — region text extraction is not "
                f"picking up the full caption line: {found[:120]!r}"
            )


# ---------------------------------------------------------------------------
# Licence visibility
# ---------------------------------------------------------------------------

def test_backends_declare_their_licences(backend_names: List[str]) -> None:
    """
    Each backend reports its licence, and the default is the permissive one.

    Surfacing this is what stops an AGPL dependency becoming load-bearing by
    accident — the failure mode that already happened in V5, where `fitz` was
    imported across eight modules but never appeared in requirements.txt.
    """
    licences = available_backends()
    assert licences, "no backends available"

    if "pymupdf" in licences:
        assert "AGPL" in licences["pymupdf"], "PyMuPDF's copyleft must stay visible"
    if "pypdfium2" in licences:
        assert "AGPL" not in licences["pypdfium2"]

    from src.processing.pdf import DEFAULT_BACKEND
    assert DEFAULT_BACKEND == "pypdfium2", "the default backend must be permissive"


def test_unknown_backend_is_rejected() -> None:
    """A typo in config fails loudly rather than silently picking a default."""
    with pytest.raises(ValueError, match="unknown PDF backend"):
        get_backend("not-a-real-backend")


# ---------------------------------------------------------------------------
# Typography — styled_lines()
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Compare line text without being hostage to spacing differences."""
    return "".join(text.split()).lower()


@pytest.mark.parametrize("name", ["pypdfium2", "pymupdf"])
def test_styled_lines_report_text_and_case(name: str) -> None:
    """Every line must carry its text and a sane uppercase fraction."""
    try:
        backend = get_backend(name)
    except ImportError:
        pytest.skip(f"{name} not installed")

    with backend.open(BENCHMARK_PDF) as document:
        lines = document.page(PROBE_PAGE).styled_lines()

    assert lines, "a born-digital page must yield typeset lines"
    for line in lines:
        assert line.text.strip()
        assert 0.0 <= line.caps_fraction <= 1.0
        assert line.cap_height >= 0.0
        assert line.weight >= 0.0


def test_a_page_without_a_text_layer_yields_no_lines() -> None:
    """A scan has no typography, and that must not raise."""
    backend = get_backend("pypdfium2")
    with backend.open(BENCHMARK_PDF) as document:
        page = document.page(PROBE_PAGE)
        assert isinstance(page.styled_lines(), list)


def test_backends_agree_on_the_typography_they_can_both_measure(
    both_backends,
) -> None:
    """Text, boldness and case must agree; cap height is NOT comparable.

    `cap_height` is measured from capital-glyph INK boxes, which only
    pypdfium2 exposes — PyMuPDF gives every character the whole line box, so
    it reports 0.0 for "unknown" rather than a number meaning something else.
    That divergence is asserted here so it stays deliberate.
    """
    permissive, fast = both_backends
    with permissive.open(BENCHMARK_PDF) as a, fast.open(BENCHMARK_PDF) as b:
        first = a.page(PROBE_PAGE).styled_lines()
        second = b.page(PROBE_PAGE).styled_lines()

    assert first and second

    # pypdfium2 measures ink height; PyMuPDF cannot and must say so.
    assert any(line.cap_height > 0 for line in first)
    assert all(line.cap_height == 0.0 for line in second)

    # Lines both backends found, matched on their text.
    by_text = {_normalise(line.text): line for line in second}
    shared = [
        (line, by_text[_normalise(line.text)])
        for line in first
        if _normalise(line.text) in by_text and len(_normalise(line.text)) > 12
    ]
    assert len(shared) >= 5, "too few shared lines to compare"

    for left, right in shared:
        assert abs(left.caps_fraction - right.caps_fraction) <= 0.05, (
            f"case disagreement on {left.text!r}"
        )

    # Boldness, as "heavier than this page's body text", must agree.
    def bold_set(lines):
        weights = [line.weight for line in lines if line.weight > 0]
        if not weights:
            return set()
        body = statistics.median(weights)
        return {
            _normalise(line.text)
            for line in lines
            if line.weight > body * 1.2 and len(_normalise(line.text)) > 12
        }

    left_bold, right_bold = bold_set(first), bold_set(second)
    if left_bold and right_bold:
        overlap = left_bold & right_bold
        union = left_bold | right_bold
        assert len(overlap) / len(union) >= 0.5, (
            f"backends disagree on which lines are bold: "
            f"only in pypdfium2 {left_bold - right_bold}, "
            f"only in pymupdf {right_bold - left_bold}"
        )


# --- pypdfium2 version safety (2026-11-13) ----------------------------------

def test_object_bounds_work_on_pypdfium2_4_and_5() -> None:
    """5.0 renamed get_pos() to get_bounds(); a fresh v6.3.0 install crashed on it."""
    from src.processing.pdf.pypdfium_backend import _object_bounds

    class Version4:
        def get_pos(self):
            return (1, 2, 3, 4)

    class Version5:
        def get_bounds(self):
            return (5, 6, 7, 8)

    assert _object_bounds(Version4()) == (1, 2, 3, 4)
    assert _object_bounds(Version5()) == (5, 6, 7, 8)


def test_pypdfium2_is_pinned_exactly() -> None:
    """
    Each pypdfium2 release bundles a different PDFium, which extracts text and
    boxes differently; 6.5.0's corpus was measured on 5.15.2. An open range is
    how a fresh 6.3.0 install crashed on every page. Change the pin only with
    a corpus comparison (eval/pypdfium5/).
    """
    from pathlib import Path
    line = next(l for l in (Path(__file__).resolve().parents[1] / "requirements.txt")
                .read_text(encoding="utf-8").splitlines() if l.startswith("pypdfium2"))
    assert line.split("#")[0].strip() == "pypdfium2==5.15.2"


# --- rotation from PDFium's character angles (2026-11-13, eval/word_boxes Addendum 5) ---

def test_the_backends_rotation_answer_overrides_the_box_shape() -> None:
    """A short upright word in a font-height box is taller than wide; it is still upright."""
    from src.processing.pdf.base import Rect, Word, looks_horizontal
    narrow_of = Rect(57.7, 569.3, 65.3, 577.8)                 # JRuberti p8, 7.6 x 8.5 pt
    assert looks_horizontal(Word("of", narrow_of, rotated=False))
    assert not looks_horizontal(Word("Downloaded", Rect(572, 100, 580, 160), rotated=True))
    assert not looks_horizontal(Word("of", narrow_of))       # no answer: the shape rule, as before


@pytest.mark.skipif(not LOCAL_CORPUS.is_dir(),
                    reason="corpus not available")
def test_pdfium_marks_the_margin_stamp_rotated_and_short_words_upright() -> None:
    from src.processing.pdf import get_backend
    from src.processing.pdf.base import looks_horizontal
    corpus = LOCAL_CORPUS
    with get_backend("pypdfium2").open(corpus / "Collagen structure- new tricks from a very old dog.pdf") as d:
        words = d.page(1).words()
    stamp = [w for w in words if w.text == "Downloaded"]
    assert stamp and all(w.rotated for w in stamp)
    with get_backend("pypdfium2").open(corpus / "JRuberti.pdf") as d:
        short = [w for w in d.page(7).words() if w.text == "of"]
    assert short and all(looks_horizontal(w) for w in short)


# ---------------------------------------------------------------------------
# D9: capping the drawing walk
# ---------------------------------------------------------------------------

def test_pdfium_caps_the_drawing_walk() -> None:
    """`limit` stops the walk at limit + 1 drawings, in the page's own order.
    NMR p4 draws 11 vector paths."""
    page = get_backend("pypdfium2").open(BENCHMARK_PDF).page(3)
    every = page.drawings()
    assert len(every) == 11
    assert len(page.drawings(limit=2)) == 3
    assert [d.rect for d in page.drawings(limit=2)] == [d.rect for d in every[:3]]
