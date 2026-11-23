#!/usr/bin/env python3
"""
Tests for the CPU page classifier and its vector-drawing clusterer.

For biology researchers: the classifier is the decision that makes V6 fast. It
looks at a page with no model at all and answers "does this page actually need
the vision model?" A page it routes to TEXT_ONLY costs milliseconds instead of
~20 seconds. So the tests below pin down each of the five routing rules, and —
just as important — the filters that stop a page header or a table's ruling
lines from being mistaken for a figure.

Two groups:

- **Logic tests** run everywhere, on hand-built pages (`tests/fakes.py`).
- **End-to-end tests** need the benchmark PDF and check the classifier against
  what we know is on those pages: figures on p2-p6, tables on p4.
"""

from pathlib import Path

import pytest

from src.processing.pdf import get_backend
from src.processing.pdf.base import DrawingBlock, ImageBlock, Rect, Word
from src.processing.utils.page_classifier import (
    PageClassifier,
    PageType,
    extract_text_stats,
)
from src.processing.utils import page_classifier as page_classifier_module
from src.processing.utils.visual_clusterer import VisualRegionClusterer

from .fakes import FakePage, word_row
from tests.local_samples import BENCHMARK_PDF as LOCAL_BENCHMARK_PDF

BENCHMARK_PDF = LOCAL_BENCHMARK_PDF

needs_pdf = pytest.mark.skipif(
    not BENCHMARK_PDF.is_file(), reason="benchmark PDF unavailable"
)

#: Page geometry used throughout: US Letter, so the header zone ends at 79.2pt
#: and the footer zone starts at 712.8pt.
PAGE_SIZE = (612.0, 792.0)


@pytest.fixture
def classifier() -> PageClassifier:
    """A classifier on stock defaults — no YAML section present."""
    return PageClassifier({})


def figure_sized_image(x0=100.0, y0=200.0, x1=400.0, y1=500.0) -> ImageBlock:
    """An image comfortably past every size filter (300x300pt)."""
    return ImageBlock(rect=Rect(x0, y0, x1, y1))


def dense_text_stats() -> dict:
    """Statistics that score a full 1.0 text density."""
    return {'word_count': 400, 'avg_words_per_line': 12, 'char_count': 2500}


# ---------------------------------------------------------------------------
# Rule 1 — the first page is never classified
# ---------------------------------------------------------------------------

def test_first_page_is_always_routed_as_the_title_page(classifier):
    """Page 1 goes down the metadata ladder, so its content is not examined."""
    busy = FakePage(images=[figure_sized_image(), figure_sized_image(120.0, 520.0, 420.0, 700.0)])
    page_type, analysis = classifier.classify(busy, 0)
    assert page_type == PageType.TEXT_ONLY
    assert analysis['classification_reason'] == 'first_page_title'
    # Nothing was inspected — the visual signal is absent, not merely empty.
    assert 'visual_count' not in analysis


# ---------------------------------------------------------------------------
# Rule 2 — equations route to COMPLEX_VISUAL
# ---------------------------------------------------------------------------

def test_latex_syntax_on_a_line_is_detected(classifier):
    page = FakePage(words=word_row([r"\frac{T_{2}}{T_{1}}", r"\sum_{i}"], y=300.0))
    equations = classifier._detect_equation_regions(page)
    latex = [e for e in equations if e['method'] == 'latex_syntax']
    assert len(latex) == 1
    assert latex[0]['confidence'] >= 0.80


def test_unicode_math_symbols_are_detected(classifier):
    page = FakePage(words=word_row(["∀x", "∈", "∑", "∫"], y=300.0))
    methods = {e['method'] for e in classifier._detect_equation_regions(page)}
    assert 'unicode_math' in methods


def test_a_short_centered_line_is_a_low_confidence_equation_candidate(classifier):
    """
    Deliberately weak (0.40): a centered short line is just as likely to be a
    section heading. It needs company before it can route a page.
    """
    centered = word_row(["x", "=", "2y", "+", "1"], y=300.0, x=280.0)
    candidates = classifier._detect_equation_regions(FakePage(words=centered))
    centered_hits = [e for e in candidates if e['method'] == 'centered_text']
    assert len(centered_hits) == 1
    assert centered_hits[0]['confidence'] == 0.40


def test_table_cells_are_not_mistaken_for_centered_equations(classifier):
    """
    The port's near-miss, pinned. V5 required a whole text BLOCK to hold one
    line — really an isolation test. Without it, every short centered cell of
    a table scores 0.40, and the benchmark's two-table page racked up 32 false
    equations (12.8 total against a 1.5 threshold), routing to COMPLEX_VISUAL
    before the visual rule could run. A table row is several lines sharing one
    height, so isolation separates it from a display equation.
    """
    table = []
    for row, y in enumerate((300.0, 320.0, 340.0)):
        table.extend(word_row(["47.3"], y=y, x=240.0))     # centered column
        table.extend(word_row(["12.8"], y=y, x=360.0))     # neighbour column
    methods = {e['method'] for e in classifier._detect_equation_regions(FakePage(words=table))}
    assert 'centered_text' not in methods


def test_a_bare_centered_numeral_is_not_an_equation(classifier):
    """A lone page number sits centered and alone, but is five characters shy."""
    page = FakePage(words=word_row(["12"], y=300.0, x=300.0))
    methods = {e['method'] for e in classifier._detect_equation_regions(page)}
    assert 'centered_text' not in methods


def test_a_left_aligned_line_is_not_a_centered_candidate(classifier):
    prose = word_row(["The", "hydration", "of", "collagen"], y=300.0, x=72.0)
    methods = {e['method'] for e in classifier._detect_equation_regions(FakePage(words=prose))}
    assert 'centered_text' not in methods


def test_small_wide_embedded_image_reads_as_an_equation_image(classifier):
    """Typeset equations are often pasted in as small, wide rasters."""
    page = FakePage(images=[ImageBlock(rect=Rect(150.0, 300.0, 400.0, 340.0))])
    methods = {e['method'] for e in classifier._detect_equation_regions(page)}
    assert 'embedded_image' in methods


def test_equation_heavy_page_routes_to_complex_visual(classifier):
    """Two LaTeX lines at 0.80 each clear the 1.5 total-confidence threshold."""
    page = FakePage(words=[
        *word_row([r"\frac{a}{b}", r"\sqrt{c}"], y=300.0),
        *word_row([r"\int_0", r"\sum_{n}"], y=340.0),
    ])
    page_type, analysis = classifier.classify(page, 3)
    assert page_type == PageType.COMPLEX_VISUAL
    assert analysis['equation_total_confidence'] > 1.5


def test_equation_thresholds_come_from_config():
    """A profile YAML must be able to loosen or tighten the equation gate."""
    strict = PageClassifier({'hybrid_processing': {'page_classifier': {
        'equation': {'total_confidence_threshold': 99.0, 'high_confidence_count': 99},
    }}})
    page = FakePage(words=[
        *word_row([r"\frac{a}{b}", r"\sqrt{c}"], y=300.0),
        *word_row([r"\int_0", r"\sum_{n}"], y=340.0),
    ])
    page_type, _ = strict.classify(page, 3)
    assert page_type != PageType.COMPLEX_VISUAL


# ---------------------------------------------------------------------------
# Rule 3 — visuals route to SIMPLE_VISUAL
# ---------------------------------------------------------------------------

def test_two_figures_route_to_simple_visual(classifier):
    page = FakePage(images=[
        figure_sized_image(100.0, 150.0, 400.0, 400.0),
        figure_sized_image(100.0, 430.0, 400.0, 680.0),
    ])
    page_type, analysis = classifier.classify(page, 3)
    assert page_type == PageType.SIMPLE_VISUAL
    assert analysis['visual_count'] == 2


def test_visual_threshold_comes_from_config():
    lenient = PageClassifier({'hybrid_processing': {'page_classifier': {
        'visual_threshold': 1,
    }}})
    page = FakePage(images=[figure_sized_image()])
    page_type, _ = lenient.classify(page, 3)
    assert page_type == PageType.SIMPLE_VISUAL


def test_header_and_footer_graphics_are_not_figures(classifier):
    """Journal logos and page furniture sit outside the body band."""
    page = FakePage(images=[
        figure_sized_image(100.0, 10.0, 400.0, 70.0),     # header
        figure_sized_image(100.0, 730.0, 400.0, 785.0),   # footer
    ])
    assert classifier._detect_visual_regions(page) == []


def test_tiny_and_thin_graphics_are_not_figures(classifier):
    """Bullet glyphs and horizontal rules must not add up to a "figure" page."""
    page = FakePage(images=[
        ImageBlock(rect=Rect(100.0, 300.0, 120.0, 320.0)),   # 20x20pt: too small
        ImageBlock(rect=Rect(72.0, 400.0, 540.0, 404.0)),    # a rule: too thin
    ])
    assert classifier._detect_visual_regions(page) == []


def test_a_real_figure_survives_every_filter(classifier):
    regions = classifier._detect_visual_regions(FakePage(images=[figure_sized_image()]))
    assert len(regions) == 1
    assert regions[0]['type'] == 'image'


# ---------------------------------------------------------------------------
# Rule 4 / 5 — text density and the fallback
# ---------------------------------------------------------------------------

def test_dense_prose_with_no_visuals_routes_to_text_only(classifier):
    """The whole point: this page never touches the GPU."""
    page = FakePage(words=word_row(["hydration", "of", "collagen", "fibrils"], y=300.0))
    page_type, analysis = classifier.classify(page, 3, text_stats=dense_text_stats())
    assert page_type == PageType.TEXT_ONLY
    assert analysis['text_density'] >= 0.85


def test_sparse_text_falls_back_to_the_vlm(classifier):
    """Caption-like pages are ambiguous, so the safe default is a VLM read."""
    sparse = {'word_count': 40, 'avg_words_per_line': 3, 'char_count': 200}
    page_type, analysis = classifier.classify(FakePage(), 3, text_stats=sparse)
    assert page_type == PageType.SIMPLE_VISUAL
    assert analysis['classification_reason'] == 'mixed_content_fallback'


def test_without_text_stats_the_page_cannot_be_routed_to_text_only(classifier):
    """Rule 4 needs the statistics; absent them the safe fallback must win."""
    page_type, _ = classifier.classify(FakePage(), 3)
    assert page_type == PageType.SIMPLE_VISUAL


def test_dense_text_beside_one_figure_still_needs_the_vlm(classifier):
    """
    One figure is below the visual threshold but not nothing — Rule 4 requires
    zero visuals, so the page falls through to the fallback rather than
    skipping the figure.
    """
    page = FakePage(images=[figure_sized_image()])
    page_type, _ = classifier.classify(page, 3, text_stats=dense_text_stats())
    assert page_type == PageType.SIMPLE_VISUAL


def test_text_density_is_a_weighted_blend(classifier):
    """Word count carries 0.4; lines and characters 0.3 each."""
    assert classifier._calculate_text_density(dense_text_stats()) == pytest.approx(1.0)
    assert classifier._calculate_text_density(
        {'word_count': 0, 'avg_words_per_line': 0, 'char_count': 0}
    ) == 0.0
    # Words alone, at the saturation point, contribute exactly their weight.
    assert classifier._calculate_text_density(
        {'word_count': 300, 'avg_words_per_line': 0, 'char_count': 0}
    ) == pytest.approx(0.4)


def test_extract_text_stats_reports_words_lines_and_characters():
    page = FakePage(text="one two three\nfour five six\n\nseven eight nine")
    stats = extract_text_stats(page)
    assert stats['word_count'] == 9
    assert stats['line_count'] == 3
    assert stats['avg_words_per_line'] == 3.0
    assert stats['char_count'] == len(page.text())


# ---------------------------------------------------------------------------
# VisualRegionClusterer
# ---------------------------------------------------------------------------

def test_scattered_strokes_of_one_chart_merge_into_one_region():
    """
    A born-digital chart is hundreds of separate strokes. Clustering is what
    turns them back into "there is a figure here".
    """
    strokes = [DrawingBlock(rect=Rect(100.0 + i * 10, 300.0, 108.0 + i * 10, 400.0)).rect
               for i in range(12)]
    clusters = VisualRegionClusterer().cluster_drawings(strokes)
    assert len(clusters) == 1
    assert clusters[0].component_count == 12
    assert clusters[0].rect.x0 == pytest.approx(100.0)
    assert clusters[0].rect.x1 == pytest.approx(218.0)


def test_two_distant_charts_stay_separate():
    left = [Rect(100.0 + i * 8, 300.0, 106.0 + i * 8, 380.0) for i in range(6)]
    right = [Rect(400.0 + i * 8, 300.0, 406.0 + i * 8, 380.0) for i in range(6)]
    assert len(VisualRegionClusterer().cluster_drawings(left + right)) == 2


def test_an_isolated_stroke_is_noise_not_a_figure():
    """min_samples=2 — one lone rule under a heading is never a figure."""
    assert VisualRegionClusterer().cluster_drawings([Rect(72.0, 300.0, 540.0, 302.0)]) == []


def test_clustering_an_empty_page_returns_nothing():
    assert VisualRegionClusterer().cluster_drawings([]) == []


def test_clustered_drawings_become_visual_regions(classifier):
    """The clusterer's output has to survive the classifier's size filters."""
    strokes = [DrawingBlock(rect=Rect(100.0 + i * 20, 300.0, 118.0 + i * 20, 500.0))
               for i in range(10)]
    regions = classifier._detect_visual_regions(FakePage(drawings=strokes))
    assert len(regions) == 1
    assert regions[0]['type'] == 'drawing_cluster'


# ---------------------------------------------------------------------------
# End to end on the benchmark paper
# ---------------------------------------------------------------------------

@needs_pdf
def test_benchmark_figure_pages_are_not_routed_past_the_vlm(classifier):
    """
    Pages 2-6 carry the paper's five figures. Routing any of them to TEXT_ONLY
    would silently lose a figure, so this is a data-loss guard.
    """
    with get_backend().open(BENCHMARK_PDF) as document:
        for page_index in range(1, 6):
            page = document.page(page_index)
            page_type, analysis = classifier.classify(
                page, page_index, text_stats=extract_text_stats(page)
            )
            assert page_type != PageType.TEXT_ONLY, (
                f"page {page_index + 1} would skip the VLM: {analysis['classification_reason']}"
            )


#: What V5's classifier routes each benchmark page to, measured directly by
#: running V5's `PageClassifier` on this PDF. This is the port's parity gate:
#: the V6 classifier must reach the same routing decision on every page, or the
#: pipeline it feeds behaves differently from the baseline the harness was
#: recorded against.
#:
#: PAGE 7 IS A DELIBERATE, DOCUMENTED DEVIATION. V5 routed it SIMPLE_VISUAL —
#: but only because two section headings ("3. Results"-style) scored as
#: equations, and Rule 4 refuses TEXT_ONLY on any page with an equation. Page 7
#: is prose-only (98.7% word coverage in the content-drop corpus, no tables, no
#: figures), so once the false positives are gone it satisfies V5's OWN Rule 4:
#: dense text, zero visuals, zero equations. V6 therefore reads it from the
#: text layer and skips the VLM.
#:
#: This is a real behavioural change and it arrives early — it is a Stage 3
#: outcome reached by fixing a Stage-2-era bug. It is recorded here rather than
#: suppressed, because reintroducing V5's false positives to preserve the old
#: routing would be preserving an accident. Revisit when the full parity
#: harness can run end to end.
V5_BENCHMARK_ROUTING = {
    1: PageType.TEXT_ONLY,      # title page (Rule 1)
    2: PageType.SIMPLE_VISUAL,
    3: PageType.SIMPLE_VISUAL,
    4: PageType.SIMPLE_VISUAL,  # the two-table page
    5: PageType.SIMPLE_VISUAL,
    6: PageType.SIMPLE_VISUAL,
    7: PageType.SIMPLE_VISUAL,
    8: PageType.TEXT_ONLY,      # references: dense prose, no visuals
}

#: Pages where V6 knowingly differs from the table above, with the reason.
DOCUMENTED_DEVIATIONS = {
    7: (PageType.TEXT_ONLY, "V5's false-positive heading detections blocked Rule 4"),
}


@needs_pdf
@pytest.mark.parametrize("page_number,v5_routing", sorted(V5_BENCHMARK_ROUTING.items()))
def test_benchmark_routing_matches_v5_page_for_page(classifier, page_number, v5_routing):
    """The port's parity gate — see `V5_BENCHMARK_ROUTING`."""
    expected, _ = DOCUMENTED_DEVIATIONS.get(page_number, (v5_routing, ""))
    with get_backend().open(BENCHMARK_PDF) as document:
        page = document.page(page_number - 1)
        page_type, analysis = classifier.classify(
            page, page_number - 1, text_stats=extract_text_stats(page)
        )
        assert page_type == expected, (
            f"page {page_number}: V6 says {page_type.value}, expected {expected.value} "
            f"(V5 said {v5_routing.value}) — {analysis['classification_reason']}"
        )


@needs_pdf
def test_page_seven_skips_the_vlm_and_that_is_the_known_deviation(classifier):
    """
    Pin the deviation itself, so it cannot change silently in either direction.

    Page 7 is prose only. Routing it TEXT_ONLY is what makes V6 faster than V5
    on this page — no vision call at all — and it is safe precisely because
    there is nothing on the page a text layer cannot carry.
    """
    with get_backend().open(BENCHMARK_PDF) as document:
        page = document.page(6)
        page_type, analysis = classifier.classify(page, 6, text_stats=extract_text_stats(page))

    assert page_type == PageType.TEXT_ONLY
    assert analysis['equation_regions'] == [], "the false positives are back"
    assert analysis['visual_regions'] == [], "a figure here would make this unsafe"
    assert analysis['text_density'] >= 0.85


@needs_pdf
def test_benchmark_table_page_reports_visual_regions(classifier):
    """
    p4 holds the two tables the repair path recovers.

    The visual signal matters beyond routing: the strategy passes
    `analysis['visual_regions']` on as `has_visuals`, which is one of the three
    conditions for running the layout detector on a page. An empty list here
    would silently skip figure extraction.
    """
    with get_backend().open(BENCHMARK_PDF) as document:
        page = document.page(3)
        _, analysis = classifier.classify(page, 3, text_stats=extract_text_stats(page))
        assert analysis.get('visual_regions'), "no visual signal on the two-table page"


@needs_pdf
def test_benchmark_body_pages_report_no_spurious_equations(classifier):
    """
    None of this paper's body pages carries a display equation, so the
    equation signal should be quiet. V5 scored a handful of section headings
    here ("3. Results", "4. Discussion"); the column-margin guard removes them,
    which is a strict improvement rather than a parity break — the routing
    decision above is unchanged either way.
    """
    with get_backend().open(BENCHMARK_PDF) as document:
        for page_index in range(1, 8):
            page = document.page(page_index)
            _, analysis = classifier.classify(
                page, page_index, text_stats=extract_text_stats(page)
            )
            assert analysis['equation_total_confidence'] < 1.5, (
                f"page {page_index + 1} would misroute to COMPLEX_VISUAL: "
                f"{[e['text'] for e in analysis['equation_regions']]}"
            )


@needs_pdf
def test_benchmark_pages_yield_real_text_statistics():
    """A body page of this paper is dense prose; the stats must reflect that."""
    with get_backend().open(BENCHMARK_PDF) as document:
        stats = extract_text_stats(document.page(6))
        assert stats['word_count'] > 200
        assert stats['avg_words_per_line'] > 3


# ---------------------------------------------------------------------------
# D9: a page that draws too many strokes to cluster
# ---------------------------------------------------------------------------

def strokes(count: int) -> list:
    """`count` tiny drawings packed into one figure area, like a network graph."""
    return [DrawingBlock(rect=Rect(100.0 + (i % 50) * 2, 200.0 + (i // 50) * 2,
                                   101.5 + (i % 50) * 2, 201.5 + (i // 50) * 2))
            for i in range(count)]


class TestDenseDrawingPages:
    """PMC10904388 p4 draws a network graph as 580,957 strokes. Clustering them
    needed ~850 GB, raised MemoryError, and the page was written empty. Above
    DENSE_DRAWING_LIMIT the page is one dense figure region instead; below it,
    nothing changes. The limit is shrunk to 5 here so the tests stay small."""

    @pytest.fixture(autouse=True)
    def small_limit(self, monkeypatch):
        monkeypatch.setattr(page_classifier_module, "DENSE_DRAWING_LIMIT", 5)

    def test_too_many_strokes_make_one_dense_region_without_clustering(self, classifier, monkeypatch):
        def must_not_cluster(_rects):
            raise AssertionError("a dense page must never reach the clusterer")
        monkeypatch.setattr(classifier.visual_clusterer, "cluster_drawings", must_not_cluster)
        regions, _ = classifier._scan_visuals(FakePage(drawings=strokes(40)))
        assert [r["type"] for r in regions] == ["dense_drawings"]

    def test_the_walk_is_always_capped(self, classifier):
        """Every page asks for at most limit + 1 drawings, so even a figure
        hidden inside one embedded form (one object to PDFium's count) is
        never listed in full."""
        asked = []

        class CountingPage(FakePage):
            def drawings(self, limit=None):
                asked.append(limit)
                return super().drawings(limit)

        classifier._scan_visuals(CountingPage(drawings=strokes(40)))
        classifier._scan_visuals(CountingPage(drawings=strokes(3)))
        assert asked == [5, 5]

    def test_a_page_under_the_limit_is_clustered_as_before(self, classifier):
        regions, _ = classifier._scan_visuals(FakePage(drawings=strokes(4)))
        assert all(r["type"] != "dense_drawings" for r in regions)

    def test_a_dense_page_is_a_visual_page(self, classifier):
        page = FakePage(drawings=strokes(40), words=word_row(["Fig."], y=700.0))
        page_type, analysis = classifier.classify(page, 3, text_stats=dense_text_stats())
        assert page_type != PageType.TEXT_ONLY
        assert analysis["visual_regions"]
