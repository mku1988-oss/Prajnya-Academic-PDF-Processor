#!/usr/bin/env python3
"""
Text with rectangles drawn on it is not a figure.

A reference list draws a small rectangle around every "CrossRef" and "PubMed"
link. Cluster a hundred of those and you get something shaped exactly like a
figure — which routed seven bibliography pages of one paper to the GPU, and
then to a second model to double-check. Pure text, with a perfect text layer,
costing ~20s a page to read and ~20s a page to verify.

**The test is DENSITY, not presence.** Real figures contain text: panel
letters, axis labels, scale bars, legends. A rule that discarded any region
containing words would throw away the figures this whole effort is trying to
recover.

MEASURED over labelled regions from the sample corpus — the populations do not
overlap, and are not close:

    decoration   32 link clusters      51.3% - 71.5% of area covered by words
    real figures 43 regions             0.0% -  8.3%   (several WITH text in
                                                        them: Hydration p5-p10
                                                        panel labels, JRuberti
                                                        confocal scale bars)

The default cutoff of 30% sits in the empty band between 8.3% and 51.3%, so it
is not a delicate number — anything from ~15% to ~45% behaves identically on
this corpus.

IMPACT, measured across 451 body pages: 7 pages re-routed to the free text
path, **0 figures put at risk**, ~2.3 min of GPU saved (plus the escalation
calls those pages were also triggering).

**BUT IT PUT A TABLE AT RISK, AND THE FIGURE MEASUREMENT COULD NOT SEE THAT.**
A table is text-dense *by construction* — dense text in ruled cells is what a
table IS. On Silberbush p4 the filter discarded a region that was exactly the
box the detector calls `table` ("Table 1. Resistivity and Contact Resistance
Parameters…"), and the page went from SIMPLE_VISUAL (read, detected, table
extracted as a retrievable object) to TEXT_ONLY (free, table gone). That is a
regression this filter introduced, invisible to a figure count.

The fix is not a lower threshold — the table measured like decoration because
it *is* dense. It is `body_pipeline._detections_the_filter_hid`: when the
filter empties a page, the CPU detector gets one look before the page is let
go. Measured over the same corpus (476 body pages): the filter empties 10
pages, the detector confirms 9 of them and overturns 1 — the table above — for
5.3s of CPU across the whole corpus.
"""

import pytest

from src.processing.pdf.base import DrawingBlock, ImageBlock, Rect, Word
from src.processing.utils.page_classifier import PageClassifier, PageType

from .fakes import FakePage

#: The measured populations, as fractions of a region's area.
DECORATION_COVERAGE = 0.62      # median of 32 link clusters
FIGURE_COVERAGE = 0.083         # the MOST text-covered real figure measured


@pytest.fixture
def classifier() -> PageClassifier:
    return PageClassifier({})


def words_covering(box: Rect, fraction: float, rows: int = 10) -> list:
    """Word boxes tiling `fraction` of `box`, as a text layer would report."""
    width, height = box.x1 - box.x0, box.y1 - box.y0
    row_height = height / rows
    word_height = row_height * 0.6
    word_width = width * fraction / 0.6            # so area works out to fraction

    words = []
    for row in range(rows):
        top = box.y0 + row * row_height
        words.append(Word(text="reference",
                          rect=Rect(box.x0, top,
                                    min(box.x0 + word_width, box.x1),
                                    top + word_height)))
    return words


def clustered_drawings(box: Rect, count: int = 12) -> list:
    """Small strokes inside `box` — what link rectangles look like."""
    width = (box.x1 - box.x0) / count
    return [DrawingBlock(rect=Rect(box.x0 + i * width, box.y0,
                                   box.x0 + (i + 1) * width, box.y1))
            for i in range(count)]


class TestDecorationIsDiscarded:
    def test_a_cluster_sitting_on_body_text_is_not_a_figure(self, classifier):
        """The references page, in miniature."""
        box = Rect(72.0, 200.0, 400.0, 500.0)
        page = FakePage(drawings=clustered_drawings(box),
                        words=words_covering(box, DECORATION_COVERAGE))

        assert classifier._detect_visual_regions(page) == []

    def test_such_a_page_routes_to_the_free_text_path(self, classifier):
        """
        The point of the whole exercise: no VLM read, no escalation, no cost.
        """
        box = Rect(72.0, 200.0, 400.0, 500.0)
        page = FakePage(drawings=clustered_drawings(box),
                        words=words_covering(box, DECORATION_COVERAGE))
        dense = {"word_count": 1470, "avg_words_per_line": 12, "char_count": 10227}

        page_type, analysis = classifier.classify(page, 20, text_stats=dense)

        assert page_type is PageType.TEXT_ONLY
        assert analysis["visual_regions"] == []


class TestRealFiguresSurvive:
    def test_a_figure_with_labels_inside_it_is_kept(self, classifier):
        """
        THE OBJECTION THAT SHAPED THIS RULE. Figures contain text — panel
        letters, axis labels, legends. At the most text-heavy real figure
        measured (8.3%), the region must survive.
        """
        box = Rect(72.0, 200.0, 400.0, 500.0)
        page = FakePage(drawings=clustered_drawings(box),
                        words=words_covering(box, FIGURE_COVERAGE))

        regions = classifier._detect_visual_regions(page)
        assert len(regions) == 1
        assert regions[0]["type"] == "drawing_cluster"

    @pytest.mark.parametrize("coverage", [0.0, 0.01, 0.05, 0.083, 0.15, 0.25])
    def test_every_plausible_figure_density_is_kept(self, classifier, coverage):
        box = Rect(72.0, 200.0, 400.0, 500.0)
        page = FakePage(drawings=clustered_drawings(box),
                        words=words_covering(box, coverage))
        assert len(classifier._detect_visual_regions(page)) == 1

    def test_an_embedded_image_is_never_discarded_by_this_rule(self, classifier):
        """
        A raster image is a picture by construction. Only CLUSTERED STROKES can
        be link rectangles pretending to be a figure, so the rule applies there
        and nowhere else — a photograph with a dense caption overlaid must not
        vanish.
        """
        box = Rect(72.0, 200.0, 400.0, 500.0)
        page = FakePage(images=[ImageBlock(rect=box)],
                        words=words_covering(box, DECORATION_COVERAGE))

        regions = classifier._detect_visual_regions(page)
        assert len(regions) == 1
        assert regions[0]["type"] == "image"


class TestTheThreshold:
    def test_the_default_sits_between_the_measured_populations(self, classifier):
        assert FIGURE_COVERAGE < classifier.text_coverage_max < 0.513

    def test_it_is_configurable(self):
        strict = PageClassifier({"hybrid_processing": {"page_classifier": {
            "text_coverage_max": 0.05}}})
        assert strict.text_coverage_max == 0.05

    def test_a_stricter_cutoff_discards_what_the_default_keeps(self):
        """Proves the threshold is what decides, not some other filter."""
        box = Rect(72.0, 200.0, 400.0, 500.0)
        page = FakePage(drawings=clustered_drawings(box),
                        words=words_covering(box, 0.20))

        default = PageClassifier({})
        strict = PageClassifier({"hybrid_processing": {"page_classifier": {
            "text_coverage_max": 0.05}}})

        assert len(default._detect_visual_regions(page)) == 1
        assert strict._detect_visual_regions(page) == []

    def test_disabling_it_restores_the_old_behaviour(self):
        box = Rect(72.0, 200.0, 400.0, 500.0)
        page = FakePage(drawings=clustered_drawings(box),
                        words=words_covering(box, DECORATION_COVERAGE))
        off = PageClassifier({"hybrid_processing": {"page_classifier": {
            "text_coverage_max": 1.01}}})

        assert len(off._detect_visual_regions(page)) == 1


class TestFailureLeansTowardsKeeping:
    def test_a_page_that_cannot_report_words_keeps_its_regions(self, classifier):
        """Dropping a real figure is permanent; an extra VLM read is not."""
        box = Rect(72.0, 200.0, 400.0, 500.0)

        class NoWords(FakePage):
            def words(self):
                raise RuntimeError("no text layer")

        page = NoWords(drawings=clustered_drawings(box))
        assert len(page.drawings()) > 0
        assert len(classifier._detect_visual_regions(page)) == 1

    def test_a_zero_area_region_is_not_fatal(self, classifier):
        page = FakePage(drawings=[DrawingBlock(rect=Rect(100.0, 100.0, 100.0, 100.0))])
        classifier._detect_visual_regions(page)      # must not raise


class TestTheAnalysisSaysWhatWasDiscarded:
    """
    What this filter throws away must be visible, because the rule is not
    safe on its own.

    It works on the sample corpus only because the one figure that would
    measure like decoration — a protein sequence alignment — is typeset by
    its publisher as VECTOR GLYPHS, so it reports 0.0% word coverage. A
    publisher setting the same alignment as selectable text would measure at
    the decoration end, the figure would be discarded, and with nothing
    visual left the page can fall to TEXT_ONLY. That path never reaches the
    figure second-opinion check, which only examines pages the VLM read.

    Every other filter in the visual scan rejects on geometry and cannot be
    wrong in this way. This one rejects on a measurement of typesetting, so
    it reports its rejections and lets the caller check them — see
    `body_pipeline._detections_the_filter_hid`.
    """

    def test_discarded_regions_are_reported_not_forgotten(self, classifier):
        box = Rect(72.0, 200.0, 400.0, 500.0)
        page = FakePage(drawings=clustered_drawings(box),
                        words=words_covering(box, DECORATION_COVERAGE))
        dense = {"word_count": 1470, "avg_words_per_line": 12, "char_count": 10227}

        page_type, analysis = classifier.classify(page, 20, text_stats=dense)

        assert page_type is PageType.TEXT_ONLY
        assert analysis["visual_regions"] == []
        assert len(analysis["text_dense_regions"]) == 1
        assert analysis["text_dense_regions"][0]["type"] == "text_dense_cluster"

    def test_a_kept_region_is_not_also_reported_as_discarded(self, classifier):
        box = Rect(72.0, 200.0, 400.0, 500.0)
        page = FakePage(drawings=clustered_drawings(box),
                        words=words_covering(box, FIGURE_COVERAGE))

        _type, analysis = classifier.classify(page, 20)
        assert len(analysis["visual_regions"]) == 1
        assert analysis["text_dense_regions"] == []

    def test_an_ordinary_prose_page_discards_nothing(self, classifier):
        page = FakePage(words=words_covering(Rect(72.0, 72.0, 400.0, 700.0), 0.4))
        _type, analysis = classifier.classify(page, 20)
        assert analysis["text_dense_regions"] == []
