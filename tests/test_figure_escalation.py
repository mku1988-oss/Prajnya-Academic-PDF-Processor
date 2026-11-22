#!/usr/bin/env python3
"""
A second opinion on figures, asked for only where the cheap pass contradicts itself.

The trigger is the user's, and it is better than the one it replaced: a page the
CLASSIFIER says holds visual content, on which the DETECTOR then found no
figure. Both cannot be right, and the disagreement is evidence about *this*
page rather than about the document.

The rejected alternative is recorded here because it is the tempting one: the
first design triggered on the read text naming a figure ("Fig. 4 | ..."). In a
research article a figure is routinely cited pages away from where it is
printed, so the text says the DOCUMENT has a Figure 4 — not that this page
holds it.

MEASURED on the sample corpus, 188 VLM-read body pages:
  18 pages fire (2.3% of all pages, ~5 min of GPU across a 49-minute run)
  of the ones inspected by eye:
    JRuberti p13     six-panel confocal figure    REAL MISS
    JRuberti p14     four-panel figure strip      REAL MISS
    Hydration p3     vector plot                  REAL MISS
    Collagen p21     references page              false alarm
    Saturated p2     scanned text page            false alarm

A SIZE THRESHOLD WAS TRIED AND REJECTED, and the numbers are why: the
references page's link decorations covered 10.6% of the page while JRuberti
p14's genuine figure covered 9.9%. Any cutoff excluding the first discards the
second. Only the scan exclusion survived — at 94.8% the "region" IS the page,
so there is no figure to crop out of it.

A false alarm costs GPU seconds. A missed figure is gone silently. The rule
leans accordingly.
"""

import pytest

from src.processing.layout.base import Detection
from src.processing.strategies.research_article.figure_escalation import (
    SCAN_PAGE_FRACTION,
    collect_suspect_pages,
    describe_escalation,
    page_needs_second_opinion,
)

PAGE_SIZE = (612.0, 792.0)          # US Letter, in PDF points
PAGE_AREA = PAGE_SIZE[0] * PAGE_SIZE[1]


def region(fraction: float) -> dict:
    """A visual region covering `fraction` of the page, as the classifier emits."""
    width = PAGE_SIZE[0]
    height = (fraction * PAGE_AREA) / width
    return {"type": "image", "bbox": (0.0, 0.0, width, height)}


def analysis(*fractions: float) -> dict:
    return {"visual_regions": [region(f) for f in fractions]}


class TestTheDisagreement:
    def test_visuals_seen_and_no_figure_found_is_a_disagreement(self):
        assert page_needs_second_opinion(analysis(0.23), detections=[],
                                         page_size=PAGE_SIZE) is True

    def test_a_found_figure_settles_it(self):
        found = [Detection(label="image", box_px=(0, 0, 100, 100), score=0.9)]
        assert page_needs_second_opinion(analysis(0.23), detections=found,
                                         page_size=PAGE_SIZE) is False

    def test_nothing_seen_means_nothing_missing(self):
        """The classifier found no visual content; there is no contradiction."""
        assert page_needs_second_opinion({"visual_regions": []}, detections=[],
                                         page_size=PAGE_SIZE) is False

    def test_a_detected_table_accounts_for_the_visual(self):
        """The page's visual content is a table, which was not lost."""
        found = [Detection(label="table", box_px=(0, 0, 100, 100), score=0.9)]
        assert page_needs_second_opinion(analysis(0.30), detections=found,
                                         page_size=PAGE_SIZE) is False

    def test_detection_never_running_is_itself_a_disagreement(self):
        """None means the page was never looked at — the strongest case to ask."""
        assert page_needs_second_opinion(analysis(0.30), detections=None,
                                         page_size=PAGE_SIZE) is True


class TestTheMeasuredPages:
    """The real pages that shaped the rule, by their measured area fractions."""

    @pytest.mark.parametrize("name,fraction", [
        ("JRuberti p13 — six-panel confocal figure", 0.234),
        ("JRuberti p14 — four-panel strip", 0.099),
        ("Hydration p3 — vector plot lost to DPI jitter", 0.283),
    ])
    def test_the_real_misses_are_escalated(self, name, fraction):
        assert page_needs_second_opinion(analysis(fraction), detections=[],
                                         page_size=PAGE_SIZE) is True, name

    def test_the_small_real_figure_is_kept_though_it_is_smaller_than_a_false_alarm(self):
        """
        THE REASON THERE IS NO SIZE THRESHOLD. JRuberti p14 is a real figure at
        9.9%; the references page is a false alarm at 10.6%. Ordering by size
        puts them the wrong way round, so size cannot be the discriminator.
        """
        real_figure = page_needs_second_opinion(analysis(0.099), detections=[],
                                                page_size=PAGE_SIZE)
        false_alarm = page_needs_second_opinion(analysis(0.106), detections=[],
                                                page_size=PAGE_SIZE)
        assert real_figure is True
        assert false_alarm is True          # we accept it, knowingly

    def test_a_scanned_page_is_not_escalated(self):
        """Saturated Solutions p2/p7: at 94.8% the region IS the page."""
        assert page_needs_second_opinion(analysis(0.948), detections=[],
                                         page_size=PAGE_SIZE) is False

    def test_a_dense_drawing_region_is_never_taken_for_a_scan(self):
        """D9: a page of 10,000+ vector strokes gets one region spanning the
        page. It is a vector figure, so a missed figure there still earns
        the second opinion."""
        dense = {"type": "dense_drawings", "bbox": (0.0, 0.0, PAGE_SIZE[0], PAGE_SIZE[1])}
        assert page_needs_second_opinion({"visual_regions": [dense]}, detections=[],
                                         page_size=PAGE_SIZE) is True

    def test_the_scan_boundary(self):
        just_under = page_needs_second_opinion(
            analysis(SCAN_PAGE_FRACTION - 0.01), detections=[], page_size=PAGE_SIZE)
        just_over = page_needs_second_opinion(
            analysis(SCAN_PAGE_FRACTION + 0.01), detections=[], page_size=PAGE_SIZE)
        assert just_under is True
        assert just_over is False

    def test_a_big_figure_beside_a_full_page_scan_still_counts_as_a_scan(self):
        """Any whole-page region means a scan, whatever else is on the page."""
        assert page_needs_second_opinion(analysis(0.30, 0.948), detections=[],
                                         page_size=PAGE_SIZE) is False


class TestUnknownGeometryLeansTowardsAsking:
    def test_no_page_size_still_escalates(self):
        """Cannot judge scan-ness; escalating costs time, skipping loses a figure."""
        assert page_needs_second_opinion(analysis(0.30), detections=[],
                                         page_size=None) is True

    def test_a_region_with_no_box_is_not_fatal(self):
        weird = {"visual_regions": [{"type": "image"}]}
        assert page_needs_second_opinion(weird, detections=[],
                                         page_size=PAGE_SIZE) is True

    def test_a_zero_area_page_is_not_fatal(self):
        assert page_needs_second_opinion(analysis(0.30), detections=[],
                                         page_size=(0.0, 0.0)) is True


class TestCollectingThemForOneModelSwap:
    class FakePage:
        def __init__(self, flag):
            self.metadata = {"needs_figure_second_opinion": flag}

    def test_only_flagged_pages_are_collected_in_order(self):
        pages = {4: self.FakePage(True), 0: self.FakePage(False),
                 2: self.FakePage(True), 7: self.FakePage(False)}
        assert collect_suspect_pages(pages) == [2, 4]

    def test_a_self_consistent_document_collects_nothing(self):
        """The common case: no swap, no grounding call, no cost at all."""
        assert collect_suspect_pages({0: self.FakePage(False)}) == []

    def test_a_page_with_no_metadata_is_not_collected(self):
        class Bare:
            metadata = None
        assert collect_suspect_pages({0: Bare()}) == []


class TestTheLogLine:
    def test_the_cost_is_never_silent(self):
        message = describe_escalation([2, 4, 12], seconds_each=17.0)
        assert "3 page(s)" in message
        assert "p3, 5, 13" in message          # 1-based, as a human reads them
        assert "51s" in message

    def test_nothing_to_do_says_so(self):
        assert "no pages" in describe_escalation([])

    def test_a_long_list_is_truncated_not_dumped(self):
        message = describe_escalation(list(range(30)))
        assert "+22 more" in message
