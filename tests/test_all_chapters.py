#!/usr/bin/env python3
"""
"Read every chapter" must survive the whole way down to pages actually read.

Found on the full sample run of session 9: `run_prajnya.py --chapters` with no
numbers — documented as "read EVERY chapter" — processed the 345-page volume in
**1.5 seconds** and logged `Chapters read: []`. It read nothing at all, having
been asked for everything.

The cause is a sentinel collision between two adjacent layers that both used a
list, and disagreed about what an empty one meant:

    layer                          None means        [] meant
    ------------------------------ ----------------- --------------------
    CLI / BookDecision / policy     read no bodies    read EVERY chapter
    reading._spans_for              EVERY chapter     read NOTHING
    router._book_body_pipeline      (falsy: no reader — swallowed [] too)

Exactly opposite readings of the same value, and `[]` being falsy meant the
mistake could not announce itself: `if not chapters` looks correct at a glance
in both files.

These tests walk the value from the flag to the page list, at every seam it
crosses, because unit-testing either end alone is what let this through.
"""

import pytest

from src.processing.batch import BookDecision, decision_for
from src.processing.policy import BookPolicy, BookSettings
from src.processing.strategies.book.chapter_boundaries import ChapterMap, ChapterSpan
from src.processing.strategies.book.reading import select_chapter_pages

CHAPTER_MAP = ChapterMap(spans=[
    ChapterSpan(number=1, start_index=10, end_index=12,
                title="First", authors=["A. One"]),
    ChapterSpan(number=2, start_index=13, end_index=20,
                title="Second", authors=["B. Two"]),
    ChapterSpan(number=3, start_index=21, end_index=25,
                title="Third", authors=["C. Three"]),
])


class TestTheValueSurvivesEverySeam:
    def test_the_policy_turns_all_into_an_empty_list(self):
        """Seam 1: `--chapters` with no numbers, or `policy: all` in the YAML."""
        decision = decision_for(BookSettings(policy=BookPolicy.ALL))
        assert decision.chapters == []

    def test_the_router_builds_a_reader_for_an_empty_list(self):
        """
        Seam 2. This is where it died: `if not chapters` returned no body
        pipeline, so the strategy had nothing to read WITH — the request for
        everything produced the cheapest possible run.
        """
        import inspect

        from src.processing.strategies import router

        source = inspect.getsource(router._book_body_pipeline)
        assert "chapters is None" in source, \
            "the router must test `is None`, not falsiness — [] means EVERY chapter"

    def test_an_empty_selection_reads_every_located_chapter(self):
        """Seam 3: the page selection itself."""
        assert select_chapter_pages(CHAPTER_MAP, []) == list(range(10, 26))

    def test_none_still_reads_everything_too(self):
        """The pre-existing convention inside reading.py is unchanged."""
        assert select_chapter_pages(CHAPTER_MAP, None) == list(range(10, 26))

    def test_naming_chapters_still_reads_only_those(self):
        """The fix must not turn a specific request into a whole-volume read."""
        assert select_chapter_pages(CHAPTER_MAP, [2]) == list(range(13, 21))

    def test_the_ceiling_still_applies_to_a_whole_volume_read(self):
        """--max-pages is the guard against an accidental hours-long run."""
        assert len(select_chapter_pages(CHAPTER_MAP, [], max_pages=6)) == 6


class TestNotRequestedIsStillNotRequested:
    """The other half: the cheap default must stay cheap."""

    def test_the_safe_default_asks_for_no_bodies(self):
        assert BookDecision.safe_default().chapters is None

    def test_skip_policy_asks_for_no_bodies(self):
        assert decision_for(BookSettings(policy=BookPolicy.SKIP)).chapters is None

    def test_the_strategy_reads_nothing_when_nothing_was_requested(self):
        """None must never be confused with the empty list going the other way."""
        from src.processing.strategies.book.strategy import BookStrategy

        strategy = BookStrategy.__new__(BookStrategy)
        strategy.body_pipeline = object()
        strategy.chapters_to_read = None

        assert strategy._read_chapter_bodies(None, CHAPTER_MAP) == []

    @pytest.mark.parametrize("chapters,expected_pipeline", [
        (None, False),      # not requested -> no reader is built
        ([], True),         # every chapter -> a reader is essential
        ([3], True),        # named chapters -> a reader is essential
    ])
    def test_the_reader_is_built_exactly_when_bodies_are_wanted(
            self, chapters, expected_pipeline, monkeypatch):
        from src.processing.strategies import router

        built = {}

        def fake_pipeline(config_root, chapters_arg, logger):
            built["called"] = chapters_arg is not None
            return None

        monkeypatch.setattr(router, "_book_body_pipeline", fake_pipeline)
        router._book_body_pipeline("config", chapters, None)

        assert built["called"] is expected_pipeline
