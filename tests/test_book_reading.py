#!/usr/bin/env python3
"""
Tests for chapter-scoped book reading.

Two things are protected here, and the first is mostly about restraint.

**Reading is opt-in.** A 345-page volume at the article pipeline's measured
~20s/page is about two hours of GPU. So `BookStrategy` must read nothing
unless a caller asked for specific chapters — the regression guard below
exists because "read everything by default" is the easy accident.

**Attribution travels with the content.** Every body page carries its
chapter's authors, so a passage retrieved from page 269 cites Meester et al.
rather than the volume's editor. Looking it up later is how it gets lost.

The page reading itself belongs to `BodyPipeline`, which has its own parity
gate, so it is faked here — what is under test is the selection and the
stamping around it.
"""

from pathlib import Path

import pytest

from src.processing.models import ContentPage, DocumentProfile
from src.processing.strategies.book import BookStrategy
from src.processing.strategies.book.chapter_boundaries import ChapterMap, ChapterSpan
from src.processing.strategies.book.reading import (
    attribute_pages,
    estimate_reading_cost,
    select_chapter_pages,
)
from tests.local_samples import SAMPLES as LOCAL_SAMPLES

SAMPLES = LOCAL_SAMPLES
FULL_BOOK = SAMPLES / "Full" / "Tendon assembly 2021.pdf"

needs_book = pytest.mark.skipif(not FULL_BOOK.is_file(),
                                reason="full Springer volume not available")

CHAPTER_MAP = ChapterMap(
    spans=[
        ChapterSpan(number=1, title="Introduction", authors=["Jaroslava Halper"],
                    start_index=10, end_index=12),
        ChapterSpan(number=2, title="Basic Structure",
                    authors=["Michael J. Mienaltowski"], start_index=13, end_index=51),
        ChapterSpan(number=3, title="Tendon Extracellular Matrix Assembly",
                    authors=["Seyed Mohammad Siadat"], start_index=52, end_index=60),
    ],
    pages_resolved_directly=40, body_pages=51,
)


class TestSelection:
    def test_one_chapter_yields_exactly_its_pages(self):
        assert select_chapter_pages(CHAPTER_MAP, [1]) == [10, 11, 12]

    def test_several_chapters_come_back_in_reading_order(self):
        pages = select_chapter_pages(CHAPTER_MAP, [3, 1])
        assert pages[0] == 10 and pages[-1] == 60
        assert pages == sorted(pages)

    def test_no_selection_means_the_whole_book(self):
        """A deliberate, expensive request — never the accidental one."""
        assert select_chapter_pages(CHAPTER_MAP) == list(range(10, 61))

    def test_an_unlocated_chapter_is_reported_not_invented(self):
        assert select_chapter_pages(CHAPTER_MAP, [99]) == []

    def test_a_mix_of_located_and_unlocated_reads_what_exists(self):
        assert select_chapter_pages(CHAPTER_MAP, [1, 99]) == [10, 11, 12]

    def test_the_page_ceiling_truncates_from_the_end(self):
        """
        Chapters asked for first stay whole.

        Truncating every chapter proportionally would give a caller several
        half-read chapters instead of a few complete ones.
        """
        pages = select_chapter_pages(CHAPTER_MAP, [1, 2], max_pages=5)
        assert pages == [10, 11, 12, 13, 14]

    def test_no_ceiling_reads_everything_selected(self):
        assert len(select_chapter_pages(CHAPTER_MAP, [2])) == 39

    def test_an_empty_map_selects_nothing(self):
        assert select_chapter_pages(ChapterMap(), [1]) == []


class TestAttribution:
    def _page(self, number):
        return ContentPage(page_number=number, content="text", metadata={"m": 1})

    def test_a_page_carries_its_own_chapters_authors(self):
        page = attribute_pages([self._page(53)], CHAPTER_MAP)[0]
        chapter = page.metadata["chapter"]

        assert chapter["number"] == 3
        assert chapter["authors"] == ["Seyed Mohammad Siadat"]
        assert chapter["title"] == "Tendon Extracellular Matrix Assembly"

    def test_the_editor_is_not_credited_with_another_authors_chapter(self):
        """The failure this whole path exists to prevent."""
        page = attribute_pages([self._page(53)], CHAPTER_MAP)[0]
        assert "Jaroslava Halper" not in page.metadata["chapter"]["authors"]

    def test_page_numbers_are_1_based_and_spans_are_0_based(self):
        """Chapter 1 spans indices 10-12, i.e. 1-based pages 11-13."""
        assert attribute_pages([self._page(11)], CHAPTER_MAP)[0] \
            .metadata["chapter"]["number"] == 1
        assert attribute_pages([self._page(10)], CHAPTER_MAP)[0] \
            .metadata["chapter"] is None

    def test_a_page_outside_every_chapter_gets_none_not_a_guess(self):
        page = attribute_pages([self._page(3)], CHAPTER_MAP)[0]
        assert page.metadata["chapter"] is None

    def test_existing_page_metadata_is_preserved(self):
        page = attribute_pages([self._page(53)], CHAPTER_MAP)[0]
        assert page.metadata["m"] == 1

    def test_the_boundary_confidence_travels_with_the_citation(self):
        """A reader of the record can tell a read boundary from an inferred one."""
        page = attribute_pages([self._page(53)], CHAPTER_MAP)[0]
        assert page.metadata["chapter"]["boundary_confidence"] == "header"


class TestCostEstimate:
    def test_the_estimate_is_reported_before_the_gpu_starts(self):
        cost = estimate_reading_cost(list(range(100)))
        assert cost["pages"] == 100
        assert cost["worst_case_minutes"] == pytest.approx(33.1, abs=0.5)

    def test_an_empty_selection_costs_nothing(self):
        assert estimate_reading_cost([])["worst_case_seconds"] == 0


class FakeBodyPipeline:
    """Stands in for the real reader, which has its own parity gate."""

    def __init__(self):
        self.calls = []

    def process_body_pages(self, cache, page_indices):
        self.calls.append(list(page_indices))
        return [ContentPage(page_number=index + 1, content=f"page {index}",
                            metadata={"processing_method": "pymupdf"})
                for index in page_indices]


@needs_book
class TestTheStrategy:
    @staticmethod
    def _profile():
        from run_prajnya import build_document_profile
        from src.processing.pdf import get_backend

        return build_document_profile(FULL_BOOK, get_backend()), get_backend()

    def test_nothing_is_read_unless_chapters_were_asked_for(self):
        """
        The regression guard: a whole volume must never be read by default.

        At ~20s/page this default is the difference between a 0.7s CPU read
        and two hours of GPU.
        """
        profile, backend = self._profile()
        pipeline = FakeBodyPipeline()

        BookStrategy(pdf_backend=backend, body_pipeline=pipeline).process(profile)

        assert pipeline.calls == []

    def test_no_pipeline_means_no_reading_even_if_chapters_are_named(self):
        profile, backend = self._profile()
        result = BookStrategy(pdf_backend=backend,
                              chapters_to_read=[12]).process(profile).processing_result

        assert all((page.metadata or {}).get("chapter") is None
                   for page in result.pages)

    def test_asking_for_a_chapter_reads_exactly_its_pages(self):
        profile, backend = self._profile()
        pipeline = FakeBodyPipeline()

        BookStrategy(pdf_backend=backend, body_pipeline=pipeline,
                     chapters_to_read=[12]).process(profile)

        assert pipeline.calls == [list(range(268, 276))]

    def test_the_body_pages_come_back_attributed(self):
        profile, backend = self._profile()
        result = BookStrategy(pdf_backend=backend, body_pipeline=FakeBodyPipeline(),
                              chapters_to_read=[12]).process(profile).processing_result

        body = [page for page in result.pages if (page.metadata or {}).get("chapter")]
        assert len(body) == 8
        assert {page.metadata["chapter"]["number"] for page in body} == {12}
        assert body[0].metadata["chapter"]["authors"][0] == "Josephina A. N. Meester"

    def test_the_front_matter_record_survives_a_body_read(self):
        profile, backend = self._profile()
        result = BookStrategy(pdf_backend=backend, body_pipeline=FakeBodyPipeline(),
                              chapters_to_read=[12]).process(profile).processing_result

        assert result.metadata.title == "Progress in Heritable Soft Connective Tissue Diseases"
        assert len(result.chapters) == 17
        assert result.errors == []

    def test_the_page_ceiling_is_honoured_end_to_end(self):
        profile, backend = self._profile()
        pipeline = FakeBodyPipeline()

        BookStrategy(pdf_backend=backend, body_pipeline=pipeline,
                     chapters_to_read=[12], max_body_pages=3).process(profile)

        assert pipeline.calls == [[268, 269, 270]]
