#!/usr/bin/env python3
"""
Tests for chapter boundary detection — the RAG attribution problem.

Knowing chapter 3 is by Siadat et al. is only half of citing it correctly; the
other half is knowing that PDF page 60 is *in* chapter 3. That is not
arithmetic: the contents page counts in the book's printed numbering, and the
offset to PDF indices drifts as unnumbered pages are inserted (measured: 7 at
chapter 3, 6 at chapter 4 of the sample volume). So the mapping is read off
the running headers the publisher already prints.

The ladder under test, strongest rung first:

  1. recto header  "3 Tendon Extracellular Matrix Assembly…"  -> chapter 3
  2. verso header  "S. M. Siadat et al."                      -> chapter 3
  3. opening page  a bare "3", corroborated by its neighbour  -> chapter 3
  4. gaps          filled from neighbours; a chapter no header ever named is
                   placed by printed-page arithmetic from its nearest located
                   neighbour — locally, never on a global offset

The fixtures are shortened from the real text layer, keeping the details that
broke the first implementation: chapter 1's title is "Introduction" (twelve
characters, so any minimum-length guard on the header loses it), and the
chapter opening page's trailing bare number is the *chapter* number while
looking exactly like a folio.

`TestTheRealBook` is the gate — the full 345-page volume, checked against
boundaries verified by hand and against the contents page's own arithmetic.
"""

from pathlib import Path

import pytest

from src.processing.strategies.book.chapter_boundaries import (
    ChapterMap,
    ChapterSpan,
    detect_chapter_spans,
)
from src.processing.strategies.book.toc import TocEntry
from tests.local_samples import SAMPLES as LOCAL_SAMPLES

SAMPLES = LOCAL_SAMPLES
FULL_BOOK = SAMPLES / "Full" / "Tendon assembly 2021.pdf"

needs_book = pytest.mark.skipif(not FULL_BOOK.is_file(),
                                reason="full Springer volume not available")


def entry(number, title, start_page, authors=()):
    return TocEntry(number=number, title=title, start_page=start_page,
                    raw_page=str(start_page), authors=list(authors))


CHAPTERS = [
    entry(1, "Introduction", 1, ["Jaroslava Halper"]),
    entry(2, "Basic Structure, Physiology, and Biochemistry", 5,
          ["Michael J. Mienaltowski", "Nicole L. Gonzales"]),
    entry(3, "Tendon Extracellular Matrix Assembly", 45,
          ["Seyed Mohammad Siadat", "Danae E. Zamboulis"]),
]


class TestTheRungs:
    def test_a_recto_header_names_its_chapter(self):
        texts = {0: "body text\n3 Tendon Extracellular Matrix Assembly"}
        chapter_map = detect_chapter_spans(texts, CHAPTERS)
        assert chapter_map.chapter_for_page(0).number == 3

    def test_a_verso_header_is_matched_through_the_contents_authors(self):
        texts = {0: "3 Tendon Extracellular Matrix Assembly",
                 1: "body text\nS. M. Siadat et al."}
        chapter_map = detect_chapter_spans(texts, CHAPTERS)
        assert chapter_map.chapter_for_page(1).number == 3

    def test_a_short_chapter_title_is_not_lost_to_a_length_guard(self):
        """Chapter 1 is called "Introduction" — twelve characters."""
        texts = {0: "prose\n1 Introduction"}
        chapter_map = detect_chapter_spans(texts, CHAPTERS)
        assert chapter_map.chapter_for_page(0).number == 1

    def test_a_numbered_list_item_is_not_a_running_header(self):
        """The title must match the contents page, not merely be long."""
        texts = {0: "method\n3 mL of buffer was added to the reaction mixture"}
        chapter_map = detect_chapter_spans(texts, CHAPTERS)
        assert chapter_map.pages_resolved_directly == 0

    def test_a_truncated_header_still_matches_its_title(self):
        """Running headers are cut to fit the page width."""
        texts = {0: "prose\n3 Tendon Extracell…"}
        assert detect_chapter_spans(texts, CHAPTERS).pages_resolved_directly == 1

    def test_a_ligature_early_in_the_title_costs_the_match(self):
        """
        A documented limitation, measured rather than assumed.

        The text layer drops ligatures ("fbrils" for "fibrils"), and one
        inside the compared prefix defeats the match. That costs a page's
        rung-1 identification, which rung 2 and the gap logic then cover — a
        coverage cost, not a correctness one. Loosening the comparison to
        absorb it would start matching the WRONG chapter.
        """
        chapters = [entry(3, "Collagen Fibrils and Function", 45)]
        texts = {0: "prose\n3 Collagen Fbrils and Function"}
        assert detect_chapter_spans(texts, chapters).pages_resolved_directly == 0


class TestTheOpeningPage:
    """A chapter's opening page carries a bare chapter number, not a header."""

    def test_the_opening_page_starts_the_chapter(self):
        texts = {
            0: "prose\n2 Basic Structure, Physiology, and Biochemistry",
            1: "© Springer Nature Switzerland AG 2021 45\nabstract\n3",
            2: "prose\nS. M. Siadat et al.",
        }
        chapter_map = detect_chapter_spans(texts, CHAPTERS)
        assert chapter_map.chapter_for_page(1).number == 3
        assert chapter_map.chapter_for_page(0).number == 2

    def test_a_bare_number_is_refused_without_corroboration(self):
        """
        Alone, a bare number is indistinguishable from a page folio.

        Here "3" is followed by chapter 2's pages, so it is a folio and must
        not be read as the start of chapter 3.
        """
        texts = {0: "prose\n3", 1: "prose\n2 Basic Structure, Physiology, and Biochemistry"}
        chapter_map = detect_chapter_spans(texts, CHAPTERS)

        span = chapter_map.chapter_for_page(0)
        assert span is None or span.number != 3
        assert 3 not in {existing.number for existing in chapter_map.spans}

    def test_a_bare_number_that_is_not_a_chapter_is_ignored(self):
        texts = {0: "prose\n97", 1: "prose\n3 Tendon Extracellular Matrix Assembly"}
        chapter_map = detect_chapter_spans(texts, CHAPTERS)

        assert chapter_map.chapter_for_page(0) is None
        assert chapter_map.chapter_for_page(1).number == 3


class TestGapsAndFallbacks:
    def test_a_gap_between_two_chapters_falls_to_the_later_one(self):
        """Opening pages carry no header, so the gap belongs to what follows."""
        texts = {
            0: "prose\n2 Basic Structure, Physiology, and Biochemistry",
            1: "an opening page with no header at all",
            2: "prose\n3 Tendon Extracellular Matrix Assembly",
        }
        chapter_map = detect_chapter_spans(texts, CHAPTERS)
        assert chapter_map.chapter_for_page(1).number == 3

    def test_a_chapter_no_header_ever_named_is_placed_from_its_neighbour(self):
        """
        Chapter 1 of the sample is four pages and carries no running header.

        It is placed by printed-page arithmetic anchored on chapter 2 — four
        printed pages earlier, so four PDF pages earlier.
        """
        texts = {
            4: "opening page, no header",
            5: "prose\n2 Basic Structure, Physiology, and Biochemistry",
        }
        chapter_map = detect_chapter_spans(texts, CHAPTERS)
        chapter_one = next(s for s in chapter_map.spans if s.number == 1)
        assert chapter_one.start_index == 1        # chapter 2 starts at 5, minus 4
        assert chapter_one.confidence == "arithmetic"

    def test_a_chapter_whose_arithmetic_would_overlap_is_dropped(self):
        """
        Chapters do not overlap or run backwards.

        Here chapter 2 has no header, and the arithmetic from chapter 1 puts
        it *after* chapter 3's observed start — so the placement is wrong.
        Dropping it leaves page 22 attributed to chapter 3; keeping it would
        attribute chapter 3's pages to chapter 2's authors, which is the exact
        failure this module exists to prevent.
        """
        texts = {
            20: "prose\n1 Introduction",
            22: "prose\n3 Tendon Extracellular Matrix Assembly",
        }
        chapter_map = detect_chapter_spans(texts, CHAPTERS)

        assert 2 not in {span.number for span in chapter_map.spans}
        assert chapter_map.chapter_for_page(22).number == 3

    def test_the_arithmetic_anchors_on_the_nearest_chapter_not_a_far_one(self):
        """
        The printed offset drifts across a book, so a distant anchor is worse.

        Chapter 2 is unobserved and sits between two observed chapters. Its
        placement must come from chapter 1 (adjacent, +4 printed pages), not
        from chapter 3 far away — both give an ordering-consistent answer, so
        only the value distinguishes them.
        """
        texts = {
            100: "prose\n1 Introduction",
            200: "prose\n3 Tendon Extracellular Matrix Assembly",
        }
        chapter_map = detect_chapter_spans(texts, CHAPTERS)

        chapter_two = next(s for s in chapter_map.spans if s.number == 2)
        assert chapter_two.start_index == 104        # 100 + (5 - 1), from chapter 1
        assert chapter_two.confidence == "arithmetic"

    def test_an_out_of_order_header_match_is_dropped(self):
        """
        The safety net for a header matched somewhere it does not belong.

        A page early in the book can quote a later chapter's number and title
        — a cross-reference — and that would place chapter 3 before chapter 1.
        Chapters cannot run backwards, so the offending chapter is dropped
        rather than allowed to claim pages belonging to another.
        """
        texts = {
            2: "see also\n3 Tendon Extracellular Matrix Assembly",
            5: "prose\n1 Introduction",
        }
        chapter_map = detect_chapter_spans(texts, CHAPTERS)

        placed = {span.number for span in chapter_map.spans}
        assert 1 in placed and 3 not in placed

    def test_an_unplaceable_chapter_is_omitted_not_guessed(self):
        chapter_map = detect_chapter_spans({0: "nothing identifiable"}, CHAPTERS)
        assert chapter_map.spans == []

    def test_no_contents_means_no_chapters(self):
        assert detect_chapter_spans({0: "text"}, []).spans == []

    def test_front_matter_is_never_attributed_to_a_chapter(self):
        texts = {0: "contents page", 1: "prose\n1 Introduction"}
        chapter_map = detect_chapter_spans(texts, CHAPTERS, first_body_index=1)
        assert chapter_map.chapter_for_page(0) is None


class TestAmbiguousAuthors:
    """An editor who wrote several chapters must not resolve a verso header."""

    def test_a_surname_in_two_chapters_resolves_nothing(self):
        chapters = [
            entry(1, "Introduction", 1, ["Jaroslava Halper"]),
            entry(4, "Basic Components", 105, ["Jaroslava Halper"]),
        ]
        chapter_map = detect_chapter_spans({0: "prose\nJ. Halper"}, chapters)
        assert chapter_map.pages_resolved_directly == 0


class TestConfidenceReporting:
    def test_coverage_is_reported_so_a_guess_is_visible(self):
        texts = {index: "prose with no header whatsoever" for index in range(10)}
        texts[0] = "prose\n1 Introduction"
        chapter_map = detect_chapter_spans(texts, CHAPTERS)
        assert chapter_map.direct_coverage == pytest.approx(0.1)
        assert chapter_map.is_confident is False

    def test_an_empty_map_is_never_confident(self):
        assert ChapterMap().is_confident is False

    def test_a_span_knows_which_pages_it_holds(self):
        span = ChapterSpan(number=3, title="t", start_index=52, end_index=110)
        assert span.contains(52) and span.contains(110) and span.contains(60)
        assert not span.contains(51) and not span.contains(111)


@needs_book
class TestTheRealBook:
    """
    The gate: the full 345-page Springer volume.

    Boundaries were verified by hand against the printed pages, and are
    independently cross-checked below against the contents page's own
    arithmetic — two sources that would have to be wrong in the same way to
    let a bad mapping through.
    """

    @pytest.fixture(scope="class")
    def result(self):
        from run_prajnya import build_document_profile
        from src.processing.pdf import get_backend
        from src.processing.strategies.book import BookStrategy

        backend = get_backend()
        profile = build_document_profile(FULL_BOOK, backend)
        return BookStrategy(pdf_backend=backend).process(profile).processing_result

    @pytest.fixture(scope="class")
    def by_number(self, result):
        return {record["number"]: record for record in result.chapters
                if record["number"] is not None}

    def test_every_chapter_gets_a_page_range(self, by_number):
        assert len(by_number) == 15
        assert all(record["page_range"] for record in by_number.values())

    @pytest.mark.parametrize("number,expected", [
        (1, [10, 12]),      # short, no running header — placed by arithmetic
        (2, [13, 51]),
        (3, [52, 110]),     # opening page 52 carries a bare "3"
        (4, [111, 131]),
        (15, [326, 344]),
    ])
    def test_hand_verified_boundaries(self, by_number, number, expected):
        assert by_number[number]["page_range"] == expected

    def test_the_ranges_are_contiguous_and_disjoint(self, by_number):
        ranges = [by_number[n]["page_range"] for n in sorted(by_number)]
        for earlier, later in zip(ranges, ranges[1:]):
            assert later[0] == earlier[1] + 1, f"{earlier} then {later}"

    def test_span_lengths_agree_with_the_contents_page(self, by_number):
        """
        The independent cross-check.

        Two consecutive start pages give a chapter's length in printed pages.
        The detected span should match, give or take the blank page many
        chapters end on so the next can open on a recto.
        """
        numbers = sorted(by_number)
        for number, following in zip(numbers, numbers[1:]):
            record, next_record = by_number[number], by_number[following]
            implied = next_record["start_page_printed"] - record["start_page_printed"]
            detected = record["page_range"][1] - record["page_range"][0] + 1
            assert abs(detected - implied) <= 2, (
                f"chapter {number}: detected {detected}p, contents implies {implied}p")

    def test_a_page_deep_inside_a_chapter_is_attributed_to_its_authors(self):
        """The whole point: cite page 60 to Siadat et al., not to the editor."""
        from src.processing.pdf import get_backend
        from src.processing.strategies.book.toc import parse_toc_entries
        from src.processing.utils.contents_leaders import contents_page_text

        backend = get_backend()
        with backend.open(FULL_BOOK) as doc:
            texts = {index: (doc.page(index).text() or "")
                     for index in range(doc.page_count)}
            # The contents pages are read the way production reads them (the
            # front-matter scan): rebuilt from geometry, so the dot leaders
            # survive under pypdfium2 5.x, which extracts none.
            entries = (parse_toc_entries(contents_page_text(doc.page(5)))
                       + parse_toc_entries(contents_page_text(doc.page(6))))

        chapter_map = detect_chapter_spans(texts, entries, first_body_index=8)
        span = chapter_map.chapter_for_page(60)

        assert span.number == 3
        assert span.authors[0] == "Seyed Mohammad Siadat"
        assert "Jaroslava Halper" not in span.authors      # the editor, not this author

    def test_the_mapping_is_reported_as_confident(self, result):
        numbered = [r for r in result.chapters if r["page_range"]]
        assert all(r["page_range_confidence"] in {"header", "arithmetic"}
                   for r in numbered)
        assert sum(1 for r in numbered
                   if r["page_range_confidence"] == "header") >= 14
