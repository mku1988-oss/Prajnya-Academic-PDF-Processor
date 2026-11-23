#!/usr/bin/env python3
"""
Tests for the front-matter identifier scan.

Two things are being protected here. First, that scanning deeper actually
finds what page 1 could not: a Springer volume prints its ISBN, ISSN and DOI
on the copyright page (page 5 in the sample corpus), and page-1-only scanning
returned nothing at all for that file. Second — and this is the one that
matters more — that scanning deeper does not start inventing identifiers. A
bare DOI on page 3 is usually a citation of somebody else's paper, and a bare
`NNNN-NNNX` is as likely to be a year range or half an ORCID as an ISSN.
Both false-positive cases below were found by running the scan over the real
corpus, not by imagining them.

The real-PDF tests are skipped when the sample corpus is absent.
"""

from pathlib import Path

import pytest

from src.processing.extractors.front_matter import (
    DEFAULT_FRONT_MATTER_PAGES,
    FrontMatterIdentifiers,
    scan_front_matter,
)
from tests.local_samples import SAMPLES as LOCAL_SAMPLES

SAMPLES = LOCAL_SAMPLES
BENCHMARK_PDF = SAMPLES / "Full" / (
    "An NMR method to characterize multiple water compartments on mammalian collagen.pdf")
SPRINGER_BOOK = SAMPLES / "Extracted" / "Tendon assembly 2021 [1-7].pdf"

needs_benchmark = pytest.mark.skipif(not BENCHMARK_PDF.is_file(),
                                     reason="benchmark PDF not available")
needs_springer = pytest.mark.skipif(not SPRINGER_BOOK.is_file(),
                                    reason="Springer sample PDF not available")


class FakeDoc:
    """A document that is nothing but page texts — no file, no backend."""

    def __init__(self, texts, unreadable=()):
        self._texts = list(texts)
        self._unreadable = set(unreadable)   # 0-based indices that raise

    @property
    def page_count(self) -> int:
        return len(self._texts)

    def page(self, index: int):
        if index in self._unreadable:
            raise RuntimeError(f"page {index} is damaged")
        return FakeTextPage(self._texts[index])


class FakeTextPage:
    def __init__(self, text):
        self._text = text

    def text(self) -> str:
        return self._text

    def leaders(self):
        """No dot leaders: the scan keeps this page's own text."""
        return []


class TestPageOnePrecedence:
    """Page 1 decides the DOI whenever it has one — the article path is unchanged."""

    def test_page_1_doi_wins_over_a_later_one(self):
        found = scan_front_matter(FakeDoc([
            "doi: 10.1016/j.first.2005",
            "",
            "https://doi.org/10.9999/later.one",
        ]))
        assert found.doi == "10.1016/j.first.2005"
        assert found.doi_confidence == 0.98
        assert found.doi_page == 1

    def test_a_labelled_doi_deeper_in_is_taken_when_page_1_is_silent(self):
        found = scan_front_matter(FakeDoc([
            "Half title page",
            "Series page",
            "",
            "",
            "ISBN 978-3-030-80613-2\nhttps://doi.org/10.1007/978-3-030-80614-9",
        ]))
        assert found.doi == "10.1007/978-3-030-80614-9"
        assert found.doi_page == 5


class TestCitationGuard:
    """A bare DOI past page 1 is a reference to someone else's work."""

    def test_a_bare_doi_after_page_1_is_refused(self):
        found = scan_front_matter(FakeDoc([
            "A title page with no identifier at all",
            "As shown previously (10.1016/j.someone.else.2003) the fibrils align.",
        ]))
        assert found.doi is None
        assert found.doi_page is None

    def test_a_bare_doi_on_page_1_is_still_accepted(self):
        found = scan_front_matter(FakeDoc(["10.1016/j.ours.2005"]))
        assert found.doi == "10.1016/j.ours.2005"
        assert found.doi_confidence == 0.85
        assert found.doi_page == 1


class TestIssn:
    """ISSN must be labelled. Both false positives below are from the real corpus."""

    @pytest.mark.parametrize("text,expected", [
        ("ISSN 0065-2598", "0065-2598"),
        ("ISSN: 1065-6995", "1065-6995"),
        ("e-ISSN 2214-4501", "2214-4501"),
        ("ISSN-L 0070-2153", "0070-2153"),
        ("ISSN 0000-002X", "0000-002X"),          # X check character
        ("ISSN 00652598", "0065-2598"),           # normalised on the way out
    ])
    def test_labelled_issn_forms(self, text, expected):
        assert scan_front_matter(FakeDoc([text])).issn == expected

    @pytest.mark.parametrize("text", [
        "Proceedings of the 2014-2015 session",   # a year range
        "ORCID 0000-0003-1234-5678",              # an ORCID
        "Grant 1234-5678 funded this work",       # a grant number
    ])
    def test_unlabelled_lookalikes_are_refused(self, text):
        assert scan_front_matter(FakeDoc([text])).issn is None

    def test_the_page_is_recorded(self):
        found = scan_front_matter(FakeDoc(["", "", "", "", "ISSN 0065-2598"]))
        assert found.issn == "0065-2598"
        assert found.issn_page == 5


class TestIsbn:
    def test_isbn_is_found_on_the_copyright_page(self):
        found = scan_front_matter(FakeDoc(["Half title", "", "", "",
                                           "ISBN-13: 978-3-030-80613-2"]))
        assert found.isbn == "9783030806132"
        assert found.isbn_confidence == 0.95
        assert found.isbn_page == 5

    def test_the_first_isbn_wins(self):
        found = scan_front_matter(FakeDoc(["ISBN 978-3-030-80613-2",
                                           "ISBN 978-1-111-11111-1"]))
        assert found.isbn == "9783030806132"
        assert found.isbn_page == 1


class TestScanBounds:
    def test_the_scan_stops_at_max_pages(self):
        pages = [""] * 12
        pages[10] = "ISBN-13: 978-3-030-80613-2"    # page 11, past the default
        found = scan_front_matter(FakeDoc(pages))
        assert found.pages_scanned == DEFAULT_FRONT_MATTER_PAGES
        assert found.isbn is None

    def test_max_pages_is_clamped_to_a_short_document(self):
        found = scan_front_matter(FakeDoc(["only one page"]), max_pages=10)
        assert found.pages_scanned == 1

    def test_page_1_text_is_reused_rather_than_re_read(self):
        doc = FakeDoc(["THE FILE SAYS THIS", "page two"])
        found = scan_front_matter(doc, page_1_text="doi: 10.1016/j.cached.2005")
        assert found.doi == "10.1016/j.cached.2005"
        assert found.page_texts[1] == "doi: 10.1016/j.cached.2005"

    def test_a_damaged_page_does_not_abort_the_scan(self):
        doc = FakeDoc(["", "", "ISBN-13: 978-3-030-80613-2"], unreadable={1})
        found = scan_front_matter(doc)
        assert found.pages_scanned == 3
        assert found.isbn == "9783030806132"

    def test_identifier_pages_omits_what_was_not_found(self):
        found = FrontMatterIdentifiers(doi="10.1/x", doi_page=1, issn="0065-2598",
                                       issn_page=5)
        assert found.identifier_pages() == {"doi": 1, "issn": 5}


@needs_benchmark
class TestBenchmarkIsUnchanged:
    """The parity guard: the article path must behave exactly as page-1-only did."""

    def test_the_nmr_benchmark_doi_still_comes_from_page_1(self):
        from src.processing.pdf import get_backend

        with get_backend().open(BENCHMARK_PDF) as doc:
            found = scan_front_matter(doc)

        assert found.doi == "10.1016/j.cellbi.2005.09.009"
        assert found.doi_confidence == 0.98
        assert found.doi_page == 1
        assert found.isbn is None

    def test_the_profile_carries_the_new_fields(self):
        from run_prajnya import build_document_profile
        from src.processing.pdf import get_backend

        profile = build_document_profile(BENCHMARK_PDF, get_backend())
        assert profile.doi == "10.1016/j.cellbi.2005.09.009"
        assert profile.doi_confidence == 0.98
        assert profile.isbn is None
        assert profile.identifier_pages == {"doi": 1}
        assert profile.front_matter_pages_scanned == 8


@needs_springer
class TestSpringerFrontMatter:
    """
    The file that motivated this work.

    `Tendon assembly 2021 [1-7].pdf` is the front matter of a Springer volume.
    Page 1 is a half-title and carries no identifier of any kind; everything
    the document declares about itself sits on the copyright page, page 5.
    Before this scan, V6 saw nothing here.
    """

    def test_all_three_identifiers_come_from_the_copyright_page(self):
        from src.processing.pdf import get_backend

        with get_backend().open(SPRINGER_BOOK) as doc:
            found = scan_front_matter(doc)

        assert found.doi == "10.1007/978-3-030-80614-9"
        assert found.doi_page == 5
        assert found.isbn == "9783030806132"
        assert found.isbn_page == 5
        # A book SERIES carries an ISSN legitimately. Recorded, not interpreted
        # — reading this as "therefore a journal article" is the single defect
        # that decides V5's classifier wrongly on this exact file.
        assert found.issn == "0065-2598"
        assert found.issn_page == 5

    def test_page_1_alone_would_have_found_nothing(self):
        from src.processing.pdf import get_backend

        with get_backend().open(SPRINGER_BOOK) as doc:
            found = scan_front_matter(doc, max_pages=1)

        assert (found.doi, found.isbn, found.issn) == (None, None, None)
