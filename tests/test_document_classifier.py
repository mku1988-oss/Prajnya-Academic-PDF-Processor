#!/usr/bin/env python3
"""
Tests for document classification.

The centre of this file is one PDF: `Tendon assembly 2021 [1-7].pdf`, the
front matter of a Springer volume. V5's classifier called it
`research_article` at 0.60, and the three causes were measured by running V5
directly, not guessed. Each has a test here that fails if the cause returns:

  1. an ISSN scored for `research_article` unconditionally, although a book
     SERIES carries one legitimately (`TestIssnInContext`);
  2. contents pages were found only by the words "table of contents", which a
     real book's contents page does not carry — it has dot leaders
     (`TestDotLeaders`);
  3. neither of the above is fixed by scanning more pages (`TestScanDepth`).

Two further defects were found by measuring THIS classifier on the corpus,
and are pinned by `TestInlineContents` and `TestBookDoiShape`.

`TestTheCorpus` runs the real files: 18 PDFs, 9 documents in both full and
extracted form. Classification must agree between a document and its extract
— if it does not, the classifier is reading length rather than evidence.
"""

from pathlib import Path

import pytest

from src.processing.classification import classify_document, gather_classification_signals
from src.processing.classification.document_classifier import (
    _looks_like_a_book_doi,
    _score_page_count,
)
from src.processing.extractors.front_matter import FrontMatterIdentifiers, scan_front_matter
from src.processing.models import DocumentType
from tests.local_samples import SAMPLES as LOCAL_SAMPLES

SAMPLES = LOCAL_SAMPLES

# Ground truth for the sample corpus. Everything not named here is an article.
CORPUS_TRUTH = {
    "Tendon assembly 2021": DocumentType.BOOK,
    "CollagenFibrilAssemblyandFunction": DocumentType.BOOK_CHAPTER,
}

needs_corpus = pytest.mark.skipif(not SAMPLES.is_dir(),
                                  reason="sample corpus not available")

# A book contents page as it is actually typeset: no "Table of Contents"
# heading anywhere, just a roman page number and dot leaders.
REAL_TOC_PAGE = """v

1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . 1
2 Tendon Assembly . . . . . . . . . . . . . . . . . . . . .  23
3 Collagen Fibrils  . . . . . . . . . . . . . . . . . . . . .  47
Index . . . . . . . . . . . . . . . . . . . . . . . . . . . . 301
"""

SPRINGER_COPYRIGHT_PAGE = """
Advances in Experimental Medicine and Biology
ISSN 0065-2598
ISBN 978-3-030-80613-2
https://doi.org/10.1007/978-3-030-80614-9
Volume 1348
"""


def _identifiers(**kwargs) -> FrontMatterIdentifiers:
    return FrontMatterIdentifiers(**kwargs)


class TestIssnInContext:
    """V5 defect 1: an ISSN is not by itself evidence of a journal article."""

    def test_an_issn_beside_an_isbn_scores_nothing_for_an_article(self):
        outcome = classify_document(
            {1: "", 5: SPRINGER_COPYRIGHT_PAGE, 6: REAL_TOC_PAGE},
            page_count=345,
            identifiers=_identifiers(issn="0065-2598", isbn="9783030806132",
                                     doi="10.1007/978-3-030-80614-9"))
        assert outcome.document_type is DocumentType.BOOK
        assert outcome.scores["book"] > outcome.scores["research_article"]

    def test_an_issn_standing_alone_still_argues_for_an_article(self):
        with_issn = classify_document({1: "A paper"}, page_count=8,
                                      identifiers=_identifiers(issn="1065-6995"))
        without = classify_document({1: "A paper"}, page_count=8)
        assert (with_issn.scores["research_article"]
                > without.scores["research_article"])

    def test_a_series_issn_is_recognised_by_the_doi_alone(self):
        """No ISBN printed — an Elsevier chapter's page has only a book DOI."""
        signals = gather_classification_signals(
            {1: "ISSN 0070-2153"}, 36,
            _identifiers(issn="0070-2153", doi="10.1016/bs.ctdb.2018.02.004"))
        assert signals.metadata["has_book_doi"] is True


class TestDotLeaders:
    """V5 defect 2: a book's contents page never says 'table of contents'."""

    def test_dot_leaders_are_recognised_without_the_words(self):
        signals = gather_classification_signals({1: REAL_TOC_PAGE}, 345)
        assert signals.structure["toc_pages"] == [1]
        assert signals.structure["dot_leader_pages"] == [1]
        assert "contents" not in REAL_TOC_PAGE.lower()

    def test_a_stray_ellipsis_in_prose_is_not_a_contents_page(self):
        prose = ("The fibrils align... and the result follows.\n"
                 "As shown ... in 2018\n")
        signals = gather_classification_signals({1: prose}, 20)
        assert signals.structure["toc_pages"] == []

    def test_the_words_still_work_when_they_are_present(self):
        signals = gather_classification_signals({3: "Table of Contents\nChapter listing"}, 200)
        assert signals.structure["toc_pages"] == [3]


class TestScanDepth:
    """V5 defect 3: the evidence is on page 5-7, not page 1."""

    def test_page_1_alone_cannot_classify_the_springer_volume(self):
        outcome = classify_document({1: "Tendon Assembly and Regeneration"},
                                    page_count=345)
        # Length alone still calls it a book, but on no real evidence.
        assert outcome.signals.structure["toc_pages"] == []
        assert outcome.signals.metadata["has_isbn"] is False

    def test_the_deeper_pages_carry_the_decision(self):
        outcome = classify_document(
            {1: "Tendon Assembly", 5: SPRINGER_COPYRIGHT_PAGE, 6: REAL_TOC_PAGE},
            page_count=345,
            identifiers=_identifiers(isbn="9783030806132", issn="0065-2598",
                                     doi="10.1007/978-3-030-80614-9"))
        assert outcome.document_type is DocumentType.BOOK
        assert outcome.confidence >= 0.9


class TestInlineContents:
    """
    A chapter's own contents list is not a book's contents page.

    Found by measuring: an Elsevier chapter opens with "Contents / 1.
    Introduction 108 …" on the same page as its title, authors and abstract.
    Read as a book TOC, it pushed the chapter into `research_article`.
    """

    CHAPTER_OPENING = """CHAPTER THREE
Collagen Fibril Assembly and Function
Contents
1. Introduction 108
1.1 Fibrillar Collagens 109
Abstract
Collagen fibrils are the major mechanical component of the matrix.
"""

    def test_contents_sharing_a_page_with_an_abstract_is_not_a_book_toc(self):
        signals = gather_classification_signals({1: self.CHAPTER_OPENING}, 36)
        assert signals.structure["toc_pages"] == []
        assert signals.structure["inline_contents_pages"] == [1]

    def test_it_counts_towards_the_chapter_instead(self):
        outcome = classify_document(
            {1: self.CHAPTER_OPENING}, page_count=36,
            identifiers=_identifiers(doi="10.1016/bs.ctdb.2018.02.004",
                                     issn="0070-2153"))
        assert outcome.document_type is DocumentType.BOOK_CHAPTER

    def test_a_standalone_contents_page_still_means_a_book(self):
        signals = gather_classification_signals({6: REAL_TOC_PAGE}, 345)
        assert signals.structure["toc_pages"] == [6]
        assert signals.structure["inline_contents_pages"] == []


class TestBookDoiShape:
    """A DOI's own shape can say 'book' when no ISBN is printed."""

    @pytest.mark.parametrize("doi", [
        "10.1007/978-3-030-80614-9",        # Springer, ISBN as the suffix
        "10.1016/bs.ctdb.2018.02.004",      # Elsevier, 'bs' = book series
        "10.1007/b97264",                   # Springer, 'b' plus digits
    ])
    def test_book_shaped_dois(self, doi):
        assert _looks_like_a_book_doi(doi) is True

    @pytest.mark.parametrize("doi", [
        "10.1016/j.cellbi.2005.09.009",     # Elsevier journal
        "10.1038/s41578-020-00251-2",       # Nature
        "10.1042/BJ20151169",               # Biochemical Journal
        "10.1021/acs.jpcb.9b07555",         # ACS
        None,
    ])
    def test_journal_dois_are_not_book_shaped(self, doi):
        assert _looks_like_a_book_doi(doi) is False


class TestPageCountIsWeakEvidence:
    """
    A book's front matter extracted to its own PDF is a handful of pages.

    V5 gave "+20 research_article" to anything under 5 pages, which fought the
    correct answer on exactly such a file. Structure must outrank length.
    """

    @pytest.mark.parametrize("page_count,expected", [
        # V5 gave a sub-5-page document +20 for research_article, twice what a
        # mid-length one got. That asymmetry is removed: a short PDF is just
        # as likely to be an extract of a long work.
        (3, {"research_article": 10, "book_chapter": 0, "book": 0}),
        (8, {"research_article": 10, "book_chapter": 10, "book": 0}),
        (36, {"research_article": 10, "book_chapter": 10, "book": 0}),
        (345, {"research_article": -20, "book_chapter": 0, "book": 40}),
    ])
    def test_the_page_count_contribution_is_pinned(self, page_count, expected):
        scores = {"research_article": 0, "book_chapter": 0, "book": 0}
        _score_page_count(scores, page_count)
        assert scores == expected

    def test_a_seven_page_book_extract_is_still_a_book(self):
        outcome = classify_document(
            {1: "Half title", 5: SPRINGER_COPYRIGHT_PAGE, 6: REAL_TOC_PAGE,
             7: REAL_TOC_PAGE},
            page_count=7,
            identifiers=_identifiers(isbn="9783030806132", issn="0065-2598",
                                     doi="10.1007/978-3-030-80614-9"))
        assert outcome.document_type is DocumentType.BOOK


class TestTheDefault:
    """An honest 0.50 beats a fabricated 0.60."""

    def test_a_scanned_pdf_with_no_text_layer_is_admitted_as_a_guess(self):
        """Length is always available. It must not masquerade as evidence."""
        outcome = classify_document({1: "", 2: "", 3: ""}, page_count=3)
        assert outcome.document_type is DocumentType.RESEARCH_ARTICLE
        assert outcome.confidence == 0.50
        assert outcome.is_confident is False

    def test_a_long_scanned_pdf_is_equally_a_guess(self):
        outcome = classify_document({1: ""}, page_count=400)
        assert outcome.document_type is DocumentType.RESEARCH_ARTICLE
        assert outcome.confidence == 0.50

    def test_a_clear_article_is_confident(self):
        outcome = classify_document(
            {1: "Abstract\nReceived: 3 May 2005; Accepted: 9 September 2005"},
            page_count=8, identifiers=_identifiers(doi="10.1016/j.cellbi.2005.09.009"))
        assert outcome.document_type is DocumentType.RESEARCH_ARTICLE
        assert outcome.is_confident is True

    def test_confidence_tracks_the_margin_over_the_runner_up(self):
        outcome = classify_document({1: "Abstract"}, page_count=8)
        margin = outcome.scores["research_article"] - max(
            outcome.scores["book_chapter"], outcome.scores["book"])
        assert outcome.confidence == pytest.approx(min(1.0, 0.5 + margin / 50.0), abs=0.01)


@needs_corpus
class TestTheCorpus:
    """
    The real files. 9 documents, each present in full and as an extract.

    The full/extract agreement is the point: a classifier that reads length
    rather than evidence gets these two forms of the same document wrong in
    opposite directions.
    """

    @staticmethod
    def _classify(pdf: Path):
        from src.processing.pdf import get_backend

        with get_backend().open(pdf) as doc:
            identifiers = scan_front_matter(doc)
            return classify_document(identifiers.page_texts, doc.page_count, identifiers)

    @staticmethod
    def _expected(pdf: Path) -> DocumentType:
        for fragment, truth in CORPUS_TRUTH.items():
            if fragment in pdf.name:
                return truth
        return DocumentType.RESEARCH_ARTICLE

    @pytest.mark.parametrize("folder", ["Full", "Extracted"])
    def test_every_sample_is_classified_correctly(self, folder):
        pdfs = sorted((SAMPLES / folder).glob("*.pdf"))
        if not pdfs:
            pytest.skip(f"no PDFs in {folder}")

        wrong = []
        for pdf in pdfs:
            outcome = self._classify(pdf)
            if outcome.document_type is not self._expected(pdf):
                wrong.append(f"{pdf.name}: {outcome.document_type.value} "
                             f"{outcome.confidence:.2f} {outcome.scores}")
        assert not wrong, "misclassified:\n" + "\n".join(wrong)

    def test_the_springer_volume_is_a_book_not_an_article(self):
        """The regression case. V5 said research_article at 0.60."""
        pdf = SAMPLES / "Extracted" / "Tendon assembly 2021 [1-7].pdf"
        if not pdf.is_file():
            pytest.skip("Springer sample not available")

        outcome = self._classify(pdf)
        assert outcome.document_type is DocumentType.BOOK
        assert outcome.is_confident
        assert outcome.signals.structure["toc_pages"] == [6, 7]
        assert outcome.signals.metadata["isbn"] == "9783030806132"

    def test_the_profile_records_the_classification_and_its_evidence(self):
        from run_prajnya import build_document_profile
        from src.processing.pdf import get_backend

        pdf = SAMPLES / "Extracted" / "Tendon assembly 2021 [1-7].pdf"
        if not pdf.is_file():
            pytest.skip("Springer sample not available")

        profile = build_document_profile(pdf, get_backend())
        assert profile.document_type is DocumentType.BOOK
        assert profile.classification_confidence >= 0.9
        assert profile.classification_signals.structure["toc_pages"] == [6, 7]
        assert profile.identifier_pages == {"doi": 5, "isbn": 5, "issn": 5}

    def test_the_nmr_benchmark_is_still_an_article(self):
        """The parity guard — the benchmark must not move."""
        pdf = SAMPLES / "Full" / (
            "An NMR method to characterize multiple water compartments "
            "on mammalian collagen.pdf")
        if not pdf.is_file():
            pytest.skip("benchmark not available")

        outcome = self._classify(pdf)
        assert outcome.document_type is DocumentType.RESEARCH_ARTICLE
        assert outcome.is_confident


class TestFigureRowsAreNotContentsEntries:
    """Session 33: a Nature Communications article became a BOOK because a
    figure on page 7 set data rows with dot leaders, read as a contents page."""

    FIGURE_PAGE = ("a b . . . . . 50\n"
                   "Nile tilapia 5 4 . . . . . 4\n"
                   "ZfWnt3-3997 + + . . . . . 0\n")
    CONTENTS_PAGE = ("1 Introduction . . . . . . . . . . . . 1\n"
                     "2 Tendon Assembly . . . . . . . . . . 23\n"
                     "3 Collagen Fibrils . . . . . . . . . 47\n"
                     "Index . . . . . . . . . . . . . . . . 301\n")

    def test_figure_data_rows_do_not_count(self):
        from src.processing.classification.document_classifier import _count_toc_evidence
        assert _count_toc_evidence(self.FIGURE_PAGE) == 0

    def test_real_contents_entries_still_count(self):
        from src.processing.classification.document_classifier import _count_toc_evidence
        assert _count_toc_evidence(self.CONTENTS_PAGE) == 4
