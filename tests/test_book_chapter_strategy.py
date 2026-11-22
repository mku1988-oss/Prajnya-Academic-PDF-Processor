#!/usr/bin/env python3
"""
Tests for the book-chapter strategy.

A chapter reads like a paper, so the page reading is delegated to the
research-article pipeline; what this strategy owns is the chapter's identity.
The tests therefore inject a fake reader — the article path has its own parity
gate and does not need re-testing here — and check what gets stamped on top.

The chapter-number case is the Elsevier sample, which prints "CHAPTER THREE"
in words. V5's extractor only understood numerals and roman numerals, so the
word form returned nothing.
"""

from pathlib import Path

import pytest

from src.processing.models import (
    ContentPage,
    DocumentProfile,
    ExtractedMetadata,
    ProcessingResult,
)
from src.processing.strategies.book_chapter.strategy import (
    BookChapterStrategy,
    _numeralize_chapter_words,
)

CONFIG_ROOT = Path(__file__).parent.parent / "config"

ELSEVIER_OPENING = """CHAPTER THREE
Collagen Fibril Assembly and Function
David F. Holmes, Yinhui Lu, Tobias Starborg, Karl E. Kadler
Abstract
Collagen fibrils are the major mechanical component in the matrix.
Current Topics in Developmental Biology, Volume 130
ISSN 0070-2153
https://doi.org/10.1016/bs.ctdb.2018.02.004
"""


class FakeArticleStrategy:
    """Stands in for the delegated reader — records that it was called."""

    def __init__(self):
        self.calls = []

    def process(self, profile, test_pages_override=None):
        self.calls.append(test_pages_override)
        profile.processing_result = ProcessingResult(
            metadata=ExtractedMetadata(title="Collagen Fibril Assembly and Function",
                                       authors=["David F. Holmes"]),
            pages=[ContentPage(page_number=1, content="body text")],
        )
        return profile


@pytest.fixture()
def profile(tmp_path):
    pdf = tmp_path / "chapter.pdf"
    pdf.write_bytes(b"%PDF-fake")
    return DocumentProfile(
        pdf_path=pdf, page_count=36, file_size=9,
        page_1_text=ELSEVIER_OPENING,
        doi="10.1016/bs.ctdb.2018.02.004", doi_confidence=0.95,
        issn="0070-2153", identifier_pages={"doi": 1, "issn": 1},
    )


@pytest.fixture()
def strategy():
    return BookChapterStrategy(CONFIG_ROOT, article_strategy=FakeArticleStrategy())


class TestDelegation:
    def test_the_page_reading_is_delegated(self, strategy, profile):
        result = strategy.process(profile).processing_result
        assert strategy.reader.calls == [None]
        assert result.metadata.title == "Collagen Fibril Assembly and Function"
        assert [page.page_number for page in result.pages] == [1]

    def test_the_page_override_is_passed_through(self, strategy, profile):
        strategy.process(profile, test_pages_override=[1, 2])
        assert strategy.reader.calls == [[1, 2]]


class TestChapterIdentity:
    def test_a_word_chapter_number_is_read(self, strategy, profile):
        """'CHAPTER THREE' — the Elsevier form V5's numeral extractor missed."""
        result = strategy.process(profile).processing_result
        assert result.metadata.chapter_number == 3

    def test_the_parent_volumes_identifiers_are_recorded(self, strategy, profile):
        profile.isbn = "9780128098028"
        result = strategy.process(profile).processing_result
        assert result.metadata.isbn == "9780128098028"
        assert result.metadata.provenance["issn"] == "0070-2153"
        assert result.metadata.provenance["document_type"] == "book_chapter"

    def test_a_chapter_with_no_heading_gets_no_number(self, strategy, profile):
        profile.page_1_text = "A chapter with no numbered heading at all."
        result = strategy.process(profile).processing_result
        assert result.metadata.chapter_number is None

    def test_the_delegated_metadata_is_not_overwritten(self, strategy, profile):
        """Identity is added to the read, never in place of it."""
        result = strategy.process(profile).processing_result
        assert result.metadata.authors == ["David F. Holmes"]


class TestWordNumerals:
    @pytest.mark.parametrize("text,expected", [
        ("CHAPTER THREE", "Chapter 3"),
        ("Chapter Twelve: Assembly", "Chapter 12: Assembly"),
        ("chapter one", "Chapter 1"),
        ("Chapter 4", "Chapter 4"),          # numerals pass through untouched
        ("a chapter of the book", "a chapter of the book"),
    ])
    def test_word_numerals_are_rewritten(self, text, expected):
        assert _numeralize_chapter_words(text) == expected
