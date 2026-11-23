#!/usr/bin/env python3
"""
Tests for `utils/contents_leaders.py`: a contents page rebuilt from geometry.

For biology researchers: pypdfium2 5.x stopped extracting the dots between a
chapter title and its page number, so the contents page could no longer be
read from its text. These tests pin the geometric rebuild that replaces it:
the dots are found as a page OBJECT filling the gap, and written back as a
standard leader.

The unit tests use hand-built pages (`FakePage`), so the geometry under test
is explicit. The real-PDF tests pin the Springer book and the journal corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.processing.pdf.base import Rect, Word
from src.processing.utils.contents_leaders import (
    LEADER_TEXT,
    MIN_LEADER_LINES,
    contents_page_text,
)
from tests.fakes import FakePage
from tests.local_samples import SAMPLES as LOCAL_SAMPLES

SAMPLES = LOCAL_SAMPLES
SPRINGER_EXTRACT = SAMPLES / "Extracted" / "Tendon assembly 2021 [1-7].pdf"
CORPUS = SAMPLES / "Full"


# --- building blocks ---------------------------------------------------------

def _word(text: str, x0: float, y: float, width: float = None) -> Word:
    """A word on the line whose top is `y`, 7 pt tall like the Springer page."""
    width = width if width is not None else 4.0 * len(text)
    return Word(text=text, rect=Rect(x0, y, x0 + width, y + 7.0))


def _leader(x0: float, x1: float, y: float) -> Rect:
    """A leader box as PDFium reports it: ~1 pt tall, on the line's baseline."""
    return Rect(x0, y + 5.0, x1, y + 6.0)


def _entry(title: str, page: str, y: float, with_leader: bool = True):
    """One contents line: title at the left, number at x=440, leader between."""
    words, x = [], 100.0
    for token in title.split():
        words.append(_word(token, x, y))
        x += 4.0 * len(token) + 3.0
    words.append(_word(page, 440.0, y))
    leaders = [_leader(x + 2.0, 438.0, y)] if with_leader else []
    return words, leaders


def _page(entries) -> FakePage:
    """A page from (title, page, with_leader) tuples, 20 pt apart."""
    words, leaders = [], []
    for index, (title, number, with_leader) in enumerate(entries):
        w, l = _entry(title, number, 100.0 + 20.0 * index, with_leader)
        words += w
        leaders += l
    return FakePage(words=words, leaders=leaders)


# --- unit tests --------------------------------------------------------------

class TestTheRebuild:
    def test_a_leader_is_written_between_title_and_number(self):
        page = _page([("1 Introduction", "1", True),
                      ("2 Collagens", "5", True),
                      ("3 Tendon Matrix", "45", True)])
        lines = contents_page_text(page).splitlines()
        assert lines == [f"1 Introduction {LEADER_TEXT} 1",
                         f"2 Collagens {LEADER_TEXT} 5",
                         f"3 Tendon Matrix {LEADER_TEXT} 45"]

    def test_a_line_without_a_leader_object_is_left_as_it_is(self):
        """A wrapped title line, or an author line, keeps its words."""
        page = _page([("1 Introduction", "1", True),
                      ("2 Collagens", "5", True),
                      ("3 Tendon Matrix", "45", True),
                      ("Seyed Siadat and", "7", False)])
        assert contents_page_text(page).splitlines()[-1] == "Seyed Siadat and 7"

    def test_a_number_alone_on_its_line_keeps_its_leader(self):
        """A long title can leave only the leader and number on the last line."""
        words = [_word("12", 440.0, 100.0)]
        leaders = [_leader(150.0, 438.0, 100.0)]
        for index in range(1, MIN_LEADER_LINES):
            w, l = _entry("Chapter", str(index), 100.0 + 20.0 * index)
            words += w
            leaders += l
        lines = contents_page_text(FakePage(words=words, leaders=leaders)).splitlines()
        assert lines[0] == f"{LEADER_TEXT} 12"

    def test_the_old_pdfiums_dot_words_are_dropped(self):
        """pypdfium2 4.30 extracts the dots as words; 5.x does not. Same result."""
        page = _page([("1 Introduction", "1", True),
                      ("2 Collagens", "5", True),
                      ("3 Tendon Matrix", "45", True)])
        dots = [_word(".", x, 100.0, width=2.0) for x in range(200, 430, 10)]
        with_dots = FakePage(words=page.words() + dots, leaders=page.leaders())
        assert contents_page_text(with_dots) == contents_page_text(page)


class TestWhenItDoesNotFire:
    def test_a_table_with_wide_gaps_but_no_dots_is_not_a_contents_page(self):
        """The reason the leader OBJECT is required: a gap alone is a table."""
        page = _page([("Collagen I", "12", False),
                      ("Collagen II", "7", False),
                      ("Collagen III", "31", False),
                      ("Elastin", "4", False)])
        assert contents_page_text(page) is None

    def test_letterless_runs_elsewhere_on_the_line_do_not_count(self):
        """A run of spaces or a dash ABOVE the line, or past the number, is not a leader."""
        words, leaders = [], []
        for index in range(4):
            w, _ = _entry("Collagen", str(index), 100.0 + 20.0 * index, with_leader=False)
            words += w
            leaders.append(_leader(460.0, 520.0, 100.0 + 20.0 * index))   # past the number
            leaders.append(Rect(150.0, 90.0 + 20.0 * index, 400.0, 91.0 + 20.0 * index))  # off the line
        assert contents_page_text(FakePage(words=words, leaders=leaders)) is None

    def test_too_few_leader_lines_is_not_a_contents_page(self):
        entries = [("Chapter", str(n), True) for n in range(MIN_LEADER_LINES - 1)]
        assert contents_page_text(_page(entries)) is None

    def test_a_page_with_no_leader_candidates_costs_no_word_walk(self):
        class Strict(FakePage):
            def words(self):
                raise AssertionError("words() must not be read without leaders")
        assert contents_page_text(Strict()) is None


# --- real PDFs ---------------------------------------------------------------

needs_springer = pytest.mark.skipif(not SPRINGER_EXTRACT.is_file(),
                                    reason="Springer book extract not on this machine")
needs_corpus = pytest.mark.skipif(not CORPUS.is_dir(),
                                  reason="sample corpus not on this machine")


@needs_springer
class TestTheSpringerContentsPages:
    """
    Pinned under pypdfium2 4.30 AND 5.15.2 (2026-11-13): the rebuilt pages
    parse to the same 17 entries, with the same authors, as 4.30's own text.
    """

    def test_both_contents_pages_are_rebuilt_and_parse_in_full(self):
        from src.processing.pdf import get_backend
        from src.processing.strategies.book.toc import parse_toc_entries

        with get_backend().open(SPRINGER_EXTRACT) as doc:
            entries = []
            for index in (5, 6):
                text = contents_page_text(doc.page(index))
                assert text is not None, f"page {index} not rebuilt"
                entries += parse_toc_entries(text)

        numbered = [e for e in entries if e.number is not None]
        assert len(entries) == 17
        assert [e.number for e in numbered] == list(range(1, len(numbered) + 1))
        tenascins = next(e for e in entries if "Tenascins" in e.title)
        assert tenascins.raw_page == "105"   # one plain space under 5.x's text

    def test_no_other_page_of_the_extract_is_rebuilt(self):
        from src.processing.pdf import get_backend
        with get_backend().open(SPRINGER_EXTRACT) as doc:
            fired = [i for i in range(doc.page_count)
                     if contents_page_text(doc.page(i)) is not None]
        assert fired == [5, 6]


@needs_corpus
class TestTheJournalCorpusIsUntouched:
    def test_no_journal_page_is_rebuilt(self):
        """Measured: fires only on the book's contents pages, all 485 pages."""
        from src.processing.pdf import get_backend
        fired = []
        for pdf in sorted(CORPUS.glob("*.pdf")):
            if pdf.name.startswith("Tendon assembly"):
                continue
            with get_backend().open(pdf) as doc:
                for index in range(min(10, doc.page_count)):
                    if contents_page_text(doc.page(index)) is not None:
                        fired.append((pdf.name, index))
        assert fired == []
