#!/usr/bin/env python3
"""
Tests for the CLI conveniences carried over from V5's run_Prajnya.py.

Two behaviours, both about not surprising someone who learned the tool on V5:
`--pages 2-5` must keep working, and re-running a folder must not re-read the
documents it already read.

The bias in both is the same. An unparseable page token is *skipped* rather
than guessed at — processing the wrong pages silently is worse than processing
fewer. And any failure in the ledger check answers "not processed", because
re-reading a document costs time while wrongly skipping it loses it from the
library.
"""

from pathlib import Path

import pytest

from src.processing.cli_support import (
    already_processed,
    parse_chapters,
    parse_pages,
)


class TestPageParsing:
    @pytest.mark.parametrize("given,expected", [
        (["4"], [4]),
        (["1", "4", "7"], [1, 4, 7]),
        (["2-5"], [2, 3, 4, 5]),                  # V5's range syntax
        (["1", "3-5"], [1, 3, 4, 5]),
        (["2-5,7"], [2, 3, 4, 5, 7]),
        (["7", "3", "3"], [3, 7]),                # sorted, deduplicated
        (["4-4"], [4]),
    ])
    def test_accepted_forms(self, given, expected):
        assert parse_pages(given) == expected

    def test_the_flag_being_absent_is_not_the_same_as_an_empty_selection(self):
        """None means 'every page'; [] would mean 'no pages at all'."""
        assert parse_pages(None) is None
        assert parse_pages([]) == []

    @pytest.mark.parametrize("given", [
        ["five"],
        ["5-"],          # incomplete range
        ["-5"],
        ["5-2"],         # backwards
        ["2--5"],
    ])
    def test_unparseable_tokens_are_skipped_not_guessed(self, given):
        assert parse_pages(given) == []

    def test_a_bad_token_does_not_lose_the_good_ones(self):
        assert parse_pages(["1", "nonsense", "3-4"]) == [1, 3, 4]

    def test_a_backwards_range_is_reported_not_silently_dropped(self, caplog):
        """
        "5-2" yields nothing either way — an empty range is empty. What
        matters is that the user is TOLD, rather than quietly getting a run
        over no pages and wondering why.
        """
        with caplog.at_level("WARNING"):
            assert parse_pages(["5-2"]) == []

        assert any("5-2" in record.getMessage() for record in caplog.records)


class FakeLedger:
    def __init__(self, known=(), explode=False):
        self.known = set(known)
        self.explode = explode
        self.asked = []

    def get_record(self, document_id):
        if self.explode:
            raise RuntimeError("ledger unavailable")
        self.asked.append(document_id)
        return _FakeRecord() if document_id in self.known else None


class _FakeRecord:
    review_status = "final"


@pytest.fixture()
def pdf(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7 contents")
    return path


class TestLedgerSkipping:
    @staticmethod
    def _hash(path):
        from src.processing.ledger.semantic_ledger import generate_hash

        return generate_hash(str(path))

    def test_a_known_document_is_skipped(self, pdf):
        assert already_processed(pdf, FakeLedger(known=[self._hash(pdf)])) is True

    def test_an_unknown_document_is_processed(self, pdf):
        assert already_processed(pdf, FakeLedger()) is False

    def test_no_ledger_means_nothing_is_skipped(self, pdf):
        assert already_processed(pdf, None) is False

    def test_a_broken_ledger_never_skips(self, pdf):
        """Failing safe means re-reading, never losing a document."""
        assert already_processed(pdf, FakeLedger(explode=True)) is False

    def test_identity_is_the_contents_so_renaming_does_not_fool_it(self, pdf, tmp_path):
        renamed = tmp_path / "different_name.pdf"
        renamed.write_bytes(pdf.read_bytes())

        ledger = FakeLedger(known=[self._hash(pdf)])
        assert already_processed(renamed, ledger) is True

    def test_editing_a_pdf_makes_it_new_again(self, pdf, tmp_path):
        ledger = FakeLedger(known=[self._hash(pdf)])
        edited = tmp_path / "edited.pdf"
        edited.write_bytes(b"%PDF-1.7 different contents")

        assert already_processed(edited, ledger) is False


class TestChapterParsing:
    """
    `--chapters` takes the same shorthand as `--pages`, and one answer here is
    worth hours: the difference between "not requested" and "every chapter" on
    a 345-page volume is minutes versus two hours of GPU.
    """

    def test_the_flag_absent_means_no_bodies(self):
        assert parse_chapters(None) is None

    def test_the_flag_with_no_numbers_means_every_chapter(self):
        assert parse_chapters([]) == []

    @pytest.mark.parametrize("given,expected", [
        (["3"], [3]),
        (["3", "4", "5"], [3, 4, 5]),
        (["3-5"], [3, 4, 5]),
        (["1", "3-5"], [1, 3, 4, 5]),
        (["1,3-5"], [1, 3, 4, 5]),
        (["5", "3", "3"], [3, 5]),
    ])
    def test_numbers_and_ranges(self, given, expected):
        assert parse_chapters(given) == expected

    @pytest.mark.parametrize("typo", [["abc"], ["three"], ["-"], ["--"]])
    def test_a_typo_reads_nothing_rather_than_everything(self, typo):
        """
        The safety case. Falling back to [] would read the WHOLE volume because
        someone mistyped a chapter number — the expensive answer must never be
        what a misread flag resolves to.
        """
        assert parse_chapters(typo) is None
