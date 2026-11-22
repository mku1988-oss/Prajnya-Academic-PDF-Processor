#!/usr/bin/env python3
"""
Audit D1: a document that lost pages is "partial", and is tried again.

For biology researchers: before v6.8 a page whose reading raised an error was
written empty, the output still said `success: true`, and the ledger marked
the document done, so the next run skipped it. PMC10904388 lost its page 4
this way in every benchmark run (D9). These tests pin the three places that
must agree: the document JSON, the batch line, and the ledger.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.processing import runner
from src.processing.document_status import (
    FAILED, PARTIAL, SUCCESS, document_status, failed_page_numbers)
from src.processing.models import (
    ContentPage, DocumentProfile, ExtractedMetadata, ProcessingResult)
from src.processing.run_request import DocumentOutcome
from src.processing.worker import _outcome_from_dict, _outcome_to_dict


def page(number: int, method: str = "pymupdf") -> ContentPage:
    return ContentPage(page_number=number, content=f"page {number}",
                       metadata={"processing_method": method})


class TestTheStatus:
    def test_every_page_read_is_a_success(self):
        assert document_status([], 3, []) == SUCCESS

    def test_a_lost_page_makes_it_partial(self):
        assert document_status([], 3, [2]) == PARTIAL

    def test_a_strategy_error_or_no_pages_is_a_failure(self):
        assert document_status(["Strategy execution failed: boom"], 3, []) == FAILED
        assert document_status([], 0, []) == FAILED

    def test_failed_pages_are_found_on_models_and_on_dicts(self):
        pages = [page(1, "two_shot_title_failed"), page(2), page(4, "glm_failed")]
        assert failed_page_numbers(pages) == [1, 4]
        assert failed_page_numbers([p.model_dump() for p in pages]) == [1, 4]


class TestTheDocumentJsonSaysSo:
    @staticmethod
    def build(tmp_path, pages, errors=()):
        from run_prajnya import build_document_json
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        profile = DocumentProfile(pdf_path=pdf, page_count=len(pages), file_size=9)
        profile.processing_result = ProcessingResult(
            metadata=ExtractedMetadata(title="A paper"), pages=pages, errors=list(errors))
        return build_document_json(profile, elapsed=1.0)["document_info"]

    def test_a_complete_document(self, tmp_path):
        info = self.build(tmp_path, [page(1), page(2)])
        assert (info["status"], info["failed_pages"], info["success"]) == (SUCCESS, [], True)

    def test_a_lost_page_is_partial_while_success_keeps_its_meaning(self, tmp_path):
        info = self.build(tmp_path, [page(1), page(2, "glm_failed")])
        assert (info["status"], info["failed_pages"], info["success"]) == (PARTIAL, [2], True)

    def test_a_strategy_error_is_failed(self, tmp_path):
        info = self.build(tmp_path, [page(1)], errors=["boom"])
        assert (info["status"], info["success"]) == (FAILED, False)


class TestTheLedgerOnlyRecordsASuccess:
    @pytest.fixture
    def recorded(self, monkeypatch):
        """Every call `_result_for` makes to write a ledger record."""
        calls = []

        class Record:
            review_status, requires_review = "not_reviewed", False

        def fake_record(*args, **kwargs):
            calls.append(args[0])
            return Record()

        monkeypatch.setattr(runner, "record_outcome", fake_record)
        return calls

    def test_a_partial_document_is_not_recorded_and_says_why(self, recorded):
        outcome = DocumentOutcome(path=Path("a.pdf"), output=Path("a.json"),
                                  failed_pages=[4], page_count=12)
        line = runner._result_for(outcome, ledger=object(), log=runner.LOG)
        assert recorded == []
        assert line["status"] == PARTIAL and line["failed_pages"] == [4]
        assert line["output"] == "a.json"

    def test_a_strategy_error_is_not_recorded(self, recorded):
        outcome = DocumentOutcome(path=Path("a.pdf"), output=Path("a.json"),
                                  errors=["Strategy execution failed: boom"], page_count=3)
        assert runner._result_for(outcome, ledger=object(), log=runner.LOG)["status"] == FAILED
        assert recorded == []

    def test_a_complete_document_is_recorded(self, recorded):
        outcome = DocumentOutcome(path=Path("a.pdf"), output=Path("a.json"), page_count=3)
        line = runner._result_for(outcome, ledger=object(), log=runner.LOG)
        assert recorded == [Path("a.pdf")]
        assert line["status"] == SUCCESS

    def test_the_status_survives_the_trip_from_the_worker(self):
        """Prajnya Studio runs each document in a child process; the parent
        writes the ledger from the child's report, so the report must carry
        the failed pages."""
        outcome = DocumentOutcome(path=Path("a.pdf"), failed_pages=[4], page_count=12)
        back = _outcome_from_dict(_outcome_to_dict(outcome))
        assert (back.status, back.failed_pages, back.page_count) == (PARTIAL, [4], 12)
