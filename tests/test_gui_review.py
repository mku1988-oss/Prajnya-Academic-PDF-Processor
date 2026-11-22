#!/usr/bin/env python3
"""
Review Queue page — the pure helpers, tested without a browser.

For biology researchers: `mount()` needs a live NiceGUI page context (it
builds real ui.dialog/ui.card elements), so it is not unit-tested directly
here — the same split test_gui_library.py and test_gui_ledger.py document.
What is tested here are the pure functions: the flagged-reason line, the
textarea-to-list splitter, the document-JSON path guess, and the
in-memory queue update after a correction (so a still-incomplete record
stays visible with its new values, and a resolved one drops out) — all of
which lean on `src/processing/review.py`, the shared seam this screen
never re-implements.
"""

from __future__ import annotations

from pathlib import Path

from src.gui.pages.review import _document_json_path, _lines, _reason, _replace_or_drop
from src.processing.ledger.semantic_ledger import LedgerRecord


def _record(**overrides) -> LedgerRecord:
    defaults = dict(
        document_id="d1", pdf_filename="paper.pdf",
        validation_confidence=0.5, validation_source="test",
        last_updated="2026-08-11T00:00:00", source_hash="d1",
        document_type="research_article",
    )
    defaults.update(overrides)
    return LedgerRecord(**defaults)


class TestReason:
    def test_missing_fields_are_listed(self):
        record = _record(review_needed_fields=["doi", "journal"])
        assert _reason(record) == "Missing: doi, journal"

    def test_no_missing_fields_falls_back_to_generic_flag(self):
        record = _record(review_needed_fields=[], requires_review=True)
        assert _reason(record) == "Flagged for review"


class TestLines:
    def test_splits_on_newline_not_comma(self):
        # A raw comma-split (what apply_corrections' _as_string_list does for
        # a plain string) would wrongly treat "A, Jr., B" as three names; the
        # textarea's newline convention keeps one name per line intact.
        assert _lines("Smith, Jr., A.\nDoe, B.") == ["Smith, Jr., A.", "Doe, B."]

    def test_blank_lines_are_dropped(self):
        assert _lines("A\n\n  \nB\n") == ["A", "B"]

    def test_empty_text_is_empty_list(self):
        assert _lines("") == []


class TestDocumentJsonPath:
    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        import src.gui.pages.review as review_page
        monkeypatch.setattr(review_page, "DEFAULT_OUTPUT_ROOT", tmp_path)
        record = _record(pdf_filename="nope.pdf")
        assert _document_json_path(record) is None

    def test_existing_file_is_found_by_stem(self, tmp_path, monkeypatch):
        import src.gui.pages.review as review_page
        monkeypatch.setattr(review_page, "DEFAULT_OUTPUT_ROOT", tmp_path)
        (tmp_path / "documents").mkdir()
        (tmp_path / "documents" / "paper.v6.json").write_text("{}")
        record = _record(pdf_filename="paper.pdf")
        assert _document_json_path(record) == tmp_path / "documents" / "paper.v6.json"

    def test_no_filename_returns_none(self, tmp_path, monkeypatch):
        import src.gui.pages.review as review_page
        monkeypatch.setattr(review_page, "DEFAULT_OUTPUT_ROOT", tmp_path)
        assert _document_json_path(_record(pdf_filename="")) is None


class TestReplaceOrDrop:
    def test_resolved_record_is_dropped_from_the_queue(self):
        original = _record(title="Old", authors=["A"], doi="10.1/x", journal="J",
                           review_needed_fields=["journal"])
        updated = original.model_copy(update={"journal": "J", "review_needed_fields": [],
                                               "requires_review": False,
                                               "review_status": "reviewed"})
        ui_state = {"records": [original]}
        _replace_or_drop(ui_state, original, updated)
        assert ui_state["records"] == []

    def test_still_incomplete_record_stays_with_new_values(self):
        original = _record(title="Old", review_needed_fields=["doi", "journal"])
        updated = original.model_copy(update={"title": "New", "review_needed_fields": ["doi"],
                                               "requires_review": True,
                                               "review_status": "provisional"})
        ui_state = {"records": [original]}
        _replace_or_drop(ui_state, original, updated)
        assert ui_state["records"] == [updated]

    def test_other_records_are_untouched(self):
        original = _record(document_id="a", review_needed_fields=["doi"])
        other = _record(document_id="b", review_needed_fields=["doi"])
        updated = original.model_copy(update={"review_needed_fields": [], "requires_review": False,
                                               "review_status": "reviewed"})
        ui_state = {"records": [original, other]}
        _replace_or_drop(ui_state, original, updated)
        assert ui_state["records"] == [other]
