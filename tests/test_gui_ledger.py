#!/usr/bin/env python3
"""
Ledger page — the pure helpers, tested without a browser.

For biology researchers: `mount()` needs a live NiceGUI page context (it
builds real ui.table/ui.dialog elements), so it is not unit-tested
directly here — the same split test_gui_library.py documents. What is
tested here are the pure functions the page uses to filter records, shape
table rows, and build the CSV export, since none of those need a page.
"""

from __future__ import annotations

from src.gui.pages.ledger import _author_summary, _filtered_records, _to_row, build_csv
from src.processing.ledger.semantic_ledger import LedgerRecord


def _record(**overrides) -> LedgerRecord:
    defaults = dict(
        document_id="d1", pdf_filename="paper.pdf", title="A Study",
        authors=["A. Author"], doi="10.1/x", journal="Journal", year=2020,
        validation_confidence=0.9, validation_source="test",
        review_status="not_reviewed", last_updated="2026-08-11T00:00:00",
        source_hash="d1",
    )
    defaults.update(overrides)
    return LedgerRecord(**defaults)


class TestFilteredRecords:
    def test_no_filters_returns_everything(self):
        records = [_record(document_id="a"), _record(document_id="b")]
        assert len(_filtered_records(records, "", "all", 0.0)) == 2

    def test_search_matches_title_case_insensitive(self):
        records = [_record(title="Collagen Structure"), _record(title="NMR Method")]
        result = _filtered_records(records, "collagen", "all", 0.0)
        assert [r.title for r in result] == ["Collagen Structure"]

    def test_search_matches_author(self):
        records = [_record(authors=["Wheeler, T."]), _record(authors=["Doe, J."])]
        result = _filtered_records(records, "wheeler", "all", 0.0)
        assert len(result) == 1

    def test_search_matches_doi(self):
        records = [_record(doi="10.1016/j.cellbi.2005"), _record(doi="10.9999/other")]
        result = _filtered_records(records, "cellbi", "all", 0.0)
        assert len(result) == 1

    def test_status_filter_not_reviewed_includes_provisional(self):
        records = [
            _record(document_id="a", review_status="not_reviewed"),
            _record(document_id="b", review_status="provisional"),
            _record(document_id="c", review_status="reviewed"),
        ]
        result = _filtered_records(records, "", "not_reviewed", 0.0)
        assert {r.document_id for r in result} == {"a", "b"}

    def test_status_filter_reviewed_is_exact(self):
        records = [
            _record(document_id="a", review_status="provisional"),
            _record(document_id="b", review_status="reviewed"),
        ]
        result = _filtered_records(records, "", "reviewed", 0.0)
        assert {r.document_id for r in result} == {"b"}

    def test_confidence_threshold_excludes_below(self):
        records = [
            _record(document_id="a", validation_confidence=0.5),
            _record(document_id="b", validation_confidence=0.9),
        ]
        result = _filtered_records(records, "", "all", 0.8)
        assert {r.document_id for r in result} == {"b"}


class TestToRow:
    def test_shapes_expected_fields(self):
        record = _record(title="An NMR method", authors=["A. Author", "B. Author"],
                         validation_confidence=0.876, review_status="provisional")
        row = _to_row(record)
        assert row["title"] == "An NMR method"
        assert row["authors"] == "A. Author et al. (2)"
        assert row["review_status"] == "Provisional"
        assert row["confidence"] == 88  # rounded percentage, not a formatted string
        assert row["last_updated"] == "2026-08-11"

    def test_missing_title_and_doi_show_placeholders(self):
        row = _to_row(_record(title=None, doi=None))
        assert row["title"] == "Untitled"
        assert row["doi"] == "—"


class TestAuthorSummary:
    def test_no_authors(self):
        assert _author_summary([]) == "Unknown"

    def test_single_author(self):
        assert _author_summary(["A. Author"]) == "A. Author"

    def test_multiple_authors_shows_et_al_with_count(self):
        assert _author_summary(["A", "B", "C"]) == "A et al. (3)"


class TestBuildCsv:
    def test_header_and_one_row(self):
        content = build_csv([_record(title="A Study", authors=["A. Author"])])
        lines = content.strip().splitlines()
        assert lines[0].startswith("Title,Authors,DOI")
        assert "A Study" in lines[1]
        assert "A. Author" in lines[1]

    def test_empty_list_is_header_only(self):
        content = build_csv([])
        assert len(content.strip().splitlines()) == 1
