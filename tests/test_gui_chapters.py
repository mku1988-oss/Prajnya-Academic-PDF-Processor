#!/usr/bin/env python3
"""
Book Chapter Map page — the pure helpers, tested without a browser.

For biology researchers: `mount()` needs a live NiceGUI page context (it
builds real ui.select/ui.table-style elements and drives a
BatchController), so it is not unit-tested directly here — the same split
test_gui_library.py, test_gui_ledger.py and test_gui_review.py all
document. What is tested here are the pure functions: pulling the chapter
list out of a document's raw JSON shape, and the PDF-locating fallback
search (the same honest guess ledger.py makes for "Open PDF").
"""

from __future__ import annotations

from src.gui.pages.chapters import _chapters_of, _index_books, _locate_pdf


class TestChaptersOf:
    def test_reads_nested_processing_result_chapters(self):
        document = {"profile": {"processing_result": {
            "chapters": [{"number": 1, "title": "Introduction"}]
        }}}
        assert _chapters_of(document) == [{"number": 1, "title": "Introduction"}]

    def test_missing_profile_is_empty_list(self):
        assert _chapters_of({}) == []

    def test_missing_processing_result_is_empty_list(self):
        assert _chapters_of({"profile": {}}) == []

    def test_missing_chapters_key_is_empty_list(self):
        assert _chapters_of({"profile": {"processing_result": {}}}) == []


class TestIndexBooks:
    def test_no_documents_directory_is_empty(self, tmp_path, monkeypatch):
        import src.gui.pages.chapters as chapters_page
        monkeypatch.setattr(chapters_page, "_documents_dir", lambda: tmp_path / "documents")
        assert _index_books() == []

    def test_only_book_type_documents_are_listed(self, tmp_path, monkeypatch):
        import json
        import src.gui.pages.chapters as chapters_page

        documents_dir = tmp_path / "documents"
        documents_dir.mkdir()
        (documents_dir / "a_book.v6.json").write_text(json.dumps({
            "document_info": {"document_type": "book"},
            "metadata_final": {"title": "A Book"},
        }))
        (documents_dir / "an_article.v6.json").write_text(json.dumps({
            "document_info": {"document_type": "research_article"},
            "metadata_final": {"title": "An Article"},
        }))
        monkeypatch.setattr(chapters_page, "_documents_dir", lambda: documents_dir)

        entries = _index_books()
        assert [entry["label"] for entry in entries] == ["A Book"]

    def test_unreadable_json_is_skipped_not_raised(self, tmp_path, monkeypatch):
        import src.gui.pages.chapters as chapters_page

        documents_dir = tmp_path / "documents"
        documents_dir.mkdir()
        (documents_dir / "broken.v6.json").write_text("{not valid json")
        monkeypatch.setattr(chapters_page, "_documents_dir", lambda: documents_dir)

        assert _index_books() == []


class TestLocatePdf:
    def test_no_filename_returns_none(self):
        assert _locate_pdf(None) is None
        assert _locate_pdf("") is None

    def test_absolute_existing_path_is_used_directly(self, tmp_path):
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        assert _locate_pdf(str(pdf)) == pdf

    def test_falls_back_to_settings_default_folder(self, tmp_path, monkeypatch):
        import src.gui.pages.chapters as chapters_page

        folder = tmp_path / "library"
        folder.mkdir()
        pdf = folder / "book.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        monkeypatch.setattr(
            chapters_page.persistent_settings.SettingsManager, "load",
            staticmethod(lambda: {"default_batch_folder": str(folder)}),
        )
        assert _locate_pdf("book.pdf") == pdf

    def test_no_settings_folder_and_not_absolute_returns_none(self, monkeypatch):
        import src.gui.pages.chapters as chapters_page

        monkeypatch.setattr(
            chapters_page.persistent_settings.SettingsManager, "load",
            staticmethod(lambda: {}),
        )
        assert _locate_pdf("book.pdf") is None
