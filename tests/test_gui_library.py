#!/usr/bin/env python3
"""
Library/Batch page — the pure helpers, tested without a browser.

For biology researchers: `mount()` itself needs a live NiceGUI page context
to run (it builds real UI elements), so it is not unit-tested directly here
— that is covered by the live HTTP smoke check this session ran by hand
(all expected controls render server-side with no error) and by
`test_gui_controller.py`, which covers everything `mount()` delegates to.
What *is* tested here are the small pure functions the page uses to turn
pipeline data into text: the preflight summary line and the transcript
line for one event. Both are pure — same input, same output — so they do
not need a page at all.
"""

from __future__ import annotations

from pathlib import Path

from src.gui.pages import library
from src.gui.pages.library import (
    _apply_settings_overrides,
    _book_overrides_from_settings,
    _format_event,
    _optional_int,
    _summarize,
)
from src.processing.batch import PreflightEntry
from src.processing.events import RunEvent
from src.processing.models import DocumentType


def _entry(**overrides) -> PreflightEntry:
    defaults = dict(path=Path("paper.pdf"), document_type=DocumentType.RESEARCH_ARTICLE,
                    confidence=0.9, page_count=8)
    defaults.update(overrides)
    return PreflightEntry(**defaults)


class TestSummarize:
    def test_no_entries(self):
        assert _summarize([]) == "No PDFs."

    def test_counts_by_document_type(self):
        entries = [
            _entry(document_type=DocumentType.RESEARCH_ARTICLE),
            _entry(document_type=DocumentType.RESEARCH_ARTICLE),
            _entry(document_type=DocumentType.BOOK),
        ]
        summary = _summarize(entries)
        assert "3 PDF(s)" in summary
        assert "2 research_article" in summary
        assert "1 book" in summary

    def test_errored_entries_are_reported_but_not_typed(self):
        entries = [_entry(), _entry(document_type=None, error="cannot open")]
        summary = _summarize(entries)
        assert "2 PDF(s)" in summary
        assert "1 could not be read" in summary
        # The errored entry must not show up as a fake "unknown"-typed count
        # for a document that was never actually classified.
        assert "unknown" not in summary


class TestSummarizeAHeldBackPdf:
    def test_a_pdf_held_back_for_its_name_is_not_called_unreadable(self):
        """Audit D3: it was readable; another PDF in the batch has its name."""
        entries = [_entry(), _entry(document_type=None, error="same output name",
                                    held_back=True)]
        summary = _summarize(entries)
        assert "1 not run (same name as another PDF)" in summary
        assert "could not be read" not in summary


class TestFormatEvent:
    def test_a_page_event_shows_progress(self):
        event = RunEvent(kind="page", page=4, page_total=8, message="Page 4")
        assert _format_event(event) == "  page 4/8: Page 4"

    def test_a_page_event_without_totals_falls_back_to_the_message(self):
        event = RunEvent(kind="page", message="Page, no totals yet")
        assert _format_event(event) == "Page, no totals yet"

    def test_a_plain_log_event_shows_its_message(self):
        event = RunEvent(kind="log", message="Routing to ResearchArticleStrategy")
        assert _format_event(event) == "Routing to ResearchArticleStrategy"

    def test_an_event_with_no_message_is_dropped(self):
        event = RunEvent(kind="log", message=None)
        assert _format_event(event) is None


class TestOptionalInt:
    def test_blank_is_none(self):
        assert _optional_int("") is None
        assert _optional_int(None) is None

    def test_digits_parse(self):
        assert _optional_int("42") == 42

    def test_non_numeric_is_none(self):
        assert _optional_int("abc") is None

    def test_zero_is_a_real_value_not_none(self):
        # RunRequest.max_pages=0 would mean "read zero pages", not "no cap" —
        # must not be conflated with the blank/unset case.
        assert _optional_int("0") == 0


class TestBookOverridesFromSettings:
    def test_blank_policy_is_no_override(self):
        assert _book_overrides_from_settings({"book_policy": ""}) is None
        assert _book_overrides_from_settings({}) is None

    def test_skip_policy_has_no_chapters_key(self):
        overrides = _book_overrides_from_settings({"book_policy": "skip"})
        assert overrides == {"policy": "skip"}

    def test_chapters_policy_parses_the_chapter_string(self):
        overrides = _book_overrides_from_settings(
            {"book_policy": "chapters", "book_chapters": "1 3-5"})
        assert overrides == {"policy": "chapters", "chapters": [1, 3, 4, 5]}

    def test_chapters_policy_with_no_chapters_typed_is_empty_list(self):
        overrides = _book_overrides_from_settings(
            {"book_policy": "chapters", "book_chapters": ""})
        assert overrides == {"policy": "chapters", "chapters": []}


class TestApplySettingsOverrides:
    def test_all_blank_settings_add_nothing(self):
        kwargs = {"paths": [], "recursive": False, "noninteractive": False}
        _apply_settings_overrides(kwargs, {})
        assert kwargs == {"paths": [], "recursive": False, "noninteractive": False}

    def test_output_root_and_ledger_path_become_paths(self):
        kwargs = {}
        _apply_settings_overrides(kwargs, {
            "output_root": "D:/out", "ledger_path": "D:/ledger.csv",
        })
        assert kwargs["out_root"] == Path("D:/out")
        assert kwargs["ledger_path"] == Path("D:/ledger.csv")

    def test_force_default_only_added_when_true(self):
        kwargs = {}
        _apply_settings_overrides(kwargs, {"force_default": False})
        assert "force" not in kwargs

        kwargs = {}
        _apply_settings_overrides(kwargs, {"force_default": True})
        assert kwargs["force"] is True

    def test_book_max_pages_and_policy_both_flow_through(self):
        kwargs = {}
        _apply_settings_overrides(kwargs, {
            "book_max_pages": "50", "book_policy": "all",
        })
        assert kwargs["max_pages"] == 50
        assert kwargs["book_overrides"] == {"policy": "all"}


class TestDisplayName:
    """
    A pre-flight row and the book dialog both print a document's name. That
    name is EXTRACTED text, so nothing bounds its length — see
    `_display_name`'s docstring for the live failure this prevents.
    """

    def _entry(self, title, name="doc.pdf"):
        return PreflightEntry(path=Path(name), document_type=DocumentType.BOOK,
                              page_count=36, title=title)

    def test_a_short_title_is_left_alone(self):
        assert library._display_name(self._entry("Collagen Fibril Assembly")) == \
            "Collagen Fibril Assembly"

    def test_a_runaway_title_is_truncated(self):
        """The real case: title detection returned the whole first page, and
        the dialog grew tall enough to push its own buttons off-screen."""
        runaway = "CHAPTER THREE " + ("Collagen fibrils are the major component " * 60)
        shown = library._display_name(self._entry(runaway))

        assert len(shown) <= library.DISPLAY_NAME_LIMIT
        assert shown.endswith("…")
        assert shown.startswith("CHAPTER THREE")

    def test_newlines_from_page_text_are_collapsed(self):
        """Page text arrives with line breaks; a table cell wants one line."""
        assert library._display_name(self._entry("Chapter\n Three\n\nAssembly")) == \
            "Chapter Three Assembly"

    def test_a_missing_title_falls_back_to_the_filename(self):
        assert library._display_name(self._entry(None, name="paper.pdf")) == "paper.pdf"

    def test_a_long_filename_is_also_truncated(self):
        """The fallback is not exempt — the limit is about the display, not
        about which field the text came from."""
        entry = self._entry(None, name="x" * 400 + ".pdf")
        assert len(library._display_name(entry)) <= library.DISPLAY_NAME_LIMIT
