#!/usr/bin/env python3
"""
Settings page — the pure helpers, tested without a browser.

For biology researchers: `mount()` builds real ui.input/ui.select
elements and needs a live NiceGUI page, so it isn't unit-tested directly
— the same split every other GUI page in this project follows. What's
tested here are the two validators `_save` runs before writing to disk:
the chapter-list round trip (so a typo is caught here, not silently
carried into a batch run later) and the integer-or-blank guard for
book_max_pages.
"""

from __future__ import annotations

from src.gui.pages.settings import _validated_chapters, _validated_int


class TestValidatedChapters:
    def test_ranges_and_lists_normalize_to_sorted_numbers(self):
        assert _validated_chapters("3-5 1") == "1 3 4 5"

    def test_blank_stays_blank(self):
        assert _validated_chapters("") == ""

    def test_garbage_input_is_dropped_not_raised(self):
        assert _validated_chapters("not a chapter list") == ""

    def test_duplicates_collapse(self):
        assert _validated_chapters("2 2 2") == "2"


class TestValidatedInt:
    def test_digits_pass_through(self):
        assert _validated_int("50") == "50"

    def test_blank_stays_blank(self):
        assert _validated_int("") == ""
        assert _validated_int(None) == ""

    def test_non_numeric_becomes_blank(self):
        assert _validated_int("abc") == ""

    def test_whitespace_is_stripped(self):
        assert _validated_int("  50  ") == "50"
