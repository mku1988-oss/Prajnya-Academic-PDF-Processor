"""
Tests for ledger T31 (v6.9 item 3, S3a): raised citation numbers that the
line grouper set off their line, a false gutter found in a short window at
a page top, and markers the vision model wrote as superscript characters.

For biology researchers: tiny hand-built pages. A raised "1–5" after
"apoptosis." must stay on its line so it reads as a citation; a short last
line holding more raised digits than letters must keep the letters'
baseline; a superscript "¹⁻³" in a page's content is the marker "1-3".
"""
from typing import List

from src.processing.citations.markers import locate_markers
from src.processing.pdf.base import PrintedChar, Rect
from src.processing.utils.citation_markers import raised_digit_runs
from src.processing.utils.layout_lines import best_window_gutters, page_layout_lines

WIDTH = 4.5
HEIGHT = 8.91          # the paper's body type
RISE = 4.5             # just over half the line height (the g5 paper's 4.42-4.5)
SMALL = 5.94           # its raised digits


def printed(text: str, x: float, baseline: float, height: float = HEIGHT, width: float = WIDTH) -> List[PrintedChar]:
    """Characters for `text` from x on one baseline; a space marks a word boundary."""
    chars, space = [], False
    for character in text:
        if character == " ":
            space, x = True, x + width
            continue
        rect = Rect(x, baseline - 0.8 * height, x + width, baseline + 0.2 * height)
        chars.append(PrintedChar(text=character, rect=rect, baseline=baseline, space_before=space))
        space, x = False, x + width
    return chars


def end_of(chars: List[PrintedChar]) -> float:
    return chars[-1].rect.x1


def raised(text: str, x: float, line_baseline: float) -> List[PrintedChar]:
    return printed(text, x, line_baseline - RISE, height=SMALL, width=3.0)


class TestRaisedRunStaysOnItsLine:
    def test_a_run_set_off_by_its_rise_rejoins_the_line_it_follows(self):
        before = printed("growth and apoptosis.", 40, 336.4)
        marker = raised("1–5", end_of(before), 336.4)
        after = printed(" Its abnormal regulation", end_of(marker) + 1, 336.4)
        lines = page_layout_lines(before + marker + after, 1)
        assert [line.text for line in lines] == ["growth and apoptosis.1–5 Its abnormal regulation"]
        runs = raised_digit_runs(lines[0])
        assert ["".join(c.text for c in lines[0].chars[a:b]) for a, b in runs] == ["1–5"]

    def test_a_short_line_of_mostly_raised_digits_keeps_the_letters_baseline(self):
        word = printed("partner,", 40, 299.7)
        marker = printed("17,19,21,24–26", end_of(word), 299.7 - 4.42, height=SMALL, width=3.0)
        lines = page_layout_lines(word + marker, 1)
        assert len(lines) == 1 and lines[0].baseline == 299.7 and abs(lines[0].height - HEIGHT) < 1e-6
        runs = raised_digit_runs(lines[0])
        assert ["".join(c.text for c in lines[0].chars[a:b]) for a, b in runs] == ["17,19,21,24–26"]

    def test_a_line_with_one_odd_letter_keeps_its_digits_height(self):
        # NMR's list: "1997;94(21):11312e7.", the "e" a dash in a symbol font with a small box.
        chars = printed("1997;94(21):11312", 40, 364.9, height=7.1)
        dash = printed("e", end_of(chars), 364.9, height=4.1)
        tail = printed("7.", end_of(dash), 364.9, height=7.1)
        lines = page_layout_lines(chars + dash + tail, 1)
        assert len(lines) == 1 and abs(lines[0].height - 7.1) < 1e-6

    def test_a_subscript_is_not_joined_as_a_raised_run(self):
        word = printed("CO", 40, 100)
        lowered = printed("2", end_of(word), 100 + 4.5, height=SMALL, width=3.0)
        lines = page_layout_lines(word + lowered, 1)
        assert all(not raised_digit_runs(line) for line in lines)

    def test_small_text_far_to_the_right_is_not_joined(self):
        word = printed("apoptosis.", 40, 336.4)
        far = raised("12", end_of(word) + 60, 336.4)
        lines = page_layout_lines(word + far, 1)
        assert sorted(line.text for line in lines) == ["12", "apoptosis."]


class TestWindowTieBreak:
    def test_of_windows_finding_as_many_gutters_the_longest_wins(self):
        # Near a page top the upward window is cut to 5 rows, where word gaps
        # can line up into a false gutter (g1 p15: 362 pt instead of 396.5).
        found = [((221.0, 362.0), 5), ((221.0, 396.5), 10), ((221.0, 396.5), 10)]
        assert best_window_gutters(found, 30.0) == (221.0, 396.5)

    def test_a_full_window_that_disagrees_with_the_other_two_loses(self):
        # g1 p17: the window above finds 316.5; the windows below and around find 392.5.
        found = [((217.0, 316.5), 10), ((232.0, 392.5), 10), ((217.0, 392.5), 10)]
        assert best_window_gutters(found, 30.0)[1] == 392.5

    def test_with_no_agreement_the_widest_window_wins(self):
        found = [((221.0, 362.0), 5), ((221.0, 396.5), 10)]
        assert best_window_gutters(found, 30.0) == (221.0, 396.5)

    def test_more_gutters_still_beat_a_longer_window(self):
        # Where a page changes layout, the window on the row's own side finds its columns.
        found = [((221.0, 396.5), 6), ((300.0,), 10)]
        assert best_window_gutters(found, 30.0) == (221.0, 396.5)


class TestSuperscriptOffsets:
    def marker(self, text, context):
        return {"page_number": 1, "text": text, "kind": "superscript", "numbers": [],
                "context_before": context, "content_offset": None}

    def test_a_superscript_marker_in_the_content_is_found(self):
        content = "used in electronic applications, in recent years.¹⁻³ In particular"
        located = locate_markers([self.marker("1−3", "applications, in recent years.")], {1: content})
        assert located[0]["content_offset"] == content.index("¹")

    def test_a_misread_superscript_is_not_taken(self):
        content = "between remote electrodes.⁷,¹²\nAmong the canonical"
        located = locate_markers([self.marker("5,7,11", "between remote electrodes.")], {1: content})
        assert located[0]["content_offset"] is None
