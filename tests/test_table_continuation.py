"""
Tests for the header carry onto continuation pages (utils/table_continuation.py).

For biology researchers: these draw small ruled tables with PIL, so they need
no PDF and no model. A fake reader stands in for GLM and says what it was given.
The corpus measurement behind the feature is ledger T22 (eval/header_carry/).
"""

import logging

from PIL import Image, ImageDraw

from src.processing.strategies.research_article.tables import _grid_for
from src.processing.utils.table_continuation import (
    ContinuationTracker,
    column_rules,
    column_shift,
    drop_pasted_header,
    first_row_cells,
    header_band,
    paste_header,
)

GREY = 149                     # printed rules are thin mid-grey, as on the Springer book
COLUMNS = (0, 200, 300, 420, 599)
HEADER_GRID = "<table><tr><td>Entity</td><td>ELN</td><td>FBLN4</td></tr><tr><td>Thin skin</td><td>-</td><td>+</td></tr></table>"


def ruled_table(columns=COLUMNS, rows=(0, 50, 100, 150), width=600, top=0, shift=0):
    """A white image with full-width horizontal rules at `rows` and vertical ones at `columns`."""
    height = top + rows[-1] + 2
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for y in rows:
        draw.line([(0, top + y), (width - 1, top + y)], fill=(GREY,) * 3, width=2)
    for x in columns:
        draw.line([(x + shift, top), (x + shift, height - 1)], fill=(GREY,) * 3, width=2)
    return image


class FakeReader:
    """Stands in for the model: records each image, returns scripted grids."""

    def __init__(self, *grids):
        self.grids = list(grids)
        self.images = []

    def __call__(self, image):
        self.images.append(image)
        return self.grids.pop(0) if self.grids else ""


def keep(image, text):
    """A `finish` step that changes nothing (production's is the section check)."""
    return text


class TestHeaderBand:
    def test_the_band_runs_from_the_top_rule_to_the_second(self):
        band = header_band(ruled_table())
        assert band is not None and 50 <= band.height <= 53

    def test_a_caption_above_the_table_is_left_out(self):
        """t21: the crop starts with its caption; the band starts at the top rule."""
        band = header_band(ruled_table(top=40))
        assert band is not None and band.height <= 53

    def test_an_unruled_table_gives_no_band(self):
        assert header_band(Image.new("RGB", (600, 200), "white")) is None


class TestColumnsLineUp:
    def test_the_column_rules_are_found(self):
        assert [round(x) for x in column_rules(ruled_table())] == [0, 200, 300, 420, 599]

    def test_a_constant_shift_still_lines_up(self):
        """t18: the two pages' crops start 18 px apart; the columns are the same."""
        assert column_shift([10, 200, 300], [28, 218, 318]) == 18

    def test_a_different_number_of_columns_is_another_table(self):
        assert column_shift([10, 200, 300], [10, 200, 300, 400]) is None

    def test_different_spacing_is_another_table(self):
        assert column_shift([10, 200, 300], [10, 230, 300]) is None

    def test_one_rule_is_not_enough_evidence(self):
        assert column_shift([10], [10]) is None


class TestPasteAndDrop:
    def test_the_header_goes_on_top_at_the_table_width(self):
        strip = Image.new("RGB", (580, 50), "black")
        table = Image.new("RGB", (600, 100), "white")
        pasted = paste_header(strip, table, shift=10)
        assert pasted.size == (600, 150)
        assert pasted.getpixel((5, 25)) == (255, 255, 255)     # left of the shifted strip
        assert pasted.getpixel((20, 25)) == (0, 0, 0)
        assert pasted.getpixel((20, 120)) == (255, 255, 255)

    def test_the_pasted_header_row_is_removed(self):
        cells = first_row_cells(HEADER_GRID)
        grid = "<table><tr><td>Entity</td><td>ELN</td><td>FBLN 4</td></tr><tr><td>Apnea</td><td>-</td><td>+</td></tr></table>"
        assert drop_pasted_header(grid, cells) == \
            "<table><tr><td>Apnea</td><td>-</td><td>+</td></tr></table>"

    def test_a_read_that_did_not_return_the_header_is_rejected(self):
        grid = "<table><tr><td>Apnea</td><td>-</td><td>+</td></tr></table>"
        assert drop_pasted_header(grid, first_row_cells(HEADER_GRID)) is None


class TestTheTracker:
    def tracker_after_first_page(self):
        tracker = ContinuationTracker(logger=logging.getLogger("test"))
        tracker.begin_page(288)
        tracker.remember(ruled_table(), HEADER_GRID, was_headed=False)
        return tracker

    def test_the_next_page_first_table_is_read_with_the_header(self):
        tracker = self.tracker_after_first_page()
        tracker.begin_page(289)
        reader = FakeReader("<table><tr><td>Entity</td><td>ELN</td><td>FBLN4</td></tr>"
                            "<tr><td>Apnea</td><td>-</td><td>+</td></tr></table>")
        grid = tracker.headed_read(ruled_table(rows=(0, 50, 100)), reader, keep)
        assert grid == "<table><tr><td>Apnea</td><td>-</td><td>+</td></tr></table>"
        assert reader.images[0].height > 102            # the header was pasted on
        assert tracker.headed_tables == [(289, 1)]

    def test_a_page_gap_drops_the_header(self):
        tracker = self.tracker_after_first_page()
        tracker.begin_page(291)
        assert tracker.headed_read(ruled_table(), FakeReader("x"), keep) is None

    def test_only_the_first_table_on_the_page_is_a_candidate(self):
        tracker = self.tracker_after_first_page()
        tracker.begin_page(289)
        tracker.remember(ruled_table(), "<table><tr><td>a</td></tr></table>", was_headed=False)
        reader = FakeReader("x")
        assert tracker.headed_read(ruled_table(), reader, keep) is None
        assert reader.images == []                        # no model call wasted

    def test_columns_that_do_not_line_up_are_never_read_headed(self):
        tracker = self.tracker_after_first_page()
        tracker.begin_page(289)
        reader = FakeReader("x")
        other = ruled_table(columns=(0, 150, 300, 599))
        assert tracker.headed_read(other, reader, keep) is None
        assert reader.images == []

    def test_a_chain_keeps_passing_the_original_header(self):
        """Table 9.1 runs over three pages: page 3 needs page 1's header, not page 2's top row."""
        tracker = self.tracker_after_first_page()
        tracker.begin_page(289)
        tracker.remember(ruled_table(), "<table><tr><td>Apnea</td></tr></table>", was_headed=True)
        assert tracker.carried.cells == first_row_cells(HEADER_GRID)
        assert tracker.carried.page_number == 289


class TestTheTablesPathFailsSafe:
    """`tables._grid_for` with the carry: any doubt reads the table exactly as before."""

    class Region:
        box = (0, 0, 600, 152)
        reading_box = (0, 0, 600, 152)

    def run(self, tracker, *grids):
        reader = FakeReader(*grids)
        page_image = ruled_table()
        grid = _grid_for(self.Region(), {}, page_image, reader, 289,
                         logging.getLogger("test"), page=None, continuation=tracker)
        return grid, reader

    def test_no_carry_reads_plain_and_offers_this_header_on(self):
        tracker = ContinuationTracker()
        tracker.begin_page(288)
        grid, reader = self.run(tracker, HEADER_GRID)
        assert grid == HEADER_GRID and len(reader.images) == 1
        assert tracker.carried is not None

    def test_a_headed_read_without_the_header_falls_back_to_a_plain_read(self):
        tracker = ContinuationTracker()
        tracker.begin_page(288)
        tracker.remember(ruled_table(), HEADER_GRID, was_headed=False)
        tracker.begin_page(289)
        plain = "<table><tr><td>Apnea</td><td>-</td><td>+</td></tr></table>"
        grid, reader = self.run(tracker, "<table><tr><td>nonsense</td></tr></table>", plain)
        assert grid == plain and len(reader.images) == 2

    def test_a_failing_headed_read_never_loses_the_table(self):
        tracker = ContinuationTracker()
        tracker.begin_page(288)
        tracker.remember(ruled_table(), HEADER_GRID, was_headed=False)
        tracker.begin_page(289)
        calls = []

        def reader(image):
            calls.append(image)
            if len(calls) == 1:
                raise RuntimeError("server went away")
            return "<table><tr><td>Apnea</td></tr></table>"

        grid = _grid_for(self.Region(), {}, ruled_table(), reader, 289,
                         logging.getLogger("test"), page=None, continuation=tracker)
        assert grid == "<table><tr><td>Apnea</td></tr></table>"
