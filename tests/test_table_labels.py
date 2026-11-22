"""Printed table titles read from the text layer (`utils/table_labels.py`).

Lines are (text, x0, y0, x1, y1) in page points, y growing downwards, as
`figure_label_pass.build_text_lines` returns them. Coordinates are taken from
the corpus pages each case comes from.
"""
from src.processing.utils.table_labels import find_table_title, table_label_of


class TestTableLabelOf:
    def test_journal_spellings(self):
        assert table_label_of("Table 2 Intrinsic preferences") == "Table 2"
        assert table_label_of("TABLE II. Relative humidity") == "Table II"
        assert table_label_of("Tab. 3 Results") == "Table 3"
        assert table_label_of("Table 11.1 Clinical features") == "Table 11.1"

    def test_not_a_label(self):
        assert table_label_of("Adapted from [44]") is None
        assert table_label_of("") is None and table_label_of(None) is None
        assert table_label_of("Tables 1 and 2 show") is None

    def test_ocr_noise_after_the_number_is_refused(self):
        """The 1960 scan reads 'TABLE IV.' as 'TABLE I N.': not Table I."""
        assert table_label_of("TABLE I N. Sources of vapor pressure") is None


class TestFindTableTitle:
    def test_title_above_the_box_ends_where_the_table_begins(self):
        """Collagen structure p10: title above, footnote inside the box."""
        lines = [("Table 2 Intrinsic preferences for the conformation", 308, 67, 555, 75),
                 ("Pro derivatives and abbreviations used.", 308, 76, 555, 83),
                 ("Adapted from [44]: Shoulders, M.D.", 308, 90, 555, 97)]
        title = find_table_title(lines, (301, 88, 555, 308))
        assert title == ("Table 2 Intrinsic preferences for the conformation "
                         "Pro derivatives and abbreviations used.")

    def test_title_inside_the_box_continues_only_after_a_hyphen(self):
        """Humidity p4: 'solu-' wraps to a centred 'tions at ...'."""
        lines = [("TABLE II. Relative humidity values over saturated solu-", 281, 207, 493, 215),
                 ("tions at given temperatures", 333, 216.6, 440, 224.2),
                 ("At 200 C.", 371, 239, 403, 246)]
        title = find_table_title(lines, (273, 165, 498, 632))
        assert title == ("TABLE II. Relative humidity values over saturated "
                         "solutions at given temperatures")

    def test_a_label_straddling_the_box_top_belongs_to_it(self):
        """Tendon p257: the label line starts 5 pt above the box."""
        lines = [("Table 11.1 Clinical features at initial diagnosis of LDS", 258, 59, 450, 70)]
        assert find_table_title(lines, (253, 64, 457, 502)).startswith("Table 11.1")

    def test_a_label_in_another_column_is_not_this_tables(self):
        lines = [("Table 3 Something else", 45, 67, 290, 75)]
        assert find_table_title(lines, (301, 88, 555, 308)) is None

    def test_a_label_far_above_is_not_this_tables(self):
        lines = [("Table 3 Something else", 308, 10, 555, 18)]
        assert find_table_title(lines, (301, 188, 555, 308)) is None
