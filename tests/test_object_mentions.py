"""In-text figure and table mentions (`citations/object_mentions.py`, fix 6).

Every case is a sentence form seen in the benchmark's pilot papers.
"""
from src.processing.citations.object_mentions import build_mentions_block, find_mentions


def linked(content, captions=()):
    """(number, text) of every mention to link, in order."""
    return [(r["number"], r["text"]) for r, reason in find_mentions(content, captions)
            if not reason]


def skipped(content, captions=()):
    return [(r["number"], reason) for r, reason in find_mentions(content, captions) if reason]


class TestWhatIsAMention:
    def test_journal_spellings(self):
        assert [n for n, _ in linked("(Fig. 3B) and (Figure 2) and (FIGURE 4) and (Fig 5A)")] \
            == ["3", "2", "4", "5"]

    def test_two_figures_give_two_records(self):
        assert [n for n, _ in linked("as shown (Figures 4A and 5B).")] == ["4", "5"]
        assert [n for n, _ in linked("(Figs 2 and 3)")] == ["2", "3"]

    def test_panels_of_one_figure_give_one_record(self):
        """PLOS 'Fig 1A and 1D' and 'Figures 3A, B': one link to the figure."""
        assert linked("(Fig 1A and 1D)") == [("1", "Fig 1A and 1D")]
        assert [n for n, _ in linked("(Figures 3A, B)")] == ["3"]
        assert [n for n, _ in linked("(Figures 1A-1C)")] == ["1"]

    def test_a_range_of_figures_is_expanded(self):
        assert [n for n, _ in linked("(Figures 2-4)")] == ["2", "3", "4"]

    def test_tables(self):
        assert [n for n, _ in linked("listed in Table 2 and TABLE II.")] == ["2", "II"]
        assert [n for n, _ in linked("see Table 11.1")] == ["11.1"]

    def test_a_mention_broken_over_a_line(self):
        assert [n for n, _ in linked("as in Figure\r\n3B, the")] == ["3"]

    def test_not_a_mention(self):
        assert linked("configure 3 dishes; figures/PMC1/p4_fig1.png") == []
        assert linked("Figure legends are below.") == []


class TestSupplementaryIsNeverLinked:
    def test_s_numbers(self):
        """Cell Press 'Figure S1' must never become Figure 1."""
        assert linked("(Figure S1)") == []
        assert skipped("(Figure S1; Table S2)") == [("1", "supplementary"),
                                                    ("2", "supplementary")]

    def test_supplementary_words_before(self):
        assert linked("(Supplementary Fig. 3)") == []
        assert linked("(Supplementary\r\nFigure 2)") == []
        assert linked("(Extended Data Fig. 2)") == []
        assert linked("(Appendix 1—figure 2)") == []

    def test_plos_s1_fig(self):
        """'Fig 1A and S1 Fig': Figure 1 yes, S1 no."""
        assert [n for n, _ in linked("(Fig 1A and S1 Fig)")] == ["1"]

    def test_elife_figure_supplement(self):
        """'Figure 2—figure supplement 1' is not Figure 2."""
        assert linked("(Figure 2—figure supplement 1A-B)") == []
        assert [n for n, _ in linked("(Figure 2B; Figure 2—source data 1)")] == ["2"]

    def test_a_main_figure_beside_a_supplementary_one(self):
        assert [n for n, _ in linked("(Figure 1A and Supplementary Figure 3)")] == ["1"]


class TestALegendsOwnLabel:
    def test_label_at_the_start_of_its_caption(self):
        caption = "Figure 3. IpaB-specific antibodies mediate protection".replace(" ", "")[:24]
        content = "end of a paragraph.\nFigure 3. IpaB-specific antibodies mediate protection"
        assert skipped(content, [caption]) == [("3", "own_label")]

    def test_frontiers_bar(self):
        assert skipped("\nFIGURE 1 | Expression of genes") == [("1", "own_label")]

    def test_a_mention_at_a_line_start_is_still_a_mention(self):
        """Text-layer lines wrap anywhere: '...shown in\\nFigure 2A, the ...'."""
        assert [n for n, _ in linked("as shown in\nFigure 2A, the cells")] == ["2"]

    def test_placeholders_are_not_text(self):
        content = ("<!-- figure crop (layout detection) -->\n[[FIGURE p4_fig1 | Figure 1. "
                   "Challenge (see Figure 2) | figures/PMC1/p4_fig1.png]]")
        assert linked(content) == []


class TestTheBlock:
    PAGES = [
        {"page_number": 1, "content": "Cells grew (Fig. 1B) but not (Fig. 4; Table 1).",
         "metadata": {"figures": [
             {"figure_id": "p2_fig1", "printed_label": "Fig. 1", "caption": "Fig. 1. Growth"},
             {"figure_id": "p2_fig2", "printed_label": "Fig. 1", "caption": "Fig. 1. Growth"}],
             "tables": [{"table_id": "p3_tab1", "caption": "Table 1 Strains used"}]}},
        {"page_number": 2, "content": "(Figure S1)", "metadata": {}},
    ]

    def test_links_to_every_crop_of_the_figure(self):
        block = build_mentions_block(self.PAGES)
        assert block["figures"][0] == {"page_number": 1, "content_offset": 12,
                                       "text": "Fig. 1B", "number": "1",
                                       "targets": ["p2_fig1", "p2_fig2"]}

    def test_a_figure_not_found_keeps_its_mention_with_no_target(self):
        block = build_mentions_block(self.PAGES)
        assert block["figures"][1]["number"] == "4" and block["figures"][1]["targets"] == []

    def test_tables_link_by_their_caption_label(self):
        block = build_mentions_block(self.PAGES)
        assert block["tables"] == [{"page_number": 1, "content_offset": 38, "text": "Table 1",
                                    "number": "1", "targets": ["p3_tab1"]}]

    def test_supplementary_is_counted(self):
        assert build_mentions_block(self.PAGES)["skipped"]["supplementary"] == 1
