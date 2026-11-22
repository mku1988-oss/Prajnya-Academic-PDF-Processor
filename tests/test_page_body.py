"""Each page's running text, `body` (`utils/page_body.py`, fix 1).

Every case is a page form seen in the benchmark pilots, the corpus or the
Springer book.
"""
from src.processing.utils.page_body import outside_body_spans, pages_with_body, text_without


def page(number, content, figures=(), chapter=None):
    metadata = {"figures": [{"caption": c} for c in figures]}
    if chapter is not None:
        metadata["chapter"] = chapter
    return {"page_number": number, "content": content, "metadata": metadata}


def kinds(pages):
    return {n: [(s["kind"], pages[[p["page_number"] for p in pages].index(n)]["content"][s["start"]:s["end"]])
                for s in spans] for n, spans in outside_body_spans(pages).items()}


LEGEND = "Fig 2. BHRF1 stimulates MT hyperacetylation in HeLa cells over 24 hours of culture."


class TestWhatIsNotBody:
    def test_placeholders_and_comments(self):
        content = ("Cells grew.\n<!-- figure crop (layout detection) -->\n"
                   "[[FIGURE p4_fig1 | Figure 1. Growth | figures/x/p4_fig1.png]]\nThey died.")
        written = pages_with_body([page(4, content)])[0]
        assert written["body"] == "Cells grew.\nThey died."

    def test_tables_html_and_markdown(self):
        content = "Before.\n| a | b |\n| :--- | :--- |\n| 1 | 2 |\nAfter.\n<table><tr><td>x</td></tr></table>"
        found = [k for k, _ in kinds([page(1, content)])[1]]
        assert found == ["table", "table"]
        assert pages_with_body([page(1, content)])[0]["body"] == "Before.\nAfter.\n"

    def test_a_legend_written_with_markdown_and_another_micro_sign(self):
        """The stored legend says 'μm', the page text '**Fig 2.**' and 'µm'."""
        legend = LEGEND + " Scale bar, 10 μm."
        content = "Text before.\n**Fig 2.** " + LEGEND[7:] + " Scale bar, 10 µm.\nText after."
        written = pages_with_body([page(6, content, figures=[legend])])[0]
        assert written["body"] == "Text before.\nText after."

    def test_a_legend_only_inside_a_placeholder_is_left_with_it(self):
        content = f"Body text here.\n[[FIGURE p6_fig1 | {LEGEND} | figures/x/p6_fig1.png]]"
        found = kinds([page(6, content, figures=[LEGEND])])[6]
        assert [k for k, _ in found] == ["placeholder"]

    def test_overlapping_legends_the_longest_wins_in_any_order(self):
        """Two stored legends share their opening words; the page prints the
        longer. Whatever order the objects come in, the whole legend leaves
        the body (set order once let the shorter one win and the tail stay)."""
        longer = LEGEND + " Scale bar, 10 um. Error bars show the SD of three repeats."
        content = "Text before.\n" + longer + "\nText after."
        for figures in ([LEGEND, longer], [longer, LEGEND]):
            written = pages_with_body([page(6, content, figures=figures)])[0]
            assert written["body"] == "Text before.\nText after."

    def test_a_short_legend_is_not_looked_for(self):
        content = "As Figure 2 shows, cells grew.\nFigure 2. Growth"
        assert kinds([page(1, content, figures=["Figure 2. Growth"])]) == {}


class TestTheReferenceList:
    def test_from_the_heading_to_a_heading_that_follows_lists(self):
        content = ("Discussion ends here.\nReferences\n1. Lee AJ, Doi K. Cell 12, 1-9 (2005).\n"
                   "2. Ono B. Nature 3, 4-5 (2010).\nConflict of Interest: none.")
        written = pages_with_body([page(10, content)])[0]
        assert written["body"] == "Discussion ends here.\nConflict of Interest: none."

    def test_the_list_runs_over_pages_until_cell_press_methods(self):
        pages = [page(12, "Body.\nREFERENCES\nAbe, K., and Ito, T. (2001). Cell 3, 1-9.\nMore, A. (2002)."),
                 page(13, "Zhu, Q. (2019). Immunity 50, 1-12.\nYang, R. (2020). Cell 9, 2-3."),
                 page(14, "STAR METHODS\nKEY RESOURCES TABLE\nThe methods text.")]
        found = kinds(pages)
        assert found[12][0][1].startswith("REFERENCES") and found[13][0][0] == "reference_list"
        assert 14 not in found

    def test_an_appendix_at_the_top_of_the_next_page_ends_it(self):
        pages = [page(18, "References\nAldao A, Nolen S. 2010. Emotion 10:1-9.\n"),
                 page(19, "Appendix 1\nComparison of results in the two samples.")]
        assert 19 not in kinds(pages)

    def test_a_jstor_cover_heading_starts_no_list(self):
        content = ("Stable URL: https://www.jstor.org/stable/1931961\n\nREFERENCES\n"
                   "Linked references are available on JSTOR for this article:\n"
                   "You may need to log in to JSTOR to access the linked references.")
        pages = [page(1, content), page(2, "The article's first page of running text.")]
        assert kinds(pages) == {}

    def test_vancouver_entries_without_stops_after_initials(self):
        content = ("References\nArroyo-Crespo JJ, Arminan A, Charbonnier D, Deladriere C,\n"
                   "Forteza J, Vicent MJ. 2019. Characterization of triple\nnegative models.")
        assert kinds([page(19, content)])[19][0][0] == "reference_list"

    def test_a_book_chapters_list_stops_at_the_next_chapter(self):
        entry = "Adams JC, Lawler J (2004) The thrombospondins. Int J Biochem 36:961-968"
        pages = [page(44, f"Text.\nReferences\n{entry}\n{entry}", chapter={"number": 2}),
                 page(45, f"{entry}\n{entry}", chapter={"number": 2}),
                 page(46, "Basic Components of Connective Tissues\nAbstract\nText.",
                      chapter={"number": 3})]
        found = kinds(pages)
        assert 45 in found and 46 not in found

    def test_old_papers_print_legends_after_the_list(self):
        content = "References\n1. Doi K. Cell 12, 1-9 (2005).\nFig. 1. Electron micrograph of fibrils."
        written = pages_with_body([page(7, content)])[0]
        assert written["body"] == "Fig. 1. Electron micrograph of fibrils."


class TestBody:
    def test_whole_lines_go_with_their_line_break(self):
        content = "Running head 3\r\nFirst line of text\r\nlast line.\r\nJournal 2020 3"
        last = content.index("Journal")
        spans = [{"start": 0, "end": 14}, {"start": last, "end": len(content)}]
        assert text_without(content, spans) == "First line of text\r\nlast line.\r\n"

    def test_part_of_a_line_leaves_the_line_break(self):
        content = "Cells, the Fig 2. legend text here and more.\nNext line."
        start = content.index("Fig 2.")
        end = content.index(" and more")
        assert text_without(content, [{"start": start, "end": end}]) == \
            "Cells, the  and more.\nNext line."

    def test_every_page_gets_the_keys_and_input_is_not_changed(self):
        pages = [page(1, "Plain running text only.")]
        written = pages_with_body(pages)[0]
        assert "body" not in pages[0] and "outside_body" not in pages[0]["metadata"]
        assert written["body"] == written["content"]
        assert written["metadata"]["furniture"] == [] and written["metadata"]["outside_body"] == []

    def test_furniture_leaves_the_body_too(self):
        pages = [page(n, f"Sentence on page {'abcdefgh'[n]} ends.\nFrontiers in Genetics {n:02d} frontiersin.org")
                 for n in range(2, 8)]
        written = pages_with_body(pages)
        assert all("frontiersin" not in p["body"] for p in written)
