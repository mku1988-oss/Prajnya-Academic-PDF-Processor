"""Tests for src/gui/links.py: the links a document carries, for the Results screen.

For biology researchers: each test builds a tiny document JSON by hand (one
or two pages, a mention or a citation) and checks that the screen would show
the right sentence under the right figure, table or reference.
"""

from src.gui import links

PAGE_ONE = ("Collagen fibrils form in the matrix. The D-period was measured "
            "(Fig. 3B) and found constant. Table 2 lists the values.")
PAGE_TWO = "Hydration shortens the fibril [2]. Water is bound in two pools [1, 2]."


def document():
    return {
        "pages": [{"page_number": 1, "content": PAGE_ONE},
                  {"page_number": 2, "content": PAGE_TWO}],
        "mentions": {
            "figures": [{"page_number": 1, "content_offset": PAGE_ONE.index("Fig. 3B"),
                         "text": "Fig. 3B", "number": "3", "targets": ["p4_fig3"]}],
            "tables": [{"page_number": 1, "content_offset": PAGE_ONE.index("Table 2"),
                        "text": "Table 2", "number": "2", "targets": ["p5_tab1"]}],
            "skipped": {"supplementary": 0, "own_label": 0}},
        "citations": {
            "level": "online",
            "references": [{"position": 1, "number": 1, "first_author": "Doi", "year": 2005,
                            "title": "Water in tendon"},
                           {"position": 2, "number": 2, "first_author": "Bella", "year": 2016,
                            "title": "Fibril hydration"}],
            "markers": [{"page_number": 2, "text": "[2]", "numbers": [2],
                         "content_offset": PAGE_TWO.index("[2]")},
                        {"page_number": 2, "text": "[1, 2]", "numbers": [1, 2],
                         "content_offset": PAGE_TWO.index("[1, 2]")}]},
    }


class TestMentionsByObject:
    def test_a_figure_gets_the_sentence_that_mentions_it(self):
        passages = links.mentions_by_object(document())["p4_fig3"]
        assert passages == [{"page": 1, "text": "Fig. 3B",
                             "sentence": "The D-period was measured (Fig. 3B) and found constant."}]

    def test_a_table_gets_its_own_sentence(self):
        assert links.mentions_by_object(document())["p5_tab1"][0]["sentence"] == \
            "Table 2 lists the values."

    def test_a_figure_split_into_crops_lists_the_passage_under_each_crop(self):
        doc = document()
        doc["mentions"]["figures"][0]["targets"] = ["p4_fig3", "p4_fig4"]
        found = links.mentions_by_object(doc)
        assert found["p4_fig3"] == found["p4_fig4"]

    def test_an_unlinked_mention_is_filed_under_nothing(self):
        doc = document()
        doc["mentions"]["figures"][0]["targets"] = []
        assert "p4_fig3" not in links.mentions_by_object(doc)

    def test_a_document_without_a_mentions_block_has_none(self):
        assert links.mentions_by_object({"pages": []}) == {}


class TestCitationsByReference:
    def test_each_entry_gets_every_sentence_citing_it(self):
        rows = links.citations_by_reference(document())
        assert [len(r["passages"]) for r in rows] == [1, 2]
        assert rows[0]["passages"][0]["sentence"] == "Water is bound in two pools [1, 2]."

    def test_no_reference_list_means_no_rows(self):
        doc = document()
        doc["citations"] = {"level": "local"}
        assert links.citations_by_reference(doc) == []

    def test_a_marker_without_an_offset_shows_the_text_printed_before_it(self):
        doc = document()
        doc["citations"]["markers"][0].update(content_offset=None,
                                              context_before="ration shortens the fibril")
        assert links.citations_by_reference(doc)[1]["passages"][0]["sentence"] == \
            "… shortens the fibril [2]"

    def test_an_author_year_citation_after_its_bracket_has_no_stray_space(self):
        doc = document()
        doc["citations"]["markers"][0].update(content_offset=None, text="Doi et al., 2005",
                                              context_before="ive crop species (")
        assert links.citations_by_reference(doc)[1]["passages"][0]["sentence"] == \
            "… crop species (Doi et al., 2005"


class TestLinksOnPage:
    def test_a_page_lists_its_links_in_reading_order(self):
        rows = links.links_on_page(document(), 1)
        assert [(r["kind"], r["text"]) for r in rows] == [("figure", "Fig. 3B"),
                                                          ("table", "Table 2")]

    def test_a_citation_points_at_its_reference_entries(self):
        rows = links.links_on_page(document(), 2)
        assert [[t["first_author"] for t in r["targets"]] for r in rows] == \
            [["Bella"], ["Doi", "Bella"]]

    def test_a_page_with_no_links_is_empty(self):
        assert links.links_on_page(document(), 9) == []


class TestSentenceAround:
    def test_an_abbreviation_does_not_end_the_sentence(self):
        text = "Fibrils were aligned (see Fig. 2 and Smith et al. 2005) in the tendon. Next."
        assert links.sentence_around(text, text.index("Fig.")) == \
            "Fibrils were aligned (see Fig. 2 and Smith et al. 2005) in the tendon."

    def test_a_decimal_number_does_not_end_the_sentence(self):
        text = "The period was 67.5 nm as in Table 1. Next."
        assert links.sentence_around(text, text.index("Table")) == \
            "The period was 67.5 nm as in Table 1."

    def test_markdown_and_line_breaks_are_tidied(self):
        text = "First.\n\nThe **fibril** was\r\nimaged (Fig. 1). After."
        assert links.sentence_around(text, text.index("Fig.")) == \
            "The fibril was imaged (Fig. 1)."

    def test_a_long_sentence_is_cut_around_the_link(self):
        text = "word " * 200 + "(Fig. 1)" + " word" * 200 + "."
        sentence = links.sentence_around(text, text.index("(Fig. 1)"))
        assert "(Fig. 1)" in sentence and sentence.startswith("…") and sentence.endswith("…")
        assert len(sentence) <= links.MAX_PASSAGE_CHARACTERS + 4

    def test_an_offset_outside_the_text_gives_nothing(self):
        assert links.sentence_around("short", 99) == ""


class TestLabelsAndCounts:
    def test_a_reference_label_names_number_author_year_and_title(self):
        assert links.reference_label({"position": 3, "number": 12, "first_author": "Doi",
                                      "year": 2005, "title": "Water"}) == \
            "[12] Doi, 2005: Water"

    def test_counts_only_linked_mentions_and_citations(self):
        doc = document()
        doc["mentions"]["tables"][0]["targets"] = []
        assert links.link_counts(doc) == {"figures": 1, "tables": 0, "citations": 2}


class TestOrderOfUnplacedCitations:
    def test_a_citation_without_an_offset_keeps_its_place_in_reading_order(self):
        doc = document()
        doc["citations"]["markers"] = [
            {"page_number": 2, "text": "[1]", "numbers": [1], "content_offset": None},
            {"page_number": 2, "text": "[2]", "numbers": [2], "content_offset": 40},
            {"page_number": 2, "text": "[1, 2]", "numbers": [1, 2], "content_offset": None}]
        assert [r["text"] for r in links.links_on_page(doc, 2)] == ["[1]", "[2]", "[1, 2]"]

    def test_a_long_label_is_cut_at_a_word_with_an_ellipsis(self):
        label = links.reference_label({"position": 1, "first_author": "Doi", "year": 2005,
                                       "title": "Water in tendon and its many pools"}, limit=24)
        assert label == "[1] Doi, 2005: Water in …"


class TestTidyKeepsNames:
    def test_an_underscore_inside_a_gene_name_is_kept(self):
        text = "The hrpG mutant lowered *Aac5_2166* expression (Figure 8A)."
        assert links.sentence_around(text, text.index("Figure")) == \
            "The hrpG mutant lowered Aac5_2166 expression (Figure 8A)."
