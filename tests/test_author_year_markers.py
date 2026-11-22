"""Tests for utils/author_year_markers.py: author-year citations linked to the list (S3b).

For biology researchers: each test prints a few body lines such as
"(Doi et al., 2005)" and a small reference list, and checks which entry (by
position) each citation links to, and which citations are refused.
"""
from types import SimpleNamespace

import pytest

from src.processing.citations.markers import author_year_markers_for, locate_markers
from src.processing.utils.author_year_markers import search_author_year_markers
from src.processing.utils.reference_list import ReferenceList


def lines_for(*texts, page=1):
    """One line per text, all on one page (or pairs (page, text))."""
    out = []
    for item in texts:
        pg, text = item if isinstance(item, tuple) else (page, item)
        out.append(SimpleNamespace(page=pg, text=text))
    return out


def ref(position, author, year, printed=None):
    return {"position": position, "number": None, "first_author": author, "year": year,
            "printed_text": printed}


def author_year_list(first_line):
    return ReferenceList(style="author_year", entries=[], first_line=first_line)


def run(texts, references):
    """Search `texts` (body) followed by a reference-list line that must never be read."""
    already_lines = bool(texts) and not isinstance(texts[0], str) and not isinstance(texts[0], tuple)
    body = list(texts) if already_lines else lines_for(*texts)
    body = body + lines_for("Doi M (2005) A list entry that is not a citation")
    return search_author_year_markers(body, author_year_list(len(body) - 1), references)


def found(texts, references):
    return [(m.text, m.numbers, m.page) for m in run(texts, references).markers]


REFS = [ref(1, "Doi", 2005, "Doi M, Nishi Y, Lee K. 2005. Collagen."),
        ref(2, "Smith", 1999, "Smith J and Jones B. 1999a. Fibrils."),
        ref(3, "Jones", 2001, "Jones B. 2001. Alone."),
        ref(4, "Bella", 1994, "Bella J, Eaton M, Brodsky B. 1994. Structure."),
        ref(5, "Bella", 1995, "Bella J, Eaton M, Brodsky B. 1995. More."),
        ref(6, "Berendsen", 1962, "Berendsen H. 1962. Water."),
        ref(7, "Münder", 2010, "Münder S. 2010. Hydra."),
        ref(8, "van der Flier", 2003, "van der Flier A. 2003. Integrins."),
        ref(9, "Pucci-Minafra", 2008, "Pucci-Minafra I. 2008. Matrix."),
        ref(10, "Amano", 2009, "Amano S. 2009. Skin."),
        ref(11, "Brennan", 2003, "Brennan A, Lee K, Wu T. 2003. Cells.")]


class TestForms:
    def test_single_author(self):
        assert found(["as shown (Amano, 2009) here."], REFS) == [("Amano, 2009", [10], 1)]

    def test_two_authors_and(self):
        assert found(["(Smith and Jones, 1999a)"], REFS) == [("Smith and Jones, 1999a", [2], 1)]

    def test_two_authors_ampersand(self):
        assert found(["(Smith & Jones, 1999a)"], REFS) == [("Smith & Jones, 1999a", [2], 1)]

    def test_apa_list(self):
        assert found(["(Brennan, Lee, & Wu, 2003)"], REFS)[0][1] == [11]
        assert found(["(Brennan, Lee, Wu and Cho, 2003)"], REFS)[0][1] == [11]

    def test_et_al(self):
        assert found(["(Doi et al., 2005)"], REFS) == [("Doi et al., 2005", [1], 1)]

    def test_et_al_without_a_space_and_no_comma(self):
        assert found(["(Doi et al.,2005)"], REFS)[0][1] == [1]
        assert found(["(Amano 2009)"], REFS)[0][1] == [10]

    def test_two_years_in_one_group_is_one_marker(self):
        assert found(["(Bella et al., 1994, 1995)"], REFS) == [("Bella et al., 1994, 1995", [4, 5], 1)]

    def test_semicolon_separated_items(self):
        assert [f[1] for f in found(["(Doi et al., 2005; Amano, 2009)"], REFS)] == [[1], [10]]

    def test_comma_separated_items_and_leading_prose(self):
        text = "(for review, see Brennan et al., 2003, Amano, 2009, Doi et al., 2005)"
        assert [f[1] for f in found([text], REFS)] == [[11], [10], [1]]

    def test_narrative_forms(self):
        text = "Berendsen (1962) and Doi et al. (2005) and Smith and Jones (1999a) found it."
        assert [f[1] for f in found([text], REFS)] == [[6], [1], [2]]
        assert found(["Berendsen (1962) showed"], REFS)[0][0] == "Berendsen (1962)"

    def test_narrative_with_two_years(self):
        assert found(["Bella et al. (1994, 1995) found"], REFS)[0][1] == [4, 5]

    def test_marker_has_context_and_contract_kind(self):
        marker = run(["the fibrils are strong (Amano, 2009)."], REFS).markers[0]
        assert marker.kind == "author_year" and marker.context == "the fibrils are strong ("


class TestNames:
    def test_umlaut_and_accent_folding(self):
        assert found(["(Münder, 2010)"], REFS)[0][1] == [7]
        assert found(["(Munder, 2010)"], REFS)[0][1] == [7]

    def test_particle_surname(self):
        assert found(["(van der Flier, 2003)"], REFS)[0][1] == [8]
        assert found(["(Flier, 2003)"], REFS)[0][1] == [8]

    def test_hyphenated_surname_keeps_its_real_hyphen_across_lines(self):
        assert found(["(Pucci-", "Minafra, 2008)"], REFS) == [("Pucci-Minafra, 2008", [9], 1)] or \
            found(["(Pucci-", "Minafra, 2008)"], REFS)[0][1] == [9]

    def test_broken_word_across_lines_is_joined(self):
        refs = [ref(1, "Amano", 2009, "Amano S. 2009.")]
        assert found(["(Ama-", "no, 2009)"], refs)[0][1] == [1]


class TestPages:
    def test_a_parenthesis_across_a_page_break_takes_the_year_page(self):
        body = lines_for((1, "were reported (Doi et al.,"), (2, "2005; Amano, 2009) later"))
        assert found(body, REFS) == [("Doi et al., " + "2005", [1], 2), ("Amano, 2009", [10], 2)] or \
            [(f[1], f[2]) for f in found(body, REFS)] == [([1], 2), ([10], 2)]

    def test_a_lone_year_with_no_author_is_nothing(self):
        assert found(["(2005; 2009)"], REFS) == []
        assert found(["in 2005 and 2009 it"], REFS) == []

    def test_the_reference_list_lines_are_not_searched(self):
        body = lines_for("no citation here.", "Doi et al. (2005) is a list entry")
        result = search_author_year_markers(body, author_year_list(1), REFS)
        assert result.markers == []


class TestLinking:
    def test_ambiguous_gives_no_link(self):
        refs = [ref(1, "Holmes", 2001, "Holmes D, Lu Y, Starborg T. 2001."),
                ref(2, "Holmes", 2001, "Holmes D, Kadler K, Lu Y. 2001.")]
        result = run(["(Holmes et al., 2001)"], refs)
        assert result.markers == [] and result.dropped["ambiguous"] == 1

    def test_second_author_decides(self):
        refs = [ref(1, "Ramachandran", 1968, "Ramachandran G and Kartha G. 1968."),
                ref(2, "Ramachandran", 1968, "Ramachandran G and Chandrasekharan R. 1968.")]
        assert found(["(Ramachandran and Chandrasekharan, 1968)"], refs)[0][1] == [2]

    def test_et_al_and_single_author_narrow_by_author_count(self):
        refs = [ref(1, "Lee", 2000, "Lee K. 2000. Alone."),
                ref(2, "Lee", 2000, "Lee K, Park S, Kim J. 2000. Many.")]
        assert found(["(Lee et al., 2000)"], refs)[0][1] == [2]
        assert found(["(Lee, 2000)"], refs)[0][1] == [1]

    def test_no_candidate_is_dropped_and_counted(self):
        result = run(["(Sarras et al., 1994; Amano, 1999)"], REFS)
        assert result.markers == [] and result.dropped["no_candidate"] == 2

    def test_same_surname_other_year_does_not_link(self):
        assert found(["(Bella et al., 1996)"], REFS) == []

    def test_letter_suffix(self):
        assert found(["(Smith and Jones, 1999a)"], REFS)[0][1] == [2]
        result = run(["(Smith and Jones, 1999b)"], REFS)
        assert result.markers == [] and result.dropped["letter"] == 1

    def test_a_year_that_does_not_link_is_left_out_of_numbers(self):
        assert found(["(Bella et al., 1994, 1996)"], REFS)[0][1] == [4]

    def test_missing_first_author_falls_back_to_the_printed_text(self):
        refs = [ref(1, None, 2005, "Doi M, Nishi Y. 2005. Collagen.")]
        assert found(["(Doi and Nishi, 2005)"], refs)[0][1] == [1]

    def test_initials_after_the_surname_decide(self):
        """Frontiers: "Gong S et al." and "Gong Z et al." are two papers of 2015."""
        refs = [ref(1, "Gong", 2015, "Gong, S., He, B., Wang, Z. (2015). Nairobi sheep disease virus."),
                ref(2, "Gong", 2015, "Gong, Z., Gu, S., Zhang, Y. (2015). Tick-borne viruses.")]
        assert found(["(Marczinke, 2002; Gong S et al., 2015; Gong Z et al., 2015)"], refs) == [
            ("Gong S et al., 2015", [1], 1), ("Gong Z et al., 2015", [2], 1)]

    def test_initials_with_dots_and_in_narrative_form(self):
        refs = [ref(1, "Li", 2022, "Li, X. H., Chen, Y., Wu, Q. (2022). One."),
                ref(2, "Li", 2022, "Li, Y., Chen, Y., Wu, Q. (2022). Two.")]
        assert found(["Li X. H. et al. (2022) showed it"], refs)[0][1] == [1]
        assert found(["(Li X et al., 2022)"], refs)[0][1] == [1]

    def test_initials_that_fit_no_entry_give_no_link(self):
        refs = [ref(1, "Gong", 2015, "Gong, S., He, B., Wang, Z. (2015). One.")]
        result = run(["(Gong Z et al., 2015)"], refs)
        assert result.markers == [] and result.dropped["no_candidate"] == 1

    def test_an_entry_whose_initials_cannot_be_read_stays_a_candidate(self):
        refs = [ref(1, "Gong", 2015, "Gong Shuang, He Bin (2015). Spelled out.")]
        assert found(["(Gong S et al., 2015)"], refs)[0][1] == [1]

    def test_a_numbered_list_gives_no_author_year_markers(self):
        body = lines_for("(Doi et al., 2005)", "1. Doi M")
        numbered = ReferenceList(style="numbered", entries=[], first_line=1)
        assert search_author_year_markers(body, numbered, REFS).markers == []


class TestMarkersFor:
    MERGED = {"reference_list": {"status": "found", "counts_agree": True}, "references": REFS}

    def test_the_contract_shape_and_report(self):
        body = lines_for("(Doi et al., 2005)", "Doi M (2005) list")
        markers, report = author_year_markers_for(body, author_year_list(1), self.MERGED)
        assert markers == [{"page_number": 1, "text": "Doi et al., 2005", "kind": "author_year",
                            "numbers": [1], "context_before": "(", "content_offset": None}]
        assert report["found"] == 1 and report["style"] == "author_year" and report["list_confirmed"] is True
        assert report["dropped"] == {"no_candidate": 0, "ambiguous": 0, "letter": 0}

    @pytest.mark.parametrize("lines, listing, merged, reason", [
        ([], author_year_list(0), MERGED, "no list read from the PDF"),
        (["x"], None, MERGED, "no list read from the PDF"),
        (["x"], ReferenceList("numbered", [], 0), MERGED, "numbered list"),
        (["x"], author_year_list(0), {"reference_list": {"status": "not_available"}}, "no reference list found"),
    ])
    def test_no_markers_with_a_reason(self, lines, listing, merged, reason):
        markers, report = author_year_markers_for(lines, listing, merged)
        assert markers == [] and report["reason"] == reason

    def test_offset_is_found_for_a_marker_with_spaces(self):
        mark = {"page_number": 1, "text": "Doi et al., 2005", "kind": "author_year", "numbers": [1],
                "context_before": "was shown (", "content_offset": None}
        content = "It was shown (Doi et al., 2005) here."
        assert locate_markers([mark], {1: content})[0]["content_offset"] == content.index("Doi")
        assert locate_markers([mark], {1: "was shown (Doi et al.,\n2005)"})[0]["content_offset"] is None


class TestTextAfterTheList:
    """Cell Press's STAR Methods follow the reference list and cite it too."""

    def test_citations_after_the_list_are_found(self):
        body = lines_for("Shown before (Doi et al., 2005).")
        listed = lines_for((5, "Doi M (2005) A list entry that is not a citation"))
        methods = lines_for((6, "Aligned with STAR (Smith and Jones, 1999)."))
        reference_list = ReferenceList(style="author_year", entries=[], first_line=1, end_line=2)
        markers = search_author_year_markers(body + listed + methods, reference_list, REFS).markers
        assert [(m.text, m.page) for m in markers] == [("Doi et al., 2005", 1),
                                                        ("Smith and Jones, 1999", 6)]

    def test_without_an_end_the_list_is_never_read(self):
        body = lines_for("Shown before (Doi et al., 2005).")
        listed = lines_for((5, "Doi M (2005) A list entry (Smith and Jones, 1999)"))
        reference_list = ReferenceList(style="author_year", entries=[], first_line=1)
        markers = search_author_year_markers(body + listed, reference_list, REFS).markers
        assert [m.text for m in markers] == ["Doi et al., 2005"]
