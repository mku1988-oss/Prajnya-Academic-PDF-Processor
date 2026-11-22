"""Tests for utils/citation_markers.py: numeric citation markers linked to entries.

For biology researchers: each test prints one short line, with some digits
raised like a superscript where needed, and checks which numbers come back
as citations, and which raised numbers (powers, charges, isotopes) do not.
"""
from typing import List

from src.processing.pdf.base import PrintedChar, Rect
from src.processing.utils.citation_markers import (expand, find_numeric_markers, is_author_line, line_words,
                                                      search_numeric_markers)
from src.processing.utils.layout_lines import page_layout_lines
from src.processing.utils.reference_list import ReferenceEntry, ReferenceList

WIDTH, HEIGHT = 5.0, 10.0


def run(text: str, x: float, *, raised: bool = False, space: bool = False, baseline: float = 100.0):
    """Characters for `text` from x; raised ones sit 4 pt up and are 70% the size."""
    chars: List[PrintedChar] = []
    for index, character in enumerate(text):
        if character == " ":
            space = True
            x += WIDTH
            continue
        height = HEIGHT * (0.7 if raised else 1.0)
        base = baseline - (4.0 if raised else 0.0)
        chars.append(PrintedChar(text=character, rect=Rect(x, base - 0.8 * height, x + WIDTH, base + 0.2 * height),
                                 baseline=base, space_before=space))
        space = False
        x += WIDTH
    return chars


def body_line(*pieces):
    """One printed line from (text, raised) pieces, set left to right."""
    chars, x = [], 40.0
    for text, raised in pieces:
        part = run(text, x, raised=raised)
        chars += part
        x = part[-1].rect.x1 if part else x
    return page_layout_lines(chars, 1)


def numbered_list(n: int = 40, first_line: int = 1) -> ReferenceList:
    entries = [ReferenceEntry(number=k, page=9, text="", first_author="A", year="2000") for k in range(1, n + 1)]
    return ReferenceList(style="numbered", entries=entries, first_line=first_line)


def numbers_found(lines, entries: int = 40) -> List[List[int]]:
    return [m.numbers for m in find_numeric_markers(lines, numbered_list(entries, first_line=len(lines)))]


class TestSuperscripts:
    def test_a_raised_number_glued_to_a_word_cites(self):
        assert numbers_found(body_line(("thermal stability", False), ("24", True))) == [[24]]

    def test_a_range_with_a_full_size_dash_is_one_marker(self):
        chars = run("nanotubes,", 40) + run("26", 90, raised=True) + run("−", 100) + run("28", 105, raised=True)
        assert numbers_found(page_layout_lines(chars, 1)) == [[26, 27, 28]]

    def test_an_exponent_is_not_a_citation(self):
        assert numbers_found(body_line(("0.3 mS cm", False), ("−1", True))) == []

    def test_a_square_after_a_lone_capital_is_not_a_citation(self):
        assert numbers_found(body_line(("fit, R", False), ("2", True), (" > 90%", False))) == []

    def test_a_charge_is_not_a_citation(self):
        assert numbers_found(body_line(("cations, Zn", False), ("2+", True))) == []

    def test_an_isotope_is_not_a_citation(self):
        assert numbers_found(body_line(("water ", False), ("1", True), ("H NMR", False))) == []

    def test_a_power_of_ten_is_not_a_citation(self):
        assert numbers_found(body_line(("values of 10", False), ("3", True))) == []


class TestBrackets:
    def test_brackets_win_over_raised_digits(self):
        lines = body_line(("collagen [3", False), ("–", False), ("5] and Hyp", False), ("4", True),
                          ("S [7] and [12,14].", False))
        assert numbers_found(lines) == [[3, 4, 5], [7], [12, 14]]


class TestSelfCheck:
    def test_a_number_beyond_the_list_drops_every_link(self):
        lines = body_line(("shown before [3] and [5] and [77].", False))
        assert find_numeric_markers(lines, numbered_list(40, first_line=len(lines))) == []

    def test_an_author_year_list_gets_no_numeric_links(self):
        lines = body_line(("shown before [3] and [5] and [7].", False))
        author_year = ReferenceList(style="author_year", entries=numbered_list().entries, first_line=len(lines))
        assert find_numeric_markers(lines, author_year) == []


def test_expand():
    assert expand("3,5–7") == [3, 5, 6, 7]
    assert expand("7–5") == []


class TestParentheses:
    def test_parenthesised_numbers_cite_when_they_are_the_style(self):
        lines = body_line(("as in (1) and (2, 3) the domain (20). Zhang et al. (22) and shadowed (23) "
                           "as by Grens et al. (24, 25) and (26\u201328).", False))
        assert numbers_found(lines, entries=28) == [[1], [2, 3], [20], [22], [23], [24, 25], [26, 27, 28]]

    def test_parentheses_citing_too_little_of_the_list_are_not_the_style(self):
        lines = body_line(("domain (20). Zhang et al. (22) and shadowed (23).", False))
        assert numbers_found(lines, entries=40) == []

    def test_quantities_figures_and_equation_numbers_are_not_citations(self):
        lines = body_line(("in Fig. (2) and Eq. (3) with (500) cells in (5) and (7) and (9).", False))
        assert numbers_found(lines, entries=10) == [[5], [7], [9]]

    def test_a_few_parentheses_do_not_outvote_raised_markers(self):
        lines = body_line(("as in (3) and (5) and (7) shown before", False), ("12", True),
                          (" and also", False), ("14", True), (" and", False), ("15", True),
                          (" and", False), ("16", True))
        assert numbers_found(lines) == [[12], [14], [15], [16]]


def lines_for(*sentences, page=1):
    """One printed line per sentence, each a list of (text, raised) pieces, all on one page, stacked downwards."""
    chars = []
    for index, pieces in enumerate(sentences):
        x = 40.0
        for text, raised in pieces:
            part = run(text, x, raised=raised, baseline=100.0 + 20.0 * index)
            chars += part
            x = part[-1].rect.x1 if part else x
    return page_layout_lines(chars, page)


def search(lines, entries=40, **options):
    return search_numeric_markers(lines, numbered_list(entries, first_line=len(lines)), **options)


AUTHORS = [[("Jean-Philippe Lambert", False), ("1,13", True), (", Gordana Ivosev", False), ("2,13", True),
            (", Amber L Couzens", False), ("1", True), (",", False)],
           [("Zhen-Yuan Lin", False), ("1", True), (", Quan Zhong", False), ("4\u20136", True), (",", False)]]
BODY = [("first shown", False), ("1", True), (" then", False), ("2", True), (" then", False), ("3", True)]


class TestListLength:
    BRACKETS = [("seen [1] and [2] and [3] but also [999] and [4].", False)]

    def test_a_confirmed_list_drops_only_the_marker_beyond_it(self):
        found = search(lines_for(self.BRACKETS), list_confirmed=True)
        assert [m.numbers for m in found.markers] == [[1], [2], [3], [4]]
        assert found.dropped["beyond_list"] == 1

    def test_an_unconfirmed_list_gives_no_markers_for_the_same_input(self):
        found = search(lines_for(self.BRACKETS))
        assert found.markers == [] and found.dropped["beyond_list"] == 0

    def test_last_given_explicitly_overrides_the_lists_own_maximum(self):
        lines = lines_for([("seen [1] and [2] and [3] and [30].", False)])
        assert [m.numbers for m in search(lines, last=5).markers] == []
        assert [m.numbers for m in search(lines, last=5, list_confirmed=True).markers] == [[1], [2], [3]]
        assert len(search(lines).markers) == 4

    def test_a_confirmed_list_also_drops_a_raised_marker_beyond_it(self):
        lines = lines_for([("alpha", False), ("1", True), (" beta", False), ("2", True), (" gamma", False),
                           ("3", True), (" delta", False), ("77", True)])
        found = search(lines, list_confirmed=True)
        assert [m.numbers for m in found.markers] == [[1], [2], [3]] and found.dropped["beyond_list"] == 1


class TestOrderCheck:
    def test_affiliation_marks_on_an_author_line_are_dropped(self):
        found = search(lines_for(*AUTHORS, BODY, page=4))
        assert [m.numbers for m in found.markers] == [[1], [2], [3]]
        assert found.dropped["order_check"] == 5

    def test_the_same_result_when_the_list_is_confirmed(self):
        found = search(lines_for(*AUTHORS, BODY, page=4), list_confirmed=True)
        assert [m.numbers for m in found.markers] == [[1], [2], [3]]

    def test_a_name_run_on_the_same_page_goes_with_the_one_that_broke_the_order(self):
        quiet = [("Ruedi Aebersold", False), ("1", True), (",", False)]       # alone it keeps the order
        found = search(lines_for(AUTHORS[0], quiet, BODY, page=4))
        assert [m.numbers for m in found.markers] == [[1], [2], [3]]
        assert found.dropped["order_check"] == 4

    def test_a_name_run_that_continues_the_order_stays(self):
        lines = lines_for([("known", False), ("15", True), (" and", False), ("16", True), (" and", False),
                           ("17", True), (" by Collins", False), ("18", True), (", Smith", False),
                           ("19", True), (" and Jones", False)])
        found = search(lines, entries=25)
        assert [m.numbers for m in found.markers] == [[15], [16], [17], [18], [19]]
        assert found.dropped["order_check"] == 0

    def test_a_name_like_word_followed_by_a_full_stop_is_not_name_attached(self):
        lines = lines_for([("by SWATH", False), ("18", True), (". Then", False), ("2", True)], page=4)
        assert [m.numbers for m in search(lines).markers] == [[18], [2]]

    def test_brackets_and_parentheses_are_not_touched_by_the_order_check(self):
        brackets = lines_for([("Smith [13], Jones [2], Lee [30] and Kim [5].", False)])
        assert [m.numbers for m in search(brackets).markers] == [[13], [2], [30], [5]]
        parens = lines_for([("as in Smith (13), Jones (2), Lee (17) and Kim (5, 6) and (7), (8), (9).", False)])
        found = search(parens, entries=20)
        assert [m.numbers[0] for m in found.markers][:4] == [13, 2, 17, 5] and found.dropped["order_check"] == 0


class TestRedundantNameRuns:
    """Author-line marks that keep the order (1, 2, 3) are still dropped when the body does not need them."""

    BODY_1_TO_4 = [("first shown", False), ("1", True), (" then", False), ("2", True), (" then", False),
                   ("3", True), (" then", False), ("4", True)]

    def test_affiliation_marks_that_keep_the_order_are_dropped(self):
        authors = [("Jinyong Luo", False), ("1,2", True), (",", False), (" Jin Chen", False), ("1,2", True),
                   (",", False), (" Zhong Deng", False), ("2,3", True), (",", False)]
        found = search(lines_for(authors, self.BODY_1_TO_4, page=4))
        assert [m.numbers for m in found.markers] == [[1], [2], [3], [4]]
        assert found.dropped["order_check"] == 3

    def test_a_comma_printed_before_the_raised_number_is_name_attached(self):
        authors = [("Alexandra Silverman,", False), ("1", True), (" Jason Olszewski,", False), ("1", True),
                   (" Seyed Siadat,", False), ("1", True), (" and Jeffrey Ruberti", False), ("1,2", True),
                   (",", False), (" *", False)]
        body = [("first shown", False), ("1", True), (" then", False), ("2", True), (" then", False), ("3", True)]
        found = search(lines_for(authors, body, page=4))
        assert [m.numbers for m in found.markers] == [[1], [2], [3]]
        assert found.dropped["order_check"] == 4

    def test_a_corresponding_author_sign_after_the_mark_is_name_attached(self):
        authors = [("Alexandra Silverman", False), ("1", True), (" *", False), (" and Ruberti", False),
                   ("1,2", True), (",", False)]
        body = [("first shown", False), ("1", True), (" then", False), ("2", True), (" then", False), ("3", True)]
        found = search(lines_for(authors, body, page=4))
        assert [m.numbers for m in found.markers] == [[1], [2], [3]] and found.dropped["order_check"] == 2

    def test_a_genuine_opening_citation_stays_when_the_body_starts_at_three(self):
        opening = [("by Collins", False), ("1", True), (", Smith", False), ("2", True), (" and Jones", False)]
        body = [("then", False), ("3", True), (" and", False), ("4", True), (" and", False), ("1", True)]
        found = search(lines_for(opening, body, page=4))
        assert [m.numbers for m in found.markers] == [[1], [2], [3], [4], [1]]
        assert found.dropped["order_check"] == 0

    def test_a_run_without_a_first_appearance_of_one_is_not_redundant(self):
        known = [("known", False), ("1", True), (" and", False), ("2", True), (" and", False), ("3", True),
                 (" by Collins", False), ("4", True), (", Smith", False), ("5", True), (" and Jones", False)]
        found = search(lines_for(known))
        assert [m.numbers for m in found.markers] == [[1], [2], [3], [4], [5]]

    def test_a_second_name_run_on_the_page_goes_with_a_redundant_one(self):
        first = [("Jinyong Luo", False), ("1,2", True), (",", False), (" Jin Chen", False), ("1,2", True),
                 (",", False)]
        second = [("Rex Haydon", False), ("2", True), (",", False), (" Hue Luu", False), ("2", True), (" and", False)]
        found = search(lines_for(first, second, self.BODY_1_TO_4, page=4))
        assert [m.numbers for m in found.markers] == [[1], [2], [3], [4]]
        assert found.dropped["order_check"] == 4


def plain(*sentences):
    """Lines for plain sentences, one printed line each (nothing raised)."""
    return lines_for(*[[(sentence, False)] for sentence in sentences])


CITED_IN_TEXT = "As shown (1) and (2) and (3) and (4) and (5) and (6) and (7) before."
LANE_LEGEND = ("PFGE of DNA digested with ClaI (2), NruI (3), MluI (4), NarI (5), PvuI (6), SmaI (7), "
               "or undigested (1) and hybridized.")


class TestEnumerations:
    def test_lane_numbers_in_a_legend_after_the_text_cited_them_are_dropped(self):
        found = search(plain(CITED_IN_TEXT, LANE_LEGEND), entries=10)
        assert [m.numbers for m in found.markers] == [[k] for k in range(1, 8)]
        assert found.dropped["enumeration"] == 7

    def test_the_same_legend_in_a_paper_that_has_not_cited_those_numbers_is_kept(self):
        found = search(plain("As shown (8) and (9) and (10) before.", LANE_LEGEND), entries=10)
        assert [m.numbers for m in found.markers] == [[8], [9], [10], [2], [3], [4], [5], [6], [7], [1]]
        assert found.dropped["enumeration"] == 0

    def test_five_recited_brackets_starting_at_seven_are_kept(self):
        first = "Seen " + ", ".join(f"[{k}]" for k in range(1, 13)) + " before."
        found = search(plain(first, "Listed [7], [8], [9], [10], [11]."))
        assert [m.numbers[0] for m in found.markers][-5:] == [7, 8, 9, 10, 11]
        assert found.dropped["enumeration"] == 0

    def test_five_recited_brackets_starting_at_one_are_dropped(self):
        first = "Seen " + ", ".join(f"[{k}]" for k in range(1, 13)) + " before."
        found = search(plain(first, "Listed [1], [2], [3], [4], [5]."))
        assert len(found.markers) == 12
        assert found.dropped["enumeration"] == 5

    def test_three_recited_parentheses_are_kept(self):
        found = search(plain("As shown (1) and (2) and (3) and (4) and (5) and (6) before.",
                             "Then (1), (2), (3)."), entries=10)
        assert [m.numbers for m in found.markers][-3:] == [[1], [2], [3]]
        assert found.dropped["enumeration"] == 0


class TestRangesAndBrokenParentheses:
    def test_a_range_with_a_plain_keyboard_hyphen_is_linked(self):
        found = search(plain("Seen [1] and [3-12] and [20-23] and [7] before."), entries=30)
        assert [m.numbers[0] for m in found.markers] == [1, 3, 20, 7]
        assert found.markers[1].numbers == list(range(3, 13))

    def test_a_parenthesised_range_with_a_plain_hyphen_is_linked(self):
        found = search(plain("Seen (1) and (5-7) and (9-12) and (2) and (3) and (4) before."), entries=20)
        assert [m.numbers for m in found.markers][1:3] == [[5, 6, 7], [9, 10, 11, 12]]

    def test_a_parenthesis_split_across_two_lines_is_one_marker(self):
        found = search(plain("As shown before (15,", "19, 24, 27, 29) and also (1) and (2) and (5) and (6) and "
                             "(7) and (8) and (9)."), entries=30)
        first = found.markers[0]
        assert first.numbers == [15, 19, 24, 27, 29]
        assert first.page == 1 and first.context == "As shown before"
        assert [m.numbers[0] for m in found.markers[1:3]] == [1, 2]

    def test_a_parenthesis_closing_after_a_figure_legend_is_one_marker(self):
        found = search(plain("hydrozoan (15, 18, 22,", "Fig. 4. A legend line read between the two halves.",
                             "23) species appeared as a family (1) and (2) and (3) and (4) and (5) and (6) and (7).",
                             ), entries=30)
        assert found.markers[0].numbers == [15, 18, 22, 23]
        assert found.markers[0].context == "hydrozoan"


class TestAuthorLines:
    """Raised numbers on an author line (first 3 pages, a line of names) are never citations."""

    NAMES = [("Huong Le, MD", False), ("1", True), ("; Rebecca Kleinerman, MD", False), ("1", True),
             ("; Oren Z. Lerman, MD", False), ("2", True), (";", False)]
    CITING = [("first shown", False), ("1", True), (" then", False), ("2", True), (" then", False), ("3", True)]

    def test_an_author_line_with_degrees_and_semicolons_gives_no_marker(self):
        found = search(lines_for(self.NAMES, self.CITING))
        assert [m.numbers for m in found.markers] == [[1], [2], [3]]
        assert found.dropped["author_line"] == 3

    def test_a_body_line_with_lowercase_prose_is_kept(self):
        lines = lines_for([("in the developing durable tissue", False), ("1", True), (" and others", False),
                           ("2", True)])
        found = search(lines)
        assert [m.numbers for m in found.markers] == [[1], [2]] and found.dropped["author_line"] == 0

    def test_the_same_line_on_page_four_is_not_an_author_line(self):
        chars = []
        for text, raised in self.NAMES:
            chars += run(text, 40.0 + 5 * len(chars), raised=raised)
        lines = page_layout_lines(chars, 4)
        assert not is_author_line(lines[0])
        assert search(lines).dropped["author_line"] == 0

    def test_one_lowercase_word_and_two_names_is_still_an_author_line(self):
        lines = lines_for([("nd Xuelian Luo", False), ("1", True)])
        assert is_author_line(lines[0])
        assert search(lines).dropped["author_line"] == 1

    def test_line_words_leave_out_raised_characters(self):
        lines = lines_for([("Huong Le", False), ("12", True), (", Kim", False)])
        assert line_words(lines[0]) == ["Huong", "Le", "Kim"]

    def test_particles_are_skipped(self):
        lines = lines_for([("Annette van der Flier", False), ("1", True)])
        assert is_author_line(lines[0])
        assert not is_author_line(lines_for([("Annette is here", False)])[0])
