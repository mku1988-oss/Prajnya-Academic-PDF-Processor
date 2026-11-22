"""Tests for utils/reference_list.py: finding a reference list and splitting it.

For biology researchers: these build tiny documents line by line and check
that the list is found where it is printed, split where entries start, and
that each entry's first author and year are read correctly.
"""
import re

import pytest

from src.processing.utils.layout_lines import LayoutLine
from src.processing.utils.reference_list import entry_start_at, entry_year, find_reference_list, first_author


def line(text: str, *, page: int = 1, x0: float = 40.0, baseline: float = 100.0,
         column: int = 0) -> LayoutLine:
    return LayoutLine(page=page, band=0, column=column, text=text, x0=x0, x1=x0 + 5 * len(text),
                      baseline=baseline, height=10.0, chars=[])


def numbered(rows, page=2):
    return [line(text, page=page, baseline=100 + 12 * i) for i, text in enumerate(rows)]


ENTRIES = [f"{n} Author{n}, A.B. and Other, C. ({1990 + n}) A title. J. Biol. Chem. {n}, 1-9" for n in range(1, 9)]


class TestNumberedLists:
    def test_the_list_is_found_without_a_heading(self):
        found = find_reference_list(numbered(ENTRIES))
        assert found.style == "numbered"
        assert [e.number for e in found.entries] == list(range(1, 9))
        assert found.entries[2].first_author == "Author3" and found.entries[2].year == "1993"

    def test_a_numbered_heading_in_the_body_does_not_start_the_list(self):
        body = [line("1 Introduction", page=1), line("2 Methods", page=1, baseline=200)]
        found = find_reference_list(body + numbered(ENTRIES))
        assert found.entries[0].page == 2 and found.entries[1].page == 2

    def test_numbered_sections_are_not_a_list(self):
        sections = [f"{n} Section heading number {n} since {1990 + n} the field" for n in range(1, 9)]
        assert find_reference_list(numbered(sections)) is None

    def test_continuation_lines_join_their_entry(self):
        rows = []
        for text in ENTRIES:
            rows += [text, "continued on a second line"]
        found = find_reference_list(numbered(rows))
        assert found.entries[0].text.endswith("continued on a second line")
        assert len(found.entries) == 8


class TestAuthorYearLists:
    def test_hanging_indent_splits_entries_at_the_left_edge(self):
        rows = [line("Introduction text", page=1), line("References", page=3, baseline=90)]
        y = 100
        for n in range(6):
            rows.append(line(f"Smith{n}, J., & Jones, K. ({2000 + n}). A long title that", page=3, baseline=y))
            rows.append(line("continues here. Journal, 12, 1-9.", page=3, x0=52, baseline=y + 10))
            y += 20
        found = find_reference_list(rows)
        assert found.style == "author_year"
        assert [e.first_author for e in found.entries] == [f"Smith{n}" for n in range(6)]
        assert found.entries[0].text.endswith("Journal, 12, 1-9.")

    def test_block_style_splits_entries_at_wider_gaps(self):
        rows = [line("REFERENCES", page=3, baseline=90)]
        y = 100
        for n in range(6):
            rows.append(line(f"Lee{n}, B., and Park, C. ({2000 + n}). A title", page=3, baseline=y))
            rows.append(line("Cell 12, 1-9.", page=3, baseline=y + 9.5))
            y += 21.5
        found = find_reference_list(rows)
        assert [e.first_author for e in found.entries] == [f"Lee{n}" for n in range(6)]

    def test_the_list_ends_at_a_following_heading(self):
        rows = [line("References", page=3, baseline=90)]
        y = 100
        for n in range(6):
            rows.append(line(f"Smith{n}, J., & Jones, K. ({2000 + n}). Title.", page=3, baseline=y))
            rows.append(line("Journal 1, 2.", page=3, x0=52, baseline=y + 10))
            y += 20
        rows.append(line("Acknowledgements", page=3, baseline=y))
        rows.append(line("Funder, A., & Grant, B. (2020). We thank.", page=3, baseline=y + 10))
        assert len(find_reference_list(rows).entries) == 6


@pytest.mark.parametrize("text, surname", [
    ("Fan, P., Li, M.H., Brodsky, B. (1993) Backbone", "Fan"),
    ("Edzes HT, Samulski ET. Cross relaxation", "Edzes"),
    ("AMANO S (2009). Possible involvement", "AMANO"),
    ("van der Flier, L.G., van Gijn, M.E.", "van der Flier"),
    ("Maki, N. et al. Expression profiles", "Maki"),
    ("Lee, J.; Choe, I. R.; Kim, Y.-O.", "Lee"),
    ("Wikimedia Commons. Pine cone (2017)", "Wikimedia Commons"),
    ("Montero de Espinosa, L., Meesorn, W.", "Montero de Espinosa"),
])
def test_first_author(text, surname):
    assert first_author(text) == surname


@pytest.mark.parametrize("text, year", [
    ("Fan, P. (1993) Backbone dynamics. Biochemistry 32, 13299-13309", "1993"),
    ("Craig, A. S. (1981a). Growth of fibrils.", "1981a"),
    ("Edzes HT. Nature 1977;265:521-3.", "1977"),
    ("Kim, Y. Proc. Natl Acad. Sci. 105, 1973-1995 (2008).", "2008"),
    ("Miserez, A. Biomaterials 1973-1985, 2012.", "2012"),
    ("Kukkola, L. et al. A third prolyl 4-hydroxylase. J. Biol. Chem. (in press)", None),
])
def test_entry_year(text, year):
    assert entry_year(text) == year


@pytest.mark.parametrize("text, surname", [
    ("P. W. Reddien, A. Sánchez Alvarado, Annu. Rev. Cell Dev. Biol. 20, 725 (2004).", "Reddien"),
    ("A. Sánchez Alvarado, P. A. Newmark, Proc. Natl. Acad. Sci.", "Sánchez Alvarado"),
    ("A. R. Kidd 3rd, J. A. Miskowski, K. R. Siegfried", "Kidd"),
    ("A. Adoutte et al., Proc. Natl. Acad. Sci. U.S.A. 97, 4453 (2000).", "Adoutte"),
    ("Kaelin Jr WG. The concept of synthetic lethality", "Kaelin"),
])
def test_first_author_initials_first_and_suffixes(text, surname):
    assert first_author(text) == surname


def test_a_year_before_1900_is_read():
    assert entry_year("H. Randolph, Arch. Entw. Mech. Org. 5, 352 (1897).") == "1897"


class TestNumberOnItsOwnLine:
    """MDPI prints each entry's number alone on a line, the entry on the next."""

    @staticmethod
    def mdpi_rows(count=8):
        rows = []
        for n in range(1, count + 1):
            rows += [f"{n}.", f"Author{n}, A.B.; Other, C. A title. J. Biol. Chem. {1990 + n}, {n}, 1-9."]
        return rows

    def test_a_list_with_numbers_on_their_own_lines_is_found(self):
        found = find_reference_list(numbered(self.mdpi_rows()))
        assert found.style == "numbered"
        assert [e.number for e in found.entries] == list(range(1, 9))
        assert found.entries[0].text.startswith("Author1,") and found.entries[0].first_author == "Author1"

    def test_the_list_starts_at_the_real_first_line_not_at_body_numbers(self):
        body = [line("1. Introduction", page=1), line("2 of 36", page=2, baseline=200),
                line("3 of 36", page=3, baseline=200)]
        rows = body + numbered(self.mdpi_rows(), page=4)
        found = find_reference_list(rows)
        assert found.first_line == 3 and rows[found.first_line].text == "1."

    def test_a_bare_number_without_punctuation_does_not_start_an_entry(self):
        lines = [line("3"), line("Smith, J. A title (2001).")]
        assert entry_start_at(lines, 0) is None

    @staticmethod
    def centred_rows():
        """Entries 1-7 number-first; 8 centred (number after its first line)."""
        rows = []
        for n in range(1, 7):
            rows += [f"{n}.", f"Author{n}, A.B.; Other, C. A title. J. Biol. Chem. {1990 + n}, {n}, 1-9."]
        rows += ["7.", "Shimizu, H. Transplantation Analysis of Developmental Mechanisms in Hy", "[PubMed]",
                 "Vogg, M.C.; Wenger, Y.; Galliot, B. How Somatic Adult Tissues Develop 2019.",
                 "8.", "[CrossRef] [PubMed]",
                 "9.", "Vogg, M.C.; Galliot, B.; Tsiairis, C.D. Model Systems for Regeneration 2020.",
                 "10.", "Zed, A.; Other, B. Last entry. J. Biol. Chem. 2021, 3, 1-9."]
        return rows

    def test_a_number_centred_after_its_entry_first_line_is_found(self):
        found = find_reference_list(numbered(self.centred_rows()))
        assert [e.number for e in found.entries] == list(range(1, 11))
        assert found.entries[7].text.startswith("Vogg")
        assert "Vogg" not in found.entries[6].text
        assert not any(re.search(r"(?:^|\s)8\.(?:\s|$)", e.text) for e in found.entries)

    def test_a_last_line_that_is_only_a_page_number_is_kept_in_the_ordinary_layout(self):
        rows = list(ENTRIES)
        rows[4] = "5 Author5, A.B. and Other, C. (1995) A title. Nat. Chem. 2014, 6,"
        rows.insert(5, "596.")
        found = find_reference_list(numbered(rows))
        assert [e.number for e in found.entries] == list(range(1, 9))
        assert found.entries[4].text.endswith("6, 596.")

    def test_an_mdpi_entry_loses_its_own_number_line_only(self):
        found = find_reference_list(numbered(self.mdpi_rows()))
        assert not any(re.search(r"(?:^|\s)\d\.(?:\s|$)", e.text) for e in found.entries)

    def test_an_mdpi_entry_ending_in_a_page_number_keeps_it(self):
        rows = self.mdpi_rows()
        rows[1] = "Author1, A.B.; Other, C. A title. J. Biol. Chem. 2004, 169, 65-"
        rows.insert(2, "73.")
        found = find_reference_list(numbered(rows))
        assert [e.number for e in found.entries] == list(range(1, 9))
        assert found.entries[0].text.endswith("65- 73.")
        assert not found.entries[1].text.startswith("73.")

    def test_the_list_start_ignores_body_numbers_tying_up_to_a_centred_number(self):
        body = [line("1. Introduction", page=1)]
        body += [line(f"{n} of 36", page=n, baseline=200) for n in range(2, 9)]
        rows = body + numbered(self.centred_rows(), page=9)
        found = find_reference_list(rows)
        assert rows[found.first_line].text == "1."
        assert found.first_line == len(body)


class TestCellPressMethodsAfterTheList:
    """Cell Press prints STAR Methods AFTER the reference list (item 000a, fix 5)."""

    def rows(self):
        rows = [line("Introduction text", page=1), line("REFERENCES", page=3, baseline=90)]
        y = 100
        for n in range(6):
            rows.append(line(f"Smith{n}, J., and Jones, K. ({2000 + n}). A long title that", page=3, baseline=y))
            rows.append(line("continues here. Cell 12, 1-9.", page=3, x0=52, baseline=y + 10))
            y += 20
        rows.append(line("STAR+METHODS", page=4, baseline=90))
        rows.append(line("KEY RESOURCES TABLE", page=4, baseline=102))
        rows.append(line("EXPERIMENTAL MODEL AND SUBJECT DETAILS", page=4, baseline=114))
        return rows

    def test_the_methods_heading_ends_the_list(self):
        found = find_reference_list(self.rows())
        assert [e.first_author for e in found.entries] == [f"Smith{n}" for n in range(6)]
        assert found.entries[-1].text.endswith("Cell 12, 1-9.")      # no KEY RESOURCES TABLE

    def test_the_list_says_where_it_ends(self):
        rows = self.rows()
        found = find_reference_list(rows)
        assert rows[found.end_line].text == "STAR+METHODS"

    def test_a_list_running_to_the_end_has_no_end_line(self):
        assert find_reference_list(self.rows()[:-3]).end_line is None


def sized(text: str, *, height: float, x0: float = 50.0, column: int = 0, baseline: float = 100.0,
          page: int = 8) -> LayoutLine:
    """A line with its own text size (`height`), as Frontiers' back matter needs."""
    return LayoutLine(page=page, band=0, column=column, text=text, x0=x0, x1=x0 + 5 * len(text),
                      baseline=baseline, height=height, chars=[])


class TestFrontiersNotesInsideTheList:
    """Frontiers (2023 on): the list starts in the left column, the right column
    first finishes the back matter ("Conflict of interest", in a larger size),
    then carries on with the list; and each entry's FIRST line is indented.
    Shape measured on PMC10977656 (entries 9.44, notes 12.14, headings 14.88)."""

    ENTRY, NOTE, HEADING = 9.44, 12.14, 14.88

    def entry(self, n, column=0):
        x = 50.0 if column == 0 else 305.0
        return [sized(f"Surname{n}, A., and Other, B. ({2000 + n}). A title that runs", height=self.ENTRY,
                      x0=x + 6, column=column, baseline=100 + 30 * n),
                sized("on to a second line. J. Biol. 12, 1-9.", height=self.ENTRY, x0=x, column=column,
                      baseline=110 + 30 * n)]

    def rows(self, note_size=None):
        note_size = note_size or self.NOTE
        rows = [sized("Body text of the paper.", height=self.NOTE, page=7),
                sized("References", height=self.HEADING)]
        for n in range(6):
            rows += self.entry(n)
        rows += [sized("Conflict of interest", height=self.HEADING, column=1, x0=305),
                 sized("The authors declare no commercial relationships.", height=note_size, column=1, x0=305)]
        for n in range(6, 12):
            rows += self.entry(n, column=1)
        return rows

    def test_the_list_goes_on_after_the_note(self):
        found = find_reference_list(self.rows())
        assert [e.first_author for e in found.entries] == [f"Surname{n}" for n in range(12)]
        assert found.end_line is None
        assert not any("declare" in e.text for e in found.entries)

    def test_entries_start_at_the_indented_first_line(self):
        found = find_reference_list(self.rows())
        assert found.entries[0].text.endswith("J. Biol. 12, 1-9.")

    def test_a_box_in_the_lists_own_size_still_ends_it(self):
        """lib02 (Int. J. Dev. Biol.) closes with a "Further Related Reading" box
        set in the list's size: no note, so the list ends at its heading."""
        found = find_reference_list(self.rows(note_size=self.ENTRY))
        assert len(found.entries) == 6


class TestFragmentsOfTheNeighbouringColumn:
    """Cell Reports (PMC8333197 p12): one- and two-word fragments of the LEFT
    column ("their", "s. We", "red in") sit ~29 points left of the right
    column's real edge. Taken as the column's edge, they made every real line
    look indented, the column was read as Frontiers' first-line indent, and an
    author list wrapped onto a line starting "Bhatt, A.S. (2019)." became a
    second entry."""

    EDGE, FRAGMENT = 304.0, 275.0

    def entry(self, n, top):
        """A block-style entry whose author list wraps onto a line that starts like an author."""
        return [line(f"Surname{n}, R., Tamburini, F.B., Asiimwe, E., Oduaran, O., and", x0=self.EDGE,
                     baseline=top, column=1, page=12),
                line(f"Wrapped{n}, A.S. ({2000 + n}). Surveying gut microbiome research in",
                     x0=self.EDGE, baseline=top + 10, column=1, page=12),
                line("Africans. Trends Microbiol. 27, 824-835.", x0=self.EDGE,
                     baseline=top + 20, column=1, page=12)]

    def rows(self, with_fragments=True):
        rows = [line("References", x0=self.EDGE, baseline=80, column=1, page=12)]
        for n in range(6):
            rows += self.entry(n, top=100 + 36 * n)
        if with_fragments:
            fragments = ["their", "s. We", "red in", "as an", "Clos", "etics,"]
            rows += [line(text, x0=self.FRAGMENT, baseline=110.5 + 36 * n, column=1, page=12)
                     for n, text in enumerate(fragments)]
            rows.sort(key=lambda row: row.baseline)
        return rows

    def test_without_fragments_each_entry_is_one_reference(self):
        found = find_reference_list(self.rows(with_fragments=False))
        assert [e.first_author for e in found.entries] == [f"Surname{n}" for n in range(6)]

    def test_fragments_do_not_split_wrapped_author_lists(self):
        found = find_reference_list(self.rows())
        assert [e.first_author for e in found.entries] == [f"Surname{n}" for n in range(6)]
