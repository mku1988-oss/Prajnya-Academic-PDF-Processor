"""
Tests for ledger T30 (v6.9 item 3, S3b): names broken over a line, folded
letters, and a reference heading that opens no list.

For biology researchers: tiny hand-built documents. A surname hyphenated
over a line end must read as one name again, a name printed with a dotless
i and a combining accent must match its reference, and a publisher's cover
sheet with its own "References" heading must not hide the real list.
"""
from types import SimpleNamespace

from src.processing.utils.author_year_markers import fold_letters, search_author_year_markers
from src.processing.utils.layout_lines import LayoutLine, join_line_texts
from src.processing.utils.reference_list import ReferenceList, find_reference_list


def line(text, *, page=1, x0=40.0, baseline=100.0):
    return LayoutLine(page=page, band=0, column=0, text=text, x0=x0, x1=x0 + 5 * len(text),
                      baseline=baseline, height=10.0, chars=[])


class TestJoinLineTexts:
    def test_a_broken_word_is_mended(self):
        assert join_line_texts(["work of Lomvar-", "das et al."]) == "work of Lomvardas et al."

    def test_a_real_hyphen_before_a_capital_is_kept(self):
        assert join_line_texts(["Martinez-", "Palomo, A."]) == "Martinez-Palomo, A."

    def test_a_hyphen_after_a_digit_is_left_alone(self):
        assert join_line_texts(["pages 123-", "130."]) == "pages 123- 130."

    def test_plain_lines_are_joined_with_one_space(self):
        assert join_line_texts([" Smith, J. ", "", "(2001) Title."]) == "Smith, J. (2001) Title."


class TestFolding:
    def test_dotless_i_with_a_combining_accent_folds_to_i(self):
        assert fold_letters("Calzado-Martı́n") == fold_letters("Calzado-Martín")


class TestCitationsWithBrokenOrAccentedNames:
    REFS = [{"position": 1, "number": None, "first_author": "Calzado-Martín", "year": 2016,
             "printed_text": "Calzado-Martín A, Encinar M. 2016. Effect."},
            {"position": 2, "number": None, "first_author": "Lomvardas", "year": 2006,
             "printed_text": "Lomvardas S, Barnea G. 2006. Interchromosomal."}]

    def run(self, *texts):
        body = [SimpleNamespace(page=1, text=t) for t in texts]
        body.append(SimpleNamespace(page=1, text="Lomvardas S (2006) the list itself"))
        listed = ReferenceList(style="author_year", entries=[], first_line=len(body) - 1)
        return [(m.text, m.numbers) for m in search_author_year_markers(body, listed, self.REFS).markers]

    def test_a_name_with_a_combining_accent_links(self):
        found = self.run("rigidity (Calzado-Martı́n et al., 2016). Using")
        assert found == [("Calzado-Martı́n et al., 2016", [1])]

    def test_a_name_hyphenated_over_a_line_end_links(self):
        found = self.run("interactions (Lomvar-", "das et al., 2006). The")
        assert found == [("Lomvardas et al., 2006", [2])]


class TestHeadingThatOpensNoList:
    def entries(self, page):
        rows, y = [], 100
        for n in range(6):
            rows.append(line(f"Smith{n}, J., & Jones, K. ({2000 + n}). A long title that",
                             page=page, baseline=y))
            rows.append(line("continues here. Journal, 12, 1-9.", page=page, x0=52, baseline=y + 10))
            y += 20
        return rows

    def test_a_cover_sheet_heading_after_the_list_does_not_hide_it(self):
        rows = ([line("Body text", page=1), line("References", page=3, baseline=90)]
                + self.entries(3)
                + [line("References", page=5, baseline=90),
                   line("http://genome.cshlp.org/content/11/9/1484.full.html", page=5, baseline=100),
                   line("This article cites 53 articles", page=5, baseline=110)])
        found = find_reference_list(rows)
        assert found is not None and found.style == "author_year"
        assert [e.first_author for e in found.entries] == [f"Smith{n}" for n in range(6)]
        # The list ends where the later heading begins.
        assert found.end_line == len(rows) - 3

    def test_the_last_heading_still_wins_when_it_opens_a_list(self):
        rows = ([line("References", page=2, baseline=90)]
                + [line(f"Old{n}, A. ({1990 + n}). Title.", page=2, baseline=100 + 12 * n) for n in range(2)]
                + [line("References", page=3, baseline=90)] + self.entries(3))
        found = find_reference_list(rows)
        assert [e.first_author for e in found.entries] == [f"Smith{n}" for n in range(6)]
