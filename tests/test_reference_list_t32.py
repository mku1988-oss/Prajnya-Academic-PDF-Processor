"""
Tests for ledger T32 (v6.9 item 3, S2): an author start with no space after
the comma (OUP's "Siegel,R."), and years before 1800.

For biology researchers: an OUP reference list prints "Siegel,R., Ma,J." and
must still be read as a list; an entry from 1743 must get its year, while a
page number between 1500 and 1799 must never be taken for one.
"""
from src.processing.utils.layout_lines import LayoutLine
from src.processing.utils.reference_list import entry_year, find_reference_list, first_author


def line(text, *, page=1, x0=40.0, baseline=100.0):
    return LayoutLine(page=page, band=0, column=0, text=text, x0=x0, x1=x0 + 5 * len(text),
                      baseline=baseline, height=10.0, chars=[])


class TestNoSpaceAfterTheComma:
    def test_an_oup_list_is_read(self):
        authors = ["Siegel,R., Ma,J.", "Omenn,G.S.", "Jain,K.K.", "Hanahan,D. and Weinberg,R.A.",
                   "Nickel,W.", "Rothman,J.E. and Orci,L."]
        rows = [line("References", baseline=80)]
        for n, name in enumerate(authors, start=1):
            rows.append(line(f"{n}. {name} ({2000 + n}) A title of the paper. J. Biol., {n}, 1-9.",
                             baseline=80 + 12 * n))
        found = find_reference_list(rows)
        assert found is not None and found.style == "numbered"
        assert [e.first_author for e in found.entries] == ["Siegel", "Omenn", "Jain", "Hanahan", "Nickel", "Rothman"]

    def test_a_word_glued_to_a_comma_is_not_an_author(self):
        assert first_author("Siegel,R., Ma,J.") == "Siegel"
        from src.processing.utils.reference_list import _AUTHOR_START
        assert not _AUTHOR_START.match("Smith,ABC went to the market")


class TestEarlyYears:
    def test_a_vancouver_year_before_1800(self):
        assert entry_year("De la Peyronie F. Sur quelques obstacles. Mém Acad Roy Chir 1743;1:318-333.") == "1743"

    def test_a_year_in_parentheses_before_1800(self):
        assert entry_year("Hooke R (1665) Micrographia. London.") == "1665"

    def test_a_page_range_is_not_a_year(self):
        assert entry_year("Jones A. A book. Publisher; pp. 1650-1700.") is None

    def test_a_later_year_wins(self):
        assert entry_year("Doe J. Title. J Biol 2001;12:1743-1750.") == "2001"
