"""Page furniture labelled in page text (`utils/page_furniture.py`, fix 1).

Every case is a line form seen in the benchmark pilots, the corpus or the
Springer book.
"""
from src.processing.utils.page_furniture import label_page_furniture

PROSE = ["The cells were grown in DMEM with ten per cent serum.",
         "Each sample was read twice by two observers.",
         "Counts were compared with a two-sided test.",
         "The results are shown below in the next section.",
         "No sample was excluded from the analysis.",
         "Means and standard deviations are given.",
         "The protocol was approved by the local committee."]


def page(number, top=(), bottom=(), prose=PROSE, newline="\r\n"):
    """A page whose prose differs from every other page's, as real pages do."""
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    tag = " ".join(words[int(d)] for d in str(number))       # no digits: prose does
    text = [f"{line[:-1]}, {tag} {words[i % 12]}." for i, line in enumerate(prose)]
    return {"page_number": number,
            "content": newline.join(list(top) + text + list(bottom)),
            "metadata": {}}


def labelled_text(pages):
    return {n: [s["text"] for s in spans] for n, spans in label_page_furniture(pages).items()}


class TestArticles:
    def test_frontiers_folio_and_author_line(self):
        pages = [page(n, bottom=[f"Frontiers in Genetics {n:02d} frontiersin.org",
                                 "Liang et al. 10.3389/fgene.2023.1219297"]) for n in range(2, 8)]
        found = labelled_text(pages)
        assert found[2] == ["Frontiers in Genetics 02 frontiersin.org",
                            "Liang et al. 10.3389/fgene.2023.1219297"]
        assert set(found) == set(range(2, 8))

    def test_cell_reports_folio_alternates_sides(self):
        """'1055' ends the line on odd pages, '1056 Cell Reports ...' starts it on even."""
        pages = []
        for n in range(2, 9):
            folio = 1053 + n
            line = (f"Cell Reports 22, 1054-1066, January 23, 2018 {folio}" if n % 2
                    else f"{folio} Cell Reports 22, 1054-1066, January 23, 2018")
            pages.append(page(n, bottom=[line]))
        assert all(len(spans) == 1 for spans in label_page_furniture(pages).values())
        assert len(label_page_furniture(pages)) == 7

    def test_a_line_on_few_pages_is_not_furniture(self):
        """PLOS '(TIF)' after two supporting-information items, eLife 'Appendix 3'."""
        pages = [page(n, bottom=["Journal of Things | 2020 | " + str(n)]) for n in range(1, 11)]
        pages[3] = page(4, bottom=["(TIF)"])
        pages[5] = page(6, bottom=["(TIF)"])
        for n in (7, 8, 9):
            pages[n - 1] = page(n, top=[f"Appendix {n - 4}"],
                                bottom=["Journal of Things | 2020 | " + str(n)])
        found = labelled_text(pages)
        assert all("(TIF)" not in lines for lines in found.values())
        assert all(not line.startswith("Appendix") for lines in found.values() for line in lines)

    def test_figure_labels_that_step_with_the_page_are_not_folios(self):
        """Frontiers 'FIGURE 1' on page 4 and 'FIGURE 6' on page 9."""
        pages = [page(4, top=["FIGURE 1"]), page(9, top=["FIGURE 6"])]
        assert label_page_furniture(pages) == {}

    def test_lines_in_the_middle_are_never_furniture(self):
        """The same line on every page, but deep inside the text."""
        pages = [page(n) for n in range(1, 6)]
        for p in pages:
            lines = p["content"].split("\r\n")
            p["content"] = "\r\n".join(lines[:6] + ["Repeated Line Text"] + lines * 2)
        assert all(span["text"] != "Repeated Line Text"
                   for spans in label_page_furniture(pages).values() for span in spans)

    def test_placeholders_are_not_page_text(self):
        placeholder = "[[FIGURE p4_fig1 | Figure 1. Growth | figures/x/p4_fig1.png]]"
        pages = [page(n, bottom=["<!-- figure crop (layout detection) -->", placeholder])
                 for n in range(1, 6)]
        assert label_page_furniture(pages) == {}


class TestBooks:
    def test_chapter_headers_are_dense_in_their_chapter(self):
        """Each chapter has its own running head; 'Abstract' opens every chapter."""
        pages = []
        for n in range(1, 61):
            chapter = (n - 1) // 20
            top = ["Abstract"] if (n - 1) % 20 == 0 else (
                [f"M. J. Author{chapter} et al."] if n % 2 else [f"{chapter + 2} Chapter {chapter} Title"])
            pages.append(page(n, top=top))
        found = labelled_text(pages)
        assert "M. J. Author0 et al." in found[3]
        assert "2 Chapter 0 Title" in found[4]
        assert all("Abstract" not in lines for lines in found.values())

    def test_bare_page_numbers_shift_between_chapters(self):
        """A book stitched from chapters: printed = page - 9, later page - 7."""
        pages = [page(n, top=[str(n - 9 if n < 30 else n - 7)]) for n in range(12, 50)]
        found = label_page_furniture(pages)
        assert len(found) == len(pages)
        assert found[20][0]["kind"] == "page_number" and found[20][0]["text"] == "11"

    def test_a_stray_number_is_not_a_page_number(self):
        pages = [page(n, top=[str(n - 9)]) for n in range(12, 30)]
        pages[5]["content"] += "\r\n3"            # page 17: a table cell's last line
        assert [span["text"] for span in label_page_furniture(pages)[17]] == ["8"]
