"""Tests for the equation-table second read (src/processing/utils/table_formulas.py).

The fixtures are real reads of the NMR benchmark's p4_tab1 (zonal relaxation
equations, Table 1 on page 4): the table read GLM-OCR gives it, and the same
crop read with its formula prompt (2026-11-22).
"""
from src.processing.utils.table_formulas import (
    flat_equation_cells, latex_cells, skeleton, tidy_latex, with_formula_cells,
)

TABLE_READ = (
    '<table border="1"><tr><td>Zone</td><td>Hydration constraint</td>'
    '<td>Relaxation rate(s-1)</td></tr>'
    '<tr><td>1</td><td>h&gt;24hb</td><td>1/T1=Ro+(24hb/h)(5/6Rim+1/8Rc+1/24Rbp-Ro)</td></tr>'
    '<tr><td>2</td><td>24hb≥h≥4hb</td><td>1/T1=Rim+(4hb/h)(3/4Rc+1/4Rbp-Rim)</td></tr>'
    '<tr><td>3</td><td>4hb≥h≥hb</td><td>1/T1=Rc+hb/h(Rbp-Rc)</td></tr>'
    '<tr><td>4</td><td>hb≥h</td><td>1/T1=Rpr+h/hb(Rbp-Rpr)</td></tr></table>'
)

FORMULA_READ = r"""$$
\begin{array}{l l l} \mathrm {Z o n e} & \mathrm {H y d r a t i o n} & \mathrm {R e l a x a t i o n r a t e (s ^ {- 1})} \\ 1 & h > 2 4 h _ {\mathrm {b}} & 1 / T _ {1} = R _ {\mathrm {o}} + \left(2 4 h _ {\mathrm {b}} / h\right) \\ & & \left(5 / 6 R _ {\mathrm {i m}} + 1 / 8 R _ {\mathrm {c}} + 1 / 2 4 R _ {\mathrm {b p}} - R _ {\mathrm {o}}\right) \\ 2 & 2 4 h _ {\mathrm {b}} \geq h \geq 4 h _ {\mathrm {b}} & 1 / T _ {1} = R _ {\mathrm {i m}} + \left(4 h _ {\mathrm {b}} / h\right) \\ & & \left(3 / 4 R _ {\mathrm {c}} + 1 / 4 R _ {\mathrm {b p}} - R _ {\mathrm {i m}}\right) \\ 3 & 4 h _ {\mathrm {b}} \geq h \geq h _ {\mathrm {b}} & 1 / T _ {1} = R _ {\mathrm {c}} + h _ {\mathrm {b}} / h \left(R _ {\mathrm {b p}} - R _ {\mathrm {c}}\right) \\ 4 & h _ {\mathrm {b}} \geq h & 1 / T _ {1} = R _ {\mathrm {p r}} + h / h _ {\mathrm {b}} \left(R _ {\mathrm {b p}} - R _ {\mathrm {p r}}\right) \end{array}
$$"""


def test_the_nmr_equation_table_has_eight_flat_equation_cells():
    assert flat_equation_cells(TABLE_READ) == 8


def test_statistics_prose_and_web_addresses_are_not_equations():
    grid = ("<table><tr><td>n = 12</td><td>P &lt; 0.05</td>"
            "<td>A score≥7 indicates systemic involvement</td>"
            "<td>Model1:MD=IMCL+Race+Sex</td>"
            "<td>https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36552</td>"
            "<td>$1/T_{1}=0.347$</td></tr></table>")
    assert flat_equation_cells(grid) == 0


def test_tidy_latex_drops_the_readers_spaces_but_keeps_needed_ones():
    assert tidy_latex(r"2 4 h _ {\mathrm {b}} \geq h") == r"24h_{\mathrm{b}}\geq h"


def test_skeletons_of_the_two_reads_agree():
    assert skeleton("1/T1=Rc+hb/h(Rbp-Rc)") == skeleton(
        r"1/T_{1}=R_{\mathrm{c}}+h_{\mathrm{b}}/h\left(R_{\mathrm{bp}}-R_{\mathrm{c}}\right)")
    assert skeleton("24hb≥h≥4hb") == skeleton(r"24h_{\mathrm{b}}\geq h\geq 4h_{\mathrm{b}}")


def test_a_wrapped_equation_is_joined_back_onto_its_first_line():
    cells = latex_cells(FORMULA_READ)
    zone1 = [cell for cell in cells if cell.startswith("1/T_{1}=R_{\\mathrm{o}}")]
    assert len(zone1) == 1
    assert zone1[0].endswith(r"-R_{\mathrm{o}}\right)")


def test_every_equation_cell_is_replaced_and_every_other_cell_kept():
    merged = with_formula_cells(TABLE_READ, FORMULA_READ)
    assert flat_equation_cells(merged) == 0
    assert "<td>Hydration constraint</td>" in merged  # the formula read dropped "constraint"
    assert "<td>Relaxation rate(s-1)</td>" in merged
    assert r"<td>$1/T_{1}=R_{\mathrm{c}}+h_{\mathrm{b}}/h\left(R_{\mathrm{bp}}-R_{\mathrm{c}}\right)$</td>" in merged
    assert r"<td>$h&gt;24h_{\mathrm{b}}$</td>" in merged
    assert merged.count("<tr>") == TABLE_READ.count("<tr>")


def test_a_formula_read_without_an_array_changes_nothing():
    assert with_formula_cells(TABLE_READ, "1/T_1 = R_c") == TABLE_READ
    assert with_formula_cells(TABLE_READ, "") == TABLE_READ


def test_a_cell_with_no_confident_match_is_left_as_read():
    other = r"\begin{array}{l} x^{2}+y^{2}=z^{2} \end{array}"
    assert with_formula_cells(TABLE_READ, other) == TABLE_READ


# --- the pipeline step (tables._with_formulas) ---------------------------

import logging

from PIL import Image

from src.processing.layout.regions import TableRegion
from src.processing.strategies.research_article.tables import _with_formulas

PAGE = Image.new("RGB", (400, 300), "white")
REGION = TableRegion(box=(10, 10, 390, 290))
LOG = logging.getLogger("test")


def _reader(reply, calls):
    def read(image):
        calls.append(image.size)
        if isinstance(reply, Exception):
            raise reply
        return reply
    return read


def test_an_equation_table_is_read_again_and_merged():
    calls = []
    merged = _with_formulas(TABLE_READ, REGION, PAGE, None,
                            _reader(FORMULA_READ, calls), 4, LOG)
    assert len(calls) == 1
    assert flat_equation_cells(merged) == 0


def test_an_ordinary_table_costs_no_second_read():
    calls = []
    grid = "<table><tr><td>n = 12</td><td>0.347</td></tr></table>"
    assert _with_formulas(grid, REGION, PAGE, None, _reader(FORMULA_READ, calls), 4, LOG) == grid
    assert calls == []


def test_a_failed_second_read_keeps_the_table_read():
    calls = []
    kept = _with_formulas(TABLE_READ, REGION, PAGE, None,
                          _reader(RuntimeError("server gone"), calls), 4, LOG)
    assert kept == TABLE_READ


def test_no_formula_reader_means_no_second_read():
    assert _with_formulas(TABLE_READ, REGION, PAGE, None, None, 4, LOG) == TABLE_READ
