"""Tests for src/gui/table_html.py: formulas and header row in the Results tables."""
from src.gui.table_html import prepare_grid_html

GRID = (
    '<table border="1"><tr><td colspan="2">Caption repeated</td></tr>'
    "<tr><td>Constant</td><td>Value</td></tr>"
    r"<tr><td>$R_{\mathrm{b}}(\mathrm{s}^{-1})$</td><td>0.347</td></tr></table>"
)


def test_latex_becomes_mathml():
    out = prepare_grid_html(GRID)
    assert "<math" in out and "$" not in out and "mathrm" not in out


def test_caption_row_dropped_and_header_marked():
    out = prepare_grid_html(GRID)
    assert "Caption repeated" not in out
    assert "<th>Constant</th><th>Value</th>" in out


def test_plain_header_row_kept_and_marked():
    out = prepare_grid_html("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>")
    assert "<th>A</th><th>B</th>" in out and "<td>1</td>" in out


def test_unparsable_formula_left_untouched():
    out = prepare_grid_html(r"<table><tr><td>$\frac{$</td></tr></table>")
    assert "frac" in out and "<math" not in out
