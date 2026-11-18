#!/usr/bin/env python3
"""
Prepare a table's stored HTML grid for display in Prajnya Studio.

For biology researchers: the table reader (a vision model) returns each table
as a small HTML grid. When a cell holds a formula it writes it as LaTeX between
dollar signs, e.g. ``$R_{\\mathrm{b}}(\\mathrm{s}^{-1})$`` (R with subscript b,
and units s to the power -1). A web page does not understand LaTeX, so it showed
the raw text. This module converts those formulas to MathML, which the browser
draws natively and offline, and tidies the header row.

Nothing here changes the stored JSON: it only affects what the screen shows.
"""

import html
import re
from typing import Optional

import latex2mathml.converter  # MIT licence; installed as a dependency of NiceGUI

# $$...$$ (display) or $...$ (inline). Non-greedy; no newline inside a formula.
_MATH_PATTERN = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$")
_ROW_PATTERN = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
_CELL_PATTERN = re.compile(r"<td(\s[^>]*)?>(.*?)</td>", re.DOTALL)


def prepare_grid_html(grid_html: str) -> str:
    """Return display-ready HTML: formulas as MathML, a proper header row."""
    # Step 1: formulas first, so later steps see finished cell text.
    with_math = convert_latex_to_mathml(grid_html)
    # Step 2: drop a duplicated caption row and mark the header row.
    return mark_header_row(with_math)


def convert_latex_to_mathml(text: str) -> str:
    """Replace every ``$...$`` formula with MathML; leave it as-is if unparsable."""
    return _MATH_PATTERN.sub(_replace_one_formula, text)


def mark_header_row(grid_html: str) -> str:
    """Turn the first real row into <th> cells.

    The reader sometimes puts the table caption in a first row of one wide cell
    (colspan). The caption is already shown above the table, so that row is
    dropped for display and the NEXT row becomes the header.
    """
    rows = _ROW_PATTERN.findall(grid_html)
    if not rows:
        return grid_html

    first_row = rows[0]
    if _is_caption_row(first_row) and len(rows) > 1:
        grid_html = grid_html.replace(f"<tr>{first_row}</tr>", "", 1)
        header_row = rows[1]
    else:
        header_row = first_row

    header_as_th = _cells_to_headers(header_row)
    return grid_html.replace(f"<tr>{header_row}</tr>", f"<tr>{header_as_th}</tr>", 1)


# --- helpers (kept after the main functions, per project rule) -----------

def _replace_one_formula(match: "re.Match[str]") -> str:
    display_block = match.group(1) is not None
    latex = html.unescape(match.group(1) if display_block else match.group(2))
    mathml = _latex_to_mathml(latex.strip(), display_block)
    return mathml if mathml else match.group(0)


def _latex_to_mathml(latex: str, display_block: bool) -> Optional[str]:
    try:
        return latex2mathml.converter.convert(
            latex, display="block" if display_block else "inline"
        )
    except Exception:  # an unparsable formula must never break the page
        return None


def _is_caption_row(row_html: str) -> bool:
    cells = _CELL_PATTERN.findall(row_html)
    return len(cells) == 1 and "colspan" in (cells[0][0] or "")


def _cells_to_headers(row_html: str) -> str:
    return _CELL_PATTERN.sub(
        lambda m: f"<th{m.group(1) or ''}>{m.group(2)}</th>", row_html
    )
