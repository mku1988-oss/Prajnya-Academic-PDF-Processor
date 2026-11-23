"""
Equations inside tables: a second read that keeps their sub- and superscripts.

For biology researchers: the table reader returns a table as HTML, and for
most tables that is all we need. But a table whose cells ARE equations (a
methods table listing relaxation-rate formulas, say) can come back flat:
"1/T1=Rc+hb/h(Rbp-Rc)", with every subscript lost, so R_c reads as "Rc" and a
reader cannot tell a symbol from a word. The same crop read with the model's
formula prompt returns proper LaTeX ("1/T_{1}=R_{c}+h_{b}/h(R_{bp}-R_{c})"),
but as one big LaTeX array that wraps long equations over two rows and may
drop words from the header. So neither read is right on its own.

This module joins them: the table read keeps the table's shape and every
non-equation cell, and only the flat equation cells are swapped for their
LaTeX version, matched by CONTENT (the same letters, digits and operators),
never by row and column position, which the wrapped rows break.

The second read costs one more model call (~4-6 s) and is made only for a
table with at least MIN_FLAT_CELLS flat equation cells, which is rare (one
of the corpus's 31 tables). A cell that finds no confident match is left as
the table read gave it: the second read can improve a table, never lose one.

Entry points:
    flat_equation_cells(grid)            -> how many cells look like flat equations
    with_formula_cells(grid, latex_read) -> the grid with those cells in LaTeX
"""
from __future__ import annotations

import html
import re
from difflib import SequenceMatcher
from typing import List, Optional

# A table needs at least this many flat equation cells before the second read
# is worth its cost. One stray "x = y" cell in a results table is not an
# equation table; NMR p4_tab1 has four (one per hydration zone).
MIN_FLAT_CELLS = 2

# How closely a LaTeX cell's skeleton must match a flat cell's skeleton to
# replace it (0..1, difflib ratio). The two reads of one equation agree on
# nearly every character once markup is stripped; a lower match means they
# are probably different cells.
MIN_MATCH = 0.85

# The longest run of letters an equation cell may hold: symbol names are short
# (R_bp, T_1, Rpr), words are not. Measured on 133 stored tables (corpus,
# benchmark run f41, s1 confirmation run): 4 keeps every NMR equation and
# rejects every cell of prose, web address or regression model.
MAX_SYMBOL_LETTERS = 4

_LETTER_RUN = re.compile(r"[A-Za-zα-ωΑ-Ω]+")
_WEB_ADDRESS = re.compile(r"://|www\.|\.(?:org|com|gov|net|cgi)\b|[?@]")
_CELL = re.compile(r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]>)", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_RELATION = re.compile(r"[=<>≥≤≈∝]")
_LETTER = re.compile(r"[A-Za-zα-ωΑ-Ω]")
_ARRAY = re.compile(r"\\begin\{array\}\s*\{[^}]*\}(.*?)\\end\{array\}", re.DOTALL)
_CONTROL_WORD = re.compile(r"\\[A-Za-z]+")
_SYMBOLS = {r"\geq": "≥", r"\leq": "≤", r"\ge": "≥", r"\le": "≤", r"\times": "×",
            r"\approx": "≈", r"\propto": "∝", r"\pm": "±", r"\cdot": "·"}


def flat_equation_cells(grid_html: Optional[str]) -> int:
    """
    How many cells of this table look like an equation written as plain text.

    A flat equation cell has no LaTeX in it, has a relation sign (=, <, >, ≥,
    ≤), and has a letter on BOTH sides of it: "1/T1=Rc+hb/h(Rbp-Rc)" and
    "h>24hb" count; "n = 12" and "P < 0.05" (a number on one side) do not.
    """
    if not grid_html:
        return 0
    return sum(1 for cell in _cell_texts(grid_html) if _is_flat_equation(cell))


def with_formula_cells(grid_html: str, latex_read: str) -> str:
    """
    The table read with each flat equation cell replaced by its LaTeX version.

    `latex_read` is the same crop read with the formula prompt (one LaTeX
    array). Returns `grid_html` unchanged when that read holds no array, or
    when no flat cell finds a confident match in it.
    """
    candidates = latex_cells(latex_read)
    if not candidates:
        return grid_html

    def replace(match: "re.Match[str]") -> str:
        opening, inner, closing = match.groups()
        text = _plain(inner)
        if not _is_flat_equation(text):
            return match.group(0)
        latex = _best_match(text, candidates)
        if latex is None:
            return match.group(0)
        return f"{opening}{html.escape('$' + latex + '$', quote=False)}{closing}"

    return _CELL.sub(replace, grid_html)


def latex_cells(latex_read: Optional[str]) -> List[str]:
    """
    Every cell of the first LaTeX array in a formula read, tidied, with a
    wrapped equation's second line joined back onto its first.

    The formula reader wraps a long equation by starting a new array row
    whose leading cells are empty ("& & (5/6 R_{im} + ...)"). Such a row is a
    continuation: its text belongs at the end of the row above.
    """
    found = _ARRAY.search(latex_read or "")
    if not found:
        return []
    rows: List[List[str]] = []
    for raw_row in re.split(r"\\\\", found.group(1)):
        cells = [tidy_latex(cell) for cell in raw_row.split("&")]
        if not any(cells):
            continue
        if rows and not cells[0] and len(cells) == len(rows[-1]):
            rows[-1] = [tidy_latex(above + " " + below)
                        for above, below in zip(rows[-1], cells)]
        else:
            rows.append(cells)
    return [cell for row in rows for cell in row if cell]


def tidy_latex(latex: str) -> str:
    """
    LaTeX without the spaces the reader puts between characters.

    The formula reader writes "2 4 h _ {\\mathrm {b}}" for 24h_b. In math
    mode TeX ignores those spaces, so dropping them changes nothing on
    screen and makes the stored text searchable. The one space kept is after
    a command word followed by a letter ("\\geq h"), where it is needed.
    """
    keep = "\x00"  # marks the spaces that must stay, while all others go
    marked = re.sub(r"(\\[A-Za-z]+)\s+(?=[A-Za-z])", r"\1" + keep, latex.strip())
    return re.sub(r"\s+", "", marked).replace(keep, " ")


def skeleton(text: str) -> str:
    """
    The part of a cell both reads agree on: letters, digits, relation and
    arithmetic signs, casefolded, with LaTeX markup and spaces removed.
    "1/T_{1}=R_{\\mathrm{c}}" and "1/T1=Rc" have the same skeleton.
    """
    for command, symbol in _SYMBOLS.items():
        text = re.sub(re.escape(command) + r"(?![A-Za-z])", symbol, text)
    text = _CONTROL_WORD.sub("", text)
    return "".join(ch for ch in text.casefold() if ch.isalnum() or ch in "=<>≥≤≈∝+-/×±·")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cell_texts(grid_html: str) -> List[str]:
    """The plain text of every cell, in reading order."""
    return [_plain(match.group(2)) for match in _CELL.finditer(grid_html)]


def _plain(cell_html: str) -> str:
    """A cell's text: tags removed, HTML entities (&gt;) decoded."""
    return html.unescape(_TAG.sub("", cell_html)).strip()


def _is_flat_equation(text: str) -> bool:
    """
    A relation with a letter on each side, written in symbols, no LaTeX yet.

    "Written in symbols" means no run of letters longer than
    MAX_SYMBOL_LETTERS: equations name things R_bp, T_1, h_b, while a cell
    with a word in it ("A score≥7 indicates...", "Model 1: MD=IMCL+Race") is
    prose or a regression formula, which the formula prompt would only
    mangle. Web addresses ("...acc.cgi?acc=GSE36552") are never equations.
    """
    if not text or "$" in text or "\\" in text or _WEB_ADDRESS.search(text):
        return False
    if any(len(run) > MAX_SYMBOL_LETTERS for run in _LETTER_RUN.findall(text)):
        return False
    parts = _RELATION.split(text)
    if len(parts) < 2:
        return False
    return bool(_LETTER.search(parts[0])) and bool(_LETTER.search(parts[-1]))


def _best_match(flat_text: str, candidates: List[str]) -> Optional[str]:
    """The LaTeX cell whose skeleton best matches this flat cell, if close enough."""
    target = skeleton(flat_text)
    if not target:
        return None
    best, best_score = None, 0.0
    for latex in candidates:
        score = SequenceMatcher(None, target, skeleton(latex)).ratio()
        if score > best_score:
            best, best_score = latex, score
    return best if best_score >= MIN_MATCH else None
