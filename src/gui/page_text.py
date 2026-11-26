#!/usr/bin/env python3
"""
Prepare a page's stored text for display in Prajnya Studio.

For biology researchers: the page readers (vision models) write formulas in
the text in LaTeX, but not all in the same way. GLM-OCR wraps them in dollar
signs, ``$h_2 = 0.264$``; olmOCR uses the other standard LaTeX delimiters,
``\\( h_2 = 0.264 \\)`` inline and ``\\[ ... \\]`` on a line of their own.
Studio's text view (markdown with the 'latex' extra) draws only the dollar
form as real maths. In Markdown a backslash before a bracket means "a plain
bracket", so ``\\( h_2 \\)`` came out as ``( h2 )`` with the underscores read
as italics: the subscript was lost, and the text looked garbled.

This module rewrites the bracket form into the dollar form, so every formula
is drawn the same way, offline, as MathML. It also guards the page against a
formula the maths converter cannot read: that one is shown as the reader's
LaTeX, where before it stopped the whole page from drawing.

Nothing here changes the stored JSON: it only affects what the screen shows.
The same rule as `table_html.py`.
"""

import re
from typing import Dict

import latex2mathml.converter   # MIT; already required (requirements.txt)

# How markdown2's 'latex' extra (the one ui.markdown runs) finds code and
# formulas. Copied from markdown2 2.5's `Latex` class ON PURPOSE: the guard
# must see exactly the formulas that extra will try to convert.
# tests/test_gui_page_text.py checks the guard against the real extra.
_MD_PRE_RE = re.compile(r"<pre>(.*?)</pre>", re.DOTALL)
_MD_TRIPLE_TICK_RE = re.compile(r"```(.*?)```", re.DOTALL)
_MD_SINGLE_TICK_RE = re.compile(r"(?<!`)(`)(.*?)(?<!`)\1(?!`)")
_MD_SINGLE_DOLLAR_RE = re.compile(r"(?<!\$)\$(?!\$)(.*?)\$")
_MD_DOUBLE_DOLLAR_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)

# \( ... \): inline maths. Non-greedy; it may wrap onto a second line (readers
# break long lines), but never crosses a blank line (a paragraph break means
# the closing \) is missing, and we must not swallow a paragraph).
_INLINE_PATTERN = re.compile(r"(?<!\\)\\\((?P<latex>(?:(?!\n[ \t]*\n).)+?)\\\)", re.DOTALL)

# \[ ... \]: display maths. May span several lines, same paragraph rule.
_DISPLAY_PATTERN = re.compile(r"(?<!\\)\\\[(?P<latex>(?:(?!\n[ \t]*\n).)+?)\\\]", re.DOTALL)

# Fenced code blocks (```...```) and inline code (`...`) are shown as written:
# a backslash-bracket inside code is code, not a formula.
_CODE_PATTERN = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


def prepare_page_markdown(text: str) -> str:
    """Return page text ready for Studio's markdown view.

    1. every \\( \\) and \\[ \\] formula is put in dollar form;
    2. any formula the maths converter cannot read is shown as code text
       (see `guard_unparsable_formulas`), so one bad formula can no longer
       stop the whole page from drawing.

    A page needing neither comes back as the very same string, so it
    displays exactly as before.
    """
    # Step 1: a cheap check first; most pages have no bracket formulas at all.
    if "\\(" in text or "\\[" in text:
        text = _rewrite_brackets_outside_code(text)
    # Step 2: only formulas that would make the markdown converter raise.
    return guard_unparsable_formulas(text)


def guard_unparsable_formulas(text: str) -> str:
    """Wrap every formula latex2mathml cannot convert in backticks.

    Studio draws maths with markdown2's 'latex' extra, which RAISES on the
    first formula it cannot convert (a "double subscript" such as ``a_b_c``
    is enough), and the whole page then fails to draw: 11 of the 4,955
    stored pages on this machine did (session 48). This finds the formulas
    exactly as that extra does (code spans first, then ``$..$``, then
    ``$$..$$``), tries each, and puts a failing one in backticks: the extra
    skips code, so the reader's LaTeX is shown as written, and every other
    formula on the page is still drawn.
    """
    if "$" not in text:
        return text
    kept: Dict[str, str] = {}
    failed = [False]

    def keep(fragment: str) -> str:
        # A placeholder with no $ or backtick in it, restored at the end.
        key = f"\x00{len(kept)}\x00"
        kept[key] = fragment
        return key

    def check(match: "re.Match[str]", display: bool) -> str:
        if _converts(match.group(1), display):
            return keep(match.group(0))
        failed[0] = True
        return keep(f"`{match.group(0)}`")

    working = _MD_PRE_RE.sub(lambda m: keep(m.group(0)), text)
    working = _MD_SINGLE_TICK_RE.sub(lambda m: keep(m.group(0)), working)
    working = _MD_TRIPLE_TICK_RE.sub(lambda m: keep(m.group(0)), working)
    working = _MD_SINGLE_DOLLAR_RE.sub(lambda m: check(m, display=False), working)
    working = _MD_DOUBLE_DOLLAR_RE.sub(lambda m: check(m, display=True), working)
    if not failed[0]:
        return text                     # nothing to guard: the same string
    for key in reversed(list(kept)):    # later keys may hold earlier ones
        working = working.replace(key, kept[key])
    return working


def convert_bracket_delimiters(text: str) -> str:
    """Rewrite ``\\[x\\]`` as ``$$x$$`` and ``\\(x\\)`` as ``$x$``."""
    # Display first: its content may itself be long, and doing it first means
    # the inline pattern never sees a display formula's insides.
    text = _DISPLAY_PATTERN.sub(_to_display_dollars, text)
    return _INLINE_PATTERN.sub(_to_inline_dollars, text)


# --- helpers (kept after the main functions, per project rule) -----------

def _rewrite_brackets_outside_code(text: str) -> str:
    """Splitting on the code spans keeps them byte-for-byte; the pieces
    between them are converted."""
    pieces = []
    position = 0
    for code in _CODE_PATTERN.finditer(text):
        pieces.append(convert_bracket_delimiters(text[position:code.start()]))
        pieces.append(code.group(0))
        position = code.end()
    pieces.append(convert_bracket_delimiters(text[position:]))
    return "".join(pieces)


def _converts(latex: str, display: bool) -> bool:
    """Would markdown2's 'latex' extra convert this formula without raising?
    (It removes literal backslash-n from display formulas first; so do we.)"""
    try:
        if display:
            latex2mathml.converter.convert(latex.replace(r"\n", ""), display="block")
        else:
            latex2mathml.converter.convert(latex)
        return True
    except Exception:
        return False


def _to_inline_dollars(match: "re.Match[str]") -> str:
    latex = match.group("latex").strip()
    # A dollar inside would end the formula early in the dollar form, and an
    # empty formula draws nothing: leave either exactly as the reader wrote it.
    if not latex or "$" in latex:
        return match.group(0)
    # One line: the dollar form must not break across a line, or the
    # markdown reader sees two half-formulas.
    return "$" + " ".join(latex.split()) + "$"


def _to_display_dollars(match: "re.Match[str]") -> str:
    latex = match.group("latex").strip()
    if not latex or "$" in latex:
        return match.group(0)
    return "$$" + latex + "$$"
