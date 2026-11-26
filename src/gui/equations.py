#!/usr/bin/env python3
"""
List a document's display equations for Prajnya Studio's "Equations" panel.

For biology researchers: figures and tables are separate items in Prajnya's
output, each with a label and a page. Equations are not (yet): the page
readers write them into the page text as LaTeX between double dollar signs,
``$$R_1 = f_o R_o + f_3 R_3$$``. This module finds those equations in the
page text, so Studio can list them in one place, like figures and tables.

It reads the stored page text and changes nothing. Making equations real
output items, with their numbers and links from "Eq. 3" in the text, is a
bigger change planned for v7 (eval/studio_equations/GATE.md).

**What counts as a display equation.** A ``$$...$$`` on a line of its own:
its opening ``$$`` starts the line, and after its closing ``$$`` only
punctuation or its number may follow. Two dollar-formulas that merely touch
(``$x$$^{43}$``) do not start a line; a table row written as text
(``$$Fbln4^{R/R}$$ + Adulthood Aortic aneurysm``) goes on with prose.
Known miss: maths printed INSIDE a figure and read out by the model as
``$$t = t_0$$`` looks exactly like an equation and is listed.

**Equation numbers, only when unambiguous.** Papers print the number beside
the equation, and the readers put it in several places: ``\\tag{3}`` inside
the LaTeX, ``(3)`` after the closing ``$$`` or alone on the next line,
``[3]``, or ``ð3Þ`` (some PDF fonts map the brackets to ð and Þ). Anything
less certain is left unnumbered: a wrong equation number is worse than none.
"""

import re
from typing import Any, Dict, List, Optional

from .page_text import prepare_page_markdown

# $$...$$ whose opening $$ starts a line (only spaces before it on the line).
_DISPLAY_PATTERN = re.compile(r"(?m)^[ \t]*\$\$(?P<latex>.+?)\$\$", re.DOTALL)

# An equation number as printed: (3), [3], ð3Þ; digits plus an optional
# letter (19a). Optionally after the punctuation that ended the equation.
_NUMBER = r"(?:\((?P<a>\d{1,3}[a-z]?)\)|\[(?P<b>\d{1,3}[a-z]?)\]|ð(?P<c>\d{1,3}[a-z]?)Þ)"
_NUMBER_ALONE = re.compile(r"^[ \t]*[.,:;]?[ \t]*" + _NUMBER + r"[ \t]*[.,]?[ \t]*$")

# Most characters allowed after the closing $$ on its line. "ti 1Þð1Þ" (8) is
# debris before an equation number; a table row's "+ Adulthood Aortic
# aneurysm Hanada et al. (2007)" is prose. The shortest such row tail in the
# corpus is far longer than this.
_LONGEST_TAIL = 12

# Inside the LaTeX: \tag{3} / \tag {3}, or a trailing \quad (3) / \qquad (3).
_TAG_INSIDE = re.compile(r"\\tag\s*\{\s*(\d{1,3}[a-z]?)\s*\}")
_QUAD_NUMBER_AT_END = re.compile(r"\\q?quad\s*\(\s*(\d{1,3}[a-z]?)\s*\)\s*[.,]?\s*$")


def list_display_equations(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every display equation in the document, in reading order.

    Returns a list of ``{"page": int, "latex": str, "number": str | None}``.
    `latex` is ready to draw between ``$$``; a ``\\tag`` is removed from it,
    because the number is shown beside the equation instead.
    """
    equations: List[Dict[str, Any]] = []
    for index, page in enumerate(document.get("pages") or []):
        page_number = page.get("page_number", index + 1)
        # The same rewrite the page view uses, so olmOCR's \[ \] count too.
        text = prepare_page_markdown(page.get("content") or "")
        equations.extend(equations_in_text(text, page_number))
    return equations


def equations_in_text(text: str, page_number: int) -> List[Dict[str, Any]]:
    """The display equations of one page's text (see the module docstring)."""
    found = []
    for match in _DISPLAY_PATTERN.finditer(text):
        latex = match.group("latex").strip()
        if not latex or not _ends_its_line(text, match.end()):
            continue
        number = _number_inside(latex) or _number_after(text, match.end())
        found.append({
            "page": page_number,
            "latex": _without_tag(latex),
            "number": number,
        })
    return found


# --- helpers (kept after the main functions, per project rule) -----------

def _number_inside(latex: str) -> Optional[str]:
    """A number the reader wrote into the LaTeX itself."""
    tag = _TAG_INSIDE.search(latex)
    if tag:
        return tag.group(1)
    quad = _QUAD_NUMBER_AT_END.search(latex)
    return quad.group(1) if quad else None


def _ends_its_line(text: str, end: int) -> bool:
    """A display equation ends its line: after the closing $$ only a short
    tail may follow (punctuation, its number, or a scrap of text-layer
    debris such as "ti 1Þð1Þ"). A table row written as text
    ("$$Fbln4^{R/R}$$ + Adulthood Aortic aneurysm") goes on with prose."""
    rest_of_line = text[end:].partition("\n")[0].strip()
    return len(rest_of_line) <= _LONGEST_TAIL


def _number_after(text: str, end: int) -> Optional[str]:
    """A number on the rest of the closing line, or alone on the next line.

    Only a line holding NOTHING but the number counts: "(2) the Berendsen
    chain..." starting the next paragraph is prose, not a label.
    """
    rest_of_line, _, after = text[end:].partition("\n")
    number = _number_alone(rest_of_line)
    if number or rest_of_line.strip():
        return number           # the line had something: a number or prose
    for line in after.split("\n"):
        if line.strip():
            return _number_alone(line)   # the first non-empty line decides
    return None


def _number_alone(line: str) -> Optional[str]:
    match = _NUMBER_ALONE.match(line.rstrip("\r"))
    if not match:
        return None
    return match.group("a") or match.group("b") or match.group("c")


def _without_tag(latex: str) -> str:
    """Remove \\tag{N}: the number is shown beside the equation instead."""
    return _TAG_INSIDE.sub("", latex).strip()
