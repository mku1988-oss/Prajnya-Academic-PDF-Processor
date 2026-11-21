"""
Printed table titles ("Table 2 Intrinsic preferences ...") read from the
text layer.

For biology researchers: the layout detector boxes a table and, inside that
box, whatever it believes is the caption. Often that is not the title: in the
Biochemical Journal the "Table N" title sits ABOVE the table and the detector
picks the footnote under it ("Adapted from [44] ..."), and on a scanned 1960
page it finds no caption at all, although "TABLE I." is printed at the top.
A table without its label cannot be found by "Table 2", and a table mention
in the text cannot be linked to it.

This finds the "Table N" line that belongs to a table box: just above the box
(where most journals print a table's title) or at the top inside it (where a
detector folded the title into the table), and reads the title from it.

It is only consulted when the detector's caption carries no label, so a table
that already has one is never touched.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

Line = Tuple[str, float, float, float, float]       # text, x0, y0, x1, y1 (top-down)
Rect = Tuple[float, float, float, float]

#: A table label at the START of a line: "Table 2", "TABLE II.", "Tab. 3",
#: "Table 11.1", "Table S1" is excluded on purpose (a supplementary table is
#: never printed in the article).
_TABLE_LABEL = re.compile(
    r"^\s*(?:TABLE|Table|Tab\.)\s*(?P<number>[0-9]{1,3}(?:\.[0-9]{1,3})?|[IVXL]{1,6})"
    r"(?![0-9A-Za-z])(?!\s+[A-Za-z][.:]?(?:\s|$))")
# The last look-ahead refuses a number followed by a stray single letter:
# the 1960 scan's OCR reads "TABLE IV." as "TABLE I N.", which would
# otherwise be labelled Table I, a second Table I.

#: How far above a table box its title may end (PDF points). Measured on the
#: corpus: 7 to 16 pt (Collagen structure p4, p10, p18).
TITLE_ABOVE_MAX_GAP = 40.0

#: How deep inside the box a folded-in title may start, as a share of the
#: box height. Measured: 0.01 (Tendon p257) to 0.09 (humidity p4).
TITLE_INSIDE_MAX_SHARE = 0.25

#: Same as the figure caption walk (`figure_label_pass`).
TITLE_LINE_GAP = 1.8
TITLE_MARGIN_TOLERANCE = 12.0
MAX_TITLE_LINES = 6


def table_label_of(text: Optional[str]) -> Optional[str]:
    """'Table 2' for a caption that starts with a table label, else None."""
    match = _TABLE_LABEL.match(text or "")
    return f"Table {match.group('number')}" if match else None


def find_table_title(lines: Sequence[Line], box: Rect) -> Optional[str]:
    """
    The printed title of the table in `box`, or None.

    Args:
        lines: the page's text lines (`figure_label_pass.build_text_lines`),
            in page coordinates with y growing downwards.
        box: the table's box in the same coordinates.

    Returns:
        The title as one string, starting at its label ("Table 2 Intrinsic
        preferences ... used throughout the text."), or None when no label
        line belongs to this box.
    """
    label = _label_line_for(lines, box)
    if label is None:
        return None
    above = label[4] <= box[1] + 1.0
    collected = [label]
    for line in _lines_after(label, lines):
        previous = collected[-1]
        if above and line[2] >= box[1] - 1.0:
            break                                   # the table itself begins
        if not above and not previous[0].rstrip().endswith("-"):
            break                                   # inside: only a hyphenated wrap
        if line[2] - previous[4] > (line[4] - line[2]) * TITLE_LINE_GAP:
            break
        hyphenated = previous[0].rstrip().endswith("-")
        if not hyphenated and abs(line[1] - label[1]) > TITLE_MARGIN_TOLERANCE:
            break                                   # (a wrapped title may be centred)
        if table_label_of(line[0]):
            break
        collected.append(line)
        if len(collected) >= MAX_TITLE_LINES:
            break
    return _join(collected)


# --- helpers (after the main functions, per CLAUDE.md) ---------------------

def _label_line_for(lines: Sequence[Line], box: Rect) -> Optional[Line]:
    """The label line just above the box or at the top inside it, nearest
    the box's top edge; it must overlap the box horizontally."""
    bx0, by0, bx1, by1 = box
    candidates = []
    for line in lines:
        text, x0, y0, x1, y1 = line
        if not table_label_of(text) or min(x1, bx1) - max(x0, bx0) <= 0:
            continue
        ends_above = 0.0 <= by0 - y1 <= TITLE_ABOVE_MAX_GAP
        # "Inside" includes a line straddling the box's top edge (Tendon
        # p257: "Table 11.1" starts 5 pt above the box and ends inside it).
        starts_inside_top = (by0 - (y1 - y0) <= y0
                             <= by0 + TITLE_INSIDE_MAX_SHARE * (by1 - by0))
        if ends_above or starts_inside_top:
            candidates.append((abs(y0 - by0), line))
    return min(candidates)[1] if candidates else None


def _lines_after(label: Line, lines: Sequence[Line]) -> list:
    """Lines below the label in its own column, nearest first."""
    below = [line for line in lines
             if line[2] > label[2] and min(line[3], label[3]) - max(line[1], label[1]) > 0]
    return sorted(below, key=lambda line: (line[2], line[1]))


def _join(lines: Sequence[Line]) -> str:
    """Join title lines, closing a hyphenated wrap ('solu-' + 'tions')."""
    text = ""
    for line in lines:
        part = line[0].strip()
        if text.endswith("-") and part[:1].islower():
            text = text[:-1] + part
        else:
            text = f"{text} {part}".strip()
    return text
