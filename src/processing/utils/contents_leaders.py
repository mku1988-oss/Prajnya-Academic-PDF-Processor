#!/usr/bin/env python3
"""
Rebuild a contents page's text from its geometry, so its dot leaders survive.

For biology researchers: a book's contents page lists each chapter as a title,
a row of dots, and a page number:

    3 Tendon Extracellular Matrix Assembly,
      Maintenance and Dysregulation Throughout Life . . . . . . .  45

Those dots (the "leader") are how the system recognises a contents page at
all, and how the contents parser splits a title from its page number.

## Why this module exists

**PDFium as bundled with pypdfium2 5.x extracts no characters from a dot
leader.** The same line reads, depending on the call:

    4.30, whole-page text:     "Life . . . . . . . . . . . 45"
    5.x,  whole-page text:     "Life45"
    5.x,  character stream:    "Life  45"   (two en spaces; on some lines one
                                             plain space, indistinguishable
                                             from an ordinary word space)

So under 5.x the text alone can no longer say where a leader was. The page
GEOMETRY still can, under both versions alike: the leader is still a text
object on the page, with a box spanning exactly the gap between title and
number (`PdfPage.leaders()`). This module rebuilds each line from the word
boxes and writes a standard leader (`LEADER_TEXT`) wherever such an object
fills the gap before a trailing page number. The parser and the classifier
then read the same text whichever PDFium produced it.

## Why the leader OBJECT, not just a wide gap

A wide gap before a number is not enough evidence: a table on an article's
first pages ("Collagen I        12") has exactly that shape, and three such
rows would make the classifier call the page a book's contents page. A table's
gap is empty; a leader's gap holds printed dots. Requiring the object keeps
the old rule's meaning ("there are dots here") and only changes how the dots
are seen.

Measured 2026-11-13, pypdfium2 4.30.0 against 5.15.2 (PDFium 8149), on the
Springer book's two contents pages: the same 9 + 8 leader objects at the same
positions under both.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from ..pdf.base import PdfPage, Rect, Word, group_words_into_lines
from .regex_patterns import TOC_PAGE_NUMBER

#: The leader written into rebuilt text. Five spaced dots: the contents
#: parser's `DOT_LEADER_RUN` accepts three or more, so this is unambiguous.
LEADER_TEXT = ". . . . ."

#: A page needs this many leader lines before its text is rebuilt. The same
#: count the classifier asks for before calling a page a contents page
#: (`MIN_DOT_LEADER_LINES`); fewer, and the page keeps its own text untouched.
MIN_LEADER_LINES = 3

#: A whole word that can be a contents page number: 45, xii, C1.
_PAGE_NUMBER_WORD = re.compile(rf'^{TOC_PAGE_NUMBER}$')

#: Never split a line at a horizontal gap: the gap before a page number IS
#: the leader, and splitting there would separate a title from its number.
#: A contents page is a single column, so nothing else is lost.
_NO_COLUMN_SPLIT = float("inf")


def contents_page_text(page: PdfPage) -> Optional[str]:
    """
    The text of a contents page with every dot leader written out, or None.

    Args:
        page: any `PdfPage`.

    Returns:
        One line per typeset line, top to bottom, words joined by single
        spaces, and `LEADER_TEXT` between a title and its page number wherever
        a leader object fills that gap. **None** when the page has fewer than
        `MIN_LEADER_LINES` such lines, which is every page that is not a
        contents page: the caller then keeps the page's own text.

    Example:
        A line "Life" ........ "45" (with a leader object between them)
        comes back as "Life . . . . . 45".
    """
    leader_boxes = page.leaders()
    if len(leader_boxes) < MIN_LEADER_LINES:
        return None

    words = _words_outside_leaders(page.words(), leader_boxes)
    lines = group_words_into_lines(words, column_gap_factor=_NO_COLUMN_SPLIT)

    rendered: List[str] = []
    leader_lines = 0
    for line in lines:
        text, has_leader = _render_line(line, leader_boxes)
        rendered.append(text)
        leader_lines += 1 if has_leader else 0

    if leader_lines < MIN_LEADER_LINES:
        return None
    return "\n".join(rendered)


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def _words_outside_leaders(words: Sequence[Word],
                           leader_boxes: Sequence[Rect]) -> List[Word]:
    """
    Drop the leader's own dots from the word list.

    Under pypdfium2 4.30 (and PyMuPDF) the dots are extracted as words (".");
    under 5.x they are not. Removing them here makes the two identical before
    the line is rebuilt. Only letterless words are dropped, and only inside a
    leader's span, so a full stop ending a real word is never touched.
    """
    kept = []
    for word in words:
        letterless = not any(ch.isalnum() for ch in word.text)
        if letterless and any(_inside_leader(word.rect, box) for box in leader_boxes):
            continue
        kept.append(word)
    return kept


def _render_line(line: Sequence[Word], leader_boxes: Sequence[Rect]) -> tuple:
    """
    One rebuilt line, and whether it is a leader line.

    A leader line ends in a page-number word with a leader object lying on
    the same line, between the previous word (if any) and that number. A line
    holding only the number is allowed: a long title can leave its leader and
    number alone on the last line.
    """
    words = [w.text for w in line]
    last = line[-1]
    if not _PAGE_NUMBER_WORD.match(last.text):
        return " ".join(words), False

    gap_start = line[-2].rect.x1 if len(line) > 1 else None
    if not any(_fills_gap(box, line, gap_start, last.rect.x0) for box in leader_boxes):
        return " ".join(words), False

    title = " ".join(words[:-1])
    parts = [part for part in (title, LEADER_TEXT, last.text) if part]
    return " ".join(parts), True


def _fills_gap(box: Rect, line: Sequence[Word], gap_start: Optional[float],
               gap_end: float) -> bool:
    """
    True when a leader box sits on this line, inside the gap before the number.

    Vertical: the leader's centre lies within the line's band (the dots sit on
    the baseline, and PDFium's box for them is ~1 pt tall). Horizontal: the
    leader ends at or before the number begins, and overlaps the gap. A small
    tolerance absorbs the half point PDFium leaves between a box and its
    neighbour's.
    """
    tolerance = 1.0
    band_top = min(w.rect.y0 for w in line)
    band_bottom = max(w.rect.y1 for w in line)
    centre_y = (box.y0 + box.y1) / 2
    if not band_top <= centre_y <= band_bottom:
        return False
    if box.x1 > gap_end + tolerance:
        return False
    return gap_start is None or box.x1 > gap_start - tolerance


def _inside_leader(rect: Rect, box: Rect) -> bool:
    """True when a word's centre lies within a leader's horizontal span and
    near its line (a word box is taller than a leader's ~1 pt box)."""
    centre_x = (rect.x0 + rect.x1) / 2
    on_line = rect.y0 <= (box.y0 + box.y1) / 2 <= rect.y1
    return box.x0 - 1.0 <= centre_x <= box.x1 + 1.0 and on_line
