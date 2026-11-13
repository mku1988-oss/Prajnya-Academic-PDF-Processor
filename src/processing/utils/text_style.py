#!/usr/bin/env python3
"""
Typography of a document's running text, for telling a caption from the body.

For biology researchers: journals print figure captions in a different type
from the article's body text: another font (Helvetica captions under Times
body text), or the same font smaller (8 pt captions under 10 pt text). A
reader sees at a glance where a caption stops and the article resumes. The
text layer records the same thing, character by character: each character's
font name, and its height. This module turns those into two questions:

- what does this document's BODY text look like? (`document_body_style`)
- is this line body text, rather than more of the caption? (`PageTypography`)

## Why the caption harvest needs it

`figure_label_pass.harvest_caption_block` reads a caption downward from its
label line until the caption ends. Its geometric stops (a paragraph gap, a
different left margin, the next figure's label) cannot see the most common
ending of all: the caption's last line followed directly, at the same
spacing and margin, by body text. Measured on the 8 corpus articles under
pypdfium2 4.30 (eval/caption_endings/GATE.md): 9 captions ran on into body
prose, one through "Acknowledgments" into the reference list; and 3 more
ended EARLY, only because a stray symbol line happened to stop them.

## The measurement behind the two tests

Across the corpus's six journals, caption and body text always differ in
one of two ways (x-heights in points, median of the letters a, c, e, ...):

    Collagen structure   Helvetica-Condensed 4.33   vs  Times-Roman 4.44
    Hydration (Nature)   DiverdaSans-Light   3.90   vs  MinionPro   4.23
    JRuberti (Matter)    AdvPSA5CD           3.38   vs  AdvPSA5B8   4.15
    CollagenFibril       AdvOT46dcae81       4.52   vs  AdvP800D    4.46
    Silberbush (ACS)     AdvOT2e364b11       3.78   vs  (same font) 4.19
    NMR (Elsevier)       AdvPSA88A           3.73   vs  (same font) 4.66

So: when the caption's font differs from the body font, a line in the BODY
font ends the caption; when they share a font, a line at the body's size,
clearly away from the caption's, ends it. Italics or bold inside a caption
are a third font, never the body's, so they never end it.

The body style is measured per DOCUMENT, not per page: on a page that is
mostly figure, the commonest font IS the caption font (Hydration p5, p7,
p16), and a per-page rule cut three captions short.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from ..pdf.base import GlyphStyle, TextStyle

#: Letters whose height is the x-height (no ascender, no descender). Their
#: median height tracks the font size and ignores capitals and digits.
X_HEIGHT_LETTERS = frozenset("acemnorsuvwxz")

#: A line counts as a different SIZE from the caption when its x-height
#: differs from the caption's by more than this fraction. Caption and body
#: sizes in the corpus differ by 11-25% where they share a font; the same
#: font at one size varies by under 3% between lines.
SIZE_CHANGE = 0.06

#: Sizes within this many points of each other are one size (`_running_text_size`).
SIZE_CLUSTER_PT = 0.12

#: A size must carry at least this share of the body font's characters to be
#: a candidate for the running text's size (`_running_text_size`).
MIN_SIZE_SHARE = 0.20

#: How many pages `document_body_style` reads, spread through the document.
#: Running text dominates any dozen pages of an article; reading every page
#: of a long one would cost time for no change in the answer.
MAX_SAMPLE_PAGES = 12


@dataclass(frozen=True)
class PageTypography:
    """
    One page's characters and its document's body style, for the harvest.

    Attributes:
        glyphs: every character on the page (`PdfPage.glyph_styles`).
        body: the document's running-text style (`document_body_style`).
    """

    glyphs: Sequence[GlyphStyle]
    body: TextStyle

    def style_in(self, rect: Tuple[float, float, float, float]) -> Optional[TextStyle]:
        """The dominant style of the characters inside `rect` (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = rect
        inside = [g for g in self.glyphs
                  if x0 - 1.0 <= g.x <= x1 + 1.0 and y0 - 1.0 <= g.y <= y1 + 1.0]
        return dominant_style(inside)

    def is_body_text(self, line_style: Optional[TextStyle],
                     caption_style: Optional[TextStyle]) -> bool:
        """
        True when a line is the document's running text, not the caption's.

        Two tests, chosen by whether the caption has its own font:

        - caption font differs from the body font: the line is body text
          when it is set in the body font;
        - caption shares the body font: the line is body text when its size
          is clearly not the caption's (`SIZE_CHANGE`) and is the body's.

        Never True when a style is unknown: no evidence, no stop.
        """
        if line_style is None or caption_style is None:
            return False
        if caption_style.font != self.body.font:
            return line_style.font == self.body.font
        return _sized_like_body(line_style, caption_style, self.body)


def document_body_style(pages: Sequence) -> Optional[TextStyle]:
    """
    The style of a document's running text: its commonest font, and the
    size that font's running text is set at (`_running_text_size`).

    Args:
        pages: the document's `PdfPage`s (any sequence; up to
            `MAX_SAMPLE_PAGES` of them are read, spread evenly).

    Returns:
        The body style, or None for a document with no text layer (a scan),
        which turns the typography stop off rather than guessing.
    """
    glyphs: List[GlyphStyle] = []
    for page in _spread(pages, MAX_SAMPLE_PAGES):
        try:
            glyphs.extend(page.glyph_styles())
        except Exception:                                   # pragma: no cover
            continue
    if not glyphs:
        return None
    font = Counter(g.font for g in glyphs).most_common(1)[0][0]
    heights = [g.x_height for g in glyphs if g.font == font and g.x_height]
    return TextStyle(font=font, x_height=_running_text_size(heights))


def dominant_style(glyphs: Iterable[GlyphStyle]) -> Optional[TextStyle]:
    """The commonest font among `glyphs`, with the median x-height of all of them."""
    glyphs = list(glyphs)
    if not glyphs:
        return None
    font = Counter(g.font for g in glyphs).most_common(1)[0][0]
    heights = [g.x_height for g in glyphs if g.x_height]
    return TextStyle(font=font, x_height=statistics.median(heights) if heights else 0.0)


# --- helpers (after the main functions, per CLAUDE.md) -----------------------

def _running_text_size(heights: List[float]) -> float:
    """
    The x-height of the running text: the LARGEST size holding at least
    `MIN_SIZE_SHARE` of the characters.

    Not the median, and not the commonest size. In a journal that sets
    everything in one font, small print can outnumber the body text: the NMR
    paper has 7,758 characters at x-height ~3.7 (captions, tables, references)
    against 7,071 at ~4.6 (the article). Running text is the largest size
    used at length; headings are larger but rare.
    """
    if not heights:
        return 0.0
    clusters: List[List[float]] = []
    for height in sorted(heights):
        if clusters and height - clusters[-1][-1] <= SIZE_CLUSTER_PT:
            clusters[-1].append(height)
        else:
            clusters.append([height])
    large_enough = [c for c in clusters if len(c) >= MIN_SIZE_SHARE * len(heights)]
    chosen = max(large_enough, key=lambda c: c[0]) if large_enough else max(clusters, key=len)
    return statistics.median(chosen)


def _sized_like_body(line: TextStyle, caption: TextStyle, body: TextStyle) -> bool:
    """Same font throughout: is this line at the body's size, not the caption's?"""
    if not (line.x_height and caption.x_height and body.x_height):
        return False
    away_from_caption = abs(line.x_height - caption.x_height) > SIZE_CHANGE * caption.x_height
    near_body = abs(line.x_height - body.x_height) <= (SIZE_CHANGE / 2) * body.x_height
    return away_from_caption and near_body


def _spread(items: Sequence, count: int) -> List:
    """Up to `count` items, evenly spaced through `items` (all of them if fewer)."""
    items = list(items)
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]
