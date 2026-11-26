#!/usr/bin/env python
"""
Figure and table pages read from the PDF's text layer (T38b, `body_read_mode: text_first`).

For biology researchers: a born-digital page with a figure or a table is
read today by the vision model, ~12 s a page. Its prose is also in the PDF's
own text layer, free and exact. This module decides whether a page can take
that free route and builds the page text for it:

1. **Guards** (`route_verdict`): the page keeps the vision read when
   - G1 it has no text layer (a scan);
   - G2 it holds 5 or more characters set in a maths font or from a Unicode
     maths block (the vision model writes LaTeX; the text layer cannot);
   - G3 it holds 5 or more broken characters (replacement, private-use or
     control characters: a font with a damaged character map);
   - G4 any sub- or superscript letter is glued to its word ("tRNA" + raised
     "Trp" would become "tRNATrp").
2. **Text** (`filtered_text`): `page.text()` without the characters printed
   inside figure, table and caption boxes (axis labels, cell text, legends)
   and without characters no layout box covers at all (a journal's margin
   sidebar, a figure DOI, a stray page number).
3. **Legends** (`without_legends`): each figure and table legend stored on
   its object is deleted from the page text by content, because a caption
   box sometimes holds only the label ("FIGURE 2") and leaves the legend.
   Then (`without_legend_fragments`, T38d) the rest of a legend whose
   title the box took: a run of 8+ consecutive legend words still in the
   text, when that run is at least a third of its legend.

Figures and tables themselves are cut and read from their crops exactly as
on the vision route; the body pipeline does that. Measured in
`eval/text_first_body/GATE_T38b.md` (10 unseen papers: same quality,
-19% wall time).
"""
from __future__ import annotations

import difflib
import re
import statistics
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..layout.base import (LABEL_IMAGE, LABEL_IMAGE_CAPTION, LABEL_TABLE,
                           LABEL_TABLE_CAPTION, LABEL_TABLE_FOOTNOTE, Detection)
from ..pdf.base import PdfPage, Rect
from .content_drop_check import is_scanned_page

#: The two body read modes. `vlm_full` is every release up to v6.9.
BODY_READ_MODES = ("vlm_full", "text_first")
#: The route verdict that means "read this page from the text layer".
TEXT_LAYER_ROUTE = "text_layer"

#: Detection labels whose printed characters leave the page text.
DROPPED_LABELS = {LABEL_TABLE, LABEL_TABLE_CAPTION, LABEL_TABLE_FOOTNOTE,
                  LABEL_IMAGE, LABEL_IMAGE_CAPTION}

#: G2 / G3 thresholds (fixed in eval/text_first_body/GATE.md).
MAX_MATHS_CHARACTERS = 5
MAX_BROKEN_CHARACTERS = 5
#: Maths fonts. Not "STIX" alone: STIX-Regular / STIX-Italic are a TEXT family
#: PNAS and Science Advances set prose in; only STIXMath is a maths font.
MATHS_FONT = re.compile(r"Math|CMMI|CMSY|CMEX|(?<![A-Za-z])Symbol|MT ?Extra|STIX-?Math"
                        r"|MathematicalPi|Cambria ?Math", re.IGNORECASE)
MATHS_BLOCKS = ((0x2200, 0x22FF), (0x27C0, 0x27EF), (0x2980, 0x2AFF), (0x1D400, 0x1D7FF))

#: G4: a glued script letter is below this share of the previous letter's
#: height, with its baseline off by more than BASELINE_SHIFT of that height.
SCRIPT_HEIGHT = 0.8
BASELINE_SHIFT = 0.15
#: ... and the previous letter is body-sized (this share of the page median).
BODY_SIZED = 0.9

#: A legend's leading label ("Fig. 3.", "TABLE 2 |"), optional when looked for.
LEGEND_LABEL = r"(?:(?:fig(?:ure)?s?|table)\.?\s*\d+[A-Za-z]?\s*[.:|\-–—]?\s*)?"
#: A legend shorter than this many words is not looked for (it would match prose).
MIN_LEGEND_WORDS = 4
#: T38c: the shortest run of legend words deleted as a leftover piece (fixed in
#: eval/text_first_body/GATE_T38c.md, not tuned).
MIN_FRAGMENT_WORDS = 8
#: T38d: ... and the smallest share of its legend such a run must hold. A box
#: that took a legend's title leaves most of the legend (49-96% on the corpus);
#: a prose line the caption reader swallowed is a sliver of it (6%, PMC9674220
#: p8). Fixed in eval/text_first_body/GATE_T38d.md, not tuned.
MIN_FRAGMENT_SHARE = 1 / 3


def route_verdict(page: PdfPage) -> Dict[str, Any]:
    """
    G1-G4 for one page: which route it takes, and the counts behind it.

    Returns a dict with `route` ("text_layer", or "vlm:<guard>" naming the
    guard that kept the vision read) and the counts each guard used.
    """
    characters = page.characters()
    verdict: Dict[str, Any] = {
        "maths_characters": sum(1 for c in characters if is_maths(c)),
        "broken_characters": sum(1 for c in page.text() if is_broken(c)),
        "printed_characters": len(characters),
    }
    if not characters or is_scanned_page(page):
        verdict["route"] = "vlm:G1_no_text_layer"
    elif verdict["maths_characters"] >= MAX_MATHS_CHARACTERS:
        verdict["route"] = "vlm:G2_maths"
    elif verdict["broken_characters"] >= MAX_BROKEN_CHARACTERS:
        verdict["route"] = "vlm:G3_broken_characters"
    elif glued_script_letters(characters) > 0:
        verdict["route"] = "vlm:G4_scripts"
    else:
        verdict["route"] = TEXT_LAYER_ROUTE
    return verdict


def filtered_text(page: PdfPage, dropped: Sequence[Rect],
                  covered: Sequence[Rect]) -> Tuple[str, int]:
    """
    `page.text()` without the characters inside `dropped` boxes, and without
    those outside every `covered` box.

    The text and `characters()` list the same printed characters in the same
    order (whitespace and unprintables aside; the line-end hyphen U+0002 is
    "-" in `characters()`), so position k of one is position k of the other.
    A character that cannot be paired is kept. Lines the removal emptied are
    removed. Returns (text, number of characters dropped).
    """
    text, characters = page.text(), page.characters()
    kept: List[str] = []
    k = dropped_count = 0
    for character in text:
        if character.isspace() or (not character.isprintable() and character != "\x02"):
            kept.append(character)
            continue
        wanted = "-" if character == "\x02" else character
        if k < len(characters) and characters[k].text == wanted:
            rect = characters[k].rect
            k += 1
            if centre_inside(rect, dropped) or not centre_inside(rect, covered):
                dropped_count += 1
                continue
        kept.append(character)
    return tidy("".join(kept), text), dropped_count


def without_legends(text: str, objects: Sequence[Dict[str, Any]]) -> Tuple[str, int]:
    """
    Delete each stored legend from the page text, by content.

    A legend is looked for as one run of its words (its leading "Figure 3."
    optional, any whitespace between words), longest legend first. A legend
    that cannot be found is left alone. Returns (text, legends removed).
    """
    removed = 0
    captions = sorted({(o.get("caption") or "").strip() for o in objects},
                      key=len, reverse=True)
    for caption in captions:
        words = re.sub("^" + LEGEND_LABEL, "", caption, flags=re.IGNORECASE).split()
        if len(words) < MIN_LEGEND_WORDS:
            continue
        pattern = LEGEND_LABEL + r"\s*".join(re.escape(w) for w in words)
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            text = text[:found.start()] + text[found.end():]
            removed += 1
    return tidy(text, text), removed


def without_legend_fragments(text: str, objects: Sequence[Dict[str, Any]]
                             ) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Delete the PIECES of stored legends left in the page text (T38d, F1d).

    A caption box sometimes covers only part of a legend: its first
    sentence leaves with the box, the panel descriptions below it stay
    ("(A) Fibre diffraction photograph ..."). `without_legends` cannot find
    such a legend whole. Here a run of MIN_FRAGMENT_WORDS or more
    consecutive words found both in the page text and in one of the page's
    legends is deleted when it holds at least MIN_FRAGMENT_SHARE of that
    legend's words. Words are compared lower-cased with only letters and
    digits kept, so a line-end hyphen or punctuation does not break a run.

    Why the share (T38c failed without it): a stored legend can end with a
    line of body prose the caption reader swallowed; that line is a short
    run of the legend, and it must stay in the body.

    Returns (text, one record per deleted run: its word count, its share of
    the legend and its opening).
    """
    removed: List[Dict[str, Any]] = []
    for caption in {(o.get("caption") or "").strip() for o in objects}:
        legend = [w for w in (_comparable(t) for t in caption.split()) if w]
        if len(legend) < MIN_FRAGMENT_WORDS:
            continue
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        page_words = [_comparable(text[a:b]) for a, b in spans]
        matcher = difflib.SequenceMatcher(None, page_words, legend, autojunk=False)
        cuts = [(spans[block.a][0], spans[block.a + block.size - 1][1], block.size / len(legend))
                for block in matcher.get_matching_blocks()
                if block.size >= MIN_FRAGMENT_WORDS
                and block.size >= MIN_FRAGMENT_SHARE * len(legend)]
        for start, end, share in sorted(cuts, reverse=True):
            removed.append({"words": len(text[start:end].split()), "share": round(share, 2),
                            "opening": " ".join(text[start:end].split())[:80]})
            text = text[:start] + text[end:]
    return tidy(text, text), removed


def boxes_in_points(detections: Optional[Sequence[Detection]], page: PdfPage,
                    image_width_px: int, padding_px: int,
                    labels: Optional[set] = None) -> List[Rect]:
    """
    Detection boxes (pixels of the extraction render, drawn with `padding_px`
    of white around the page) converted to PDF points. With `labels`, only
    the detections carrying one of them.
    """
    width_pt, _ = page.size
    scale = (image_width_px - 2 * padding_px) / width_pt
    return [Rect(*((v - padding_px) / scale for v in d.box_px))
            for d in (detections or []) if labels is None or d.label in labels]


# ---------------------------------------------------------------------------
# Helpers (after the main functions, per the project rule)
# ---------------------------------------------------------------------------

def is_maths(character) -> bool:
    """A character set in a maths font, or from a Unicode maths block (G2)."""
    if MATHS_FONT.search(character.font or ""):
        return True
    code = ord(character.text[0]) if character.text else 0
    return any(low <= code <= high for low, high in MATHS_BLOCKS)


def is_broken(character: str) -> bool:
    """A replacement, private-use or control character (G3). The line-end
    hyphen U+0002 is not broken: text normalisation turns it back into '-'."""
    if character in "\x02\n\r\t":
        return False
    code = ord(character)
    return (character == "�" or 0xE000 <= code <= 0xF8FF
            or unicodedata.category(character) == "Cc")


def glued_script_letters(characters) -> int:
    """G4: letters set smaller and off the baseline, glued (no space) to a
    body-sized letter before them: "tRNA" + raised "Trp", "BhaB" + "Trp"."""
    heights = [c.rect.y1 - c.rect.y0 for c in characters if c.text.isalpha()]
    body = statistics.median(heights) if heights else 0.0
    count = 0
    for previous, character in zip(characters, characters[1:]):
        height = character.rect.y1 - character.rect.y0
        previous_height = previous.rect.y1 - previous.rect.y0
        if (character.text.isalpha() and not character.space_before
                and height < SCRIPT_HEIGHT * previous_height
                and previous_height >= BODY_SIZED * body
                and abs(character.baseline - previous.baseline)
                > BASELINE_SHIFT * previous_height):
            count += 1
    return count


def centre_inside(rect: Rect, boxes: Sequence[Rect]) -> bool:
    x, y = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
    return any(b.x0 <= x <= b.x1 and b.y0 <= y <= b.y1 for b in boxes)


def tidy(filtered: str, original: str) -> str:
    """Drop lines the removal emptied and the doubled spaces it left; lines
    that were blank in the original stay (they are paragraph breaks)."""
    lines_out = []
    for new, old in zip(filtered.split("\n"), original.split("\n")):
        if new.strip() or not old.strip():
            lines_out.append(re.sub(r"(?<=\S) {2,}(?=\S)", " ", new).rstrip())
    return "\n".join(lines_out)


def _comparable(word: str) -> str:
    """A word as F1c compares it: lower case, letters and digits only."""
    return "".join(ch for ch in word.lower() if ch.isalnum())
