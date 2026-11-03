#!/usr/bin/env python3
"""
Give every figure crop the number the paper calls it by — including the crops
that are panels of one printed figure.

For biology researchers: the layout detector finds *panels*, not figures. A
figure printed as "Fig. 11" with three stacked panels comes back as three
separate crops, and only the panel that happens to sit next to the caption
gets the caption. The other two arrive with no caption, no label, and nothing
tying them to the figure they belong to — so a RAG asked for Figure 11 finds
one third of it.

This module fixes that with the standard remedy from the scientific-document
literature (see `docs/layout_detectors_and_parsers.md`): **assume one caption
per printed figure**, find the caption lines in the text layer, and let every
crop between two caption lines belong to the same figure.

WHY THE TEXT LAYER AND NOT THE DETECTOR. Measured over the 57 figures whose
captions failed: 23% got no caption box at all, 49% got a box containing a
panel sub-label ("d", "f", "( A ) Top view…") rather than the caption. No
permissive detector distinguishes those two things. But the caption itself is
sitting in the text layer of the same page, in full, even where the detector
saw nothing.

THE TRAP THIS MUST NOT FALL INTO. A figure is routinely CITED pages away from
where it is PRINTED — the tendon volume's page 20 reads "…packing is affected
(Fig. 2.2). The rigid COL2 domain…" in running prose. Adopting that citation
as a label is confidently wrong, and nothing downstream could tell. Three
constraints separate a caption from a citation, and all three are applied:

1. **Anchoring** — a caption BEGINS its line; a mid-sentence mention does not.
2. **Adjacency** — a caption sits against a figure, within `MAX_GAP_FRACTION`
   of the page height. A citation buried in a column of prose does not.
3. **Uniqueness** — a printed figure has exactly ONE caption in a document but
   may be cited a dozen times, so a label claimed by several pages is resolved
   in favour of the closest claim and dropped everywhere else.

The design rule inherited from `chapter_boundaries` applies here too:
**an unplaceable figure is left unlabelled rather than guessed at.** A missing
label is recoverable; a wrong one silently attaches the wrong image to a
citation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.processing.utils.figure_labels import parse_printed_label

#: How far from a figure's edge a caption line may sit, as a fraction of page
#: height. A caption is directly against its figure; anything further away is
#: another figure's caption or a citation in body text. 12% of a page is
#: roughly three lines of body text plus the usual figure margin.
MAX_GAP_FRACTION = 0.12

#: A caption line must share this much horizontal span with its figure, as a
#: fraction of the narrower of the two. A single-column caption under a
#: single-column figure overlaps almost completely; a caption in the OTHER
#: column of a two-column paper overlaps not at all.
MIN_HORIZONTAL_OVERLAP = 0.25

#: How far a caption may reach INSIDE its figure's box, as a fraction of page
#: height. Measured, and not optional: the figure crop is snapped outwards to
#: whitespace, which routinely swallows the first line or two of the caption.
#: On the Springer chapter's page 3 the caption begins at y=531.3 while the
#: crop ends at y=533.0 — a gap of −1.7pt. A `gap >= 0` test rejects it, and
#: that single sign error lost 35 of the 50 captions we already trusted.
CAPTION_OVERLAP_TOLERANCE = 0.04

#: A typographic ORNAMENT printed before a caption's label, which must be
#: stripped before the label can be recognised. Nature-family journals set a
#: "◀" before a caption that belongs to the figure on the FACING page --
#: so the very mark meaning "the figure is overleaf" was what stopped the
#: overleaf figure being found.
#:
#: The two backends disagree here and that is why this went unnoticed: PyMuPDF
#: drops the ornament, pypdfium2 returns it as a word. Anything relying on one
#: backend's text will not see the problem at all.
#:
#: MEASURED over the 9-document corpus (485 pages), because the obvious wider
#: rule is actively harmful. Stripping any leading punctuation unblocks 21
#: lines, of which only 2 are captions -- the other 19 are parenthesised
#: CITATIONS ("(Figure 7), whereas ...", "(Fig. 3.3 ) (Kadler et al. 1996 )"),
#: exactly what this module exists to keep out. So "(" and "[" are pointedly
#: NOT in this set: they mark a citation, not an ornament. Restricted to
#: geometric shapes, dingbats and bullets, it unblocks 2 lines and both are
#: real captions.
ORNAMENT_PREFIX = re.compile(
    "^["
    "\\s"
    "■-◿"      # geometric shapes, including U+25C0 BLACK LEFT-POINTING TRIANGLE
    "☀-➿"      # miscellaneous symbols and dingbats
    "•‣"       # bullet, triangular bullet
    "]+"
)

#: Words further apart than this (as a fraction of page width) are in
#: DIFFERENT COLUMNS, not the same line. Without this, a two-column paper
#: interleaves both columns into one line at each y, and a caption starting
#: "Fig. 1 …" is prefixed with the neighbouring column's prose — so it no
#: longer starts with a label and vanishes from the scan.
COLUMN_GAP_FRACTION = 0.06


@dataclass(frozen=True)
class LabelLine:
    """
    One line of page text that begins with a figure label.

    Attributes:
        printed: the label as printed, e.g. "Fig. 2.1".
        key: the normalized matching key, e.g. "figure 2.1".
        text: the full line, which is the start of the caption itself.
        rect: (x0, y0, x1, y1) of the line, in the same units as the figures.
        starts_a_block: True when no body line runs directly into this one
            from above at the same left margin. A caption begins a block of
            text; a sentence that happens to open with "Figure 6 and Video S5
            show..." is the continuation of a paragraph. Only used by the
            last-resort pairing in `_adopt_orphan_captions`, which is the one
            place that cannot rely on distance to tell them apart.
    """

    printed: str
    key: str
    text: str
    rect: Tuple[float, float, float, float]
    starts_a_block: bool = True


@dataclass(frozen=True)
class LabelAssignment:
    """
    The outcome for one figure crop.

    Attributes:
        printed: label as printed, or None when nothing could be established.
        key: normalized key, or None.
        caption: the caption as read from the text layer — the whole block
            when `text_lines` was supplied, otherwise its first line only.
        opening: the caption's FIRST line alone, which is the part a
            detector-supplied caption is missing when it begins part way
            down. Kept separate because the two are used for different
            repairs: see `figures._should_replace_caption`.
        shared_with: how many crops on the page carry this same label. Greater
            than 1 means these crops are PANELS OF ONE PRINTED FIGURE.
        distance: gap in page units between the crop and its caption line,
            kept so a document-level pass can resolve competing claims.
    """

    printed: Optional[str]
    key: Optional[str]
    caption: str
    shared_with: int
    distance: float
    opening: str = ""


def build_text_lines(words: Sequence[Tuple[str, float, float, float, float]],
                     page_width: float,
                     ) -> List[Tuple[str, float, float, float, float]]:
    """
    Group words into text lines, keeping columns apart.

    For biology researchers: a research paper is usually printed in two
    columns, so at any given height there are words from BOTH columns at the
    same distance down the page. Joining everything at one height into a
    single line produces nonsense — the left column's prose runs straight into
    the right column's caption — and a caption that no longer *starts* with
    "Fig. 1" becomes invisible to the scan. So words are grouped by height
    and then split wherever a horizontal gap is wide enough to be a column
    boundary.

    Args:
        words: `(text, x0, y0, x1, y1)` per word, in page units.
        page_width: the page's width, which sets the column-gap threshold.

    Returns:
        `(text, x0, y0, x1, y1)` per line, top-to-bottom then left-to-right.
    """
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (round(w[2], 1), w[1]))
    rows: List[List[Tuple[str, float, float, float, float]]] = []
    current: List[Tuple[str, float, float, float, float]] = []
    top: Optional[float] = None

    for word in ordered:
        if top is None or abs(word[2] - top) <= 3.0:
            current.append(word)
            top = word[2] if top is None else top
        else:
            rows.append(current)
            current, top = [word], word[2]
    if current:
        rows.append(current)

    column_gap = page_width * COLUMN_GAP_FRACTION
    lines: List[Tuple[str, float, float, float, float]] = []
    for row in rows:
        # Reading order is left-to-right, and the grouping above cannot supply
        # it: words are sorted by (y, x) globally, so a word sitting a third
        # of a point lower than its neighbours sorts AFTER all of them. A bold
        # "Fig. 1" set slightly off the baseline of its own caption therefore
        # lands at the END of the line — the caption stops beginning with its
        # label, and the whole scan misses it. Measured: this alone accounted
        # for 35 of the 50 trusted captions being lost.
        row = sorted(row, key=lambda w: w[1])
        run = [row[0]]
        for previous, word in zip(row, row[1:]):
            if word[1] - previous[3] > column_gap:
                lines.append(_as_line(run))
                run = [word]
            else:
                run.append(word)
        lines.append(_as_line(run))
    return lines


def find_label_lines(lines: Sequence[Tuple[str, float, float, float, float]]
                     ) -> List[LabelLine]:
    """
    Pick out the text lines that begin with a figure label.

    For biology researchers: given every line of text on a page, this returns
    the ones that look like the start of a figure caption — "Fig. 2.1
    Hierarchical structure of tendon…". A line that merely *mentions* a figure
    part-way through is not returned, because `parse_printed_label` only reads
    the beginning of the text.

    A leading typographic ORNAMENT is stripped first — see `ORNAMENT_PREFIX`.

    Args:
        lines: `(text, x0, y0, x1, y1)` per line, in page units.

    Returns:
        One `LabelLine` per caption-looking line, page order preserved. Its
        text carries no leading ornament, so the caption stored downstream
        begins at the label.
    """
    found: List[LabelLine] = []
    for text, x0, y0, x1, y1 in lines:
        cleaned = ORNAMENT_PREFIX.sub("", text)
        printed, key = parse_printed_label(cleaned)
        if printed and key:
            found.append(LabelLine(
                printed=printed, key=key, text=cleaned,
                rect=(x0, y0, x1, y1),
                starts_a_block=_starts_a_block((x0, y0, x1, y1), lines),
            ))
    return found


#: How far below one caption line the next may begin, as a multiple of the
#: line's own height, and still belong to the same caption. Captions are set
#: solid like body text; the gap to the paragraph AFTER a caption is larger.
CAPTION_LINE_GAP = 1.8

#: How far a continuation line's left edge may sit from the caption's own left
#: margin, in page units. Journals indent nothing inside a caption, but the
#: line reconstruction wobbles by a point or two, and some captions carry a
#: hanging first line.
CAPTION_MARGIN_TOLERANCE = 12.0

#: A caption longer than this many lines is not a caption -- the walk has run
#: into body text. Measured: the longest real caption in the 9-document corpus
#: is 21 lines (the tendon volume's Fig. 3.5), so this leaves generous room
#: while still bounding a runaway.
MAX_CAPTION_LINES = 40


def harvest_caption_block(
    label_line: LabelLine,
    lines: Sequence[Tuple[str, float, float, float, float]],
) -> str:
    """
    Read a whole caption out of the text layer, not just its first line.

    For biology researchers: the layout detector draws a box around what it
    thinks the caption is, and that box very often starts PART WAY DOWN the
    caption. The stored caption then begins mid-sentence -- "respective
    overlaps and gaps in the fibril. (b) ..." -- and the opening sentence that
    says what the figure actually shows, "Fig. 2.2 Fibril-forming collagens
    and their assembly", is missing. That opening is exactly the part a search
    needs, so 26 of this corpus's 107 captions are unsearchable for their own
    subject.

    The page's text layer has the caption whole. This walks down from the
    label line, taking each following line while it still reads as the same
    block, and stops at the first sign that the caption has ended:

    * a vertical gap larger than `CAPTION_LINE_GAP` -- a new paragraph;
    * a left margin outside `CAPTION_MARGIN_TOLERANCE` -- a different column
      or a different element;
    * a line that begins a caption of its OWN -- the next figure;
    * `MAX_CAPTION_LINES`, so a page whose layout defeats all three cannot
      swallow the rest of the article.

    Args:
        label_line: the caption's opening line, from `find_label_lines`.
        lines: every text line on the page, as `build_text_lines` returns them.

    Returns:
        The caption as one space-joined string, beginning at its label. Never
        empty: at worst it is the label line alone, which is what the caller
        had before.
    """
    left = label_line.rect[0]
    collected = [label_line.text]
    previous = label_line.rect

    for text, x0, y0, x1, y1 in _lines_below(label_line, lines):
        if y0 - previous[3] > (y1 - y0) * CAPTION_LINE_GAP:
            break                                   # a new paragraph begins
        if abs(x0 - left) > CAPTION_MARGIN_TOLERANCE:
            break                                   # another column or element
        printed, _ = parse_printed_label(ORNAMENT_PREFIX.sub("", text))
        if printed:
            break                                   # the next figure's caption
        collected.append(text)
        previous = (x0, y0, x1, y1)
        if len(collected) >= MAX_CAPTION_LINES:
            break

    return " ".join(part.strip() for part in collected if part.strip())


def _lines_below(label_line: LabelLine,
                 lines: Sequence[Tuple[str, float, float, float, float]]
                 ) -> List[Tuple[str, float, float, float, float]]:
    """The page's lines that sit below the label line, nearest first."""
    below = [line for line in lines if line[2] > label_line.rect[1]]
    return sorted(below, key=lambda line: (line[2], line[1]))



def assign_labels_on_page(
    figure_boxes: Sequence[Tuple[float, float, float, float]],
    label_lines: Sequence[LabelLine],
    page_height: float,
    logger: Optional[logging.Logger] = None,
    text_lines: Optional[Sequence[Tuple[str, float, float, float, float]]] = None,
) -> List[LabelAssignment]:
    """
    Attach each figure crop on a page to the caption it belongs to.

    For biology researchers: this is the step that rejoins panels. Each crop
    looks for the caption line directly beneath it — that is where a figure
    caption goes — and failing that, directly above (some journals caption
    above the figure). Several crops finding the SAME caption is not a
    conflict: it is the signal that they are panels of one printed figure, and
    they all get that figure's label.

    Args:
        figure_boxes: one `(x0, y0, x1, y1)` per crop, in page units.
        label_lines: caption-looking lines from `find_label_lines`.
        page_height: the page's height, for the adjacency limit.
        logger: optional logger.
        text_lines: every line on the page, from `build_text_lines`. When
            given, each assignment carries the WHOLE caption rather than only
            its first line — see `harvest_caption_block`. Omitting it keeps
            the first-line-only behaviour, which is all the geometry needs.

    Returns:
        One `LabelAssignment` per crop, in the order the crops were given.
        Crops with nothing credible nearby come back with `printed=None` —
        the honest answer, and the one that cannot mislead a reader.
    """
    log = logger or logging.getLogger("FigureLabelPass")
    limit = page_height * MAX_GAP_FRACTION
    inside = page_height * CAPTION_OVERLAP_TOLERANCE

    picks: List[Optional[Tuple[LabelLine, float]]] = []
    for box in figure_boxes:
        picks.append(_nearest_caption(box, label_lines, limit, inside))

    picks = _propagate_to_panels(figure_boxes, picks, label_lines)
    picks = _adopt_orphan_captions(figure_boxes, picks, label_lines)
    shared = _count_by_key(picks)

    assignments: List[LabelAssignment] = []
    for box, pick in zip(figure_boxes, picks):
        if pick is None:
            assignments.append(LabelAssignment(None, None, "", 0, float("inf")))
            continue
        line, gap = pick
        caption = (harvest_caption_block(line, text_lines)
                   if text_lines is not None else line.text)
        assignments.append(LabelAssignment(
            printed=line.printed, key=line.key, caption=caption,
            shared_with=shared[line.key], distance=gap, opening=line.text,
        ))

    for key, count in shared.items():
        if count > 1:
            log.debug("%d crops share %s — panels of one printed figure",
                      count, key)
    return assignments


def resolve_across_document(
    per_page: Dict[int, List[LabelAssignment]],
    logger: Optional[logging.Logger] = None,
) -> Dict[int, List[LabelAssignment]]:
    """
    Drop labels that two pages both claim — a citation beat a caption.

    For biology researchers: a printed figure is captioned once but may be
    mentioned on many pages. If "Figure 3" is claimed on two different pages,
    at most one of those is the real caption, so the closer claim keeps it and
    the other is cleared. Clearing is deliberate: an unlabelled figure can be
    fixed later, a figure labelled with someone else's number cannot be
    noticed at all.

    Args:
        per_page: page number -> the assignments from `assign_labels_on_page`.
        logger: optional logger.

    Returns:
        The same structure with losing claims replaced by empty assignments.
    """
    log = logger or logging.getLogger("FigureLabelPass")

    best: Dict[str, Tuple[int, float]] = {}
    for page, assignments in per_page.items():
        for assignment in assignments:
            if not assignment.key:
                continue
            current = best.get(assignment.key)
            if current is None or assignment.distance < current[1]:
                best[assignment.key] = (page, assignment.distance)

    resolved: Dict[int, List[LabelAssignment]] = {}
    for page, assignments in per_page.items():
        kept: List[LabelAssignment] = []
        for assignment in assignments:
            if assignment.key and best[assignment.key][0] != page:
                log.info("p%d: dropping %s — page %d has a closer claim, so "
                         "this one is a citation, not a caption.",
                         page, assignment.printed, best[assignment.key][0])
                kept.append(LabelAssignment(None, None, "", 0, float("inf")))
            else:
                kept.append(assignment)
        resolved[page] = kept
    return resolved


# ---------------------------------------------------------------------------
# Geometry helpers (after the main functions, per CLAUDE.md)
# ---------------------------------------------------------------------------

def _nearest_caption(box: Tuple[float, float, float, float],
                     label_lines: Sequence[LabelLine],
                     limit: float,
                     inside: float) -> Optional[Tuple[LabelLine, float]]:
    """
    The caption line belonging to one crop: nearest below, else nearest above.

    Below is preferred because that is where a figure caption is printed. The
    `limit` keeps a citation elsewhere on the page out of reach; `inside`
    allows the caption to begin slightly WITHIN the crop, which it usually
    does because the crop was snapped outwards to whitespace.
    """
    below: List[Tuple[float, LabelLine]] = []
    above: List[Tuple[float, LabelLine]] = []

    for line in label_lines:
        if _horizontal_overlap(box, line.rect) < MIN_HORIZONTAL_OVERLAP:
            continue
        gap_below = line.rect[1] - box[3]
        gap_above = box[1] - line.rect[3]
        if -inside <= gap_below <= limit:
            below.append((max(gap_below, 0.0), line))
        elif -inside <= gap_above <= limit:
            above.append((max(gap_above, 0.0), line))

    for bucket in (below, above):
        if bucket:
            bucket.sort(key=lambda pair: pair[0])
            return bucket[0][1], bucket[0][0]
    return None


def _propagate_to_panels(
    figure_boxes: Sequence[Tuple[float, float, float, float]],
    picks: Sequence[Optional[Tuple[LabelLine, float]]],
    label_lines: Sequence[LabelLine],
) -> List[Optional[Tuple[LabelLine, float]]]:
    """
    Give an unlabelled crop the label of the figure it is a panel of.

    For biology researchers: a figure printed as "Fig. 11" with three stacked
    panels arrives as three crops, and only the bottom one — the one touching
    the caption — found that caption. The other two are the *same figure*.
    They cannot find the caption themselves because the panel in between puts
    it beyond the adjacency limit, so instead each unlabelled crop adopts the
    label of the nearest labelled crop BELOW it.

    THE GUARD THAT MAKES THIS SAFE: nothing is adopted across a caption line.
    If another figure's caption sits between the two crops, then the crops are
    on opposite sides of a figure boundary and are different figures. Without
    that check, every figure on a page would collapse into the last one.

    Args:
        figure_boxes: crop boxes, page units, in the caller's order.
        picks: what each crop found for itself; None where it found nothing.
        label_lines: every caption line on the page — the boundaries.

    Returns:
        A new picks list; crops that adopted a neighbour's label now carry it.
    """
    resolved = list(picks)
    order = sorted(range(len(figure_boxes)), key=lambda i: figure_boxes[i][1])

    for position, index in enumerate(order):
        if resolved[index] is not None:
            continue
        box = figure_boxes[index]

        for lower in order[position + 1:]:
            if resolved[lower] is None:
                continue
            owner = figure_boxes[lower]
            if _horizontal_overlap(box, owner) < MIN_HORIZONTAL_OVERLAP:
                continue
            if _a_caption_lies_between(box, owner, label_lines):
                break               # a different figure starts here
            # Distance stays the owner's: this crop's claim is exactly as
            # strong as the crop it belongs to, no stronger.
            resolved[index] = resolved[lower]
            break

    return resolved


def _adopt_orphan_captions(
    figure_boxes: Sequence[Tuple[float, float, float, float]],
    picks: Sequence[Optional[Tuple[LabelLine, float]]],
    label_lines: Sequence[LabelLine],
) -> List[Optional[Tuple[LabelLine, float]]]:
    """
    Pair a caption nobody claimed with the lone crop sitting above it.

    For biology researchers: the two tests above are geometric — a caption has
    to be CLOSE to its figure and SIDE BY SIDE with it. Both fail on real
    pages, for reasons that have nothing to do with the caption being wrong:

    * **A panel the detector could not see.** The `Collagen structure` review's
      page 5 prints Figure 4 as panels A, B and C, where panel C is a block of
      amino-acid *text*. The layout detector correctly reads C as text rather
      than as a picture, so no crop is made for it — and the 344pt of panel C
      now sitting between panels A/B and the caption puts the caption far
      outside the adjacency limit.
    * **A caption whose first line is a short bold title.** The same review's
      page 7 sets "Figure 5 Water-mediated hydrogen bonding in collagen" as a
      narrow heading over a full-width, five-panel figure. Only one panel was
      cropped, and it happens to sit to the RIGHT of that short heading, so
      the two share no horizontal span at all.

    This last pass therefore stops measuring distance and uses ORDER instead,
    which is the model stated at the top of this module: captions divide a
    page into bands, and the crops in a band belong to the caption that closes
    it. It is deliberately the last resort, and it is safe because of two
    restrictions that make it unable to disturb any pairing already made:

    1. **Only captions nobody claimed** are offered. A caption that a crop
       already paired with geometrically is never taken away from it.
    2. **Only crops that found nothing** can adopt. A crop with a label keeps
       it.

    So this can add pairings; by construction it cannot change or remove one.

    THE AMBIGUITY GUARD. A band holding more than one unlabelled crop is left
    alone. Those crops are either panels of one figure or several figures the
    detector merged, and nothing here can tell which — so the module's
    standing rule applies: an unplaceable figure stays unlabelled rather than
    being guessed at.

    Args:
        figure_boxes: crop boxes, page units, in the caller's order.
        picks: what each crop has after the geometric passes; None = nothing.
        label_lines: every caption line found on the page.

    Returns:
        A new picks list, with orphan captions adopted where unambiguous.
    """
    resolved = list(picks)
    if not label_lines:
        return resolved

    claimed = {pick[0].key for pick in resolved if pick is not None}
    ordered_lines = sorted(label_lines, key=lambda line: line.rect[1])

    # Step 1: the band above each caption starts at the caption before it.
    band_top = 0.0
    for line in ordered_lines:
        band_bottom = line.rect[1]
        if line.key not in claimed and line.starts_a_block:
            _adopt_within_band(figure_boxes, resolved, line,
                               band_top, band_bottom)
        band_top = line.rect[3]
    return resolved


def _adopt_within_band(
    figure_boxes: Sequence[Tuple[float, float, float, float]],
    resolved: List[Optional[Tuple[LabelLine, float]]],
    line: LabelLine,
    band_top: float,
    band_bottom: float,
) -> None:
    """
    Give `line` to the single unlabelled crop lying in its band, if there is one.

    A crop counts as being in the band when its vertical MIDPOINT is, which
    tolerates a crop snapped outwards into the caption above or below it.
    Mutates `resolved` in place; does nothing when the band holds zero crops
    or more than one.
    """
    candidates = []
    for index, box in enumerate(figure_boxes):
        if resolved[index] is not None:
            continue
        midpoint = (box[1] + box[3]) / 2.0
        if band_top <= midpoint <= band_bottom:
            candidates.append(index)

    if len(candidates) != 1:
        return
    index = candidates[0]
    # The distance recorded is the real gap to the caption, so a document-level
    # pass comparing claims still sees an honest — and deliberately weak —
    # number for a pairing that geometry alone would not have made.
    gap = max(line.rect[1] - figure_boxes[index][3], 0.0)
    resolved[index] = (line, gap)


#: How close a line directly above must be, as a multiple of its own height,
#: to count as running INTO the line below it. Body text is set solid: one
#: line follows the next at roughly its own height. A caption is separated
#: from whatever precedes it by the figure or by a paragraph break.
BLOCK_CONTINUATION_LINES = 1.6

#: How far two lines' left edges may differ, in page units, and still count as
#: the same margin. A few points of kerning, not an indent.
SAME_MARGIN_TOLERANCE = 3.0


def _starts_a_block(rect: Tuple[float, float, float, float],
                    lines: Sequence[Tuple[str, float, float, float, float]]
                    ) -> bool:
    """
    True when no paragraph runs directly into this line from above.

    For biology researchers: this separates a real caption from a sentence
    that merely opens with a figure's name. The `JRuberti` article's page 9
    carries the line "Figure 6 and Video S5 show a clear example of flow
    aggregation" -- which begins with a figure label, sits at the start of its
    line, and is nonetheless body prose, not a caption. What gives it away is
    that the previous line of the same paragraph ends immediately above it at
    the same left margin. A caption has whitespace or a figure above it.
    """
    x0, y0, _, y1 = rect
    height = max(y1 - y0, 1.0)
    for _, other_x0, other_y0, _, other_y1 in lines:
        if other_y1 > y0 or other_y0 >= y0:
            continue                          # not above this line
        if y0 - other_y1 > height * BLOCK_CONTINUATION_LINES:
            continue                          # too far above to be the same block
        if abs(other_x0 - x0) <= SAME_MARGIN_TOLERANCE:
            return False
    return True


def _a_caption_lies_between(upper: Sequence[float], lower: Sequence[float],
                            label_lines: Sequence[LabelLine]) -> bool:
    """True when a caption line separates two crops — so they are two figures."""
    top, bottom = upper[3], lower[1]
    for line in label_lines:
        if top < line.rect[1] < bottom and (
                _horizontal_overlap(upper, line.rect) >= MIN_HORIZONTAL_OVERLAP):
            return True
    return False


def _as_line(run):
    """Collapse a run of words into one line tuple."""
    return (" ".join(w[0] for w in run),
            min(w[1] for w in run), min(w[2] for w in run),
            max(w[3] for w in run), max(w[4] for w in run))


def _horizontal_overlap(first: Sequence[float], second: Sequence[float]) -> float:
    """Shared horizontal span, as a fraction of the narrower box."""
    overlap = min(first[2], second[2]) - max(first[0], second[0])
    if overlap <= 0:
        return 0.0
    narrower = min(first[2] - first[0], second[2] - second[0])
    return overlap / narrower if narrower > 0 else 0.0


def _count_by_key(picks: Sequence[Optional[Tuple[LabelLine, float]]]
                  ) -> Dict[str, int]:
    """How many crops chose each label — >1 means panels of one figure."""
    counts: Dict[str, int] = {}
    for pick in picks:
        if pick is not None:
            counts[pick[0].key] = counts.get(pick[0].key, 0) + 1
    return counts
