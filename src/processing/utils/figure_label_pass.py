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
    """

    printed: str
    key: str
    text: str
    rect: Tuple[float, float, float, float]


@dataclass(frozen=True)
class LabelAssignment:
    """
    The outcome for one figure crop.

    Attributes:
        printed: label as printed, or None when nothing could be established.
        key: normalized key, or None.
        caption: the caption line's text — recovered even when the detector
            gave us no caption box at all.
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

    Args:
        lines: `(text, x0, y0, x1, y1)` per line, in page units.

    Returns:
        One `LabelLine` per caption-looking line, page order preserved.
    """
    found: List[LabelLine] = []
    for text, x0, y0, x1, y1 in lines:
        printed, key = parse_printed_label(text)
        if printed and key:
            found.append(LabelLine(printed=printed, key=key, text=text,
                                   rect=(x0, y0, x1, y1)))
    return found


def assign_labels_on_page(
    figure_boxes: Sequence[Tuple[float, float, float, float]],
    label_lines: Sequence[LabelLine],
    page_height: float,
    logger: Optional[logging.Logger] = None,
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
    shared = _count_by_key(picks)

    assignments: List[LabelAssignment] = []
    for box, pick in zip(figure_boxes, picks):
        if pick is None:
            assignments.append(LabelAssignment(None, None, "", 0, float("inf")))
            continue
        line, gap = pick
        assignments.append(LabelAssignment(
            printed=line.printed, key=line.key, caption=line.text,
            shared_with=shared[line.key], distance=gap,
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
