#!/usr/bin/env python3
"""
Recover the caption of a figure whose caption is printed on the NEXT page.

For biology researchers: a journal that runs a full-page figure often prints
the figure on one page and its caption on the following one. Prajnya's
labelling pass reads captions from the page the figure sits on, so for those
figures it finds nothing at all — the crop arrives with no caption, no label,
and no way for a RAG to know what it shows, even though the caption is sitting
in the corpus one page away.

Measured on the 9-document sample corpus (107 figures):

    13 figures have no usable caption
     6 of those have the caption on their OWN page   <- a different defect
     6 have it on an adjacent page                   <- THIS module
     1 has no caption anywhere nearby

Two of the recoverable ones are figures the downstream evaluation set asks
questions about, and both are full-page figures captioned overleaf:

    JRuberti p15         Figure 10, caption on p16, panels A-E
    Hydration-induced    Fig. 10,   caption on p15, panels a-e

THE TRAP THIS MUST NOT FALL INTO, WHICH IS THE SAME ONE `figure_label_pass`
NAMES BUT A SHARPER VERSION OF IT.

That module keeps captions and citations apart with three constraints:
anchoring, adjacency and uniqueness. **Adjacency does not survive the crossing
to another page** — a caption on the next page has no geometric relationship
to a crop on this one — so anchoring and uniqueness have to carry the whole
load, and anchoring alone is not enough. This line

    "Figure 4 shows the progression of the cell culture ..."

is running prose from the middle of a paragraph, and it BEGINS ITS LINE, so
`find_label_lines` returns it as a perfectly good label line. Attaching it
would put a sentence about cell culture on a figure it does not describe.

The discriminator is grammatical: **a caption is a label followed by a TITLE,
a citation is a label followed by a VERB.** So the token after the number must
be a caption separator or a capitalised word:

    "Fig. 10 | Bioinspired self-shaping ..."   separator     CAPTION
    "Figure 10. Mechanistic models ..."        separator     CAPTION
    "Figure 4 Variation of the unit twist"     capitalised   CAPTION
    "Fig. 2.2 Fibril-forming collagens ..."    capitalised   CAPTION
    "Figure 4 shows the progression ..."       lowercase     citation

WHY THERE IS NO GEOMETRIC GUARD HERE. An obvious extra safety condition would
be "the crop must sit at the bottom of its page and the caption at the top of
the next". It is deliberately absent: the four non-geometric conditions below
already resolve every case in the sample corpus correctly, and an unmeasured
geometric threshold would risk REJECTING correct recoveries to guard against a
failure that has not been observed. Add one when a corpus produces a case that
needs it, and measure it then.

The design rule inherited from `figure_label_pass` holds here too, and matters
more: **an unplaceable figure is left unlabelled rather than guessed at.** A
missing caption fails visibly; a wrong caption silently mis-describes an image
in a system whose entire purpose is correct citation.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence

from src.processing.utils.figure_labels import parse_printed_label

#: Which neighbouring pages to consult, in the order they are tried. The page
#: AFTER comes first, and that ordering is measured rather than assumed: both
#: verified cases have their caption overleaf, and in one of them
#: (Hydration-induced p14) the PRECEDING page carries a different figure's
#: caption, so a symmetric "nearest page" rule picks the wrong one.
NEIGHBOUR_OFFSETS = (1, -1)

#: A caption of fewer than this many characters is a panel letter ("d") or a
#: fragment, not a caption. Matches the threshold the downstream document
#: model uses, so both agree on what "uncaptioned" means.
MIN_USABLE_CAPTION = 15

#: What may follow the figure number in a real caption. See the module
#: docstring: this is what separates a caption from a sentence that happens to
#: start with a figure number.
_CAPTION_SEPARATOR = re.compile(r"^\s*[.|:–—-]")
_CAPITALISED_WORD = re.compile(r"^\s+[A-Z(\[]")

#: A PANEL sub-caption: a lowercase panel letter, bare or bracketed, then
#: optionally "|", ")" or ".", then a capitalised word. "a The shape recovery
#: of the shaft of a bird feather" (Hydration p10, panel a of Fig. 7, whose
#: caption is printed overleaf). Lowercase only, so a real caption that
#: happens to begin "A schematic of ..." never matches.
_PANEL_CAPTION = re.compile(r"^\s*\(?[a-z]\)?\s*[|.)]?\s+[A-Z]")


def recover_overleaf_captions(
    figures_by_page: Dict[int, List[dict]],
    label_lines_by_page: Dict[int, Sequence],
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Give uncaptioned crops the caption printed on an adjacent page.

    For biology researchers: this runs once per document, after every page has
    been read, because it is the only point at which one page's figures and
    another page's text are both available.

    A page's uncaptioned crops are treated together and all receive the same
    caption, because that is what they are: several panels of one printed
    figure that the layout detector returned separately.

    Args:
        figures_by_page: page number -> the figure records for that page, as
            dicts. Modified IN PLACE, which is why nothing is returned but a
            count.
        label_lines_by_page: page number -> `LabelLine`s from
            `figure_label_pass.find_label_lines` for that page.
        logger: optional logger.

    Returns:
        How many figure records were given a caption.
    """
    log = logger or logging.getLogger("CaptionOverleaf")

    claimed = _claimed_keys(figures_by_page)
    captioned_pages = _pages_holding_a_captioned_figure(figures_by_page)

    recovered = 0
    for page in sorted(figures_by_page):
        orphans = [f for f in figures_by_page[page] if _needs_a_caption(f)]
        if not orphans:
            continue

        candidate = _find_overleaf_caption(
            page, label_lines_by_page, claimed, captioned_pages
        )
        if candidate is None:
            log.debug("p%d: %d uncaptioned crop(s), nothing safe on either "
                      "neighbouring page — left unlabelled.", page, len(orphans))
            continue

        offset, line = candidate
        for figure in orphans:
            # A panel keeps the text printed under it: the overleaf line is
            # only the caption's FIRST line, and the panel's own sentence is
            # what is specific to that crop.
            if not is_panel_caption((figure.get("caption") or "").strip()):
                figure["caption"] = line.text
            figure["printed_label"] = line.printed
            figure["label_key"] = line.key
            figure["panel_group_size"] = max(len(orphans), 1)
            recovered += 1

        # One printed figure owns its number, so nothing further may take it.
        claimed.add(line.key)
        log.info("p%d: %d crop(s) adopt %s from page %d — a full-page figure "
                 "captioned overleaf.",
                 page, len(orphans), line.printed, page + offset)

    return recovered


# ---------------------------------------------------------------------------
# Helpers (after the main function, per CLAUDE.md)
# ---------------------------------------------------------------------------


def _needs_a_caption(figure: dict) -> bool:
    """Whether a figure record arrived with no label and no real caption.

    A crop that already carries a label was placed by the geometric pass on
    its own page, which is stronger evidence than anything available across
    a page boundary, so it is never touched. Otherwise the crop needs a
    caption when it has none worth the name: shorter than
    `MIN_USABLE_CAPTION`, or only its panel's sub-caption (`is_panel_caption`).
    """
    if figure.get("printed_label"):
        return False
    caption = (figure.get("caption") or "").strip()
    return len(caption) < MIN_USABLE_CAPTION or is_panel_caption(caption)


def is_panel_caption(caption: str) -> bool:
    """
    True for a PANEL's sub-caption carrying no figure label.

    The layout detector sometimes boxes the text printed under one panel
    ("a The shape recovery of the shaft of a bird feather") as that crop's
    caption. It is long enough to pass for a caption, so until 2026-11-13
    such a crop was never treated as uncaptioned, and when its figure's
    caption was printed on the facing page its sibling panels adopted the
    label and it did not (Hydration p10: Fig. 7's panels b and c labelled,
    panel a not; the last unlabelled crop of the corpus).
    """
    printed, _ = parse_printed_label(caption)
    return printed is None and bool(_PANEL_CAPTION.match(caption))


def _claimed_keys(figures_by_page: Dict[int, List[dict]]) -> set:
    """Every figure number already spoken for somewhere in this document.

    This is the uniqueness constraint, and it is what stops a caption being
    duplicated onto a second figure: a printed figure is captioned once.
    """
    claimed = set()
    for figures in figures_by_page.values():
        for figure in figures:
            if figure.get("label_key"):
                claimed.add(figure["label_key"])
                continue
            _, key = parse_printed_label(figure.get("caption") or "")
            if key:
                claimed.add(key)
    return claimed


def _pages_holding_a_captioned_figure(
    figures_by_page: Dict[int, List[dict]]
) -> set:
    """Pages whose own figures are captioned.

    A neighbouring page that has its own captioned figure is ambiguous: a
    caption found there plausibly belongs to that figure rather than to the
    orphan next door. Such a page is skipped rather than guessed about.
    """
    pages = set()
    for page, figures in figures_by_page.items():
        for figure in figures:
            if not _needs_a_caption(figure):
                pages.add(page)
                break
    return pages


def _find_overleaf_caption(
    page: int,
    label_lines_by_page: Dict[int, Sequence],
    claimed: set,
    captioned_pages: set,
):
    """Pick a safe caption from a neighbouring page, or None.

    Returns `(offset, LabelLine)`. The four conditions, in the order applied:
    the neighbour must not hold its own captioned figure; the line must read
    like a caption rather than a citation; its number must be unclaimed; and
    the page after is preferred over the page before.
    """
    for offset in NEIGHBOUR_OFFSETS:
        neighbour = page + offset
        if neighbour in captioned_pages:
            continue
        for line in label_lines_by_page.get(neighbour, ()):
            if line.key in claimed:
                continue
            if not reads_like_a_caption(line.text):
                continue
            return offset, line
    return None


def reads_like_a_caption(text: str) -> bool:
    """
    Whether a label-opening line is a caption rather than a citation.

    For biology researchers: "Figure 4. Progression of cell culture" is a
    caption — a label, then the figure's title. "Figure 4 shows the
    progression of the cell culture" is a sentence in the body text that
    merely mentions the figure. Both begin with "Figure 4", so telling them
    apart needs the word that comes next: a title starts with a capital, a
    sentence continues with a lowercase verb.

    Args:
        text: the full text of the line.

    Returns:
        True when the line looks like the start of a caption.
    """
    printed, _ = parse_printed_label(text)
    if not printed:
        return False

    # Where the label ends in the ORIGINAL text. `printed` is tidied, so its
    # length cannot be used directly on a line whose label carried the extra
    # whitespace a two-column text layer sprinkles in.
    rest = _text_after_label(text)
    return bool(_CAPTION_SEPARATOR.match(rest) or _CAPITALISED_WORD.match(rest))


def _text_after_label(text: str) -> str:
    """The remainder of a line once its leading figure label is removed."""
    match = re.match(
        r"^\s*(?:Fig(?:ure|s|ures)?|FIG|Scheme|Chart|Plate|Panel)"
        r"\s*\.?\s*(?:[0-9]{1,3}(?:\.[0-9]{1,3})*|[IVXLivxl]{1,6})",
        text,
        re.IGNORECASE,
    )
    return text[match.end():] if match else ""
