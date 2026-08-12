#!/usr/bin/env python3
"""
Document anchoring — a sparse text "skeleton" of a page for the VLM.

For biology researchers: before the vision model reads a page image, we hand
it a few coordinates-plus-words hints from the text layer ("at position X,Y
the text starts 'Materials and methods…'"). Per the OlmOCR paper
(arXiv 2502.18443) this anchoring improves extraction quality by 15-20%,
because the model can align what it sees with what the text layer already
knows.

Ported from V5's `DocumentAnchoringExtractor`, with the access rewritten onto
the `PdfBackend` seam (V5 called `fitz` directly — the licence rule bans
that). One behavioural note, recorded for the parity run: V5 sampled the
first 8 words of each PyMuPDF text *block*; the seam exposes words, so V6
groups words into lines and lines into blocks by vertical gap. The anchor
format is unchanged (`[x,y]text`, bottom-left origin, 6000-char cap), but
block boundaries may differ slightly from PyMuPDF's. The anchor is context
for the VLM, not extracted output — if title-page parity ever fails, check
here first.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

from ..pdf.base import PdfPage, Word, group_words_into_lines, line_text

#: Words sampled from the start of each block (V5's value).
WORDS_PER_BLOCK = 8

#: Cap on the final anchor text (V5's value).
MAX_ANCHOR_CHARS = 6000

#: A vertical gap larger than this multiple of the line height starts a new
#: block (paragraph break, heading, column jump).
BLOCK_GAP_FACTOR = 1.5

logger = logging.getLogger("AnchorExtractor")


def extract_document_anchoring(page: PdfPage, max_chars: int = MAX_ANCHOR_CHARS) -> str:
    """
    Build the sparse anchor text for one page.

    Args:
        page: the page (any PdfBackend implementation).
        max_chars: cap on the returned text.

    Returns:
        One `[x,y]first words…` line per text block, y in bottom-left origin
        (the convention OlmOCR was trained on). Falls back to a bare
        page-dimensions line when the page has no usable text layer.
    """
    width_pts, height_pts = page.size
    try:
        blocks = _group_lines_into_blocks(group_words_into_lines(page.words()))
    except Exception as error:
        logger.warning("Document anchoring extraction failed: %s", error)
        return "Page dimensions: 612x792\n"

    anchor_text = ""
    for block in blocks:
        words = [w for line in block for w in line]
        sampled = " ".join(w.text for w in words[:WORDS_PER_BLOCK]).strip()
        if not sampled:
            continue
        x0 = min(w.rect.x0 for w in words)
        y0 = min(w.rect.y0 for w in words)
        y_lower_left = height_pts - y0  # top-left origin -> PDF bottom-left
        block_anchor = f"[{int(x0)},{int(y_lower_left)}]{sampled}\n"
        if len(anchor_text) + len(block_anchor) > max_chars:
            break
        anchor_text += block_anchor

    if anchor_text:
        return anchor_text
    return f"Page dimensions: {width_pts}x{height_pts}\n"


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def _group_lines_into_blocks(lines: Sequence[Sequence[Word]]) -> List[List[Sequence[Word]]]:
    """
    Group consecutive text lines into blocks by vertical gap.

    A gap larger than `BLOCK_GAP_FACTOR` line heights means a new block —
    the same visual cue a human uses to see where a paragraph ends.
    """
    blocks: List[List[Sequence[Word]]] = []
    current: List[Sequence[Word]] = []
    previous_bottom: float | None = None

    for line in lines:
        if not line:
            continue
        top = min(w.rect.y0 for w in line)
        height = max(w.rect.y1 - w.rect.y0 for w in line)
        gap_limit = BLOCK_GAP_FACTOR * max(height, 1.0)
        if current and previous_bottom is not None and (top - previous_bottom) > gap_limit:
            blocks.append(current)
            current = []
        current.append(line)
        previous_bottom = max(w.rect.y1 for w in line)

    if current:
        blocks.append(current)
    return blocks
