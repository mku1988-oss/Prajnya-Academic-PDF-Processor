#!/usr/bin/env python3
"""
Deciding which pages deserve a second opinion about their figures.

For biology researchers: the cheap CPU detector finds figures on most pages for
almost no cost. Occasionally it misses one — and a missed figure is invisible,
because the page still reads fine and `figure_count: 0` looks like an honest
answer. This module spots the pages where the pipeline **contradicted itself**,
so they can be re-examined by a slower, better model.

The contradiction is the whole idea:

    the page classifier says   "this page has visual content on it"
    the layout detector says   "there is no figure here"

Both cannot be right. One escalation per disagreement, and nothing anywhere
else — which is what keeps this affordable.

**Why not use the page's text instead?** A first attempt triggered on the read
text naming a figure ("Fig. 4 | ..."). That is bad evidence: in a research
article a figure is routinely cited pages away from where it is printed, so the
text tells you the document has a Figure 4, not that THIS page holds it. The
classifier's visual regions are the page's own geometry — embedded images and
clustered vector strokes — so they are evidence about this page specifically.

**Why no size threshold?** Measured on the sample corpus, and it does not
separate. A references page's link decorations covered 10.6% of the page; a
genuine four-panel confocal figure the detector missed (JRuberti p14, Figure 9)
covered 9.9%. Any area cutoff that excluded the first would have discarded the
second. The only geometric exclusion here is the one that means something
structural — see `SCAN_PAGE_FRACTION`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

LOG = logging.getLogger(__name__)

#: A "visual region" this large IS the page: a scanned page is one big image,
#: so there is no separate figure to crop out of it. Both sample pages at this
#: size were scans of pure text. This is the one exclusion that is structural
#: rather than fitted to examples.
SCAN_PAGE_FRACTION = 0.85

#: Labels that count as "a figure was found here" — the same set the figure
#: extractor crops.
FIGURE_LABELS = ("figure", "image", "chart")


def page_needs_second_opinion(analysis: Dict[str, Any],
                              detections: Optional[Sequence] = None,
                              page_size: Optional[tuple] = None) -> bool:
    """
    True when the classifier saw visual content and detection found no figure.

    Args:
        analysis: the page classifier's analysis dict — its `visual_regions`
            are the evidence that something visual is on this page.
        detections: the layout detections for the page, or None when detection
            never ran (which is itself a disagreement worth escalating).
        page_size: (width, height) in PDF points, for the scan test.

    Returns:
        Whether this page should be re-detected by the grounding model.

        A page is NOT escalated when a table was detected — the visual content
        is accounted for and nothing was lost — nor when its single region
        covers the whole page, which means a scan rather than a figure.
    """
    regions = (analysis or {}).get("visual_regions") or []
    if not regions:
        return False                      # nothing was seen; nothing is missing

    for detection in detections or ():
        label = getattr(detection, "label", None)
        if label in FIGURE_LABELS:
            return False                  # a figure WAS found; no disagreement
        if label == "table":
            return False                  # the visual content is a table

    return not _is_whole_page(regions, page_size)


def collect_suspect_pages(pages: Dict[int, Any]) -> List[int]:
    """
    The page indices a document should re-examine, in reading order.

    Args:
        pages: {page index -> ContentPage}, as the body pipeline accumulates
            them. A page carries its own verdict in
            `metadata['needs_figure_second_opinion']`, set while it was read —
            deciding later would mean re-running the classifier.

    Returns:
        Sorted page indices. Empty when the document is self-consistent, which
        on the sample corpus is the overwhelming majority of documents.
    """
    return sorted(
        index for index, page in pages.items()
        if (getattr(page, "metadata", None) or {}).get("needs_figure_second_opinion")
    )


def describe_escalation(suspects: Sequence[int], seconds_each: float = 17.0) -> str:
    """A one-line summary for the log, so the cost is never silent."""
    if not suspects:
        return "no pages need a second opinion about figures"
    pages = ", ".join(str(index + 1) for index in suspects[:8])
    if len(suspects) > 8:
        pages += f", … (+{len(suspects) - 8} more)"
    return (f"{len(suspects)} page(s) disagree about figures (p{pages}); "
            f"re-detecting with the grounding model, ~{len(suspects) * seconds_each:.0f}s")


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _is_whole_page(regions: Sequence[Dict[str, Any]],
                   page_size: Optional[tuple]) -> bool:
    """
    True when a region covers essentially the whole page — i.e. it is a scan.

    Without a page size we cannot judge, and the safe answer is "not a scan":
    escalating costs time, while skipping loses a figure permanently.
    """
    if not page_size:
        return False

    width, height = page_size
    page_area = float(width) * float(height)
    if page_area <= 0:
        return False

    for region in regions:
        bbox = region.get("bbox") or region.get("rect")
        if bbox is None:
            continue
        try:
            x0, y0, x1, y1 = (bbox if isinstance(bbox, (tuple, list))
                              else (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
        except (TypeError, ValueError, AttributeError):   # pragma: no cover
            continue
        if abs((x1 - x0) * (y1 - y0)) / page_area >= SCAN_PAGE_FRACTION:
            return True

    return False
