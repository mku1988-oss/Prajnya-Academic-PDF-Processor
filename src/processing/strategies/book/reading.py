#!/usr/bin/env python3
"""
Reading a book's chapters — selection, attribution, and cost control.

For biology researchers: a 345-page book cannot be read like an 8-page paper.
At the research-article pipeline's measured ~20s/page that would be about two
hours of GPU time for one volume, and most of it would be wasted: 73% of this
book's pages are ordinary prose whose text layer is already perfect.

So book reading rests on two decisions, and this module is only those two —
the actual page reading is delegated to the article pipeline's `BodyPipeline`,
which already does classify → text layer or VLM → coverage gate → table repair
→ figure extraction, and is covered by the parity harness. Duplicating it here
would mean maintaining a second copy of a benchmarked path.

**Decision 1: which pages.** Reading is *opt-in and chapter-scoped*. A caller
asks for chapters — "chapter 3", "chapters 1-4" — and gets exactly those
pages. Nothing reads a whole book by accident, which matters when the
difference between a chapter and a volume is two hours of GPU.

**Decision 2: whose words these are.** Every page that comes back is stamped
with its chapter's number, title and **authors**, from the chapter map. This
is the point of the whole book path: a passage retrieved from page 60 must
carry Siadat et al., not the volume's editor. Attribution travels with the
content rather than being looked up later, because later is when it gets lost.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from ...models import ContentPage
from .chapter_boundaries import ChapterMap, ChapterSpan

LOG = logging.getLogger(__name__)


def select_chapter_pages(chapter_map: ChapterMap,
                         chapters: Optional[Sequence[int]] = None,
                         max_pages: Optional[int] = None) -> List[int]:
    """
    Which PDF pages to read, for the chapters asked for.

    Args:
        chapter_map: the detected chapter spans.
        chapters: chapter numbers to read. None means every located chapter —
            which for a whole volume is a large, deliberate request.
        max_pages: a hard ceiling on how many pages are returned, as a guard
            against an accidental whole-book read. Pages are dropped from the
            END of the selection, so the chapters asked for first are complete
            rather than every chapter being truncated.

    Returns:
        0-based PDF page indices, ascending and without duplicates.

    Example:
        >>> select_chapter_pages(chapter_map, chapters=[3])      # doctest: +SKIP
        [52, 53, 54, ..., 110]
    """
    spans = _spans_for(chapter_map, chapters)
    if not spans:
        LOG.warning("No chapters selected for reading (asked for %s)", chapters)
        return []

    indices = sorted({index for span in spans
                      for index in range(span.start_index, span.end_index + 1)})

    if max_pages is not None and len(indices) > max_pages:
        LOG.warning("Selection is %d pages; truncating to the %d-page limit. "
                    "Ask for fewer chapters to read them whole.",
                    len(indices), max_pages)
        indices = indices[:max_pages]

    LOG.info("Reading %d pages across %d chapter(s): %s",
             len(indices), len(spans), [span.number for span in spans])
    return indices


def attribute_pages(pages: Sequence[ContentPage],
                    chapter_map: ChapterMap) -> List[ContentPage]:
    """
    Stamp each page with the chapter it belongs to — and whose work it is.

    For biology researchers: this is what makes a book citable. Without it a
    retrieved passage knows only which *book* it came from, and every chapter
    of an edited volume would be attributed to the editor.

    Args:
        pages: the pages `BodyPipeline` returned. `page_number` is 1-based.
        chapter_map: the detected chapter spans.

    Returns:
        The same pages, each with a `chapter` block in its metadata. A page
        outside every span gets `None` rather than a nearest guess.
    """
    attributed = []

    for page in pages:
        span = chapter_map.chapter_for_page(page.page_number - 1)
        page.metadata = {**page.metadata, "chapter": _attribution(span)}
        attributed.append(page)

    unattributed = sum(1 for page in attributed if page.metadata["chapter"] is None)
    if unattributed:
        LOG.warning("%d of %d pages fall outside every chapter span and carry no "
                    "attribution", unattributed, len(attributed))

    return attributed


def estimate_reading_cost(page_indices: Sequence[int]) -> Dict[str, float]:
    """
    A rough idea of what a selection will cost, before it is paid.

    For biology researchers: this exists so a request for a whole volume is
    visibly expensive *before* the GPU starts, not two hours later.

    The per-page figure is the research-article pipeline's measured average on
    the NMR benchmark (158.8s over 8 pages). It is a deliberate OVER-estimate
    for a book: most book pages are prose that never reaches the GPU at all,
    while the benchmark's are dense with tables and figures. Treat it as a
    ceiling, not a forecast.

    Returns:
        `pages`, `worst_case_seconds`, and `worst_case_minutes`.
    """
    pages = len(page_indices)
    worst_case = pages * SECONDS_PER_PAGE_CEILING
    return {
        "pages": pages,
        "worst_case_seconds": round(worst_case, 1),
        "worst_case_minutes": round(worst_case / 60, 1),
    }


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

#: Measured on the NMR benchmark: 158.8s for 8 dense pages. See
#: `estimate_reading_cost` for why this is a ceiling rather than a forecast.
SECONDS_PER_PAGE_CEILING = 19.85


def _spans_for(chapter_map: ChapterMap,
               chapters: Optional[Sequence[int]]) -> List[ChapterSpan]:
    """
    The spans a caller asked for, in reading order.

    Both None and an empty selection mean **every located chapter**. The empty
    list is what the CLI produces for `--chapters` with no numbers, and it
    reached here as "no chapters" — so a whole-volume read silently returned
    nothing after the user explicitly asked for everything. Nobody asks to read
    exactly zero chapters through a list; the only way to read nothing is to
    not request bodies at all, which `BookStrategy` decides above this.
    """
    if not chapters:
        return sorted(chapter_map.spans, key=lambda span: span.start_index)

    wanted = set(chapters)
    found = [span for span in chapter_map.spans if span.number in wanted]

    missing = wanted - {span.number for span in found}
    if missing:
        LOG.warning("No located span for chapter(s) %s; they cannot be read",
                    sorted(missing))

    return sorted(found, key=lambda span: span.start_index)


def _attribution(span: Optional[ChapterSpan]) -> Optional[Dict]:
    """The citation block stored on a page, or None outside every chapter."""
    if span is None:
        return None
    return {
        "number": span.number,
        "title": span.title,
        "authors": list(span.authors),
        "page_range": [span.start_index, span.end_index],
        "boundary_confidence": span.confidence,
    }
