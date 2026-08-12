#!/usr/bin/env python3
"""
Table Repair — recover the table a full-page read silently dropped.

For biology researchers: the page reader is excellent on prose but has no model
of page LAYOUT, so when a table shares a page with running text it sometimes
skips the table entirely. The content-drop check detects that loss by comparing
the read against the page's digital text layer. THIS module is what happens
next: it takes the layout detector's map of the page, finds where the table
sits, crops it out, has the reader re-read just that clean crop (where it is
accurate), and splices the recovered table back into the page text.

Why re-read a crop instead of trusting what we have:

  A recognition model reading a whole mixed page can skip the table. Reading a
  *cropped* table it is highly accurate (measured at 96.7-100% numeric recall
  in V5). So the crop is the fix. Note that nothing here trusts the layout
  detector's own transcription — a grounding VLM fabricates on dense tables,
  inventing values and rowspans. We take its BOXES and nothing else.

The recovered grid is APPENDED to the page text rather than re-inserted at its
true reading-order position. That is deliberate: the repair gate fires
precisely because those tables were MISSING, so appending restores exactly the
lost content while keeping the reader's (good) prose undisturbed.

VLM-AGNOSTIC: this module names no model and carries no prompt. Model identity
lives entirely in the `read_table_crop` callable, which the caller binds to the
extraction provider and the prompt resolved from that model's config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from src.processing.layout.base import Detection
from src.processing.layout.regions import TableRegion, locate_table_regions

#: A caller-supplied reader: given a cropped table image, return its
#: transcription (e.g. HTML). Bound by the caller to the extraction provider,
#: which is what keeps this module free of any model name or prompt string.
CropReader = Callable[["Image.Image"], str]

#: Marks a spliced-in table so the recovery is auditable in the output.
RECOVERED_TABLE_MARKER = "<!-- table recovered by repair path -->"


@dataclass
class RecoveredTable:
    """
    One table this page recovered: where it was, and what it says.

    Pairing the box with its transcription is what lets the caller tag the
    table as a retrievable object downstream — the crop was already read
    here, so the RAG gets the grid for free rather than paying a second
    model call for it.

    Attributes:
        box: (x0, y0, x1, y1) in pixels of the rendered page image.
        grid: the re-read transcription of that crop.
    """

    box: Tuple[int, int, int, int]
    grid: str


@dataclass
class RepairResult:
    """
    Outcome of repairing one page.

    Attributes:
        text: page text with any recovered tables appended (unchanged if none).
        tables_recovered: how many table crops were re-read and spliced in.
        region_boxes: every located box, including ones that re-read empty.
        tables: the successfully recovered tables, box paired with grid, in
            page order. `tables_recovered == len(tables)`.
    """

    text: str
    tables_recovered: int
    region_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    tables: List[RecoveredTable] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        return self.tables_recovered > 0


def repair_page_tables(
    page_image: "Image.Image",
    full_page_text: str,
    detections: Sequence[Detection],
    read_table_crop: CropReader,
    logger: Optional[logging.Logger] = None,
) -> RepairResult:
    """
    Locate the dropped table(s), re-read them from clean crops, splice them in.

    Args:
        page_image: the rendered page the detections were made against.
        full_page_text: the extraction model's full-page read, which dropped
            the table(s).
        detections: this page's layout detections, from any `LayoutDetector`.
            V5 took a raw grounding string plus a spec here; the parse now
            belongs to the detector.
        read_table_crop: callable that re-reads one cropped table image.
        logger: optional logger.

    Returns:
        A `RepairResult`. On a page with no locatable table it returns the text
        unchanged with `tables_recovered == 0`, so the caller can apply it
        unconditionally.
    """
    log = logger or logging.getLogger("TableRepair")

    regions = locate_table_regions(detections, page_image, logger=log)
    if not regions:
        log.info("Repair: layout detector located no table to recover")
        return RepairResult(text=full_page_text, tables_recovered=0)

    recovered = _read_table_regions(page_image, regions, read_table_crop, log)
    if not recovered:
        log.warning("Repair: located %d table region(s) but re-read returned nothing",
                    len(regions))
        return RepairResult(text=full_page_text, tables_recovered=0,
                            region_boxes=[r.box for r in regions])

    repaired_text = _splice_tables(full_page_text, [t.grid for t in recovered])
    log.info("Repair: recovered %d table(s) and spliced them into the page",
             len(recovered))
    return RepairResult(
        text=repaired_text,
        tables_recovered=len(recovered),
        region_boxes=[r.box for r in regions],
        tables=recovered,
    )


# --- helpers (after the main function, per CLAUDE.md) ----------------------

def _read_table_regions(
    page_image: "Image.Image",
    regions: Sequence[TableRegion],
    read_table_crop: CropReader,
    log: logging.Logger,
) -> List[RecoveredTable]:
    """
    Crop each located table region and re-read it, keeping non-empty results.

    Returns one `RecoveredTable` per table successfully re-read, in page order,
    each carrying the box it came from so the caller can tag it.
    """
    recovered: List[RecoveredTable] = []
    for index, region in enumerate(regions, start=1):
        crop = page_image.crop(region.box)
        log.debug("Repair: reading table crop %d/%d at %s", index, len(regions), region.box)
        transcription = read_table_crop(crop)
        if transcription and transcription.strip():
            recovered.append(RecoveredTable(box=region.box, grid=transcription.strip()))
        else:
            log.warning("Repair: crop %d at %s returned empty text", index, region.box)
    return recovered


def _splice_tables(full_page_text: str, recovered_grids: Sequence[str]) -> str:
    """
    Append recovered table grids to the page text, each behind an audit marker.

    The marker makes every recovery visible in the output and easy to find
    downstream.
    """
    blocks = [full_page_text.rstrip()]
    for grid in recovered_grids:
        blocks.append(f"{RECOVERED_TABLE_MARKER}\n{grid}")
    return "\n\n".join(blocks)
