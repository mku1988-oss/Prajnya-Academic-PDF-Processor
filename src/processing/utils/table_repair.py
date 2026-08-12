"""
Table Repair — recover the table a full-page read silently dropped.

For biology researchers: our page reader is excellent on prose but has no model of page LAYOUT,
so when a table shares a page with running text it sometimes skips the table entirely. The
content-drop check (content_drop_check.py) detects that loss by comparing the read against the
page's digital text layer. THIS module is what happens next: it takes the layout model's map of
the page, finds where the table sits, crops it out, has the reader re-read just that clean crop
(where it is accurate), and splices the recovered table back into the page text.

Design (from TIER_OVERHAUL_PLAN_GLM.md §0):
  * The table is located by a LAYOUT model's grounding boxes, parsed by the VLM-agnostic
    table_region_detector using a GroundingSpec built from that model's config.
  * The crop is re-read by the extraction model through a caller-supplied `read_table_crop`
    callable — so this module names no model and carries no prompt. The table grid comes from the
    reader; nothing here trusts the layout model's own transcription (it fabricates on dense
    tables).
  * The recovered grid is APPENDED to the full-page text. The gate fired precisely because those
    tables were MISSING from the read, so appending them restores exactly the lost content while
    keeping the reader's (good) prose for the rest of the page.

VLM-AGNOSTIC: this module knows nothing about which extraction or layout model is in use. Model
identity lives entirely in the GroundingSpec (from the layout model's config) and in the
`read_table_crop` callable (bound by the caller to the extraction provider + its configured
prompt). Swapping either model is a config change, not a code change.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from PIL import Image

from .table_region_detector import GroundingSpec, TableRegion, locate_table_regions

# A caller-supplied reader: given a cropped table image, return its transcription (e.g. HTML).
# The caller binds this to the extraction provider and the prompt resolved from that model's
# config, so this module stays free of any model name or prompt string.
CropReader = Callable[[Image.Image], str]

# Marks a spliced-in table so the recovery is auditable in the output and downstream parsing.
RECOVERED_TABLE_MARKER = "<!-- table recovered by repair path -->"


@dataclass
class RepairResult:
    """
    Outcome of repairing one page.

    Attributes:
        text: the page text with any recovered tables appended (unchanged if nothing recovered).
        tables_recovered: how many table crops were re-read and spliced in.
        region_boxes: the pixel boxes that were cropped, for logging / the ledger.
        repaired: True when at least one table was recovered.
    """
    text: str
    tables_recovered: int
    region_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        return self.tables_recovered > 0


def repair_page_tables(
    page_image: Image.Image,
    full_page_text: str,
    grounding_output: str,
    spec: GroundingSpec,
    read_table_crop: CropReader,
    logger: Optional[logging.Logger] = None,
) -> RepairResult:
    """
    Locate the dropped table(s), re-read them from clean crops, and splice them back in.

    Args:
        page_image: the rendered page image the layout model saw (source of the crops).
        full_page_text: the extraction model's full-page read, which dropped the table(s).
        grounding_output: the layout model's raw grounding response for this page.
        spec: the layout model's grounding format (regex + scale + labels), built from its config.
        read_table_crop: callable that re-reads one cropped table image and returns its text.
        logger: optional logger.

    Returns:
        A RepairResult. On a page with no locatable table it returns the text unchanged with
        tables_recovered == 0, so the caller can apply it unconditionally.
    """
    log = logger or logging.getLogger("TableRepair")

    regions = locate_table_regions(grounding_output, page_image, spec, logger=log)
    if not regions:
        log.info("Repair: layout model located no table to recover")
        return RepairResult(text=full_page_text, tables_recovered=0)

    recovered_grids = _read_table_regions(page_image, regions, read_table_crop, log)
    if not recovered_grids:
        log.warning("Repair: located %d table region(s) but re-read returned nothing", len(regions))
        return RepairResult(text=full_page_text, tables_recovered=0,
                            region_boxes=[r.box for r in regions])

    repaired_text = _splice_tables(full_page_text, recovered_grids)
    log.info("Repair: recovered %d table(s) and spliced them into the page", len(recovered_grids))
    return RepairResult(
        text=repaired_text,
        tables_recovered=len(recovered_grids),
        region_boxes=[r.box for r in regions],
    )



def _read_table_regions(
    page_image: Image.Image,
    regions: List[TableRegion],
    read_table_crop: CropReader,
    log: logging.Logger,
) -> List[str]:
    """
    Crop each located table region and re-read it, keeping only non-empty transcriptions.

    Args:
        page_image: the full rendered page.
        regions: crop-safe table regions from the detector.
        read_table_crop: the caller's crop reader.
        log: logger.

    Returns:
        One transcription string per table successfully re-read, in page order.
    """
    grids = []
    for index, region in enumerate(regions, start=1):
        crop = page_image.crop(region.box)
        log.debug("Repair: reading table crop %d/%d at %s", index, len(regions), region.box)
        transcription = read_table_crop(crop)
        if transcription and transcription.strip():
            grids.append(transcription.strip())
        else:
            log.warning("Repair: crop %d at %s returned empty text", index, region.box)
    return grids


def _splice_tables(full_page_text: str, recovered_grids: List[str]) -> str:
    """
    Append recovered table grids to the page text, each behind an auditable marker.

    Appending (rather than trying to re-insert at the exact reading-order position) is deliberate:
    the tables were absent from the read, so appending restores the missing content losslessly
    without disturbing the prose the reader already got right. The marker makes every recovery
    visible in the output and easy to find downstream.

    Args:
        full_page_text: the reader's full-page text.
        recovered_grids: the re-read table transcriptions.

    Returns:
        The page text with the recovered tables appended.
    """
    blocks = [full_page_text.rstrip()]
    for grid in recovered_grids:
        blocks.append(f"{RECOVERED_TABLE_MARKER}\n{grid}")
    return "\n\n".join(blocks)
