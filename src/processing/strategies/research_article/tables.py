#!/usr/bin/env python3
"""
Table tagging — from layout detections to saved crops + manifest.

For biology researchers: a figure in this pipeline is a retrievable *object* —
it has an id, a saved PNG, a caption, and a page, so a multimodal search can
return "Figure 3 from the NMR paper" and show it. Before this module, a table
was none of those things. Its text was spliced into the page's prose blob and
the only trace it left was a count (`tables_recovered: 2`). You could not
retrieve a table, cite one, or display it.

This gives tables the same treatment as figures: crop, save, caption, id,
manifest. Two things differ from `figures.py`, both inherited from how tables
actually sit on a page:

  1. **The caption is inside the crop.** `locate_table_regions` deliberately
     folds a table's caption and footnote into its box, because a table read
     without its header row or units footnote is worse than useless. So the
     PNG contains the caption, and we ALSO read the caption text separately
     (from the text layer) for the manifest — the same string, two uses.

  2. **Every table leaves with a grid.** When the repair path ran, it already
     re-read the crop, and that transcription is carried through at zero extra
     cost. When repair did NOT run, the crop is read here, with the same
     reader. An earlier version skipped that read, reasoning that the table's
     text was already in the page content. Measured downstream on 2026-09-11,
     that reasoning failed: the retrieval layer drops table markup from page
     text BECAUSE the table object is meant to carry it, so 14 of 31 corpus
     tables reached no index at all. See `_grid_for`.

VLM-AGNOSTIC: no model name and no prompt appears here. A transcription
arrives already made, via `recovered`, or through the `read_table_crop`
callable the caller binds to its extraction model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from ...events import emit
from ...layout.base import LABEL_TABLE_CAPTION, Detection
from ...layout.regions import locate_table_regions
from ...pdf.base import PdfPage
from ...utils.figure_extractor import safe_doc_id
from ...utils.text_normalization import normalize_harvested_text
from ...utils.table_repair import CropReader, RecoveredTable
from .figures import image_box_to_pdf_rect

#: Overlap needed before a caption detection counts as belonging to a table
#: region. The caption is folded INTO the region box, so a real caption sits
#: almost entirely inside it; anything less is a neighbouring block.
CAPTION_CONTAINMENT = 0.6

#: Audit marker for the anchor block, mirroring the figure one.
TABLE_ANCHOR_MARKER = "<!-- table crop (layout detection) -->"


@dataclass
class TableAsset:
    """
    One tagged table, the table-side twin of `FigureAsset`.

    Attributes:
        table_id: stable per-document id, e.g. "p4_tab1".
        image_path: saved crop, relative to `path_root` when one was given.
        caption: the caption text, or "" when the page has none.
        page: 1-indexed page number.
        bbox: the crop box in pixels of the rendered page image.
        grid: the transcription when the repair path already read this crop,
            otherwise None (the text lives in the page content).
    """

    table_id: str
    image_path: str
    caption: str
    page: int
    bbox: Tuple[int, int, int, int]
    grid: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def anchor(self) -> str:
        """The inline anchor linking this table into the page's reading flow."""
        caption = " ".join(self.caption.split()) if self.caption else ""
        return f"[[TABLE {self.table_id} | {caption} | {self.image_path}]]"


def append_table_anchors(page_text: str, assets: Sequence[TableAsset]) -> str:
    """
    Append an inline anchor per table to the page text, behind an audit marker.

    Same contract as `append_figure_anchors`: the anchor is what lets a RAG
    hit on the page's prose find the table's crop and caption. Appended rather
    than re-inserted, so prose the reader already got right is undisturbed.
    """
    if not assets:
        return page_text
    blocks = [page_text.rstrip(), TABLE_ANCHOR_MARKER]
    blocks.extend(asset.anchor() for asset in assets)
    return "\n".join(blocks)


def extract_page_tables(
    page: PdfPage,
    page_image: "Image.Image",
    detections: Optional[Sequence[Detection]],
    pdf_path: Path | str,
    tables_base_dir: Path,
    page_idx: int,
    recovered: Optional[Sequence[RecoveredTable]] = None,
    path_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    read_table_crop: Optional[CropReader] = None,
) -> List[TableAsset]:
    """
    Crop and save every table on this page, tagging each with its caption.

    Uses the page's already-computed layout detections — no new model call,
    and no new detection pass.

    Args:
        page: the PDF page (for its text layer, which supplies the caption).
        page_image: the render the detections were made against.
        detections: this page's layout detections, or None when layout never
            ran (returns []).
        pdf_path: the source PDF, used to derive the document id.
        tables_base_dir: root output directory; tables land in
            `<tables_base_dir>/<doc_id>/`.
        page_idx: 0-indexed page number.
        recovered: tables the repair path already re-read, so their
            transcriptions can be attached for free.
        path_root: manifest paths are made relative to this when given.
        logger: optional logger.
        read_table_crop: reads one cropped table (the extraction model with
            its table prompt). A table the repair path did NOT transcribe is
            read with it, so every table leaves with a grid. None keeps the
            old behaviour: such a table's grid stays None.

    Returns:
        One `TableAsset` per saved table, in page order; [] when none.
    """
    log = logger or logging.getLogger("Tables")
    if not detections:
        return []

    regions = locate_table_regions(detections, page_image, logger=log)
    if not regions:
        return []

    out_dir = Path(tables_base_dir) / safe_doc_id(Path(pdf_path).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_lookup = _grids_by_box(recovered)

    assets: List[TableAsset] = []
    for index, region in enumerate(regions, start=1):
        try:
            assets.append(_save_one_table(
                page=page, page_image=page_image, box=region.box,
                detections=detections, out_dir=out_dir,
                page_number=page_idx + 1, index=index,
                grid=_grid_for(region.box, grid_lookup, page_image,
                               read_table_crop, page_idx + 1, log),
                path_root=path_root, logger=log,
            ))
        except Exception as error:
            # A table crop that fails to save must not lose the page.
            log.warning("Could not save table %d on page %d: %s",
                        index, page_idx + 1, error)

    if assets:
        emit(log, "stage", stage="tables",
             message=f"Page {page_idx + 1}: saved {len(assets)} table(s)",
             page=page_idx + 1, count=len(assets))
    return assets


def write_table_manifest(
    pdf_path: Path | str,
    tables_base_dir: Path,
    tables: Sequence[dict],
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """
    Write the document-level table manifest (JSON) the RAG consumes.

    Deliberately the same shape as the figure manifest, so a downstream
    indexer can walk both with one code path.

    Args:
        pdf_path: the source PDF, used to derive the document id.
        tables_base_dir: root table directory (same as extraction used).
        tables: every page's table dicts, already flattened, in page order.
        logger: optional logger.

    Returns:
        The manifest path, or None when there was nothing to write or the
        write failed (a missing manifest must not fail the document).
    """
    log = logger or logging.getLogger("Tables")
    if not tables:
        return None
    doc_id = safe_doc_id(Path(pdf_path).stem)
    manifest_dir = Path(tables_base_dir) / doc_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    try:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"document": doc_id, "table_count": len(tables),
                 "tables": list(tables)},
                handle, indent=2, ensure_ascii=False,
            )
        emit(log, "stage", stage="tables",
             message=f"Table manifest: {len(tables)} table(s) -> {manifest_path}",
             count=len(tables), manifest=str(manifest_path))
        return manifest_path
    except Exception as error:
        log.warning("Could not write table manifest: %s", error)
        return None


# --- helpers (after the main functions, per CLAUDE.md) ---------------------

def _save_one_table(
    page: PdfPage,
    page_image: "Image.Image",
    box: Tuple[int, int, int, int],
    detections: Sequence[Detection],
    out_dir: Path,
    page_number: int,
    index: int,
    grid: Optional[str],
    path_root: Optional[Path],
    logger: logging.Logger,
) -> TableAsset:
    """Crop, save, and tag one table."""
    table_id = f"p{page_number}_tab{index}"
    image_path = out_dir / f"{table_id}.png"
    page_image.crop(box).save(image_path)

    return TableAsset(
        table_id=table_id,
        image_path=_portable_path(image_path, path_root),
        caption=_caption_for(page, page_image, box, detections),
        page=page_number,
        bbox=tuple(box),
        grid=grid,
    )


def _caption_for(
    page: PdfPage,
    page_image: "Image.Image",
    box: Tuple[int, int, int, int],
    detections: Sequence[Detection],
) -> str:
    """
    The caption text for a table region, read from the text layer.

    The caption box was folded into the region, so we find the caption
    detection sitting inside this region and read just that rectangle —
    giving the manifest the caption as text even though the PNG already
    shows it.
    """
    for detection in detections:
        if detection.label != LABEL_TABLE_CAPTION:
            continue
        if _containment(detection.box_px, box) < CAPTION_CONTAINMENT:
            continue
        rect = image_box_to_pdf_rect(tuple(detection.box_px), page, page_image)
        # Same repair as a figure caption: this is stored text, and it is often
        # the only prose a table object carries into the index.
        text = normalize_harvested_text(page.text_in(rect)).strip()
        if text:
            return text
    return ""


def _containment(inner: Sequence[int], outer: Sequence[int]) -> float:
    """Fraction of `inner`'s area that lies inside `outer` (0.0–1.0)."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    inner_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inner_area <= 0:
        return 0.0
    overlap_w = max(0, min(ix1, ox1) - max(ix0, ox0))
    overlap_h = max(0, min(iy1, oy1) - max(iy0, oy0))
    return (overlap_w * overlap_h) / inner_area


def _grids_by_box(
    recovered: Optional[Sequence[RecoveredTable]],
) -> Dict[Tuple[int, int, int, int], str]:
    """Index already-read transcriptions by the box they came from."""
    return {tuple(table.box): table.grid for table in (recovered or [])}


def _grid_for(
    box: Sequence[int],
    grid_lookup: Dict[Tuple[int, int, int, int], str],
    page_image: "Image.Image",
    read_table_crop: Optional[CropReader],
    page_number: int,
    logger: logging.Logger,
) -> Optional[str]:
    """A table's transcription: the repair path's if it ran, else a crop read.

    Why every table is read, measured downstream on 2026-09-11 (Shastrasaara,
    docs/upstream_tables_headings_gate.md): only the repair path ever
    transcribed a table, so 14 of the corpus's 31 tables left with no grid.
    The retrieval layer drops table markup from page text because the table
    object is supposed to carry it -- so those 14 tables' contents reached no
    index at all, including a Table 9.1 row two evaluation questions need.
    Reading the crop costs ~3.5 s per table with GLM-OCR's `Table
    Recognition:`, which returns HTML.

    A failed or empty read leaves the grid None: losing a transcription must
    never lose the table itself.
    """
    existing = grid_lookup.get(tuple(box))
    if existing is not None or read_table_crop is None:
        return existing

    try:
        transcription = read_table_crop(page_image.crop(tuple(box)))
    except Exception as error:  # a failed read must not lose the table
        logger.warning("Table crop read failed on page %d at %s: %s",
                       page_number, tuple(box), error)
        return None

    text = (transcription or "").strip()
    if not text:
        logger.warning("Table crop on page %d at %s read back empty",
                       page_number, tuple(box))
        return None
    return text


def _portable_path(image_path: Path, path_root: Optional[Path]) -> str:
    """Relative to `path_root` when possible, so manifests are movable."""
    if path_root is not None:
        try:
            return image_path.resolve().relative_to(path_root.resolve()).as_posix()
        except ValueError:
            pass  # not under path_root -> fall back to absolute
    return image_path.as_posix()
