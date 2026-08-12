#!/usr/bin/env python3
"""
Figure extraction glue — from layout detections to saved crops + manifest.

For biology researchers: figures are the evidence in a paper. This module cuts
each detected figure out of the rendered page image, saves it as a PNG, pairs
it with its caption text, and writes one `manifest.json` per document so the
multimodal RAG can find them all later.

Ported from V5's `_extract_page_figures` / `_make_caption_reader` /
`_image_box_to_pdf_rect` / `_write_figure_manifest`
(research_article_strategy.py L1705–1799), with the seams V6 introduced:

- Detections arrive as `List[Detection]` from any `LayoutDetector` — never a
  raw grounding string (the parse belongs to the detector).
- Page text access goes through `PdfPage` (`text_in`), never `fitz`.
- The manifest writer takes a plain list of figure dicts, not V5's
  `ContentPage` objects, so the module has no dependency on result models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image

from ...events import emit
from ...layout.base import Detection
from ...pdf.base import PdfPage, Rect
from ...utils.content_drop_check import is_scanned_page
from ...utils.figure_extractor import FigureAsset, extract_and_save_figures, safe_doc_id
from ...utils.figure_label_pass import (
    assign_labels_on_page,
    build_text_lines,
    find_label_lines,
)
from .rendering import RENDER_PADDING_PX

#: Extra clip slop in PDF points, so a caption line that merely touches the
#: detected box is still captured whole.
CAPTION_CLIP_SLOP_PTS = 4.0

#: Reads a caption crop with the extraction model; None when no model is
#: available (born-digital pages never need it).
CropReader = Callable[[Image.Image], str]


def extract_page_figures(
    page: PdfPage,
    page_image: Image.Image,
    detections: Optional[Sequence[Detection]],
    pdf_path: Path | str,
    figures_base_dir: Path,
    page_idx: int,
    read_crop: Optional[CropReader] = None,
    path_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> List[FigureAsset]:
    """
    Crop and save every figure on this page, tagging each with its caption.

    Uses the page's already-computed layout detections — no new model call.

    Args:
        page: the PDF page (for its text layer).
        page_image: the padded render the detections were made against.
        detections: this page's layout detections, or None when layout was
            never run (returns []).
        pdf_path: the source PDF, used to derive the document id.
        figures_base_dir: root directory for figure output; figures land in
            `<figures_base_dir>/<doc_id>/`.
        page_idx: 0-indexed page number.
        read_crop: optional model reader for captions on scanned pages.
        path_root: manifest paths are made relative to this when given.
        logger: optional logger.

    Returns:
        One `FigureAsset` per saved figure; [] when the page has none.
    """
    log = logger or logging.getLogger("Figures")
    if not detections:
        return []
    doc_id = safe_doc_id(Path(pdf_path).stem)
    out_dir = Path(figures_base_dir) / doc_id
    read_caption = make_caption_reader(page, page_image, read_crop=read_crop, logger=log)
    assets = extract_and_save_figures(
        page_image=page_image,
        detections=list(detections),
        out_dir=out_dir,
        page_number=page_idx + 1,
        read_caption=read_caption,
        path_root=path_root,
        logger=log,
    )
    return attach_printed_labels(page, page_image, assets, logger=log)


def attach_printed_labels(
    page: PdfPage,
    page_image: Image.Image,
    assets: List[FigureAsset],
    logger: Optional[logging.Logger] = None,
) -> List[FigureAsset]:
    """
    Label each crop from the page's own text, and rejoin panels of one figure.

    For biology researchers: the detector finds panels, not figures, and only
    the panel touching the caption gets it. This reads the caption lines
    straight out of the text layer — which has them even on pages where the
    detector saw no caption at all — and gives every crop of one printed
    figure the same label, so asking a RAG for "Figure 11" returns all three
    of its panels rather than one.

    A label the caption already supplied is never overwritten: the caption box
    was paired geometrically and is the stronger evidence. Measured over the
    corpus, this pass agrees with all but two of those and contradicts none.

    Args:
        page: the PDF page, for its text layer and dimensions.
        page_image: the render the crop boxes are expressed in, for scaling.
        assets: the figures saved for this page.
        logger: optional logger.

    Returns:
        The same assets, with `printed_label`, `label_key`, `panel_group_size`
        and (only where it was empty) `caption` filled in.
    """
    log = logger or logging.getLogger("Figures")
    if not assets:
        return assets

    try:
        width, height = page.size
        scale = width / page_image.width if page_image.width else 1.0
        words = [(w.text, w.rect.x0, w.rect.y0, w.rect.x1, w.rect.y1)
                 for w in page.words()]
        label_lines = find_label_lines(build_text_lines(words, width))
        boxes = [tuple(value * scale for value in asset.bbox) for asset in assets]
        assignments = assign_labels_on_page(boxes, label_lines, height, logger=log)
    except Exception as error:                              # pragma: no cover
        # A page with no text layer is a scan; it simply has no labels to
        # read. That must never cost us the figures themselves.
        log.warning("Figure label pass failed on page %d (%s); crops keep "
                    "whatever their captions gave them.",
                    assets[0].page, error)
        return assets

    for asset, assignment in zip(assets, assignments):
        asset.panel_group_size = max(1, assignment.shared_with)
        if asset.printed_label or not assignment.printed:
            continue
        asset.printed_label = assignment.printed
        asset.label_key = assignment.key
        if not asset.caption:
            # The detector gave this crop no caption at all. The caption's
            # first line is a poor substitute for the whole caption, but it
            # is what the page actually says, and it beats an empty string.
            asset.caption = assignment.caption
        log.debug("p%d %s -> %s", asset.page, asset.figure_id, asset.printed_label)

    return assets


def make_caption_reader(
    page: PdfPage,
    page_image: Image.Image,
    read_crop: Optional[CropReader] = None,
    logger: Optional[logging.Logger] = None,
) -> Callable[[Optional[Tuple[int, int, int, int]]], str]:
    """
    Build the caption reader for one page: box (page-image pixels) -> text.

    Born-digital pages read the caption verbatim from the text layer
    (authoritative, free, no fabrication). A scanned page has no usable text
    layer, so it falls back to the extraction model reading the caption crop —
    when a `read_crop` was provided.
    """
    log = logger or logging.getLogger("Figures")

    def _read(box_px: Optional[Tuple[int, int, int, int]]) -> str:
        if not box_px:
            return ""
        rect = image_box_to_pdf_rect(box_px, page, page_image)
        text = page.text_in(rect).strip()
        if text:
            return text
        if read_crop is not None and is_scanned_page(page):
            crop = page_image.crop(box_px)
            return (read_crop(crop) or "").strip()
        return ""

    return _read


def image_box_to_pdf_rect(
    box_px: Tuple[int, int, int, int],
    page: PdfPage,
    page_image: Image.Image,
    padding: int = RENDER_PADDING_PX,
) -> Rect:
    """
    Map a box in rendered-page-image pixels back to PDF points.

    The page image comes from `rendering.render_page`: uniform scale plus a
    fixed white border. The scale is derived from the actual content width so
    DPI rounding cannot drift; the padding is subtracted; a small point-slop is
    added so a caption line intersecting the box is captured.
    """
    width_pts, _ = page.size
    content_width = max(1, page_image.width - 2 * padding)
    scale = content_width / width_pts  # pixels per PDF point, uniform
    x0, y0, x1, y1 = box_px
    return Rect(
        (x0 - padding) / scale - CAPTION_CLIP_SLOP_PTS,
        (y0 - padding) / scale - CAPTION_CLIP_SLOP_PTS,
        (x1 - padding) / scale + CAPTION_CLIP_SLOP_PTS,
        (y1 - padding) / scale + CAPTION_CLIP_SLOP_PTS,
    )


def write_figure_manifest(
    pdf_path: Path | str,
    figures_base_dir: Path,
    figures: Sequence[dict],
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """
    Write the document-level figure manifest (JSON) the RAG consumes.

    One flat list of {figure_id, image_path, caption, page, bbox}. Nothing is
    written when the document has no figures.

    Args:
        pdf_path: the source PDF, used to derive the document id.
        figures_base_dir: root figure directory (same as extraction used).
        figures: every page's figure dicts, already flattened, in page order.
        logger: optional logger.

    Returns:
        The manifest path, or None when there was nothing to write or the
        write failed (a missing manifest must not fail the document).
    """
    log = logger or logging.getLogger("Figures")
    if not figures:
        return None
    doc_id = safe_doc_id(Path(pdf_path).stem)
    manifest_dir = Path(figures_base_dir) / doc_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    try:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"document": doc_id, "figure_count": len(figures),
                 "figures": list(figures)},
                handle, indent=2, ensure_ascii=False,
            )
        emit(log, "stage", stage="figures",
             message=f"Figure manifest: {len(figures)} figure(s) -> {manifest_path}",
             count=len(figures), manifest=str(manifest_path))
        return manifest_path
    except Exception as error:
        log.warning("Could not write figure manifest: %s", error)
        return None
