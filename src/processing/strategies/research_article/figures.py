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
from ...utils.text_normalization import normalize_harvested_text
from ...utils.figure_extractor import FigureAsset, extract_and_save_figures, safe_doc_id
from ...utils.figure_label_pass import (
    assign_labels_on_page,
    build_text_lines,
    find_label_lines,
)
from ...utils.figure_labels import parse_printed_label
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
        text_lines = build_text_lines(words, width)
        label_lines = find_label_lines(text_lines)
        boxes = [tuple(value * scale for value in asset.bbox) for asset in assets]
        assignments = assign_labels_on_page(boxes, label_lines, height, logger=log,
                                            text_lines=text_lines)
    except Exception as error:                              # pragma: no cover
        # A page with no text layer is a scan; it simply has no labels to
        # read. That must never cost us the figures themselves.
        log.warning("Figure label pass failed on page %d (%s); crops keep "
                    "whatever their captions gave them.",
                    assets[0].page, error)
        return assets

    for asset, assignment in zip(assets, assignments):
        asset.panel_group_size = max(1, assignment.shared_with)
        if not assignment.printed:
            continue
        if not asset.printed_label:
            asset.printed_label = assignment.printed
            asset.label_key = assignment.key
        repaired = _repair_caption(asset, assignment)
        if repaired is not None:
            # This is the SECOND place a caption is stored: the first is
            # `figure_extractor._build_asset`. Both read the text layer, so
            # both need the control-character repair — normalising only the
            # other one left two damaged captions in a full corpus run, which
            # is how half a fix ships (compare the ledger literal that was
            # once defined in two places).
            asset.caption = normalize_harvested_text(repaired)
        log.debug("p%d %s -> %s", asset.page, asset.figure_id, asset.printed_label)

    return assets


def _repair_caption(asset, assignment) -> Optional[str]:
    """
    The caption this crop should carry, or None to keep what it has.

    For biology researchers: the layout detector draws a box around what it
    believes is the caption, and that box is right most of the time. Three
    times it is not, and each failure is worse than an empty caption, because
    nothing downstream can tell a broken caption from a good one:

    1. **Nothing at all** — the box was never found.
    2. **A panel letter** — the box caught the "d" printed inside a panel.
    3. **A caption missing its own opening** — MEASURED: 26 of this corpus's
       107 captions begin part way down, so the sentence naming the figure
       ("Fig. 2.2 Fibril-forming collagens and their assembly") is absent and
       the caption cannot be found by its own subject.

    THE TWO REPAIRS ARE NOT THE SAME, and using one instrument for both was
    measured and rejected. Where a caption already exists (case 3) the stored
    text is CLEAN and merely headless, so only the missing opening is put
    back. Replacing it wholesale with the text-layer block loses badly: on the
    `Collagen structure` review's page 19 that trades 1343 characters for 44,
    and on the tendon volume's two-column captions the block comes back with
    the columns interleaved. Where there is no caption to protect (cases 1 and
    2) the whole block is taken, because a jumbled caption still beats "d".

    Returns:
        The caption to store, or None when the stored one is already right.
    """
    if not assignment.caption:
        return None
    if asset.printed_label and asset.label_key != assignment.key:
        return None                        # two passes, two different figures

    # Cases 1 and 2: nothing worth keeping, so take the whole caption.
    if not asset.caption or _is_panel_sublabel(asset.caption):
        return assignment.caption

    # Case 3: put back only the opening, and only when that is what is wrong —
    # a caption already beginning at its own label is never touched.
    stored_label, _ = parse_printed_label(asset.caption.strip())
    opening_label, _ = parse_printed_label(assignment.opening.strip())
    if stored_label is not None or opening_label is None:
        return None
    head = _without_overlap(assignment.opening.strip(), asset.caption.strip())
    if not head:
        return None
    return f"{head} {asset.caption.strip()}"


#: The shortest run of characters that counts as a real overlap between the
#: recovered opening and the stored caption, rather than a coincidence. Two
#: captions sharing eight consecutive characters at exactly this seam are the
#: same text; sharing two ("a ") are not.
MIN_OVERLAP_CHARS = 8


def _without_overlap(opening: str, stored: str) -> str:
    """
    The opening with any text the stored caption already repeats cut off.

    For biology researchers: on a page whose caption is set in TWO COLUMNS,
    the line reconstruction reads straight across the gutter, so the caption's
    first line arrives with the second column's words mixed into it -- "Fig.
    2.2 Fibril-forming collagens and their assembly. respective overlaps and
    gaps in the fbril." Those trailing words are exactly where the stored
    caption begins, so pasting the two together says everything twice.

    This finds the longest tail of the opening that the stored caption already
    starts with, and removes it, keeping only the genuinely missing head.
    """
    limit = min(len(opening), len(stored))
    for length in range(limit, MIN_OVERLAP_CHARS - 1, -1):
        if stored.startswith(opening[-length:]):
            return opening[:-length].strip()
    return opening



#: A caption box holding only a PANEL SUB-LABEL -- "d", "( A )", "f" -- rather
#: than a caption. Measured over the 9-document corpus: the detector returns
#: one of these for the three crops of the ice-plant figure on
#: `Hydration-induced` page 7, which is how a figure ends up correctly labelled
#: "Fig. 4" and captioned "d". Length is the discriminator, and it is set well
#: below the shortest real caption in the corpus ("Helical twisting", 16
#: characters) so no genuine caption can be mistaken for one.
MAX_PANEL_SUBLABEL_CHARS = 15


def _is_panel_sublabel(caption: str) -> bool:
    """
    True when a stored caption is really a panel letter, not a caption.

    For biology researchers: the layout detector often hands back the little
    "(a)" or "d" printed inside a panel instead of the figure's caption. That
    string is worse than nothing -- it looks like a caption to everything
    downstream -- so it is replaced by the caption the text layer actually
    holds.

    A caption that begins with its own figure label ("Fig. 4 | Hygroscopic
    ...") is never a sub-label however short it is, so that case is excluded
    explicitly rather than left to the length test.
    """
    stripped = caption.strip()
    if not stripped or len(stripped) > MAX_PANEL_SUBLABEL_CHARS:
        return False
    # `parse_printed_label` always returns a PAIR, and signals "no label" with
    # (None, None) rather than with None itself — so the printed half is what
    # has to be tested. Testing the pair is silently always-false.
    printed, _ = parse_printed_label(stripped)
    return printed is None


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
