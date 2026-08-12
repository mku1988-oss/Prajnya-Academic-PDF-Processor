#!/usr/bin/env python3
"""
Figure Extractor — save each figure as a picture, tagged with its caption.

For biology researchers: a research article's figures (micrographs, plots,
schematics) carry data that prose cannot. An earlier design asked a vision
model to *describe* every figure in words — that was slow and it sometimes
invented details, getting legend colours wrong or conclusions backwards. For a
multimodal Retrieval-Augmented-Generation index we do not want a lossy
word-picture; we want the FIGURE ITSELF.

So this module crops each figure out of the rendered page, saves it as a PNG,
and reads its caption verbatim from the digital text layer. The image and its
true caption become retrievable together — no fabrication, and no extra model
pass, because the layout detector already emitted `image` and `image_caption`
boxes in the same pass used to find dropped tables.

VLM-AGNOSTIC: this module names no model. Reading a caption is delegated to a
caller-supplied `read_caption` callable, which the caller binds to the digital
text layer (with a scanned-page fallback to the extraction model).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from src.processing.layout.base import Detection
from src.processing.layout.regions import FigureRegion, locate_figure_regions
from src.processing.utils.figure_labels import parse_printed_label

#: A caller-supplied caption reader: given a caption box in page-image pixels
#: (or None), return the caption text. The caller maps pixels to the text layer,
#: so this module stays free of any coordinate or model detail.
CaptionReader = Callable[[Optional[Tuple[int, int, int, int]]], str]

#: Marks the block of inline figure anchors appended to a page's text.
FIGURE_ANCHOR_MARKER = "<!-- figure crop (layout detection) -->"


@dataclass
class FigureAsset:
    """
    One saved figure, ready for the multimodal RAG index.

    Attributes:
        figure_id: stable id, e.g. "p4_fig1" (page 4, first figure).
        image_path: path to the saved PNG, relative to the run root when possible.
        caption: caption text, verbatim from the text layer ("" if none found).
        page: 1-indexed page number the figure sits on.
        bbox: (x0, y0, x1, y1) in pixels of the rendered page — the crop box.
        printed_label: what the PAPER calls this figure ("Fig. 2.1"), as
            printed, or None when the caption carried no label. `figure_id`
            says where the figure sits; this says what to call it, and the two
            disagree for 45 of 107 figures in the sample corpus.
        label_key: the same label normalized for matching ("figure 2.1"), so a
            body-text mention of "Fig. 2.1" joins to this crop regardless of
            which spelling either side used.
    """

    figure_id: str
    image_path: str
    caption: str
    page: int
    bbox: Tuple[int, int, int, int]
    printed_label: Optional[str] = None
    label_key: Optional[str] = None
    panel_group_size: int = 1

    def to_dict(self) -> dict:
        """Plain dict for page metadata and the document manifest (JSON-friendly)."""
        return {
            "figure_id": self.figure_id,
            "image_path": self.image_path,
            "caption": self.caption,
            "page": self.page,
            "bbox": list(self.bbox),
            "printed_label": self.printed_label,
            "label_key": self.label_key,
            "panel_group_size": self.panel_group_size,
        }

    def anchor(self) -> str:
        """The inline anchor linking this figure into the page's reading flow."""
        caption = " ".join(self.caption.split()) if self.caption else ""
        return f"[[FIGURE {self.figure_id} | {caption} | {self.image_path}]]"


def extract_and_save_figures(
    page_image: "Image.Image",
    detections: Sequence[Detection],
    out_dir: Path,
    page_number: int,
    read_caption: CaptionReader,
    path_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> List[FigureAsset]:
    """
    Crop and save every figure on a page, tagging each with its caption.

    Args:
        page_image: the rendered page the detections were made against.
        detections: this page's layout detections, from any `LayoutDetector`.
            V5 took a raw grounding string plus a spec here; the parse now
            belongs to the detector.
        out_dir: directory to write the figure PNGs into (created if missing).
        page_number: 1-indexed page number, used in the figure id.
        read_caption: callable mapping a caption box (pixels, or None) to text.
        path_root: when given, image paths are made relative to it so the
            manifest carries portable paths like "figures/<doc>/p4_fig1.png".
        logger: optional logger.

    Returns:
        One `FigureAsset` per figure saved, in page order. Empty when the page
        has no figure.
    """
    log = logger or logging.getLogger("FigureExtractor")

    regions = locate_figure_regions(detections, page_image, logger=log)
    if not regions:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    assets: List[FigureAsset] = []
    for index, region in enumerate(regions, start=1):
        try:
            assets.append(_save_one_figure(
                page_image, region, out_dir, page_number, index,
                read_caption, path_root, log,
            ))
        except Exception as error:
            # One unsaveable figure must not cost the whole page.
            log.warning("Figure %d on page %d could not be saved: %s",
                        index, page_number, error)

    log.info("Page %d: saved %d figure(s)", page_number, len(assets))
    return assets


def append_figure_anchors(page_text: str, assets: Sequence[FigureAsset]) -> str:
    """
    Append an inline anchor per figure to the page text, behind an audit marker.

    The anchors let the RAG link a figure's image and caption back to the
    page's text. They are appended rather than re-inserted at the exact reading
    position for the same reason recovered tables are: it restores the link
    losslessly without disturbing prose the reader already got right.
    """
    if not assets:
        return page_text
    blocks = [page_text.rstrip(), FIGURE_ANCHOR_MARKER]
    blocks.extend(asset.anchor() for asset in assets)
    return "\n".join(blocks)


_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def safe_doc_id(name: str) -> str:
    """Filesystem-safe document id from a PDF stem (for the figures subdirectory)."""
    cleaned = _SAFE_ID.sub("_", name).strip("_")
    return cleaned or "document"


# --- helpers (after the main functions, per CLAUDE.md) --------------------

def _save_one_figure(
    page_image: "Image.Image",
    region: FigureRegion,
    out_dir: Path,
    page_number: int,
    index: int,
    read_caption: CaptionReader,
    path_root: Optional[Path],
    log: logging.Logger,
) -> FigureAsset:
    """Crop one figure, write its PNG, read its caption, assemble the asset."""
    figure_id = f"p{page_number}_fig{index}"
    crop = page_image.crop(region.image_box)

    image_path = out_dir / f"{figure_id}.png"
    crop.save(image_path, format="PNG")

    caption = ""
    try:
        caption = (read_caption(region.caption_box) or "").strip()
    except Exception as error:
        # A missing caption degrades the RAG entry; it must not lose the image.
        log.warning("Caption read failed for %s: %s", figure_id, error)

    printed_label, label_key = parse_printed_label(caption)

    return FigureAsset(
        figure_id=figure_id,
        image_path=_portable_path(image_path, path_root),
        caption=caption,
        page=page_number,
        bbox=tuple(region.image_box),
        printed_label=printed_label,
        label_key=label_key,
    )


def _portable_path(image_path: Path, path_root: Optional[Path]) -> str:
    """The image path relative to `path_root` when possible, else absolute."""
    if path_root is not None:
        try:
            return image_path.resolve().relative_to(path_root.resolve()).as_posix()
        except ValueError:
            pass  # not under path_root -> fall back to absolute
    return image_path.as_posix()
