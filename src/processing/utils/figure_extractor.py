"""
Figure Extractor — save each figure as a picture and tag it with its caption, for the RAG.

For biology researchers: a research article's figures (micrographs, plots, schematics) carry
data that prose cannot. Earlier we asked a vision model to *describe* every figure in words —
but that was slow and it sometimes invented details (wrong legend colours, back-to-front
conclusions). For a multimodal Retrieval-Augmented-Generation index we do not want a lossy
word-picture; we want the FIGURE ITSELF. So this module crops each figure out of the rendered
page, saves it as a PNG, and reads its caption verbatim from the digital text layer. The image
and its true caption are then retrievable together — no fabrication, no extra model pass.

Where the boxes come from: the layout model's `<|grounding|>` call ALREADY emits `image` and
`image_caption` boxes in the same response we use to find dropped tables. We simply consume the
figure labels too (see table_region_detector.locate_figure_regions). This module therefore
costs no additional VLM call.

VLM-AGNOSTIC: this module names no model. The figure/caption label vocabulary lives in the
`GroundingSpec` (from the layout model's config); reading a caption is delegated to a
caller-supplied `read_caption` callable (the caller binds it to the digital text layer, with a
scanned fallback to the extraction model). Swapping models is a config change, not a code change.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image

from .table_region_detector import FigureRegion, GroundingSpec, locate_figure_regions

# A caller-supplied caption reader: given a caption box in page-image pixels (or None), return the
# caption text. The caller maps pixels -> text layer (born-digital) with a scanned fallback, so
# this module stays free of any coordinate/model detail.
CaptionReader = Callable[[Optional[Tuple[int, int, int, int]]], str]

# Inline anchor spliced into the page text where a figure sits, linking text <-> image <-> caption
# for the multimodal RAG. Format: [[FIGURE <id> | <caption> | <relative image path>]].
FIGURE_ANCHOR_MARKER = "<!-- figure crop (layout grounding) -->"


@dataclass
class FigureAsset:
    """
    One saved figure, ready for the multimodal RAG index.

    Attributes:
        figure_id: stable id, e.g. "p4_fig1" (page 4, first figure).
        image_path: path to the saved PNG (relative to the run output root when possible).
        caption: the figure's caption text, verbatim from the text layer ("" if none found).
        page: 1-indexed page number the figure sits on.
        bbox: (x0, y0, x1, y1) in pixels of the rendered page image (the crop box).
    """
    figure_id: str
    image_path: str
    caption: str
    page: int
    bbox: Tuple[int, int, int, int]

    def to_dict(self) -> dict:
        """Plain dict for page metadata and the document-level manifest (JSON-friendly)."""
        return {
            "figure_id": self.figure_id,
            "image_path": self.image_path,
            "caption": self.caption,
            "page": self.page,
            "bbox": list(self.bbox),
        }

    def anchor(self) -> str:
        """The inline anchor text linking this figure into the page's reading flow."""
        caption = " ".join(self.caption.split()) if self.caption else ""
        return f"[[FIGURE {self.figure_id} | {caption} | {self.image_path}]]"


def extract_and_save_figures(
    page_image: Image.Image,
    grounding_output: str,
    spec: GroundingSpec,
    out_dir: Path,
    page_number: int,
    read_caption: CaptionReader,
    path_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> List[FigureAsset]:
    """
    Crop and save every figure on a page, tagging each with its caption.

    Args:
        page_image: the rendered page image the layout model saw (source of the crops).
        grounding_output: the layout model's raw grounding response for this page.
        spec: the layout model's grounding format (labels + scale), from its config.
        out_dir: directory to write the figure PNGs into (created if missing).
        page_number: 1-indexed page number, used in the figure id and asset record.
        read_caption: callable mapping a caption box (pixels, or None) to caption text.
        path_root: if given, image paths in the returned assets are made relative to it (so the
            manifest carries portable paths like "figures/<doc>/p4_fig1.png"); otherwise the
            absolute path is stored.
        logger: optional logger.

    Returns:
        One FigureAsset per figure saved, in page order. Empty when the page has no figure or the
        detector lacks figure support.
    """
    log = logger or logging.getLogger("FigureExtractor")

    regions = locate_figure_regions(grounding_output, page_image, spec, logger=log)
    if not regions:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    assets = []
    for index, region in enumerate(regions, start=1):
        try:
            asset = _save_one_figure(
                page_image, region, out_dir, page_number, index, read_caption, path_root, log
            )
            assets.append(asset)
        except Exception as error:
            log.warning("Figure %d on page %d could not be saved: %s", index, page_number, error)

    log.info("Page %d: saved %d figure(s)", page_number, len(assets))
    return assets



def _save_one_figure(
    page_image: Image.Image,
    region: FigureRegion,
    out_dir: Path,
    page_number: int,
    index: int,
    read_caption: CaptionReader,
    path_root: Optional[Path],
    log: logging.Logger,
) -> FigureAsset:
    """Crop one figure, write its PNG, read its caption, and assemble the FigureAsset."""
    figure_id = f"p{page_number}_fig{index}"
    crop = page_image.crop(region.image_box)

    image_path = out_dir / f"{figure_id}.png"
    crop.save(image_path, format="PNG")

    caption = ""
    try:
        caption = (read_caption(region.caption_box) or "").strip()
    except Exception as error:
        log.warning("Caption read failed for %s: %s", figure_id, error)

    stored_path = _portable_path(image_path, path_root)
    return FigureAsset(
        figure_id=figure_id,
        image_path=stored_path,
        caption=caption,
        page=page_number,
        bbox=tuple(region.image_box),
    )


def _portable_path(image_path: Path, path_root: Optional[Path]) -> str:
    """Return the image path relative to path_root when possible, else its absolute form."""
    if path_root is not None:
        try:
            return image_path.resolve().relative_to(path_root.resolve()).as_posix()
        except ValueError:
            pass  # not under path_root -> fall back to absolute
    return image_path.as_posix()


def append_figure_anchors(page_text: str, assets: List[FigureAsset]) -> str:
    """
    Append an inline anchor for each figure to the page text, behind an auditable marker.

    The anchors let the RAG link a figure's image and caption back to the page's text. They are
    appended (not re-inserted at the exact reading position) for the same reason recovered tables
    are: it restores the link losslessly without disturbing the prose the reader got right.
    """
    if not assets:
        return page_text
    blocks = [page_text.rstrip(), FIGURE_ANCHOR_MARKER]
    blocks.extend(asset.anchor() for asset in assets)
    return "\n".join(blocks)


_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def safe_doc_id(name: str) -> str:
    """Filesystem-safe document id from a PDF stem (for the per-document figures subdir)."""
    cleaned = _SAFE_ID.sub("_", name).strip("_")
    return cleaned or "document"
