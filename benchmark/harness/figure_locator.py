"""
Figure crops (amendment A11 §6): where each figure really sits on its page.

For biology researchers: PMC keeps each figure's publisher image (the same
picture printed in the PDF, at its own resolution). Sliding that picture
over the rendered PDF pages at many sizes and keeping the best match gives
the figure's true box on the page, with no tool involved. A tool's crop is
then scored by how much it overlaps that box (IoU: shared area / combined
area; 1.0 = the same box). "Crop right" is IoU >= 0.8.

The match is OpenCV's normalised correlation (TM_CCOEFF_NORMED), on grey
pixels: a score near 1 is a clean find; below the threshold fixed on the
pilots, the figure counts as "not located" and is left out of the score.

Boxes are in PDF points, origin top-left: (x0, y0, x1, y1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CROP_RIGHT_IOU = 0.8
COARSE_WIDTH_PX = 700          # the page is searched at this width first
FINE_DPI = 100                 # then the best match is refined at this resolution
SCALE_STEPS = 40               # template widths tried between the two limits
MIN_WIDTH_SHARE = 0.15         # a figure is at least this share of the page width
MAX_WIDTH_SHARE = 1.0


@dataclass
class Location:
    page_index: int            # 0-based
    box: tuple                 # (x0, y0, x1, y1) in PDF points, top-left origin
    score: float               # match correlation, -1..1


def locate_figure(figure_image: Path | Image.Image, pages: list) -> Location | None:
    """The best match of one publisher figure image over the document's pages.

    `pages` is a list of (page_image: PIL.Image, page_width_pt, page_height_pt),
    as `render_pages` gives. None when the image cannot be matched at any
    size (e.g. it is larger than every page).
    """
    template = grey(load(figure_image))
    best = None
    for index, (picture, width_pt, height_pt) in enumerate(pages):
        found = best_match(template, grey(picture))
        if found and (best is None or found[0] > best[0]):
            best = (*found, index, picture.width / width_pt)
    if best is None:
        return None
    score, (x, y, w, h), index, px_per_pt = best
    box = (x / px_per_pt, y / px_per_pt, (x + w) / px_per_pt, (y + h) / px_per_pt)
    return Location(page_index=index, box=box, score=float(score))


def crop_iou(true_box: tuple, crop_box: tuple) -> float:
    """Intersection over union of two boxes (x0, y0, x1, y1)."""
    ix = max(0.0, min(true_box[2], crop_box[2]) - max(true_box[0], crop_box[0]))
    iy = max(0.0, min(true_box[3], crop_box[3]) - max(true_box[1], crop_box[1]))
    shared = ix * iy
    union = area(true_box) + area(crop_box) - shared
    return shared / union if union > 0 else 0.0


def score_crops(located: dict, tool_crops: dict) -> dict:
    """Mean IoU and share of crops right, over the located key figures.

    `located`: key figure id -> Location (unlocated figures already left out).
    `tool_crops`: key figure id -> (page_index, box) of the tool's crop for
    that figure (matched by label before this call). A located figure the
    tool gave no crop for, or a crop on another page, scores IoU 0.
    """
    if not located:
        return {"crop_iou": None, "crop_right": None, "crops_scored": 0}
    ious = []
    for figure_id, where in located.items():
        crop = tool_crops.get(figure_id)
        same_page = crop is not None and crop[0] == where.page_index
        ious.append(crop_iou(where.box, crop[1]) if same_page else 0.0)
    return {"crop_iou": sum(ious) / len(ious),
            "crop_right": sum(i >= CROP_RIGHT_IOU for i in ious) / len(ious),
            "crops_scored": len(ious)}


def render_pages(pdf_path: Path, dpi: int = FINE_DPI) -> list:
    """Every page as (grey-able PIL image, width_pt, height_pt), via pypdfium2."""
    import pypdfium2 as pdfium
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        pages = []
        for page in document:
            width_pt, height_pt = page.get_size()
            picture = page.render(scale=dpi / 72).to_pil()
            pages.append((picture, width_pt, height_pt))
            page.close()
        return pages
    finally:
        document.close()


# ---------------------------------------------------------------------------
# Helpers (after the main functions, per the project rule)
# ---------------------------------------------------------------------------

def best_match(template: np.ndarray, page: np.ndarray):
    """(score, (x, y, w, h) in page pixels) of the best size and place, or None.

    Coarse: page shrunk to COARSE_WIDTH_PX, template tried at SCALE_STEPS
    widths. Fine: at full page resolution, sizes around the coarse best.
    """
    shrink = min(1.0, COARSE_WIDTH_PX / page.shape[1])
    small = cv2.resize(page, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    widths = np.linspace(MIN_WIDTH_SHARE, MAX_WIDTH_SHARE, SCALE_STEPS) * small.shape[1]
    coarse = best_over_widths(template, small, widths)
    if coarse is None:
        return None
    step = (widths[1] - widths[0]) / shrink
    centre = coarse[1][2] / shrink
    return best_over_widths(template, page, np.linspace(centre - step, centre + step, 9))


def best_over_widths(template: np.ndarray, page: np.ndarray, widths) -> tuple | None:
    best = None
    aspect = template.shape[0] / template.shape[1]
    for width in widths:
        w, h = int(round(width)), int(round(width * aspect))
        if w < 8 or h < 8 or w > page.shape[1] or h > page.shape[0]:
            continue
        resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(page, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(result)
        if np.isfinite(score) and (best is None or score > best[0]):
            best = (score, (x, y, w, h))
    return best


def load(image) -> Image.Image:
    """A PIL image from a path (jpg, png, webp) or an image already loaded."""
    return image if isinstance(image, Image.Image) else Image.open(image)


def grey(image: Image.Image) -> np.ndarray:
    """8-bit grey pixels; transparency is put on white, as on a printed page."""
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image)
    return np.asarray(image.convert("L"))


def area(box: tuple) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
