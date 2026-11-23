"""
Display equations on text-layer pages (T35): read from their crop, not the text layer.

For biology researchers: a page of plain prose is read straight from the
PDF's own text, with no model (fast and exact). But some PDFs draw their
equations in a font whose characters are mapped to ordinary letters, so
the text layer turns "∂t md = Dm ∇²m md + Rd" into "qtmd ¼ Dm r 2 mmd þRd".
Nothing on the page says "equation" to the page classifier, so the page
goes the free route and the equation is lost.

The layout model (PP-DocLayout, CPU) marks display equations as
`display_formula`. On a text-layer page this module:

1. runs the layout model on a small render of the page (~0.6 s);
2. for each display-formula box, finds the text-layer words inside it in
   the page text, as one unbroken run (only spaces and line breaks between
   them); a box whose words cannot be found that way is left alone;
3. reads the box's crop with the model's formula prompt and puts
   `$$<LaTeX>$$` in place of that run.

Every word outside a box stays exactly where it was, and a read that
fails or comes back empty leaves the page text as it was: the step can
only replace an equation's garbled text, never lose prose.

Entry points:
    formula_boxes(page, detector)                         -> boxes in PDF points
    with_display_formulas(text, page, boxes, read_crop)   -> (new text, records)
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

from ..pdf.base import PdfPage, Rect

#: The layout model's label for an equation set on its own line(s).
LABEL_DISPLAY_FORMULA = "display_formula"
#: Longest edge of the render the layout model looks at. The model resizes
#: its input to 800 x 800, so a larger render costs time and buys nothing.
DETECTION_EDGE_PX = 1024
#: White kept around an equation's box when it is cropped for reading, in
#: PDF points (sub- and superscripts often touch the detected box).
CROP_MARGIN_PT = 2.0


def formula_boxes(page: PdfPage, detector) -> List[Rect]:
    """The page's display-formula boxes, in PDF points (top-left origin)."""
    image = page.render(max_edge=DETECTION_EDGE_PX)
    width_pt, _ = page.size
    scale = image.width / width_pt                       # pixels per point
    return [Rect(*(v / scale for v in detection.box_px))
            for detection in detector.detect(image, page=page)
            if detection.label == LABEL_DISPLAY_FORMULA]


def with_display_formulas(
    text: str,
    page: PdfPage,
    boxes: Sequence[Rect],
    read_crop: Callable[[Rect], Optional[str]],
) -> Tuple[str, List[dict]]:
    """The page text with each display formula's run replaced by its LaTeX.

    Args:
        text: the page's text-layer text (`page.text()`).
        page: the page, for its word boxes.
        boxes: display-formula boxes in PDF points (from `formula_boxes`).
        read_crop: reads one box (PDF points) and returns the model's text;
            None or "" when the read failed.

    Returns:
        The new text, and one record per box: {"bbox": [x0, y0, x1, y1],
        "status": "read" | "not_in_text" | "empty_read"}. Boxes are read only
        when their run was found, so a box that cannot be placed costs no
        model call.
    """
    words = page.words()
    replacements, records = [], []
    for box in boxes:
        record = {"bbox": [round(v, 1) for v in (box.x0, box.y0, box.x1, box.y1)]}
        records.append(record)
        span = run_of_words(text, words_inside(words, box))
        if span is None or overlaps(span, replacements):
            record["status"] = "not_in_text"
            continue
        latex = clean_latex(read_crop(padded(box, page)))
        if not latex:
            record["status"] = "empty_read"
            continue
        record["status"] = "read"
        replacements.append((span[0], span[1], f"$${latex}$$"))
    return splice(text, replacements), records


# ---------------------------------------------------------------------------
# Helpers (after the main functions, per the project rule)
# ---------------------------------------------------------------------------

def words_inside(words, box: Rect) -> List[str]:
    """The words whose centre lies inside the box, in the page's word order."""
    inside = []
    for word in words:
        x = (word.rect.x0 + word.rect.x1) / 2
        y = (word.rect.y0 + word.rect.y1) / 2
        if box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1:
            inside.append(word.text)
    return [w for w in inside if w.strip()]


def run_of_words(text: str, words: Sequence[str]) -> Optional[Tuple[int, int]]:
    """(start, end) of the first place the words follow each other in the
    text with only whitespace between them; None when there is none."""
    if not words:
        return None
    gap = r"\s*"
    pattern = gap.join(re.escape(w) for w in words)
    match = re.search(pattern, text)
    return (match.start(), match.end()) if match else None


def overlaps(span: Tuple[int, int], replacements: Sequence[tuple]) -> bool:
    """Whether a run overlaps one already being replaced (two boxes, one run)."""
    return any(span[0] < end and start < span[1] for start, end, _ in replacements)


def padded(box: Rect, page: PdfPage) -> Rect:
    """The box with CROP_MARGIN_PT around it, kept on the page."""
    width, height = page.size
    return Rect(max(0.0, box.x0 - CROP_MARGIN_PT), max(0.0, box.y0 - CROP_MARGIN_PT),
                min(width, box.x1 + CROP_MARGIN_PT), min(height, box.y1 + CROP_MARGIN_PT))


def clean_latex(read: Optional[str]) -> str:
    """The model's formula read without its $$ / \\[ \\] wrapping; one line."""
    latex = (read or "").strip()
    latex = re.sub(r"^\$\$|\$\$$|^\\\[|\\\]$|^\$|\$$", "", latex).strip()
    return re.sub(r"\s*\n\s*", " ", latex)


def splice(text: str, replacements: Sequence[tuple]) -> str:
    """Apply (start, end, new) replacements, last first so offsets hold."""
    for start, end, new in sorted(replacements, key=lambda r: r[0], reverse=True):
        text = text[:start] + new + text[end:]
    return text
