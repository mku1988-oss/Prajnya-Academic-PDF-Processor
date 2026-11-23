"""
The two measures amendment A11 adds, for one tool on one paper.

For biology researchers: `score_a11` gives the formula scores (display and
inline apart) and the figure-crop scores. Figure crops need each key
figure's true box, found once per paper by `ground_truth/locate_key_figures.py`
and stored in `outputs/key_boxes/<PMCID>.json`; a figure whose best match
is below LOCATE_THRESHOLD counts as "not located" and is left out (and
counted).
"""
from __future__ import annotations

from .figure_locator import Location, score_crops
from .formulas import match_formulas, score_formulas

#: Fixed on the 10 pilots before any tool is scored (A11 §6). None until then:
#: crop scores stay None ("not measured") rather than use an unchecked value.
LOCATE_THRESHOLD: float | None = 0.6   # fixed on the pilots: results/A11_locator_pilot_check.md

#: Tools that give a crop box per figure. GROBID gives figure coordinates
#: only for bitmaps and saves no crop: its crops are "not supported".
CROPPING_TOOLS = {"prajnya", "prajnya_online", "docling", "docling_formula", "marker", "mineru"}


def score_a11(key: dict, tool: dict, boxes: dict | None) -> dict:
    """Formula and crop scores; None where not supported or not applicable."""
    result = formula_scores(key, tool)
    result.update(crop_scores(key, tool, boxes))
    return result


def formula_scores(key: dict, tool: dict) -> dict:
    """Display and inline formulas apart; only key formulas with MathML count."""
    scored = [f for f in key.get("formulas") or [] if f.get("mathml")]
    if tool.get("formulas") is None:                     # tool writes no formulas
        return {"formula_display": None, "formula_inline": None,
                "formula_display_count": 0, "formula_inline_count": 0}
    return score_formulas(scored, match_formulas(scored, tool["formulas"]))


def crop_scores(key: dict, tool: dict, boxes: dict | None) -> dict:
    """Mean crop IoU and share right over the located key figures (by label)."""
    empty = {"crop_iou": None, "crop_right": None, "crops_scored": 0, "figures_unlocated": None}
    if boxes is None or LOCATE_THRESHOLD is None or tool.get("tool") not in CROPPING_TOOLS:
        return empty
    located, unlocated = {}, 0
    for figure in key.get("figures") or []:
        found = (boxes.get("figures") or {}).get(figure["id"])
        if found and found["score"] >= LOCATE_THRESHOLD:
            located[figure["id"]] = Location(found["page_index"], tuple(found["box"]), found["score"])
        else:
            unlocated += 1
    number_of = {f["id"]: f["label_number"] for f in key.get("figures") or []}
    by_number = {f["label_number"]: f for f in tool.get("figures") or []}
    tool_crops = {}
    for figure_id, where in located.items():
        figure = by_number.get(number_of[figure_id])
        crops = [in_points(c, boxes["page_sizes"]) for c in (figure or {}).get("crops") or []]
        on_page = [box for page, box in crops if page == where.page_index]
        if on_page:                       # several crops of one figure: their joint box
            tool_crops[figure_id] = (where.page_index, joint_box(on_page))
    result = score_crops(located, tool_crops)
    result["figures_unlocated"] = unlocated
    return result


# ---------------------------------------------------------------------------
# Helpers (after the main functions, per the project rule)
# ---------------------------------------------------------------------------

def in_points(crop: dict, page_sizes: list) -> tuple:
    """(page_index, box in PDF points) for a crop given in points or page fractions."""
    page, box = crop["page_index"], crop["box"]
    if crop.get("unit") == "fraction" and 0 <= page < len(page_sizes):
        width, height = page_sizes[page]
        box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
    return page, tuple(box)


def joint_box(boxes: list) -> tuple:
    """The smallest box holding them all."""
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
