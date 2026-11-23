"""
Docling JSON (DoclingDocument) -> common format (protocol §3).

For biology researchers: Docling stores every text item with a label (text,
list_item, section_header, caption, page_header, ...) and the reading order
as a tree under `body`. A caption belongs to a picture or table either through
the item's `captions` list or because the caption's `parent` is that item;
both are Docling's own links. Docling does not extract authors, DOI, year,
journal, an abstract field, references or in-text links: those stay
"not supported".
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import (cells_to_html, collapse_figures, crop, empty_document,
                     figure_entry, formulas_of_pieces, table_entry)

BODY_LABELS = {"text", "paragraph", "list_item"}


def convert(json_path: Path, paper_id: str) -> dict:
    """Read one Docling JSON into the common format."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    items = index_items(data)
    document = empty_document(paper_id, "docling", "2.151.1", "offline")
    titles = [t["text"] for t in data.get("texts", []) if t.get("label") == "title"]
    document["metadata"]["title"] = titles[0] if titles else None
    document["body_paragraphs"] = [items[ref]["text"] for ref in reading_order(data, items)
                                   if items[ref].get("label") in BODY_LABELS
                                   and items[ref].get("text")]
    figures, unlabelled = [], []
    for picture in data.get("pictures", []):
        caption = caption_of(picture, data, items)
        entry = figure_entry(picture["self_ref"], caption, crops=picture_crops(picture, data))
        if entry is None:
            unlabelled.append(caption)
        else:
            figures.append(entry)
    document["figures"] = collapse_figures(figures)
    document["unlabelled_figure_objects"] = unlabelled
    # A11: items labelled "formula" are display formulas, in reading order.
    document["formulas"] = formulas_of_pieces(
        [("display" if items[ref].get("label") == "formula" else "text", items[ref].get("text") or "")
         for ref in reading_order(data, items)
         if items[ref].get("label") in BODY_LABELS | {"formula"}])
    document["tables"] = [table_entry(t["self_ref"], caption_of(t, data, items),
                                      table_html(t)) for t in data.get("tables", [])]
    return document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def index_items(data: dict) -> dict:
    items = {}
    for collection in ("texts", "groups", "pictures", "tables"):
        for item in data.get(collection, []):
            items[item["self_ref"]] = item
    return items


def reading_order(data: dict, items: dict) -> list[str]:
    """Depth-first walk of body's children; pictures/tables are not entered."""
    order: list[str] = []

    def walk(ref: str) -> None:
        item = items.get(ref)
        if item is None:
            return
        order.append(ref)
        if ref.startswith("#/pictures") or ref.startswith("#/tables"):
            return
        for child in item.get("children", []):
            walk(child["$ref"])

    for child in data.get("body", {}).get("children", []):
        walk(child["$ref"])
    return order


def picture_crops(picture: dict, data: dict) -> list | None:
    """A11: each provenance box in PDF points, turned to a top-left origin
    with the page height Docling records."""
    boxes = []
    for prov in picture.get("prov") or []:
        box, page_no = prov.get("bbox") or {}, prov.get("page_no")
        height = ((data.get("pages") or {}).get(str(page_no)) or {}).get("size", {}).get("height")
        if not box or not page_no or height is None:
            continue
        if box.get("coord_origin", "BOTTOMLEFT").upper() == "BOTTOMLEFT":
            top, bottom = height - box["t"], height - box["b"]
        else:
            top, bottom = box["t"], box["b"]
        boxes.append(crop(page_no - 1, (box["l"], min(top, bottom), box["r"], max(top, bottom))))
    return boxes or None


def caption_of(item: dict, data: dict, items: dict) -> str:
    refs = [c["$ref"] for c in item.get("captions", [])]
    refs += [t["self_ref"] for t in data.get("texts", [])
             if t.get("label") == "caption"
             and t.get("parent", {}).get("$ref") == item["self_ref"]
             and t["self_ref"] not in refs]
    return " ".join(items[r]["text"] for r in refs if r in items)


def table_html(table: dict) -> str | None:
    cells = (table.get("data") or {}).get("table_cells") or []
    if not cells:
        return None
    return cells_to_html([{"row": c["start_row_offset_idx"],
                           "col": c["start_col_offset_idx"],
                           "rowspan": c.get("row_span", 1),
                           "colspan": c.get("col_span", 1),
                           "text": c.get("text", "")} for c in cells])
