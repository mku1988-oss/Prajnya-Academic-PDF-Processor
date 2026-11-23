"""
MinerU content list -> common format (protocol §3).

For biology researchers: MinerU's "content list" is a flat list of blocks in
reading order: text (with `text_level` for headings), list (its
`sub_type` says whether it is the reference list), image, chart, table,
equation, header, footer, page_number. Images, charts and tables carry their
own caption lists. MinerU recognises reference-list entries but does not
split them into title, year or DOI, so its references are scored on
finding the entry only (raw text); the per-field scores are "not supported".
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import (collapse_figures, crop, empty_document, figure_entry,
                     formulas_of_pieces, html_to_text, math_pieces, table_entry)


def convert(content_list_path: Path, paper_id: str) -> dict:
    """Read one MinerU content list into the common format."""
    blocks = json.loads(content_list_path.read_text(encoding="utf-8"))
    document = empty_document(paper_id, "mineru", "4.2.3", "offline")
    document["body_paragraphs"] = body_of(blocks)
    figures, unlabelled = [], []
    for i, block in enumerate(blocks):
        if block.get("type") in ("image", "chart"):
            caption = join(block.get("image_caption") or block.get("chart_caption"))
            # A11: content-list boxes are on a 0-1000 scale of the page.
            box = [crop(block.get("page_idx", 0), [v / 1000 for v in block["bbox"]], "fraction")]                 if block.get("bbox") else None
            entry = figure_entry(f"b{i}", caption, crops=box)
            if entry is None:
                unlabelled.append(caption)
            else:
                figures.append(entry)
    document["figures"] = collapse_figures(figures)
    document["unlabelled_figure_objects"] = unlabelled
    document["formulas"] = formulas_of_pieces(formula_pieces(blocks))     # A11
    document["tables"] = [table_entry(f"b{i}", join(b.get("table_caption")),
                                      b.get("table_body") or None)
                          for i, b in enumerate(blocks) if b.get("type") == "table"]
    references = [item for b in blocks
                  if b.get("type") == "list" and b.get("sub_type") == "ref_text"
                  for item in b.get("list_items") or []]
    document["references"] = [{"id": f"r{n}", "position": n, "title": None,
                               "year": None, "doi": None,
                               "first_author_surname": None, "raw": html_to_text(raw),
                               "parsed": False}
                              for n, raw in enumerate(references, start=1)] or None
    return document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def join(value) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return html_to_text(value)
    return " ".join(html_to_text(v) for v in value)


def formula_pieces(blocks: list[dict]) -> list[tuple[str, str]]:
    """Reading order as (kind, text) pieces: equation blocks are display
    formulas; inline $...$ is split out of the running text (A11)."""
    pieces = []
    for block in blocks:
        kind = block.get("type")
        if kind == "equation":
            pieces.append(("display", (block.get("text") or "").strip().strip("$").strip()))
        elif kind == "text" and not int(block.get("text_level") or 0):
            pieces.extend(math_pieces(block.get("text") or ""))
        elif kind == "list" and block.get("sub_type") != "ref_text":
            for item in block.get("list_items") or []:
                pieces.extend(math_pieces(item))
    return pieces


def body_of(blocks: list[dict]) -> list[str]:
    """Running text: plain text blocks (not headings) and non-reference lists."""
    paragraphs = []
    for block in blocks:
        kind = block.get("type")
        if kind == "text" and not int(block.get("text_level") or 0):
            text = html_to_text(block.get("text"))
            if text:
                paragraphs.append(text)
        elif kind == "list" and block.get("sub_type") != "ref_text":
            paragraphs.extend(html_to_text(i) for i in block.get("list_items") or [])
    return paragraphs
