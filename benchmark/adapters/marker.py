"""
Marker JSON -> common format (protocol §3).

For biology researchers: Marker's JSON is a tree of blocks, page by page, in
reading order. Each block has a type (Text, SectionHeader, Caption, Figure,
Picture, Table, PageHeader, ...) and its content as HTML. Marker groups a
figure with its caption in a FigureGroup (or PictureGroup), and a table with
its caption in a TableGroup. Marker does not extract metadata fields, an
abstract field, references or in-text links: those stay "not supported".
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import (collapse_figures, empty_document, figure_entry, html_to_text,
                     table_entry)

BODY_TYPES = {"Text", "TextInlineMath", "ListItem"}
FIGURE_TYPES = {"Figure", "Picture"}
GROUP_TYPES = {"FigureGroup", "PictureGroup"}


def convert(json_path: Path, paper_id: str) -> dict:
    """Read one Marker JSON into the common format."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    blocks = list(walk(data))
    document = empty_document(paper_id, "marker", "2.2.0", "offline")
    document["body_paragraphs"] = [html_to_text(b.get("html")) for b in blocks
                                   if b.get("block_type") in BODY_TYPES
                                   and html_to_text(b.get("html"))]
    figures, unlabelled = figures_of(data)
    document["figures"] = collapse_figures(figures)
    document["unlabelled_figure_objects"] = unlabelled
    document["tables"] = tables_of(data)
    return document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def walk(block: dict):
    """Every block in document order (pre-order); figure and table groups are
    not entered for body text."""
    for child in block.get("children") or []:
        yield child
        if child.get("block_type") not in GROUP_TYPES | {"TableGroup"}:
            yield from walk(child)


def all_blocks(block: dict):
    for child in block.get("children") or []:
        yield child
        yield from all_blocks(child)


def captions_in(group: dict) -> str:
    return " ".join(html_to_text(c.get("html")) for c in group.get("children") or []
                    if c.get("block_type") == "Caption")


def figures_of(data: dict) -> tuple[list[dict], int]:
    """A FigureGroup/PictureGroup is one figure; a Figure/Picture outside any
    group is one figure without a caption (so it carries no label)."""
    figures, unlabelled = [], []
    grouped_ids = set()
    for block in all_blocks(data):
        if block.get("block_type") in GROUP_TYPES:
            grouped_ids.update(c.get("id") for c in block.get("children") or [])
            entry = figure_entry(block.get("id"), captions_in(block))
            if entry is None:
                unlabelled.append(captions_in(block))
            else:
                figures.append(entry)
    for block in all_blocks(data):
        if block.get("block_type") in FIGURE_TYPES and block.get("id") not in grouped_ids:
            unlabelled.append("")
    return figures, unlabelled


def tables_of(data: dict) -> list[dict]:
    tables, grouped_ids = [], set()
    for block in all_blocks(data):
        if block.get("block_type") == "TableGroup":
            children = block.get("children") or []
            grouped_ids.update(c.get("id") for c in children)
            table = next((c for c in children if c.get("block_type") == "Table"), None)
            tables.append(table_entry(block.get("id"), captions_in(block),
                                      extract_table(table)))
    for block in all_blocks(data):
        if block.get("block_type") == "Table" and block.get("id") not in grouped_ids:
            tables.append(table_entry(block.get("id"), "", extract_table(block)))
    return tables


def extract_table(block: dict | None) -> str | None:
    if block is None:
        return None
    html = block.get("html") or ""
    start, end = html.find("<table"), html.rfind("</table>")
    return html[start:end + 8] if start >= 0 and end > start else None
