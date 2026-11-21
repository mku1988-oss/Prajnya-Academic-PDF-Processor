#!/usr/bin/env python
"""In-text figure and table mentions, linked to the figures and tables found.

For biology researchers: a results paragraph says "... was reduced (Fig. 3B)"
or "... are listed in Table 2". This module finds each such mention in the
page text Prajnya writes out and links it to the figure or table object that
carries that printed label, the way the citation markers (`markers.py`) link
"[12]" to entry 12 of the reference list. A reader (or a retrieval system)
can then jump from the sentence to the figure crop and its legend.

What it refuses, because a wrong link is worse than none:

- **Supplementary material.** "Figure S1", "Supplementary Fig. 3",
  "S1 Fig" (PLOS), "Extended Data Fig. 2", "Appendix 1-figure 2" and eLife's
  "Figure 2-figure supplement 1" are not printed in the article. Linking
  "Figure S1" to Figure 1 is the commonest error of other tools.
- **A figure's own label.** "Figure 3." at the start of Figure 3's legend is
  the label, not a mention.

A mention whose number no object carries (the figure was not found) is kept
with an empty target list: the mention is real, the object is missing.

Derived on every write from the stored page text and objects, never cached,
so a rewritten document gets offsets that match the text it holds.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from ..utils.table_labels import table_label_of

#: The word that starts a mention: Fig., Figs, Figure(s), FIGURE, Table(s),
#: TABLE, Tab. Followed (after spaces, possibly a line break) by a number.
_HEAD = re.compile(
    r"(?<![A-Za-z0-9_/])(?P<head>(?:Fig(?:ure)?s?|FIG(?:URE)?S?|fig(?:ure)?s?)\.?"
    r"|(?:Tables?|TABLES?|tables?|Tabs?\.))"
    r"[ \t ]*(?:\r?\n[ \t ]*)?(?=S?\d|[IVX]+\b)")

#: One printed number: an optional "S" (supplementary), the number (book
#: tables have "11.1"), then optional panel letters ("3B", "3Bi").
_ITEM = re.compile(
    r"(?P<supp>S)?(?P<number>\d{1,3}(?:\.\d{1,2})?)(?![\d])"
    r"(?P<panels>[A-Za-z](?:[ivx]{1,3})?(?![A-Za-z]))?")

#: A roman table number ("TABLE II.") on an old scan.
_ROMAN_ITEM = re.compile(r"(?P<number>[IVXL]{1,6})(?![A-Za-z])")

#: Between two items of one mention: ", ", " and ", " & ", or a range dash.
_SEPARATOR = re.compile(r"\s*(?P<sep>,|&|and|[-‐-―−])\s*(?:and\s+)?")

#: A panel letter after a separator ("3A, B", "3A and C", "3A-C"): it adds
#: no new figure.
_PANEL_ONLY = re.compile(r"[A-Za-z](?:[ivx]{1,3})?(?![A-Za-z0-9])")

#: Words just before the head that make the whole mention supplementary.
_SUPPLEMENTARY_BEFORE = re.compile(
    r"(?:Supplement\w*|Suppl\.?|Extended\s+Data|Appendix\s*\d*\s*[-‐-―]?"
    r"|Source\s+Data|Online\s+Resource|Additional\s+file\s*\d*|S\d+)\s*$",
    re.IGNORECASE)

#: eLife's "Figure 2-figure supplement 1", "Figure 1-source data 1": the
#: number names the main figure, but the thing meant is not printed.
_SUPPLEMENT_AFTER = re.compile(
    r"\s*[-‐-―]\s*(?:figure|source|video|animation|table)\s+"
    r"(?:supplement|data|\d)", re.IGNORECASE)

#: Parts of a page text that are not prose: Prajnya's own object placeholders
#: ("[[FIGURE p4_fig1 | Figure 1. ... | figures/.../p4_fig1.png]]", which
#: repeat the caption and name the crop file), HTML comments and tables read
#: as HTML. A mention there is not a mention in the text.
_NOT_PROSE = re.compile(r"\[\[(?:FIGURE|TABLE) .*?\]\]|<!--.*?-->|<table.*?</table>",
                        re.DOTALL | re.IGNORECASE)

#: How far back the supplementary words may stand ("Supplementary\nFig.").
_LOOK_BACK = 40

#: How much of a caption must match the text at a label to call it the label.
_CAPTION_PREFIX = 24

#: A range wider than this ("Figures 1-90") is a misreading, not a range.
_MAX_RANGE = 20


def build_mentions_block(pages: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    The `mentions` block of the document JSON (contract 1.2).

    Args:
        pages: the document's pages as written (`page_number`, `content`,
            `metadata.figures`, `metadata.tables`).

    Returns:
        {"figures": [...], "tables": [...], "skipped": {...}}. Each mention is
        {"page_number", "content_offset", "text", "number", "targets"}:
        `text` is the whole printed mention ("Figures 3A and 4"), `number`
        the one printed number this record links ("4"), `targets` the
        `figure_id`s / `table_id`s that carry that label (empty when none
        was found). A mention naming two numbers gives two records.
    """
    figure_ids, table_ids = _objects_by_number(pages)
    block = {"figures": [], "tables": [],
             "skipped": {"supplementary": 0, "own_label": 0}}
    for page in pages:
        content = page.get("content")
        if not isinstance(content, str) or not isinstance(page.get("page_number"), int):
            continue
        captions = _caption_starts(page)
        for record, reason in find_mentions(content, captions):
            if reason:
                block["skipped"][reason] += 1
                continue
            record["page_number"] = page["page_number"]
            kind = record.pop("kind")
            lookup = figure_ids if kind == "figure" else table_ids
            record["targets"] = list(lookup.get(record["number"], []))
            block["figures" if kind == "figure" else "tables"].append(_ordered(record))
    return block


def find_mentions(content: str, captions: Sequence[str] = ()
                  ) -> Iterator[Tuple[Dict[str, Any], str]]:
    """
    Every mention in one page's text, one record per printed number.

    Args:
        content: the page text.
        captions: the starts of the captions of the objects on this page
            (whitespace collapsed), to recognise a legend's own label.

    Yields:
        (record, reason). `reason` is "" for a mention to link, else why it
        is skipped ("supplementary", "own_label"). A record holds `kind`
        ("figure"/"table"), `number`, `text`, `content_offset`.
    """
    not_prose = [match.span() for match in _NOT_PROSE.finditer(content)]
    for head in _HEAD.finditer(content):
        if any(start <= head.start() < end for start, end in not_prose):
            continue
        kind = "table" if head.group("head").lower().startswith("tab") else "figure"
        items, end = _items_after(content, head.end(), kind)
        if not items:
            continue
        text = content[head.start():end]
        whole_reason = _reason_to_skip_whole(content, head.start(), end, captions)
        seen = set()
        for number, offset, supplementary in items:
            reason = whole_reason or ("supplementary" if supplementary else "")
            if not reason and number in seen:
                continue                    # "Fig 1A and 1D": one link to Figure 1
            seen.add(number)
            yield ({"kind": kind, "number": number, "text": " ".join(text.split()),
                    "content_offset": head.start() if offset == head.end() else offset},
                   reason)


# --- helpers (after the main functions, per CLAUDE.md) ---------------------

def _items_after(content: str, start: int, kind: str
                 ) -> Tuple[List[Tuple[str, int, bool]], int]:
    """The printed numbers of one mention: [(number, offset, supplementary)],
    ranges expanded, and where the mention ends."""
    first = _match_item(content, start, kind)
    if first is None:
        return [], start
    items = [_item_tuple(first, content)]
    position = first.end()
    while True:
        separator = _SEPARATOR.match(content, position)
        if separator is None:
            break
        after = separator.end()
        item = _match_item(content, after, kind)
        if item is None:
            panel = _PANEL_ONLY.match(content, after)
            if panel is None or separator.group("sep") == "&":
                break
            position = panel.end()                  # "3A, B": a panel, no new figure
            continue
        new = _item_tuple(item, content)
        if separator.group("sep") not in (",", "&", "and") and new[0] != items[-1][0]:
            items.extend(_range_between(items[-1], new))
        items.append(new)
        position = item.end()
    return items, position


def _match_item(content: str, position: int, kind: str):
    """A number at `position`, Arabic or (tables only) roman."""
    item = _ITEM.match(content, position)
    if item is None and kind == "table":
        item = _ROMAN_ITEM.match(content, position)
    return item


def _item_tuple(item, content: str) -> Tuple[str, int, bool]:
    """(number, offset, supplementary) of one matched number."""
    groups = item.groupdict()
    supplementary = bool(groups.get("supp")) or bool(
        _SUPPLEMENT_AFTER.match(content, item.end()))
    return item.group("number"), item.start(), supplementary


def _range_between(first: Tuple[str, int, bool], last: Tuple[str, int, bool]
                   ) -> List[Tuple[str, int, bool]]:
    """The numbers strictly between a range's ends ("Figures 2-5": 3, 4)."""
    if not (first[0].isdigit() and last[0].isdigit()) or first[2] or last[2]:
        return []
    low, high = int(first[0]), int(last[0])
    if not 0 < high - low <= _MAX_RANGE:
        return []
    return [(str(n), last[1], False) for n in range(low + 1, high)]


def _reason_to_skip_whole(content: str, start: int, end: int,
                          captions: Sequence[str]) -> str:
    """'supplementary' or 'own_label' when the whole mention is skipped."""
    before = content[max(0, start - _LOOK_BACK):start]
    if _SUPPLEMENTARY_BEFORE.search(before):
        return "supplementary"
    if _is_own_label(content, start, end, captions):
        return "own_label"
    return ""


def _is_own_label(content: str, start: int, end: int, captions: Sequence[str]) -> bool:
    """A legend's label: at the start of a line, and either followed by '|'
    (Frontiers) or the start of one of this page's captions."""
    line_start = content.rfind("\n", 0, start) + 1
    if content[line_start:start].strip(" \t *#>") != "":
        return False
    after = content[end:end + 3].lstrip(" * ")
    if after.startswith("|"):
        return True
    here = _collapse(content[start:start + 4 * _CAPTION_PREFIX])
    return any(here.startswith(caption) for caption in captions)


def _caption_starts(page: Mapping[str, Any]) -> List[str]:
    """The first characters of each caption on the page, whitespace collapsed."""
    metadata = page.get("metadata") or {}
    starts = []
    for item in list(metadata.get("figures") or []) + list(metadata.get("tables") or []):
        caption = _collapse(item.get("caption") or "")
        if len(caption) >= 8:
            starts.append(caption[:_CAPTION_PREFIX])
    return starts


def _collapse(text: str) -> str:
    """Whitespace and markdown emphasis removed, for comparing text."""
    return re.sub(r"[\s* ]+", "", text)


def _objects_by_number(pages: Sequence[Mapping[str, Any]]
                       ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """{printed number: [ids]} for figures (by label) and tables (by caption)."""
    figures: Dict[str, List[str]] = {}
    tables: Dict[str, List[str]] = {}
    for page in pages:
        metadata = page.get("metadata") or {}
        for figure in metadata.get("figures") or []:
            number = _figure_number(figure)
            if number and figure.get("figure_id"):
                figures.setdefault(number, []).append(figure["figure_id"])
        for table in metadata.get("tables") or []:
            label = table_label_of(table.get("caption"))
            if label and table.get("table_id"):
                tables.setdefault(label.split(" ", 1)[1], []).append(table["table_id"])
    return figures, tables


def _figure_number(figure: Mapping[str, Any]) -> Optional[str]:
    """'3' for a figure labelled 'Figure 3' (supplementary labels excluded)."""
    label = figure.get("printed_label") or ""
    found = re.match(r"\s*(?:Fig(?:ure)?s?|FIG(?:URE)?S?)\.?\s*(\d{1,3}(?:\.\d{1,2})?)(?![\d])",
                     label)
    return found.group(1) if found else None


def _ordered(record: Dict[str, Any]) -> Dict[str, Any]:
    """The record's keys in the contract's order."""
    keys = ("page_number", "content_offset", "text", "number", "targets")
    return {key: record[key] for key in keys}
