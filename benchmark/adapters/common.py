"""
Shared pieces for the per-tool adapters (protocol §3).

For biology researchers: every adapter turns one tool's native output into
the same JSON shape (the "common format"). Two conventions matter:

* `body_paragraphs` is the tool's running text in the tool's own order,
  WITHOUT what the tool itself marks as a heading, caption, page header or
  footer, table or figure. Where a tool gives no paragraph breaks (Prajnya's
  text layer gives lines), a "paragraph" is whatever unit the tool gives.
* every in-text link records `paragraph_index` and `offset` (characters into
  that paragraph). The scorer places links by position, so tools that split
  paragraphs differently are still compared fairly.

A field the tool does not produce is None ("not supported"), never [].
"""
from __future__ import annotations

import html as html_lib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.text_utils import clean_doi, parse_label, year_of   # noqa: E402,F401


def empty_document(paper_id: str, tool: str, version: str, track: str) -> dict:
    """The common format with every field 'not supported' until filled."""
    return {"paper_id": paper_id, "tool": tool, "tool_version": version,
            "track": track,
            "metadata": {"title": None, "authors": None, "doi": None,
                         "year": None, "journal": None},
            "abstract": None, "body_paragraphs": None, "figures": None,
            "tables": None, "references": None, "citation_links": None,
            "figure_links": None, "table_links": None,     # table_links: A7
            "unlabelled_figure_objects": None,
            "formulas": None}                               # A11


def figure_entry(fid: str, caption: str, label_source: str | None = None,
                 crops: list | None = None) -> dict | None:
    """A figure object, labelled from its label text or its caption start;
    None when it carries no main-figure label (it is then 'unlabelled')."""
    parsed = parse_label(label_source) or parse_label(caption)
    if parsed is None or parsed[0] != "figure":
        return None
    return {"id": fid, "label": f"Figure {parsed[1]}", "label_number": parsed[1],
            "caption": caption or "", "object_count": 1,
            "crops": crops}                    # A11: None = the tool gives no crop box


def table_entry(tid: str, caption: str, html: str | None,
                label_source: str | None = None) -> dict:
    """A table; label_number is None when its caption gives no 'Table N'."""
    parsed = parse_label(label_source) or parse_label(caption)
    number = parsed[1] if parsed and parsed[0] == "table" else None
    return {"id": tid, "label": f"Table {number}" if number else None,
            "label_number": number, "caption": caption or "", "html": html}


def collapse_figures(figures: list[dict]) -> list[dict]:
    """One entry per label number; object_count = how many separate objects
    the tool gave that label (2+ means the figure was split)."""
    by_number: dict[str, dict] = {}
    for figure in figures:
        kept = by_number.get(figure["label_number"])
        if kept is None:
            by_number[figure["label_number"]] = dict(figure)
        else:
            kept["object_count"] += 1
            if figure.get("crops"):            # A11: every crop given this label
                kept["crops"] = (kept.get("crops") or []) + figure["crops"]
            if len(figure["caption"]) > len(kept["caption"]):
                kept["caption"] = figure["caption"]
    return list(by_number.values())


def html_to_text(fragment: str | None) -> str:
    """Visible text of an HTML fragment (tags dropped, entities decoded)."""
    if not fragment:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", fragment)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", html_lib.unescape(text)).strip()


def cells_to_html(cells: list[dict]) -> str:
    """Rows of {'row', 'col', 'rowspan', 'colspan', 'text'} -> minimal HTML,
    the same shape the answer key uses (PubTabNet style)."""
    rows: dict[int, list[dict]] = {}
    for cell in cells:
        rows.setdefault(cell["row"], []).append(cell)
    parts = []
    for row in sorted(rows):
        cells_html = []
        for cell in sorted(rows[row], key=lambda c: c["col"]):
            spans = ""
            if cell.get("colspan", 1) > 1:
                spans += f' colspan="{cell["colspan"]}"'
            if cell.get("rowspan", 1) > 1:
                spans += f' rowspan="{cell["rowspan"]}"'
            text = (cell.get("text") or "").replace("&", "&amp;").replace("<", "&lt;")
            cells_html.append(f"<td{spans}>{text}</td>")
        parts.append("<tr>" + "".join(cells_html) + "</tr>")
    return "<table>" + "".join(parts) + "</table>"


# --- amendment A11: formulas and figure crops ------------------------------------

#: Math a tool writes into its text: display first, so "$$" is never read as
#: two inline "$". Marker writes <math display="block"> and <math>.
MATH_SPANS = re.compile(
    r'\$\$(?P<d1>.*?)\$\$|\\\[(?P<d2>.*?)\\\]|<math display="block">(?P<d3>.*?)</math>'
    r'|(?<![\\$])\$(?!\$)(?P<i1>[^$\n]+?)(?<!\\)\$|\\\((?P<i2>.*?)\\\)|<math>(?P<i3>.*?)</math>',
    re.DOTALL)


def math_pieces(text: str) -> list[tuple[str, str]]:
    """Split a text into ("text", ...), ("display", latex), ("inline", latex) pieces."""
    pieces, at = [], 0
    for match in MATH_SPANS.finditer(text or ""):
        pieces.append(("text", text[at:match.start()]))
        kind = next(name for name, value in match.groupdict().items() if value is not None)
        pieces.append(("display" if kind.startswith("d") else "inline", match.group(kind).strip()))
        at = match.end()
    pieces.append(("text", (text or "")[at:]))
    return pieces


def formulas_of_pieces(pieces: list[tuple[str, str]]) -> list[dict]:
    """Formulas in reading order, each with the CONTEXT_CHARS of plain text before it.

    `pieces` is the tool's reading order as (kind, text): kind "text" for
    running text, "display"/"inline" for a formula. Formula text never
    enters a context (each tool writes it differently)."""
    formulas, before = [], ""
    for kind, value in pieces:
        if kind == "text":
            before = (before + " " + html_to_text(value))[-4 * CONTEXT_CHARS:]
        else:
            formulas.append({"kind": kind, "text": value,
                             "context_before": re.sub(r"\s+", " ", before)[-CONTEXT_CHARS:].strip()})
    return formulas


CONTEXT_CHARS = 80        # as the key (ground_truth/jats_to_common.py)


def crop(page_index: int, box, unit: str = "pt") -> dict:
    """One crop of a figure: 0-based page, box (x0, y0, x1, y1) with the
    origin top-left, in PDF points ("pt") or page fractions ("fraction");
    the scorer turns fractions into points with the PDF's page size."""
    return {"page_index": int(page_index), "box": [float(v) for v in box], "unit": unit}
