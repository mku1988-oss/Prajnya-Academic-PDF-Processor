"""
Prajnya contract JSON -> common format (protocol §3).

For biology researchers: Prajnya writes one JSON per document. Each page's
`content` is its text (line by line from the text layer, or Markdown from a
vision-model read). Figures and tables sit in each page's metadata; crops of
one printed figure share a `label_key`. At the `online` citation level the
`citations` block carries the reference list and the in-text markers, each
marker placed by page and `content_offset`.

Parts of the page text that Prajnya itself marks as headings, tables or
images are blanked with spaces (not deleted), so marker offsets stay valid.

Amendment A7: from contract 1.3, each page also marks what is not running
text (`metadata.furniture`: running heads, page numbers; `metadata.outside_body`:
figure and table placeholders, legends, the reference list). Those spans are
blanked the same way. From contract 1.2, the `mentions` block links in-text
"Fig. 3" / "Table 2" to the objects found; those become figure and table
links. Older outputs carry neither and convert as before.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .common import (clean_doi, collapse_figures, crop, empty_document, figure_entry,
                     formulas_of_pieces, math_pieces, table_entry, year_of)


def convert(json_path: Path, paper_id: str, track: str = "offline") -> dict:
    """Read one Prajnya document JSON into the common format."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    tool = "prajnya_online" if track == "online" else "prajnya"
    version = (data.get("document_info") or {}).get("pipeline_version") or "v6.6.4"
    document = empty_document(paper_id, tool, version, track)
    document["metadata"] = metadata_of(data.get("metadata_final") or {})
    pages = data.get("pages") or []
    document["body_paragraphs"] = [body_text(p) for p in pages]
    figures, unlabelled = figures_of(pages)
    document["figures"] = collapse_figures(figures)
    document["unlabelled_figure_objects"] = unlabelled
    # A11: formulas as written in the page text ($$..$$ display, $..$ inline).
    document["formulas"] = formulas_of_pieces(
        [piece for text in document["body_paragraphs"] for piece in math_pieces(text)])
    document["tables"] = [table_entry(t.get("table_id"), t.get("caption") or "",
                                      t.get("grid"))
                          for p in pages for t in (p.get("metadata") or {}).get("tables") or []]
    mentions = data.get("mentions")
    if mentions is not None:                             # contract 1.2 and later
        document["figure_links"] = mention_links(mentions.get("figures") or [], pages, "figure_ids")
        document["table_links"] = mention_links(mentions.get("tables") or [], pages, "table_ids")
    citations = data.get("citations") or {}
    if (citations.get("reference_list") or {}).get("status") not in (None, "not_requested"):
        document["references"] = references_of(citations.get("references") or [])
        document["citation_links"] = markers_of(citations.get("markers") or [], pages)
    return document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def metadata_of(meta: dict) -> dict:
    authors = registry_names(meta) or [split_name(name) for name in meta.get("authors") or []]
    return {"title": meta.get("title"), "authors": authors,
            "doi": clean_doi(meta.get("doi")), "year": year_of(meta.get("year")),
            "journal": meta.get("journal")}


def registry_names(meta: dict) -> list:
    """Contract 1.4 (amendment A10): the surnames Prajnya kept apart, when it
    wrote them, one per author ([] for an older file or an unknown list)."""
    parts = meta.get("author_names") or []
    if not parts or len(parts) != len(meta.get("authors") or []):
        return []
    return [{"surname": (p.get("family") or "").strip(), "given": (p.get("given") or "").strip()}
            for p in parts]


def split_name(name: str) -> dict:
    """One full name ("Dennis van der Meer") -> surname and given names.

    Prajnya writes each author as one string; the key (and GROBID) carry the
    surname as its own field. The surname is the last word that has letters
    (affiliation marks such as "{,}" or "✉" printed after a name are not
    part of it), together with any particles before it ("van der
    Meer", "de la Cruz", "dos Santos")."""
    words = [w for w in name.replace(",", " ").split() if any(c.isalpha() for c in w)]
    if not words:
        return {"surname": "", "given": ""}
    start = len(words) - 1
    while start > 1 and words[start - 1].lower() in SURNAME_PARTICLES:
        start -= 1
    return {"surname": " ".join(words[start:]), "given": " ".join(words[:start])}


# Lowercase words that belong to the surname that follows them.
SURNAME_PARTICLES = {"van", "von", "der", "den", "de", "del", "della", "di", "da",
                     "das", "do", "dos", "du", "la", "le", "ten", "ter", "bin", "al",
                     "el", "y"}


def body_text(page: dict) -> str:
    """The page text with Prajnya's own headings, tables and images blanked,
    and (contract 1.3) its furniture and the spans outside the body."""
    content = blank_spans(page.get("content") or "", page.get("metadata") or {})
    content = blank(content, r"<table.*?</table>", re.DOTALL | re.IGNORECASE)
    content = blank(content, r"!\[[^\]]*\]\([^)]*\)")
    content = blank(content, r"^\s{0,3}#{1,6} .*$", re.MULTILINE)
    for heading in (page.get("metadata") or {}).get("headings") or []:
        text = (heading.get("text") or "").strip()
        if len(text) >= 3:
            content = blank(content, r"^\s*" + re.escape(text) + r"\s*$",
                            re.MULTILINE, count=1)
    return content


def blank_spans(content: str, metadata: dict) -> str:
    """Blank every furniture and outside-body span (offsets index `content`)."""
    characters = list(content)
    for span in (metadata.get("furniture") or []) + (metadata.get("outside_body") or []):
        start, end = max(0, span.get("start") or 0), min(len(content), span.get("end") or 0)
        for i in range(start, end):
            if characters[i] not in "\r\n":
                characters[i] = " "
    return "".join(characters)


def mention_links(mentions: list[dict], pages: list[dict], field: str) -> list[dict]:
    """Mentions linked to an object found -> links naming its printed number."""
    index_of_page = {p.get("page_number"): i for i, p in enumerate(pages)}
    return [{"paragraph_index": index_of_page[m["page_number"]],
             "offset": m.get("content_offset") or 0, field: [str(m["number"])]}
            for m in mentions
            if m.get("targets") and m.get("page_number") in index_of_page]


def blank(text: str, pattern: str, flags: int = 0, count: int = 0) -> str:
    """Replace each match by the same number of spaces (offsets survive)."""
    return re.sub(pattern, lambda m: re.sub(r"[^\n]", " ", m.group(0)), text,
                  count=count, flags=flags)


def figures_of(pages: list[dict]) -> tuple[list[dict], int]:
    """One object per label group (crops sharing a label_key count once)."""
    figures, groups, unlabelled = [], {}, []
    for index, page in enumerate(pages):
        for figure in (page.get("metadata") or {}).get("figures") or []:
            key = figure.get("label_key")
            box = crop_of(index, figure.get("bbox"))
            if key and key in groups:                     # another panel of a figure seen
                groups[key]["crops"] = (groups[key]["crops"] or []) + box
                continue
            entry = figure_entry(figure.get("figure_id"), figure.get("caption") or "",
                                 label_source=figure.get("printed_label"), crops=box or None)
            if entry is None:
                unlabelled.append(figure.get("caption") or "")
                continue
            groups[key] = entry
            figures.append(entry)
    return figures, unlabelled


#: Contract rule 2: a figure `bbox` is in pixels of the rendered page image.
#: Prajnya renders that image at its extraction resolution (300 dpi, the
#: `target_dpi` of its extraction profile) with a 25 px white border
#: (RENDER_PADDING_PX); the contract does not record either, so they are
#: stated here and checked against the saved crops (A11).
RENDER_DPI = 300
RENDER_PADDING_PX = 25


def crop_of(page_index: int, bbox) -> list:
    """A Prajnya figure bbox (rendered-page pixels) -> one crop in PDF points."""
    if not bbox or len(bbox) != 4:
        return []
    points = [(v - RENDER_PADDING_PX) * 72.0 / RENDER_DPI for v in bbox]
    return [crop(page_index, points)]


def references_of(references: list[dict]) -> list[dict]:
    return [{"id": str(r.get("position")), "position": r.get("position"),
             "title": r.get("title"), "year": year_of(r.get("year")),
             "doi": clean_doi(r.get("doi")),
             "first_author_surname": r.get("first_author"),
             "raw": r.get("printed_text") or ""} for r in references]


def markers_of(markers: list[dict], pages: list[dict]) -> list[dict]:
    """Each marker -> (page index, offset, the reference positions it names).
    A marker with no offset is placed at the start of its page."""
    index_of_page = {p.get("page_number"): i for i, p in enumerate(pages)}
    links = []
    for marker in markers:
        index = index_of_page.get(marker.get("page_number"))
        if index is None or not marker.get("numbers"):
            continue
        offset = marker.get("content_offset")
        if offset is None:
            offset = offset_from_context(pages[index].get("content") or "",
                                         marker.get("context_before") or "")
        links.append({"paragraph_index": index, "offset": offset or 0,
                      "reference_ids": [str(n) for n in marker["numbers"]]})
    return links


def offset_from_context(content: str, context_before: str) -> int | None:
    """Where the marker stands, from the text just before it (whitespace-
    insensitive search of its last 30 characters)."""
    tail = re.sub(r"\s+", " ", context_before).strip()[-30:]
    if len(tail) < 8:
        return None
    pattern = r"\s+".join(re.escape(w) for w in tail.split(" "))
    found = re.search(pattern, content)
    return found.end() if found else None
