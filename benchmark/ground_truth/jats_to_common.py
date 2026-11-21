#!/usr/bin/env python
"""
Turn a publisher's JATS XML into the benchmark's common format (protocol §2-§3).

For biology researchers: JATS is the structured version of an article that the
publisher deposits in PMC. It already marks the title, authors, every
paragraph, each figure and table, each reference, and which reference every
in-text citation points to. This module reads those marks and writes the
answer key in the same shape every tool's output is converted to, so the
scorer compares like with like.

Run (from src/v6):
    python benchmark/ground_truth/jats_to_common.py
writes benchmark/outputs/key/<PMCID>.json for every paper in the manifest.
"""
from __future__ import annotations

import copy
import csv
import json
import re
import sys
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.text_utils import clean_doi, parse_label, year_of   # noqa: E402

V6_ROOT = Path(__file__).resolve().parents[2]
CORPUS = V6_ROOT / "benchmark" / "corpus"
KEY_DIR = V6_ROOT / "benchmark" / "outputs" / "key"
EXCLUDED_ANCESTORS = {"fig", "table-wrap", "caption", "fn", "fn-group",
                      "supplementary-material", "disp-formula", "boxed-text",
                      "ref-list", "table", "app", "app-group", "ack"}
# Objects anchored inside a paragraph whose text is not the paragraph's (A7).
NESTED_OBJECTS = ("disp-formula", "fig", "table-wrap")
# The common format's name for each kind of link's targets.
LINK_FIELD = {"bibr": "reference_ids", "fig": "figure_ids", "table": "table_ids"}
RANGE_JOINERS = re.compile(r"^\s*[-\u2010-\u2015\u2212]\s*$")


def main() -> int:
    """Write the answer key for every manifest paper."""
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS / "MANIFEST.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        xml_path = CORPUS / "cache" / f"{row['version']}.xml"
        key = jats_to_common(xml_path.read_bytes(), row["pmcid"])
        (KEY_DIR / f"{row['pmcid']}.json").write_text(
            json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8",
            newline="")
    print(f"wrote {len(rows)} keys to {KEY_DIR}")
    return 0


def jats_to_common(xml_bytes: bytes, paper_id: str) -> dict:
    """The answer key for one article, in the common format."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)
    root = etree.fromstring(xml_bytes, parser)
    strip_namespaces(root)
    meta = root.find(".//front/article-meta")
    paragraphs, paragraph_elements = body_paragraphs(root)
    figures = figures_of(root)
    tables = tables_of(root)
    references = references_of(root)
    return {
        "paper_id": paper_id, "tool": "key", "tool_version": "JATS", "track": "key",
        "metadata": metadata_of(root, meta),
        "abstract": abstract_of(meta),
        "body_paragraphs": paragraphs,
        "figures": figures,
        "tables": tables,
        "references": references,
        "citation_links": links_of(paragraph_elements, "bibr",
                                   [r["id"] for r in references]),
        "figure_links": links_of(paragraph_elements, "fig",
                                 [f["id"] for f in figures]),
        # Amendment A7: in-text table mentions, scored like figure mentions.
        "table_links": links_of(paragraph_elements, "table",
                                [t["id"] for t in tables]),
    }


# ---------------------------------------------------------------------------
# Helpers (after the main functions, per the project rule)
# ---------------------------------------------------------------------------

def strip_namespaces(root) -> None:
    """MathML and xlink carry namespaces; the JATS elements we read do not.
    Removing prefixes makes every lookup a plain tag name."""
    for element in root.iter():
        if isinstance(element.tag, str) and "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]


def text_of(element, drop: tuple[str, ...] = ("disp-formula",)) -> str:
    """All text inside an element, skipping display formulas (§5 step 6).
    Works on a copy: each dropped element is removed, its tail text kept."""
    if element is None:
        return ""
    clone = copy.deepcopy(element)
    for node in [n for n in clone.iter(*drop) if n is not clone]:
        parent, previous = node.getparent(), node.getprevious()
        if node.tail:
            if previous is not None:
                previous.tail = (previous.tail or "") + node.tail
            else:
                parent.text = (parent.text or "") + node.tail
        parent.remove(node)
    return re.sub(r"\s+", " ", "".join(clone.itertext())).strip()


def metadata_of(root, meta) -> dict:
    authors = []
    for contrib in meta.findall(".//contrib-group/contrib[@contrib-type='author']"):
        name = contrib.find("name")
        if name is not None:
            authors.append({"surname": text_of(name.find("surname")),
                            "given": text_of(name.find("given-names"))})
        elif contrib.find("collab") is not None:
            authors.append({"surname": text_of(contrib.find("collab")), "given": ""})
    journal_meta = root.find(".//front/journal-meta")
    journals = [text_of(j) for j in journal_meta.iter("journal-title")]
    abbreviations = [text_of(j) for j in journal_meta.iter("abbrev-journal-title")]
    abbreviations += [text_of(j) for j in journal_meta.findall("journal-id")
                      if j.get("journal-id-type") in ("nlm-ta", "iso-abbrev")]
    doi = meta.find("article-id[@pub-id-type='doi']")
    return {"title": text_of(meta.find(".//title-group/article-title")),
            "authors": authors,
            "doi": clean_doi(doi.text if doi is not None else None),
            "year": earliest_year(meta),
            "journal": journals[0] if journals else None,
            "journal_alternates": sorted({j for j in journals + abbreviations if j})}


def earliest_year(meta) -> int | None:
    years = [year_of(text_of(d.find("year"))) for d in meta.findall("pub-date")]
    years = [y for y in years if y]
    return min(years) if years else None


def abstract_of(meta) -> str | None:
    abstracts = meta.findall("abstract")
    plain = [a for a in abstracts if not a.get("abstract-type")] or abstracts
    if not plain:
        return None
    return " ".join(text_of(p) for p in plain[0].iter("p")) or text_of(plain[0])


def body_paragraphs(root) -> tuple[list[str], list]:
    """Every body paragraph in document order, outside figures, tables,
    captions, footnotes and formulas; returns texts and the elements."""
    body = root.find(".//body")
    texts, elements = [], []
    if body is None:
        return texts, elements
    for p in body.iter("p"):
        if any(a.tag in EXCLUDED_ANCESTORS for a in p.iterancestors()):
            continue
        if is_wrapper(p):                       # its inner p count instead
            continue
        # Amendment A7: a <fig> or <table-wrap> anchored inside the paragraph
        # (Cell Press, Nature Communications) is not paragraph text.
        text = text_of(p, drop=NESTED_OBJECTS)
        if text:
            texts.append(text)
            elements.append(p)
    return texts, elements


def is_wrapper(p) -> bool:
    """A <p> holding body paragraphs of its own (amendment A7).

    Only a nested <p> that is itself body text counts. Before A7 any nested
    <p> did, so a paragraph holding an anchored figure (whose caption has
    its own <p>) was dropped from the key with its links."""
    return any(not inside_excluded(q, p) for q in p.iter("p") if q is not p)


def inside_excluded(node, paragraph) -> bool:
    """Whether a node sits inside a figure, table, caption, ... below `paragraph`."""
    for ancestor in node.iterancestors():
        if ancestor is paragraph:
            return False
        if ancestor.tag in EXCLUDED_ANCESTORS:
            return True
    return False


def figures_of(root) -> list[dict]:
    figures = []
    for fig in root.iter("fig"):
        if any(a.tag == "supplementary-material" for a in fig.iterancestors()):
            continue
        if fig.get("specific-use") == "child-fig":      # eLife figure supplements
            continue
        label = text_of(fig.find("label"))
        parsed = parse_label(label)
        if parsed is None or parsed[0] != "figure":
            continue
        figures.append({"id": fig.get("id"), "label": label,
                        "label_number": parsed[1],
                        "caption": text_of(fig.find("caption")),
                        "object_count": 1})
    return figures


def tables_of(root) -> list[dict]:
    tables = []
    for wrap in root.iter("table-wrap"):
        if any(a.tag == "supplementary-material" for a in wrap.iterancestors()):
            continue
        label = text_of(wrap.find("label"))
        parsed = parse_label(label)
        if parsed is None or parsed[0] != "table":
            continue
        table = wrap.find(".//table")
        tables.append({"id": wrap.get("id"), "label": label,
                       "label_number": parsed[1],
                       "caption": text_of(wrap.find("caption")),
                       "html": table_html(table) if table is not None else None,
                       "image_only": table is None})
    return tables


def table_html(table) -> str:
    """A minimal <table> with only rows and cells (+ spans), like PubTabNet."""
    rows = []
    for tr in table.iter("tr"):
        cells = []
        for cell in tr:
            if cell.tag not in ("td", "th"):
                continue
            spans = "".join(f' {a}="{cell.get(a)}"' for a in ("colspan", "rowspan")
                            if cell.get(a) and cell.get(a) != "1")
            cells.append(f"<td{spans}>{escape(text_of(cell))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def references_of(root) -> list[dict]:
    references = []
    for position, ref in enumerate(root.findall(".//back//ref-list/ref"), start=1):
        citation = ref.find("element-citation")
        if citation is None:
            citation = ref.find("mixed-citation")
        if citation is None:
            citation = ref.find("citation") if ref.find("citation") is not None else ref
        title = (citation.find("article-title") if citation.find("article-title") is not None
                 else citation.find("chapter-title"))
        if title is None and citation.get("publication-type") in ("book", "thesis"):
            title = citation.find("source")
        doi = citation.find("pub-id[@pub-id-type='doi']")
        doi_text = doi.text if doi is not None else None
        if doi_text is None:
            found = re.search(r"10\.\d{4,9}/\S+", text_of(citation))
            doi_text = found.group(0).rstrip(".,;") if found else None
        first = citation.find(".//person-group[@person-group-type='author']//surname")
        if first is None:
            first = citation.find(".//surname")
        collab = citation.find(".//collab")
        references.append({
            "id": ref.get("id"), "position": position,
            "title": text_of(title) or None,
            "year": year_of(text_of(citation.find("year"))),
            "doi": clean_doi(doi_text),
            "first_author_surname": text_of(first) if first is not None
            else (text_of(collab) if collab is not None else None),
            "raw": re.sub(r"\s+", " ", " ".join(citation.itertext())).strip()})
    return references


def links_of(paragraph_elements: list, ref_type: str, ordered_ids: list[str]) -> list[dict]:
    """For each paragraph, the targets of its xrefs of one type, with
    numbered ranges ('3–7' written as two xrefs joined by a dash) expanded."""
    order = {rid: i for i, rid in enumerate(ordered_ids)}
    links = []
    for index, p in enumerate(paragraph_elements):
        targets: list[str] = []
        previous = None
        for xref in p.iter("xref"):
            if inside_excluded(xref, p):         # an anchored figure's caption (A7)
                continue
            if xref.get("ref-type") != ref_type:
                previous = None
                continue
            rids = [r for r in (xref.get("rid") or "").split() if r in order]
            if previous is not None and rids and RANGE_JOINERS.match(previous.tail or ""):
                start, end = order[targets[-1]], order[rids[0]]
                targets.extend(ordered_ids[start + 1:end])
            targets.extend(rids)
            previous = xref
        if targets:
            links.append({"paragraph_index": index, LINK_FIELD[ref_type]: targets})
    return links


if __name__ == "__main__":
    sys.exit(main())
