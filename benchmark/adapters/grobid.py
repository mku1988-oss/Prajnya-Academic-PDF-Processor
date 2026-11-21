"""
GROBID TEI XML -> common format (protocol §3).

For biology researchers: GROBID writes TEI, an XML format for scholarly text.
Its header holds the metadata, <body> the paragraphs (with <ref> elements
marking in-text citations and figure mentions), <figure> the figures and
tables, and <listBibl> the parsed reference list.
"""
from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

from .common import (cells_to_html, clean_doi, collapse_figures, empty_document,
                     figure_entry, table_entry, year_of)

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def convert(tei_path: Path, paper_id: str, tool: str = "grobid",
            track: str = "offline") -> dict:
    """Read one GROBID TEI file into the common format."""
    root = etree.parse(str(tei_path), etree.XMLParser(recover=True,
                                                      resolve_entities=False,
                                                      no_network=True)).getroot()
    document = empty_document(paper_id, tool, "0.9.1", track)
    document["metadata"] = metadata_of(root)
    document["abstract"] = " ".join(text(p) for p in root.iter(f"{TEI}abstract")
                                    for p in p.iter(f"{TEI}p")) or None
    paragraphs, links, fig_links, table_links = body_of(root)
    document["body_paragraphs"] = paragraphs
    document["citation_links"] = links
    document["figure_links"] = fig_links
    figures, unlabelled = figures_of(root)
    label_of_id = {f["id"]: f["label_number"] for f in figures}
    for link in fig_links:
        link["figure_ids"] = [label_of_id[i] for i in link["figure_ids"] if i in label_of_id]
    document["figure_links"] = [l for l in fig_links if l["figure_ids"]]
    document["figures"] = collapse_figures(figures)
    document["unlabelled_figure_objects"] = unlabelled
    document["tables"] = tables_of(root)
    # Amendment A7: <ref type="table"> -> the table's printed number.
    table_number = {t["id"]: t["label_number"] for t in document["tables"]}
    for link in table_links:
        link["table_ids"] = [table_number[i] for i in link["table_ids"] if table_number.get(i)]
    document["table_links"] = [l for l in table_links if l["table_ids"]]
    document["references"] = references_of(root)
    return document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text(element) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def metadata_of(root) -> dict:
    header = root.find(f".//{TEI}teiHeader")
    source = header.find(f".//{TEI}sourceDesc/{TEI}biblStruct")
    authors = []
    if source is not None:
        for person in source.findall(f"{TEI}analytic/{TEI}author/{TEI}persName"):
            authors.append({"surname": text(person.find(f"{TEI}surname")),
                            "given": " ".join(text(f) for f in person.findall(f"{TEI}forename"))})
    doi = header.find(f".//{TEI}sourceDesc//{TEI}idno[@type='DOI']")
    date = header.find(f".//{TEI}sourceDesc//{TEI}date")
    journal = header.find(f".//{TEI}sourceDesc//{TEI}monogr/{TEI}title")
    return {"title": text(header.find(f".//{TEI}titleStmt/{TEI}title")) or None,
            "authors": authors,
            "doi": clean_doi(text(doi)) if doi is not None else None,
            "year": year_of(date.get("when") or text(date)) if date is not None else None,
            "journal": text(journal) or None}


def body_of(root) -> tuple[list[str], list[dict], list[dict], list[dict]]:
    """Paragraphs of <body>, and each <ref> with its character offset
    (citations, figures, and tables: amendment A7)."""
    body = root.find(f".//{TEI}text/{TEI}body")
    paragraphs, citation_links, figure_links, table_links = [], [], [], []
    if body is None:
        return paragraphs, citation_links, figure_links, table_links
    for p in body.iter(f"{TEI}p"):
        if any(a.tag == f"{TEI}figure" for a in p.iterancestors()):
            continue
        index = len(paragraphs)
        offset = 0
        pieces = []
        for node in p.iter():
            if node.tag == f"{TEI}ref" and node.get("target"):
                target = node.get("target").lstrip("#")
                if node.get("type") == "bibr":
                    citation_links.append({"paragraph_index": index, "offset": offset,
                                           "reference_ids": [target]})
                elif node.get("type") == "figure":
                    figure_links.append({"paragraph_index": index, "offset": offset,
                                         "figure_ids": [target]})
                elif node.get("type") == "table":
                    table_links.append({"paragraph_index": index, "offset": offset,
                                        "table_ids": [target]})
            if node.text:
                pieces.append(node.text)
                offset += len(node.text)
            if node is not p and node.tail:
                pieces.append(node.tail)
                offset += len(node.tail)
        paragraphs.append("".join(pieces))
    return paragraphs, citation_links, figure_links, table_links


def figures_of(root) -> tuple[list[dict], int]:
    figures, unlabelled = [], []
    for figure in root.iter(f"{TEI}figure"):
        if figure.get("type") == "table":
            continue
        head = text(figure.find(f"{TEI}head"))
        label = text(figure.find(f"{TEI}label"))
        caption = text(figure.find(f"{TEI}figDesc"))
        entry = figure_entry(figure.get(XML_ID), f"{head} {caption}".strip(),
                             label_source=head or (f"Figure {label}" if label else None))
        if entry is None:
            unlabelled.append(caption)
        else:
            entry["source_ids"] = [figure.get(XML_ID)]
            figures.append(entry)
    return figures, unlabelled


def tables_of(root) -> list[dict]:
    tables = []
    for figure in root.iter(f"{TEI}figure"):
        if figure.get("type") != "table":
            continue
        head = text(figure.find(f"{TEI}head"))
        label = text(figure.find(f"{TEI}label"))
        caption = text(figure.find(f"{TEI}figDesc"))
        table = figure.find(f"{TEI}table")
        html = None
        if table is not None:
            cells = []
            for r, row in enumerate(table.findall(f"{TEI}row")):
                col = 0
                for cell in row.findall(f"{TEI}cell"):
                    span = int(cell.get("cols") or 1)
                    cells.append({"row": r, "col": col, "colspan": span,
                                  "rowspan": int(cell.get("rows") or 1),
                                  "text": text(cell)})
                    col += span
            html = cells_to_html(cells)
        tables.append(table_entry(figure.get(XML_ID), f"{head} {caption}".strip(), html,
                                  label_source=head or (f"Table {label}" if label else None)))
    return tables


def references_of(root) -> list[dict]:
    references = []
    for position, bibl in enumerate(root.iter(f"{TEI}biblStruct"), start=0):
        if bibl.get(XML_ID) is None or not any(
                a.tag == f"{TEI}listBibl" for a in bibl.iterancestors()):
            continue
        title = bibl.find(f"{TEI}analytic/{TEI}title")
        if title is None or not text(title):
            title = bibl.find(f"{TEI}monogr/{TEI}title")
        surname = bibl.find(f".//{TEI}author/{TEI}persName/{TEI}surname")
        date = bibl.find(f".//{TEI}imprint/{TEI}date")
        doi = bibl.find(f".//{TEI}idno[@type='DOI']")
        raw = bibl.find(f"{TEI}note[@type='raw_reference']")
        references.append({
            "id": bibl.get(XML_ID), "position": len(references) + 1,
            "title": text(title) or None,
            "year": year_of(date.get("when") or text(date)) if date is not None else None,
            "doi": clean_doi(text(doi)) if doi is not None else None,
            "first_author_surname": text(surname) or None,
            "raw": text(raw) if raw is not None else text(bibl)})
    return references
