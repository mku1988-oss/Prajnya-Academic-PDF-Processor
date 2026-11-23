#!/usr/bin/env python
"""
Draw the v6.9 benchmark round (PROTOCOL.md amendment A11).

For biology researchers: this picks the 40 scored papers (20 biology in four
sub-fields, 10 physics, 10 chemistry) and 10 pilots from seven CC BY
journals. A paper's field comes from the publisher's own label in its JATS
XML, or for the Nature journals from Springer Nature's subject taxonomy
(`subject_tree.py`); never from any tool. Each (field, journal) list of
candidates is shuffled once with the seed fixed in A11, and eligible papers
are taken in that order to fill the quotas.

Run (from src/v6):
    python benchmark/corpus_tools/build_manifest_a11.py

Writes benchmark/corpus/MANIFEST_A11.tsv and DRAW_REPORT_A11.json. Listings,
records, XML and PDFs are cached (corpus/cache/), so a re-run gives the
identical draw. Run 2's MANIFEST.tsv is not touched.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_manifest as base  # noqa: E402  (download, cache and DOI helpers)
from subject_tree import SubjectTree  # noqa: E402

# --- Fixed by amendment A11. Changing any of these is a new amendment. ------
SEED = 20261123
YEARS = range(2018, 2026)
QUOTAS = [   # (field, [(journal, papers), ...]) in table order
    ("biochemistry", [("Proc Natl Acad Sci U S A", 2), ("Nat Commun", 2), ("Nature", 1)]),
    ("biophysics", [("Sci Adv", 2), ("Nat Commun", 2), ("Sci Rep", 1)]),
    ("developmental biology", [("Proc Natl Acad Sci U S A", 2), ("Nat Commun", 1), ("Sci Rep", 1),
                               ("Nature", 1)]),
    ("computational biology", [("PLoS Comput Biol", 5)]),
    ("physics", [("Proc Natl Acad Sci U S A", 3), ("Sci Adv", 3), ("Nat Commun", 2), ("Sci Rep", 2)]),
    ("chemistry", [("Front Chem", 3), ("Nat Commun", 2), ("Sci Rep", 2), ("Proc Natl Acad Sci U S A", 2),
                   ("Sci Adv", 1)]),
]
PILOTS = 10
NATURE_FAMILY = {"Nature", "Nat Commun", "Sci Rep"}
SCI_ADV_PHYSICS = {"Physics", "Condensed Matter Physics", "Applied Physics"}
SCI_ADV_CHEMISTRY = {"Chemistry"}
SCI_ADV_BOTH = {"Chemical Physics"}
ESEARCH_PAGE = 10000

CORPUS_DIR = base.CORPUS_DIR
TREE = None   # SubjectTree, loaded once in main()


@dataclass
class Drawn:
    pmcid: str
    version: str
    journal: str
    field: str
    role: str = "scored"
    doi: str = ""
    title: str = ""
    licence: str = ""
    pages: int = 0
    draw_rank: int = 0
    counts: dict = field(default_factory=dict)


def main() -> int:
    """List, shuffle, walk and fill every quota cell, then the pilots; write the manifest."""
    global TREE
    base.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TREE = SubjectTree()
    excluded = exclusions()
    print(f"excluded DOIs: {len(excluded)}")
    walkers = {}
    drawn, shortfalls = [], {}
    for field_name, cells in QUOTAS:
        carry = 0
        for journal, wanted in cells:
            walker = walkers.setdefault((field_name, journal),
                                        Walker(field_name, journal, excluded))
            need = wanted + carry
            got = [paper for paper in (walker.next() for _ in range(need)) if paper]
            drawn.extend(got)
            carry = need - len(got)
            print(f"{field_name:24s} {journal:26s} wanted {need}, drawn {len(got)}")
        if carry:
            shortfalls[field_name] = carry
    pilots = draw_pilots(walkers)
    drawn.extend(pilots)
    write_outputs(drawn, walkers, shortfalls)
    return 0


# --- helpers (after main, per the project rule) -----------------------------

def exclusions() -> set:
    """A11 rule 4: everything we developed against, both earlier draws, and the probed papers."""
    found = base.load_exclusion_dois() | base.manifest_dois()
    probed = json.loads((CORPUS_DIR / "PROBED.json").read_text(encoding="utf-8"))
    found |= {base.clean_doi(p["doi"]) for p in probed["papers"] if p.get("doi")}
    return found


class Walker:
    """One (field, journal) list, shuffled once with the seed, handed out in order."""

    def __init__(self, field_name: str, journal: str, excluded: set) -> None:
        self.field = field_name
        self.journal = journal
        self.excluded = excluded
        self.candidates = shuffled_candidates(field_name, journal)
        self.position = 0
        self.skipped: dict = {}

    def next(self) -> Drawn | None:
        while self.position < len(self.candidates):
            number = self.candidates[self.position]
            self.position += 1
            paper, reason = check(number, self.journal, self.field, self.excluded)
            if paper is None:
                self.skipped[reason] = self.skipped.get(reason, 0) + 1
                continue
            paper.draw_rank = self.position
            self.excluded.add(paper.doi)          # never drawn twice
            return paper
        return None


def shuffled_candidates(field_name: str, journal: str) -> list:
    numbers = set()
    for year in YEARS:
        numbers.update(esearch_all(journal, year))
    ordered = sorted(numbers, key=int)
    random.Random(f"{SEED}:{field_name}:{journal}").shuffle(ordered)
    return ordered


def esearch_all(journal: str, year: int) -> list:
    """Every CC BY PMC number of one journal-year (cached).

    E-utilities returns at most 9,999 records per query (retstart cannot
    pass 9,998), so a journal-year larger than that (Scientific Reports) is
    asked month by month. The union is the same set either way, and it is
    sorted before the seeded shuffle, so the draw does not depend on how
    the list was fetched.
    """
    whole = esearch_page(journal, f"{year}[pdat]", f"{year}")
    if int(whole["count"]) <= len(whole["idlist"]):
        return whole["idlist"]
    found = []
    for month in range(1, 13):
        span = f'"{year}/{month:02d}/01"[pdat] : "{year}/{month:02d}/31"[pdat]'
        part = esearch_page(journal, span, f"{year}_m{month:02d}")
        if int(part["count"]) > len(part["idlist"]):
            raise RuntimeError(f"{journal} {year}-{month:02d}: {part['count']} records, more than one query returns")
        found.extend(part["idlist"])
    return found


def esearch_page(journal: str, date_term: str, label: str) -> dict:
    """One E-utilities query (up to ESEARCH_PAGE ids) for a journal and date range."""
    term = f'"{journal}"[journal] AND "cc by license"[filter] AND {date_term}'
    query = urllib.parse.urlencode({"db": "pmc", "term": term, "retmax": ESEARCH_PAGE,
                                    "retmode": "json", "tool": "prajnya_benchmark"})
    name = f"a11_esearch_{journal.replace(' ', '_')}_{label}_0.json"
    data = json.loads(base.fetch_cached(f"{base.EUTILS}/esearch.fcgi?{query}", name,
                                        pause=base.NCBI_PAUSE_S), strict=False)["esearchresult"]
    if "ERROR" in data or "idlist" not in data:
        raise RuntimeError(f"E-utilities refused {term}: {data.get('ERROR', data)}")
    return data


def check(number: str, journal: str, field_name: str, excluded: set):
    """A11 eligibility: CC BY, not retracted, research article with a body, not excluded, and of this field."""
    pmcid = f"PMC{number}"
    version = base.latest_version(pmcid)
    if not version:
        return None, "not in bucket"
    record = json.loads(base.fetch_cached(f"{base.BUCKET}/{version}/{version}.json", f"{version}.json"))
    if record.get("license_code") != "CC BY":
        return None, f"licence {record.get('license_code')}"
    if record.get("is_retracted") or record.get("is_manuscript"):
        return None, "retracted or author manuscript"
    if not record.get("pdf_url") or not record.get("xml_url"):
        return None, "no PDF or no XML"
    doi = base.clean_doi(record.get("doi") or "")
    if doi and doi in excluded:
        return None, "excluded (used or probed)"
    root = base.parse_jats(base.fetch_cached(f"{base.BUCKET}/{version}/{version}.xml", f"{version}.xml"))
    if root is None:
        return None, "XML does not parse"
    if root.get("article-type") != "research-article":
        return None, "not a research article"
    if root.find(".//body") is None or not root.findall(".//body//p"):
        return None, "no body in XML"
    found = field_of(root, journal)
    if found != field_name:
        return None, f"field: {found or 'none or two'}"
    paper = Drawn(pmcid=pmcid, version=version, journal=journal, field=field_name, doi=doi,
                  title=record.get("title", ""), licence=record["license_code"])
    paper.pages = page_count(version)
    paper.counts = counts(root)
    return paper, "eligible"


def field_of(root: ET.Element, journal: str) -> str | None:
    """The paper's field by A11 rule 3, or None (none, or two)."""
    if journal == "PLoS Comput Biol":
        return "computational biology"
    if journal == "Front Chem":
        return "chemistry"
    if journal in NATURE_FAMILY:
        return TREE.field_of([k.text for k in root.iter("kwd") if k.text])
    groups = subject_groups(root)
    if journal == "Proc Natl Acad Sci U S A":
        heading = set(groups.get("heading", []))
        reached = set()
        if "Biological Sciences" in heading and "Biochemistry" in heading:
            reached.add("biochemistry")
        if "Biological Sciences" in heading and "Developmental Biology" in heading:
            reached.add("developmental biology")
        if "Physical Sciences" in heading and "Chemistry" in heading:
            reached.add("chemistry")
        if "Physical Sciences" in heading and heading & {"Physics", "Applied Physical Sciences"}:
            reached.add("physics")
        return reached.pop() if len(reached) == 1 else None
    if journal == "Sci Adv":
        heading, fields = set(groups.get("heading", [])), set(groups.get("field", []))
        reached = set()
        if "Biophysics" in fields:
            reached.add("biophysics")
        if "Physical and Materials Sciences" in heading:
            if fields & SCI_ADV_BOTH:
                reached |= {"physics", "chemistry"}
            if fields & SCI_ADV_PHYSICS:
                reached.add("physics")
            if fields & SCI_ADV_CHEMISTRY:
                reached.add("chemistry")
        return reached.pop() if len(reached) == 1 else None
    return None


def subject_groups(root: ET.Element) -> dict:
    """{subj-group-type: subjects}, nested groups' subjects included (PNAS nests
    "Biochemistry" inside the "Biological Sciences" heading group)."""
    groups: dict = {}
    for group in root.iter("subj-group"):
        groups.setdefault(group.get("subj-group-type") or "untyped", []).extend(
            s.text.strip() for s in group.iter("subject") if s.text)
    return groups


def counts(root: ET.Element) -> dict:
    mml = "{http://www.w3.org/1998/Math/MathML}math"
    return {"figures": len(root.findall(".//fig")),
            "tables": len(root.findall(".//table-wrap")),
            "display_formulas": sum(1 for f in root.iter("disp-formula") if f.find(f".//{mml}") is not None),
            "inline_formulas": sum(1 for f in root.iter("inline-formula") if f.find(f".//{mml}") is not None),
            "references": len(root.findall(".//ref-list/ref"))}


def page_count(version: str) -> int:
    import pypdfium2 as pdfium
    path = base.CACHE_DIR / f"{version}.pdf"
    if not path.exists():
        path.write_bytes(base.http_get(f"{base.BUCKET}/{version}/{version}.pdf"))
    document = pdfium.PdfDocument(str(path))
    try:
        return len(document)
    finally:
        document.close()


def draw_pilots(walkers: dict) -> list:
    """A11 rule 5: the next eligible papers, in turn across the rows (each row's journals in order)."""
    pilots, row = [], 0
    rows = [(field_name, [journal for journal, _ in cells]) for field_name, cells in QUOTAS]
    tried = 0
    while len(pilots) < PILOTS and tried < len(rows) * 4:
        field_name, journals = rows[row % len(rows)]
        for journal in journals:
            paper = walkers[(field_name, journal)].next()
            if paper:
                paper.role = "pilot"
                pilots.append(paper)
                break
        row += 1
        tried += 1
    return pilots


def write_outputs(drawn: list, walkers: dict, shortfalls: dict) -> None:
    columns = ["role", "field", "journal", "pmcid", "version", "doi", "licence", "pages", "draw_rank",
               "figures", "tables", "display_formulas", "inline_formulas", "references", "title"]
    with open(CORPUS_DIR / "MANIFEST_A11.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for paper in drawn:
            record = {**asdict(paper), **paper.counts}
            writer.writerow([record.get(c, "") for c in columns])
    scored = [p for p in drawn if p.role == "scored"]
    report = {
        "amendment": "A11", "seed": SEED, "scored": len(scored), "pilot": len(drawn) - len(scored),
        "per_field": {f: sum(1 for p in scored if p.field == f) for f, _ in QUOTAS},
        "per_journal": {j: sum(1 for p in scored if p.journal == j) for j in sorted({p.journal for p in drawn})},
        "shortfalls": shortfalls,
        "skipped": {f"{f} / {j}": w.skipped for (f, j), w in walkers.items()},
        "candidates": {f"{f} / {j}": len(w.candidates) for (f, j), w in walkers.items()},
    }
    (CORPUS_DIR / "DRAW_REPORT_A11.json").write_text(json.dumps(report, indent=2), encoding="utf-8",
                                                     newline="")
    print(json.dumps({k: report[k] for k in ("scored", "pilot", "per_field", "shortfalls")}))


if __name__ == "__main__":
    sys.exit(main())
