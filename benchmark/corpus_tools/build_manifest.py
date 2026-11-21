#!/usr/bin/env python
"""
Draw the benchmark corpus from PMC's open-data bucket (protocol §7 + A1).

For biology researchers: this picks the papers every tool will be scored on.
It never looks at any tool's output. It lists CC BY research articles per
publisher, shuffles each list once with the seed fixed in the protocol, keeps
the first eligible papers, and writes benchmark/corpus/MANIFEST.tsv plus each
paper's PDF and JATS XML (the publisher's structured version = answer key).

Run (from src/v6):
    python benchmark/corpus_tools/build_manifest.py

Re-running reuses the cached listings and downloads, so the draw is identical.

Amendment A8 (a fresh scored set, after development had seen the first):
    python benchmark/corpus_tools/build_manifest.py --a8 --data-root <the checkout holding the draw's caches>
draws 40 new scored papers with seed A8_SEED from the same cached listings,
excluding every paper in the existing manifest; the old scored papers become
role "seen", the pilots stay. Report: corpus/DRAW_REPORT_A8.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- Fixed by the protocol (§7, A1). Changing any of these is an amendment. ---
SEED = 20261119
YEARS = range(2018, 2026)
SCORED_PER_STRATUM = 8
PILOT_PER_STRATUM = 2
STRATA = {
    "Frontiers": ["Front Cell Dev Biol", "Front Genet", "Front Physiol",
                  "Front Microbiol", "Front Immunol"],
    "PLOS": ["PLoS Biol", "PLoS Genet", "PLoS Pathog"],
    "eLife": ["eLife"],
    "NatureComms": ["Nat Commun"],
    "CellPress": ["Cell Rep"],
}
MESH_BIOLOGY = {"Animals", "Humans", "Bacteria", "Plants", "Viruses", "Fungi",
                "Cells, Cultured"}
# Amendment A8: the fresh draw's seed, fixed in PROTOCOL.md before the draw.
A8_SEED = 20261120
# Minimums over the 40 scored papers: flag name -> required count.
QUOTAS = {"two_column": 15, "one_column": 15, "has_cell_table": 15,
          "multi_panel_figure": 15, "numbered": 10, "author_year": 10,
          "refs_60_plus": 10}

V6_ROOT = Path(__file__).resolve().parents[2]
# Where the exclusion sources (ledger, run_outputs, eval, examples) live; A8's
# --data-root points at the development checkout when run from a worktree.
DATA_ROOT = V6_ROOT
CORPUS_DIR = V6_ROOT / "benchmark" / "corpus"
CACHE_DIR = CORPUS_DIR / "cache"
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_PAUSE_S = 0.4          # NCBI allows 3 requests/s without an API key
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\"'<>,;|)\]]+", re.IGNORECASE)


@dataclass
class Paper:
    """One drawn paper and the features the quotas look at."""
    pmcid: str
    stratum: str
    journal: str
    version: str = ""
    doi: str = ""
    title: str = ""
    licence: str = ""
    pages: int = 0
    draw_rank: int = 0                 # position in its stratum's shuffled list
    flags: dict = field(default_factory=dict)
    role: str = "scored"               # scored | pilot


def main() -> int:
    """Build the manifest: list, exclude, shuffle, draw, balance, download."""
    global SEED, PILOT_PER_STRATUM, DATA_ROOT
    args = parse_arguments()
    DATA_ROOT = Path(args.data_root) if args.data_root else V6_ROOT
    if args.a8:
        SEED, PILOT_PER_STRATUM = A8_SEED, 0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Step 1: everything we have ever developed against is excluded.
    excluded_dois = load_exclusion_dois()
    if args.a8:
        excluded_dois |= manifest_dois()        # the 50 papers of the first draw
    print(f"excluded DOIs found in this repository: {len(excluded_dois)}")
    # Step 2: list and shuffle each stratum's candidates once, with the seed.
    shuffled = {name: shuffle_candidates(name, journals)
                for name, journals in STRATA.items()}
    # Step 3: walk each list, keep the first eligible papers.
    walkers = {name: EligibleWalker(name, ids, excluded_dois)
               for name, ids in shuffled.items()}
    scored = {name: [w.next_eligible() for _ in range(SCORED_PER_STRATUM)]
              for name, w in walkers.items()}
    # Step 4: meet the protocol's minimums by swaps (A1 rule 6).
    shortfalls = balance_quotas(scored, walkers)
    # Step 5: the pilot papers are the next eligible ones in each list.
    pilot = {name: [w.next_eligible() for _ in range(PILOT_PER_STRATUM)]
             for name, w in walkers.items()}
    for papers in pilot.values():
        for paper in papers:
            paper.role = "pilot"
    # Step 6: write the manifest and the quota report.
    all_papers = [p for ps in scored.values() for p in ps] + \
                 [p for ps in pilot.values() for p in ps]
    if args.a8:
        write_a8_manifest(all_papers, walkers, shortfalls)
    else:
        write_manifest(all_papers, walkers, shortfalls)
    return 0


# ---------------------------------------------------------------------------
# Helpers (after main, per the project rule)
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a8", action="store_true",
                        help="amendment A8: a fresh scored set (see the module docstring)")
    parser.add_argument("--data-root", default=None,
                        help="checkout holding the exclusion sources (default: this one)")
    return parser.parse_args()


def manifest_dois() -> set[str]:
    """Every DOI in the existing manifest (A8: none may be drawn again)."""
    with open(CORPUS_DIR / "MANIFEST.tsv", encoding="utf-8") as f:
        return {clean_doi(r["doi"]) for r in csv.DictReader(f, delimiter="\t") if r["doi"]}


def load_exclusion_dois() -> set[str]:
    """Every DOI printed anywhere in our ledger, corpus outputs, eval keys and
    contract examples. Broad on purpose: a paper we ever looked at is out."""
    sources = [DATA_ROOT / "persistence" / "semantic_ledger.csv",
               DATA_ROOT / "run_outputs", DATA_ROOT / "eval",
               DATA_ROOT / "contract" / "examples"]
    found: set[str] = set()
    for source in sources:
        files = [source] if source.is_file() else \
            [f for f in source.rglob("*") if f.suffix.lower() in
             {".json", ".csv", ".tsv", ".md", ".txt"}] if source.exists() else []
        for path in files:
            if path.stat().st_size > 60_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            found.update(clean_doi(d) for d in DOI_PATTERN.findall(text))
    return found


def shuffle_candidates(stratum: str, journals: list[str]) -> list[tuple[str, str]]:
    """All CC BY candidates of a stratum as (pmcid number, journal), sorted then
    shuffled once with the seed, so the order never depends on NCBI's order."""
    candidates: set[tuple[str, str]] = set()
    for journal in journals:
        for year in YEARS:
            for pmc_number in esearch_pmc(journal, year):
                candidates.add((pmc_number, journal))
    ordered = sorted(candidates, key=lambda c: int(c[0]))
    random.Random(f"{SEED}:{stratum}").shuffle(ordered)
    print(f"{stratum}: {len(ordered)} candidates")
    return ordered


class EligibleWalker:
    """Walks one stratum's shuffled list and hands out eligible papers in
    order, remembering why each skipped paper was skipped."""

    def __init__(self, stratum: str, candidates: list[tuple[str, str]],
                 excluded_dois: set[str]):
        self.stratum = stratum
        self.candidates = candidates
        self.excluded_dois = excluded_dois
        self.position = 0
        self.skipped: dict[str, int] = {}

    def next_eligible(self, needs_flag: str | None = None,
                      max_looks: int = 30) -> Paper | None:
        """The next eligible paper; with `needs_flag`, the next that has it,
        giving up after `max_looks` eligible papers without it (A3)."""
        looks = 0
        while self.position < len(self.candidates):
            if needs_flag and looks >= max_looks:
                return None
            pmc_number, journal = self.candidates[self.position]
            self.position += 1
            paper, reason = check_paper(pmc_number, journal, self.stratum,
                                        self.excluded_dois)
            if paper is None:
                self.skipped[reason] = self.skipped.get(reason, 0) + 1
                continue
            paper.draw_rank = self.position
            if needs_flag and not paper.flags.get(needs_flag):
                looks += 1
                self.skipped["kept back: lacks needed flag"] = \
                    self.skipped.get("kept back: lacks needed flag", 0) + 1
                continue
            return paper
        return None


def check_paper(pmc_number: str, journal: str, stratum: str,
                excluded_dois: set[str]) -> tuple[Paper | None, str]:
    """Eligibility (protocol §7, A1): CC BY, not retracted, research article,
    has a body, not already used by us; Nat Commun must be biology."""
    pmcid = f"PMC{pmc_number}"
    version = latest_version(pmcid)
    if not version:
        return None, "not in bucket"
    record = json.loads(fetch_cached(f"{BUCKET}/{version}/{version}.json",
                                     f"{version}.json"))
    if record.get("license_code") != "CC BY":
        return None, f"licence {record.get('license_code')}"
    if record.get("is_retracted") or record.get("is_manuscript"):
        return None, "retracted or author manuscript"
    if not record.get("pdf_url") or not record.get("xml_url"):
        return None, "no PDF or no XML"
    doi = clean_doi(record.get("doi") or "")
    if doi and doi in excluded_dois:
        return None, "already used in development"
    xml_text = fetch_cached(f"{BUCKET}/{version}/{version}.xml", f"{version}.xml")
    root = parse_jats(xml_text)
    if root is None:
        return None, "XML does not parse"
    if root.get("article-type") != "research-article":
        return None, f"article-type {root.get('article-type')}"
    if root.find(".//body") is None or not root.findall(".//body//p"):
        return None, "no body in XML"
    if journal == "Nat Commun" and not is_biology_by_mesh(record.get("pmid")):
        return None, "Nat Commun not biology (MeSH)"
    paper = Paper(pmcid=pmcid, stratum=stratum, journal=journal,
                  version=version, doi=doi, title=record.get("title", ""),
                  licence=record["license_code"])
    paper.flags = jats_flags(root)
    paper.pages, two_column = pdf_layout(version)
    paper.flags["two_column"] = two_column
    paper.flags["one_column"] = not two_column
    return paper, "eligible"


def balance_quotas(scored: dict[str, list[Paper]],
                   walkers: dict[str, EligibleWalker]) -> dict[str, int]:
    """A1 rule 6: for each unmet minimum, swap the latest-drawn paper lacking
    the feature for the next eligible paper with it, stratum by stratum."""
    shortfalls = {}
    for flag, minimum in QUOTAS.items():
        for stratum, papers in scored.items():
            if count_flag(scored, flag) >= minimum:
                break
            lacking = [p for p in papers if p and not p.flags.get(flag)]
            if not lacking:
                continue
            replacement = walkers[stratum].next_eligible(needs_flag=flag)
            if replacement is None:
                continue
            victim = max(lacking, key=lambda p: p.draw_rank)
            papers[papers.index(victim)] = replacement
            print(f"quota {flag}: {stratum} swapped {victim.pmcid} -> "
                  f"{replacement.pmcid}")
        short = minimum - count_flag(scored, flag)
        if short > 0:
            shortfalls[flag] = short
    return shortfalls


def count_flag(scored: dict[str, list[Paper]], flag: str) -> int:
    return sum(1 for ps in scored.values() for p in ps if p and p.flags.get(flag))


def jats_flags(root: ET.Element) -> dict:
    """Read the quota features from the answer key itself (A1 rule 7)."""
    flags = {"has_cell_table": any(tw.find(".//table") is not None
                                   for tw in root.iter("table-wrap"))}
    flags["multi_panel_figure"] = any(
        len(panel_letters(fig)) >= 3 for fig in root.iter("fig"))
    marker_texts = ["".join(x.itertext()).strip()
                    for x in root.iter("xref") if x.get("ref-type") == "bibr"]
    numbered = bool(marker_texts) and all(
        re.fullmatch(r"[\d\s,\-–—]+", t or "0") for t in marker_texts)
    flags["numbered"] = numbered
    flags["author_year"] = bool(marker_texts) and not numbered
    reference_count = len(root.findall(".//ref-list/ref"))
    flags["reference_count"] = reference_count
    flags["refs_60_plus"] = reference_count >= 60
    flags["figure_count"] = len(list(root.iter("fig")))
    flags["table_count"] = len(list(root.iter("table-wrap")))
    return flags


def panel_letters(fig: ET.Element) -> set[str]:
    """Distinct panel letters named in a figure caption: (A), bold A, 'A,'."""
    caption = fig.find("caption")
    if caption is None:
        return set()
    text = "".join(caption.itertext())
    letters = set(re.findall(r"\(([A-L])\)", text))
    letters |= set(re.findall(r"(?:^|[\s.;])([A-L])[,:]\s", text))
    for bold in caption.iter("bold"):
        token = "".join(bold.itertext()).strip(" ().,:")
        if re.fullmatch(r"[A-L]", token):
            letters.add(token)
    return letters


def pdf_layout(version: str) -> tuple[int, bool]:
    """Page count and column layout from the PDF (amendment A3).

    For biology researchers: in a two-column page almost no line of text runs
    across the middle of the page (that is the gutter); in a one-column page
    most do. So on pages 2-4 we group pypdfium2's text runs into lines by
    their vertical position and count the share of lines that cross the
    page centre. Two-column when fewer than 30% of lines cross it."""
    import pypdfium2 as pdfium   # Prajnya's own PDF library (Apache-2.0)
    pdf_path = CACHE_DIR / f"{version}.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(http_get(f"{BUCKET}/{version}/{version}.pdf"))
    document = pdfium.PdfDocument(str(pdf_path))
    page_count = len(document)
    crossing, lines = 0, 0
    for index in range(1, min(page_count, 4)) or [0]:
        page_crossing, page_lines = centre_crossing_lines(document[index])
        crossing += page_crossing
        lines += page_lines
    document.close()
    return page_count, lines > 0 and crossing / lines < 0.30


def centre_crossing_lines(page) -> tuple[int, int]:
    """(lines crossing the page centre, all lines) on one page. A line is the
    set of text runs whose vertical centres fall in the same 3-point band."""
    width = page.get_width()
    textpage = page.get_textpage()
    bands: dict[int, list[tuple[float, float]]] = {}
    for i in range(textpage.count_rects()):
        left, bottom, right, top = textpage.get_rect(i)
        if right - left < 2:
            continue
        bands.setdefault(round((bottom + top) / 2 / 3), []).append((left, right))
    centre = width / 2
    crossing = sum(1 for spans in bands.values() if line_crosses(spans, centre))
    return crossing, len(bands)


def line_crosses(spans: list[tuple[float, float]], centre: float) -> bool:
    """True when the line's text runs reach 20 points either side of the
    centre with no gap wider than 6 points (a word space) at the centre. Two
    columns leave a gutter of 10-30 points there."""
    if min(l for l, _ in spans) > centre - 20 or max(r for _, r in spans) < centre + 20:
        return False
    if any(l <= centre <= r for l, r in spans):
        return True
    gap_left = max((r for _, r in spans if r < centre), default=None)
    gap_right = min((l for l, _ in spans if l > centre), default=None)
    return gap_left is not None and gap_right is not None and gap_right - gap_left <= 6


def is_biology_by_mesh(pmid) -> bool:
    """A1 rule 3: a Nat Commun paper counts as biology if PubMed gives it an
    organism heading such as Animals or Humans."""
    if not pmid:
        return False
    xml_text = fetch_cached(f"{EUTILS}/efetch.fcgi?db=pubmed&retmode=xml&id={pmid}"
                            "&tool=prajnya_benchmark", f"pubmed_{pmid}.xml",
                            pause=NCBI_PAUSE_S)
    headings = set(re.findall(r'<DescriptorName[^>]*>([^<]+)</DescriptorName>',
                              xml_text))
    return bool(headings & MESH_BIOLOGY)


def esearch_pmc(journal: str, year: int) -> list[str]:
    """PMC numbers of CC BY articles in one journal and year (cached)."""
    term = f'"{journal}"[journal] AND "cc by license"[filter] AND {year}[pdat]'
    query = urllib.parse.urlencode({"db": "pmc", "term": term, "retmax": 10000,
                                    "retmode": "json", "tool": "prajnya_benchmark"})
    name = f"esearch_{journal.replace(' ', '_')}_{year}.json"
    data = json.loads(fetch_cached(f"{EUTILS}/esearch.fcgi?{query}", name,
                                   pause=NCBI_PAUSE_S))
    return data["esearchresult"]["idlist"]


def latest_version(pmcid: str) -> str:
    """The highest version folder in the bucket, e.g. 'PMC123.2' (cached)."""
    listing = fetch_cached(f"{BUCKET}/?list-type=2&delimiter=/&prefix={pmcid}.",
                           f"list_{pmcid}.xml")
    versions = re.findall(rf"<Prefix>({pmcid}\.(\d+))/</Prefix>", listing)
    if not versions:
        return ""
    return max(versions, key=lambda v: int(v[1]))[0]


def parse_jats(xml_text: str) -> ET.Element | None:
    """Parse JATS; ElementTree does not fetch external DTDs or entities."""
    try:
        return ET.fromstring(xml_text.encode("utf-8"))
    except ET.ParseError:
        return None


def fetch_cached(url: str, name: str, pause: float = 0.0) -> str:
    """GET a text resource once and keep it in the cache, so re-runs give the
    identical draw even if the bucket or NCBI changes."""
    path = CACHE_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    time.sleep(pause)
    text = http_get(url).decode("utf-8", errors="replace")
    path.write_text(text, encoding="utf-8", newline="")
    return text


def http_get(url: str, attempts: int = 3) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "prajnya-benchmark/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return b""


def clean_doi(doi: str) -> str:
    return doi.strip().rstrip(".").lower()


def write_manifest(papers: list[Paper | None], walkers: dict[str, EligibleWalker],
                   shortfalls: dict[str, int]) -> None:
    """MANIFEST.tsv (one row per paper) + DRAW_REPORT.json (counts, skips,
    quotas met or short)."""
    rows = [p for p in papers if p]
    columns = ["role", "stratum", "pmcid", "version", "doi", "journal", "licence",
               "pages", "draw_rank", "two_column", "has_cell_table",
               "multi_panel_figure", "numbered", "author_year", "reference_count",
               "figure_count", "table_count", "title"]
    with open(CORPUS_DIR / "MANIFEST.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for p in rows:
            record = {**asdict(p), **p.flags}
            writer.writerow([record.get(c, "") for c in columns])
    scored = [p for p in rows if p.role == "scored"]
    report = {
        "seed": SEED,
        "scored": len(scored), "pilot": len(rows) - len(scored),
        "quota_counts": {flag: sum(1 for p in scored if p.flags.get(flag))
                         for flag in QUOTAS},
        "quota_minimums": QUOTAS, "shortfalls": shortfalls,
        "skipped_per_stratum": {n: w.skipped for n, w in walkers.items()},
    }
    (CORPUS_DIR / "DRAW_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="")
    print(json.dumps(report["quota_counts"]), "shortfalls:", shortfalls)


def write_a8_manifest(papers: list[Paper | None], walkers: dict[str, EligibleWalker],
                      shortfalls: dict[str, int]) -> None:
    """A8: the old scored rows become "seen", the pilots stay, the new draw is
    "scored". Same columns; the report goes to DRAW_REPORT_A8.json."""
    with open(CORPUS_DIR / "MANIFEST.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns, old_rows = reader.fieldnames, list(reader)
    for row in old_rows:
        if row["role"] == "scored":
            row["role"] = "seen"
    new_rows = [{**asdict(p), **p.flags} for p in papers if p]
    with open(CORPUS_DIR / "MANIFEST.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for record in old_rows + new_rows:
            writer.writerow([record.get(c, "") for c in columns])
    drawn = [p for p in papers if p]
    report = {
        "amendment": "A8", "seed": SEED, "scored": len(drawn),
        "quota_counts": {flag: sum(1 for p in drawn if p.flags.get(flag)) for flag in QUOTAS},
        "quota_minimums": QUOTAS, "shortfalls": shortfalls,
        "skipped_per_stratum": {n: w.skipped for n, w in walkers.items()},
    }
    (CORPUS_DIR / "DRAW_REPORT_A8.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="")
    print(json.dumps(report["quota_counts"]), "shortfalls:", shortfalls)


if __name__ == "__main__":
    sys.exit(main())
