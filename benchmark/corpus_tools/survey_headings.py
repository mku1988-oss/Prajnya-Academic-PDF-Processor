"""
Survey of subject headings before the item-9 draw (biology sub-fields).

For biology researchers: the researcher asked for the 20 biology papers to be
5 biochemistry, 5 biophysics, 5 developmental biology and 5 computational
biology. A paper's sub-field must come from its own JATS headings, so this
counts which heading values PNAS and Science Advances actually use, on a
sample of CC BY papers. Only the headings are read (never the text), and
every surveyed paper is added to `corpus/PROBED.json`, which the draw
excludes.

Run:  python benchmark/corpus_tools/survey_headings.py [per_journal_per_year]
"""
import collections
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBED = HERE.parent / "corpus" / "PROBED.json"
SURVEY = HERE.parent / "corpus" / "HEADING_SURVEY.json"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
HEADERS = {"User-Agent": "prajnya-benchmark/1.0"}
JOURNALS = ["Sci Adv", "Proc Natl Acad Sci U S A"]
YEARS = [2019, 2021, 2023, 2025]


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def ids_for(journal: str, year: int, count: int) -> list:
    term = f'"{journal}"[journal] AND "cc by license"[filter] AND {year}[pdat]'
    query = urllib.parse.urlencode({"db": "pmc", "term": term, "retmax": count, "retmode": "json"})
    time.sleep(0.4)
    return json.loads(get(f"{EUTILS}/esearch.fcgi?{query}"))["esearchresult"]["idlist"]


def headings_of(pmcid: str):
    keys = re.findall(r"<Key>([^<]+)</Key>", get(f"{BUCKET}/?list-type=2&prefix={pmcid}.").decode())
    if not keys:
        return None
    version = sorted({k.split("/")[0] for k in keys}, key=lambda v: int(v.rsplit(".", 1)[1]))[-1]
    root = ET.fromstring(get(f"{BUCKET}/{version}/{version}.xml"))
    groups = {}
    for group in root.iter("subj-group"):
        groups.setdefault(group.get("subj-group-type") or "untyped", []).extend(
            s.text for s in group.iter("subject") if s.text)
    doi = next((a.text for a in root.iter("article-id") if a.get("pub-id-type") == "doi"), None)
    article = root.get("article-type")
    return version, doi, article, groups


def main() -> int:
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    probed = json.loads(PROBED.read_text(encoding="utf-8"))
    rows = []
    for journal in JOURNALS:
        for year in YEARS:
            for number in ids_for(journal, year, per):
                found = headings_of("PMC" + number)
                if found is None:
                    continue
                version, doi, article, groups = found
                rows.append({"journal": journal, "year": year, "pmcid": version, "doi": doi,
                             "article_type": article, "groups": groups})
                probed["papers"].append({"journal": journal, "pmcid": version, "doi": doi,
                                         "headings_only": True})
                time.sleep(0.1)
    PROBED.write_text(json.dumps(probed, indent=1, ensure_ascii=False), encoding="utf-8")
    SURVEY.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    for journal in JOURNALS:
        tally = collections.Counter()
        for row in rows:
            if row["journal"] != journal or row["article_type"] != "research-article":
                continue
            labels = row["groups"].get("heading", []) + row["groups"].get("field", [])
            for label in set(labels):
                tally[label] += 1
        print("==", journal, sum(1 for r in rows if r["journal"] == journal), "papers")
        for label, count in tally.most_common(40):
            print(f"  {count:4d}  {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
