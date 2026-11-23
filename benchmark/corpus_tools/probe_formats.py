"""
Format probe for the next benchmark round (v6.9 item 9), run BEFORE the draw.

For biology researchers: the next round draws papers from five journals and
labels each paper biology, physics or chemistry from its own subject headings.
Before the draw rule can be written, one recent CC BY article per journal is
looked at to see HOW each journal writes its subject headings in the JATS XML,
and whether PMC's open-data bucket holds the figure images (needed to score
figure crops). Nothing is scored; no tool is run.

The probed papers are written to `corpus/PROBED.json` and are excluded from
the draw, since they have been looked at.

Run:  python benchmark/corpus_tools/probe_formats.py
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "corpus" / "PROBED.json"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
JOURNALS = ["Nature", "Proc Natl Acad Sci U S A", "Sci Adv", "Sci Rep", "Front Chem"]
HEADERS = {"User-Agent": "prajnya-benchmark/1.0"}


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def first_article(journal: str) -> str:
    term = f'"{journal}"[journal] AND "cc by license"[filter] AND 2024[pdat]'
    query = urllib.parse.urlencode({"db": "pmc", "term": term, "retmax": 1, "retmode": "json"})
    time.sleep(0.4)
    return "PMC" + json.loads(get(f"{EUTILS}/esearch.fcgi?{query}"))["esearchresult"]["idlist"][0]


def probe(journal: str) -> dict:
    pmcid = first_article(journal)
    listing = get(f"{BUCKET}/?list-type=2&prefix={pmcid}.").decode("utf-8")
    keys = re.findall(r"<Key>([^<]+)</Key>", listing)
    version = sorted({k.split("/")[0] for k in keys}, key=lambda v: int(v.rsplit(".", 1)[1]))[-1]
    files = [k.split("/", 1)[1] for k in keys if k.startswith(version + "/")]
    root = ET.fromstring(get(f"{BUCKET}/{version}/{version}.xml"))
    groups = []
    for group in root.iter("subj-group"):
        groups.append({"type": group.get("subj-group-type"),
                       "subjects": [s.text for s in group.iter("subject") if s.text]})
    keywords = [k.text for k in root.iter("kwd") if k.text][:6]
    doi = next((a.text for a in root.iter("article-id") if a.get("pub-id-type") == "doi"), None)
    graphics = [g.get("{http://www.w3.org/1999/xlink}href") for g in root.iter("graphic")][:4]
    maths = sum(1 for _ in root.iter("{http://www.w3.org/1998/Math/MathML}math"))
    return {"journal": journal, "pmcid": version, "doi": doi, "subj_groups": groups,
            "keywords": keywords, "files": sorted(files)[:15], "file_count": len(files),
            "graphic_hrefs": graphics, "mathml_formulas": maths}


def main() -> int:
    found = [probe(journal) for journal in JOURNALS]
    OUT.write_text(json.dumps({"why": "format probe before the item-9 draw; excluded from it",
                               "papers": found}, indent=1, ensure_ascii=False), encoding="utf-8")
    for paper in found:
        print(json.dumps(paper, ensure_ascii=False)[:900])
    return 0


if __name__ == "__main__":
    sys.exit(main())
