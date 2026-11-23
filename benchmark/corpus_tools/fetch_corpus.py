#!/usr/bin/env python
"""
Download the benchmark's papers, exactly as listed in the manifest.

For biology researchers: the manifest (benchmark/corpus/MANIFEST.tsv) names
every paper by its PMC version (for example PMC10231790.1). This script
fetches that version's PDF and its JATS XML (the publisher's structured
version, used as the answer key) from PMC's Open Access bucket into
benchmark/corpus/cache/. A file already there is not downloaded again.

Fetching by version, not by searching PMC again, gives everyone the same
files, even after PMC adds new papers or new versions.

Run (from the repository root):
    python benchmark/corpus_tools/fetch_corpus.py                 # every paper
    python benchmark/corpus_tools/fetch_corpus.py --roles scored  # the 40 scored papers
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
CACHE_DIR = CORPUS_DIR / "cache"
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
FILE_KINDS = ("pdf", "xml")


def main() -> int:
    """Fetch the PDF and XML of every manifest paper in the chosen roles."""
    args = parse_arguments()
    papers = manifest_papers(args.roles, args.manifest)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fetched, kept = 0, 0
    for paper in papers:
        for kind in FILE_KINDS:
            if fetch_one(paper["version"], kind):
                fetched += 1
            else:
                kept += 1
    print(f"{len(papers)} papers: {fetched} files downloaded, {kept} already present "
          f"in {CACHE_DIR}")
    return 0


# ---------------------------------------------------------------------------
# Helpers (after main, per the project rule)
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", default="scored,pilot,seen",
                        help="comma-separated manifest roles to fetch (default: all)")
    parser.add_argument("--manifest", default="MANIFEST.tsv",
                        help="the manifest under corpus/ (round 2: MANIFEST_A11.tsv)")
    args = parser.parse_args()
    args.roles = set(args.roles.split(","))
    return args


def manifest_papers(roles: set[str], manifest: str = "MANIFEST.tsv") -> list[dict]:
    """The manifest rows whose role is one of `roles`."""
    with open(CORPUS_DIR / manifest, encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f, delimiter="\t") if row["role"] in roles]


def fetch_one(version: str, kind: str) -> bool:
    """Download one file of one paper; False if it was already in the cache."""
    path = CACHE_DIR / f"{version}.{kind}"
    if path.exists() and path.stat().st_size > 0:
        return False
    data = http_get(f"{BUCKET}/{version}/{version}.{kind}")
    path.write_bytes(data)
    print(f"  {path.name}  {len(data) / 1024:.0f} KB", flush=True)
    return True


def http_get(url: str, attempts: int = 3) -> bytes:
    """GET with a few retries (the bucket occasionally drops a connection)."""
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "prajnya-benchmark/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return b""


if __name__ == "__main__":
    sys.exit(main())
