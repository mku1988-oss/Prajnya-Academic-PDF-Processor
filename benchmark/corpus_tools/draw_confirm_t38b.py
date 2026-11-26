#!/usr/bin/env python
"""
Draw T38b's confirmation set: the next 10 eligible papers after the A11
pilots, by A11's own seeded walk (eval/text_first_body/GATE_T38b.md).

For biology researchers: A11 shuffled each (field, journal) list of papers
once with a fixed seed and took eligible papers in that order, first for the
40 scored papers, then 10 pilots "in turn across the rows". This replays that
walk exactly, CHECKS that it reproduces MANIFEST_A11.tsv paper for paper (if
not, it stops and draws nothing), and then takes the next 10 papers the same
way. Nobody chooses them, and they have not been looked at.

Writes benchmark/corpus/MANIFEST_T38B.tsv (role `confirm_t38b`), which every
later benchmark draw must exclude (PROTOCOL exclusion rule).

Run (from src/v6):  python benchmark/corpus_tools/draw_confirm_t38b.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_manifest_a11 as a11  # noqa: E402
from subject_tree import SubjectTree  # noqa: E402

CONFIRM = 10
ROLE = "confirm_t38b"
OUT = a11.CORPUS_DIR / "MANIFEST_T38B.tsv"


def main() -> int:
    a11.TREE = SubjectTree()
    published = read_manifest(a11.CORPUS_DIR / "MANIFEST_A11.tsv")
    excluded = a11.exclusions() - {row["doi"] for row in published}
    walkers, replayed = replay_a11(excluded)
    if [p.pmcid for p in replayed] != [row["pmcid"] for row in published]:
        print("STOP: the replay does not reproduce MANIFEST_A11.tsv; nothing drawn.")
        for got, want in zip([p.pmcid for p in replayed], [r["pmcid"] for r in published]):
            print(f"  replay {got}  published {want}" + ("" if got == want else "  <-- differs"))
        return 1
    print(f"replay reproduces MANIFEST_A11.tsv: {len(replayed)} papers, same order")
    # The pilots took one paper from each of 10 rows in turn (checked against
    # their fields below), so the walk resumes at the 11th row.
    pilot_fields = [p.field for p in replayed if p.role == "pilot"]
    rows = [field_name for field_name, _ in a11.QUOTAS]
    assert pilot_fields == [rows[i % len(rows)] for i in range(len(pilot_fields))], pilot_fields
    confirm = next_in_turn(walkers, CONFIRM, start_row=len(pilot_fields))
    write_manifest(confirm)
    for paper in confirm:
        print(f"{paper.pmcid:12s} {paper.field:24s} {paper.journal:26s} {paper.pages:3d} pages")
    return 0


# ---------------------------------------------------------------------------
# Helpers (after the main function, per the project rule)
# ---------------------------------------------------------------------------

def read_manifest(path: Path) -> list:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def replay_a11(excluded: set):
    """A11's main() walk (quotas, then pilots) without writing anything."""
    walkers, drawn = {}, []
    for field_name, cells in a11.QUOTAS:
        carry = 0
        for journal, wanted in cells:
            walker = walkers.setdefault((field_name, journal),
                                        a11.Walker(field_name, journal, excluded))
            need = wanted + carry
            got = [paper for paper in (walker.next() for _ in range(need)) if paper]
            drawn.extend(got)
            carry = need - len(got)
    drawn.extend(a11.draw_pilots(walkers))
    return walkers, drawn


def next_in_turn(walkers: dict, wanted: int, start_row: int) -> list:
    """A11 rule 5 continued: the next eligible papers, in turn across the rows,
    resuming at the row after the last pilot's."""
    rows = [(field_name, [journal for journal, _ in cells]) for field_name, cells in a11.QUOTAS]
    taken, row, tried = [], start_row, 0
    while len(taken) < wanted and tried < len(rows) * 4:
        field_name, journals = rows[row % len(rows)]
        for journal in journals:
            paper = walkers[(field_name, journal)].next()
            if paper:
                paper.role = ROLE
                taken.append(paper)
                break
        row += 1
        tried += 1
    return taken


def write_manifest(papers: list) -> None:
    columns = ["role", "field", "journal", "pmcid", "version", "doi", "licence", "pages",
               "draw_rank", "figures", "tables", "display_formulas", "inline_formulas",
               "references", "title"]
    with open(OUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for p in papers:
            writer.writerow([p.role, p.field, p.journal, p.pmcid, p.version, p.doi, p.licence,
                             p.pages, p.draw_rank, *(p.counts.get(k, 0) for k in
                             ("figures", "tables", "display_formulas", "inline_formulas",
                              "references")), p.title])


if __name__ == "__main__":
    sys.exit(main())
