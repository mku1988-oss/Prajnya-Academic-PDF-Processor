#!/usr/bin/env python
"""
Draw T38d's confirmation set: the next 10 eligible papers after T38b's,
by A11's own seeded walk (eval/text_first_body/GATE_T38d.md).

For biology researchers: this replays A11's walk (checked against
MANIFEST_A11.tsv), then T38b's draw (checked against MANIFEST_T38B.tsv), and
only if both reproduce paper for paper takes the next 10 the same way. Nobody
chooses them, and they have not been looked at.

Writes benchmark/corpus/MANIFEST_T38D.tsv (role `confirm_t38d`), which every
later benchmark draw must exclude (PROTOCOL exclusion rule).

Run (from src/v6):  python benchmark/corpus_tools/draw_confirm_t38d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_manifest_a11 as a11  # noqa: E402
import draw_confirm_t38b as t38b  # noqa: E402
from subject_tree import SubjectTree  # noqa: E402

CONFIRM = 10
ROLE = "confirm_t38d"
OUT = a11.CORPUS_DIR / "MANIFEST_T38D.tsv"


def main() -> int:
    a11.TREE = SubjectTree()
    published = t38b.read_manifest(a11.CORPUS_DIR / "MANIFEST_A11.tsv")
    excluded = a11.exclusions() - {row["doi"] for row in published}
    walkers, replayed = t38b.replay_a11(excluded)
    if [p.pmcid for p in replayed] != [row["pmcid"] for row in published]:
        print("STOP: the replay does not reproduce MANIFEST_A11.tsv; nothing drawn.")
        return 1
    pilots = sum(1 for p in replayed if p.role == "pilot")
    t38b_rows = [r["pmcid"] for r in t38b.read_manifest(t38b.OUT)]
    previous, row = next_in_turn(walkers, CONFIRM, start_row=pilots, role=t38b.ROLE)
    if [p.pmcid for p in previous] != t38b_rows:
        print("STOP: the replay does not reproduce MANIFEST_T38B.tsv; nothing drawn.")
        print("  replay   ", [p.pmcid for p in previous])
        print("  published", t38b_rows)
        return 1
    print(f"replay reproduces MANIFEST_A11.tsv ({len(replayed)}) and MANIFEST_T38B.tsv ({len(previous)})")
    confirm, _ = next_in_turn(walkers, CONFIRM, start_row=row, role=ROLE)
    t38b.OUT = OUT                       # write_manifest writes to t38b.OUT
    t38b.write_manifest(confirm)
    for paper in confirm:
        print(f"{paper.pmcid:12s} {paper.field:24s} {paper.journal:26s} {paper.pages:3d} pages")
    return 0


# ---------------------------------------------------------------------------
# Helpers (after the main function, per the project rule)
# ---------------------------------------------------------------------------

def next_in_turn(walkers: dict, wanted: int, start_row: int, role: str):
    """`draw_confirm_t38b.next_in_turn`, also returning the row it stopped at,
    so a second draw resumes exactly where the first left off."""
    rows = [(field_name, [journal for journal, _ in cells]) for field_name, cells in a11.QUOTAS]
    taken, row, tried = [], start_row, 0
    while len(taken) < wanted and tried < len(rows) * 4:
        field_name, journals = rows[row % len(rows)]
        for journal in journals:
            paper = walkers[(field_name, journal)].next()
            if paper:
                paper.role = role
                taken.append(paper)
                break
        row += 1
        tried += 1
    return taken, row


if __name__ == "__main__":
    sys.exit(main())
