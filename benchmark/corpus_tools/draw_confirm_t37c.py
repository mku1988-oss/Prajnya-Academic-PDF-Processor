#!/usr/bin/env python
"""
Draw T37c's confirmation set: the next 20 eligible papers after T38d's,
by A11's own seeded walk (eval/title_text_layer/GATE_T37c.md).

For biology researchers: this replays A11's walk (checked against
MANIFEST_A11.tsv), then T38b's draw (MANIFEST_T38B.tsv), then T38d's
(MANIFEST_T38D.tsv), and only if all three reproduce paper for paper takes
the next 20 the same way. Nobody chooses them, and they have not been looked
at.

Writes benchmark/corpus/MANIFEST_T37C.tsv (role `confirm_t37c`), which every
later benchmark draw must exclude (PROTOCOL exclusion rule).

Run (from src/v6):  python benchmark/corpus_tools/draw_confirm_t37c.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_manifest_a11 as a11  # noqa: E402
import draw_confirm_t38b as t38b  # noqa: E402
import draw_confirm_t38d as t38d  # noqa: E402
from subject_tree import SubjectTree  # noqa: E402

CONFIRM = 20
ROLE = "confirm_t37c"
OUT = a11.CORPUS_DIR / "MANIFEST_T37C.tsv"


def main() -> int:
    """Replay A11, T38b and T38d; draw the next 20 only if all three match."""
    a11.TREE = SubjectTree()
    published = t38b.read_manifest(a11.CORPUS_DIR / "MANIFEST_A11.tsv")
    # The papers being REPLAYED are not exclusions, or the replay would skip
    # them. T38d's draw only needed this for A11's papers; since then the T37
    # and T37b outputs under eval/ carry the T38b and T38d papers' DOIs, which
    # `exclusions()` (it scans eval/) now picks up, so they are taken out too.
    replayed_dois = {row["doi"] for manifest in
                     ("MANIFEST_A11.tsv", t38b.OUT.name, t38d.OUT.name)
                     for row in t38b.read_manifest(a11.CORPUS_DIR / manifest)}
    excluded = a11.exclusions() - replayed_dois
    walkers, replayed = t38b.replay_a11(excluded)
    if [p.pmcid for p in replayed] != [row["pmcid"] for row in published]:
        print("STOP: the replay does not reproduce MANIFEST_A11.tsv; nothing drawn.")
        return 1
    pilots = sum(1 for p in replayed if p.role == "pilot")

    row = pilots
    for manifest, role in ((t38b.OUT, t38b.ROLE), (t38d.OUT, t38d.ROLE)):
        expected = [r["pmcid"] for r in t38b.read_manifest(manifest)]
        previous, row = t38d.next_in_turn(walkers, len(expected), start_row=row, role=role)
        if [p.pmcid for p in previous] != expected:
            print(f"STOP: the replay does not reproduce {manifest.name}; nothing drawn.")
            print("  replay   ", [p.pmcid for p in previous])
            print("  published", expected)
            return 1
    print(f"replay reproduces MANIFEST_A11.tsv ({len(replayed)}), "
          f"MANIFEST_T38B.tsv and MANIFEST_T38D.tsv")

    confirm, _ = t38d.next_in_turn(walkers, CONFIRM, start_row=row, role=ROLE)
    t38b.OUT = OUT                       # write_manifest writes to t38b.OUT
    t38b.write_manifest(confirm)
    print(f"{len(confirm)} papers drawn -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
