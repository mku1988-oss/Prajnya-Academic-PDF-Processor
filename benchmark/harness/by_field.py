#!/usr/bin/env python
"""
Benchmark means per field (stratum), to see where each tool is strong or weak.

For biology researchers: the A11 round drew 40 papers from six fields (four
biology fields, physics, chemistry). This reads the per-paper scores that
run_scoring.py wrote and prints, for each field and for "biology" as a whole,
the mean of each measure per tool, as a Markdown table. Nothing is re-scored.

Run (from src/v6):
    python benchmark/harness/by_field.py --run a11 > benchmark/results/A11_by_field.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
BIOLOGY = ["biochemistry", "biophysics", "developmental biology", "computational biology"]
GROUPS = [("Biology (all four fields)", BIOLOGY)] + [(f.capitalize(), [f]) for f in BIOLOGY] \
    + [("Physics", ["physics"]), ("Chemistry", ["chemistry"])]
TOOLS = ["prajnya", "prajnya_online", "grobid", "grobid_consolidated", "docling",
         "docling_formula", "marker", "mineru"]
SHORT = {"prajnya": "Prajnya", "prajnya_online": "Prajnya online", "grobid": "GROBID",
         "grobid_consolidated": "GROBID +lookup", "docling": "Docling",
         "docling_formula": "Docling formulas (A12)", "marker": "Marker", "mineru": "MinerU"}
MEASURES = [
    ("Title exact", "title_exact"), ("Authors F1", "authors_f1"),
    ("Author order", "authors_order"), ("DOI", "doi"), ("Year", "year"),
    ("Journal", "journal"),
    ("Body coverage", "body_coverage"), ("Body similarity", "body_similarity"),
    ("Figures found", "figures_recall"), ("Caption similarity", "figure_caption_similarity"),
    ("Crop right", "crop_right"), ("Crop IoU", "crop_iou"),
    ("Tables found", "tables_recall"), ("TEDS", "teds"),
    ("Display formulas", "formula_display"), ("Inline formulas", "formula_inline"),
    ("References found", "references_recall"), ("Reference title", "ref_title"),
    ("Reference DOI", "ref_doi"), ("Citation precision", "citation_precision"),
    ("Citation recall", "citation_recall"),
    ("Figure mentions found", "figure_mention_recall"),
    ("Table mentions found", "table_mention_recall"),
]


def main() -> int:
    args = parse_arguments()
    rows = load_rows(args.run, args.file)
    tools = [t for t in TOOLS if any(r["tool"] == t for r in rows)]
    print(f"# A11 benchmark by field (run {args.run}, {args.file})\n")
    print("Mean per measure; best per row in bold; `-` = the tool does not produce it "
          "or no paper of the field has it; n = papers.\n")
    for title, fields in GROUPS:
        group = [r for r in rows if r["stratum"] in fields]
        print(group_table(title, group, tools))
    return 0


# ---------------------------------------------------------------------------
# Helpers (after main, per the project rule)
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="a11")
    parser.add_argument("--file", default="per_paper_scored_key.jsonl")
    return parser.parse_args()


def load_rows(run: str, name: str) -> list[dict]:
    path = BENCH / "outputs" / f"run{run}" / "scores" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def mean_of(rows: list[dict], tool: str, measure: str) -> float | None:
    values = [r[measure] for r in rows if r["tool"] == tool and r.get(measure) is not None]
    return statistics.fmean(values) if values else None


def group_table(title: str, rows: list[dict], tools: list[str]) -> str:
    papers = len({r["pmcid"] for r in rows})
    lines = [f"## {title} (n = {papers})\n",
             "| Measure | " + " | ".join(SHORT[t] for t in tools) + " |",
             "|---|" + "---|" * len(tools)]
    for label, measure in MEASURES:
        means = [mean_of(rows, t, measure) for t in tools]
        if all(m is None for m in means):
            continue
        lines.append(f"| {label} | " + " | ".join(cell(m, means) for m in means) + " |")
    return "\n".join(lines) + "\n"


def cell(value: float | None, row: list[float | None]) -> str:
    if value is None:
        return "-"
    best = max(m for m in row if m is not None)
    text = f"{value:.3f}"
    return f"**{text}**" if abs(value - best) < 5e-4 else text


if __name__ == "__main__":
    raise SystemExit(main())
