#!/usr/bin/env python
"""
The A11 round's tables for BENCHMARK.md, as Markdown, from the saved scores.

For biology researchers: prints (1) every measure for every tool setting over
the 40 scored papers, with the number of papers behind each mean; (2) the main
measures per field group (biology, physics, chemistry); (3) the cost table
(seconds per page, median seconds per paper, peak GPU memory). Nothing is
re-scored; paste the output into BENCHMARK.md.

Run (from src/v6):  python benchmark/harness/a11_report.py > benchmark/results/A11_tables.md
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
RUN = BENCH / "outputs" / "runa11"
COLUMNS = [("text layer", "pdf_text_layer"), ("Prajnya local", "prajnya"),
           ("Prajnya online", "prajnya_online"), ("GROBID", "grobid"),
           ("GROBID consol.", "grobid_consolidated"), ("Docling", "docling"),
           ("Docling formulas", "docling_formula"), ("Marker", "marker"), ("MinerU", "mineru")]
SECTIONS = [
    ("Metadata", [("title exact", "title_exact"), ("authors found (surname F1)", "authors_f1"),
                  ("author list exact, in order", "authors_order"), ("DOI", "doi"),
                  ("year", "year"), ("journal", "journal")]),
    ("Body text", [("paragraphs found (coverage)", "body_coverage"),
                   ("text similarity", "body_similarity"),
                   ("reading order (Kendall τ)", "reading_order_tau")]),
    ("Figures", [("found under its printed label", "figures_recall"),
                 ("found by label or by caption", "figures_recall_with_caption_match"),
                 ("kept in one piece", "figures_kept_together"),
                 ("caption similarity", "figure_caption_similarity"),
                 ("crop right (IoU ≥ 0.8)", "crop_right"), ("crop IoU", "crop_iou")]),
    ("Tables", [("found under its printed label", "tables_recall"),
                ("TEDS (cells and structure)", "teds"), ("TEDS-S (structure only)", "teds_structure")]),
    ("Formulas", [("display formulas", "formula_display"), ("inline formulas", "formula_inline")]),
    ("References", [("entries found", "references_recall"), ("precision", "references_precision"),
                    ("title right", "ref_title"), ("year right", "ref_year"),
                    ("DOI right", "ref_doi"), ("first author right", "ref_first_author")]),
    ("In-text links", [("citations: precision", "citation_precision"),
                       ("citations: recall", "citation_recall"),
                       ("figure mentions: precision", "figure_mention_precision"),
                       ("figure mentions: recall", "figure_mention_recall"),
                       ("table mentions: precision", "table_mention_precision"),
                       ("table mentions: recall", "table_mention_recall")]),
]
GROUPS = {"biology (20)": ["biochemistry", "biophysics", "developmental biology",
                           "computational biology"],
          "physics (10)": ["physics"], "chemistry (10)": ["chemistry"]}
FIELD_MEASURES = [("figures found", "figures_recall"), ("crop right", "crop_right"),
                  ("table TEDS", "teds"), ("display formulas", "formula_display"),
                  ("inline formulas", "formula_inline"), ("body text similarity", "body_similarity"),
                  ("reference title right", "ref_title"), ("citations: precision", "citation_precision"),
                  ("citations: recall", "citation_recall"),
                  ("figure mentions: recall", "figure_mention_recall")]
COST = [("GROBID", "grobid", "CPU"), ("GROBID consolidated", "grobid_consolidated", "CPU + Crossref"),
        ("Docling", "docling", "GPU"), ("Docling, formulas on", "docling_formula", "GPU"),
        ("MinerU", "mineru", "GPU"), ("Marker", "marker", "GPU (llama.cpp)"),
        ("Prajnya local", "prajnya", "GPU (llama.cpp)")]


def main() -> int:
    rows = [json.loads(line) for line in (RUN / "scores" / "per_paper_scored_key.jsonl")
            .read_text(encoding="utf-8").splitlines() if line]
    for title, measures in SECTIONS:
        print(overall_table(title, measures, rows))
    print(field_table(rows))
    print(cost_table())
    return 0


# ---------------------------------------------------------------------------
# Helpers (after main, per the project rule)
# ---------------------------------------------------------------------------

def mean_and_n(rows: list[dict], column: str, measure: str, strata=None) -> tuple:
    vals = [r[measure] for r in rows if r["tool"] == column and r.get(measure) is not None
            and (strata is None or r["stratum"] in strata)]
    return (statistics.fmean(vals), len(vals)) if vals else (None, 0)


def overall_table(title: str, measures: list, rows: list[dict]) -> str:
    """One section; columns that support none of its measures are left out."""
    columns = [(n, c) for n, c in COLUMNS
               if any(mean_and_n(rows, c, m)[0] is not None for _, m in measures)]
    lines = [f"#### {title}\n", "| | " + " | ".join(n for n, _ in columns) + " |",
             "|---|" + "---|" * len(columns)]
    for label, measure in measures:
        cells = []
        for _, column in columns:
            mean, n = mean_and_n(rows, column, measure)
            cells.append("n/s" if mean is None else f"{mean:.3f}")
        papers = max(mean_and_n(rows, c, measure)[1] for _, c in columns)
        lines.append(f"| {label} ({papers}) | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def field_table(rows: list[dict]) -> str:
    """Each tool at its best setting per measure, per field group."""
    tools = {"Prajnya": ["prajnya", "prajnya_online"], "GROBID": ["grobid", "grobid_consolidated"],
             "Docling": ["docling", "docling_formula"], "Marker": ["marker"], "MinerU": ["mineru"]}
    lines = ["#### By field\n", "| | field | " + " | ".join(tools) + " |",
             "|---|---|" + "---|" * len(tools)]
    for label, measure in FIELD_MEASURES:
        for group, strata in GROUPS.items():
            cells = []
            for settings in tools.values():
                means = [mean_and_n(rows, s, measure, strata)[0] for s in settings]
                means = [m for m in means if m is not None]
                cells.append(f"{max(means):.3f}" if means else "n/s")
            lines.append(f"| {label} | {group} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def cost_table() -> str:
    with open(BENCH / "corpus" / "MANIFEST_A11.tsv", encoding="utf-8") as f:
        scored = {r["pmcid"] for r in csv.DictReader(f, delimiter="\t") if r["role"] == "scored"}
    timings = [json.loads(line) for line in (RUN / "timings.jsonl").read_text(encoding="utf-8")
               .splitlines() if line]
    lines = ["#### Cost (reported, not ranked)\n",
             "| | seconds per page | median seconds per paper | peak GPU memory | runs on |",
             "|---|---|---|---|---|"]
    for label, column, runs_on in COST:
        own = [t for t in timings if t.get("tool") == column and t.get("pmcid") in scored
               and "seconds" in t]
        per_page = sum(t["seconds"] for t in own) / sum(t["pages"] for t in own)
        median = statistics.median(t["seconds"] for t in own)
        vram = max(t.get("peak_vram_mib") or 0 for t in own) / 1024
        gpu = "none" if runs_on.startswith("CPU") else f"{vram:.1f} GB"
        lines.append(f"| {label} | {per_page:.2f} | {median:.1f} | {gpu} | {runs_on} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
