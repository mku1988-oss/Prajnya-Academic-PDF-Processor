#!/usr/bin/env python
"""
Build the results tables (Markdown) from scores and timings (protocol §6, C6).

For biology researchers: this reads benchmark/outputs/run<N>/scores/
per_paper_<role>.jsonl and timings.jsonl and writes
benchmark/results/tables_<role>.md: one table per metric family (mean, with
the number of papers in brackets), the cost table, a per-publisher view of
the main metrics, and, when a second run exists, the determinism table.
`n/s` = the tool does not produce this. Nothing here ranks tools.

Run (from src/v6):
    python benchmark/harness/report.py --run 1 --role scored
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
TOOL_NAMES = {"pdf_text_layer": "Raw text layer (baseline)", "prajnya": "Prajnya (local)", "prajnya_online": "Prajnya (online)",
              "grobid": "GROBID", "grobid_consolidated": "GROBID (consolidated)",
              "docling": "Docling", "marker": "Marker", "mineru": "MinerU"}
NETWORK = {"prajnya_online": "DOI/title to Crossref, OpenAlex",
           "grobid_consolidated": "extracted fields to Crossref"}
SECTIONS = {
    "Metadata (§6.1)": ["title_exact", "authors_f1", "authors_order", "doi", "year", "journal"],
    "Abstract and body text (§6.2)": ["abstract_similarity", "body_coverage",
                                      "body_similarity", "reading_order_tau"],
    "Figures (§6.3)": ["figures_recall", "figures_recall_with_caption_match",
                       "figures_precision", "figure_caption_similarity",
                       "figures_kept_together", "figures_kept_together_with_caption_match"],
    "Tables (§6.4)": ["tables_recall", "tables_precision", "teds", "teds_structure",
                      "tables_unlabelled"],
    "References (§6.5)": ["references_recall", "references_recall_frozen_rule",
                          "references_precision", "ref_title",
                          "ref_year", "ref_doi", "ref_first_author"],
    "In-text links (§6.6, §6.7)": ["citation_precision", "citation_recall",
                                   "figure_mention_precision", "figure_mention_recall",
                                   "table_mention_precision", "table_mention_recall"],
}
BY_PUBLISHER = ["body_coverage", "figures_recall", "teds", "references_recall",
                "citation_recall"]


def main() -> int:
    """Write the Markdown tables for one run and role."""
    args = parse_arguments()
    run_dir = BENCH / "outputs" / f"run{args.run}"
    rows = read_jsonl(run_dir / "scores" / f"per_paper_{args.role}{key_suffix(args.key)}.jsonl")
    tools = [t for t in TOOL_NAMES if any(r["tool"] == t for r in rows)]
    parts = [f"# Results tables: run {args.run}, {args.role} papers\n",
             papers_line(rows, tools)]
    for title, metrics in SECTIONS.items():
        parts.append(metric_table(title, metrics, rows, tools))
    parts.append(cost_table(run_dir, args.role, tools))
    parts.append(publisher_table(rows, tools))
    second = BENCH / "outputs" / f"run{args.second}" / "common" if args.second else None
    if second is not None and second.exists():
        parts.append(determinism_table(run_dir / "common", second, rows, tools))
    out = BENCH / "results"
    out.mkdir(exist_ok=True)
    path = out / f"tables_run{args.run}_{args.role}{key_suffix(args.key)}.md"
    path.write_text("\n".join(parts), encoding="utf-8", newline="")
    print(f"wrote {path}")
    return 0


# ---------------------------------------------------------------------------
# Helpers (after main, per the project rule)
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="1")
    parser.add_argument("--role", default="scored")
    parser.add_argument("--key", default="",
                        help="answer-key folder the scores were made with ('key', 'key_frozen'); "
                             "empty reads run 1's original per_paper_<role>.jsonl")
    parser.add_argument("--second", default="",
                        help="the repeat run for the determinism table (§6.8), e.g. 3")
    return parser.parse_args()


def key_suffix(key: str) -> str:
    """Scores files carry the key folder in their name (amendment A7)."""
    return f"_{key}" if key else ""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def header(tools: list[str]) -> str:
    return ("| metric | " + " | ".join(TOOL_NAMES[t] for t in tools) + " |\n"
            + "|---|" + "---|" * len(tools))


def cell(values: list[float]) -> str:
    if not values:
        return "n/s"
    return f"{statistics.fmean(values):.3f} ({len(values)})"


def papers_line(rows: list[dict], tools: list[str]) -> str:
    lines = ["Each cell: mean over papers (number of papers scored). `n/s`: not "
             "supported by the tool. Failed papers are scored as empty output.\n",
             "| tool | papers | failed | network |", "|---|---|---|---|"]
    for t in tools:
        tool_rows = [r for r in rows if r["tool"] == t]
        lines.append(f"| {TOOL_NAMES[t]} | {len(tool_rows)} | "
                     f"{sum(r['failed'] for r in tool_rows)} | {NETWORK.get(t, 'none')} |")
    return "\n".join(lines) + "\n"


def metric_table(title: str, metrics: list[str], rows: list[dict], tools: list[str]) -> str:
    lines = [f"## {title}\n", header(tools)]
    for metric in metrics:
        cells = [cell([r[metric] for r in rows if r["tool"] == t and r.get(metric) is not None])
                 for t in tools]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def cost_table(run_dir: Path, role: str, tools: list[str]) -> str:
    timings = [r for r in read_jsonl(run_dir / "timings.jsonl")
               if r.get("role") == role and "seconds" in r and r.get("tool") in tools]
    lines = ["## Cost (§6.9; reported, not ranked)\n",
             "| tool | papers | pages | seconds per page | median seconds per paper "
             "| peak VRAM (MiB) | peak system RAM in use (GB) |",
             "|---|---|---|---|---|---|---|"]
    for t in tools:
        tool_rows = [r for r in timings if r["tool"] == t]
        if not tool_rows:
            continue
        pages = sum(r["pages"] for r in tool_rows)
        seconds = sum(r["seconds"] for r in tool_rows)
        lines.append(f"| {TOOL_NAMES[t]} | {len(tool_rows)} | {pages} | "
                     f"{seconds / pages:.2f} | "
                     f"{statistics.median(r['seconds'] for r in tool_rows):.1f} | "
                     f"{max(r['peak_vram_mib'] for r in tool_rows)} | "
                     f"{max(r['peak_ram_used_gb'] for r in tool_rows):.1f} |")
    return "\n".join(lines) + "\n"


def publisher_table(rows: list[dict], tools: list[str]) -> str:
    strata = sorted({r["stratum"] for r in rows})
    lines = ["## By publisher (main metrics)\n"]
    for metric in BY_PUBLISHER:
        lines.append(f"**{metric}**\n")
        lines.append("| publisher | " + " | ".join(TOOL_NAMES[t] for t in tools) + " |")
        lines.append("|---|" + "---|" * len(tools))
        for stratum in strata:
            cells = [cell([r[metric] for r in rows if r["tool"] == t and r["stratum"] == stratum
                           and r.get(metric) is not None]) for t in tools]
            lines.append(f"| {stratum} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def determinism_table(first: Path, second: Path, rows: list[dict], tools: list[str]) -> str:
    """§6.8: share of papers whose common-format output is identical, and
    for the others, which fields changed (counted over this role's papers)."""
    lines = ["## Determinism (§6.8)\n", "| tool | papers in both runs | identical | "
             "fields that changed |", "|---|---|---|---|"]
    for t in tools:
        if not (second / t).exists():
            continue
        same, both, changed = 0, 0, {}
        papers = {r["pmcid"] for r in rows if r["tool"] == t}       # this role only
        for path in sorted(p for p in (second / t).glob("*.json") if p.stem in papers):
            other = first / t / path.name
            if not other.exists():
                continue
            a = json.loads(other.read_text(encoding="utf-8"))
            b = json.loads(path.read_text(encoding="utf-8"))
            both += 1
            diff = [k for k in a if k not in ("tool_version",) and a.get(k) != b.get(k)]
            if not diff:
                same += 1
            for k in diff:
                changed[k] = changed.get(k, 0) + 1
        if both:
            lines.append(f"| {TOOL_NAMES[t]} | {both} | {same} | "
                         + ", ".join(f"{k} ({n})" for k, n in sorted(changed.items())) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
