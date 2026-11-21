#!/usr/bin/env python
"""
Convert every tool's raw output to the common format and score it (§6).

For biology researchers: run this after (or during) the batch. It finds each
tool's output for each paper, converts it with that tool's adapter, scores it
against the JATS answer key, and writes:

  benchmark/outputs/run<N>/common/<tool>/<PMCID>.json   converted outputs
  benchmark/outputs/run<N>/scores/per_paper.jsonl        one line per tool x paper
  benchmark/outputs/run<N>/scores/summary_<role>.json    means and spread per tool

A paper a tool failed on (no output) is scored as an empty output for the
fields that tool supports, so failures count against the tool.

Run (from src/v6):
    python benchmark/harness/run_scoring.py --run 1 --role pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from adapters import docling, grobid, marker, mineru, prajnya   # noqa: E402
from harness import metrics                                       # noqa: E402

TOOLS = ["pdf_text_layer", "prajnya", "prajnya_online", "grobid",
         "grobid_consolidated", "docling", "marker", "mineru"]


def main() -> int:
    """Convert, score and summarise one run for one role (pilot/scored)."""
    args = parse_arguments()
    run_dir = BENCH / "outputs" / f"run{args.run}"
    papers = load_papers(args.role)
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tool in [t for t in TOOLS if t in args.tools]:
        if not (run_dir / tool).exists() and tool != "pdf_text_layer":
            continue
        for paper in papers:
            key = load_key(paper["pmcid"], args.key)
            common = convert(tool, paper, run_dir)
            if common is None:
                common = failed_output(tool, paper["pmcid"])
            write_common(run_dir, tool, paper["pmcid"], common)
            row = {"tool": tool, "pmcid": paper["pmcid"], "role": paper["role"],
                   "stratum": paper["stratum"], "failed": common.get("failed", False),
                   **score(key, common)}
            rows.append(row)
            print(f"{tool:20s} {paper['pmcid']:12s} done", flush=True)
    name = scores_name(args)
    write_rows(scores_dir / f"per_paper_{name}.jsonl", rows)
    summary = summarise(rows)
    (scores_dir / f"summary_{name}.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8", newline="")
    print_summary(summary)
    return 0


# ---------------------------------------------------------------------------
# Helpers (after main, per the project rule)
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="1")
    parser.add_argument("--role", default="pilot")
    parser.add_argument("--tools", default=",".join(TOOLS),
                        help="comma-separated subset (default: every tool found)")
    parser.add_argument("--key", default="key",
                        help="answer-key folder under outputs/ (A7: 'key_frozen' is "
                             "the key as frozen before amendment A7)")
    args = parser.parse_args()
    args.partial = args.tools != ",".join(TOOLS)
    args.tools = args.tools.split(",")
    return args


def scores_name(args: argparse.Namespace) -> str:
    """The scores files' name: `<role>_<key folder>`, plus `_partial` for a
    subset of tools. Run 1's recorded `summary_<role>.json` (scored before
    amendment A7, on what is now `key_frozen`) is never written again."""
    name = f"{args.role}_{args.key}"
    if args.partial:
        name += "_partial"
    return name


def load_papers(role: str) -> list[dict]:
    with open(BENCH / "corpus" / "MANIFEST.tsv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["role"] == role]


def load_key(pmcid: str, folder: str = "key") -> dict:
    return json.loads((BENCH / "outputs" / folder / f"{pmcid}.json").read_text(encoding="utf-8"))


def convert(tool: str, paper: dict, run_dir: Path) -> dict | None:
    """The tool's raw output for this paper, converted; None if missing."""
    pmcid, version = paper["pmcid"], paper["version"]
    if tool == "pdf_text_layer":
        return text_layer_baseline(pmcid, version)
    folder = run_dir / tool
    paths = {
        "prajnya": folder / "rundir" / "documents" / f"{pmcid}.v6.json",
        "prajnya_online": folder / "rundir" / "documents" / f"{pmcid}.v6.json",
        "grobid": folder / f"{pmcid}.tei.xml",
        "grobid_consolidated": folder / f"{pmcid}.tei.xml",
        "docling": folder / f"{version}.json",
        "marker": folder / version / f"{version}.json",
        "mineru": folder / f"{version}.content_list.json",
    }
    path = paths[tool]
    if not path.exists():
        return None
    if tool == "prajnya":
        return prajnya.convert(path, pmcid)
    if tool == "prajnya_online":
        return prajnya.convert(path, pmcid, track="online")
    if tool.startswith("grobid"):
        return grobid.convert(path, pmcid, tool=tool,
                              track="online" if "consolidated" in tool else "offline")
    return {"docling": docling, "marker": marker, "mineru": mineru}[tool].convert(path, pmcid)


def text_layer_baseline(pmcid: str, version: str) -> dict:
    """Amendment A4.9: the PDF's raw text layer, page by page, as a reference
    row for body text only (no tool involved)."""
    import pypdfium2 as pdfium
    from adapters.common import empty_document
    document = empty_document(pmcid, "pdf_text_layer", "pypdfium2", "baseline")
    pdf = pdfium.PdfDocument(str(BENCH / "corpus" / "cache" / f"{version}.pdf"))
    document["body_paragraphs"] = [pdf[i].get_textpage().get_text_range()
                                   for i in range(len(pdf))]
    pdf.close()
    return document


def failed_output(tool: str, pmcid: str) -> dict:
    """What a failed run 'produced': nothing, in every field the tool supports
    (taken from a successful conversion's shape for that tool)."""
    from adapters.common import empty_document
    document = empty_document(pmcid, tool, "", "")
    supported = SUPPORTED_FIELDS[tool]
    for name in supported:
        document[name] = [] if name != "abstract" else ""
    if "metadata" in supported:
        document["metadata"] = {"title": "", "authors": [], "doi": None, "year": None,
                                "journal": None}
    elif "title" in supported:
        document["metadata"]["title"] = ""
    document["failed"] = True
    return document


SUPPORTED_FIELDS = {
    # figure_links and table_links for Prajnya, table_links for GROBID: A7.
    "prajnya": ["metadata", "body_paragraphs", "figures", "tables",
                "figure_links", "table_links"],
    "prajnya_online": ["metadata", "body_paragraphs", "figures", "tables",
                       "references", "citation_links", "figure_links", "table_links"],
    "grobid": ["metadata", "abstract", "body_paragraphs", "figures", "tables",
               "references", "citation_links", "figure_links", "table_links"],
    "grobid_consolidated": ["metadata", "abstract", "body_paragraphs", "figures",
                            "tables", "references", "citation_links", "figure_links",
                            "table_links"],
    "docling": ["title", "body_paragraphs", "figures", "tables"],
    "marker": ["body_paragraphs", "figures", "tables"],
    "mineru": ["body_paragraphs", "figures", "tables", "references"],
}


def score(key: dict, tool: dict) -> dict:
    """Every §6 metric for one tool on one paper."""
    result = {}
    result.update(metrics.score_metadata(key, tool))
    result.update(metrics.score_abstract(key, tool))
    body, spans, stream = metrics.score_body(key, tool)
    result.update(body)
    result.update(metrics.score_figures(key, tool))
    result.update(metrics.score_tables(key, tool))
    references, pairs = metrics.score_references(key, tool)
    result.update(references)
    result.update(metrics.score_links(key, tool, spans, stream, "citations", pairs))
    result.update(metrics.score_links(key, tool, spans, stream, "figures"))
    result.update(metrics.score_links(key, tool, spans, stream, "tables"))
    return result


def write_common(run_dir: Path, tool: str, pmcid: str, document: dict) -> None:
    folder = run_dir / "common" / tool
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{pmcid}.json").write_text(json.dumps(document, ensure_ascii=False),
                                          encoding="utf-8", newline="")


def write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def summarise(rows: list[dict]) -> dict:
    """Per tool and metric: papers scored, mean, median, min, max."""
    summary: dict[str, dict] = {}
    metric_names = [k for k in rows[0] if k not in
                    ("tool", "pmcid", "role", "stratum", "failed")] if rows else []
    for tool in dict.fromkeys(r["tool"] for r in rows):
        tool_rows = [r for r in rows if r["tool"] == tool]
        entry = {"papers": len(tool_rows), "failed": sum(r["failed"] for r in tool_rows)}
        for name in metric_names:
            values = [r[name] for r in tool_rows if r.get(name) is not None]
            entry[name] = None if not values else {
                "n": len(values), "mean": round(statistics.fmean(values), 4),
                "median": round(statistics.median(values), 4),
                "min": round(min(values), 4), "max": round(max(values), 4)}
        summary[tool] = entry
    return summary


def print_summary(summary: dict) -> None:
    tools = list(summary)
    names = [n for n in next(iter(summary.values())) if n not in ("papers", "failed")] \
        if summary else []
    print(f"{'metric':28s}" + "".join(f"{t[:14]:>15s}" for t in tools))
    for name in names:
        cells = []
        for tool in tools:
            value = summary[tool].get(name)
            cells.append(f"{'n/s':>15s}" if value is None else f"{value['mean']:>15.3f}")
        print(f"{name:28s}" + "".join(cells))


if __name__ == "__main__":
    sys.exit(main())
