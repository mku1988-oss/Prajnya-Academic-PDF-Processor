#!/usr/bin/env python
"""
The A11 round's plots, drawn from the saved scores (nothing is re-scored).

For biology researchers: this turns `outputs/runa11/scores/per_paper_scored_key.jsonl`
into PNG images, each in a light and a dark version (GitHub shows the one that
matches the reader's theme; a Reddit post uses the light one):

  compare_figures_tables_formulas   every tool, 6 measures (dot = mean, line = 95% range)
  compare_text_references_links     every tool, 6 measures
  compare_metadata                  the two tools that read metadata, 6 measures
  compare_speed                     seconds per page, every tool and setting
  compare_crops                     one real page: the true figure box and each tool's crop
  prajnya_by_field                  Prajnya alone, biology / physics / chemistry (README)
  prajnya_crop                      the same page, the true box and Prajnya's crop (README)

Rules fixed before looking at any plot:
  * a tool with several settings is shown at its best setting for each measure
    (Prajnya local/online, GROBID with/without lookup, Docling default/formulas on);
  * the 95% range is a bootstrap over papers (2,000 resamples, seed 20261123);
  * the example page is chosen by rule, not by eye: among located figures that
    every cropping tool cropped, the one whose Prajnya IoU is closest to
    Prajnya's median IoU over all figures (a typical figure, not its best).

Run (from src/v6):  python benchmark/harness/plots.py
Writes benchmark/plots/<name>_light.png and <name>_dark.png.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from harness.a11_measures import LOCATE_THRESHOLD, in_points, joint_box   # noqa: E402
from harness.figure_locator import crop_iou                              # noqa: E402

RUN = BENCH / "outputs" / "runa11"
OUT = BENCH / "plots"
SEED, RESAMPLES = 20261123, 2000
FOOTER = ("A11 round: 40 open-access papers not seen during development "
          "(20 biology, 10 physics, 10 chemistry), 2026-11-23. Method and every score: BENCHMARK.md")

# Colours: the dataviz reference palette (validated, light and dark), slot order fixed.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]},
}
#: Tool name -> its settings (score columns). Colour = slot in this order.
TOOLS = {"Prajnya": ["prajnya", "prajnya_online"],
         "GROBID": ["grobid", "grobid_consolidated"],
         "Docling": ["docling", "docling_formula"],
         "Marker": ["marker"], "MinerU": ["mineru"]}
CROPPERS = {"Prajnya": "prajnya", "Docling": "docling", "Marker": "marker", "MinerU": "mineru"}
FIELDS = {"Biology": ["biochemistry", "biophysics", "developmental biology",
                      "computational biology"],
          "Physics": ["physics"], "Chemistry": ["chemistry"]}
PANELS = {
    "compare_figures_tables_formulas": ("Figures, tables and formulas", [
        ("Figures found", "figures_recall"), ("Figure crops right (IoU ≥ 0.8)", "crop_right"),
        ("Tables found", "tables_recall"), ("Table structure and content (TEDS)", "teds"),
        ("Display formulas", "formula_display"), ("Inline formulas", "formula_inline")]),
    "compare_text_references_links": ("Body text, references and links", [
        ("Body text similarity", "body_similarity"), ("Reference titles", "ref_title"),
        ("Citation links, precision", "citation_precision"),
        ("Citation links, recall", "citation_recall"),
        ("Figure mentions found", "figure_mention_recall"),
        ("Table mentions found", "table_mention_recall")]),
    "compare_metadata": ("Bibliographic metadata", [
        ("Title exact", "title_exact"), ("Authors (F1)", "authors_f1"),
        ("Author order", "authors_order"), ("DOI", "doi"), ("Year", "year"),
        ("Journal", "journal")]),
}
README_MEASURES = [("Figures found", "figures_recall"), ("Figure crops right", "crop_right"),
                   ("Table structure (TEDS)", "teds"), ("Display formulas", "formula_display"),
                   ("Figure mentions found", "figure_mention_recall"),
                   ("Title exact", "title_exact"), ("DOI", "doi"), ("Year", "year")]


def main() -> int:
    rows = load_rows()
    OUT.mkdir(exist_ok=True)
    example = choose_example_figure()
    for mode, theme in THEMES.items():
        style(theme)
        for name, (title, measures) in PANELS.items():
            save(compare_panels(rows, title, measures, theme), name, mode)
        save(speed_chart(theme), "compare_speed", mode)
        save(prajnya_by_field(rows, theme), "prajnya_by_field", mode)
        save(crop_pictures(example, list(CROPPERS), theme), "compare_crops", mode)
        save(crop_pictures(example, ["Prajnya"], theme), "prajnya_crop", mode)
    print(f"example figure: {example['pmcid']} {example['label']} "
          f"(Prajnya IoU {example['ious']['Prajnya']:.3f})")
    return 0


# ---------------------------------------------------------------------------
# Helpers (after main, per the project rule)
# ---------------------------------------------------------------------------

def load_rows() -> list[dict]:
    path = RUN / "scores" / "per_paper_scored_key.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def values(rows: list[dict], column: str, measure: str, strata=None) -> list[float]:
    return [r[measure] for r in rows if r["tool"] == column and r.get(measure) is not None
            and (strata is None or r["stratum"] in strata)]


def best_setting(rows: list[dict], tool: str, measure: str, strata=None) -> list[float]:
    """The tool's per-paper values at its best-scoring setting for this measure."""
    candidates = [values(rows, c, measure, strata) for c in TOOLS[tool]]
    candidates = [v for v in candidates if v]
    return max(candidates, key=statistics.fmean) if candidates else []


def bootstrap(vals: list[float]) -> tuple[float, float]:
    """95% range of the mean, resampling papers."""
    rng = random.Random(SEED)
    means = sorted(statistics.fmean(rng.choices(vals, k=len(vals))) for _ in range(RESAMPLES))
    return means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def style(theme: dict) -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"],
        "text.color": theme["ink"], "axes.labelcolor": theme["ink2"],
        "xtick.color": theme["muted"], "ytick.color": theme["ink"],
        "axes.edgecolor": theme["axis"], "savefig.facecolor": theme["surface"]})


def compare_panels(rows: list[dict], title: str, measures: list, theme: dict):
    """Small multiples: one panel per measure, one row per tool, dot + 95% range."""
    fig, axes = plt.subplots(len(measures) // 2, 2, figsize=(11, 2.15 * len(measures) // 2 + 1.3))
    for ax, (label, measure) in zip(axes.flat, measures):
        papers = 0
        for slot, tool in enumerate(TOOLS):
            y = len(TOOLS) - 1 - slot
            vals = best_setting(rows, tool, measure)
            if not vals:
                ax.text(0.01, y, "not produced", va="center", fontsize=9, color=theme["muted"])
                continue
            papers = max(papers, len(vals))
            low, high = bootstrap(vals)
            mean = statistics.fmean(vals)
            colour = theme["series"][slot]
            ax.plot([low, high], [y, y], color=colour, lw=2, solid_capstyle="round", alpha=0.55)
            ax.plot(mean, y, "o", ms=9, color=colour, mec=theme["surface"], mew=2)
            ax.text(1.03, y, f"{mean:.2f}", va="center", fontsize=9.5, color=theme["ink"],
                    fontweight="bold" if tool == "Prajnya" else "normal",
                    transform=ax.get_yaxis_transform())
        panel_axes(ax, label, f"{papers} papers", theme)
    header(fig, title, "Dot = mean score (0 to 1, higher is better); line = 95% range over papers. "
           "Each tool at its best setting.", theme)
    fig.subplots_adjust(left=0.09, right=0.95, top=0.83, bottom=0.09, hspace=0.75, wspace=0.32)
    return fig


def panel_axes(ax, label: str, note: str, theme: dict) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(TOOLS) - 0.4)
    ax.set_yticks(range(len(TOOLS)), list(TOOLS)[::-1])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.grid(axis="x", color=theme["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(label, loc="left", fontsize=11, fontweight="bold", color=theme["ink"], pad=16)
    ax.text(0, 1.04, note, transform=ax.transAxes, fontsize=8.5, color=theme["muted"])


def header(fig, title: str, subtitle: str, theme: dict) -> None:
    height = fig.get_size_inches()[1]          # place the header in inches, any figure height
    fig.text(0.02, 1 - 0.15 / height, title, fontsize=15, fontweight="bold", va="top",
             color=theme["ink"])
    fig.text(0.02, 1 - 0.47 / height, subtitle, fontsize=9.5, va="top", color=theme["ink2"])
    footer = FOOTER if fig.get_size_inches()[0] > 9 else FOOTER.replace(", 2026", ",\n2026")
    fig.text(0.02, 0.012, footer, fontsize=8, color=theme["muted"])


def speed_chart(theme: dict):
    """Seconds per page on the 40 scored papers, one bar per tool setting."""
    settings = [("Prajnya", "prajnya", 0), ("GROBID", "grobid", 1),
                ("GROBID + online lookup", "grobid_consolidated", 1),
                ("Docling", "docling", 2), ("Docling, formulas on", "docling_formula", 2),
                ("Marker", "marker", 3), ("MinerU", "mineru", 4)]
    seconds = seconds_per_page()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (label, column, slot) in enumerate(settings):
        y = len(settings) - 1 - i
        ax.barh(y, seconds[column], height=0.5, color=theme["series"][slot])
        ax.text(seconds[column] + 0.2, y, f"{seconds[column]:.1f} s", va="center",
                fontsize=9.5, color=theme["ink"])
    ax.set_yticks(range(len(settings)), [s[0] for s in settings][::-1])
    ax.grid(axis="x", color=theme["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("seconds per page (lower is faster)")
    header(fig, "Time per page", "Wall time per page on one PC (8 GB GPU), 40 papers, "
           "each tool run once. Online lookups included; Prajnya's adds about 0.6 s per page.", theme)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.17)
    return fig


def seconds_per_page() -> dict:
    with open(BENCH / "corpus" / "MANIFEST_A11.tsv", encoding="utf-8") as f:
        scored = {r["pmcid"] for r in csv.DictReader(f, delimiter="\t") if r["role"] == "scored"}
    seconds, pages = {}, {}
    for line in (RUN / "timings.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("pmcid") in scored and "seconds" in r:
            seconds[r["tool"]] = seconds.get(r["tool"], 0) + r["seconds"]
            pages[r["tool"]] = pages.get(r["tool"], 0) + r["pages"]
    return {t: seconds[t] / pages[t] for t in seconds}


def prajnya_by_field(rows: list[dict], theme: dict):
    """README: Prajnya's scores, one row per measure, one dot per field."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    count = len(README_MEASURES)
    for slot, (field, strata) in enumerate(FIELDS.items()):
        colour = theme["series"][slot]
        for i, (_, measure) in enumerate(README_MEASURES):
            vals = best_setting(rows, "Prajnya", measure, strata)
            if vals:
                y = count - 1 - i + (1 - slot) * 0.22
                ax.plot(statistics.fmean(vals), y, "o", ms=9, color=colour,
                        mec=theme["surface"], mew=2, label=field if i == 0 else None)
    for i, (_, measure) in enumerate(README_MEASURES):
        vals = best_setting(rows, "Prajnya", measure)
        ax.text(1.02, count - 1 - i, f"{statistics.fmean(vals):.2f}", va="center",
                fontsize=10, fontweight="bold", transform=ax.get_yaxis_transform())
    ax.text(1.02, count - 0.35, "all", fontsize=8.5, color=theme["muted"],
            transform=ax.get_yaxis_transform())
    ax.set_yticks(range(count), [m[0] for m in README_MEASURES][::-1])
    ax.set_xlim(0.7, 1.01)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0], ["0.7", "0.8", "0.9", "1"])
    ax.set_ylim(-0.6, count - 0.2)
    ax.grid(axis="x", color=theme["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(-0.01, 0.99),
              labelcolor=theme["ink"], handletextpad=0.2, columnspacing=1.2)
    header(fig, "Prajnya on papers it never saw", "Mean score per field (0 to 1, higher is "
           "better); right: all 40 papers. Metadata with online lookup.", theme)
    fig.subplots_adjust(left=0.22, right=0.93, top=0.78, bottom=0.1)
    return fig


def choose_example_figure() -> dict:
    """The typical figure (rule in the module docstring), with every tool's box."""
    candidates, prajnya_ious = [], []
    for boxes_path in sorted((BENCH / "outputs" / "key_boxes").glob("*.json")):
        pmcid = boxes_path.stem
        common = {t: RUN / "common" / c / f"{pmcid}.json" for t, c in CROPPERS.items()}
        if not all(p.exists() for p in common.values()):
            continue
        boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
        key = json.loads((BENCH / "outputs" / "key" / f"{pmcid}.json").read_text(encoding="utf-8"))
        number_of = {f["id"]: f["label_number"] for f in key["figures"]}
        tools = {t: json.loads(p.read_text(encoding="utf-8")) for t, p in common.items()}
        for figure_id, found in boxes["figures"].items():
            if found["score"] < LOCATE_THRESHOLD:
                continue
            crops = {t: crop_on_page(tools[t], number_of[figure_id], found["page_index"],
                                     boxes["page_sizes"]) for t in CROPPERS}
            if crops["Prajnya"] is not None:
                prajnya_ious.append(crop_iou(tuple(found["box"]), crops["Prajnya"]))
            if all(c is not None for c in crops.values()):
                candidates.append({"pmcid": pmcid, "label": found["label"], "true": found["box"],
                                   "page_index": found["page_index"], "crops": crops,
                                   "ious": {t: crop_iou(tuple(found["box"]), c)
                                            for t, c in crops.items()}})
    median = statistics.median(prajnya_ious)
    return min(candidates, key=lambda c: (abs(c["ious"]["Prajnya"] - median), c["pmcid"]))


def crop_on_page(tool: dict, number: str, page_index: int, page_sizes: list):
    for figure in tool.get("figures") or []:
        if figure.get("label_number") == number:
            boxes = [in_points(c, page_sizes) for c in figure.get("crops") or []]
            on_page = [b for p, b in boxes if p == page_index]
            return joint_box(on_page) if on_page else None
    return None


def crop_pictures(example: dict, tools: list, theme: dict):
    """The example page once per tool: the true box and that tool's crop."""
    page, width_pt = render_page(example["pmcid"], example["page_index"])
    scale = page.width / width_pt
    size = (3.3 * len(tools) + 0.6, 5.4) if len(tools) > 1 else (7.6, 8.2)
    fig, axes = plt.subplots(1, len(tools), figsize=size, squeeze=False)
    for ax, tool in zip(axes.flat, tools):
        ax.imshow(page)
        draw_box(ax, example["true"], scale, theme["ink"], fill=False)
        slot = list(TOOLS).index(tool)
        draw_box(ax, example["crops"][tool], scale, theme["series"][slot], fill=True)
        ax.set_axis_off()
        ax.set_title(f"{tool}   IoU {example['ious'][tool]:.2f}", fontsize=11,
                     fontweight="bold", color=theme["ink"], loc="left")
    title = "Figure crops on a real page" if len(tools) > 1 else "Prajnya's figure crop"
    joiner = " " if len(tools) > 1 else "\n"
    header(fig, title, f"Black or white outline = the figure's true box; coloured box = the "
           f"tool's crop.{joiner}{example['label']}, PMC open-access paper {example['pmcid']} "
           f"(CC BY).", theme)
    if len(tools) > 1:
        fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.05, wspace=0.06)
    else:
        fig.subplots_adjust(left=0.08, right=0.92, top=0.85, bottom=0.06)
    return fig


def render_page(pmcid: str, page_index: int):
    import pypdfium2 as pdfium
    with open(BENCH / "corpus" / "MANIFEST_A11.tsv", encoding="utf-8") as f:
        version = next(r["version"] for r in csv.DictReader(f, delimiter="\t")
                       if r["pmcid"] == pmcid)
    document = pdfium.PdfDocument(str(BENCH / "corpus" / "cache" / f"{version}.pdf"))
    try:
        page = document[page_index]
        width_pt = page.get_size()[0]
        image = page.render(scale=150 / 72).to_pil()
        page.close()
        return image, width_pt
    finally:
        document.close()


def draw_box(ax, box, scale: float, colour: str, fill: bool) -> None:
    x0, y0, x1, y1 = (v * scale for v in box)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2.2 if fill else 2.6,
                           edgecolor=colour, linestyle="-"))
    if fill:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, lw=0,
                               facecolor=colour, alpha=0.12))


def save(fig, name: str, mode: str) -> None:
    fig.savefig(OUT / f"{name}_{mode}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
