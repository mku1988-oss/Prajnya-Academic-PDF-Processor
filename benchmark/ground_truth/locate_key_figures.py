#!/usr/bin/env python
"""
Find each key figure's true box on its page (amendment A11 §6), once per paper.

For biology researchers: for every paper of a manifest, this fetches each
figure's publisher image from the PMC bucket, renders the PDF's pages, and
slides the image over them (harness/figure_locator.py). It writes, per paper,
`outputs/key_boxes/<PMCID>.json`:

    {"page_sizes": [[width_pt, height_pt], ...],
     "figures": {"<key figure id>": {"page_index": 2, "box": [x0, y0, x1, y1],
                                     "score": 0.91, "image": "<file name>"}}}

No tool output is read. A figure without a publisher image gets no entry
(it counts as "not located"). Optionally `--draw` saves each page with the
found box drawn on it, for checking by eye (the pilot check).

Run (from src/v6):
    python benchmark/ground_truth/locate_key_figures.py --manifest MANIFEST_A11.tsv --role pilot --draw
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "corpus_tools"))

from ground_truth.jats_to_common import jats_to_common        # noqa: E402
from harness.figure_locator import locate_figure, render_pages  # noqa: E402
import build_manifest as base                                  # noqa: E402

CORPUS = BENCH / "corpus"
BOXES_DIR = BENCH / "outputs" / "key_boxes"
FIGURE_CACHE = CORPUS / "cache" / "figures"


def main() -> int:
    args = parse_arguments()
    BOXES_DIR.mkdir(parents=True, exist_ok=True)
    for paper in load_papers(args.manifest, args.role):
        target = BOXES_DIR / f"{paper['pmcid']}.json"
        if target.exists() and not args.force:
            continue
        result = locate_paper(paper, args.draw)
        target.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="")
        scores = [f["score"] for f in result["figures"].values()]
        print(f"{paper['pmcid']:12s} figures {len(scores):2d}  lowest match "
              f"{min(scores) if scores else float('nan'):.3f}", flush=True)
    return 0


def locate_paper(paper: dict, draw: bool) -> dict:
    """Every key figure of one paper located on the rendered PDF."""
    version = paper["version"]
    key = jats_to_common((CORPUS / "cache" / f"{version}.xml").read_bytes(), paper["pmcid"])
    pages = render_pages(CORPUS / "cache" / f"{version}.pdf")
    found = {}
    for figure in key["figures"]:
        image = publisher_image(version, figure.get("image"))
        if image is None:
            continue
        where = locate_figure(image, pages)
        if where is None:
            continue
        found[figure["id"]] = {"page_index": where.page_index, "box": list(where.box),
                               "score": round(where.score, 4), "image": image.name,
                               "label": figure["label"]}
    if draw:
        draw_boxes(paper["pmcid"], pages, found)
    return {"page_sizes": [[w, h] for _, w, h in pages], "figures": found}


# ---------------------------------------------------------------------------
# Helpers (after the main functions, per the project rule)
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="MANIFEST_A11.tsv")
    parser.add_argument("--role", default=None, help="pilot or scored (default: both)")
    parser.add_argument("--draw", action="store_true", help="save pages with boxes drawn")
    parser.add_argument("--force", action="store_true", help="locate again")
    return parser.parse_args()


def load_papers(manifest: str, role: str | None) -> list[dict]:
    with open(CORPUS / manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [r for r in rows if role is None or r["role"] == role]


def publisher_image(version: str, name: str | None) -> Path | None:
    """The figure's image from the bucket (cached). JATS often names it
    without its extension; the bucket record lists the real file."""
    if not name:
        return None
    record = json.loads((CORPUS / "cache" / f"{version}.json").read_text(encoding="utf-8"))
    for url in record.get("media_urls") or []:
        file_name = Path(url.split("?")[0]).name
        if without_image_extension(file_name) == without_image_extension(name):
            local = FIGURE_CACHE / version / file_name
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                web = url.split("?")[0].replace("s3://pmc-oa-opendata", base.BUCKET)
                local.write_bytes(base.http_get(web))
            return local
    return None


def without_image_extension(name: str) -> str:
    """A file name without an image extension. Only real image extensions
    are cut: PLOS names a figure "pcbi.1006201.g001", whose ".g001" is part
    of the name (Path.stem would cut it and match the paper's first file)."""
    return re.sub(r"\.(?:jpe?g|png|gif|tiff?|webp)$", "", name, flags=re.IGNORECASE)


def draw_boxes(pmcid: str, pages: list, found: dict) -> None:
    """Pages holding a located figure, with its box and match score drawn."""
    from PIL import ImageDraw
    folder = BENCH / "outputs" / "key_boxes" / "drawn" / pmcid
    folder.mkdir(parents=True, exist_ok=True)
    for index, (picture, width_pt, _) in enumerate(pages):
        here = {k: v for k, v in found.items() if v["page_index"] == index}
        if not here:
            continue
        canvas = picture.convert("RGB")
        pen, scale = ImageDraw.Draw(canvas), picture.width / width_pt
        for entry in here.values():
            box = [v * scale for v in entry["box"]]
            pen.rectangle(box, outline=(220, 0, 0), width=4)
            pen.text((box[0] + 6, box[1] + 6), f"{entry['label']} {entry['score']:.2f}", fill=(220, 0, 0))
        canvas.save(folder / f"page{index + 1:02d}.png")


if __name__ == "__main__":
    sys.exit(main())
