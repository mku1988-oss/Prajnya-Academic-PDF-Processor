#!/usr/bin/env python
"""
Run MinerU's parsing engine on one PDF, locally (benchmark, protocol §4).

For biology researchers: MinerU 4 is normally driven through its "document
library" server, which also carries telemetry and an optional upload to
mineru.net. This runner skips that layer and calls the engine itself,
`doc_analyze`, with the effort that MinerU's own default local tier uses
(tier "standard" = effort "high", from mineru/parser/tier.py). Nothing leaves
the machine; models are downloaded once on first use.

Run inside MinerU's own environment (never Prajnya's):
    <bench_mineru env>/python.exe benchmark/runners/run_mineru.py <pdf> <out_dir> [effort]

Writes <out_dir>/<stem>.middle.json, <stem>.content_list.json, <stem>.md and
<stem>.timing.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MINERU_TELEMETRY_DISABLED", "1")   # belt and braces

DEFAULT_EFFORT = "high"     # = MinerU's default local tier "standard"


def main() -> int:
    """Parse one PDF and write MinerU's three outputs plus wall time."""
    pdf_path, out_dir, effort = read_arguments(sys.argv[1:])
    out_dir.mkdir(parents=True, exist_ok=True)
    # Step 1: the engine (layout pipeline + VLM, at the chosen effort).
    started = time.perf_counter()
    middle_json = analyse_pdf(pdf_path, effort)
    seconds = time.perf_counter() - started
    # Step 2: MinerU's own renderers, so we score what MinerU users get.
    write_outputs(out_dir, pdf_path.stem, middle_json, effort, seconds)
    print(f"mineru {pdf_path.name}: effort={effort} {seconds:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# Helpers (after main, per the project rule)
# ---------------------------------------------------------------------------

def read_arguments(argv: list[str]) -> tuple[Path, Path, str]:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    effort = argv[2] if len(argv) > 2 else DEFAULT_EFFORT
    return Path(argv[0]), Path(argv[1]), effort


def analyse_pdf(pdf_path: Path, effort: str) -> dict:
    from mineru.backend.analyze import doc_analyze
    middle_json, _model_json = doc_analyze(pdf_path.read_bytes(), effort=effort,
                                           file_suffix="pdf")
    return middle_json      # MinerU's own object: its renderers require it


def write_outputs(out_dir: Path, stem: str, middle_json, effort: str,
                  seconds: float) -> None:
    from mineru.render.content_list import render_content_list
    from mineru.render.markdown import render_markdown
    write_json(out_dir / f"{stem}.middle.json", to_plain(middle_json))
    write_json(out_dir / f"{stem}.content_list.json",
               to_plain(render_content_list(middle_json)))
    markdown = render_markdown(middle_json)
    if not isinstance(markdown, str):
        markdown = getattr(markdown, "content", None) or str(markdown)
    (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8", newline="")
    write_json(out_dir / f"{stem}.timing.json",
               {"effort": effort, "seconds": round(seconds, 2)})


def to_plain(value):
    """MiddleJson may be a pydantic model or a TypedDict; make it plain JSON."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                    encoding="utf-8", newline="")


if __name__ == "__main__":
    sys.exit(main())
