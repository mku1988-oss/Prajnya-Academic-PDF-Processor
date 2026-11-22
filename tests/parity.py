#!/usr/bin/env python3
"""
Parity comparison helpers for the V6 benchmark harness.

For biology researchers: "parity" here just means *did we keep getting the same
answer*. Every speed optimization in V6 risks quietly losing content — a dropped
table, a mangled author name, a missing figure. This module holds the checks that
catch that, so the speed work can proceed without guessing.

These are plain functions with no pytest dependency, which means you can also run
them from the command line against a fresh pipeline output:

    python -m tests.parity path/to/fresh_run.json

The pytest wrapper that uses them lives in `test_parity_nmr.py`.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.processing.utils.text_normalization import normalize_harvested_text


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Check:
    """One pass/fail assertion, with enough context to debug a failure."""

    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        line = f"[{mark}] {self.name}"
        if not self.passed:
            line += f"\n        expected: {self.expected!r}"
            line += f"\n        actual:   {self.actual!r}"
            if self.detail:
                line += f"\n        note:     {self.detail}"
        return line


@dataclass
class ParityReport:
    """The full set of checks for one document run."""

    checks: List[Check] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def passed(self) -> bool:
        return not self.failures

    def render(self) -> str:
        """Human-readable summary — failures first, then the timing table."""
        lines: List[str] = []
        total, failed = len(self.checks), len(self.failures)
        lines.append(f"Parity: {total - failed}/{total} checks passed")
        lines.append("")
        for check in self.checks:
            lines.append(str(check))
        if self.timings:
            lines.append("")
            lines.append(render_timing_table(self.timings))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading and normalization
# ---------------------------------------------------------------------------

def load_document(path: Path | str) -> Dict[str, Any]:
    """Load a pipeline document JSON (either a baseline or a fresh run)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


#: Typographic characters that PDF text layers use freely and that carry no
#: meaning for a content comparison. Folded to ASCII before comparing.
#:
#: Note NFKC does NOT do this — it leaves U+2018/U+2019 alone. Journal PDFs
#: often build a double quote from two single quotes (the LaTeX ``...'' idiom),
#: which is what the p4 figure caption in our baseline does.
_PUNCTUATION_FOLD = str.maketrans({
    "‘": "'", "’": "'",           # single curly quotes
    "“": '"', "”": '"',           # double curly quotes
    "‐": "-", "‑": "-", "‒": "-",
    "–": "-", "—": "-", "―": "-",   # dashes
    " ": " ", " ": " ", " ": " ",   # exotic spaces
})


def normalize_text(value: Optional[str]) -> str:
    """
    Collapse a string to a form that survives cosmetic differences.

    Unicode is NFKC-folded, typographic punctuation is folded to ASCII, quote
    marks are dropped entirely, control-character damage is repaired, whitespace
    runs collapse to single spaces, and case is dropped. This lets a caption
    comparison ignore how a PDF happened to render its quotes without ignoring
    real content changes.

    **On the control-character fold.** Some PDFs encode a line-break hyphen as a
    private font code, so a word arrives as "measure<02>ments". The archived V5
    baseline contains ten of these; V6 repairs them at extraction (session 22).
    Folding both sides here keeps the comparison honest in the only way that
    preserves the gate's purpose: the baseline is V5's output and must stay
    byte-fixed, per tests/baselines/README.md, so it cannot simply be
    regenerated. Note `\\s` does NOT match these characters, so the whitespace
    collapse below never removed them.
    """
    if not value:
        return ""
    folded = unicodedata.normalize("NFKC", value).translate(_PUNCTUATION_FOLD)
    # Quote marks are pure presentation here and vary between the text layer and
    # a VLM transcription of the same caption.
    folded = folded.replace("'", "").replace('"', "")
    folded = normalize_harvested_text(folded)
    return re.sub(r"\s+", " ", folded).strip().lower()


def normalize_author(name: str) -> str:
    """
    Normalize an author name for order-sensitive comparison.

    Author *order* is data in biology — it encodes contribution — so the harness
    compares the sequence, not a set. Punctuation and initials spacing vary
    between the publisher record and the PDF text layer, so those are folded out.
    """
    folded = normalize_text(name)
    return re.sub(r"[.,]", "", folded)


# ---------------------------------------------------------------------------
# Accessors — tolerate schema drift between V5 and V6 output
# ---------------------------------------------------------------------------

def get_pages(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    return doc.get("pages") or []


def page_by_number(doc: Dict[str, Any], number: int) -> Optional[Dict[str, Any]]:
    for page in get_pages(doc):
        if page.get("page_number") == number:
            return page
    return None


def page_meta(page: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not page:
        return {}
    return page.get("metadata") or {}


def all_figures(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every figure record across all pages, in page order."""
    figures: List[Dict[str, Any]] = []
    for page in get_pages(doc):
        figures.extend(page_meta(page).get("figures") or [])
    return figures


# ---------------------------------------------------------------------------
# Timing buckets
# ---------------------------------------------------------------------------

#: The buckets every V6 run should report, so a stage's claimed saving is
#: measured rather than estimated. Ported-as-is code will not populate all of
#: them; absent buckets are reported as missing rather than failing.
TIMING_BUCKETS = ("title", "reads", "detection", "repair", "swap", "total")


def extract_timings(doc: Dict[str, Any]) -> Dict[str, float]:
    """
    Pull per-bucket wall time out of a run.

    V5 only recorded title-page timings (shot1/shot2) plus the document total, so
    that is all this can recover from the baseline. V6 instrumentation should
    populate `document_info['timings']` with the full bucket set; when it does,
    that takes precedence.
    """
    info = doc.get("document_info") or {}

    explicit = info.get("timings")
    if isinstance(explicit, dict) and explicit:
        return {k: float(v) for k, v in explicit.items() if isinstance(v, (int, float))}

    # Fall back to reconstructing what V5-era output does expose.
    timings: Dict[str, float] = {}
    if isinstance(info.get("processing_time"), (int, float)):
        timings["total"] = float(info["processing_time"])

    title_meta = page_meta(page_by_number(doc, 1))
    if isinstance(title_meta.get("total_time"), (int, float)):
        timings["title"] = float(title_meta["total_time"])
    for shot in ("shot1_time", "shot2_time"):
        if isinstance(title_meta.get(shot), (int, float)):
            timings[f"title_{shot.replace('_time', '')}"] = float(title_meta[shot])

    return timings


def render_timing_table(timings: Dict[str, float]) -> str:
    """Format timings as a table, with each bucket's share of the total."""
    if not timings:
        return "Timings: none recorded"

    total = timings.get("total")
    lines = ["Timing buckets:"]
    width = max(len(k) for k in timings)
    for name, seconds in sorted(timings.items(), key=lambda kv: -kv[1]):
        share = ""
        if total and total > 0 and name != "total":
            share = f"  ({seconds / total * 100:4.1f}%)"
        lines.append(f"  {name:<{width}}  {seconds:8.2f}s{share}")

    missing = [b for b in TIMING_BUCKETS if b not in timings]
    if missing:
        lines.append(f"  -- not instrumented: {', '.join(missing)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: List[str]) -> int:
    """Compare a fresh run against the checked-in expectations."""
    if len(argv) != 2:
        print(__doc__)
        print("usage: python -m tests.parity <run_output.json>", file=sys.stderr)
        return 2

    # Imported lazily so this module stays free of test-suite dependencies.
    from tests.test_parity_nmr import build_report  # noqa: PLC0415

    report = build_report(load_document(argv[1]))
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
