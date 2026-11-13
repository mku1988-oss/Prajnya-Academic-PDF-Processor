#!/usr/bin/env python3
"""
The document JSON — the canonical interchange shape, and how it is written.

For biology researchers: this is the file a run produces for each paper. It is
also the file the parity harness reads to check that a speed change did not
quietly damage quality, which is why its *shape* is fixed and must not drift.

**Why this is its own module.** Writing a finished document is work the CLI and
the GUI both do, identically. It used to live in `run_prajnya.py`, which meant
anything under `src/` that wanted to write a document would have had to import
the entrypoint. Moved here, both front ends call the same three functions and
the envelope stays single-sourced.

Nothing about the payload changed in the move — the harness compares against an
archived V5 known-good run, and a byte that shifts here is a false alarm there.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .models import DocumentProfile

LOG = logging.getLogger(__name__)

#: Stamped into every document's `document_info`, so an output file says which
#: pipeline produced it.
PIPELINE_VERSION = "v6.5.0"

#: The version of the output FORMAT, which moves independently of the release.
#: Stamped into `document_info.contract_version`, and pinned against
#: `contract/prajnya_document.schema.json` (`x-contract-version`) by
#: `tests/test_output_contract.py`: change one without the other and the
#: suite fails. Raise the MAJOR number when a consumer's code could break
#: (a key removed, renamed, retyped, or given a new meaning); raise the MINOR
#: number for anything purely additive. Rules in `contract/README.md`.
CONTRACT_VERSION = "1.0"


def write_result(profile: DocumentProfile, out_path: Path, elapsed: float,
                 timings: Optional[Dict[str, float]] = None) -> None:
    """
    Write the run as a canonical document JSON (see `build_document_json`).

    An unserializable value must never destroy a finished document. Everything
    above this line has already been paid for in GPU minutes; if some corner of
    the payload holds a type neither pydantic nor `json` knows — a numpy scalar
    escaping from OpenCV was the real case — the value is recovered or written
    as text, and the loss is one field, loudly logged, instead of the whole
    document. The dump inside `build_document_json` carries the same fallback.

    Args:
        profile: the processed document.
        out_path: where to write. Its parent directory is created.
        elapsed: total wall time in seconds.
        timings: optional per-bucket wall time.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_document_json(profile, elapsed, timings)

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  default=last_resort_encoder)


def build_document_json(profile: DocumentProfile, elapsed: float,
                        timings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Serialize a processed document into the canonical interchange shape.

    For biology researchers: this is the file the parity harness reads to check
    that a speed change did not quietly damage quality. It has to be the SAME
    shape as the archived V5 known-good run, or the comparison is meaningless.

    V6's internal `DocumentProfile` nests everything under `processing_result`,
    which is convenient in code but is not what the harness (or V5) speaks. The
    per-page and per-figure records already match exactly; only this outer
    envelope differs, so this function is a re-wrap, not a conversion:

        document_info   run-level facts — filename, wall time, success, timings
        metadata_final  the merged bibliographic record
        pages           top-level, one entry per processed page
        summary         errors and counts

    Args:
        profile: the processed document
        elapsed: total wall time in seconds
        timings: optional per-bucket wall time (title / reads / detection /
            repair / swap). Recorded under document_info['timings'], which the
            harness prefers over anything it can reconstruct.

    Returns:
        A JSON-serializable dict in the canonical shape.
    """
    result = profile.processing_result
    dumped = profile.model_dump(mode="json", fallback=last_resort_encoder)
    dumped_result = dumped.get("processing_result") or {}

    errors = list(result.errors) if result and result.errors else []
    pages = dumped_result.get("pages") or []

    info: Dict[str, Any] = {
        "filename": profile.pdf_path.name,
        "document_type": enum_value(profile.document_type),
        "processing_time": elapsed,
        "success": not errors,
        "pipeline_version": PIPELINE_VERSION,
        "contract_version": CONTRACT_VERSION,
    }
    if timings:
        info["timings"] = dict(timings)

    return {
        "document_info": info,
        "metadata_final": dumped_result.get("metadata") or {},
        "pages": pages,
        "summary": {
            "errors": errors,
            "page_count": len(pages),
            "figure_count": sum(
                (page.get("metadata") or {}).get("figure_count") or 0 for page in pages
            ),
        },
        # The un-flattened profile, so nothing is lost by the re-wrap.
        "profile": dumped,
    }


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def last_resort_encoder(value: Any) -> Any:
    """
    Serialize what `json` cannot, rather than raising at the finish line.

    Numpy scalars have a real value worth keeping (`.item()` gives the Python
    one). Anything else becomes its `repr`, which is lossy but preserves the
    document — and the warning names the type so the leak can be fixed at its
    source, which is where it should be fixed.
    """
    item = getattr(value, "item", None)
    if callable(item):
        try:
            recovered = item()
            LOG.warning("Coerced a %s in the output JSON — fix this at its source.",
                        type(value).__name__)
            return recovered
        except Exception:                                     # pragma: no cover
            pass

    LOG.warning("Wrote a %s as text in the output JSON — fix this at its source.",
                type(value).__name__)
    return repr(value)


def enum_value(value: Any) -> Any:
    """DocumentType may arrive as an enum or an already-dumped string."""
    return getattr(value, "value", value)
