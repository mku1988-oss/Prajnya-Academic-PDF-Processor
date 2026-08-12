#!/usr/bin/env python3
"""
Where things live — the one place that answers "which directory?".

For biology researchers: this module holds the fixed locations a run uses —
where the configuration is read from, and where results are written to. It is
deliberately tiny and deliberately dependency-free, so that everything else
can ask it a question without dragging in the rest of the pipeline.

**Why this is its own module, and not part of the entrypoint.** These values
used to live in `run_prajnya.py`. Code under `src/` must never import the
entrypoint — the entrypoint is a front end, and a second front end (the GUI)
will exist alongside it. Anything both front ends need therefore has to sit
below them, which is here.

**Anchored to the repository, never to the working directory.** Every path is
derived from this file's own location, so the answer does not change depending
on which folder the command was typed in. That was a real defect: a folder run
once created its `run_outputs/` *inside whatever folder was scanned*, so the
same library processed twice from two different places left two unrelated
result trees with no way to tell which was current.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

#: The V6 tree itself — `src/v6`. This file is `src/v6/src/processing/paths.py`,
#: so two levels up from `src/processing/` is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The `config/` directory: strategy YAML, VLM profiles, task definitions.
CONFIG_ROOT = REPO_ROOT / "config"

#: Where results go when `--out` is not given.
#:
#: Anchored to the repository, NOT to the input, deliberately: writing beside
#: the PDF scatters generated JSON through the user's reference library, which
#: should contain only papers. One predictable location makes a run findable
#: afterwards.
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "run_outputs"

#: Persistent state that belongs to the library rather than to any one run:
#: the ledger and the Zotero cache. Anchored for the same reason as everything
#: else here, but the consequence of getting it wrong is worse.
#:
#: These were CWD-relative. A run launched from another directory therefore
#: created a SECOND ledger there — so "have I already processed this?" silently
#: answered no, every document was re-processed, and the review queue split in
#: two with neither half knowing about the other. Nothing failed; the run just
#: quietly stopped remembering. `SARA_LEDGER_PATH` and `SARA_CACHE_PATH` still
#: override, which is the supported way to keep a library elsewhere.
PERSISTENCE_ROOT = REPO_ROOT / "persistence"
DEFAULT_LEDGER_PATH = str(PERSISTENCE_ROOT / "semantic_ledger.csv")
DEFAULT_ZOTERO_CACHE_PATH = str(PERSISTENCE_ROOT / "zotero_cache")


def anchor_output_path(value: Optional[Any], output_root: Optional[Path],
                       default_name: Optional[str] = None) -> Path:
    """
    Pin one output location to the run's root, wherever the run was launched.

    For biology researchers: this is what stops a run scattering its own
    results. Figures and tables used to be written to a path *relative to
    whatever folder you happened to be standing in* when you started the run —
    so `--out D:/foo` moved the JSON documents to `D:/foo` and quietly left the
    figures behind, in a `run_outputs/figures` next to your shell. One run, two
    unrelated places, and nothing said so.

    The rule now, in order:

    - **nothing configured** -> `output_root/default_name` (or `output_root`
      itself when no name is given). This is the case that always applies
      today, because these keys are absent from the strategy YAML.
    - **a relative path** -> resolved against `output_root`, never against the
      working directory. This is the fix.
    - **an absolute path** -> taken exactly as given. Someone who names a
      whole path means it.

    Args:
        value: the configured path, or None.
        output_root: the run's root; None falls back to `DEFAULT_OUTPUT_ROOT`,
            so a caller that never passes one still writes somewhere sane.
        default_name: the subdirectory to use when nothing is configured.
            None or empty means the root itself.

    Returns:
        An absolute path under the run's root, unless an absolute path was
        explicitly configured.
    """
    root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT

    if value is None or value == "":
        return root / default_name if default_name else root

    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_batch_output(requested: Optional[str]) -> Path:
    """
    Where a folder run writes its documents, figures, tables and summary.

    For biology researchers: `--out` wins if you give it. Otherwise results go
    to one fixed place inside the V6 tree, NOT inside the folder you scanned.

    Args:
        requested: the `--out` value, or None.

    Returns:
        The run directory. `documents/` is created beneath it.
    """
    return Path(requested) if requested else DEFAULT_OUTPUT_ROOT


def resolve_document_output(pdf_path: Path, requested: Optional[str]) -> Path:
    """
    Where a single PDF's JSON is written.

    For biology researchers: previously this was `<pdf>.v6.json` — *beside the
    PDF*. It now lands beside every other run's output, named after the PDF.

    Args:
        pdf_path: the PDF being processed (its stem names the output).
        requested: the `--out` value, or None. An explicit path is taken
            exactly as given, including its own filename.

    Returns:
        The JSON path to write. Its parent is created by the writer.
    """
    if requested:
        return Path(requested)
    return DEFAULT_OUTPUT_ROOT / "documents" / f"{pdf_path.stem}.v6.json"
