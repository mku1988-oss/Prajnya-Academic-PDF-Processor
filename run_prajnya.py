#!/usr/bin/env python
"""
Shastrasaara V6 entrypoint — process a PDF, or a folder of them.

For biology researchers: point this at a paper and it runs the whole pipeline
— CPU pre-pass, classification, title-page metadata, body reading, table
repair, figure extraction — and writes the result as JSON.

    run_prajnya.py --pdf "paper.pdf"
    run_prajnya.py --pdf "paper.pdf" --pages 1 4 7
    run_prajnya.py --pdf "C:/papers"                    a whole folder
    run_prajnya.py --pdf "C:/papers" --recursive --noninteractive
    run_prajnya.py --review                             fix flagged metadata

**Every document is recorded, and incomplete ones are flagged.** A citation
needs a title, authors, a DOI and a journal; anything short of all four is
written to the ledger as `provisional` and offered for correction — at the end
of a batch, never in the middle of one. `--review` picks up whatever is still
flagged, from any earlier run, without reading a PDF or calling a model.

**Books are asked about, not assumed.** Whatever a document turns out to be is
decided before it is read, because the difference between a paper and a
345-page volume is minutes versus hours of GPU. Point this at a folder and it
classifies everything first, shows you what it found, and settles what to do
with any book *before* the first GPU call. Then it runs unattended.

`--chapters` answers the same question from a script, and `--noninteractive`
takes the cheap default (front matter and chapter map only) — never the
expensive one, since nobody is watching to stop it.

The local llama-server (router mode) must already be running for the offline
path; see CLAUDE.md, "Running things".

---

**This file is a front end, and only a front end.** The work itself lives in
`src/processing/runner.py`, so that the graphical interface can do exactly the
same work without importing argparse or a terminal prompt. What stays here is
the command line, logging setup, the terminal's way of asking questions, and
the exit code. Everything below the line "hand it to the runner" is shared.
"""

from __future__ import annotations

import sys

# =============================================================================
# Interpreter guard -- see run_prajnya_studio.py for the full reasoning
# =============================================================================
#
# Short version, all of it measured: the shebang above says `python`, not
# `python3`, because running this file by its own filename on Windows hands it
# to `C:\WINDOWS\py.exe`, which resolves a `/usr/bin/env <name>` shebang
# against PATH -- and a conda env ships only `python.exe`, no `python3.exe`.
# `python3` therefore misses and py.exe falls back to its highest REGISTERED
# interpreter (C:\Python313), ignoring the activated environment. `python`
# hits the env and the obvious command works.
#
# The version check is what makes that safe: with `python` and the env NOT
# activated, PATH falls through to miniconda3 base, which has Python 3.12.9,
# numpy 2.0.1 AND python-dotenv -- it would run happily on the wrong numpy
# major version, which is the silent failure this project cannot afford.
#
# Duplicated in the two entrypoints rather than shared: an entrypoint that
# imports a helper in order to report that imports are broken has a
# bootstrapping problem. Messages are ASCII-only (Windows console is cp1252).

#: The `shastrasaara` conda env is Python 3.10.16. A pin, not a discovery --
#: change this line if the environment is ever deliberately upgraded.
_REQUIRED_PYTHON = (3, 10)


def _wrong_interpreter(problem: str) -> None:
    """Stop with an explanation instead of a traceback about a package that,
    on the correct interpreter, is installed."""
    sys.exit(
        f"\nShastrasaara could not start: {problem}\n"
        "\n"
        f"  interpreter : {sys.executable}\n"
        f"  version     : {sys.version.split()[0]}"
        f"   (this project needs {_REQUIRED_PYTHON[0]}.{_REQUIRED_PYTHON[1]}.x)\n"
        "\n"
        "Use the project's environment -- a conda env named 'shastrasaara',\n"
        "created from requirements.txt:\n"
        "\n"
        "  conda activate shastrasaara\n"
        "  python run_prajnya.py --pdf \"paper.pdf\"\n"
        "\n"
        "If activation cannot find it by name, it is registered by path.\n"
        "'conda env list' prints where it is, and naming its interpreter\n"
        "directly always works:\n"
        "\n"
        "  <env>/python.exe run_prajnya.py --pdf \"paper.pdf\"\n"
        "\n"
        "See README.md, 'The Python environment'.\n"
    )


if sys.version_info[:2] != _REQUIRED_PYTHON:
    _wrong_interpreter("the wrong Python is running this file.")

# load_dotenv() BEFORE any project import: config/loader.py interpolates
# ${ENV} from os.environ but does not read .env itself (see CLAUDE.md,
# "Modularity contracts"). Skipping this silently disables Zotero.
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    _wrong_interpreter("this Python has none of the project's dependencies.")

load_dotenv()

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.processing.batch import PreflightEntry, find_pdfs, write_batch_summary
from src.processing.book_prompt import ask_about_book, show_preflight
from src.processing.cli_support import parse_chapters, parse_pages

# Several names below are imported but not called here. That is deliberate:
# they used to be DEFINED in this file, and callers — the test suite among them
# — still reach for `run_prajnya.build_document_json`, `record_document`,
# `resolve_document_output` and friends. Re-exporting keeps this module the
# stable public face of a run while the implementations live under `src/`,
# where a second front end can reach them without importing a command line.
from src.processing.document_json import (
    PIPELINE_VERSION,
    build_document_json,
    last_resort_encoder as _last_resort_encoder,
    write_result as _write_result,
)
from src.processing.extractors.front_matter import DEFAULT_FRONT_MATTER_PAGES
from src.processing.ledger.semantic_ledger import LedgerRecord
from src.processing.paths import (
    CONFIG_ROOT,
    DEFAULT_LEDGER_PATH,
    DEFAULT_OUTPUT_ROOT,
    resolve_batch_output,
    resolve_document_output,
)
from src.processing.review import regenerate_document_json, regenerate_from_ledger
from src.processing.review_prompt import (
    offer_deferred_review,
    review_ledger,
    review_record,
)
from src.processing.run_request import RunRequest
from src.processing.runner import (
    build_document_profile,
    decide_for_single,
    describe_book,
    open_ledger as _open_ledger,
    process_batch,
    process_one,
    record_document,
    record_outcome,
    resolve_decisions,
    run_preflight,
)


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)
    log = logging.getLogger("run_prajnya")

    # V5 accepted "--pages 2-5"; keep that habit working, and let --chapters
    # take the same shorthand.
    args.pages = parse_pages(args.pages)
    args.chapters = parse_chapters(args.chapters, log)

    if args.review:
        return run_review(log)

    if not args.pdf:
        log.error("Give me a --pdf (a file or a folder), or --review.")
        return 1

    target = Path(args.pdf)
    if target.is_dir():
        return run_folder(target, args, log)
    if not target.is_file():
        log.error("Not a PDF or a folder: %s", target)
        return 1
    return run_one_document(target, args, log)


def run_review(log: logging.Logger) -> int:
    """
    Work through the ledger's flagged records — no PDF, no GPU, no model.

    For biology researchers: this is `--review` on its own. It opens the
    library's record book, finds every document whose metadata came out
    incomplete, and lets you fix them one at a time. Nothing is re-read.
    """
    ledger = _open_ledger(log, create=False)
    if ledger is None:
        log.error("No ledger to review (looked for %s).",
                  os.getenv("SARA_LEDGER_PATH", DEFAULT_LEDGER_PATH))
        return 1

    review_ledger(ledger, logger=log)
    return 0


def run_folder(folder: Path, args: argparse.Namespace,
               log: logging.Logger) -> int:
    """
    Process every PDF in a folder, deciding the expensive questions first.

    For biology researchers: this classifies everything up front — cheap, CPU
    only — shows you what it found, and settles what to do with any book
    BEFORE any GPU work starts. Then it runs without interruption, so the
    batch can be left alone. A document that fails is recorded and the rest
    continue.

    All this function does is supply the terminal's answers — a table to show
    the pre-flight, a prompt to ask about a book, a question at the end about
    incomplete metadata. The run itself belongs to `runner.process_batch`.
    """
    paths = find_pdfs(folder, recursive=args.recursive)
    if not paths:
        log.error("No PDFs in %s", folder)
        return 1

    request = RunRequest.from_args(args, paths=paths)
    entries = run_preflight(request, logger=log)
    show_preflight(entries)
    resolve_decisions(entries, request,
                      prompt=None if args.noninteractive else ask_about_book,
                      logger=log)

    ledger = _open_ledger(log)
    start = time.time()
    results = process_batch(entries, request, ledger, logger=log)
    elapsed = time.time() - start

    _offer_review(results, ledger, args, log)

    summary_path = write_batch_summary(entries, results, request.out_root, elapsed)
    failed = sum(1 for value in results.values() if value.get("error"))
    log.info("Batch done: %d processed, %d failed, %.1fs -> %s",
             len(results) - failed, failed, elapsed, summary_path)
    return 2 if failed else 0


def run_one_document(pdf_path: Path, args: argparse.Namespace,
                     log: logging.Logger) -> int:
    """
    Process a single PDF, asking about a book unless told not to.

    A single document takes the SAME path a batch takes — it is built into a
    one-entry pre-flight list and handed to `runner.process_one`. Keeping one
    document and thirty on one code path is what stops them drifting apart.
    """
    request = RunRequest.from_args(args)
    backend = _backend_for(request)
    profile = build_document_profile(pdf_path, backend, args.front_matter_pages)
    decision = decide_for_single(
        profile, pdf_path, request,
        prompt=None if args.noninteractive else ask_about_book,
        backend=backend, logger=log)

    entry = PreflightEntry(path=pdf_path, document_type=profile.document_type,
                           confidence=profile.classification_confidence or 0.0,
                           page_count=profile.page_count, decision=decision)
    # The profile is handed on rather than rebuilt: deciding whether this was a
    # book already cost one full front-matter scan, and the answer has not
    # changed since.
    outcome = process_one(entry, request, backend=backend, logger=log,
                          profile=profile)

    if outcome.error:
        log.error("Failed: %s", outcome.error)
        return 2

    ledger = _open_ledger(log)
    record = record_outcome(pdf_path, outcome.metadata, outcome.document_type,
                            outcome.output, ledger, log)
    if record.requires_review and not args.noninteractive:
        _review_now(record, outcome.output, ledger, log)

    if outcome.errors:
        log.error("Completed with errors: %s", outcome.errors)
        return 2
    return 0


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _backend_for(request: RunRequest):
    """The PDF backend this run uses — built once and shared, as a batch does."""
    from src.processing.pdf import get_backend

    return get_backend(request.backend)


def _review_now(record: LedgerRecord, out_path: Optional[Path], ledger,
                log: logging.Logger) -> None:
    """
    Fix one document's metadata straight away, then re-write its JSON.

    Only ever reached for a SINGLE document at a terminal. A batch defers all
    of this to the end (see `offer_deferred_review`) — a run that stops to ask
    a question halfway is a run nobody can leave alone.
    """
    corrected = review_record(record)
    if corrected is None:            # EOF or Ctrl-C: leave it flagged
        return

    from src.processing.review import save_record

    save_record(ledger, corrected, log)
    if out_path is not None:
        regenerate_document_json(out_path, corrected, log)


def _offer_review(results: Dict[str, Any], ledger, args: argparse.Namespace,
                  log: logging.Logger) -> None:
    """
    Once the batch is done, offer to fix everything that came out incomplete.

    For biology researchers: this is the end-of-run moment. Whatever needed
    your attention was collected while the batch ran without interrupting it;
    now you are asked once. Corrections are written back into the affected
    JSON files, and only those.
    """
    pending = {Path(path): Path(value["output"])
               for path, value in results.items()
               if value.get("requires_review") and value.get("output")}

    if not pending:
        return

    if args.noninteractive or ledger is None:
        log.info("%d document(s) need metadata review — run with --review.",
                 len(pending))
        return

    if not offer_deferred_review(list(pending), ledger, logger=log):
        return

    updated = regenerate_from_ledger(pending, ledger, log)
    log.info("Re-wrote %d output file(s) from the corrected ledger.", updated)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a research article PDF (V6)")
    parser.add_argument("--pdf", default=None,
                        help="path to a PDF, or to a FOLDER of PDFs")
    parser.add_argument("--review", action="store_true",
                        help="review flagged metadata in the ledger and stop "
                             "(no PDF is read, no model is called)")
    parser.add_argument("--recursive", action="store_true",
                        help="with a folder, search subdirectories too")
    parser.add_argument("--noninteractive", action="store_true",
                        help="never prompt; books take the safe default "
                             "(front matter and chapter map only)")
    parser.add_argument("--pages", nargs="*", default=None,
                        help="1-based pages: '4', '2-5', or '1 3-5' (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="re-process even if the ledger already has the document")
    parser.add_argument("--out", default=None,
                        help="where results go. For one PDF, a JSON path; for a "
                             "folder, the run directory. Default: "
                             f"{DEFAULT_OUTPUT_ROOT} (documents/, figures/, tables/ "
                             "and the batch summary all live under it)")
    parser.add_argument("--chapters", nargs="*", default=None,
                        help="BOOKS ONLY: which chapters to read, e.g. --chapters 3 4 "
                             "or --chapters 3-5. "
                             "Omit to read the front matter and chapter map only "
                             "(seconds, no GPU). Passing this with no numbers reads "
                             "EVERY chapter, which for a full volume is hours of GPU")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="BOOKS ONLY: ceiling on how many body pages to read")
    parser.add_argument("--front-matter-pages", type=int,
                        default=DEFAULT_FRONT_MATTER_PAGES,
                        help="how many leading pages to scan for DOI/ISBN/ISSN "
                             f"(default: {DEFAULT_FRONT_MATTER_PAGES})")
    parser.add_argument("--backend", default=None,
                        help="PDF backend name (default: pypdfium2)")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    sys.exit(main())
