#!/usr/bin/env python3
"""
One PDF, one process — so a document that hangs can be killed.

For biology researchers: a corpus run is fifty minutes of GPU work, and
occasionally one document misbehaves. If everything runs in a single process
there is no way to stop just that document; you kill the run and lose the
twenty-nine good results with it. Running each PDF in its own child process
means the parent can end a bad one and carry on.

    python -m src.processing.worker <job.json>

The job file is a `RunRequest` plus the one document to process. Progress comes
back as **NDJSON on stdout** — one JSON object per line — and the finished
document's metadata arrives in the last event.

**Why a subprocess and NDJSON, rather than `multiprocessing`.** Three reasons,
all practical:

1. The graphical interface runs under uvicorn with a reloader, and
   `multiprocessing` under that is a known source of pain.
2. `Popen.terminate()` is a trivially correct hard kill on Windows. Killing a
   pool worker is not.
3. A job file plus a plain subprocess means **a hung run is reproducible by
   hand** — run the command above and watch. At fifty minutes a corpus, that is
   worth a great deal.

**The ledger is not written here.** The parent owns every ledger write, because
`CsvLedgerBackend` loads the whole file and rewrites it on commit — two writers
is a lost-update race. The child reports what it found; the parent records it.

**A killed document gets no ledger record, and that is correct**: a
half-processed document must not be marked "already processed" and skipped on
the next run.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .batch import BookDecision, PreflightEntry
from .events import EVENT_ATTRIBUTE, EventHandler, RunEvent, emit
from .paths import REPO_ROOT
from .run_request import DocumentOutcome, RunRequest

LOG = logging.getLogger(__name__)

#: The event a child uses to hand its result back. It is the last line the
#: child writes, produced once at the end of the work — never stored, never
#: replayed. See `runner.record_outcome` on why that distinction matters.
OUTCOME_EVENT = "worker_outcome"


def run_in_subprocess(entry: PreflightEntry, request: RunRequest,
                      on_event: Optional[Callable[[RunEvent], None]] = None,
                      on_process: Optional[Callable[[subprocess.Popen], None]] = None,
                      python: Optional[str] = None,
                      logger: Optional[logging.Logger] = None,
                      timeout: Optional[float] = None) -> DocumentOutcome:
    """
    Process one document in a child process. An `Execute` for `process_batch`.

    For biology researchers: this does exactly what processing a document
    normally does, except in a separate program that can be stopped on its own.

    Args:
        entry: the pre-flighted document, carrying its book decision.
        request: the run's order form.
        on_event: called for each progress event as it arrives. Called on a
            reader thread, so it must be cheap — append to a queue, do not
            render.
        on_process: called once, synchronously, right after the child is
            spawned, with the `Popen` handle. This is the GUI's hard-kill
            seam: `BatchController.stop()` stores this handle so a Stop
            click can call `terminate()` on the document actually running,
            not the one requested when Stop was pressed — those can differ
            if the click lands between documents.
        python: interpreter to use; defaults to the one running this code, so
            the child cannot land in a different conda environment.
        logger: optional logger.
        timeout: seconds before the child is killed, or None for no limit.

    Returns:
        A `DocumentOutcome`. A child that dies, times out, or never reports
        comes back with `error` set — never an exception.
    """
    log = logger or LOG
    job = write_job_file(entry, request)
    command = [python or sys.executable, "-u", "-m", "src.processing.worker", str(job)]

    outcome_holder: Dict[str, Any] = {}
    process = subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8",
                               errors="replace", bufsize=1, env=child_environment(request))
    if on_process is not None:
        on_process(process)

    reader = threading.Thread(
        target=_drain, args=(process.stdout, outcome_holder, on_event, log),
        daemon=True)
    reader.start()

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log.error("%s exceeded %.0fs; terminating.", entry.path.name, timeout)
        terminate(process)
        return DocumentOutcome(path=entry.path,
                               error=f"TimeoutExpired: exceeded {timeout}s")
    finally:
        reader.join(timeout=5)

    stderr = (process.stderr.read() or "").strip() if process.stderr else ""
    if outcome_holder.get("outcome") is not None:
        return outcome_holder["outcome"]

    # No result and no exception: the child died before it could report.
    detail = stderr.splitlines()[-1] if stderr else f"exit code {process.returncode}"
    return DocumentOutcome(path=entry.path,
                           error=f"WorkerFailed: {detail}")


def terminate(process: subprocess.Popen) -> None:
    """
    Stop a child, hard.

    `terminate()` first, then `kill()` if it does not go. This is the whole
    reason the work is in a subprocess rather than a thread: a thread running
    a GPU call cannot be interrupted, and a process can.
    """
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:                        # pragma: no cover
        process.kill()
        process.wait(timeout=10)


def child_environment(request: RunRequest) -> Optional[Dict[str, str]]:
    """
    The child's environment: this one, plus the run's ledger file if it set one.

    The pipeline inside the child finds its ledger through `SARA_LEDGER_PATH`
    (strategy.py). A ledger chosen in the GUI's Settings travels on the request,
    so it has to be handed over here or the child reads the default ledger.

    Returns:
        None (inherit unchanged) when the request names no ledger.
    """
    if not request.ledger_path:
        return None
    return {**os.environ, "SARA_LEDGER_PATH": str(Path(request.ledger_path).resolve())}


def write_job_file(entry: PreflightEntry, request: RunRequest,
                   directory: Optional[Path] = None) -> Path:
    """
    Write the job a child reads: the run's request plus this one document.

    Kept as a real file rather than piped on stdin so that a failure is
    reproducible by hand — `python -m src.processing.worker <job.json>` is the
    whole repro.
    """
    directory = directory or (request.out_root / ".jobs")
    directory.mkdir(parents=True, exist_ok=True)

    decision = entry.decision or BookDecision.safe_default()
    payload = {
        "request": request.to_dict(),
        "path": str(entry.path),
        # `chapters` written unconditionally: None and [] are opposite
        # instructions and one of them is falsy. See `run_request`.
        "decision": {"chapters": decision.chapters,
                     "max_pages": decision.max_pages,
                     "reason": decision.reason},
    }

    job_path = directory / f"{entry.path.stem}.job.json"
    with open(job_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return job_path


def load_job(job_path: Path) -> tuple:
    """Read a job file back into `(entry, request)`. The other half of the trip."""
    with open(job_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    request = RunRequest.from_dict(payload.get("request") or {})
    raw = payload.get("decision") or {}
    decision = BookDecision(chapters=raw.get("chapters"),
                            max_pages=raw.get("max_pages"),
                            reason=raw.get("reason") or "from job file")
    entry = PreflightEntry(path=Path(payload["path"]), decision=decision)
    return entry, request


def main(argv: Optional[List[str]] = None) -> int:
    """
    The child's entrypoint: process one document, report as NDJSON on stdout.

    Every log record in this process becomes one JSON line, so the parent sees
    the whole transcript and not merely the progress counters.
    """
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print("usage: python -m src.processing.worker <job.json>", file=sys.stderr)
        return 2

    entry, request = load_job(Path(argv[0]))
    _install_ndjson_logging(request.verbose)
    log = logging.getLogger("worker")

    from .runner import process_one            # imported late: it loads the config tree

    outcome = process_one(entry, request, logger=log)
    emit(log, OUTCOME_EVENT, message=f"Worker finished {entry.path.name}",
         path=str(entry.path), elapsed=outcome.elapsed,
         outcome=_outcome_to_dict(outcome))
    sys.stdout.flush()
    return 0 if outcome.error is None else 2


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _install_ndjson_logging(verbose: bool) -> None:
    """
    Send every log record to stdout as one JSON object per line.

    Nothing else may write to stdout in this process — that is the contract of
    the stream. Anything that does anyway becomes a `log` event rather than an
    exception, because a stray `print` in a dependency must not kill a run.
    """
    # THE STREAM IS UTF-8, ALWAYS. Windows gives a redirected stdout cp1252 by
    # default, and the pipeline's own log messages contain em dashes and Greek
    # letters — so without this the child writes bytes the parent cannot
    # decode. Measured: an em dash arrived as 0x97 and broke the read. The
    # parent decodes as UTF-8, so this is the two halves of one contract.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):                     # pragma: no cover
        pass

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(EventHandler(_write_ndjson,
                                 level=logging.DEBUG if verbose else logging.INFO))


def _write_ndjson(event: RunEvent) -> None:
    """One event, one line, flushed — the parent is reading as we go."""
    sys.stdout.write(event.to_json() + "\n")
    sys.stdout.flush()


def _outcome_to_dict(outcome: DocumentOutcome) -> Dict[str, Any]:
    """The child's report: plain data only, since it crosses a process boundary."""
    return {
        "path": str(outcome.path),
        "output": str(outcome.output) if outcome.output else None,
        "elapsed": outcome.elapsed,
        "metadata": outcome.metadata,
        "document_type": getattr(outcome.document_type, "value",
                                 outcome.document_type),
        "errors": list(outcome.errors),
        "error": outcome.error,
        "skipped": outcome.skipped,
    }


def _outcome_from_dict(raw: Dict[str, Any]) -> DocumentOutcome:
    """Rebuild the child's report in the parent."""
    return DocumentOutcome(
        path=Path(raw.get("path") or ""),
        output=Path(raw["output"]) if raw.get("output") else None,
        elapsed=raw.get("elapsed") or 0.0,
        metadata=raw.get("metadata") or {},
        document_type=raw.get("document_type"),
        errors=list(raw.get("errors") or []),
        error=raw.get("error"),
        skipped=raw.get("skipped"),
    )


def _drain(stream, holder: Dict[str, Any],
           on_event: Optional[Callable[[RunEvent], None]],
           log: logging.Logger) -> None:
    """
    Read the child's stdout line by line, turning each into an event.

    A line that is not valid JSON becomes a plain `log` event rather than an
    error — see `RunEvent.from_json`. The child's stdout is not ours alone.
    """
    if stream is None:                                       # pragma: no cover
        return

    for line in stream:
        if not line.strip():
            continue

        event = RunEvent.from_json(line)
        if event.kind == OUTCOME_EVENT:
            raw = (event.payload or {}).get("outcome")
            if isinstance(raw, dict):
                holder["outcome"] = _outcome_from_dict(raw)
            continue

        # The child's log becomes the parent's log, so a terminal run looks
        # exactly as it did before anything ran in a subprocess.
        log.log(logging.getLevelName(event.level or "INFO")
                if isinstance(logging.getLevelName(event.level or "INFO"), int)
                else logging.INFO,
                "%s", event.message or "",
                extra={EVENT_ATTRIBUTE: {"kind": event.kind, "stage": event.stage,
                                         "page": event.page,
                                         "page_total": event.page_total,
                                         "path": event.path}})
        if on_event is not None:
            on_event(event)


if __name__ == "__main__":
    sys.exit(main())
