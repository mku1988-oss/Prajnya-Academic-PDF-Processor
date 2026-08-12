#!/usr/bin/env python3
"""
BatchController — runs a folder of PDFs as one background batch.

For biology researchers: this is the object a Library/Batch page talks to.
It does the same work `run_prajnya.py --pdf <folder>` does — pre-flight
every PDF on the CPU, settle what to do with any book, then process each
document one subprocess at a time — except it runs on a background thread
instead of blocking a terminal, and reports its progress through a small
poll-friendly interface instead of printing to a console.

**Why a thread, not `asyncio`.** NiceGUI's page code runs on the same
event loop that serves every browser request; a fifty-minute batch run on
that loop would freeze the whole app for every user of it. So the batch
runs on a plain background `threading.Thread`, and a page reads its state
back through `snapshot()` / `events_since()` from a `ui.timer` — the "one
progress mechanism" gui_plan.md section 5 specifies, replacing V5's four
separate progress callbacks with one polled event log.

**Why book questions block the thread instead of returning a future.**
`resolve_decisions` (the same function the CLI's terminal prompt drives)
expects a plain synchronous callable — ask, get an answer, move to the
next book. Keeping that contract lets the CLI and the GUI share the exact
same decision logic; the GUI's callable just blocks on a
`threading.Event` instead of on `input()`, and the page sets that event
from `answer_book_decision()` once its dialog closes.

**Why events are captured only once, not from two places.** `on_event`
passed to `run_in_subprocess` and a `logging.Handler` attached to this
controller's own logger both see the same child events — `run_in_subprocess`
re-logs every NDJSON line from the child onto whichever logger it was
given, on top of also offering a direct callback. Subscribing to both would
double every child-process event in the transcript. This module uses the
handler only, so both the parent's own events (`document_start`,
`batch_done`) and the child's re-logged ones arrive through one path.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.processing.batch import BookDecision, PreflightEntry, write_batch_summary
from src.processing.events import EventHandler, RunEvent
from src.processing.run_request import DocumentOutcome, RunRequest
from src.processing.runner import open_ledger, process_batch, resolve_decisions, run_preflight
from src.processing.worker import run_in_subprocess, terminate

from .session import RunRegistry

#: Bounded so a very long or very verbose run cannot grow the transcript
#: without limit — same size as `worker.run_in_subprocess`'s own reasoning
#: for keeping progress consumption cheap.
MAX_EVENTS = 5000


class BatchState(str, Enum):
    """Where a batch is, for a page to render without re-deriving it."""

    IDLE = "idle"
    PREFLIGHT = "preflight"
    AWAITING_BOOK_DECISION = "awaiting_book_decision"
    RUNNING = "running"
    STOPPING = "stopping"
    DONE = "done"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class BookQuestion:
    """One book-like document waiting on a user answer, for a page dialog."""

    entry: PreflightEntry


class BatchController:
    """
    Runs `request.paths` as one batch, on a background thread.

    A page never calls into the pipeline directly. It builds a controller,
    registers it with `RunRegistry` via `start()` — which raises
    `RunAlreadyActive` if another batch is already running, since this
    project has one 8GB GPU and concurrency was tested four ways and
    rejected (see PERFORMANCE.md) — then polls `snapshot()` and
    `events_since()` from a `ui.timer`.
    """

    def __init__(self, request: RunRequest, noninteractive: bool = False,
                 logger: Optional[logging.Logger] = None) -> None:
        self.request = request
        self.noninteractive = noninteractive

        # A unique logger per controller, so its EventHandler never leaks
        # onto a shared logger between runs, and an explicit level so
        # events are captured regardless of the process's root log level
        # (which pytest, in particular, leaves at WARNING).
        self.log = logger or logging.getLogger(f"prajnya_studio.controller.{id(self):x}")
        self.log.setLevel(logging.INFO)

        self.entries: List[PreflightEntry] = []
        self.results: Dict[str, Any] = {}
        self.state: BatchState = BatchState.IDLE
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

        self._events: Deque[RunEvent] = deque(maxlen=MAX_EVENTS)
        self._events_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self._stop_requested = False
        self._current_process: Any = None
        self._process_lock = threading.Lock()

        self._pending_question: Optional[PreflightEntry] = None
        self._answer_ready: Optional[threading.Event] = None
        self._decision_answer: Optional[BookDecision] = None
        self._question_lock = threading.Lock()

    # ---- lifecycle, called from the UI thread ------------------------------

    def start(self) -> None:
        """
        Registers this controller as the one active run and launches it.

        Raises:
            RunAlreadyActive: another batch is already running. The page
                should show that, not start a second one — see
                `session.RunRegistry`.
        """
        RunRegistry.get_instance().start(self)
        self.state = BatchState.PREFLIGHT
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Hard-kills the document currently running, and starts no more.

        For biology researchers: the document being processed right now is
        stopped immediately — its partial work is discarded, same as
        closing the CLI with Ctrl-C mid-document — but any document already
        finished keeps its result. See `_execute` for why the killed
        document is reported as stopped rather than as a generic failure.
        """
        self._stop_requested = True
        if self.state not in (BatchState.DONE, BatchState.STOPPED, BatchState.FAILED):
            self.state = BatchState.STOPPING
        with self._process_lock:
            process = self._current_process
        if process is not None:
            terminate(process)

    def answer_book_decision(self, decision: BookDecision) -> None:
        """
        Unblocks a pending book question with the user's choice.

        Called from the UI thread once a dialog resolves. A no-op if
        nothing is currently waiting, so a stale or double click cannot
        answer the wrong question.
        """
        with self._question_lock:
            ready = self._answer_ready
            if ready is None:
                return
            self._decision_answer = decision
            self._answer_ready = None
        ready.set()

    # ---- polling, called from a ui.timer -----------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Everything a page needs to redraw itself from scratch on mount."""
        with self._question_lock:
            question = (BookQuestion(entry=self._pending_question)
                        if self._pending_question is not None else None)
        return {
            "state": self.state,
            "entries": list(self.entries),
            "results": dict(self.results),
            "error": self.error,
            "question": question,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def events_since(self, cursor: int) -> Tuple[List[RunEvent], int]:
        """
        Every event after `cursor`, and the cursor to pass next time.

        `cursor` is a plain count of events seen so far, not an id — the
        deque is bounded, so an old event can fall off the front. A page
        that has been idle long enough to miss events should re-mount from
        `snapshot()` rather than trust a stale cursor; this is the same
        trade `MAX_EVENTS` already makes for memory.
        """
        with self._events_lock:
            events = list(self._events)
        return events[cursor:], len(events)

    # ---- the background thread ---------------------------------------------

    def _run(self) -> None:
        handler = EventHandler(self._on_event)
        self.log.addHandler(handler)
        try:
            self.started_at = time.time()
            self.entries = run_preflight(self.request, logger=self.log)

            prompt = None if self.noninteractive else self._threaded_prompt
            self.state = BatchState.AWAITING_BOOK_DECISION
            resolve_decisions(self.entries, self.request, prompt=prompt,
                              logger=self.log)

            if self._stop_requested:
                self.state = BatchState.STOPPED
                return

            self.state = BatchState.RUNNING
            ledger = open_ledger(self.log)
            results = process_batch(self.entries, self.request, ledger,
                                    execute=self._execute, logger=self.log)
            self.results = results
            write_batch_summary(self.entries, results, self.request.out_root,
                                time.time() - self.started_at)

            self.state = BatchState.STOPPED if self._stop_requested else BatchState.DONE
        except Exception as error:                                # pragma: no cover
            self.error = f"{type(error).__name__}: {error}"
            self.state = BatchState.FAILED
            self.log.error("Batch controller failed: %s", self.error, exc_info=True)
        finally:
            self.finished_at = time.time()
            self.log.removeHandler(handler)
            RunRegistry.get_instance().finish()

    def _execute(self, entry: PreflightEntry, request: RunRequest) -> DocumentOutcome:
        """
        The `Execute` handed to `process_batch` — one document, one child.

        Checked for a stop request both before spawning (so a Stop click
        between documents starts no more) and after the child returns (so a
        document killed mid-run is reported as stopped, not as an opaque
        `WorkerFailed`).
        """
        if self._stop_requested:
            return DocumentOutcome(path=entry.path, error="Stopped by user")

        outcome = run_in_subprocess(entry, request,
                                    on_process=self._register_process,
                                    logger=self.log)
        if self._stop_requested and outcome.error:
            outcome.error = "Stopped by user"
        return outcome

    def _register_process(self, process: Any) -> None:
        """Recorded so `stop()` can kill the document actually running."""
        with self._process_lock:
            self._current_process = process

    def _on_event(self, event: RunEvent) -> None:
        with self._events_lock:
            self._events.append(event)

    def _threaded_prompt(self, entry: PreflightEntry) -> BookDecision:
        """
        Blocks the background thread until the UI answers.

        `resolve_decisions` calls this once per book-like document, in
        order — one question is ever pending at a time, which is why the
        page only ever needs to render one dialog.
        """
        ready = threading.Event()
        with self._question_lock:
            self._pending_question = entry
            self._answer_ready = ready
            self._decision_answer = None

        ready.wait()

        with self._question_lock:
            decision = self._decision_answer
            self._pending_question = None
        return decision or BookDecision.safe_default()
