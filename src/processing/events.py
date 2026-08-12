#!/usr/bin/env python3
"""
Progress a machine can read — carried by the logging you already have.

For biology researchers: a corpus run takes about fifty minutes. Something has
to tell you what it is doing while you wait, and "something" is about to be a
graphical interface rather than a scrolling terminal. This module is how the
pipeline says "page 4 of 8 of this paper" in a form a program can act on.

**Why logging rather than a callback.** The obvious design is to thread a
`Callable[[RunEvent], None]` down through `BodyPipeline` -> `figures` ->
`tables`. That adds a parameter to about six modules on the GPU-hot path for
the sake of a page counter, and every one of those signatures then exists to
serve a progress bar. The obvious alternative — parsing the log prose — breaks
the moment someone rewords a message.

Take neither. `logging` **already** threads through every module, is already
the injected `logger=` seam everywhere, and already carries a structured
payload via `extra=`. So a module that wants to report progress logs its line
exactly as before and attaches a dict; a caller that wants events installs a
handler. Nothing in `src/` changes shape.

    emit(log, "page", message="Page 4", page=4, page_total=8)

The command line sees one ordinary log line and knows nothing about events.
The GUI installs `EventHandler` and receives every record — the typed ones as
themselves, and **the entire rest of the log for free** as `log` events. For a
fifty-minute run that transcript is what a person actually watches; a
percentage bar is the smaller half of the value.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional

#: The key an event travels under on a `LogRecord`. A record without it is
#: still an event — just a plain `log` one.
EVENT_ATTRIBUTE = "sara_event"

#: A message longer than this is truncated before it becomes an event. A
#: transcript pane does not need a full page of OCR text on one line, and an
#: unbounded string here is an unbounded queue there.
MAX_MESSAGE_CHARS = 2000

#: The kinds an event can be. `log` is the catch-all every ordinary log line
#: becomes; the rest are emitted deliberately.
EVENT_KINDS = (
    "document_start",
    "document_done",
    "document_failed",
    "stage",
    "page",
    "batch_done",
    "log",
)

#: Where a document currently is. Attached to `stage` and `page` events.
STAGES = ("profile", "title", "body", "figures", "tables", "gpu_swap", "write")


@dataclass
class RunEvent:
    """
    One thing that happened during a run, as data.

    Every field is optional but `kind`, because an event is a partial report:
    a page event knows its page number, a batch event knows its totals, and
    neither needs the other's fields. Consumers read what they recognise.
    """

    kind: str = "log"
    ts: float = field(default_factory=time.time)
    path: Optional[str] = None
    position: Optional[int] = None
    total: Optional[int] = None
    stage: Optional[str] = None
    page: Optional[int] = None
    page_total: Optional[int] = None
    elapsed: Optional[float] = None
    level: Optional[str] = None
    logger: Optional[str] = None
    message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """One line of NDJSON. Never raises — see `_safe`."""
        data = {key: value for key, value in asdict(self).items()
                if value is not None}
        return json.dumps(data, ensure_ascii=False, default=_safe)

    @classmethod
    def from_json(cls, line: str) -> "RunEvent":
        """
        Parse one NDJSON line.

        A line that is not valid JSON becomes a `log` event carrying the raw
        text. That is deliberate and load-bearing: a stray `print` in some
        dependency must never kill a fifty-minute run, and a child process's
        stdout is not ours alone.
        """
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("not an object")
        except Exception:
            return cls(kind="log", level="INFO", message=line.rstrip("\n"))

        known = {key: raw.get(key) for key in cls.__dataclass_fields__
                 if key in raw}
        known.setdefault("kind", "log")
        known.setdefault("ts", time.time())
        return cls(**known)


def emit(logger: logging.Logger, kind: str, level: int = logging.INFO,
         **fields: Any) -> None:
    """
    Log a line AND attach a structured event to it.

    For biology researchers: this writes one ordinary line to the log, exactly
    as before, and quietly staples some machine-readable facts to it. The
    terminal shows the line and ignores the rest; a graphical interface reads
    the facts and moves its progress bar.

    Args:
        logger: the module's own logger — the same one it already uses.
        kind: one of `EVENT_KINDS`.
        level: logging level for the human-readable line.
        **fields: any `RunEvent` field. `message` is the line that gets
            logged; anything not a known field is collected into `payload`,
            so a caller can attach detail without this module knowing about it.

    Example:
        >>> emit(log, "page", message="Page 4 of 8", page=4, page_total=8)
        ... # doctest: +SKIP
    """
    fields = dict(fields)
    message = fields.pop("message", None) or kind

    known = {key: fields.pop(key) for key in list(fields)
             if key in RunEvent.__dataclass_fields__}
    if fields:
        # Anything left over is detail this module does not model.
        known["payload"] = {**(known.get("payload") or {}), **fields}

    known["kind"] = kind
    known["message"] = _truncate(message)
    logger.log(level, message, extra={EVENT_ATTRIBUTE: known})


class EventHandler(logging.Handler):
    """
    Turn every log record into an event, and hand it to a sink.

    For biology researchers: installing this on the root logger is how a
    graphical interface listens to a run. It sees the deliberate progress
    events AND every other log line the pipeline writes, which together are
    the transcript a person actually reads while waiting.

    The sink is called on the logging thread, so it must be cheap — append to
    a queue, never render. A sink that raises is swallowed by
    `logging.Handler.handleError`: a broken progress display must not be able
    to stop a run that is otherwise fine.
    """

    def __init__(self, sink: Callable[[RunEvent], None],
                 level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:      # noqa: A003
        try:
            self.sink(event_from_record(record))
        except Exception:                                    # pragma: no cover
            self.handleError(record)


def event_from_record(record: logging.LogRecord) -> RunEvent:
    """
    The event a log record carries — the typed one, or a plain `log`.

    A record written by `emit` has its fields under `EVENT_ATTRIBUTE`. Every
    other record in the process — third-party libraries included — becomes a
    `log` event with its level, logger name and message, which is how the
    transcript arrives complete without anyone instrumenting for it.
    """
    payload = getattr(record, EVENT_ATTRIBUTE, None)
    fields = dict(payload) if isinstance(payload, dict) else {}

    fields.setdefault("kind", "log")
    fields.setdefault("message", _truncate(record.getMessage()))
    fields.setdefault("ts", record.created)
    fields["level"] = record.levelname
    fields["logger"] = record.name

    known = {key: value for key, value in fields.items()
             if key in RunEvent.__dataclass_fields__}
    return RunEvent(**known)


# --- helpers (after the main definitions, per CLAUDE.md) --------------------

def _truncate(message: Any) -> str:
    """A message long enough to flood a transcript is cut, with a marker."""
    text = message if isinstance(message, str) else str(message)
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[:MAX_MESSAGE_CHARS] + f"... [{len(text) - MAX_MESSAGE_CHARS} more]"


def _safe(value: Any) -> Any:
    """
    Serialize what `json` cannot, so one odd field cannot cost a whole event.

    The same shape as `document_json.last_resort_encoder`, and for the same
    reason: a numpy scalar escaping from OpenCV is a real thing that has
    happened here. Degrade one field, never the line.
    """
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:                                    # pragma: no cover
            pass
    return repr(value)
