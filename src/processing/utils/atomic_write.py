#!/usr/bin/env python3
"""
Write a JSON file so that a crash never leaves half of it (audit D4).

For biology researchers: writing a file "in place" first empties it, then
fills it. If the run is killed in between (Stop, a crash, a full disk, a power
cut), what is left is a cut-off file that no reader can parse, and the good
file from the last run is already gone. A document JSON stands for minutes of
GPU work, so that is an expensive way to lose it.

The safe way, used here:

    1. write the whole file under a temporary name in the SAME folder;
    2. flush it to disk;
    3. swap it in with `os.replace`, which replaces the old file in one step.

Until step 3 the old file is untouched; after it the new file is complete.
A reader sees either the old file or the new one, never a mix. The temporary
file must sit in the same folder because `os.replace` is only a single step
within one drive.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

# On Windows the swap fails for a moment when another program (Prajnya
# Studio's file view, a virus scanner, a sync client) has the old file open.
# A few short retries ride that out; a lasting lock still raises.
REPLACE_ATTEMPTS = 5
REPLACE_PAUSE_SECONDS = 0.2


def write_json_atomically(path: Path, payload: Any,
                          default: Optional[Callable[[Any], Any]] = None) -> None:
    """
    Write `payload` as indented UTF-8 JSON to `path`, all or nothing.

    Args:
        path: the file to (re)write. Its folder must exist.
        payload: anything `json.dump` accepts.
        default: optional fallback encoder for unknown types, as in `json.dump`.

    Raises:
        Whatever the write raised. The old file is then still in place and no
        temporary file is left behind.
    """
    path = Path(path)
    temporary = _temporary_beside(path)
    try:
        _write_and_flush(temporary, payload, default)
        _replace_with_retries(temporary, path)
    except BaseException:
        _remove_quietly(temporary)
        raise


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def _temporary_beside(path: Path) -> Path:
    """A fresh, unique file name next to `path` (two writers never share one)."""
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".",
                                    suffix=".tmp")
    os.close(handle)
    return Path(name)


def _write_and_flush(temporary: Path, payload: Any,
                     default: Optional[Callable[[Any], Any]]) -> None:
    """Write the JSON and make sure it reached the disk before the swap."""
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=default)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_with_retries(temporary: Path, path: Path) -> None:
    """`os.replace`, retried briefly while another program holds the file."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_PAUSE_SECONDS)


def _remove_quietly(temporary: Path) -> None:
    """Delete a half-written temporary file; a failure here is not news."""
    try:
        temporary.unlink()
    except OSError:
        pass
