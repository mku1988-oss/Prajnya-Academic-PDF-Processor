"""
RunRegistry — the process-wide seam that keeps two GUI tabs from starting
two batch runs at once.

For biology researchers: there is one 8GB GPU on this machine. Running two
documents through the VLM at the same time was tried four different ways
(see PERFORMANCE.md) and every one of them was slower than running them one
after another, because the GPU still only does one thing at a time — the
work doesn't disappear, it just gets interleaved with itself. So the GUI
must never let a second batch start while one is already running, even if
the user opens the app in two browser tabs.

This module holds no actual run logic — the object it registers
(`BatchController`, wrapping one subprocess-per-PDF run) is built in S6.
For S5 it only needs to exist and enforce the one-at-a-time rule, so pages
built later have a single, already-tested place to ask "is anything
running right now?" and "let me be the one that runs."
"""

from __future__ import annotations

import threading
from typing import Any, Optional


class RunAlreadyActive(Exception):
    """Raised by RunRegistry.start() when a batch run is already active."""


class RunRegistry:
    """
    Process-wide singleton holding at most one active run controller.

    Not a queue: a second `start()` call while a run is active raises
    `RunAlreadyActive` rather than waiting its turn. A GUI page decides for
    itself what to do with that — offer to view the running batch, refuse,
    etc. — the registry's only job is to make "is one already running" an
    atomic, thread-safe question.
    """

    _instance: Optional["RunRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._controller: Optional[Any] = None

    @classmethod
    def get_instance(cls) -> "RunRegistry":
        """Returns the single process-wide registry, creating it on first use."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def active_controller(self) -> Optional[Any]:
        """The currently active controller, or None if no run is active."""
        with self._lock:
            return self._controller

    def is_active(self) -> bool:
        """True if a batch run is currently registered."""
        with self._lock:
            return self._controller is not None

    def start(self, controller: Any) -> None:
        """
        Registers `controller` as the single active run.

        Args:
            controller: the run object to register (a `BatchController`,
                from S6 onward). This module treats it as opaque.

        Raises:
            RunAlreadyActive: if a controller is already registered. The
                caller must call `finish()` (or wait for it to be called)
                before trying again — this method never replaces an
                existing controller silently.
        """
        with self._lock:
            if self._controller is not None:
                raise RunAlreadyActive("A batch run is already active.")
            self._controller = controller

    def finish(self) -> None:
        """Clears the active controller. Safe to call when none is active."""
        with self._lock:
            self._controller = None
