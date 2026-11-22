#!/usr/bin/env python3
"""
RunRegistry — the rule that only one batch can run at a time.

For biology researchers: this is the safety catch that stops the GUI from
starting two document batches on the same 8GB GPU at once. Concurrency was
tried four ways in this project and every one was slower, not faster (see
PERFORMANCE.md) — so "one run at a time" is a hard requirement, not a nicety,
and these tests exist to make sure a second `start()` call is refused rather
than silently replacing the first run's controller.
"""

from __future__ import annotations

import threading

import pytest

from src.gui.session import RunAlreadyActive, RunRegistry


@pytest.fixture
def registry() -> RunRegistry:
    """A fresh registry, not the process-wide singleton — tests must not
    share state with each other or with a real running app."""
    return RunRegistry()


def test_starts_with_no_active_run(registry: RunRegistry) -> None:
    assert registry.is_active() is False
    assert registry.active_controller is None


def test_start_registers_the_controller(registry: RunRegistry) -> None:
    controller = object()
    registry.start(controller)

    assert registry.is_active() is True
    assert registry.active_controller is controller


def test_second_start_raises_without_replacing_the_first(registry: RunRegistry) -> None:
    first = object()
    second = object()
    registry.start(first)

    with pytest.raises(RunAlreadyActive):
        registry.start(second)

    assert registry.active_controller is first


def test_finish_clears_the_active_run(registry: RunRegistry) -> None:
    registry.start(object())
    registry.finish()

    assert registry.is_active() is False
    assert registry.active_controller is None


def test_finish_is_safe_when_nothing_is_active(registry: RunRegistry) -> None:
    registry.finish()  # must not raise
    assert registry.is_active() is False


def test_start_after_finish_succeeds(registry: RunRegistry) -> None:
    registry.start(object())
    registry.finish()

    second = object()
    registry.start(second)

    assert registry.active_controller is second


def test_get_instance_returns_the_same_object() -> None:
    first = RunRegistry.get_instance()
    second = RunRegistry.get_instance()
    assert first is second


def test_concurrent_start_calls_let_exactly_one_through(registry: RunRegistry) -> None:
    """Ten threads race to start a run; the lock must let exactly one win.

    This is the property the whole module exists for: two GUI tabs clicking
    Start at the same instant must not both succeed.
    """
    successes = []
    failures = []
    lock = threading.Lock()

    def attempt(n: int) -> None:
        try:
            registry.start(n)
            with lock:
                successes.append(n)
        except RunAlreadyActive:
            with lock:
                failures.append(n)

    threads = [threading.Thread(target=attempt, args=(n,)) for n in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(successes) == 1
    assert len(failures) == 9
    assert registry.active_controller == successes[0]
