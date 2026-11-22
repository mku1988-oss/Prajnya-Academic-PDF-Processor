#!/usr/bin/env python3
"""
BatchController — the seam the Library/Batch page drives.

For biology researchers: these tests exercise the controller without a
GPU. A batch of nonexistent PDF paths fails pre-flight instantly (no file
to open), which is enough to walk the WHOLE lifecycle — preflight, no book
questions, the processing loop's error path, batch_done — the same way
`tests/test_events.py` uses a nonexistent PDF to test the real subprocess
path without needing a model. Only the book-question and stop mechanics
are tested against the controller's internal seams directly, since driving
those through a real subprocess would need a GPU.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from src.gui.controller import BatchController, BatchState, BookQuestion
from src.gui.session import RunAlreadyActive, RunRegistry
from src.processing.batch import BookDecision, PreflightEntry
from src.processing.run_request import DocumentOutcome, RunRequest


@pytest.fixture(autouse=True)
def clean_registry():
    """The process-wide RunRegistry must not leak an active run between tests."""
    RunRegistry.get_instance().finish()
    yield
    RunRegistry.get_instance().finish()


def _request(tmp_path: Path, n_missing: int = 1) -> RunRequest:
    paths = [tmp_path / f"missing_{i}.pdf" for i in range(n_missing)]
    return RunRequest(paths=paths, out_root=tmp_path)


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for condition")


class TestTheFullLifecycleWithNoGpu:
    """A batch of PDFs that do not exist fails pre-flight instantly, which
    is enough to drive the controller from start to a finished state."""

    def test_a_batch_of_missing_pdfs_finishes_without_error(self, tmp_path):
        controller = BatchController(_request(tmp_path, n_missing=2),
                                     noninteractive=True)
        controller.start()

        _wait_for(lambda: controller.state in
                  (BatchState.DONE, BatchState.STOPPED, BatchState.FAILED))

        assert controller.state == BatchState.DONE
        assert controller.error is None
        assert len(controller.entries) == 2
        assert all(entry.error is not None for entry in controller.entries)
        assert len(controller.results) == 2
        assert all("error" in value for value in controller.results.values())

    def test_starting_registers_with_the_run_registry(self, tmp_path):
        controller = BatchController(_request(tmp_path), noninteractive=True)
        assert RunRegistry.get_instance().is_active() is False

        controller.start()
        assert RunRegistry.get_instance().active_controller is controller

        _wait_for(lambda: controller.state in
                  (BatchState.DONE, BatchState.STOPPED, BatchState.FAILED))
        _wait_for(lambda: not RunRegistry.get_instance().is_active())

    def test_a_second_start_while_one_is_active_raises(self, tmp_path):
        first = BatchController(_request(tmp_path), noninteractive=True)
        RunRegistry.get_instance().start(first)

        second = BatchController(_request(tmp_path), noninteractive=True)
        with pytest.raises(RunAlreadyActive):
            second.start()

    def test_events_accumulate_and_the_cursor_advances(self, tmp_path):
        controller = BatchController(_request(tmp_path), noninteractive=True)
        controller.start()

        _wait_for(lambda: controller.state in
                  (BatchState.DONE, BatchState.STOPPED, BatchState.FAILED))

        events, cursor = controller.events_since(0)
        assert events, "a run that did anything must leave a transcript"
        assert cursor == len(events)

        more_events, cursor2 = controller.events_since(cursor)
        assert more_events == []
        assert cursor2 == cursor


class TestTheBookQuestionBlocksTheThreadUntilAnswered:
    """Tested against `_threaded_prompt` directly: driving it through a real
    book-like PDF would need a real classified document, which is not
    available without a GPU-processed sample. The blocking contract is the
    thing that matters here and does not depend on classification."""

    def test_the_prompt_blocks_until_answer_book_decision_is_called(self, tmp_path):
        controller = BatchController(_request(tmp_path), noninteractive=False)
        entry = PreflightEntry(path=tmp_path / "book.pdf")
        result = {}

        def call_prompt():
            result["decision"] = controller._threaded_prompt(entry)

        thread = threading.Thread(target=call_prompt)
        thread.start()

        _wait_for(lambda: controller.snapshot()["question"] is not None)
        question = controller.snapshot()["question"]
        assert isinstance(question, BookQuestion)
        assert question.entry is entry

        expected = BookDecision(chapters=[3], reason="test")
        controller.answer_book_decision(expected)
        thread.join(timeout=5)

        assert result["decision"] is expected
        assert controller.snapshot()["question"] is None

    def test_answering_with_nothing_pending_is_a_safe_no_op(self, tmp_path):
        controller = BatchController(_request(tmp_path))
        controller.answer_book_decision(BookDecision.safe_default())  # must not raise
        assert controller.snapshot()["question"] is None

    def test_no_decision_falls_back_to_the_safe_default(self, tmp_path):
        """A dialog closed with no real answer (e.g. `None` from the page)
        must never fall through to reading every chapter."""
        controller = BatchController(_request(tmp_path), noninteractive=False)
        entry = PreflightEntry(path=tmp_path / "book.pdf")
        result = {}

        def call_prompt():
            result["decision"] = controller._threaded_prompt(entry)

        thread = threading.Thread(target=call_prompt)
        thread.start()
        _wait_for(lambda: controller.snapshot()["question"] is not None)

        controller.answer_book_decision(None)
        thread.join(timeout=5)

        assert result["decision"].chapters is None


class TestStopping:
    """`stop()` is tested against the controller's own seams — driving a
    real kill through a live subprocess would need something slow enough
    to still be running when Stop is clicked, which needs a GPU."""

    def test_stop_before_a_document_starts_prevents_it_from_running(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr("src.gui.controller.run_in_subprocess",
                            lambda *a, **k: called.append(1))

        controller = BatchController(_request(tmp_path))
        controller._stop_requested = True

        entry = PreflightEntry(path=tmp_path / "paper.pdf")
        outcome = controller._execute(entry, controller.request)

        assert called == [], "a stopped controller must not spawn a child"
        assert outcome.error == "Stopped by user"

    def test_stop_relabels_a_failure_that_happens_after_the_stop_request(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.gui.controller.run_in_subprocess",
            lambda entry, request, **k: DocumentOutcome(
                path=entry.path, error="WorkerFailed: killed"))

        controller = BatchController(_request(tmp_path))
        entry = PreflightEntry(path=tmp_path / "paper.pdf")

        controller._stop_requested = True
        outcome = controller._execute(entry, controller.request)

        assert outcome.error == "Stopped by user"

    def test_stop_terminates_the_currently_registered_process(self, tmp_path):
        terminated = []
        controller = BatchController(_request(tmp_path))

        class FakeProcess:
            pass

        fake = FakeProcess()
        controller._register_process(fake)

        import src.gui.controller as controller_module
        original_terminate = controller_module.terminate
        controller_module.terminate = lambda proc: terminated.append(proc)
        try:
            controller.stop()
        finally:
            controller_module.terminate = original_terminate

        assert terminated == [fake]
        assert controller._stop_requested is True


class TestStopWhileABookQuestionIsOpen:
    """Audit D5: Stop used to do nothing while the batch waited on a book
    question (there is no child to kill then), so the run hung until the
    researcher answered. Stop must close the question and end the batch."""

    @pytest.fixture
    def two_books(self, tmp_path, monkeypatch):
        """A batch whose pre-flight finds two book-like documents."""
        entries = [PreflightEntry(path=tmp_path / f"book_{i}.pdf") for i in (1, 2)]
        asked, spawned = [], []

        def fake_resolve(entries, request, prompt=None, logger=None):
            for entry in entries:
                asked.append(entry.path.name)
                prompt(entry)

        monkeypatch.setattr("src.gui.controller.run_preflight",
                            lambda request, logger=None: entries)
        monkeypatch.setattr("src.gui.controller.resolve_decisions", fake_resolve)
        monkeypatch.setattr("src.gui.controller.run_in_subprocess",
                            lambda *a, **k: spawned.append(1))
        return asked, spawned

    def test_stop_ends_the_batch_without_an_answer(self, tmp_path, two_books):
        asked, spawned = two_books
        controller = BatchController(_request(tmp_path))
        controller.start()
        _wait_for(lambda: controller.snapshot()["question"] is not None)

        controller.stop()
        _wait_for(lambda: controller.state == BatchState.STOPPED, timeout=5)

        assert controller.snapshot()["question"] is None
        assert spawned == [], "a stopped batch must not process anything"
        # The second book is passed over at once: nobody is there to answer.
        assert asked == ["book_1.pdf", "book_2.pdf"]

    def test_a_stop_just_before_the_question_opens_does_not_hang(self, tmp_path):
        controller = BatchController(_request(tmp_path))
        controller.stop()
        result = {}
        thread = threading.Thread(target=lambda: result.update(
            decision=controller._threaded_prompt(PreflightEntry(path=tmp_path / "b.pdf"))))
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert result["decision"].chapters is None   # the safe default, nothing read
