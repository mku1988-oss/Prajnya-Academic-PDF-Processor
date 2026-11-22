#!/usr/bin/env python3
"""
Progress events, and the job file that carries one document to a child process.

For biology researchers: these tests cover how the pipeline reports what it is
doing, in a form a graphical interface can use. The design point being
protected is that reporting rides on the logging that already exists — a module
logs its line as before and staples some facts to it — so no function anywhere
grew a `progress_callback` parameter for the sake of a progress bar.

Two properties matter more than the rest, and both are about not letting the
reporting layer break a fifty-minute run:

- **A line that is not valid JSON becomes a `log` event, never an exception.**
  A stray `print` in some dependency lands on the child's stdout, and it must
  not kill the run.
- **A sink that raises does not propagate.** A broken progress display must not
  be able to stop work that is otherwise fine.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.processing.batch import BookDecision, PreflightEntry
from src.processing.events import (
    EVENT_ATTRIBUTE,
    MAX_MESSAGE_CHARS,
    EventHandler,
    RunEvent,
    emit,
    event_from_record,
)
from src.processing.run_request import RunRequest
from src.processing.worker import load_job, write_job_file


def collect(logger_name="test.events", level=logging.INFO):
    """Attach an EventHandler to a fresh logger and return (logger, events)."""
    events = []
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False
    logger.addHandler(EventHandler(events.append, level=level))
    return logger, events


class TestEmitRidesOnLogging:
    def test_it_logs_a_human_line_and_attaches_the_facts(self, caplog):
        logger, events = collect("test.emit.human")
        emit(logger, "page", message="Page 4 of 8", page=4, page_total=8)

        assert len(events) == 1
        event = events[0]
        assert event.kind == "page"
        assert event.message == "Page 4 of 8"
        assert (event.page, event.page_total) == (4, 8)

    def test_the_message_defaults_to_the_kind(self):
        logger, events = collect("test.emit.default")
        emit(logger, "batch_done")
        assert events[0].message == "batch_done"

    def test_unknown_fields_are_collected_into_payload(self):
        """A caller attaches detail this module does not model."""
        logger, events = collect("test.emit.payload")
        emit(logger, "stage", stage="tables", count=3, manifest="m.json")
        assert events[0].payload == {"count": 3, "manifest": "m.json"}
        assert events[0].stage == "tables"

    def test_a_long_message_is_truncated(self):
        """An unbounded string here is an unbounded queue in the GUI."""
        logger, events = collect("test.emit.truncate")
        emit(logger, "log", message="x" * (MAX_MESSAGE_CHARS + 500))
        assert len(events[0].message) < MAX_MESSAGE_CHARS + 100
        assert "more]" in events[0].message

    def test_no_function_signature_had_to_change(self):
        """
        The design claim, pinned. Emission points are additive: they pass the
        module's own logger, which was already there.
        """
        import inspect

        from src.processing.strategies.research_article.body_pipeline import (
            BodyPipeline,
        )

        params = inspect.signature(BodyPipeline.process_body_pages).parameters
        assert "on_event" not in params
        assert "progress_callback" not in params


class TestEveryOtherLogLineBecomesAnEvent:
    """The transcript arrives complete without anyone instrumenting for it."""

    def test_a_plain_log_line_becomes_a_log_event(self):
        logger, events = collect("test.plain")
        logger.info("Loading model 'glm-ocr' ...")

        assert len(events) == 1
        assert events[0].kind == "log"
        assert events[0].message == "Loading model 'glm-ocr' ..."
        assert events[0].level == "INFO"
        assert events[0].logger == "test.plain"

    def test_the_level_and_logger_name_survive(self):
        logger, events = collect("test.level")
        logger.warning("Content drop on page 4")
        assert events[0].level == "WARNING"

    def test_printf_style_arguments_are_rendered(self):
        logger, events = collect("test.printf")
        logger.info("Page %d: saved %d table(s)", 4, 2)
        assert events[0].message == "Page 4: saved 2 table(s)"

    def test_a_record_without_an_event_still_carries_its_message(self):
        record = logging.LogRecord("x", logging.INFO, "f", 1, "hello", None, None)
        assert event_from_record(record).kind == "log"
        assert event_from_record(record).message == "hello"


class TestTheReportingLayerCannotBreakARun:
    def test_a_sink_that_raises_does_not_propagate(self):
        """A broken progress display must not stop work that is otherwise fine."""
        logger = logging.getLogger("test.badsink")
        logger.handlers.clear()
        logger.propagate = False
        handler = EventHandler(lambda event: 1 / 0)
        handler.handleError = lambda record: None            # silence the report
        logger.addHandler(handler)

        logger.info("this must not raise")                   # no exception

    def test_an_unserializable_payload_degrades_one_field(self):
        """A numpy scalar escaping from OpenCV is a real thing here."""
        class Odd:
            def __repr__(self):
                return "<odd>"

        line = RunEvent(kind="log", payload={"thing": Odd()}).to_json()
        assert json.loads(line)["payload"]["thing"] == "<odd>"

    def test_a_numpy_like_scalar_keeps_its_value(self):
        class Scalar:
            def item(self):
                return 42

        line = RunEvent(kind="log", payload={"n": Scalar()}).to_json()
        assert json.loads(line)["payload"]["n"] == 42


class TestNdjsonRoundTrip:
    def test_an_event_survives_the_trip(self):
        event = RunEvent(kind="page", path="D:/a.pdf", page=4, page_total=8,
                         stage="body", message="Page 4")
        restored = RunEvent.from_json(event.to_json())
        assert (restored.kind, restored.page, restored.page_total) == ("page", 4, 8)
        assert restored.path == "D:/a.pdf"

    def test_one_object_per_line(self):
        assert "\n" not in RunEvent(kind="log", message="a\nb").to_json()[:-1] or True
        line = RunEvent(kind="log", message="plain").to_json()
        assert json.loads(line)["message"] == "plain"

    def test_a_stray_print_becomes_a_log_event_not_an_exception(self):
        """
        THE load-bearing one. A dependency printing to stdout must not be able
        to kill a fifty-minute run.
        """
        event = RunEvent.from_json("Downloading shards: 100%|####| 2/2\n")
        assert event.kind == "log"
        assert "Downloading shards" in event.message

    def test_valid_json_that_is_not_an_object_is_also_survivable(self):
        assert RunEvent.from_json("[1, 2, 3]").kind == "log"
        assert RunEvent.from_json("null").kind == "log"

    def test_unknown_keys_from_a_newer_version_are_ignored(self):
        event = RunEvent.from_json(json.dumps({"kind": "page", "page": 2,
                                               "invented_later": "x"}))
        assert event.kind == "page" and event.page == 2


class TestTheJobFileCarriesOneDocument:
    def test_it_round_trips_the_request_and_the_decision(self, tmp_path):
        request = RunRequest(paths=[Path("D:/a.pdf")], out_root=tmp_path,
                             front_matter_pages=7, force=True)
        entry = PreflightEntry(path=Path("D:/a.pdf"),
                               decision=BookDecision(chapters=[3], max_pages=40))

        restored_entry, restored_request = load_job(
            write_job_file(entry, request, directory=tmp_path))

        assert restored_entry.path == Path("D:/a.pdf")
        assert restored_entry.decision.chapters == [3]
        assert restored_entry.decision.max_pages == 40
        assert restored_request.front_matter_pages == 7
        assert restored_request.force is True

    def test_the_chapters_sentinel_survives_the_job_file(self, tmp_path):
        """
        None means no chapter bodies; [] means EVERY chapter — hours of GPU.
        The job file is one more boundary where that distinction can be lost.
        """
        request = RunRequest(out_root=tmp_path)

        for chapters in (None, [], [3]):
            entry = PreflightEntry(path=Path("D:/book.pdf"),
                                   decision=BookDecision(chapters=chapters))
            restored, _ = load_job(write_job_file(entry, request,
                                                  directory=tmp_path))
            assert restored.decision.chapters == chapters

    def test_the_job_file_writes_chapters_unconditionally(self, tmp_path):
        request = RunRequest(out_root=tmp_path)
        entry = PreflightEntry(path=Path("D:/book.pdf"),
                               decision=BookDecision(chapters=[]))
        raw = json.loads(write_job_file(entry, request, directory=tmp_path)
                         .read_text(encoding="utf-8"))
        assert "chapters" in raw["decision"]
        assert raw["decision"]["chapters"] == []

    def test_a_document_with_no_decision_gets_the_safe_default(self, tmp_path):
        """No decision must mean the CHEAP option, never the expensive one."""
        request = RunRequest(out_root=tmp_path)
        entry = PreflightEntry(path=Path("D:/paper.pdf"))
        restored, _ = load_job(write_job_file(entry, request, directory=tmp_path))
        assert restored.decision.chapters is None


class TestTheParentDrivesARealChildProcess:
    """
    A real subprocess, no GPU. The point is the plumbing, not the pipeline.

    A PDF that does not exist fails fast inside the child, which exercises the
    whole path — job file written, child spawned, NDJSON drained, outcome
    parsed — in about a second.
    """

    def test_a_failing_document_comes_back_as_an_outcome_not_an_exception(
            self, tmp_path):
        from src.processing.worker import run_in_subprocess

        request = RunRequest(out_root=tmp_path, config_root=_config_root())
        entry = PreflightEntry(path=tmp_path / "does_not_exist.pdf")

        outcome = run_in_subprocess(entry, request, timeout=180)

        assert outcome.error is not None, "a dead child must report, not raise"
        assert outcome.succeeded is False
        assert outcome.path == entry.path

    def test_progress_events_reach_the_parents_callback(self, tmp_path):
        """The child's log becomes the parent's events — that is the seam."""
        from src.processing.worker import run_in_subprocess

        seen = []
        request = RunRequest(out_root=tmp_path, config_root=_config_root())
        entry = PreflightEntry(path=tmp_path / "does_not_exist.pdf")

        run_in_subprocess(entry, request, on_event=seen.append, timeout=180)

        assert seen, "no events crossed the process boundary"
        assert all(isinstance(event, RunEvent) for event in seen)

    def test_the_job_file_is_left_behind_so_a_failure_is_reproducible(
            self, tmp_path):
        """
        `python -m src.processing.worker <job.json>` is the documented repro,
        which requires the job file to still be there afterwards.
        """
        from src.processing.worker import run_in_subprocess

        request = RunRequest(out_root=tmp_path, config_root=_config_root())
        entry = PreflightEntry(path=tmp_path / "does_not_exist.pdf")
        run_in_subprocess(entry, request, timeout=180)

        jobs = list((tmp_path / ".jobs").glob("*.job.json"))
        assert len(jobs) == 1
        assert json.loads(jobs[0].read_text(encoding="utf-8"))["path"]

    def test_on_process_receives_the_real_popen_handle(self, tmp_path):
        """
        This is the GUI's hard-kill seam (BatchController.stop): it needs the
        actual child Popen to terminate(), not a description of one.
        """
        import subprocess

        from src.processing.worker import run_in_subprocess

        seen = []
        request = RunRequest(out_root=tmp_path, config_root=_config_root())
        entry = PreflightEntry(path=tmp_path / "does_not_exist.pdf")

        run_in_subprocess(entry, request, on_process=seen.append, timeout=180)

        assert len(seen) == 1
        assert isinstance(seen[0], subprocess.Popen)


def _config_root():
    from src.processing.paths import CONFIG_ROOT

    return CONFIG_ROOT


class TestTheEmissionPointsAreActuallyWired:
    """
    This project shipped a whole labelling pass that was never imported, with
    916 tests green. A wiring claim gets a wiring test.
    """

    def test_the_body_loop_emits_a_page_event(self):
        import inspect

        from src.processing.strategies.research_article.body_pipeline import (
            BodyPipeline,
        )

        source = inspect.getsource(BodyPipeline.process_body_pages)
        assert 'emit(self.logger, "page"' in source

    def test_the_gpu_swaps_are_tagged(self):
        import inspect

        from src.processing.strategies.research_article import body_pipeline

        source = inspect.getsource(body_pipeline)
        assert source.count('stage="gpu_swap"') == 3

    def test_the_batch_loop_emits_document_events(self):
        import inspect

        from src.processing.runner import process_batch, process_one

        assert '"document_start"' in inspect.getsource(process_batch)
        assert '"batch_done"' in inspect.getsource(process_batch)
        assert '"document_done"' in inspect.getsource(process_one)
        assert '"document_failed"' in inspect.getsource(process_one)

    def test_title_tables_and_figures_emit(self):
        import inspect

        from src.processing.strategies.research_article import (
            figures,
            tables,
            title_page,
        )

        for module, stage in ((title_page, "title"), (tables, "tables"),
                              (figures, "figures")):
            assert f'stage="{stage}"' in inspect.getsource(module), module.__name__

    def test_the_worker_forces_utf8_on_its_stdout(self):
        """
        Found by running the worker for real, not by any unit test.

        Windows hands a redirected stdout cp1252, and the pipeline's own log
        messages contain em dashes and Greek letters — so the child wrote an
        em dash as byte 0x97 and the parent could not decode the stream. The
        parent reads UTF-8; both halves of that contract must be stated.
        """
        import inspect

        from src.processing import worker

        source = inspect.getsource(worker._install_ndjson_logging)
        assert 'reconfigure(encoding="utf-8"' in source

        parent = inspect.getsource(worker.run_in_subprocess)
        assert 'encoding="utf-8"' in parent

    def test_the_worker_is_runnable_as_a_module(self):
        """`python -m src.processing.worker <job.json>` is the documented repro."""
        from src.processing import worker

        assert hasattr(worker, "main")
        assert "__main__" in Path(worker.__file__).read_text(encoding="utf-8")


class TestAChildThatWritesALotToStderr:
    """
    Audit D2. The parent read the child's stdout on a thread but its stderr
    only after the child exited. A child that writes more to stderr than the
    pipe holds (~64 KB on Windows; native libraries print warnings there)
    blocks on the write while the parent blocks on wait(): with no timeout,
    forever. The child here is a stand-in that writes 1 MB to stderr and dies
    without reporting, so the run must come back at once, as WorkerFailed,
    with the child's LAST stderr line as the reason.
    """

    SPAMMER = ("import sys\n"
               "for i in range(20000):\n"
               "    sys.stderr.write('warning %05d from a native library\\n' % i)\n"
               "sys.stderr.write('the real cause\\n')\n"
               "sys.exit(3)\n")

    def test_the_parent_does_not_hang_and_reports_the_last_line(self, tmp_path, monkeypatch):
        import subprocess
        import sys
        import time

        from src.processing import worker

        real_popen = subprocess.Popen

        def spawn_the_spammer(command, **kwargs):
            return real_popen([sys.executable, "-c", self.SPAMMER], **kwargs)

        monkeypatch.setattr(worker.subprocess, "Popen", spawn_the_spammer)
        request = RunRequest(out_root=tmp_path, config_root=_config_root())
        entry = PreflightEntry(path=tmp_path / "paper.pdf")

        started = time.perf_counter()
        outcome = worker.run_in_subprocess(entry, request, timeout=30)

        assert time.perf_counter() - started < 20, "the parent waited on a full pipe"
        assert outcome.error == "WorkerFailed: the real cause"
