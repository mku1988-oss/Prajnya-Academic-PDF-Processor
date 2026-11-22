#!/usr/bin/env python3
"""
The Settings "Ledger File" must reach every place that opens the ledger.

Found 2026-11-04 by driving the GUI: with Ledger File pointed at an empty
scratch file, a batch still skipped a PDF as "already in the ledger
(reviewed)" -- it had read the default ledger. The path travelled on the
RunRequest and was read by nothing: not the controller's skip check and
record write, not the pipeline in the worker child (which reads
PRAJNYA_LEDGER_PATH), not the Ledger or Review Queue pages.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.gui import controller as controller_module
from src.gui import persistent_settings
from src.gui.controller import BatchController, BatchState
from src.gui.session import RunRegistry
from src.processing.run_request import RunRequest
from src.processing.worker import child_environment


@pytest.fixture(autouse=True)
def clean_registry():
    RunRegistry.get_instance().finish()
    yield
    RunRegistry.get_instance().finish()


class TestTheWorkerChild:
    def test_a_chosen_ledger_is_handed_to_the_child(self, tmp_path):
        request = RunRequest(paths=[], out_root=tmp_path, ledger_path=tmp_path / "scratch.csv")
        env = child_environment(request)
        assert env["PRAJNYA_LEDGER_PATH"] == str((tmp_path / "scratch.csv").resolve())

    def test_no_choice_inherits_the_environment_unchanged(self, tmp_path):
        assert child_environment(RunRequest(paths=[], out_root=tmp_path)) is None


class TestTheController:
    def test_the_batch_opens_the_chosen_ledger(self, tmp_path, monkeypatch):
        opened = []
        monkeypatch.setattr(controller_module, "open_ledger",
                            lambda log, ledger_path=None, **kw: opened.append(ledger_path))
        request = RunRequest(paths=[tmp_path / "missing.pdf"], out_root=tmp_path,
                             ledger_path=tmp_path / "scratch.csv")
        batch = BatchController(request, noninteractive=True)
        batch.start()
        deadline = time.time() + 10
        while batch.state not in (BatchState.DONE, BatchState.FAILED) and time.time() < deadline:
            time.sleep(0.02)
        assert opened == [tmp_path / "scratch.csv"]


class TestTheSettingsHelper:
    def _use_settings_file(self, tmp_path, monkeypatch, ledger_path: str):
        path = tmp_path / "settings.json"
        monkeypatch.setattr(persistent_settings.SettingsManager, "settings_path",
                            staticmethod(lambda: path))
        settings = persistent_settings.SettingsManager.DEFAULT_SETTINGS.copy()
        settings["ledger_path"] = ledger_path
        persistent_settings.SettingsManager.save(settings)

    def test_a_set_ledger_file_is_returned(self, tmp_path, monkeypatch):
        self._use_settings_file(tmp_path, monkeypatch, str(tmp_path / "scratch.csv"))
        assert persistent_settings.SettingsManager.ledger_path() == tmp_path / "scratch.csv"

    def test_a_blank_ledger_file_means_the_default(self, tmp_path, monkeypatch):
        self._use_settings_file(tmp_path, monkeypatch, "  ")
        assert persistent_settings.SettingsManager.ledger_path() is None
