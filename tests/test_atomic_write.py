#!/usr/bin/env python3
"""
Audit D4: a write that dies halfway keeps the previous file whole.

For biology researchers: the document JSON used to be written in place, so a
kill mid-write left a cut-off file and the good one from the last run was
already gone. These tests break a write on purpose and check what survives.
"""

from __future__ import annotations

import json

import pytest

from src.processing import document_json
from src.processing.utils import atomic_write
from src.processing.utils.atomic_write import write_json_atomically


def half_a_dump(payload, handle, **kwargs):
    """Stands in for `json.dump` dying partway: some bytes, then an error."""
    handle.write('{"document_info": {"file')
    raise OSError("disk full")


class TestAllOrNothing:
    def test_a_normal_write(self, tmp_path):
        target = tmp_path / "paper.json"
        write_json_atomically(target, {"title": "Ångström"})
        assert json.loads(target.read_text(encoding="utf-8")) == {"title": "Ångström"}

    def test_a_failed_write_keeps_the_old_file_and_leaves_no_temporary(
            self, tmp_path, monkeypatch):
        target = tmp_path / "paper.json"
        write_json_atomically(target, {"run": "previous, good"})
        monkeypatch.setattr(atomic_write.json, "dump", half_a_dump)

        with pytest.raises(OSError, match="disk full"):
            write_json_atomically(target, {"run": "new"})

        assert json.loads(target.read_text(encoding="utf-8")) == {"run": "previous, good"}
        assert [p.name for p in tmp_path.iterdir()] == ["paper.json"]

    def test_a_briefly_locked_file_is_retried(self, tmp_path, monkeypatch):
        """Windows refuses the swap while another program has the file open."""
        target = tmp_path / "paper.json"
        real_replace, calls = atomic_write.os.replace, []

        def locked_twice(source, destination):
            calls.append(1)
            if len(calls) <= 2:
                raise PermissionError("in use")
            real_replace(source, destination)

        monkeypatch.setattr(atomic_write.os, "replace", locked_twice)
        monkeypatch.setattr(atomic_write, "REPLACE_PAUSE_SECONDS", 0)
        write_json_atomically(target, {"ok": True})
        assert len(calls) == 3 and json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


class TestTheDocumentJsonUsesIt:
    def test_a_crash_while_writing_a_document_keeps_the_last_run(
            self, tmp_path, monkeypatch):
        target = tmp_path / "paper.json"
        target.write_text('{"from": "last run"}', encoding="utf-8")
        monkeypatch.setattr(document_json, "build_document_json",
                            lambda profile, elapsed, timings=None: {"new": True})
        monkeypatch.setattr(atomic_write.json, "dump", half_a_dump)

        with pytest.raises(OSError):
            document_json.write_result(profile=None, out_path=target, elapsed=1.0)

        assert json.loads(target.read_text(encoding="utf-8")) == {"from": "last run"}
