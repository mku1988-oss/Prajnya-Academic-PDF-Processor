#!/usr/bin/env python3
"""
Audit 2.4 (6.9.0 assessment, P7): what Prajnya Studio exposes, pinned.

For biology researchers: Studio has no login. Two promises keep that safe on
a lab computer, and these tests fail if either is ever broken:
  1. by default it listens on 127.0.0.1 only (this computer, not the network);
  2. the only files a browser can fetch from it are Studio's own icons and
     the processed output folder (figure and table crops). A path that tries
     to climb out of those folders ("../.env", where API keys live) is refused.
"""

from __future__ import annotations

import sys

import pytest

import run_prajnya_studio as launcher
from src.gui import app as studio_app
from src.gui import theme
from src.processing.paths import DEFAULT_OUTPUT_ROOT


def file_routes():
    """{URL prefix: the folder it serves} for every file-serving route of Studio."""
    from nicegui import app
    routes = {}
    for route in app.routes:
        path = getattr(route, "path", "")
        handler = _handler_of(route)
        if handler is not None and path.endswith("/{path:path}"):
            routes[path[:-len("/{path:path}")]] = handler
    return routes


class TestLoopbackByDefault:
    def test_the_launcher_binds_studio_to_loopback_without_host(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(sys, "argv", ["run_prajnya_studio.py", "--no-browser"])
        monkeypatch.setattr(launcher, "_port_is_taken", lambda host, port: False)
        monkeypatch.setattr(studio_app, "main", lambda **kwargs: seen.update(kwargs))
        assert launcher.main() == 0
        assert seen["host"] == "127.0.0.1"

    def test_the_app_itself_defaults_to_loopback(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(studio_app.ui, "run", lambda **kwargs: seen.update(kwargs))
        studio_app.main(show=False)
        assert seen["host"] == "127.0.0.1"


class TestFilesServed:
    def test_only_the_icons_and_the_output_folder_are_served(self):
        routes = file_routes()
        assert set(routes) == {"/static", "/run_outputs"}
        assert routes["/static"].directory == str(theme.ASSETS_DIR)
        assert routes["/run_outputs"].directory == str(DEFAULT_OUTPUT_ROOT)

    @pytest.mark.parametrize("attempt", ["../.env", "..\.env", "documents/../../.env",
                                         "../run_prajnya.py", "../config/model.ini"])
    def test_a_path_out_of_the_output_folder_is_refused(self, attempt):
        full_path, _ = file_routes()["/run_outputs"].lookup_path(attempt)
        assert full_path == ""

    def test_a_file_inside_the_output_folder_is_found(self, tmp_path):
        probe = DEFAULT_OUTPUT_ROOT / "_studio_exposure_probe.txt"
        probe.write_text("ok", encoding="utf-8")
        try:
            full_path, _ = file_routes()["/run_outputs"].lookup_path(probe.name)
            assert full_path.endswith(probe.name)
        finally:
            probe.unlink()


# --- helpers (after the tests, per CLAUDE.md) --------------------------------

def _handler_of(route):
    """The static-files handler nicegui's `add_static_files` closes over, or None."""
    endpoint = getattr(route, "endpoint", None)
    code = getattr(endpoint, "__code__", None)
    if code is None or not endpoint.__closure__:
        return None
    cells = dict(zip(code.co_freevars, (cell.cell_contents for cell in endpoint.__closure__)))
    handler = cells.get("handler")
    return handler if hasattr(handler, "lookup_path") else None
