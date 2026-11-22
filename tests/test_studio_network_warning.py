#!/usr/bin/env python3
"""
Audit L2: Prajnya Studio warns when it is opened to other computers.

For biology researchers: Studio has no login. On its default address
(127.0.0.1) only this computer can reach it. Started with `--host` on any
other address, anyone on the network could read every processed document and
figure, so it says so loudly at start-up.
"""

from __future__ import annotations

import logging

import pytest

import run_prajnya_studio as launcher


class TestWhichAddressesAreOnlyThisComputer:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost",
                                      "LOCALHOST", "::1", "[::1]"])
    def test_loopback_needs_no_warning(self, host):
        assert launcher._is_loopback(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20",
                                      "10.0.0.5", "my-lab-pc"])
    def test_anything_else_may_be_reached_from_the_network(self, host):
        assert not launcher._is_loopback(host)

    def test_the_default_is_loopback(self):
        assert launcher._is_loopback(launcher.DEFAULT_HOST)


class TestTheWarning:
    def test_it_names_the_address_and_the_risk(self):
        text = launcher.network_warning("0.0.0.0", 8850)
        assert "0.0.0.0:8850" in text and "NO login" in text and "--host" in text

    def _start(self, monkeypatch, host):
        """Run main() with the server and port check stubbed out."""
        import sys
        from src.gui import app as studio_app
        monkeypatch.setattr(sys, "argv", ["run_prajnya_studio.py", "--host", host,
                                          "--no-browser"])
        monkeypatch.setattr(launcher, "_port_is_taken", lambda host, port: False)
        monkeypatch.setattr(studio_app, "main", lambda **kwargs: None)
        assert launcher.main() == 0

    def test_a_network_bind_warns_at_start(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger="prajnya_studio"):
            self._start(monkeypatch, "0.0.0.0")
        assert any("NO login" in r.getMessage() for r in caplog.records)

    def test_the_default_bind_does_not(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger="prajnya_studio"):
            self._start(monkeypatch, "127.0.0.1")
        assert not any("NO login" in r.getMessage() for r in caplog.records)
