#!/usr/bin/env python3
"""Tests for run_prajnya_studio.py — the GUI's top-level entrypoint.

Until this file's subject existed, the GUI had no launcher: `python
src/gui/app.py` fails outright (app.py uses relative imports) and
`python -m src.gui.app` only works from the repository root, which is why
every scratch launcher from S5 onward had to work around it.

These tests cover the two things a live launch cannot conveniently prove
again later: that the default port does not collide with the pipeline's own
llama-server, and that the port-in-use guard actually detects a listener.
"""

from __future__ import annotations

import re
import socket
import sys
import threading

import pytest

import run_prajnya_studio as launcher
from src.gui import app as studio_app


class TestDefaultPort:
    """9931 is llama-server's (8080 before v6.4.0). Choosing it would have made the GUI unable to
    start on exactly the machine configured correctly to run the pipeline —
    which is what happened live while this entrypoint was being built."""

    #: The port CLAUDE.md documents for the local llama-server router, and
    #: the one the offline-verification grep looks for (`localhost:9931`).
    LLAMA_SERVER_PORT = 9931

    def test_launcher_default_is_not_the_llama_server_port(self):
        assert launcher.DEFAULT_PORT != self.LLAMA_SERVER_PORT

    def test_app_main_default_is_not_the_llama_server_port(self):
        """`python -m src.gui.app` takes app.main()'s own defaults and gets
        no port check, so its default must be safe on its own."""
        import inspect
        default = inspect.signature(studio_app.main).parameters["port"].default
        assert default != self.LLAMA_SERVER_PORT

    def test_both_entrypoints_agree_on_the_port(self):
        """Two defaults that drift apart mean the URL printed by one path is
        wrong for the other."""
        import inspect
        assert (inspect.signature(studio_app.main).parameters["port"].default
                == launcher.DEFAULT_PORT)

    def test_default_host_is_loopback(self):
        """This app opens any folder path typed into it, so binding beyond
        loopback must be an explicit --host, never the default."""
        assert launcher.DEFAULT_HOST == "127.0.0.1"


class TestPortIsTaken:
    """The guard exists because a busy port does NOT always fail loudly:
    session 15 lost time to SLM's dashboard holding 8765 while nicegui
    printed its usual 'ready' line and every route 404'd."""

    def test_detects_a_real_listener(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))          # 0 = let the OS pick a free one
        server.listen(1)
        port = server.getsockname()[1]

        # Accept in the background so the probe's connect() completes.
        stop = threading.Event()

        def _accept():
            server.settimeout(0.2)
            while not stop.is_set():
                try:
                    server.accept()[0].close()
                except OSError:
                    pass

        thread = threading.Thread(target=_accept, daemon=True)
        thread.start()
        try:
            assert launcher._port_is_taken("127.0.0.1", port) is True
        finally:
            stop.set()
            thread.join(timeout=2)
            server.close()

    def test_free_port_reads_as_free(self):
        # Bind then immediately close: the OS just told us this port was
        # free, and nothing has taken it since.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        assert launcher._port_is_taken("127.0.0.1", port) is False

    @pytest.mark.parametrize("wildcard", ["0.0.0.0", "::"])
    def test_wildcard_host_is_probed_via_loopback(self, wildcard):
        """You cannot connect TO 0.0.0.0; the guard must probe an interface
        it actually covers rather than raising."""
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        assert launcher._port_is_taken(wildcard, port) is False


class TestLauncherIsOnlyAFrontEnd:
    """run_prajnya.py's docstring states the rule for the CLI; this is the
    GUI's half of it. The launcher must not grow pipeline logic — the work
    lives in src/, so both front ends call the same code."""

    def test_launcher_does_not_import_the_pipeline_at_module_level(self):
        source = (launcher.__file__)
        with open(source, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        # Module-level imports are the unindented `import`/`from` lines.
        top_level = [ln for ln in lines
                     if (ln.startswith("import ") or ln.startswith("from "))]
        joined = "".join(top_level)
        assert "src.processing" not in joined
        assert "src.gui" not in joined, (
            "src.gui must be imported INSIDE main(), after load_dotenv() and "
            "after --help has had its chance to exit cheaply."
        )


class TestInterpreterGuard:
    r"""Launching by filename on Windows hands the file to `C:\WINDOWS\py.exe`,
    which resolves a `/usr/bin/env <name>` shebang against PATH. A conda env
    ships only `python.exe` -- no `python3.exe` -- so a `python3` shebang
    MISSES and py.exe falls back to its highest registered interpreter,
    ignoring the environment the user activated. Measured live:

        #!/usr/bin/env python3  -> C:\Python313\python.exe        WRONG
        #!/usr/bin/env python   -> <the conda env>\python.exe     RIGHT

    The version pin is what makes that shebang safe: unactivated, PATH falls
    through to miniconda3 base, which has python-dotenv AND numpy 2.0.1, so it
    would run the pipeline on the wrong numpy major version silently.
    """

    def _shebang(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.readline().strip()

    def test_both_entrypoints_use_the_bare_python_shebang(self):
        from src.processing.paths import REPO_ROOT
        for name in ("run_prajnya.py", "run_prajnya_studio.py"):
            shebang = self._shebang(str(REPO_ROOT / name))
            assert shebang == "#!/usr/bin/env python", (
                f"{name} must not use `python3`: conda ships no python3.exe, so "
                f"py.exe would ignore the activated env. Got: {shebang!r}"
            )

    def test_required_python_matches_the_conda_env(self):
        """3.10 uniquely identifies the correct env against BOTH wrong
        interpreters on this machine (base 3.12, system 3.13)."""
        assert launcher._REQUIRED_PYTHON == (3, 10)

    def test_the_suite_itself_runs_on_the_required_interpreter(self):
        """If this fails, pytest is running on the wrong Python -- which is
        the very condition the guard exists to prevent."""
        assert sys.version_info[:2] == launcher._REQUIRED_PYTHON

    def test_wrong_interpreter_exits_rather_than_returning(self):
        with pytest.raises(SystemExit) as excinfo:
            launcher._wrong_interpreter("test reason")
        message = str(excinfo.value)
        assert "test reason" in message
        assert sys.executable in message

    def test_the_guidance_names_the_environment_not_one_machine(self):
        """This assertion used to require the literal path of the
        development machine's interpreter: it pinned that machine's own path
        into the one message a stranger is most likely to see. V4 never did
        this: it says `python run_Prajnya.py`. A cloned tree has no such
        folder, so the advice was unfollowable exactly where it was most
        needed.

        What the message must still do is be ACTIONABLE, which is why the fix
        is not simply deleting the path: it names the environment, says how to
        activate it, and says how to find it when activation misses (the env is
        registered by path, so `conda activate prajnya` can fail)."""
        with pytest.raises(SystemExit) as excinfo:
            launcher._wrong_interpreter("test reason")
        message = str(excinfo.value)

        assert "conda activate prajnya" in message
        assert "conda env list" in message

        #: `sys.executable` is legitimately in the message — it reports which
        #: Python actually ran — so only the GUIDANCE is checked here.
        guidance = message.split("Use the project's environment", 1)[1]
        assert re.search(r"\b[A-Za-z]:[\\/]", guidance) is None, "the guidance names a drive path"

    def test_guard_message_is_ascii_only(self):
        """The Windows console is cp1252; a non-ASCII character in the one
        message that must survive a broken environment is a bad trade."""
        with pytest.raises(SystemExit) as excinfo:
            launcher._wrong_interpreter("reason")
        str(excinfo.value).encode("ascii")  # raises UnicodeEncodeError if not
