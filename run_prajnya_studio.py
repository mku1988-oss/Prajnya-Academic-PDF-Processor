#!/usr/bin/env python
"""
Prajnya Studio (V6) entrypoint — open the graphical interface.

For biology researchers: this is the GUI counterpart of `run_prajnya.py`.
That one processes a paper and prints to the terminal; this one opens a
window in your browser where you can pick a folder, watch a batch run, read
the results, review flagged metadata, and change which AI model does what.

    run_prajnya_studio.py                    open it, default port 8080
    run_prajnya_studio.py --port 8973        use a different port
    run_prajnya_studio.py --no-browser       start the server, don't open a tab
    run_prajnya_studio.py --verbose          DEBUG logging

Underneath it is the same pipeline. Both entrypoints end up in
`src/processing/runner.py` — the GUI's call arrives there through a
subprocess worker (`src/processing/worker.py`) so a 50-minute batch cannot
freeze the interface. Neither front end reimplements "process a document";
that was the whole point of S1.

The local llama-server (router mode) must already be running before you
start a batch, exactly as for the command line — see CLAUDE.md, "Running
things".

---

**This file is a front end, and only a front end** — the same rule
`run_prajnya.py` states about itself. What lives here is the command line,
logging setup, the port check, and starting the server. Everything you
actually see and click is defined in `src/gui/app.py`, which this file
imports and hands its options to.
"""

from __future__ import annotations

import sys

# =============================================================================
# Interpreter guard -- runs before anything else, and must stay at module level
# =============================================================================
#
# **Why the shebang above says `python` and not `python3`.** On Windows,
# running `.\run_prajnya_studio.py` by its own filename hands the file to the
# `.py` association, which is `C:\WINDOWS\py.exe` -- the Python Launcher.
# py.exe DOES honour a `/usr/bin/env <name>` shebang by searching PATH for
# `<name>.exe`, but a conda environment ships only `python.exe`; there is no
# `python3.exe`. So `python3` misses on PATH, py.exe silently falls back to its
# highest REGISTERED interpreter (measured: C:\Python313\python.exe -- conda
# envs are not in the PEP 514 registry at all), and the activated environment
# is ignored entirely. With `python`, the PATH lookup hits the activated env
# and the obvious command simply works. Measured, all three cases:
#     #!/usr/bin/env python3  -> C:\Python313\python.exe          WRONG
#     #!/usr/bin/env python   -> the activated conda env            RIGHT
#     (no shebang)            -> C:\Python313\python.exe          WRONG
#
# **Why that change alone would have made things WORSE, and hence the version
# check.** With `python` on the shebang and the env NOT activated, PATH falls
# through to miniconda3 base -- which has Python 3.12.9, numpy 2.0.1, AND
# python-dotenv installed. It would import cleanly and run the whole pipeline
# on the wrong numpy major version: the silent wrong-numbers failure CLAUDE.md
# calls the worst possible mode on a project gated by IoU thresholds and word
# coverage ratios. The version check is what makes the shebang change safe --
# 3.10 uniquely identifies the correct env against both wrong interpreters
# (base is 3.12, system is 3.13) without hardcoding a path.
#
# Messages are deliberately ASCII-only: the Windows console is cp1252.

#: The `shastrasaara` conda env is Python 3.10.16. If the environment is ever
#: deliberately upgraded, change this line -- it is a pin, not a discovery.
_REQUIRED_PYTHON = (3, 10)


def _wrong_interpreter(problem: str) -> None:
    """Stop with an explanation instead of a traceback about a package that,
    on the correct interpreter, is installed."""
    sys.exit(
        f"\nPrajnya Studio could not start: {problem}\n"
        "\n"
        f"  interpreter : {sys.executable}\n"
        f"  version     : {sys.version.split()[0]}"
        f"   (this project needs {_REQUIRED_PYTHON[0]}.{_REQUIRED_PYTHON[1]}.x)\n"
        "\n"
        "Use the project's environment -- a conda env named 'shastrasaara',\n"
        "created from requirements.txt:\n"
        "\n"
        "  conda activate shastrasaara\n"
        "  python run_prajnya_studio.py\n"
        "\n"
        "If activation cannot find it by name, it is registered by path.\n"
        "'conda env list' prints where it is, and naming its interpreter\n"
        "directly always works:\n"
        "\n"
        "  <env>/python.exe run_prajnya_studio.py\n"
        "\n"
        "See README.md, 'The Python environment'.\n"
    )


if sys.version_info[:2] != _REQUIRED_PYTHON:
    _wrong_interpreter("the wrong Python is running this file.")

# load_dotenv() BEFORE any project import, for the same reason run_prajnya.py
# does it: config/loader.py interpolates ${ENV} from os.environ but never
# reads .env itself. src/gui/app.py also calls it (it has to work under
# `python -m src.gui.app` too); load_dotenv is idempotent, so calling it in
# both places is harmless and neither one can be the only guard.
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    _wrong_interpreter("this Python has none of the project's dependencies.")

load_dotenv()

import argparse
import logging
import socket

DEFAULT_HOST = "127.0.0.1"

#: NOT 8080, and not any of the obvious neighbours. **The local llama-server
#: — the thing this whole pipeline talks to — listens on 127.0.0.1:8080**
#: (see CLAUDE.md, "Running things", and the offline-verification grep for
#: `localhost:8080`). Defaulting the GUI there means the GUI cannot start on
#: exactly the machine that is set up correctly to run it. Also taken on this
#: machine: 8765 and 8767 (SLM's dashboard), 9080. 8850 is clear of all of
#: them and of the usual 8000/8888/3000/5000 crowd.
DEFAULT_PORT = 8850


def main() -> int:
    """Parse the command line, check the port, start the server.

    Returns:
        A process exit code. 0 is only ever reached by the user stopping the
        server with Ctrl-C — until then `app.main()` does not return, because
        it is serving. 1 means we refused to start.
    """
    args = _parse_args()
    _setup_logging(args.verbose)
    logger = logging.getLogger("prajnya_studio")

    if _port_is_taken(args.host, args.port):
        logger.error(
            "Port %d on %s is already in use — refusing to start.\n"
            "        Something else is already listening there. Stop it, or "
            "pick another port with --port.",
            args.port, args.host,
        )
        return 1

    # Imported here, not at module top, for two reasons: load_dotenv() above
    # must run first (importing app.py builds config that reads os.environ),
    # and an --help run should not pay for importing nicegui and the whole
    # pipeline just to print usage text.
    from src.gui import app as studio_app

    logger.info("Prajnya Studio starting on http://%s:%d", args.host, args.port)
    try:
        studio_app.main(host=args.host, port=args.port, show=not args.no_browser)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    return 0


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open Prajnya Studio, the graphical interface for Shastrasaara V6.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"TCP port to serve on (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="interface to bind. The default is loopback only. "
                             "This app reads and writes files anywhere on this "
                             "machine by design, so putting it on a network has "
                             f"to be deliberate (default: {DEFAULT_HOST})")
    parser.add_argument("--no-browser", action="store_true",
                        help="start the server without opening a browser tab")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    """Same format as run_prajnya.py, so a GUI log and a CLI log read alike."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _port_is_taken(host: str, port: int) -> bool:
    """True if something is ALREADY listening on this port.

    This check exists because of a real hour lost in session 15: SLM's own
    dashboard was already bound to port 8765, and starting the GUI there
    printed nicegui's usual "ready" line anyway while every single route
    returned 404 — the other server was quietly answering instead. A busy
    port does not always fail loudly, so we look before we leap.

    We test by CONNECTING, not by binding. Binding is the obvious approach
    and it is wrong on Windows: `SO_REUSEADDR` there permits binding a port
    that is already in use (its meaning differs from Linux's), so a bind
    test can succeed against an occupied port and report it free. A
    successful connection is unambiguous — someone accepted it.
    """
    # 0.0.0.0 means "every interface"; you cannot connect TO it, so probe
    # loopback, which is one of the interfaces it covers.
    target = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((target, port), timeout=0.5):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(main())
