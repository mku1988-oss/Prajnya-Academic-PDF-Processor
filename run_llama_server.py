#!/usr/bin/env python
"""
Start the local llama-server with this repository as its working directory.

For biology researchers: the pipeline does not contain an AI model. It talks to
a separate program — `llama-server`, from llama.cpp — which loads the models and
answers requests on `http://127.0.0.1:9931`. That program has to be running
before you process anything. This file starts it correctly.

    run_llama_server.py                          start it
    run_llama_server.py --check                  verify the setup, start nothing
    run_llama_server.py --port 8081              a different port
    run_llama_server.py -- --verbose             pass extra flags to llama-server

**Why this exists rather than a line in the README, and it is not tidiness.**
`model.ini` tells llama-server where each model file is. In a released tree
those paths are RELATIVE to the repository (`models/GLM/GLM-OCR-f16.gguf`), and
llama.cpp resolves them against **the server process's own working directory** —
not against the location of `model.ini`. So starting the server from the wrong
folder makes every model path point at nothing.

And it fails in the worst possible way. `--models-preset` validates **nothing**
at startup: the server launches happily, the pipeline's pre-flight pass succeeds
(it never asks the server anything), and the failure arrives **mid-run**, at the
first page that needs a model — after the CPU pass has already been paid for. A
README line asking the user to `cd` first is a step that can be forgotten
silently. This file makes it unforgettable by doing it.

The second half of the same problem is that a path can be wrong for reasons a
correct working directory does not fix — a model that was never downloaded, or
a rename. So this launcher also **resolves every path in `model.ini` before it
starts anything**, and refuses to launch when one is missing. That converts a
mid-run failure into a message you get in the first second.

Finding `llama-server` itself is deliberately NOT hardcoded: it lives outside
this repository (on the development machine, in a llama.cpp build tree) and will
be somewhere else for everyone else. In order:

  1. `--llama-server <path>`
  2. the `SARA_LLAMA_SERVER` environment variable (put it in `.env`)
  3. `llama-server` / `llama-server.exe` on PATH
"""

from __future__ import annotations

import argparse
import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

#: Anchored to this file, never to the shell's folder — the whole point of the
#: module. `src/processing/paths.py` is the project's single answer to "which
#: directory?", and this file is a front end like the other two entrypoints, so
#: it asks that module rather than computing its own answer.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.processing.paths import CONFIG_ROOT, REPO_ROOT  # noqa: E402

#: The preset file llama-server is pointed at, relative to the repository root.
#: Passed to the server as a RELATIVE path on purpose: combined with the pinned
#: working directory, that is what makes a released tree portable.
MODEL_INI_RELATIVE = "config/vlm_profiles/model.ini"

#: The keys in each preset that name a file on disk.
_PATH_KEYS = ("model", "mmproj")

#: One model at a time. The 8 GB card cannot hold the title model and the body
#: model together, and llama-server sheds models by COUNT rather than by
#: measuring VRAM — so with a higher limit a model left resident by the previous
#: document makes the next one spill to system RAM and decode about three times
#: slower, with byte-identical output. The failure is invisible in the results
#: and visible only in the clock, which is why it is pinned here rather than
#: left to be remembered. See CLAUDE.md, "VRAM residency".
DEFAULT_MODELS_MAX = 1

#: Where the pipeline expects to find the server. Changing this means changing
#: `config/research_article_strategy.yaml`'s `vlm_provider.local_server.port` too.
DEFAULT_PORT = 9931
DEFAULT_HOST = "127.0.0.1"


def main(argv: Optional[List[str]] = None) -> int:
    """Check the setup, then hand over to llama-server. Returns an exit code."""
    args = _parse_args(argv)

    ini_path = REPO_ROOT / MODEL_INI_RELATIVE
    if not ini_path.exists():
        print(f"No preset file at {ini_path}", file=sys.stderr)
        return 2

    presets, missing = check_preset_paths(ini_path)
    _report_check(ini_path, presets, missing)

    if missing and not args.allow_missing:
        print(
            "\nRefusing to start: the server would launch, and then fail at the\n"
            "first page that needs one of these models — after the run had\n"
            "already started. Download the missing files, or pass\n"
            "--allow-missing to start anyway.",
            file=sys.stderr,
        )
        return 1

    executable = resolve_llama_server(args.llama_server)
    if executable is None:
        print(_not_found_message(), file=sys.stderr)
        return 2

    if args.check:
        print(f"\nWould start: {executable}")
        print("--check given, so nothing was started.")
        return 0

    command = build_command(executable, args.host, args.port, args.models_max, args.extra)
    return launch(command)


def check_preset_paths(ini_path: Path,
                       repo_root: Path = REPO_ROOT) -> Tuple[List[Tuple[str, str, Path]], List[Tuple[str, str, Path]]]:
    """
    Resolve every model file named in `model.ini`, before starting anything.

    For biology researchers: this is the eager check. It answers "are all the
    model files actually where the configuration says they are?" — a question
    llama-server itself does not ask until the middle of a run.

    A relative path is resolved against the repository root, which is exactly
    what the server will do once this launcher has pinned its working
    directory. An absolute path is taken as given (that is the development
    machine's shared model store, and it is correct there).

    Args:
        ini_path: the `model.ini` to read.
        repo_root: what a relative path is resolved against.

    Returns:
        `(found, missing)` — each entry is `(section, key, resolved_path)`.
    """
    #: `strict=False` because `model.ini` is hand-maintained and has carried a
    #: duplicate key before; llama-server's own behaviour is last-flag-wins, so
    #: refusing to READ the file over a duplicate would be stricter than the
    #: program the file is written for.
    parser = configparser.ConfigParser(strict=False)
    parser.read(ini_path, encoding="utf-8")

    found: List[Tuple[str, str, Path]] = []
    missing: List[Tuple[str, str, Path]] = []

    for section in parser.sections():
        for key in _PATH_KEYS:
            raw = parser[section].get(key)
            if not raw:
                continue
            resolved = _resolve(raw.strip(), repo_root)
            (found if resolved.exists() else missing).append((section, key, resolved))

    return found, missing


def resolve_llama_server(explicit: Optional[str] = None) -> Optional[Path]:
    """
    Find the llama-server binary, or return None.

    Order: the explicit argument, then `SARA_LLAMA_SERVER`, then PATH. Never a
    hardcoded location — the binary lives outside this repository and is in a
    different place on every machine.
    """
    candidates = [explicit, os.environ.get("SARA_LLAMA_SERVER")]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path

    on_path = shutil.which("llama-server")
    return Path(on_path) if on_path else None


def build_command(executable: Path, host: str, port: int, models_max: int,
                  extra: Sequence[str]) -> List[str]:
    """
    The exact command line, with the preset passed as a RELATIVE path.

    Relative is the load-bearing detail: combined with the working directory
    this launcher pins, it is what lets the same command work in a released
    tree on a machine that has never heard of this one's model store.
    """
    return [
        str(executable),
        "--models-preset", MODEL_INI_RELATIVE,
        "--models-max", str(models_max),
        "--host", host,
        "--port", str(port),
        *extra,
    ]


def launch(command: Sequence[str], cwd: Path = REPO_ROOT) -> int:
    """
    Run llama-server with `cwd` as its working directory, in the foreground.

    Foreground on purpose: the server's log is what tells you a model loaded,
    and Ctrl-C should stop it. Background it with your shell if you want to.
    """
    print(f"\nWorking directory : {cwd}")
    print(f"Starting          : {' '.join(command)}\n")

    try:
        return subprocess.call(list(command), cwd=str(cwd))
    except KeyboardInterrupt:
        return 0


# =============================================================================
# Helpers — after the functions they serve, never inside them
# =============================================================================

def _resolve(raw: str, repo_root: Path) -> Path:
    """One configured model path, made absolute the way the server will."""
    path = Path(raw)
    return path if path.is_absolute() else (repo_root / path)


def _report_check(ini_path: Path, found: Sequence[Tuple[str, str, Path]],
                  missing: Sequence[Tuple[str, str, Path]]) -> None:
    """Print what the eager check saw. ASCII only — the console is cp1252."""
    print(f"Preset file : {ini_path}")
    print(f"Model files : {len(found)} found, {len(missing)} missing")

    for section, key, path in missing:
        print(f"  MISSING  [{section}] {key} -> {path}")


def _not_found_message() -> str:
    """Say how to point the launcher at llama-server, not just that it failed."""
    return (
        "\nCould not find llama-server.\n"
        "\n"
        "It is a separate program (from llama.cpp) and lives outside this\n"
        "repository. Tell this launcher where it is, in any of these ways:\n"
        "\n"
        "  run_llama_server.py --llama-server <path to llama-server.exe>\n"
        "  set SARA_LLAMA_SERVER=<path>   (or add it to .env)\n"
        "  put its folder on PATH\n"
    )


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    """The command line. Everything after `--` goes to llama-server unchanged."""
    parser = argparse.ArgumentParser(
        description="Start llama-server with this repository as its working directory.")
    parser.add_argument("--llama-server", default=None,
                        help="path to the llama-server binary")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--models-max", type=int, default=DEFAULT_MODELS_MAX,
                        help="models kept resident (default 1 — see the VRAM note)")
    parser.add_argument("--check", action="store_true",
                        help="verify the setup and exit without starting anything")
    parser.add_argument("--allow-missing", action="store_true",
                        help="start even when a model file is missing")
    parser.add_argument("extra", nargs="*",
                        help="extra flags passed straight to llama-server")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
