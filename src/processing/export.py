#!/usr/bin/env python3
"""
A portable copy of a Prajnya output file, safe to share (audit L1).

For biology researchers: every output file records `profile.pdf_path`, the
FULL path of the PDF on the computer that ran Prajnya
("D:\\Lab\\Projects\\Collagen\\paper.pdf"). That names folders on your machine,
which you may not want in a file you send to a colleague or publish with a
dataset. This writes a copy in which that field holds only the PDF's file
name ("paper.pdf"). Nothing else changes, and the copy is still a valid
output file under the same contract version (the field stays a string).

    python -m src.processing.export run_outputs/documents/paper.v6.json
    python -m src.processing.export paper.v6.json --out shared/paper.json

The copy is written beside the original as `<name>.portable.json` unless
`--out` says otherwise. Figure and table images are NOT copied: the file
refers to them by paths relative to the output folder, as the original does.
`profile.pdf_path` itself stays required until contract 2.0 (v7).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional

from .utils.atomic_write import write_json_atomically

PORTABLE_SUFFIX = ".portable.json"


def portable_copy(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    A deep copy of an output document with `profile.pdf_path` cut to the file name.

    Args:
        document: a Prajnya output document (the parsed `.v6.json`).

    Returns:
        The copy. The input is not changed. A document without the field is
        copied unchanged.

    Example:
        >>> portable_copy({"profile": {"pdf_path": "D:\\\\Lab\\\\paper.pdf"}})["profile"]["pdf_path"]
        'paper.pdf'
    """
    shared = copy.deepcopy(document)
    profile = shared.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("pdf_path"), str):
        profile["pdf_path"] = file_name_only(profile["pdf_path"])
    return shared


def export_document(source: Path, destination: Optional[Path] = None) -> Path:
    """
    Read an output file and write its portable copy.

    Args:
        source: the `.v6.json` file.
        destination: where to write; default `<source stem>.portable.json`
            beside the source.

    Returns:
        The path written.
    """
    document = json.loads(Path(source).read_text(encoding="utf-8"))
    target = Path(destination) if destination else default_destination(Path(source))
    write_json_atomically(target, portable_copy(document))
    return target


def main(argv: Optional[List[str]] = None) -> int:
    """Command line: export one or more output files."""
    arguments = parse_arguments(argv)
    if arguments.out and len(arguments.documents) > 1:
        print("--out names one file; give one document with it.", file=sys.stderr)
        return 2
    for name in arguments.documents:
        written = export_document(Path(name), Path(arguments.out) if arguments.out else None)
        print(f"Wrote {written}")
    return 0


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def file_name_only(path: str) -> str:
    """The last part of a path written on Windows or on Linux/macOS."""
    return PureWindowsPath(path).name


def default_destination(source: Path) -> Path:
    """`paper.v6.json` -> `paper.portable.json`, in the same folder."""
    stem = source.name[:-len(".v6.json")] if source.name.endswith(".v6.json") else source.stem
    return source.with_name(stem + PORTABLE_SUFFIX)


def parse_arguments(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.processing.export",
        description="Write a copy of a Prajnya output file with profile.pdf_path cut to the file name.")
    parser.add_argument("documents", nargs="+", help="output files (.v6.json)")
    parser.add_argument("--out", help="where to write (one document only)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
