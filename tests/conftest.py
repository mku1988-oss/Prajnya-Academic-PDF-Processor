#!/usr/bin/env python3
"""
Shared pytest configuration for the V6 test suite.

The main job here is letting the parity harness point at either the archived
baseline (fast, no GPU, always available) or a fresh pipeline run:

    pytest tests/ -v                                    # baseline
    pytest tests/ -v --run-output run_outputs/foo.json  # fresh run
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Make the repo root importable so `from tests.parity import ...` works no matter
# which directory pytest was invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_DOCUMENT = REPO_ROOT / "tests" / "baselines" / "nmr_known_good_document.json"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-output",
        action="store",
        default=None,
        metavar="PATH",
        help="Compare a fresh pipeline output JSON instead of the archived baseline.",
    )


@pytest.fixture(scope="session")
def document_path(request: pytest.FixtureRequest) -> Path:
    """Which document JSON the parity harness should check."""
    supplied = request.config.getoption("--run-output")
    if supplied:
        path = Path(supplied)
        if not path.is_file():
            pytest.fail(f"--run-output file not found: {path}")
        return path

    if not BASELINE_DOCUMENT.is_file():
        pytest.fail(
            f"baseline document missing: {BASELINE_DOCUMENT}\n"
            "See tests/baselines/README.md."
        )
    return BASELINE_DOCUMENT


@pytest.fixture(scope="module")
def document(document_path: Path) -> Dict[str, Any]:
    """The loaded document JSON under test."""
    from tests.parity import load_document  # noqa: PLC0415

    return load_document(document_path)


def pytest_report_header(config: pytest.Config) -> str:
    target = config.getoption("--run-output") or f"{BASELINE_DOCUMENT.name} (archived baseline)"
    return f"parity target: {target}"
