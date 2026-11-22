"""
The factory defaults must be the configuration that actually works today.

For biology researchers: config/factory_defaults/ is what Strategy Lab's
"Restore Defaults" writes back. On 2026-11-04 two of its files were found
stale: its model.ini still used `mlock`, an option llama.cpp 0.6.0 removed (a
restore would have left a preset the server refuses to start on) and lacked six
presets; its glm-ocr.yaml lacked the table output cap, so restoring GLM's
defaults would have brought back cut-off tables. Nothing had noticed, because
nothing compared the two.

So, in the committed repository, every factory default must equal the live
file it backs up. When a default is changed on purpose, change both. If this
fails right after editing settings in Strategy Lab, that is the point: the
defaults no longer match what you are running, and you decide which is right.
"""
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[1] / "config"
FACTORY = CONFIG / "factory_defaults"
FACTORY_FILES = sorted(p for p in FACTORY.rglob("*") if p.is_file())


def _live_counterpart(factory_file: Path) -> Path:
    """config/factory_defaults/<x> backs up config/<x>, except model.ini (vlm_profiles)."""
    relative = factory_file.relative_to(FACTORY)
    if relative.name == "model.ini":
        return CONFIG / "vlm_profiles" / "model.ini"
    return CONFIG / relative


def test_there_are_factory_defaults_to_check():
    assert FACTORY_FILES, "config/factory_defaults is empty"


@pytest.mark.parametrize("factory_file", FACTORY_FILES, ids=lambda p: p.relative_to(FACTORY).as_posix())
def test_each_factory_default_matches_the_live_file(factory_file):
    live = _live_counterpart(factory_file)
    assert live.exists(), f"{live} has no live counterpart"
    assert factory_file.read_bytes() == live.read_bytes(), (
        f"{factory_file.relative_to(CONFIG)} differs from {live.relative_to(CONFIG)}: "
        "'Restore Defaults' would write back something other than what runs today")
