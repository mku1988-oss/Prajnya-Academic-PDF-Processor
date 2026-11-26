#!/usr/bin/env python
"""
Pins the llama-server preset settings that were measured, so nobody "tunes"
them back by accident.

For biology researchers: llama-server reads each model's settings from
config/vlm_profiles/model.ini. Some of those numbers look like free speed
knobs but were measured on this 8 GB GPU; these tests fail if one is changed
back to the value that was measured and rejected.
"""
from __future__ import annotations

import configparser
from pathlib import Path

PRESET = Path(__file__).resolve().parents[1] / "config" / "vlm_profiles" / "model.ini"

#: olmocr-2-7b's largest prompt-processing chunk that stays on the 8 GB card.
#: At 2048 ~566 MB of its batch buffers spill into system RAM and prompt
#: processing drops from ~870 to ~103 tokens/s (T40, eval/title_page_cache/GATE.md).
OLMOCR_MAX_UBATCH = 1024


def test_olmocr_ubatch_fits_the_gpu():
    olmocr = _read_preset()["olmocr-2-7b"]
    assert int(olmocr["ubatch-size"]) <= OLMOCR_MAX_UBATCH, (
        "olmocr-2-7b ubatch-size above 1024 spills into system RAM on the 8 GB card "
        "and halves title-page speed (T40); re-run the T40 gate before raising it")


def test_olmocr_batch_is_not_below_ubatch():
    # llama-server caps ubatch at batch, so a smaller batch would silently
    # change the chunk size that T40 measured.
    olmocr = _read_preset()["olmocr-2-7b"]
    assert int(olmocr["batch-size"]) >= int(olmocr["ubatch-size"])


# ---------------------------------------------------------------------------
# Helpers (after the tests, per the project rule)
# ---------------------------------------------------------------------------

def _read_preset() -> configparser.ConfigParser:
    """model.ini as llama-server sees it: ';' comment lines, no interpolation."""
    parser = configparser.ConfigParser(interpolation=None, strict=False,
                                       comment_prefixes=(";", "#"))
    parser.read(PRESET, encoding="utf-8")
    return parser
