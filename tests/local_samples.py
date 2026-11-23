"""
Where the local sample PDFs are, read from one setting (v6.9 item 8).

For biology researchers: some tests open real papers (the 8-page NMR
benchmark, the corpus) to check that reading them still works. Those PDFs
are copyrighted and never shipped, and their folder is different on every
computer. So the folder is a setting, `TEST_SAMPLES` in `.env` (with the
project's usual prefix), pointing at the folder that holds `Full/` with the
corpus PDFs. When it is not set, or the folder is missing, the tests that need
a PDF skip and say why; nothing else changes.

Only this one key is read from `.env`: the test session does not load the
whole file, which would hand every test the API keys and change what some
of them check.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values

from src.processing.env_settings import env_setting, setting_name

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The setting's name, without the prefix.
SETTING = "TEST_SAMPLES"

#: Returned when the folder is not set: a path that never exists, so every
#: test's own "is the PDF there?" check skips as it always did.
NOT_SET = REPO_ROOT / "tests" / "_sample_pdfs_not_set"


def sample_root() -> Path:
    """The sample-PDF folder (it holds `Full/`), or `NOT_SET`."""
    value = env_setting(SETTING) or dotenv_values(REPO_ROOT / ".env").get(setting_name(SETTING))
    return Path(value) if value else NOT_SET


def corpus_dir() -> Path:
    """The corpus PDFs: `<sample folder>/Full`."""
    return sample_root() / "Full"


def skip_reason() -> str:
    """Why a test that needs a sample PDF skipped."""
    return f"sample PDFs not available: set {setting_name(SETTING)} in .env to the folder holding Full/"


SAMPLES = sample_root()
CORPUS = corpus_dir()
BENCHMARK_PDF = CORPUS / "An NMR method to characterize multiple water compartments on mammalian collagen.pdf"
