#!/usr/bin/env python3
"""
Title-page readings kept on disk between the two phases of a batch (T39).

For biology researchers: a batch is processed "by model" — first the title
model (olmocr, ~5.5 GB) reads the title page of every paper in a chunk, then
the body model (GLM, ~3 GB) reads the rest of each paper. The two models do
not fit on the 8 GB GPU together, and swapping one for the other costs ~14 s,
so doing all the title pages first saves two swaps per paper.

Between the two phases, each title-page reading has to wait somewhere. It
waits HERE, as one small JSON file per paper in a hidden folder of the run's
output directory, never in memory (researcher, session 47: a large corpus
must not grow the memory a run needs). Writing a file costs milliseconds; a
title page costs ~28 s of GPU time.

What is stored is the FACT the GPU produced (the title-page `ContentPage`),
never anything derived from it: metadata extraction and enrichment run again
in phase 2 from this reading ("cache facts, derive on read", CLAUDE.md).

A file is only trusted when it matches the PDF it was made from (a SHA-256 of
the file's bytes) AND the title-page settings that made it (a fingerprint of
the model, its profile, its two tasks and the resolution). Anything else is
ignored and the title page is read again — a stale reading must never leak
into a result.

The folder is removed after a full run. After a crash it stays, so a re-run
reuses the readings already paid for, and so they can be inspected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .models import ContentPage

#: The hidden folder, under the run's output root, holding the readings.
TITLE_PAGE_FOLDER = ".title_pages"

#: Bumped if the file layout ever changes, so an old file is never misread.
FILE_FORMAT = 1

LOG = logging.getLogger(__name__)


@dataclass
class SavedTitlePage:
    """
    One title-page reading as phase 2 receives it.

    Attributes:
        page: the title page exactly as the title model returned it.
        elapsed: seconds phase 1 spent on this document (profile + title page),
            added to phase 2's time so `processing_time` stays the whole cost.
    """
    page: ContentPage
    elapsed: float


class TitlePageStore:
    """
    The folder of saved title-page readings for one run.

    Args:
        out_root: the run's output directory (`RunRequest.out_root`).
        logger: optional logger.
    """

    def __init__(self, out_root: Path, logger: Optional[logging.Logger] = None) -> None:
        self.folder = Path(out_root) / TITLE_PAGE_FOLDER
        self.logger = logger or LOG

    def save(self, pdf_path: Path, settings_key: str, page: ContentPage,
             elapsed: float) -> Path:
        """
        Write one reading to disk, then check it reads back unchanged.

        The check is gate mark 6 (eval/batch_by_model/GATE.md): both arms of
        the test go through this file, so a loss in the round trip would be
        invisible to the byte-identity mark. It is logged on every save.

        Args:
            pdf_path: the PDF the reading came from.
            settings_key: `ResearchArticleStrategy.title_settings_key()`.
            page: the title page as the model returned it.
            elapsed: seconds spent on this document in phase 1.

        Returns:
            The file written.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self._path_for(pdf_path)
        record = {
            "format": FILE_FORMAT,
            "pdf_name": pdf_path.name,
            "pdf_sha256": _file_sha256(pdf_path),
            "settings_key": settings_key,
            "elapsed": elapsed,
            "page": page.model_dump(mode="json"),
        }
        _write_json(path, record)
        self._check_round_trip(path, page)
        return path

    def load(self, pdf_path: Path, settings_key: str) -> Optional[SavedTitlePage]:
        """
        The saved reading for this PDF, or None if there is none to trust.

        None when the file is missing, unreadable, from another file format,
        made from different PDF bytes, or made with different title settings.
        The caller then reads the title page itself, as a single run does.
        """
        path = self._path_for(pdf_path)
        if not path.is_file():
            return None
        try:
            record = _read_json(path)
            if not self._matches(record, pdf_path, settings_key):
                return None
            return SavedTitlePage(page=ContentPage.model_validate(record["page"]),
                                  elapsed=float(record.get("elapsed") or 0.0))
        except Exception as error:
            self.logger.warning("Saved title page for %s is unreadable (%s); "
                                "it will be read again.", pdf_path.name, error)
            return None

    def discard(self, pdf_path: Path) -> None:
        """Remove one document's reading, and the folder once it is empty."""
        path = self._path_for(pdf_path)
        if path.is_file():
            path.unlink()
        if self.folder.is_dir() and not any(self.folder.iterdir()):
            self.folder.rmdir()

    def clear(self) -> None:
        """Remove the whole folder: called once, after a full run."""
        if self.folder.is_dir():
            shutil.rmtree(self.folder)
            self.logger.info("Removed the saved title pages (%s)", self.folder)

    # --- helpers (after the main methods, per CLAUDE.md) ------------------

    def _path_for(self, pdf_path: Path) -> Path:
        """One file per PDF, named after it (as its result JSON is)."""
        return self.folder / f"{Path(pdf_path).stem}.title.json"

    def _matches(self, record: dict, pdf_path: Path, settings_key: str) -> bool:
        """True only for a file of this format, these PDF bytes, these settings."""
        if record.get("format") != FILE_FORMAT:
            reason = "an older file format"
        elif record.get("settings_key") != settings_key:
            reason = "different title-page settings"
        elif record.get("pdf_sha256") != _file_sha256(pdf_path):
            reason = "a different version of the PDF"
        else:
            return True
        self.logger.info("Saved title page for %s was made from %s; it will be "
                         "read again.", pdf_path.name, reason)
        return False

    def _check_round_trip(self, path: Path, page: ContentPage) -> None:
        """Gate mark 6: the reading read back equals the reading written."""
        reread = ContentPage.model_validate(_read_json(path)["page"])
        if reread.model_dump() == page.model_dump():
            self.logger.info("Title page saved: %s (round trip identical)", path.name)
        else:
            self.logger.warning("Title page saved: %s (round trip DIFFERS: the "
                                "re-read reading is not the one the model returned)",
                                path.name)


def _file_sha256(pdf_path: Path) -> str:
    """SHA-256 of the file's bytes: the same identity the ledger uses."""
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _write_json(path: Path, record: Any) -> None:
    """UTF-8, LF line endings on every platform (CLAUDE.md, Windows traps)."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=1)


def _read_json(path: Path) -> Any:
    """The matching reader: UTF-8, newlines untranslated."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return json.load(handle)
