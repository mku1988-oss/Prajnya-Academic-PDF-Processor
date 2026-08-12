#!/usr/bin/env python3
"""
Run-level policy — one setting, read the same way by the CLI and the GUI.

For biology researchers: this is the rule that decides what happens when the
system meets a book. It lives in `config/processing_policy.yaml` rather than
inside either front end, so the terminal and the GUI cannot disagree about it —
they read the same file and each may override it for a single run.

**Why a policy and not a yes/no flag.** The obvious design is a boolean
("interactive: true"), and it breaks as soon as there are two front ends: the
GUI has no terminal to prompt on, so it would need a second setting to
suppress a prompt that only makes sense in the CLI. Instead `ask` is one
*value* of the policy, and asking is a capability the caller supplies:

    CLI, interactive     supplies a terminal prompt  -> asks
    CLI, --noninteractive supplies nothing           -> falls back to SKIP
    GUI                  supplies a dialog           -> asks in a window
    a script             supplies nothing            -> falls back to SKIP

Neither front end needs to know the other exists, and an unattended run can
never hang waiting for an answer nor quietly spend hours of GPU.

**Precedence, highest first:** a CLI flag, then a per-run override (what a GUI
passes), then the YAML, then the built-in default — which is always the cheap
one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

POLICY_FILENAME = "processing_policy.yaml"

#: Read no chapter bodies. Every fallback resolves here, deliberately.
DEFAULT_BOOK_POLICY = "skip"
DEFAULT_IMPLAUSIBLE_CHAPTER_PAGES = 80


class BookPolicy(str, Enum):
    """What to do with a document classified as a book or a book chapter."""
    SKIP = "skip"            # front matter and chapter map only
    ASK = "ask"              # let the user decide, if there is a way to ask
    ALL = "all"              # every chapter — hours of GPU on a full volume
    CHAPTERS = "chapters"    # the chapters named in the settings

    @classmethod
    def parse(cls, value: Any) -> "BookPolicy":
        """
        Read a policy name, falling back to SKIP rather than raising.

        A typo in a config file must not stop a run, and must not be resolved
        towards the expensive option.
        """
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            LOG.warning("Unknown book policy %r; using %r.", value, DEFAULT_BOOK_POLICY)
            return cls.SKIP


@dataclass
class BookSettings:
    """
    The resolved book policy for one run.

    Attributes:
        policy: what to do with a book.
        chapters: chapter numbers, used when `policy` is CHAPTERS.
        max_pages: ceiling on body pages read, whatever the policy.
        implausible_chapter_pages: a `book_chapter` longer than this is treated
            as a misclassified book and never has its bodies read.
        source: where the policy came from, for the run summary and the log.
    """
    policy: BookPolicy = BookPolicy.SKIP
    chapters: List[int] = field(default_factory=list)
    max_pages: Optional[int] = None
    implausible_chapter_pages: int = DEFAULT_IMPLAUSIBLE_CHAPTER_PAGES
    source: str = "built-in default"

    def wants_prompt(self) -> bool:
        """True when this policy calls for asking the user."""
        return self.policy is BookPolicy.ASK


def load_book_settings(config_root: Path | str) -> BookSettings:
    """
    Read the book policy from `config/processing_policy.yaml`.

    A missing or malformed file is not an error: it resolves to the built-in
    default, which reads no chapter bodies. A pipeline that cannot read its
    settings must not fall back to the expensive behaviour.

    Args:
        config_root: the `config/` directory.

    Returns:
        The settings the file expresses, before any CLI or per-run override.

    Example:
        >>> load_book_settings("config").policy       # doctest: +SKIP
        <BookPolicy.ASK: 'ask'>
    """
    path = Path(config_root) / POLICY_FILENAME
    raw = _read_yaml(path)
    if raw is None:
        return BookSettings(source=f"built-in default ({POLICY_FILENAME} not found)")

    books = raw.get("books") or {}
    return BookSettings(
        policy=BookPolicy.parse(books.get("policy", DEFAULT_BOOK_POLICY)),
        chapters=_as_int_list(books.get("chapters")),
        max_pages=_as_optional_int(books.get("max_pages")),
        implausible_chapter_pages=_as_optional_int(
            books.get("implausible_chapter_pages")) or DEFAULT_IMPLAUSIBLE_CHAPTER_PAGES,
        source=str(path),
    )


def resolve_book_settings(config_root: Path | str,
                          cli_chapters: Optional[List[int]] = None,
                          cli_max_pages: Optional[int] = None,
                          noninteractive: bool = False,
                          overrides: Optional[Dict[str, Any]] = None) -> BookSettings:
    """
    Apply the precedence chain and produce the settings this run will use.

    Args:
        config_root: the `config/` directory.
        cli_chapters: `--chapters`. An empty list means "every chapter"; None
            means the flag was not given.
        cli_max_pages: `--max-pages`.
        noninteractive: `--noninteractive`. Downgrades ASK to SKIP, because
            there is nobody to ask.
        overrides: a per-run override, which is how a GUI passes a choice
            without writing the file. Same keys as the YAML's `books` block.

    Returns:
        The settings to use, with `source` naming what decided them.

    Example:
        >>> resolve_book_settings("config", cli_chapters=[3]).policy  # doctest: +SKIP
        <BookPolicy.CHAPTERS: 'chapters'>
    """
    settings = load_book_settings(config_root)

    if overrides:
        settings = _apply_overrides(settings, overrides)

    if cli_max_pages is not None:
        settings.max_pages = cli_max_pages

    if cli_chapters is not None:
        # An explicit flag always wins, and `--chapters` with no numbers is a
        # deliberate request for the whole book.
        settings.policy = BookPolicy.CHAPTERS if cli_chapters else BookPolicy.ALL
        settings.chapters = list(cli_chapters)
        settings.source = "--chapters"

    if noninteractive and settings.policy is BookPolicy.ASK:
        # Nobody is watching. Downgrade to the cheap option, never the costly
        # one — an unattended run must not hang, and must not overspend.
        settings.policy = BookPolicy.SKIP
        settings.source += " (--noninteractive: 'ask' downgraded to 'skip')"

    LOG.info("Book policy: %s (from %s)", settings.policy.value, settings.source)
    return settings


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    """The file as a dict, or None when it is missing or unreadable."""
    if not path.is_file():
        LOG.info("No %s; using the built-in policy (no chapter bodies read).",
                 path.name)
        return None

    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return loaded if isinstance(loaded, dict) else None
    except Exception as error:
        LOG.error("Could not read %s (%s); using the built-in policy.", path, error)
        return None


def _apply_overrides(settings: BookSettings, overrides: Dict[str, Any]) -> BookSettings:
    """Apply a per-run override — how a GUI supplies a choice for one run."""
    if "policy" in overrides:
        settings.policy = BookPolicy.parse(overrides["policy"])
    if "chapters" in overrides:
        settings.chapters = _as_int_list(overrides["chapters"])
    if "max_pages" in overrides:
        settings.max_pages = _as_optional_int(overrides["max_pages"])
    settings.source = "per-run override"
    return settings


def _as_int_list(value: Any) -> List[int]:
    """Chapter numbers from a config value, ignoring anything unparseable."""
    if not isinstance(value, (list, tuple)):
        return []
    numbers = []
    for item in value:
        try:
            numbers.append(int(item))
        except (TypeError, ValueError):
            LOG.warning("Ignoring non-numeric chapter %r", item)
    return sorted(set(numbers))


def _as_optional_int(value: Any) -> Optional[int]:
    """An int, or None for null/blank/unparseable."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        LOG.warning("Ignoring non-numeric value %r", value)
        return None
