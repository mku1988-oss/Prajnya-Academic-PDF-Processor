#!/usr/bin/env python
"""Which citation level a run uses, and who chose it.

For biology researchers: the level is the answer to "how much may leave this
computer to find out what paper this is?". It is chosen in ONE place, the
strategy YAML:

    citations:
      level: local        # local | zotero | online

A command-line `--citations` may override it for a single run. This module
turns those two sources into one answer and records which of them decided,
so every output file can say not only its level but why it had that level.

OLDER CONFIG FILES name no level. They have a `zotero:` block with
`enabled` and `offline_mode` instead. Such a file keeps behaving as it did:
Zotero switched on and reachable reads as `zotero`, anything else as
`local`. That rule is used only when `citations.level` is absent.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

LEVEL_LOCAL = "local"
LEVEL_ZOTERO = "zotero"
LEVEL_ONLINE = "online"

#: Every level, in order of how much leaves the machine.
CITATION_LEVELS: Tuple[str, ...] = (LEVEL_LOCAL, LEVEL_ZOTERO, LEVEL_ONLINE)

#: Nothing leaves the machine unless the researcher chooses it (session 31).
DEFAULT_LEVEL = LEVEL_LOCAL

#: Levels whose lookups exist in the code today. Stage S1 built `online`
#: (Crossref + OpenAlex). A level named but not listed here is refused
#: rather than quietly run as something else.
BUILT_LEVELS: Tuple[str, ...] = (LEVEL_LOCAL, LEVEL_ZOTERO, LEVEL_ONLINE)

#: Who decided the level. Written to every document as `level_set_by`.
SET_BY_CLI = "cli"                      # --citations on the command line
SET_BY_CONFIG = "config"                # citations.level in the YAML
SET_BY_LEGACY_CONFIG = "legacy_config"  # derived from the old zotero block
SET_BY_DEFAULT = "default"              # nothing said anything
LEVEL_SET_BY: Tuple[str, ...] = (SET_BY_CLI, SET_BY_CONFIG,
                                 SET_BY_LEGACY_CONFIG, SET_BY_DEFAULT)


def resolve_level(config: Optional[Dict[str, Any]],
                  override: Optional[str] = None) -> Tuple[str, str]:
    """
    Decide the citation level for a run.

    Order of authority: the command-line override, then `citations.level` in
    the config, then the older `zotero` block, then the default (`local`).

    Args:
        config: the loaded strategy config (a dict), or None.
        override: the value of `--citations`, or None when it was not given.

    Returns:
        `(level, set_by)`: the level, and which source decided it.

    Raises:
        ValueError: when the override or the config names a level that does
            not exist. A misspelt level must stop the run: quietly falling
            back to another level could send data out that the researcher
            meant to keep in, or the reverse.

    Example:
        >>> resolve_level({"citations": {"level": "online"}})
        ('online', 'config')
        >>> resolve_level({"citations": {"level": "online"}}, override="local")
        ('local', 'cli')
        >>> resolve_level({"zotero": {"enabled": True}})
        ('zotero', 'legacy_config')
        >>> resolve_level({})
        ('local', 'default')
    """
    config = config if isinstance(config, dict) else {}

    if override:
        return _checked(override, "--citations"), SET_BY_CLI

    block = config.get("citations")
    named = block.get("level") if isinstance(block, dict) else None
    if named:
        return _checked(named, "citations.level"), SET_BY_CONFIG

    if isinstance(config.get("zotero"), dict):
        return _level_from_legacy_zotero(config["zotero"]), SET_BY_LEGACY_CONFIG

    return DEFAULT_LEVEL, SET_BY_DEFAULT


def require_built(level: str) -> str:
    """
    Refuse a level whose lookups are not written yet.

    Args:
        level: a level returned by `resolve_level`.

    Returns:
        The same level, when it can be run.

    Raises:
        NotImplementedError: for `online`, until stage S1 is built. Running
            it as `local` while the output file said `online` would be a
            false statement in every document.
    """
    if level not in BUILT_LEVELS:
        raise NotImplementedError(
            f"Citation level {level!r} is not built yet (stage S1 of "
            f"docs/citation_levels_plan.md). Use one of: {', '.join(BUILT_LEVELS)}.")
    return level


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def _checked(value: Any, where: str) -> str:
    """The level, lower-cased, or a ValueError naming the allowed ones."""
    level = str(value).strip().lower()
    if level not in CITATION_LEVELS:
        raise ValueError(
            f"{where} is {value!r}, which is not a citation level. "
            f"Use one of: {', '.join(CITATION_LEVELS)}.")
    return level


def _level_from_legacy_zotero(zotero: Dict[str, Any]) -> str:
    """
    What an older config (no `citations.level`) was already doing.

    `enabled: true` with `offline_mode: false` contacted Zotero, which is
    the `zotero` level. Every other combination looked nothing up.
    """
    if zotero.get("enabled", False) and not zotero.get("offline_mode", False):
        return LEVEL_ZOTERO
    return LEVEL_LOCAL
