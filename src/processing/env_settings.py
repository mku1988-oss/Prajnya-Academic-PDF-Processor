#!/usr/bin/env python
"""
Prajnya's settings that come from the environment (the `.env` file).

For biology researchers: a few things can be set outside the config files,
such as where the ledger lives or where the `llama-server` program is. They
are environment variables, written in `.env` as `NAME=value`.

THE NAMES. Every setting's name starts with `ENV_PREFIX` below. Development
keeps the project's historical prefix; a release names its settings after
Prajnya. The release script (`tools/stage_transform.py`, `rename_env_prefix`)
rewrites the prefix in every file it ships, except lines marked
"release-rename: keep", which is why `LEGACY_PREFIX` keeps the old one.

THE OLD NAMES STILL WORK. Releases up to 6.7.0 used the `LEGACY_PREFIX`
names, and someone who upgrades keeps their `.env`. Ignoring it would move their ledger and caches
without a word (the ledger would look empty and every paper would be
processed again). So a setting found only under its old name is used, with a
warning that says which name to write instead.

Every reader of these settings goes through `env_setting`; never call
`os.getenv` with the full name (`tests/test_env_settings.py` checks).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Set

#: The prefix of every setting's name.
ENV_PREFIX = "PRAJNYA_"

#: The prefix releases up to 6.7.0 used; still read, with a warning.
LEGACY_PREFIX = "SARA_"  # release-rename: keep

LOG = logging.getLogger("prajnya.settings")

#: Settings already warned about in this process (one warning each is enough).
_warned: Set[str] = set()


def env_setting(name: str, default: Any = None) -> Any:
    """
    One setting from the environment, by the part of its name after the prefix.

    Args:
        name: the name without its prefix, e.g. "LEDGER_PATH".
        default: what to return when the setting is not set (or is empty).

    Returns:
        The value under the current name; else under the old name (with a
        one-time warning); else `default`.
    """
    value = os.environ.get(ENV_PREFIX + name)
    if value:
        return value
    if LEGACY_PREFIX != ENV_PREFIX:
        legacy = os.environ.get(LEGACY_PREFIX + name)
        if legacy:
            warn_old_name(name)
            return legacy
    return default


def setting_name(name: str) -> str:
    """The full name a setting is written under, e.g. "LEDGER_PATH" -> its `.env` name."""
    return ENV_PREFIX + name


# --- helpers (after the main function, per CLAUDE.md) --------------------------

def warn_old_name(name: str) -> None:
    """Say once that a setting was found under its old name, and which name to use."""
    if name in _warned:
        return
    _warned.add(name)
    LOG.warning("%s%s is the old name of this setting and still works; "
                "write it as %s%s in .env.", LEGACY_PREFIX, name, ENV_PREFIX, name)
