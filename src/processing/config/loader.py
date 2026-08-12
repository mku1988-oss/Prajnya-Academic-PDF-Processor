#!/usr/bin/env python3
"""
Configuration loading — YAML, environment interpolation, and registry references.

For biology researchers: nearly everything about how a document gets processed
lives in YAML files rather than in code — which model reads the pages, at what
resolution, which prompt it is given. This module is what turns those files
into usable settings, and it does three jobs:

1. **Load** the YAML (always UTF-8; Windows would otherwise default to cp1252
   and choke on the first accented author name).
2. **Interpolate the environment.** A config may write `api_key: ${ZOTERO_API_KEY}`
   so that secrets live in the environment, never in a file that gets committed.
3. **Resolve registry references.** A config may write
   `profile: '@profile_registry.default'` instead of naming a model directly.
   That indirection is the VLM-agnostic rule in practice: swapping the model
   for a whole role means editing one registry line, not hunting through the
   file. See CLAUDE.md, "Modularity contracts".

## Why the environment interpolation is new code, not a port

`plan.md` and `CLAUDE.md` both describe V5's config loader as already having
`${ENV}` interpolation, to be "extracted untouched". **It does not exist.**
Every loader in V5 is a bare `yaml.safe_load`, so `api_key: ${ZOTERO_API_KEY}`
was never expanded — that literal string sat in the config as-is.

**This was not breaking anything in V5**, and it is worth being precise about
why, because the obvious guess is wrong. `run_Prajnya.py` and
`PrajnyaBootstrap.initialize_environment()` both call `load_dotenv()` at
startup, which populates `os.environ` from `gemini_refactor/.env`. Every
consumer of these values then reads the environment itself first — e.g.
`zotero_service` does `os.getenv("ZOTERO_API_KEY") or config.get("api_key")` —
so the real value always wins and the literal is simply dead config.

The defect is therefore latent rather than active, and it is a trap for what
comes next rather than a bug to be fixed in what came before. It holds only
because every consumer independently re-reads the same variable. A new config
key written as `${VAR}` whose consumer lacks that belt-and-braces receives the
literal string — and a truthy literal walks straight past an `if not value:`
guard, which is how it would eventually surface as a confusing downstream
failure rather than a clear one.

Interpolating once, here, removes the trap and lets consumers stop
double-reading the environment.

**This module reads `os.environ`; it does not load `.env` itself.** V6's
entrypoint must call `load_dotenv()`, exactly as V5's does (`python-dotenv` is
already in `requirements.txt`). There is no V6 entrypoint yet.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

#: `${VAR}` or `${VAR:-fallback}` — standard shell substitution syntax, so the
#: meaning of a config line is obvious to anyone who has written a shell script.
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_config(path: Path | str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Read a YAML config file and interpolate environment variables into it.

    Args:
        path: the YAML file to read.
        logger: optional logger.

    Returns:
        The parsed configuration, with every `${VAR}` replaced.

    Raises:
        FileNotFoundError: if the file does not exist. Deliberately not caught
            here — a caller that has a sensible default should supply it, and
            V5's habit of falling back to a hardcoded config on a missing file
            hid genuine path bugs behind a silently different pipeline.
        yaml.YAMLError: if the file is not valid YAML.
    """
    log = logger or logging.getLogger("ConfigLoader")
    config_path = Path(path)

    # UTF-8 explicitly: Windows defaults to cp1252, which fails on the first
    # accented author name or typographic dash in a prompt.
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config is None:
        log.warning("Config file is empty: %s", config_path)
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping, got {type(config).__name__}: {config_path}")

    log.debug("Loaded config: %s", config_path)
    return interpolate_env(config, logger=log)


def interpolate_env(value: Any, logger: Optional[logging.Logger] = None) -> Any:
    """
    Replace every `${VAR}` in a config tree with its environment value.

    Walks dicts, lists, and strings; leaves numbers, booleans, and None alone.

    Two forms are supported:

    - `${VAR}` — the variable's value, or **None** if it is not set.
    - `${VAR:-fallback}` — the variable's value, or the literal fallback.

    An unset `${VAR}` becoming None (rather than staying as the literal text,
    which is what V5 effectively did) is the whole point: None is falsy, so a
    consumer's `if not api_key:` guard fires and the problem surfaces where it
    happens instead of three layers down as a failed API call. A warning is
    logged so the cause is never a mystery.

    A string that is only a placeholder resolves to the value's own type — so
    `${PORT:-8080}` yields the string "8080", and a bare `${MISSING}` yields
    None rather than the string "None". A placeholder embedded in surrounding
    text always yields a string, with an unset variable contributing "".
    """
    log = logger or logging.getLogger("ConfigLoader")

    if isinstance(value, dict):
        return {key: interpolate_env(item, logger=log) for key, item in value.items()}
    if isinstance(value, list):
        return [interpolate_env(item, logger=log) for item in value]
    if not isinstance(value, str):
        return value

    whole_match = ENV_PATTERN.fullmatch(value.strip())
    if whole_match:
        return _resolve_placeholder(whole_match, log)

    # Embedded in surrounding text: substitute in place, unset -> empty string.
    def substitute(match: "re.Match") -> str:
        resolved = _resolve_placeholder(match, log)
        return "" if resolved is None else str(resolved)

    return ENV_PATTERN.sub(substitute, value)


# --- helpers (after the main functions, per CLAUDE.md) ---------------------

def _resolve_placeholder(match: "re.Match", log: logging.Logger) -> Any:
    """
    Turn one `${VAR}` / `${VAR:-fallback}` match into its value.

    Returns None for an unset variable with no fallback — see `interpolate_env`
    for why None rather than the literal placeholder text.
    """
    name, fallback = match.group(1), match.group(2)
    if name in os.environ:
        return os.environ[name]
    if fallback is not None:
        log.debug("Environment variable %s unset; using config fallback", name)
        return fallback
    log.warning(
        "Environment variable %s is not set; config value resolves to None. "
        "Any feature depending on it will be unavailable.", name,
    )
    return None


def resolve_reference(value: Any, config: Dict[str, Any],
                      logger: Optional[logging.Logger] = None) -> Any:
    """
    Resolve an `@registry.key` reference against the config's registries.

    This is the role-to-model indirection that keeps model names out of `src/`.
    Ported from V5's `_resolve_reference` with its semantics unchanged,
    including the deliberate refusal to recurse.

    Supported:
        `@profile_registry.default`  -> config['profile_registry']['default']
        `@task_registry.raw_ocr`     -> config['task_registry']['raw_ocr']

    Non-reference values (strings without an `@` prefix, numbers, booleans,
    None) are returned unchanged, so it is safe to call on any config value.

    Args:
        value: a config value, which may or may not be a reference.
        config: the full config containing the registries.
        logger: optional logger.

    Returns:
        The resolved value, or `value` itself when it is not a reference.

    Raises:
        ValueError: on invalid syntax, a missing key, or a registry value that
            is itself a reference.

    Examples:
        >>> resolve_reference("@profile_registry.default", config)
        'olmocr-2-7b'
        >>> resolve_reference("literal-value", config)
        'literal-value'
        >>> resolve_reference(123, config)
        123
    """
    log = logger or logging.getLogger("ConfigLoader")

    if not isinstance(value, str) or not value.startswith("@"):
        return value

    reference_path = value[1:].split(".")
    if len(reference_path) < 2:
        raise ValueError(
            f"Invalid reference syntax: '{value}'. "
            f"Expected format: @registry_name.key (e.g. @profile_registry.default)"
        )

    current: Any = config
    for index, key in enumerate(reference_path):
        if not isinstance(current, dict):
            raise ValueError(
                f"Invalid reference: '{value}'. "
                f"Path segment '{reference_path[index - 1]}' is not a mapping."
            )
        if key not in current:
            available = ", ".join(sorted(current.keys()))
            raise ValueError(
                f"Invalid reference: '{value}'. "
                f"Key '{key}' not found in {'.'.join(reference_path[:index]) or '<root>'}. "
                f"Available keys: {available}"
            )
        current = current[key]

    # No recursion: a registry entry must be a concrete value. Allowing chains
    # would make a typo produce an infinite loop instead of an error.
    if isinstance(current, str) and current.startswith("@"):
        raise ValueError(
            f"Invalid reference: '{value}'. "
            f"Registry values cannot themselves be references. Found: {current}"
        )

    log.debug("Resolved reference '%s' -> '%s'", value, current)
    return current


def resolve_references(section: Any, config: Dict[str, Any],
                       logger: Optional[logging.Logger] = None) -> Any:
    """
    Resolve every `@registry.key` reference in a config subtree at once.

    V5 resolved references one value at a time, at each call site, which meant
    a typo in a rarely-taken branch stayed invisible until that branch ran.
    Resolving a whole section up front surfaces the error immediately.

    Args:
        section: the subtree to resolve (dict, list, or scalar).
        config: the full config containing the registries.
        logger: optional logger.

    Returns:
        The subtree with all references replaced by their values.
    """
    log = logger or logging.getLogger("ConfigLoader")

    if isinstance(section, dict):
        return {key: resolve_references(item, config, logger=log) for key, item in section.items()}
    if isinstance(section, list):
        return [resolve_references(item, config, logger=log) for item in section]
    return resolve_reference(section, config, logger=log)


class ConfigLoader:
    """
    Loads the three kinds of config file from an explicit root directory.

    The root is a constructor argument rather than a `Path(__file__)`-relative
    guess. V5 derived every config path from the location of the module doing
    the loading, so moving a module — exactly what the decomposition in
    `plan.md` does — silently broke its config lookups. `plan.md` flags this as
    a Stage 4 risk; making the root explicit removes it.

    Expected layout under `root`:

        config/
          research_article_strategy.yaml
          vlm_profiles/<name>.yaml
          task_definitions/<name>.yaml
    """

    def __init__(self, root: Path | str, logger: Optional[logging.Logger] = None) -> None:
        self.root = Path(root)
        self.logger = logger or logging.getLogger("ConfigLoader")

    def strategy(self, name: str = "research_article_strategy") -> Dict[str, Any]:
        """Load a strategy config by name."""
        return load_config(self.root / f"{name}.yaml", logger=self.logger)

    def vlm_profile(self, name: str) -> Dict[str, Any]:
        """
        Load a model profile — everything model-specific lives in one of these.

        Args:
            name: profile name without the extension, e.g. "glm-ocr".
        """
        return self._load_or_explain(self.root / "vlm_profiles" / f"{name}.yaml",
                                     kind="VLM profile", name=name)

    def task_definition(self, name: str) -> Dict[str, Any]:
        """
        Load a task definition — the instruction given to a model for one job.

        Args:
            name: task name without the extension, e.g. "raw_ocr_extraction".
        """
        return self._load_or_explain(self.root / "task_definitions" / f"{name}.yaml",
                                     kind="task definition", name=name)

    # --- helpers (after the main methods, per CLAUDE.md) ------------------

    def _load_or_explain(self, path: Path, kind: str, name: str) -> Dict[str, Any]:
        """Load a config, turning a missing file into a message that says where to look."""
        try:
            return load_config(path, logger=self.logger)
        except FileNotFoundError:
            available = (
                ", ".join(sorted(p.stem for p in path.parent.glob("*.yaml")))
                if path.parent.is_dir() else "<directory does not exist>"
            )
            raise FileNotFoundError(
                f"{kind} '{name}' not found.\n"
                f"  Expected: {path}\n"
                f"  Available: {available}"
            ) from None
