#!/usr/bin/env python
"""Builds the outside services a citation level is allowed to contact.

For biology researchers: each citation level names who may be asked about a
paper. This is the ONE place that turns a level into live connections, used
by a normal run (the strategy) and by `--recite` alike, so the two can never
disagree about what a level contacts.

    local    nothing is built. Building the Zotero service already contacts
             Zotero (it checks the key), and `local` promises that nothing
             leaves the machine.
    zotero   the Zotero service, from the `zotero:` credentials.
    online   Crossref and OpenAlex (`online_chain_for`); Zotero too only when
             `citations.online.prefer_zotero` is true.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..paths import DEFAULT_CITATION_CACHE_PATH, DEFAULT_ZOTERO_CACHE_PATH
from ..utils.zotero_service import ZoteroService
from .chain import OnlineChain
from .crossref import CrossrefService
from .http import CitationHttp
from .level import LEVEL_ONLINE, LEVEL_ZOTERO
from .openalex import OpenAlexService
from .reference_lists import ReferenceListFinder


def zotero_service_for(level: str, config: Optional[Dict[str, Any]],
                       logger: Optional[logging.Logger] = None) -> Optional[ZoteroService]:
    """
    The Zotero service for this level, or None.

    Args:
        level: the citation level in force.
        config: the loaded strategy config (its `zotero:` block holds the
            credentials).
        logger: optional logger.

    Returns:
        A connected `ZoteroService` at level `zotero`. None at every other
        level, and None at `zotero` when the service cannot be built
        (missing credentials, `pyzotero` not installed): that is logged as an
        error and the run continues on Prajnya's own reading, because a
        missing key must not end a document run.
    """
    log = logger or logging.getLogger("citations")
    prefer_zotero = bool(_online_settings(config).get("prefer_zotero", False))
    if level == LEVEL_ONLINE and prefer_zotero:
        pass        # `online` with Zotero first: build it below
    elif level != LEVEL_ZOTERO:
        log.info("Citation level %s: Zotero is not contacted.", level)
        return None

    log.info("Citation level zotero: initializing the Zotero service.")
    zotero_config = (config or {}).get("zotero") or {}
    try:
        cache_dir = Path(os.getenv("SARA_CACHE_PATH", DEFAULT_ZOTERO_CACHE_PATH))
        return ZoteroService(config=zotero_config, cache_dir=str(cache_dir.resolve()))
    except (ImportError, ValueError) as error:
        # ValueError: an unset ZOTERO_LIBRARY_ID reaches the service as None
        # (V6's real ${ENV} interpolation), and the service's own guard fires.
        log.error("ZoteroService unavailable (%s). Continuing without Zotero.", error)
        return None


def online_chain_for(level: str, config: Optional[Dict[str, Any]],
                     zotero_service: Optional[ZoteroService] = None,
                     logger: Optional[logging.Logger] = None) -> Optional[OnlineChain]:
    """
    The Crossref + OpenAlex chain for this level, or None.

    Args:
        level: the citation level in force.
        config: the loaded strategy config; its `citations.online` block
            holds hosts, pacing, thresholds and the contact e-mail.
        zotero_service: the Zotero service, used first only when
            `citations.online.prefer_zotero` is true.

    Returns:
        An `OnlineChain` at level `online`; None at every other level.
    """
    log = logger or logging.getLogger("citations")
    if level != LEVEL_ONLINE:
        return None

    settings = _online_settings(config)
    services, cache_dir = _crossref_and_openalex(settings, log)
    rows = int(settings.get("max_hits", 5))
    log.info("Citation level online: Crossref and OpenAlex will be asked (cache %s).", cache_dir)
    return OnlineChain(
        services, zotero_service=zotero_service,
        prefer_zotero=bool(settings.get("prefer_zotero", False)),
        min_similarity=float(settings.get("title_similarity", 0.90)),
        max_hits=rows, year_tolerance=int(settings.get("year_tolerance", 1)), logger=log)


def reference_finder_for(level: str, config: Optional[Dict[str, Any]],
                         logger: Optional[logging.Logger] = None) -> Optional[ReferenceListFinder]:
    """
    The reference-list finder (stage S2) for this level, or None.

    Args:
        level: the citation level in force.
        config: the loaded strategy config (`citations.online` settings).

    Returns:
        A `ReferenceListFinder` at level `online`; None at every other level.
        It has its own HTTP client, sharing the same cache directory.
    """
    log = logger or logging.getLogger("citations")
    if level != LEVEL_ONLINE:
        return None
    (crossref, openalex), _cache = _crossref_and_openalex(_online_settings(config), log)
    return ReferenceListFinder(crossref, openalex, logger=log)


def _crossref_and_openalex(settings: Dict[str, Any], log: logging.Logger):
    """
    Build the Crossref and OpenAlex services from `citations.online` settings.

    Returns:
        `([CrossrefService, OpenAlexService], cache_dir)`.
    """
    cache_dir = Path(os.getenv("SARA_CITATION_CACHE_PATH", DEFAULT_CITATION_CACHE_PATH))
    http = CitationHttp(
        cache_dir.resolve(), pacing_seconds=settings.get("pacing_seconds", 1.2),
        timeout_seconds=settings.get("timeout_seconds", 20.0),
        contact_email=settings.get("contact_email") or None, logger=log)
    rows = int(settings.get("max_hits", 5))
    services = [
        CrossrefService(http, host=settings.get("crossref_host") or "https://api.crossref.org",
                        rows=rows),
        OpenAlexService(http, host=settings.get("openalex_host") or "https://api.openalex.org",
                        api_key=settings.get("openalex_api_key") or os.getenv("OPENALEX_API_KEY"),
                        rows=rows),
    ]
    return services, cache_dir


def _online_settings(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The `citations.online` block of the config, or {}."""
    block = (config or {}).get("citations")
    online = block.get("online") if isinstance(block, dict) else None
    return online if isinstance(online, dict) else {}
