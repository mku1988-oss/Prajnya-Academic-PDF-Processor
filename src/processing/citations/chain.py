#!/usr/bin/env python
"""The `online` lookup chain: Crossref, then OpenAlex (plan C2, C4).

For biology researchers: at citation level `online` Prajnya asks the public
registries what paper it is holding. The order is fixed and every answer
must pass `acceptance.py`:

    [Zotero, when `prefer_zotero` is on]
    Crossref by DOI  ->  OpenAlex by DOI        (a DOI was read)
    Crossref by title ->  OpenAlex by title     (no DOI, or the DOI was wrong)

The chain stops at the first accepted record and says who supplied it.
If NOTHING answered because the network was down, that is reported as
`network: unavailable`, which is different from "the registries do not
know this paper" (`network: ok`): the first deserves a retry, the second a
human look.

The chain answers in the same shape as `ZoteroService.get_canonical_metadata`
(`record`, `confidence`, `source`) so the merger needs no change.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..utils.metadata_merger import normalize_doi
from .acceptance import (DEFAULT_MAX_HITS, DEFAULT_TITLE_SIMILARITY,
                         DEFAULT_YEAR_TOLERANCE, choose_record, clean_name, surname_of)
from .http import STATUS_NOT_FOUND, STATUS_OK, STATUS_UNREACHABLE

#: Confidence of an accepted record: a DOI that was also found on the page is
#: the strongest evidence; a title match plus author is a little less.
CONFIDENCE_BY_DOI = 0.95
CONFIDENCE_BY_TITLE = 0.90

NETWORK_OK = "ok"
NETWORK_UNAVAILABLE = "unavailable"

#: A title shorter than this many words is too weak to search by.
MIN_SEARCH_WORDS = 3


class OnlineChain:
    """Runs the `online` steps for one document at a time."""

    def __init__(self, services: List[Any], zotero_service: Any = None,
                 prefer_zotero: bool = False,
                 min_similarity: float = DEFAULT_TITLE_SIMILARITY,
                 max_hits: int = DEFAULT_MAX_HITS,
                 year_tolerance: int = DEFAULT_YEAR_TOLERANCE,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            services: Crossref then OpenAlex service objects, in the order to ask.
            zotero_service: asked first when `prefer_zotero` is true.
        """
        self.services = services
        self.zotero_service = zotero_service
        self.prefer_zotero = prefer_zotero
        self.rule = dict(min_similarity=min_similarity, max_hits=max_hits,
                         year_tolerance=year_tolerance)
        self.log = logger or logging.getLogger("citations.chain")

    def get_canonical_metadata(self, vlm_metadata: Dict[str, Any],
                               title_page_text: str = "") -> Dict[str, Any]:
        """
        Find the accepted record for one document.

        Args:
            vlm_metadata: Prajnya's own reading (title, authors, year, doi).
            title_page_text: the title page's text (rule 6 referee).

        Returns:
            `{record, confidence, source, network, drop_doi}`. `record` is {}
            when nothing was accepted; `drop_doi` is true when a DOI was
            fetched but its record is not the paper on the page.
        """
        answer = self._try_zotero(vlm_metadata)
        if answer:
            return answer

        reachable = {"any_answer": False}
        doi = normalize_doi(vlm_metadata.get("doi"))
        doi_was_wrong = False

        if doi:
            record, source, wrong = self._by_doi(doi, vlm_metadata, title_page_text, reachable)
            doi_was_wrong = wrong
            if record:
                return _answer(record, source, CONFIDENCE_BY_DOI)

        record, source = self._by_title(vlm_metadata, title_page_text, reachable)
        if record:
            return _answer(record, source, CONFIDENCE_BY_TITLE, drop_doi=False)

        return {"record": {}, "confidence": 0.0, "source": "unknown",
                "network": NETWORK_OK if reachable["any_answer"] else NETWORK_UNAVAILABLE,
                "drop_doi": doi_was_wrong}

    # --- steps (after the main function, per CLAUDE.md) ---------------------

    def _try_zotero(self, vlm_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Zotero first, when the researcher prefers it and it knows the paper."""
        if not (self.prefer_zotero and self.zotero_service):
            return None
        try:
            found = self.zotero_service.get_canonical_metadata(vlm_metadata)
        except Exception as error:      # a Zotero failure must not end the chain
            self.log.error("Zotero lookup failed inside the online chain: %s", error)
            return None
        if (found.get("record") or {}).get("title"):
            return {**found, "network": NETWORK_OK, "drop_doi": False}
        return None

    def _by_doi(self, doi: str, extracted: Dict[str, Any], page_text: str,
                reachable: Dict[str, bool]) -> Tuple[Optional[Dict[str, Any]], str, bool]:
        """Each service by DOI. Returns (record, source, a_record_was_wrong)."""
        wrong = False
        for service in self.services:
            status, record = service.by_doi(doi)
            _note(status, reachable)
            if not record:
                continue
            accepted, reason = choose_record([record], extracted, page_text,
                                             by_doi=True, **self.rule)
            if accepted:
                return respell_authors(accepted, extracted), f"{service.name}_doi", False
            self.log.warning("%s DOI %s refused: %s", service.name, doi, reason)
            wrong = True
        return None, "", wrong

    def _by_title(self, extracted: Dict[str, Any], page_text: str,
                  reachable: Dict[str, bool]) -> Tuple[Optional[Dict[str, Any]], str]:
        """Each service by title + first-author surname."""
        title = extracted.get("title") or ""
        if len(title.split()) < MIN_SEARCH_WORDS:
            return None, ""
        authors = extracted.get("authors") or []
        surname = surname_of(authors[0]) if authors else ""
        for service in self.services:
            status, candidates = service.by_title(title, surname)
            _note(status, reachable)
            accepted, reason = choose_record(candidates, extracted, page_text,
                                             by_doi=False, **self.rule)
            if accepted:
                return respell_authors(accepted, extracted), f"{service.name}_title"
            self.log.info("%s title search: %s", service.name, reason)
        return None, ""


def respell_authors(record: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep the title page's spelling of an author the registry also names.

    Registries sometimes hold initials ("G Fullerton"); the title page says
    "Gary D. Fullerton". When the surnames agree, the fuller spelling wins.
    The registry's order and its authors the page did not name are kept.
    A DOI the registry spells in lower case keeps the paper's printed case.
    The registry's separate surnames (`author_names`, contract 1.4) follow:
    the given names take the page's spelling, the surname stays the registry's.
    """
    read = [a for a in (extracted.get("authors") or []) if isinstance(a, str) and a.strip()]
    by_surname = {surname_of(a): clean_name(a) for a in read}
    spelled = []
    for name in record.get("authors") or []:
        page_spelling = by_surname.get(surname_of(name))
        spelled.append(page_spelling if page_spelling and len(page_spelling) >= len(name)
                       else name)
    printed_doi = normalize_doi(extracted.get("doi"))
    doi = record.get("doi")
    if printed_doi and doi and printed_doi.lower() == doi.lower():
        doi = printed_doi       # DOIs are case-blind; keep the form the paper prints
    respelled = {**record, "authors": spelled, "doi": doi}
    parts = record.get("author_names")
    if isinstance(parts, list) and len(parts) == len(spelled):
        respelled["author_names"] = [respelled_parts(part, name)
                                     for part, name in zip(parts, spelled)]
    return respelled


# --- helpers ----------------------------------------------------------------

def respelled_parts(parts: Dict[str, str], spelling: str) -> Dict[str, str]:
    """
    {"given", "family"} for an author now spelled `spelling`.

    When the spelling ends with the registry's surname, everything before it
    is the given names ("Sietske K." + "Rosendahl Huber"). Otherwise the
    registry's parts are kept as they are.
    """
    family = (parts.get("family") or "").strip()
    if family and spelling.lower().endswith(family.lower()):
        return {"given": spelling[:len(spelling) - len(family)].strip(),
                "family": spelling[len(spelling) - len(family):]}
    return {"given": parts.get("given") or "", "family": family}


def _answer(record: Dict[str, Any], source: str, confidence: float,
            drop_doi: bool = False) -> Dict[str, Any]:
    return {"record": record, "confidence": confidence, "source": source,
            "network": NETWORK_OK, "drop_doi": drop_doi}


def _note(status: str, reachable: Dict[str, bool]) -> None:
    """Remember that at least one service actually answered (200 or 404)."""
    if status in (STATUS_OK, STATUS_NOT_FOUND):
        reachable["any_answer"] = True
