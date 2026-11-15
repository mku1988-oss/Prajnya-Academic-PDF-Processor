#!/usr/bin/env python
"""OpenAlex: a free index of scholarly works, the second opinion.

For biology researchers: OpenAlex covers papers Crossref does not describe
well (some books, preprints, older journals). It is asked only after
Crossref has not produced an accepted record. The same `acceptance.py` rule
decides whether its answer is believed.

An API key is optional. When `OPENALEX_API_KEY` is set (in `.env`) it is
sent; with none, the free tier is used.
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional

from .http import STATUS_NOT_FOUND, STATUS_OK, STATUS_UNREACHABLE, CitationHttp

DEFAULT_HOST = "https://api.openalex.org"

#: OpenAlex accepts at most 50 ids or DOIs in one "a|b|c" filter.
BATCH = 50
_SELECT = "id,doi,title,publication_year,authorships,primary_location,biblio"

# Lower-case words that belong in front of a surname ("van der Flier").
_NAME_PARTICLES = {"van", "von", "der", "den", "de", "del", "della", "di", "da",
                   "dos", "du", "la", "le", "ten", "ter"}


class OpenAlexService:
    """OpenAlex lookups by DOI and by title."""

    name = "openalex"

    def __init__(self, http: CitationHttp, host: str = DEFAULT_HOST,
                 api_key: Optional[str] = None, rows: int = 5):
        self.http = http
        self.host = host.rstrip("/")
        self.api_key = api_key or None
        self.rows = rows

    def by_doi(self, doi: str):
        status, body = self.http.get_json(f"{self.host}/works/https://doi.org/{doi}",
                                          self._params())
        if status != STATUS_OK or not isinstance(body, dict):
            return status, None
        return status, canonical_record(body)

    def by_title(self, title: str, first_author_surname: str = ""):
        params = self._params({"search": title, "per-page": self.rows})
        status, body = self.http.get_json(f"{self.host}/works", params)
        if status != STATUS_OK or not isinstance(body, dict):
            return status, []
        return status, [canonical_record(w) for w in body.get("results") or []]

    def works_by_doi(self, dois):
        """
        Many works at once, looked up by DOI (50 per request).

        Returns:
            `(status, {lower-case bare doi: cited_work dict})`. Status is
            unreachable if any batch got no answer, else ok.
        """
        found: Dict[str, Dict[str, Any]] = {}
        status = STATUS_OK
        for batch in _batches(list(dois)):
            params = self._params({"filter": "doi:" + "|".join(batch),
                                   "per-page": BATCH, "select": _SELECT})
            batch_status, body = self.http.get_json(f"{self.host}/works", params)
            if batch_status == STATUS_UNREACHABLE:
                status = STATUS_UNREACHABLE
            if batch_status != STATUS_OK or not isinstance(body, dict):
                continue
            for work in body.get("results") or []:
                entry = cited_work(work)
                if entry["doi"]:
                    found[entry["doi"].lower()] = entry
        return status, found

    def cited_works(self, doi: str):
        """
        The works this paper cites, as OpenAlex lists them (`referenced_works`).

        The order is NOT the printed order.

        Returns:
            `(status, [cited_work dicts])`; 404 on the paper gives
            `(STATUS_NOT_FOUND, [])`.
        """
        status, body = self.http.get_json(f"{self.host}/works/https://doi.org/{doi}",
                                          self._params())
        if status == STATUS_NOT_FOUND:
            return STATUS_NOT_FOUND, []
        if status != STATUS_OK or not isinstance(body, dict):
            return status, []
        ids = [url.rsplit("/", 1)[-1] for url in body.get("referenced_works") or []
               if isinstance(url, str) and url]
        works: List[Dict[str, Any]] = []
        for batch in _batches(ids):
            params = self._params({"filter": "openalex:" + "|".join(batch),
                                   "per-page": BATCH, "select": _SELECT})
            batch_status, answer = self.http.get_json(f"{self.host}/works", params)
            if batch_status == STATUS_UNREACHABLE:
                status = STATUS_UNREACHABLE
            if batch_status == STATUS_OK and isinstance(answer, dict):
                works.extend(cited_work(w) for w in answer.get("results") or [])
        return status, works

    def _params(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(extra or {})
        if self.api_key:
            params["api_key"] = self.api_key
        return params


def canonical_record(work: Dict[str, Any]) -> Dict[str, Any]:
    """An OpenAlex work as the record shape the merger consumes."""
    source = ((work.get("primary_location") or {}).get("source") or {})
    return {
        "title": _unescaped(work.get("display_name") or work.get("title")),
        "authors": [a["author"]["display_name"] for a in work.get("authorships") or []
                    if isinstance(a.get("author"), dict) and a["author"].get("display_name")],
        "doi": _bare_doi(work.get("doi")),
        "journal": _unescaped(source.get("display_name")),
        "year": work.get("publication_year"),
        "publisher": source.get("host_organization_name"),
        "type": work.get("type"),
        "payload": work,
    }


def cited_work(work: Dict[str, Any]) -> Dict[str, Any]:
    """
    A work cited by a paper, in the same shape as Crossref's `deposited_reference`.

    The first author's surname keeps the case it is printed in (OpenAlex
    gives a display name such as "Annette van der Flier").
    """
    authorships = work.get("authorships") or []
    first = ((authorships[0].get("author") or {}).get("display_name")) if authorships else None
    source = ((work.get("primary_location") or {}).get("source") or {})
    biblio = work.get("biblio") or {}
    year = work.get("publication_year")
    return {
        "first_author": printed_surname(first),
        "year": year if isinstance(year, int) else None,
        "title": _unescaped(work.get("title") or work.get("display_name")),
        "container": _unescaped(source.get("display_name")),
        "doi": _bare_doi(work.get("doi")),
        "volume": biblio.get("volume") or None,
        "first_page": biblio.get("first_page") or None,
        "deposited_text": None,
    }


def printed_surname(name: Any) -> Optional[str]:
    """"Annette van der Flier" gives "van der Flier"; "Jordi Bella" gives "Bella"; None if empty."""
    if not isinstance(name, str) or not name.strip():
        return None
    words = name.split()
    surname = [words.pop()]
    while words and words[-1] in _NAME_PARTICLES:
        surname.insert(0, words.pop())
    return " ".join(surname)


def _batches(items: List[str]) -> List[List[str]]:
    """`items` cut into lists of at most BATCH."""
    return [items[i:i + BATCH] for i in range(0, len(items), BATCH)]


def _unescaped(value: Any) -> Optional[str]:
    """Registry text with HTML entities (&amp;) turned into the characters."""
    return html.unescape(value) if isinstance(value, str) and value else None


def _bare_doi(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", value.strip(), flags=re.I)
