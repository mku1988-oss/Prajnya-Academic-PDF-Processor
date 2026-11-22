#!/usr/bin/env python
"""Crossref: the publishers' own registry of DOIs.

For biology researchers: when a publisher issues a DOI it deposits the
paper's official title, authors, journal and year with Crossref. That is the
closest thing to ground truth for a journal article.

Two questions are asked: "what is the record for this DOI?" and "which
records look like this title and first author?". Both return CANDIDATES in
one canonical shape; `acceptance.py` decides which, if any, is the paper.
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional

from ..utils.reference_list import entry_year, first_author
from .http import STATUS_OK, CitationHttp

DEFAULT_HOST = "https://api.crossref.org"

#: Initials standing beside a surname in a deposited reference: "PA", "P.A.", "J.-P.".
_INITIALS_TOKEN = re.compile(r"^[A-Z](?:\.?-?[A-Z]){0,2}\.?$")


class CrossrefService:
    """Crossref lookups by DOI and by title."""

    name = "crossref"

    def __init__(self, http: CitationHttp, host: str = DEFAULT_HOST, rows: int = 5):
        self.http = http
        self.host = host.rstrip("/")
        self.rows = rows

    def by_doi(self, doi: str):
        """Returns `(status, record or None)`; status is the HTTP layer's."""
        status, body = self.http.get_json(f"{self.host}/works/{doi}")
        if status != STATUS_OK or not isinstance(body, dict):
            return status, None
        return status, canonical_record((body.get("message") or {}))

    def deposited_references(self, doi: str):
        """
        The reference list the publisher deposited for this DOI.

        Uses the same URL as `by_doi`, so the cached answer is shared.

        Returns:
            `(status, [entries])` in deposited order; each entry is the dict
            built by `deposited_reference`. Empty when the publisher
            deposited no list (many do not).
        """
        status, body = self.http.get_json(f"{self.host}/works/{doi}")
        if status != STATUS_OK or not isinstance(body, dict):
            return status, []
        items = (body.get("message") or {}).get("reference") or []
        return status, [deposited_reference(item) for item in items if isinstance(item, dict)]

    def by_title(self, title: str, first_author_surname: str = ""):
        """Returns `(status, [records])`: up to `rows` candidates, best first."""
        params = {"query.bibliographic": title, "rows": self.rows}
        if first_author_surname:
            params["query.author"] = first_author_surname
        status, body = self.http.get_json(f"{self.host}/works", params)
        if status != STATUS_OK or not isinstance(body, dict):
            return status, []
        items = (body.get("message") or {}).get("items") or []
        return status, [canonical_record(item) for item in items]


def canonical_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    A Crossref work as the record shape the merger consumes.

    Derived from the raw answer every time it is read (it is kept in
    `payload`), so a corrected mapping is never hidden behind a stored one.
    """
    return {
        "title": _first(item.get("title")),
        "authors": [n for n in (_person(a) for a in item.get("author") or []) if n],
        # The same people with the surname kept apart (contract 1.4): a
        # compound surname ("Rosendahl Huber", "García Martínez de Artola")
        # cannot be recovered from the joined string by any rule.
        "author_names": [_name_parts(a) for a in item.get("author") or [] if _person(a)],
        "doi": item.get("DOI"),
        "journal": _first(item.get("container-title")),
        "year": _year(item),
        "publisher": html.unescape(item["publisher"]) if item.get("publisher") else None,
        "type": item.get("type"),
        "payload": item,
    }


def deposited_reference(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    One entry of a deposited reference list, in the shape the merger consumes.

    Crossref's `author` field of a reference is already the first author's
    surname. When author or year is missing but the publisher deposited the
    printed text (`unstructured`), they are read from that text.

    Returns:
        Dict with `first_author, year, title, container, doi, volume,
        first_page, deposited_text`; missing values are None.
    """
    text = _clean(item.get("unstructured"))
    author = _surname_only(_clean(item.get("author")))
    year = _reference_year(item.get("year"))
    if text and not author:
        author = _clean(first_author(text))
    if text and year is None:
        year = _reference_year(entry_year(text))
    return {
        "first_author": author,
        "year": year,
        "title": _clean(item.get("article-title") or item.get("volume-title"), unescape=True),
        "container": _clean(item.get("journal-title") or item.get("series-title"), unescape=True),
        "doi": _clean(item.get("DOI")),
        "volume": _clean(item.get("volume")),
        "first_page": _clean(item.get("first-page")),
        "deposited_text": text,
    }


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _surname_only(author: Optional[str]) -> Optional[str]:
    """
    A deposited first author without initials: "PA Tsonis" -> "Tsonis".

    Most publishers deposit the surname alone; some add the initials before
    or after it. A lone word is never touched ("AMANO" stays).
    """
    words = (author or "").split()
    kept = [word for word in words if not _INITIALS_TOKEN.match(word)]
    if len(words) < 2 or not kept:
        return author
    return " ".join(kept)


def _clean(value: Any, unescape: bool = False) -> Optional[str]:
    """A stripped string, or None when empty or not a string."""
    if not isinstance(value, str) or not value.strip():
        return None
    return html.unescape(value).strip() if unescape else value.strip()


def _reference_year(value: Any) -> Optional[int]:
    """The first 4-digit year 18xx/19xx/20xx in `value` ("1981a" gives 1981), or None."""
    match = re.search(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)", str(value or ""))
    return int(match.group(0)) if match else None

def _first(value: Any) -> Optional[str]:
    if isinstance(value, list):
        value = value[0] if value else None
    return html.unescape(value).strip() if isinstance(value, str) and value.strip() else None


def _year(item: Dict[str, Any]) -> Optional[int]:
    """Print year, else the earliest `issued` year (plan C4, merge rule)."""
    for key in ("published-print", "issued", "published-online"):
        parts = ((item.get(key) or {}).get("date-parts") or [[None]])[0]
        if parts and isinstance(parts[0], int):
            return parts[0]
    return None


def _name_parts(author: Dict[str, Any]) -> Dict[str, str]:
    """{"given", "family"} of one Crossref author, spelled as `_person` spells them."""
    person = _person(author)
    given = (author.get("given") or "").strip()
    return {"given": given, "family": person[len(given):].strip()}


def _person(author: Dict[str, Any]) -> str:
    """
    "Given Family". Some deposits write names in capitals ("G FULLERTON"):
    the family name is brought to normal case here; the title page's fuller
    spelling is restored later by `chain.respell_authors`.
    """
    given = (author.get("given") or "").strip()
    family = (author.get("family") or author.get("name") or "").strip()
    if family.isupper() and len(family) > 1:
        family = family.title()
    return f"{given} {family}".strip()
