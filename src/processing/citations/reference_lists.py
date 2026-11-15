#!/usr/bin/env python
"""Finds a paper's reference list at citation level `online` (stage S2).

For biology researchers: this asks Crossref for the bibliography the
publisher deposited, asks OpenAlex to fill in the entries that came as bare
DOIs, reads the list printed in the PDF, and hands the three to
`reference_merge.py`. Nothing here decides how they combine; it only fetches.

A reference list is never worth failing a document for: any problem reading
the PDF is logged and the online answer is used alone.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..pdf import get_backend
from ..utils.layout_lines import LayoutLine, document_layout_lines
from ..utils.reference_list import ReferenceList, find_reference_list
from .http import STATUS_UNREACHABLE
from .markers import author_year_markers_for, numeric_markers_for
from .openalex import printed_surname
from .reference_merge import merge_reference_lists

_FILLABLE_FIELDS = ("first_author", "year", "title")

#: At most this many DOI-only entries per paper are looked up one at a time.
MAX_SINGLE_LOOKUPS = 10


class ReferenceListFinder:
    """Fetches the three readings of a reference list and merges them."""

    def __init__(self, crossref, openalex, logger: Optional[logging.Logger] = None,
                 local_reader: Optional[Callable[[Optional[Path]], Optional[ReferenceList]]] = None,
                 line_reader: Optional[Callable[[Optional[Path]], Optional[List[LayoutLine]]]] = None):
        self.crossref = crossref
        self.openalex = openalex
        self.log = logger or logging.getLogger("citations")
        self.local_reader = local_reader
        self.line_reader = line_reader or read_document_lines

    def find(self, doi: Optional[str], pdf_path: Optional[Path]) -> Dict[str, Any]:
        """
        The merged reference list for one paper.

        Args:
            doi: the paper's bare DOI, or None (then only the PDF is read).
            pdf_path: the PDF, or None.

        Returns:
            The `merge_reference_lists` result (`reference_list`,
            `references`, `report`) plus `markers`: the in-text
            marks (numbered, or author-year) linked to the list (their `content_offset` is still None, the block fills it in).
            Status is "not_reached" when nothing was found and a service
            could not be contacted.
        """
        statuses: List[str] = []
        crossref_refs: List[Dict[str, Any]] = []
        openalex_works: List[Dict[str, Any]] = []
        if doi:
            crossref_refs = self._deposited(doi, statuses)
            if crossref_refs:
                self._fill_from_openalex(crossref_refs, statuses)
                self._fill_from_crossref(crossref_refs, statuses)
            else:
                openalex_works = self._cited(doi, statuses)
        lines = self.line_reader(pdf_path)
        local_list = self._local_list(lines, pdf_path)
        result = merge_reference_lists(crossref_refs, openalex_works, local_list)
        if (result["reference_list"]["status"] == "not_available"
                and STATUS_UNREACHABLE in statuses):
            result["reference_list"]["status"] = "not_reached"
        self._add_markers(result, lines, local_list)
        self._log_result(result)
        return result

    # --- fetching helpers (after the main method) ------------------------------

    def _local_list(self, lines: Optional[List[LayoutLine]],
                    pdf_path: Optional[Path]) -> Optional[ReferenceList]:
        """The list printed in the PDF: from a caller's reader, else from the lines."""
        if self.local_reader is not None:
            return self.local_reader(pdf_path)
        return list_from_lines(lines)

    def _add_markers(self, result: Dict[str, Any], lines, local_list) -> None:
        """
        Add `markers` and `report["markers"]` to the result.

        A mark is never worth failing a document for: any error is logged and
        the document simply gets no markers.
        """
        try:
            if getattr(local_list, "style", None) == "author_year":
                markers, report = author_year_markers_for(lines, local_list, result)
            else:
                markers, report = numeric_markers_for(lines, local_list, result)
        except Exception as error:
            self.log.warning("Could not link the in-text markers: %s", error)
            markers = []
            report = {"found": 0, "style": None,
                      "dropped": {"beyond_list": 0, "order_check": 0, "author_line": 0, "not_in_list": 0},
                      "list_confirmed": False, "reason": "error while linking"}
        result["markers"] = markers
        result["report"]["markers"] = report

    def _deposited(self, doi: str, statuses: List[str]) -> List[Dict[str, Any]]:
        status, entries = self.crossref.deposited_references(doi)
        statuses.append(status)
        return list(entries)

    def _cited(self, doi: str, statuses: List[str]) -> List[Dict[str, Any]]:
        status, works = self.openalex.cited_works(doi)
        statuses.append(status)
        return list(works)

    def _fill_from_openalex(self, entries: List[Dict[str, Any]], statuses: List[str]) -> None:
        """Fill ONLY the missing author, year or title of entries that carry a DOI."""
        wanted = [e["doi"] for e in entries
                  if e.get("doi") and any(not e.get(f) for f in _FILLABLE_FIELDS)]
        if not wanted:
            return
        status, found = self.openalex.works_by_doi(sorted(set(wanted)))
        statuses.append(status)
        for entry in entries:
            work = found.get((entry.get("doi") or "").lower())
            if work:
                for field in _FILLABLE_FIELDS:
                    if not entry.get(field) and work.get(field):
                        entry[field] = work[field]

    def _fill_from_crossref(self, entries: List[Dict[str, Any]], statuses: List[str]) -> None:
        """
        Identify the few DOI-only entries OpenAlex did not know, one by one.

        An entry with a DOI and nothing else cannot be lined up with the
        printed list, so the same reference would stand in the list twice
        (seen on a fresh paper: two such entries of 83). Crossref issued the
        DOI, so it can name the work. Capped, to keep a paper to a few
        requests.
        """
        unknown = [e for e in entries
                   if e.get("doi") and not e.get("first_author") and not e.get("year")]
        for entry in unknown[:MAX_SINGLE_LOOKUPS]:
            status, record = self.crossref.by_doi(entry["doi"])
            statuses.append(status)
            if not record:
                continue
            authors = record.get("authors") or []
            entry["first_author"] = printed_surname(authors[0]) if authors else None
            entry["year"] = record.get("year")
            entry["title"] = entry.get("title") or record.get("title")
            entry["container"] = entry.get("container") or record.get("journal")

    def _log_result(self, result: Dict[str, Any]) -> None:
        block = result["reference_list"]
        self.log.info("Reference list: %s (source %s, %s entries, %s read from the PDF, "
                      "case %s, %s in-text markers).",
                      block["status"], block.get("source", "-"), block["count"],
                      block["local_count"], result["report"]["case"],
                      len(result.get("markers") or []))


def read_local_reference_list(pdf_path: Optional[Path]) -> Optional[ReferenceList]:
    """
    The reference list printed in the PDF, or None.

    Returns None when there is no file, no list is found, or anything goes
    wrong (logged as a warning).
    """
    return list_from_lines(read_document_lines(pdf_path))


def read_document_lines(pdf_path: Optional[Path]) -> Optional[List[LayoutLine]]:
    """
    Every line of the PDF in reading order, or None.

    The lines are read once and serve both the reference list and the
    in-text markers. Returns None when there is no file or anything goes
    wrong (logged as a warning).
    """
    if pdf_path is None or not Path(pdf_path).is_file():
        return None
    try:
        with get_backend().open(Path(pdf_path)) as pdf:
            pages = list(pdf.pages())
            return document_layout_lines((page.characters() for page in pages),
                                         [page.size[1] for page in pages])
    except Exception as error:  # a list is never worth failing a document for
        logging.getLogger("citations").warning(
            "Could not read the reference list from %s: %s", pdf_path, error)
        return None


def list_from_lines(lines: Optional[List[LayoutLine]]) -> Optional[ReferenceList]:
    """The reference list in those lines, or None (also when the search fails)."""
    if not lines:
        return None
    try:
        return find_reference_list(lines)
    except Exception as error:  # a list is never worth failing a document for
        logging.getLogger("citations").warning(
            "Could not read the reference list: %s", error)
        return None
