#!/usr/bin/env python
"""In-text citation marks, linked to the paper's reference list (stages S3a, S3b).

For biology researchers: in a numbered paper a sentence ends "... was shown
[12,14]"; in an author-year paper it ends "... was shown (Doi et al., 2005)".
This module finds each such mark in the PDF, checks that it names an entry of
the paper's reference list, and later says where the mark sits in the page
text that Prajnya writes out (`content_offset`).

Steps, kept apart on purpose:

    numeric_markers_for   run once, when the reference list is fetched, for a
                          NUMBERED list. Needs the PDF's lines. Offsets are
                          not known yet (None).
    author_year_markers_for  the same for an AUTHOR-YEAR list; the list
                          decides which entry a name and year mean. A marker's
                          `numbers` are always positions in `references`.
    locate_markers        run every time the document is written. Needs only
                          the markers and the page texts, so a document that
                          is rewritten (a correction, a re-cite) gets offsets
                          that match the text it now holds.

A wrong link sends a reader to the wrong paper, so every rule here prefers to
say nothing: no printed numbers, no markers; an ambiguous place, no offset.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..utils.author_year_markers import search_author_year_markers
from ..utils.citation_markers import search_numeric_markers

#: The characters that all count as "a dash" when a marker's text is compared
#: with the page content ("[3-7]" printed with an en dash, stored with a hyphen).
_DASHES = "–—‒-−"

#: How many letters and digits before a marker must agree with its context.
CONTEXT_CHARS = 12

#: The marker kind that needs a check that no digit touches its text.
KIND_SUPERSCRIPT = "superscript"


def numeric_markers_for(lines, local_list, merged: Dict[str, Any]
                        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    The numbered in-text marks of one paper, linked to its reference list.

    Args:
        lines: the document's `LayoutLine`s (`document_layout_lines`).
        local_list: the `ReferenceList` the PDF's own list was read into, or
            None.
        merged: the `merge_reference_lists` result (`reference_list`,
            `references`).

    Returns:
        `(markers, report)`. `markers` is a list of dicts in the contract's
        shape (`content_offset` still None), in reading order. It is empty
        unless the list is numbered, was found, and carries printed numbers.
        `report` says how many were found, in which style, what was dropped
        and why there are none.
    """
    reason = _reason_for_no_markers(lines, local_list, merged)
    linkable = _linkable_numbers(merged) if not reason else set()
    if not reason and not linkable:
        reason = "no printed numbers"
    if reason:
        return [], _report([], {}, False, reason)

    confirmed = (merged["reference_list"] or {}).get("counts_agree") is True
    search = search_numeric_markers(lines, local_list, last=max(linkable),
                                    list_confirmed=confirmed)
    kept = [m for m in search.markers if all(n in linkable for n in m.numbers)]
    dropped = dict(search.dropped)
    dropped["not_in_list"] = len(search.markers) - len(kept)
    markers = [_marker_dict(m) for m in kept]
    return markers, _report(markers, dropped, confirmed, "" if markers else "none found in the text")


def author_year_markers_for(lines, local_list, merged: Dict[str, Any]
                            ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    The author-year in-text citations of one paper, linked to its reference list.

    Args:
        lines: the document's `LayoutLine`s (`document_layout_lines`).
        local_list: the `ReferenceList` the PDF's own list was read into, or
            None.
        merged: the `merge_reference_lists` result (`reference_list`,
            `references`).

    Returns:
        `(markers, report)` in the same shape as `numeric_markers_for`. A
        marker's `numbers` are the `position`s of the references it cites.
        Empty unless the PDF's list is author-year and the merged list was
        found.
    """
    reason = _reason_for_no_author_year(lines, local_list, merged)
    if reason:
        return [], _author_year_report([], {}, False, reason)
    confirmed = (merged["reference_list"] or {}).get("counts_agree") is True
    search = search_author_year_markers(lines, local_list, merged.get("references") or [])
    markers = [_marker_dict(m) for m in search.markers]
    return markers, _author_year_report(markers, search.dropped, confirmed,
                                        "" if markers else "none found in the text")


def locate_markers(markers: Sequence[Dict[str, Any]],
                   page_contents: Mapping[int, str]) -> List[Dict[str, Any]]:
    """
    Copies of the markers with `content_offset` set where it can be trusted.

    The offset is the place of the marker's own text in its page's content
    string. It is set only when ONE place fits both the marker's text and the
    words before it (`context_before`); two equally good places, or none, give
    None. Markers are judged one by one: two identical marks with identical
    context both get None, which is the stated contract.

    Args:
        markers: markers in the contract's shape.
        page_contents: `{1-based page number: that page's content string}`.

    Returns:
        New dicts; the input is not changed. Never raises.
    """
    located = []
    for marker in markers:
        copy = dict(marker)
        try:
            copy["content_offset"] = _offset_of(marker, page_contents)
        except Exception:  # an offset is never worth failing a document for
            copy["content_offset"] = None
        located.append(copy)
    return located


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _reason_for_no_markers(lines, local_list, merged: Dict[str, Any]) -> str:
    """Why this document gets no numeric markers, or '' when it may."""
    if not lines:
        return "no list read from the PDF"
    if local_list is None:
        return "no list read from the PDF"
    if getattr(local_list, "style", None) != "numbered":
        return "author-year list"
    block = (merged or {}).get("reference_list")
    if not isinstance(block, dict) or block.get("status") != "found":
        return "no reference list found"
    return ""


def _reason_for_no_author_year(lines, local_list, merged: Dict[str, Any]) -> str:
    """Why this document gets no author-year markers, or '' when it may."""
    if not lines or local_list is None:
        return "no list read from the PDF"
    if getattr(local_list, "style", None) != "author_year":
        return "numbered list"
    block = (merged or {}).get("reference_list")
    if not isinstance(block, dict) or block.get("status") != "found":
        return "no reference list found"
    return ""


def _author_year_report(markers: List[Dict[str, Any]], dropped: Dict[str, int],
                        confirmed: bool, reason: str) -> Dict[str, Any]:
    """The report written next to author-year markers."""
    counts = {"no_candidate": 0, "ambiguous": 0, "letter": 0}
    counts.update(dropped)
    return {"found": len(markers),
            "style": markers[0]["kind"] if markers else None,
            "dropped": counts,
            "list_confirmed": confirmed,
            "reason": reason}


def _linkable_numbers(merged: Dict[str, Any]) -> Set[int]:
    """
    The printed numbers a marker may point to.

    A reference counts only when its printed number equals its position in
    the list, so that reading a marker by number or by position gives the
    same reference.
    """
    linkable: Set[int] = set()
    for reference in merged.get("references") or []:
        number = reference.get("number")
        if number is not None and number == reference.get("position"):
            linkable.add(number)
    return linkable


def _marker_dict(marker) -> Dict[str, Any]:
    """One found marker in the shape the contract fixes (offset not yet known)."""
    return {"page_number": marker.page, "text": marker.text, "kind": marker.kind,
            "numbers": list(marker.numbers), "context_before": marker.context,
            "content_offset": None}


def _report(markers: List[Dict[str, Any]], dropped: Dict[str, int],
            confirmed: bool, reason: str) -> Dict[str, Any]:
    """The report written next to the markers."""
    counts = {"beyond_list": 0, "order_check": 0, "author_line": 0, "not_in_list": 0}
    counts.update(dropped)
    return {"found": len(markers),
            "style": markers[0]["kind"] if markers else None,
            "dropped": counts,
            "list_confirmed": confirmed,
            "reason": reason}


def _offset_of(marker: Mapping[str, Any], page_contents: Mapping[int, str]) -> Optional[int]:
    """The one trustworthy offset of a marker in its page, or None."""
    content = page_contents.get(marker.get("page_number"))
    text = marker.get("text")
    if not isinstance(content, str) or not isinstance(text, str) or not text:
        return None
    wanted = _folded(marker.get("context_before") or "")[-CONTEXT_CHARS:]
    agreeing = [i for i in _candidates(content, text, marker.get("kind"))
                if _folded(content[:i]).endswith(wanted)]
    return agreeing[0] if len(agreeing) == 1 else None


def _candidates(content: str, text: str, kind: Optional[str]) -> List[int]:
    """Every place where the marker's text stands in the content."""
    found = []
    for i in range(len(content) - len(text) + 1):
        if not _same_text(content[i:i + len(text)], text):
            continue
        if kind == KIND_SUPERSCRIPT and _touches_digit(content, i, i + len(text)):
            continue
        found.append(i)
    return found


def _same_text(piece: str, text: str) -> bool:
    """Equal character for character, any dash counting as any other dash."""
    return all(a == b or (a in _DASHES and b in _DASHES) for a, b in zip(piece, text))


def _touches_digit(content: str, start: int, end: int) -> bool:
    """Whether a digit stands right before or right after the span."""
    before = content[start - 1] if start > 0 else ""
    after = content[end] if end < len(content) else ""
    return before.isdigit() or after.isdigit()


def _folded(text: str) -> str:
    """Only the letters and digits, lower-cased (hyphens, spaces, marks go)."""
    return "".join(c for c in text if c.isalnum()).lower()
