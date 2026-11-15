#!/usr/bin/env python
"""Merges the three readings of a paper's reference list (stage S2).

For biology researchers: the bibliography of a paper can be read three ways.
Crossref holds the list the publisher deposited (in printed order, but often
with gaps and bare DOIs). OpenAlex holds the works the paper cites (complete
but in no particular order). Prajnya's own splitter reads the list printed in
the PDF (in order, with the printed numbers, but with only author and year).
This module combines them into ONE list in printed order, and says how far
the readings agree. It is pure: no network, no PDF, so every rule here can be
tested on its own.

The rules, in short:
  * Two entries "agree" when their first authors and years match.
  * Crossref and the local list are lined up on the entries that agree (a
    longest common subsequence). If too few agree, the local reading is not
    trusted to be the same list and Crossref stands alone.
  * Printed numbers are handed out only when the alignment is nearly
    complete, because a wrong number is worse than none.
  * With no Crossref list, the local list is the spine and OpenAlex works are
    attached to its entries only on evidence found in the printed text.
  * OpenAlex alone cannot give the printed order, so it gives no list.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..utils.reference_list import ReferenceList
from .acceptance import fold

#: Numbers are handed out only when at least this share of the shorter list
#: (deposited or printed) was paired.
MIN_ALIGNED_SHARE = 0.8
#: The local reading counts as "the same list" only when the anchors reach
#: this share of the shorter list.
MIN_SAME_LIST_SHARE = 0.5
#: An OpenAlex work that agrees on author and year is attached when this
#: share of its title words is printed in the entry.
TITLE_SHARE_WITH_AUTHOR = 0.6
#: Without any author/year agreement, a long title must match this closely.
TITLE_SHARE_ALONE = 0.9
#: ...and "long" means at least this many words of 3+ letters.
MIN_TITLE_WORDS_ALONE = 4

CASE_CROSSREF = "crossref"
CASE_LOCAL_SPINE = "local_spine"
CASE_LOCAL_ONLY = "local_only"
CASE_OPENALEX_UNORDERED = "openalex_unordered"
CASE_NONE = "none"


def merge_reference_lists(crossref_refs: Sequence[Dict[str, Any]],
                          openalex_works: Sequence[Dict[str, Any]],
                          local_list: Optional[ReferenceList]) -> Dict[str, Any]:
    """
    Combine the readings into the contract's `reference_list` + `references`.

    Args:
        crossref_refs: deposited entries in deposited order (may be empty),
            each a dict from `crossref.deposited_reference`.
        openalex_works: cited works in the same shape (unordered, may be empty).
        local_list: the list read from the PDF, or None.

    Returns:
        `{"reference_list": {...}, "references": [...], "report": {...}}`.
        `report` is for evaluation only and is not written to the document.
    """
    local = _local_entries(local_list)
    style = local_list.style if local_list is not None else None
    if crossref_refs:
        return _merge_with_crossref(list(crossref_refs), local, style)
    if local:
        attach = list(openalex_works)
        return _merge_local_spine(local, style, attach)
    if openalex_works:
        return _not_available(CASE_OPENALEX_UNORDERED, online_only=len(openalex_works))
    return _not_available(CASE_NONE)


# --- case: Crossref present ---------------------------------------------------

def _merge_with_crossref(crossref: List[Dict[str, Any]], local: List[Dict[str, Any]],
                         style: Optional[str]) -> Dict[str, Any]:
    """Crossref alone, or aligned with the local list when that is the same list."""
    report = _new_report(CASE_CROSSREF)
    if not local:
        return _crossref_alone(crossref, None, None, report)
    anchors = _anchor_pairs(crossref, local)
    report["anchors"] = len(anchors)
    if len(anchors) < MIN_SAME_LIST_SHARE * min(len(crossref), len(local)):
        report["conflicts"].append(
            f"local list not trusted: {len(anchors)} anchors for "
            f"{len(crossref)} deposited and {len(local)} printed entries")
        return _crossref_alone(crossref, len(local), style, report)

    steps = _alignment(crossref, local, anchors, report)
    report["paired"] = sum(1 for step in steps if step[0] == "pair")
    # Measured against the SHORTER list: a registry that holds only part of
    # the list (83 of 98 on one fresh paper) is still the same list, and the
    # numbers come from the printed entries themselves.
    numbered = (style == "numbered"
                and report["paired"] >= MIN_ALIGNED_SHARE * min(len(crossref), len(local)))
    references = [_reference_from_step(step, crossref, local, numbered) for step in steps]
    report["local_only_entries"] = sum(1 for step in steps if step[0] == "local")
    report["online_only_entries"] = sum(1 for step in steps if step[0] == "online")
    return _result(references, "crossref", style, len(local), len(crossref) == len(local), report)


def _crossref_alone(crossref: List[Dict[str, Any]], local_count: Optional[int],
                    style: Optional[str], report: Dict[str, Any]) -> Dict[str, Any]:
    """Crossref's entries in order, with no printed numbers or printed text."""
    references = [_make_reference(pos, None, entry, "crossref", None)
                  for pos, entry in enumerate(crossref, start=1)]
    agree = None if local_count is None else len(crossref) == local_count
    return _result(references, "crossref", style, local_count, agree, report)


def _alignment(crossref: List[Dict[str, Any]], local: List[Dict[str, Any]],
               anchors: List[Tuple[int, int]], report: Dict[str, Any]) -> List[Tuple[Any, ...]]:
    """
    Walk the anchors and the gaps between them into a list of steps:
    ("pair", c, l), ("online", c) or ("local", l).
    """
    steps: List[Tuple[Any, ...]] = []
    boundaries = [(-1, -1)] + list(anchors) + [(len(crossref), len(local))]
    for (c_before, l_before), (c_after, l_after) in zip(boundaries, boundaries[1:]):
        c_gap = list(range(c_before + 1, c_after))
        l_gap = list(range(l_before + 1, l_after))
        if len(c_gap) == len(l_gap):
            for c_index, l_index in zip(c_gap, l_gap):
                steps.append(("pair", c_index, l_index))
                _note_conflict(crossref[c_index], local[l_index], report)
        else:
            steps.extend(("online", c_index) for c_index in c_gap)
            steps.extend(("local", l_index) for l_index in l_gap)
        if c_after < len(crossref):
            steps.append(("pair", c_after, l_after))
    return steps


def _note_conflict(online: Dict[str, Any], printed: Dict[str, Any], report: Dict[str, Any]) -> None:
    """Record a paired gap entry whose author and year both exist but differ."""
    both_known = (online.get("first_author") and online.get("year")
                  and printed.get("first_author") and printed.get("year"))
    if both_known and not entries_agree(online, printed):
        report["conflicts"].append(
            f"paired by position, but deposited {online['first_author']} {online['year']} "
            f"differs from printed {printed['first_author']} {printed['year']}")


def _reference_from_step(step: Tuple[Any, ...], crossref: List[Dict[str, Any]],
                         local: List[Dict[str, Any]], numbered: bool) -> Dict[str, Any]:
    """One output reference (position filled in later by `_renumber`)."""
    kind = step[0]
    if kind == "pair":
        online, printed = crossref[step[1]], local[step[2]]
        merged = dict(online)
        merged["first_author"] = online.get("first_author") or printed.get("first_author")
        merged["year"] = _paired_year(online, printed)
        number = printed["number"] if numbered else None
        return _make_reference(0, number, merged, "crossref", printed["text"])
    if kind == "online":
        return _make_reference(0, None, crossref[step[1]], "crossref", None)
    printed = local[step[1]]
    number = printed["number"] if numbered else None
    return _make_reference(0, number, printed, "local", printed["text"], printed_only=True)


def _paired_year(online: Dict[str, Any], printed: Dict[str, Any]) -> Optional[int]:
    """
    The year of a paired entry.

    A registry often holds the year a paper went ONLINE, the printed entry
    the year of the PRINT issue: one apart, same work. The list is the
    paper's own, so when the surnames agree and the years are one apart the
    printed year is kept. Any larger difference keeps the registry's year
    (and is reported as a conflict).
    """
    online_year, printed_year = _year_int(online.get("year")), _year_int(printed.get("year"))
    if online_year is None or printed_year is None:
        return online_year or printed_year
    same_author = entries_agree({**online, "year": printed_year}, printed)
    if same_author and abs(online_year - printed_year) == 1:
        return printed_year
    return online_year


# --- case: local list is the spine --------------------------------------------

def _merge_local_spine(local: List[Dict[str, Any]], style: Optional[str],
                       works: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The printed entries in order, each with at most one OpenAlex work attached."""
    report = _new_report(CASE_LOCAL_SPINE if works else CASE_LOCAL_ONLY)
    unused = list(range(len(works)))
    references = []
    for position, entry in enumerate(local, start=1):
        chosen = _best_work(entry, works, unused)
        number = entry["number"] if style == "numbered" else None
        if chosen is None:
            references.append(_make_reference(position, number, entry, "local",
                                              entry["text"], printed_only=True))
            continue
        unused.remove(chosen)
        work = works[chosen]
        merged = {**entry, "doi": work.get("doi"), "title": work.get("title"),
                  "container": work.get("container")}
        references.append(_make_reference(position, number, merged, "openalex", entry["text"]))
        report["attached"] += 1
    report["online_only_entries"] = len(unused)
    return _result(references, "local", style, len(local), True, report)


def _best_work(entry: Dict[str, Any], works: List[Dict[str, Any]],
               unused: List[int]) -> Optional[int]:
    """Index of the one work to attach to this entry, or None (also on a tie)."""
    scored = []
    for index in unused:
        share = _title_share(works[index], entry["text"])
        if _attachable(entry, works[index], share):
            scored.append((share, index))
    if not scored:
        return None
    scored.sort(key=lambda pair: -pair[0])
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _attachable(entry: Dict[str, Any], work: Dict[str, Any], share: float) -> bool:
    """Rule (a): agrees on author+year and has evidence in the text; rule (b): long title match."""
    if entries_agree(entry, work):
        if share >= TITLE_SHARE_WITH_AUTHOR or _volume_and_page_printed(work, entry["text"]):
            return True
    return len(_title_words(work)) >= MIN_TITLE_WORDS_ALONE and share >= TITLE_SHARE_ALONE


def _title_words(work: Dict[str, Any]) -> List[str]:
    """The folded words of the work's title that have 3 or more letters."""
    return [word for word in fold(work.get("title")).split() if len(word) >= 3]


def _title_share(work: Dict[str, Any], printed_text: str) -> float:
    """Share of the work's title words found among the words of the printed entry."""
    words = _title_words(work)
    if not words:
        return 0.0
    printed = set(fold(printed_text).split())
    return sum(1 for word in words if word in printed) / len(words)


def _volume_and_page_printed(work: Dict[str, Any], printed_text: str) -> bool:
    """Whether the work's volume and first page both occur as whole tokens in the entry."""
    volume, page = fold(work.get("volume")), fold(work.get("first_page"))
    if not volume or not page:
        return False
    printed = set(fold(printed_text).split())
    return volume in printed and page in printed


# --- the agreement rule and the alignment ----------------------------------------

def entries_agree(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    """
    Same first author (accents and case folded; "van der flier" matches
    "flier") and the same year's first four digits. Both must be known.
    """
    a, b = fold(left.get("first_author")), fold(right.get("first_author"))
    year_a, year_b = _year_int(left.get("year")), _year_int(right.get("year"))
    if not a or not b or year_a is None or year_b is None or year_a != year_b:
        return False
    return a == b or set(a.split()) >= set(b.split()) or set(b.split()) >= set(a.split())


def _anchor_pairs(crossref: List[Dict[str, Any]], local: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Longest common subsequence of the two lists under `_same_entry`, as (c, l) index pairs."""
    rows, cols = len(crossref), len(local)
    same = _same_entry_table(crossref, local)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            if same[i][j]:
                table[i][j] = table[i + 1][j + 1] + 1
            else:
                table[i][j] = max(table[i + 1][j], table[i][j + 1])
    return _trace_back(table, same)


def _same_entry_table(crossref: List[Dict[str, Any]],
                      local: List[Dict[str, Any]]) -> List[List[bool]]:
    """
    For every (deposited, printed) pair: are they the same reference?

    Worked out once, with each entry's words folded once, because the
    alignment asks the question for every pair of a list of several hundred.
    """
    titles = [_title_words(entry) for entry in crossref]
    printed_words = [set(fold(entry.get("text")).split()) for entry in local]
    return [[_same_entry(online, printed, titles[i], printed_words[j])
             for j, printed in enumerate(local)]
            for i, online in enumerate(crossref)]


def _same_entry(online: Dict[str, Any], printed: Dict[str, Any],
                title_words: List[str], printed_words: set) -> bool:
    """
    Whether a deposited entry and a printed entry are the same reference.

    Author and year agreeing is the usual case. Two more cases were found on
    fresh papers, where the strict rule left the same reference in the list
    twice (once from each reading):
      * the registry holds the ONLINE year and the paper prints the PRINT
        year, one apart: same surname, years one apart;
      * the years differ by more, or the splitter misread the author, but
        the deposited TITLE is printed in the entry.
    """
    if entries_agree(online, printed):
        return True
    share = (sum(1 for word in title_words if word in printed_words) / len(title_words)
             if title_words else 0.0)
    if len(title_words) >= MIN_TITLE_WORDS_ALONE and share >= TITLE_SHARE_ALONE:
        return True
    if not _surnames_agree(online.get("first_author"), printed.get("first_author")):
        return False
    online_year, printed_year = _year_int(online.get("year")), _year_int(printed.get("year"))
    one_apart = (online_year is not None and printed_year is not None
                 and abs(online_year - printed_year) == 1)
    return one_apart or share >= TITLE_SHARE_WITH_AUTHOR


def _surnames_agree(left: Any, right: Any) -> bool:
    """Same surname after folding; "van der flier" matches "flier"."""
    a, b = fold(left), fold(right)
    if not a or not b:
        return False
    return a == b or set(a.split()) >= set(b.split()) or set(b.split()) >= set(a.split())


def _trace_back(table: List[List[int]], same: List[List[bool]]) -> List[Tuple[int, int]]:
    """Read the anchors out of the filled LCS table, front to back."""
    anchors: List[Tuple[int, int]] = []
    i = j = 0
    while i < len(same) and j < len(same[0]):
        if same[i][j] and table[i][j] == table[i + 1][j + 1] + 1:
            anchors.append((i, j))
            i, j = i + 1, j + 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return anchors


# --- building the output ---------------------------------------------------------

def _local_entries(local_list: Optional[ReferenceList]) -> List[Dict[str, Any]]:
    """The local list's entries as plain dicts with an integer year."""
    if local_list is None:
        return []
    return [{"number": entry.number, "text": entry.text,
             "first_author": entry.first_author or None, "year": _year_int(entry.year),
             "title": None, "container": None, "doi": None}
            for entry in local_list.entries]


def _make_reference(position: int, number: Optional[int], entry: Dict[str, Any], source: str,
                    printed_text: Optional[str], printed_only: bool = False) -> Dict[str, Any]:
    """A reference dict with exactly the nine contract keys."""
    return {
        "position": position,
        "number": number,
        "first_author": entry.get("first_author") or None,
        "year": _year_int(entry.get("year")),
        "title": None if printed_only else entry.get("title"),
        "container": None if printed_only else entry.get("container"),
        "doi": None if printed_only else entry.get("doi"),
        "source": source,
        "printed_text": printed_text,
    }


def _result(references: List[Dict[str, Any]], source: str, style: Optional[str],
            local_count: Optional[int], counts_agree: Optional[bool],
            report: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the contract block; positions are renumbered 1..N here."""
    for position, reference in enumerate(references, start=1):
        reference["position"] = position
    block: Dict[str, Any] = {"status": "found", "source": source}
    if style:
        block["style"] = style
    block.update({"count": len(references), "local_count": local_count,
                  "counts_agree": counts_agree})
    return {"reference_list": block, "references": references, "report": report}


def _not_available(case: str, online_only: int = 0) -> Dict[str, Any]:
    """No list can be given (nothing found, or only an unordered OpenAlex set)."""
    report = _new_report(case)
    report["online_only_entries"] = online_only
    block = {"status": "not_available", "count": 0, "local_count": None, "counts_agree": None}
    return {"reference_list": block, "references": [], "report": report}


def _new_report(case: str) -> Dict[str, Any]:
    return {"case": case, "anchors": 0, "paired": 0, "conflicts": [],
            "local_only_entries": 0, "online_only_entries": 0, "attached": 0}


def _year_int(value: Any) -> Optional[int]:
    """The first four digits of a year ("1981a" gives 1981); None when absent."""
    if isinstance(value, int):
        return value
    match = re.search(r"(?<!\d)\d{4}(?!\d)", str(value or ""))
    return int(match.group(0)) if match else None
