#!/usr/bin/env python
"""Is this registry record really the paper on the title page? (plan C4)

For biology researchers: a search for a title returns several records, and
the first is not always the paper. Examples met in the audit: a
"Supporting Information" record that shares the paper's title; a later
edition of the same book; a "Faculty Opinions recommendation of ..." entry.
Taking hit 1 blindly puts a wrong citation on a document, and nothing would
say so. So EVERY record from Crossref or OpenAlex, found by DOI or by
title, passes through the one rule below before it is believed.

THE RULE
  1. Records of type `component`, `peer-review`, `posted-content`, `dataset`
     are skipped while any other candidate exists.
  2. Title similarity to the reading of the title page >= 0.90 (case,
     accents and markup folded away).
  3. First-author surname matches, when one was read. When none was read,
     the year must match instead. With neither, the record is rejected.
  4. Year within one of the year read, when one was read (print and online
     years differ by one; book editions by more).
  5. Up to 5 hits are looked at. The first that passes is accepted, but if
     two DIFFERENT DOIs pass, neither is: the lookup is ambiguous.
  6. For a DOI, the record's title is also checked against the title page's
     own text. If it is not printed there, the DOI was wrong.

Every function is pure (no network), so the rule is tested on its own.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

#: Record types that are not the paper itself (rule 1). Crossref and OpenAlex
#: spell them slightly differently; both spellings are listed.
#: NOT "review": OpenAlex uses it for review ARTICLES, which are papers.
NON_PAPER_TYPES = {"component", "peer-review", "posted-content", "dataset",
                   "peer_review", "paratext"}

DEFAULT_TITLE_SIMILARITY = 0.90
DEFAULT_MAX_HITS = 5
DEFAULT_YEAR_TOLERANCE = 1

#: A title is "printed on the page" when this share of its words appear there.
TITLE_WORDS_ON_PAGE = 0.9

# Words that sit in front of a surname and belong to it.
_PARTICLES = {"van", "von", "der", "den", "de", "del", "della", "di", "da",
              "dos", "du", "la", "le", "ten", "ter"}


def fold(text: Any) -> str:
    """Lower-case, accent-free, markup-free, single-spaced: for comparing titles."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)            # <i>, <sub>, JATS tags
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                   if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def title_similarity(left: Any, right: Any) -> float:
    """0..1 similarity of two titles after folding."""
    a, b = fold(left), fold(right)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


#: Affiliation marks the vision model leaves glued to a name: "Jordi Bella*1",
#: "Smith†", "Lee¹". No real name contains a digit or these signs.
#: "+" is a mark too ("Audrey Esclatine¹**+", PLOS Pathogens).
_AFFILIATION_MARKS = re.compile(r"[*†‡§¶#+\d¹²³⁰-₟]+")


def clean_name(name: Any) -> str:
    """A name without affiliation marks, commas left by them, or stray spaces."""
    if not isinstance(name, str):
        return ""
    return " ".join(_AFFILIATION_MARKS.sub("", name).strip(" ,;").split())


def clean_author_list(authors: Any) -> List[str]:
    """
    The vision model's authors, tidied for merging with a registry record.

    Seen on the S1 confirmation: affiliation marks ("Zhang†*"), a leading
    "and" on the last author, and several authors in ONE string ("Xiaoming
    Zhang, Raymond P. Boot-Handford"). A string is split at its commas only
    when EVERY part is a full name (two or more words), so "Bella, Jordi" and
    "Michael P. Sarras, Jr." stay whole.

    Seen on the benchmark pilots (session 40): a whole byline in one string
    with numbered marks, "Paula Istvan1,2*, Einar Birkeland1,2†, ... Paula
    Berstad4 and Trine B. Roughe2,3,11". Removing "1,2" leaves an empty piece
    between two commas, and the last two names are joined by "and"; both are
    separators here, never part of a name.
    """
    cleaned: List[str] = []
    for entry in authors if isinstance(authors, (list, tuple)) else []:
        text = clean_name(entry)
        text = re.sub(r"^(and|&)\s+", "", text, flags=re.I)
        if _ELLIPSIS.match(text):
            continue            # "..." of a shortened list is not a name
        # The joined form is used only if the split succeeds; otherwise the
        # entry is kept exactly as it was ("Smith, J and Jones, K" stays).
        joined = re.sub(r"\s+(?:and|&)\s+", ", ", text, flags=re.I)
        parts = [clean_name(part) for part in joined.split(",")]
        parts = [part for part in parts                # pieces left by "1,2",
                 if part and not _ELLIPSIS.match(part)]  # and a list's "..."
        if len(parts) > 1 and all(len(part.split()) >= 2 for part in parts):
            cleaned.extend(re.sub(r"^(and|&)\s+", "", p, flags=re.I) for p in parts)
        elif text:
            cleaned.append(text)
    return cleaned


#: An author list the page itself shortened: "A, B, C, ..., Y, Z".
_ELLIPSIS = re.compile(r"^\s*(?:\.\s*){3,}\s*$|^\s*…\s*$")

#: How far past the first author the byline may run (characters).
_BYLINE_REACH = 3000

#: Spacing accents a PDF text layer leaves beside their letter
#: ("Kova´ c", "He´ le` ne"), mapped to the combining form.
_SPACING_ACCENTS = {"´": "́", "`": "̀", "¨": "̈",
                    "ˆ": "̂", "˜": "̃", "ˇ": "̌"}


def is_truncated_author_list(authors: Any) -> bool:
    """Whether the reading holds an ellipsis in place of some authors."""
    if not isinstance(authors, (list, tuple)):
        return False
    pieces = [piece for entry in authors if isinstance(entry, str)
              for piece in entry.split(",")]
    return any(_ELLIPSIS.match(piece) for piece in pieces)


def complete_truncated_authors(authors: Any, next_page_text: str) -> Optional[List[str]]:
    """
    The full author list for a reading the page itself shortened.

    For biology researchers: a Cell Press article's first page is a summary
    page whose "Authors" box shortens long lists ("Biana Bernshtein, Esther
    Ndungo, Deniz Cizmeci, ..., Galit Alter"). The vision model copies that
    faithfully. The full byline is printed on the NEXT page, starting with
    the same first author and ending with the same last author, so it is
    read from there: the text from the first author's name to the last
    author's surname, split as any byline is.

    Returns None (keep the reading as it is) unless every name in the short
    list is found in the full one and the full one is longer.
    """
    shown = _names_around_ellipsis(authors)
    if len(shown) < 2 or not next_page_text:
        return None
    text = _rejoin_spacing_accents(" ".join(next_page_text.split()))
    start = text.find(shown[0])
    if start < 0:
        return None
    last_surname = shown[-1].split()[-1]
    end = text.find(last_surname, start, start + _BYLINE_REACH)
    if end < 0:
        return None
    full = clean_author_list([text[start:end + len(last_surname)]])
    full_surnames = {surname_of(name) for name in full}
    if len(full) <= len(shown) or not all(surname_of(n) in full_surnames for n in shown):
        return None
    return full


def _names_around_ellipsis(authors: Any) -> List[str]:
    """The names a shortened list does show, in order, ellipsis removed."""
    names: List[str] = []
    for entry in authors if isinstance(authors, (list, tuple)) else []:
        if not isinstance(entry, str):
            continue
        for piece in re.split(r",|\s+(?:and|&)\s+", entry):
            if _ELLIPSIS.match(piece):
                continue
            name = clean_name(piece)
            if len(name.split()) >= 2:
                names.append(name)
    return names


def _rejoin_spacing_accents(text: str) -> str:
    """'Kova´ c' -> 'Kovác', 'He´ le` ne' -> 'Hélène'."""
    def combine(match: "re.Match[str]") -> str:
        return unicodedata.normalize(
            "NFC", match.group(1) + _SPACING_ACCENTS[match.group(2)])
    accents = "".join(_SPACING_ACCENTS)
    return re.sub(rf"([A-Za-z])\s?([{accents}])\s?", combine, text)


def surname_of(name: Any) -> str:
    """
    A person's surname, folded: "Gary D. Fullerton" -> "fullerton",
    "FULLERTON, G" -> "fullerton", "Annette van der Flier" -> "van der flier".
    """
    name = clean_name(name)
    if not name:
        return ""
    if "," in name:
        return fold(name.split(",")[0])
    words = name.split()
    surname = [words.pop()]
    while words and words[-1].lower() in _PARTICLES:
        surname.insert(0, words.pop())
    return fold(" ".join(surname))


def title_is_on_page(title: Any, page_text: str) -> bool:
    """Whether the title's words are printed on the title page (rule 6)."""
    words = fold(title).split()
    if len(words) < 3:
        return False
    page_words = set(fold(page_text).split())
    return sum(1 for w in words if w in page_words) / len(words) >= TITLE_WORDS_ON_PAGE


def check_candidate(candidate: Dict[str, Any], extracted: Dict[str, Any],
                    title_page_text: str = "", by_doi: bool = False,
                    min_similarity: float = DEFAULT_TITLE_SIMILARITY,
                    year_tolerance: int = DEFAULT_YEAR_TOLERANCE) -> Tuple[bool, str]:
    """
    Apply rules 2, 3, 4 (and 6 for a DOI) to one candidate.

    Args:
        candidate: a canonical record (`title`, `authors`, `year`, `doi`).
        extracted: Prajnya's own reading (`title`, `authors`, `year`).
        title_page_text: the title page's text, for rule 6.
        by_doi: True when the candidate was fetched by a DOI.

    Returns:
        `(accepted, reason)`. The reason is a short phrase for the log.
    """
    title_ok = title_similarity(candidate.get("title"), extracted.get("title")) >= min_similarity
    if not title_ok and by_doi:
        title_ok = title_is_on_page(candidate.get("title"), title_page_text)
    if not title_ok:
        return False, "title does not match"

    extracted_authors = [a for a in (extracted.get("authors") or []) if a]
    extracted_year = _year(extracted.get("year"))
    candidate_year = _year(candidate.get("year"))

    if extracted_authors:
        if not _first_author_matches(candidate, extracted_authors[0]):
            return False, "first author does not match"
    elif extracted_year is None or candidate_year != extracted_year:
        return False, "no author was read and the year does not match"

    if (extracted_year is not None and candidate_year is not None
            and abs(candidate_year - extracted_year) > year_tolerance):
        return False, f"year {candidate_year} is not within {year_tolerance} of {extracted_year}"
    return True, "accepted"


def choose_record(candidates: List[Dict[str, Any]], extracted: Dict[str, Any],
                  title_page_text: str = "", by_doi: bool = False,
                  max_hits: int = DEFAULT_MAX_HITS,
                  min_similarity: float = DEFAULT_TITLE_SIMILARITY,
                  year_tolerance: int = DEFAULT_YEAR_TOLERANCE
                  ) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Rules 1 and 5: pick the one record that is the paper, or none.

    Returns:
        `(record, reason)`. `record` is None when nothing passed, or when two
        different DOIs passed (reason says "ambiguous").
    """
    pool = _without_non_papers(candidates)[:max_hits]
    passed = []
    last_reason = "no candidates"
    for candidate in pool:
        ok, reason = check_candidate(candidate, extracted, title_page_text, by_doi,
                                     min_similarity, year_tolerance)
        if ok:
            passed.append(candidate)
        else:
            last_reason = reason
    dois = {(c.get("doi") or "").lower() for c in passed}
    if len(dois) > 1:
        return None, "ambiguous: two different DOIs pass"
    if passed:
        return passed[0], "accepted"
    return None, last_reason


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _without_non_papers(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rule 1: drop non-paper types, but only if something else is left."""
    papers = [c for c in candidates if str(c.get("type") or "").lower() not in NON_PAPER_TYPES]
    return papers if papers else list(candidates)


def surname_readings(name: Any) -> set:
    """
    Every surname a first-author string could mean.

    The vision model sometimes returns a whole author LINE as one string:
    "Roland Aufschnaiter1,*, Evan A. Zamir2, ..." (seen on two papers of the
    S1 confirmation). A comma then means either "Last, First" or "end of the
    first author". Both readings are offered; the registry's first author
    must equal one of them, so a wrong first author is still refused.
    """
    readings = {surname_of(name)}
    cleaned = clean_name(name)
    if "," in cleaned:
        first_person = cleaned.split(",")[0].strip()
        readings.add(surname_of(first_person))          # "First Last, First Last"
    return {r for r in readings if r}


def _first_author_matches(candidate: Dict[str, Any], extracted_first: str) -> bool:
    """Whether the candidate's first author has the surname that was read."""
    authors = candidate.get("authors") or []
    return bool(authors and surname_of(authors[0]) in surname_readings(extracted_first))


def _year(value: Any) -> Optional[int]:
    match = re.search(r"\b(1[89]\d{2}|20\d{2})\b", str(value or ""))
    return int(match.group(1)) if match else None
