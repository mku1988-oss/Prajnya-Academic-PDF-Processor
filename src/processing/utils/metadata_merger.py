# File: src/processing/utils/metadata_merger.py
"""
Intelligent Merge 2.0 - Three-Way Consensus merge logic.
Merges VLM metadata and Zotero canonical metadata using PyMuPDF title-page text as referee.
"""

from typing import Dict, Any, List, Optional
from .author_utils import merge_author_lists
import logging
import re

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')
_DOI_RE = re.compile(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', re.I)

# --- NEW HELPER FUNCTION TO SANITIZE YEAR ---
def _extract_year(year_input: Any) -> Optional[int]:
    """
    Phase 3.6 hardened year parser.
    Extracts a valid 4-digit year (1900-2099) from:
        - int (e.g., 2019)
        - string (e.g., 'Jan 2004', '01/2004', '2004.', 'Published 2008')
    Returns:
        int year or None
    """
    if year_input is None:
        return None

    # If already an integer and in valid range, return directly
    if isinstance(year_input, int) and 1900 <= year_input <= 2099:
        return year_input

    # Convert to string for regex parsing
    s = str(year_input).strip()
    if not s:
        return None

    # Find the first plausible year
    match = re.search(r'\b(19\d{2}|20\d{2})\b', s)
    if match:
        try:
            year = int(match.group(1))
            if 1900 <= year <= 2099:
                return year
        except Exception:
            return None

    return None

def normalize_doi(doi: Any) -> Optional[str]:
    """
    Phase 3.6 DOI sanitizer.
    - Accepts raw DOI, URL DOI, 'doi:' formats.
    - Returns canonical DOI (no URL prefix, no empty string).
    - Returns None for invalid or blank DOI.
    """
    if not doi:
        return None

    s = str(doi).strip()
    if not s:
        return None

    # Remove common prefixes
    s = s.replace("https://doi.org/", "")
    s = s.replace("http://doi.org/", "")
    s = s.replace("doi:", "")
    s = s.strip().strip(".")

    # Basic DOI regex
    doi_re = re.compile(r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$')
    if doi_re.match(s):
        return s

    return None  # invalid or malformed DOI

def choose_title(vlm_title: str, zotero_title: str, title_page_text: str) -> str:
    # ... (this function remains the same)
    from .author_utils import _fuzzy_score, normalize_name_raw
    candidates = []
    if zotero_title: candidates.append(('zotero', zotero_title))
    if vlm_title and vlm_title not in [c[1] for c in candidates]: candidates.append(('vlm', vlm_title))
    best = None
    best_score = -1
    for source, t in candidates:
        if not t: continue
        score = _fuzzy_score(normalize_name_raw(t), normalize_name_raw(title_page_text))
        if score > best_score:
            best_score = score
            best = (source, t, score)
    if best and best[2] >= 85:
        logger.debug(f"Title chosen from {best[0]} with score={best[2]}")
        return best[1]
    for _, t in candidates:
        if t and len(t) > 20: return t
    return vlm_title or zotero_title or ""

def merge_metadata(
    vlm_metadata: Dict[str, Any],
    zotero_record: Dict[str, Any],
    title_page_text: str
) -> Dict[str, Any]:
    """
    Merge VLM metadata, Zotero records, and title-page context into a single
    final metadata dictionary (Phase 3.6).

    This function:
    1. Normalizes and reconciles DOI, year, title, journal, and publisher.
    2. Merges authors using three-way consensus via merge_author_lists():
            - VLM authors (structured extraction)
            - Zotero authors (canonical)
            - Title-page text (referee)
        Includes Fix-D: if VLM returns a single concatenated multi-author
        block, VLM authors are discarded and Zotero becomes the reference.
    3. Returns both the clean final author list and any suspicious
        summary/aggregate author strings for review-mode triggers.

    Returns:
        merged (dict) with normalized bibliographic fields, including:
            - merged['authors'] : List[str]       Final author list
            - merged['suspicious_authors'] : List[str]
    """
    if zotero_record is None:
        zotero_record = {}

    merged = {}

    # --------------------------
    # DOI
    # --------------------------
    zotero_doi = zotero_record.get("doi") or zotero_record.get("DOI")
    vlm_doi = vlm_metadata.get("doi")
    merged["doi"] = normalize_doi(zotero_doi) or normalize_doi(vlm_doi)

    # --------------------------
    # YEAR
    # --------------------------
    merged["year"] = _extract_year(zotero_record.get("year")) or \
                     _extract_year(vlm_metadata.get("year"))

    # --------------------------
    # TITLE
    # --------------------------
    merged["title"] = choose_title(
        vlm_metadata.get("title"),
        zotero_record.get("title"),
        title_page_text
    )

    # --------------------------
    # AUTHORS  (CRITICAL FIX)
    # --------------------------
    vlm_authors = vlm_metadata.get("authors") or []
    zotero_authors = zotero_record.get("authors") or []
    
    # Detect concatenated multi-author blocks in VLM output (Fix D)
    def _looks_like_concatenated_block(a: str) -> bool:
        if not a or not isinstance(a, str):
            return False

        # Condition 1: Multiple commas → usually separate authors
        if a.count(",") < 2:
            return False

        # Condition 2: Multiple capitalized words → likely multiple names
        cap_tokens = sum(1 for t in a.split() if t and t[0].isupper())
        return cap_tokens >= 2


    # Fix-D: If VLM returned a single huge author block → ignore VLM authors
    if isinstance(vlm_authors, list) and len(vlm_authors) == 1:
        first = vlm_authors[0]
        if _looks_like_concatenated_block(first):
            logger.debug(
                "Fix-D triggered: VLM authors appear concatenated. "
                "Using Zotero authors as primary reference."
            )
            vlm_authors = []  # Force Zotero to be the reference list

    authors, suspicious = merge_author_lists(
        vlm_authors,
        zotero_authors,
        title_page_text
    )

    merged["authors"] = authors
    merged["suspicious_authors"] = suspicious

    # --------------------------
    # Journal + Publisher
    # --------------------------
    merged["journal"] = zotero_record.get("journal") or vlm_metadata.get("journal")
    merged["publisher"] = zotero_record.get("publisher") or vlm_metadata.get("publisher")

    # --------------------------
    # Carry over provenance if present
    # --------------------------
    for key in [
        "validation_confidence",
        "validation_source",
        "last_updated",
        "retrieval_provenance"
    ]:
        if key in zotero_record:
            merged[key] = zotero_record[key]
        elif key in vlm_metadata:
            merged[key] = vlm_metadata[key]

    # Normalize blanks
    for key in ["doi", "journal", "publisher", "title"]:
        if merged.get(key) == "":
            merged[key] = None

    if not isinstance(merged.get("authors"), list):
        merged["authors"] = []

    return merged