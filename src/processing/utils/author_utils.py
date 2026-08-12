# src/processing/utils/author_utils.py
"""
Author utilities:
- normalization
- canonicalization
- fuzzy matching helpers
- merging helpers for author lists
"""

from typing import Dict, List, Tuple, Optional
import re
import unicodedata

try:
    # rapidfuzz is fast and preferred
    from rapidfuzz import fuzz
except Exception:
    # fallback to difflib if not installed
    import difflib
    fuzz = None

def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _strip_accents(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))

def normalize_name_raw(name: str) -> str:
    """Return lowercased, accent-stripped, whitespace-normalized name."""
    if not name:
        return ""
    n = _normalize_whitespace(name)
    n = _strip_accents(n)
    return n.lower()

def split_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristic split into (first_names, last_name).
    If input is "Last, First" it will detect that. Otherwise, last token is last name.
    Returns (first_names or None, last_name or None)
    """
    if not name:
        return None, None
    name = _normalize_whitespace(name)
    # If comma present, assume "Last, First"
    if ',' in name:
        parts = [p.strip() for p in name.split(',', 1)]
        last = parts[0]
        first = parts[1] if len(parts) > 1 else ""
        return first or None, last or None
    # Otherwise assume Last token is last name
    tokens = name.split()
    if len(tokens) == 1:
        return None, tokens[0]
    first = " ".join(tokens[:-1])
    last = tokens[-1]
    return first or None, last or None

#: Characters journals hang off an author's name to point at an affiliation
#: footnote: digits, the classic footnote symbols, and Unicode superscripts.
_AFFILIATION_MARKER_CHARS = "0-9*†‡§¶‖#¹²³⁰-⁹"

#: A trailing run of those markers, plus any commas/spaces binding them
#: together ("*1", "1,2", " †,‡").
_TRAILING_AFFILIATION = re.compile(
    rf"[\s,]*[{_AFFILIATION_MARKER_CHARS}][{_AFFILIATION_MARKER_CHARS},\s]*$"
)


def strip_affiliation_markers(name: str) -> str:
    """
    Remove the affiliation footnote markers a journal glues onto a name.

    For biology researchers: papers print authors as "Jordi Bella*1" or
    "Michael J. Mienaltowski1,2", where the trailing symbols are pointers to
    the affiliation list, not part of anybody's name. The VLM reads the page
    faithfully and hands those markers straight through.

    Leaving them in did real damage, because `canonicalize_author` keys on the
    string: "Jordi Bella*1" canonicalizes to 'bella*1, J' while Zotero's clean
    "Jordi Bella" gives 'bella, J'. Two different keys for one person means
    the dedupe cannot pair them, so the dirty VLM form survives into the final
    record — and is then stamped `validation_source: zotero_api`, claiming a
    canonical provenance the value does not have. On a multi-author paper the
    same mismatch can list one person twice.

    Only a TRAILING run is stripped, and only of markers — never letters. A
    name is never reduced to nothing: if stripping would empty it (a
    "name" that is all digits), the original is returned untouched.

    Args:
        name: an author name, possibly carrying markers

    Returns:
        The name with trailing affiliation markers removed.

    Example:
        >>> strip_affiliation_markers("Jordi Bella*1")
        'Jordi Bella'
        >>> strip_affiliation_markers("Michael J. Mienaltowski1,2")
        'Michael J. Mienaltowski'
    """
    if not name:
        return ""
    stripped = _TRAILING_AFFILIATION.sub("", name).strip().strip(",").strip()
    return stripped or name.strip()


def canonicalize_author(name: str) -> str:
    """
    Return 'lastname, f' canonical short form for matchup.
    Example: 'Johanna Myllyharju' -> 'myllyharju, j'

    Affiliation markers are stripped first, so "Jordi Bella*1" and
    "Jordi Bella" produce the SAME key — see `strip_affiliation_markers`.
    """
    if not name:
        return ""
    name = strip_affiliation_markers(name)
    first, last = split_name(name)
    if not last:
        # fallback to normalized raw name
        return normalize_name_raw(name)
    # take first initial if first available
    if first:
        initial = first.strip().split()[0][0]
        return f"{normalize_name_raw(last)}, {initial}"
    else:
        # no first name known — use last only
        return normalize_name_raw(last)

def author_pretty(name: str) -> str:
    """Return cleaned pretty name using normalized whitespace (but keeping case)."""
    if not name:
        return ""
    return " ".join(name.split())

def _fuzzy_score(a: str, b: str) -> int:
    """
    Return fuzzy matching score 0..100 between a and b.
    Prefer rapidfuzz when available.
    """
    if not a or not b:
        return 0
    if 'rapidfuzz' in globals() and fuzz is not None:
        # use token_set_ratio style measure
        try:
            return int(fuzz.token_set_ratio(a, b))
        except Exception:
            # fallback to simple ratio
            try:
                return int(fuzz.ratio(a, b))
            except Exception:
                pass
    # fallback: difflib
    import difflib
    s = difflib.SequenceMatcher(None, a, b).ratio()
    return int(s * 100)

def names_match(candidate: str, raw_text: str, threshold: int = 90) -> Tuple[bool, int]:
    """
    Check if candidate name appears in raw_text with fuzzy match >= threshold.
    
    SPECIAL HANDLING for abbreviated names:
    - "G Fullerton" should match "Gary D. Fullerton"
    - "E Nes" should match "Elena Nes"
    
    Returns (matched_bool, score)
    """
    if not candidate or not raw_text:
        return False, 0

    candidate_norm = normalize_name_raw(candidate)
    raw = normalize_name_raw(raw_text)

    # Quick substring check
    if candidate_norm in raw:
        return True, 100

    # FIX BUG #13: Special handling for abbreviated first names
    # Check if candidate is "Initial Lastname" format
    candidate_parts = candidate.strip().split()
    if len(candidate_parts) == 2:
        first_part = candidate_parts[0]
        last_part = candidate_parts[1]
        
        # If first part is a single letter (initial)
        if len(first_part) == 1 and first_part.isalpha():
            # Search for "<Initial>... <Lastname>" in text
            # e.g., "G Fullerton" matches "Gary D. Fullerton"
            pattern = f"{first_part.lower()}[a-z]*[\\s\\.]*.*?{last_part.lower()}"
            import re
            if re.search(pattern, raw, re.IGNORECASE):
                return True, 95  # High confidence for initial match

    # Fuzzy check whole text
    score = _fuzzy_score(candidate_norm, raw)
    if score >= threshold:
        return True, score

    # Check line-by-line
    for line in raw_text.splitlines():
        line_norm = normalize_name_raw(line)
        if not line_norm:
            continue
        if candidate_norm in line_norm:
            return True, 100
        
        # FIX BUG #13: Also check initial pattern per line
        if len(candidate_parts) == 2 and len(candidate_parts[0]) == 1:
            pattern = f"{candidate_parts[0].lower()}[a-z]*[\\s\\.]*.*?{candidate_parts[1].lower()}"
            if re.search(pattern, line_norm):
                return True, 95
        
        sc = _fuzzy_score(candidate_norm, line_norm)
        if sc >= threshold:
            return True, sc

    return False, score

def merge_author_lists(
    vlm_authors: List[str], 
    zotero_authors: List[str], 
    title_page_text: str,
    containment_threshold: int = 2
) -> Tuple[List[str], List[str]]:
    """
    Three-Way Author Consensus (VLM + Zotero + Title Page)

    1. Build a unified pool of candidate author names from VLM and Zotero.
    2. Confirm each candidate using structured identity scoring 
    (AuthorMatcher) against reliable reference authors 
    (VLM if available, otherwise Zotero). 
    If none match, fall back to fuzzy matching on title-page text.
    3. Remove summary/aggregate strings (e.g., “A. Smith, B. Jones”).
    4. Canonically deduplicate by keeping the most complete form 
    (e.g., prefer “John Smith” over “J. Smith”).

    Returns a clean, deduplicated author list and any filtered suspicious strings.
    """
    # AUTHOR ORDER IS DATA, NOT PRESENTATION. In a biology paper the first author did the
    # work and the last author ran the lab, so the sequence must survive this function
    # untouched. Every collection below is therefore an ORDERED list de-duplicated by a
    # lookaside set -- never a bare set. (Sets iterate in hash order, which Python
    # randomises per process, so an earlier version of this function returned the authors
    # in a different order on every run.)
    if not title_page_text:
        return _dedupe_preserving_order(
            [strip_affiliation_markers(a)
             for a in (vlm_authors or []) + (zotero_authors or []) if a]), []

    # --- Step 1: unified candidate pool (VLM order first, then any Zotero extras) ---
    # Affiliation markers come off HERE, before anything keys on the string:
    # otherwise "Jordi Bella*1" and Zotero's "Jordi Bella" never pair up and
    # the dirty form wins on pool order. See `strip_affiliation_markers`.
    candidate_pool = _dedupe_preserving_order(
        [strip_affiliation_markers(a.strip().strip("[]").rstrip(","))
         for a in (vlm_authors or []) if a]
        + [strip_affiliation_markers(a.strip().strip("[]").rstrip(","))
           for a in (zotero_authors or []) if a]
    )

    # --- Step 2: identity scoring against reference authors ---
    # References are cleaned too, so a stripped candidate is not scored against
    # a marker-carrying reference of the same person.
    if vlm_authors:
        reference_authors = [strip_affiliation_markers(a) for a in vlm_authors if a.strip()]
    else:
        reference_authors = [strip_affiliation_markers(a) for a in zotero_authors if a.strip()]

    confirmed_candidates = []
    for cand in candidate_pool:
        for ref in reference_authors:
            is_match, score, breakdown = AuthorMatcher.match_authors(cand, ref)
            if is_match:
                confirmed_candidates.append(cand)
                break

    # --- fallback BEFORE redundancy filter and dedupe ---
    if not confirmed_candidates:
        confirmed_candidates = [
            cand for cand in candidate_pool
            if names_match(cand, title_page_text)[0]
        ]

    # --- Step 3: filter summary/superset strings ---
    # e.g. drop "A. Smith, B. Jones" when "A. Smith" and "B. Jones" are both present.
    suspicious_authors = [
        a for a in confirmed_candidates
        if sum(1 for b in confirmed_candidates
               if a != b and b.lower() in a.lower()) >= containment_threshold
    ]
    suspicious_lookup = set(suspicious_authors)

    good_authors = [a for a in confirmed_candidates if a not in suspicious_lookup]

    # --- Step 4: canonical dedupe (dict preserves first-insertion order) ---
    author_map = {}
    for name in good_authors:
        canonical_key = canonicalize_author(name)
        if not canonical_key: continue

        # If we've seen this author before, keep the longer (more complete) name,
        # but hold that author's original position in the list.
        if canonical_key in author_map:
            if len(name) > len(author_map[canonical_key]):
                author_map[canonical_key] = name
        else:
            author_map[canonical_key] = name

    final_clean_list = list(author_map.values())

    return final_clean_list, suspicious_authors


def _dedupe_preserving_order(names: List[str]) -> List[str]:
    """
    Drop duplicates from `names` while keeping the first occurrence's position.

    Used instead of set() wherever author order must be preserved.
    """
    seen = set()
    ordered = []
    for name in names:
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered

class AuthorMatcher:
    """
    Weighted, structured per-author scoring for bibliographic metadata.
    
    Weights reflect human intuition about name stability:
    - Last name: 50 points (backbone)
    - First name: 35 points (handles abbreviations)
    - Middle name: 15 points (handles variation)
    Total: 100 points
    """
    
    # Configurable weights
    WEIGHT_LAST = 50
    WEIGHT_FIRST = 35
    WEIGHT_MIDDLE = 15
    
    # Matching thresholds
    THRESHOLD_DEFINITE = 85  # Definitely same person
    THRESHOLD_LIKELY = 70    # Probably same person
    
    @staticmethod
    def parse_name(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse author name into (first, middle, last) components.
        
        Handles formats:
        - "Gary D. Fullerton"
        - "Fullerton, Gary D."
        - "G Fullerton"
        - "Fullerton"
        
        Returns: (first, middle, last)
        """
        if not name:
            return None, None, None
        
        name = name.strip()
        
        # Handle "Last, First Middle" format
        if ',' in name:
            parts = [p.strip() for p in name.split(',', 1)]
            last = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            
            # Parse first/middle from rest
            tokens = rest.split()
            first = tokens[0] if tokens else None
            middle = tokens[1] if len(tokens) > 1 else None
            
            return first, middle, last
        
        # Handle "First Middle Last" format
        tokens = name.split()
        if len(tokens) == 1:
            return None, None, tokens[0]  # Last only
        elif len(tokens) == 2:
            return tokens[0], None, tokens[1]  # First Last
        else:
            return tokens[0], tokens[1], tokens[-1]  # First Middle Last
    
    @staticmethod
    def normalize_component(comp: Optional[str]) -> str:
        """Normalize name component for comparison."""
        if not comp:
            return ""
        # Remove punctuation, lowercase
        return re.sub(r'[^\w]', '', comp.lower())
    
    @classmethod
    def score_last_name(cls, candidate_last: Optional[str], 
                        reference_last: Optional[str]) -> float:
        """
        Score last name match (0-50 points).
        
        Rules:
        - Exact match: 50 points
        - No match: 0 points (immediate reject)
        """
        if not candidate_last or not reference_last:
            return 0.0
        
        cand_norm = cls.normalize_component(candidate_last)
        ref_norm = cls.normalize_component(reference_last)
        
        if cand_norm == ref_norm:
            return cls.WEIGHT_LAST  # 50 points
        
        return 0.0  # Different last name = reject
    
    @classmethod
    def score_first_name(cls, candidate_first: Optional[str],
                         reference_first: Optional[str]) -> float:
        """
        Score first name match (0-35 points).
        
        Rules:
        - Full match: 35 points
        - Initial match (G vs Gary): 28 points (80%)
        - Both initials (G vs G): 28 points
        - Missing data: 17 points (50% - neutral)
        - Mismatch: 0 points
        """
        if not candidate_first or not reference_first:
            return cls.WEIGHT_FIRST * 0.5  # 17.5 points (neutral)
        
        cand_norm = cls.normalize_component(candidate_first)
        ref_norm = cls.normalize_component(reference_first)
        
        # Full match
        if cand_norm == ref_norm:
            return cls.WEIGHT_FIRST  # 35 points
        
        # Initial match
        if len(cand_norm) == 1 and ref_norm.startswith(cand_norm):
            return cls.WEIGHT_FIRST * 0.8  # 28 points
        if len(ref_norm) == 1 and cand_norm.startswith(ref_norm):
            return cls.WEIGHT_FIRST * 0.8  # 28 points
        
        # Both single initials but different
        if len(cand_norm) == 1 and len(ref_norm) == 1:
            return 0.0  # Different initials = reject
        
        return 0.0  # No match
    
    @classmethod
    def score_middle_name(cls, candidate_middle: Optional[str],
                          reference_middle: Optional[str]) -> float:
        """
        Score middle name match (0-15 points).
        
        Rules:
        - Full match: 15 points
        - Initial match: 12 points (80%)
        - Missing data: 10 points (67% - expected)
        - Mismatch: 5 points (33% - not penalized heavily)
        """
        if not candidate_middle or not reference_middle:
            return cls.WEIGHT_MIDDLE * 0.67  # 10 points (expected absence)
        
        cand_norm = cls.normalize_component(candidate_middle)
        ref_norm = cls.normalize_component(reference_middle)
        
        # Full match
        if cand_norm == ref_norm:
            return cls.WEIGHT_MIDDLE  # 15 points
        
        # Initial match
        if len(cand_norm) == 1 and ref_norm.startswith(cand_norm):
            return cls.WEIGHT_MIDDLE * 0.8  # 12 points
        if len(ref_norm) == 1 and cand_norm.startswith(ref_norm):
            return cls.WEIGHT_MIDDLE * 0.8  # 12 points
        
        # Mismatch - not heavily penalized
        return cls.WEIGHT_MIDDLE * 0.33  # 5 points
    
    @classmethod
    def match_authors(cls, candidate: str, reference: str) -> Tuple[bool, float, dict]:
        """
        Match two author names using weighted component scoring.
        
        Args:
            candidate: Author name from Zotero (e.g., "G Fullerton")
            reference: Author name from title page (e.g., "Gary D. Fullerton")
        
        Returns:
            (is_match, score, breakdown)
            
        Example:
            >>> match_authors("G Fullerton", "Gary D. Fullerton")
            (True, 88.0, {
                'last': 50.0,
                'first': 28.0,
                'middle': 10.0,
                'total': 88.0
            })
        """
        # Parse names
        cand_first, cand_middle, cand_last = cls.parse_name(candidate)
        ref_first, ref_middle, ref_last = cls.parse_name(reference)
        
        # Score components
        last_score = cls.score_last_name(cand_last, ref_last)
        first_score = cls.score_first_name(cand_first, ref_first)
        middle_score = cls.score_middle_name(cand_middle, ref_middle)
        
        total_score = last_score + first_score + middle_score
        
        breakdown = {
            'last': last_score,
            'first': first_score,
            'middle': middle_score,
            'total': total_score,
            'candidate_parsed': (cand_first, cand_middle, cand_last),
            'reference_parsed': (ref_first, ref_middle, ref_last)
        }
        
        # Decision
        is_match = total_score >= cls.THRESHOLD_LIKELY
        
        return is_match, total_score, breakdown