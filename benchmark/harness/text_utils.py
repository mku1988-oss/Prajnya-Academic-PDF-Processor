"""
Text normalisation and comparison for the benchmark (protocol §5).

For biology researchers: two tools can print the same sentence differently
(a ligature "ﬁ" vs "fi", an en dash vs a hyphen, a word broken over a line).
Every text is passed through the SAME steps here, for every tool and for the
answer key, before anything is compared. Nothing here is tool-specific.
"""
from __future__ import annotations

import html
import re
import unicodedata

from rapidfuzz.distance import Levenshtein

INVISIBLE = "​‌‍⁠﻿­"   # zero-width chars, soft hyphen
DASHES ="\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"
QUOTES_DOUBLE = "\u201c\u201d\u201e\u201f\u00ab\u00bb\u2033"
QUOTES_SINGLE = "\u2018\u2019\u201a\u201b\u2032"
LABEL_PATTERN = re.compile(
    r"^\s*(?:\*\*|<[^>]+>)*\s*(fig(?:ure)?s?|table|tab)\.?\s*"
    r"(s?)(\d+)", re.IGNORECASE)


def normalise_for_scoring(text: str | None) -> str:
    """§5 steps 1-6: NFKC, dashes, quotes, line-end hyphens, markup, spaces."""
    if not text:
        return ""
    text = strip_markup(text)
    text = text.translate({ord(c): None for c in INVISIBLE})
    text = unicodedata.normalize("NFKC", text)
    text = text.translate({ord(c): "-" for c in DASHES})
    text = text.translate({ord(c): '"' for c in QUOTES_DOUBLE})
    text = text.translate({ord(c): "'" for c in QUOTES_SINGLE})
    text = re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalise_for_matching(text: str | None) -> str:
    """§5 step 7: also lowercase and without punctuation (matching only)."""
    text = normalise_for_scoring(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_markup(text: str) -> str:
    """Remove HTML tags, Markdown emphasis/headings/links/images and simple
    LaTeX commands, keeping the words inside."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)          # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)        # links
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__|\*|_|`)", "", text)
    text = re.sub(r"\\(?:mathrm|textbf|textit|mathbf|text|mathit|operatorname)\{([^}]*)\}",
                  r"\1", text)
    text = re.sub(r"\\([a-zA-Z]+)", lambda m: LATEX_SYMBOLS.get(m.group(1), " "), text)
    text = re.sub(r"[\^_]", "", text)
    text = text.replace("$", " ").replace("{", "").replace("}", "")
    return text


GREEK = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
         "omicron pi rho sigma tau upsilon phi chi psi omega").split()
LATEX_SYMBOLS = {name: chr(0x3B1 + i + (1 if i >= 17 else 0)) for i, name in enumerate(GREEK)}
LATEX_SYMBOLS.update({name.capitalize(): chr(0x391 + i + (1 if i >= 17 else 0))
                      for i, name in enumerate(GREEK)})
LATEX_SYMBOLS.update({"varepsilon": "ε", "varphi": "φ", "vartheta": "θ", "pm": "±",
                      "mp": "∓", "times": "×", "cdot": "·", "leq": "≤", "le": "≤",
                      "geq": "≥", "ge": "≥", "neq": "≠", "approx": "≈", "sim": "~",
                      "circ": "°", "degree": "°", "infty": "∞", "rightarrow": "→",
                      "to": "→", "leftarrow": "←", "uparrow": "↑", "downarrow": "↓",
                      "%": "%", "prime": "′", "AA": "Å", "ell": "ℓ", "partial": "∂"})


def similarity(a: str | None, b: str | None) -> float:
    """1 - normalised Levenshtein distance on scoring-normalised text."""
    a, b = normalise_for_scoring(a), normalise_for_scoring(b)
    if not a and not b:
        return 1.0
    return Levenshtein.normalized_similarity(a, b)


def match_similarity(a: str | None, b: str | None) -> float:
    """Similarity on matching-normalised text (used to pair items up)."""
    a, b = normalise_for_matching(a), normalise_for_matching(b)
    if not a and not b:
        return 1.0
    return Levenshtein.normalized_similarity(a, b)


def parse_label(text: str | None) -> tuple[str, str] | None:
    """('figure'|'table', number) from a label or caption start; None for
    supplementary items ('Figure S2') or text with no label."""
    if not text:
        return None
    normalised = normalise_for_scoring(text)
    found = LABEL_PATTERN.match(normalised)
    if not found or found.group(2):
        return None
    if "supplement" in normalised[found.end():found.end() + 30].lower():
        return None                    # "Figure 2-figure supplement 1"
    kind = "table" if found.group(1).lower().startswith("tab") else "figure"
    return kind, str(int(found.group(3)))


def clean_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", doi)
    return doi.rstrip(".") or None


def year_of(value) -> int | None:
    if value is None:
        return None
    found = re.search(r"(1[89]\d\d|20\d\d)", str(value))
    return int(found.group(1)) if found else None


def surname_key(name: str | None) -> str:
    """Lowercase ASCII-folded surname, for comparing author lists."""
    if not name:
        return ""
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", folded.lower())
