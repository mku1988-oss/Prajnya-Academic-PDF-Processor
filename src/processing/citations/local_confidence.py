#!/usr/bin/env python
"""How far to trust a record that nothing outside the PDF has confirmed.

For biology researchers: at citation level `local` Prajnya asks nobody. No
Zotero library and no online registry says "yes, that is the paper". So the
only evidence is how well Prajnya's own two readings of the title page
agree with each other:

  - the vision model's reading (title, authors, year, journal);
  - the PDF's own text layer (the exact letters stored in the file), which
    also gives the DOI when the page prints one.

When the text layer carries the DOI next to a "doi:" label AND the model's
title is really printed on the page, two independent readings agree, and
the record is taken as good. When they do not agree, the document is
flagged for a human, as before.

THE RULE (researcher's decision, session 32; plan section C3):

    confident text-layer DOI  +  title found on the page        0.80
    title and first author found on the page, no such DOI       0.75
    anything less                                                0.60  flagged

`review.LOW_CONFIDENCE` is 0.75: a record below it is flagged for review.
Before this rule existed, every `local` document scored 0.0 and was
flagged, which would have taught people to ignore the flag.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Tuple

#: A text-layer DOI at or above this was found next to a "doi:" label or as
#: a doi.org link (0.98 / 0.95). A bare DOI in the body scores 0.85 and may
#: belong to a CITED paper, so it does not count as confirmation.
CONFIDENT_DOI = 0.95

#: The title counts as "on the page" when this share of its words are there.
TITLE_WORDS_ON_PAGE = 0.9

#: A title needs at least this many words to be checked at all: a one-word
#: "title" is on almost any page.
MIN_TITLE_WORDS = 3

CONFIDENCE_DOI_AND_TITLE = 0.80
CONFIDENCE_TITLE_AND_AUTHOR = 0.75
CONFIDENCE_UNCONFIRMED = 0.60

#: `validation_source` values written at level `local`.
SOURCE_TEXT_LAYER = "text_layer"          # the two readings agree
SOURCE_VLM_EXTRACTION = "vlm_extraction"  # the model's reading, unconfirmed

#: The `doi_source` value `prefer_text_layer_doi` writes for a text-layer
#: DOI. (The name is historical: the text layer is read with pypdfium2.)
TEXT_LAYER_DOI_SOURCE = "pymupdf_text_layer"


def local_confidence(merged: Dict[str, Any], title_page_text: str) -> Tuple[float, str]:
    """
    Score a record established with no outside help.

    Args:
        merged: the merged metadata (title, authors, doi, doi_source,
            doi_confidence).
        title_page_text: the title page's text: the PDF's text layer, or the
            model's transcription when the page is a scan.

    Returns:
        `(confidence, validation_source)`.

    Example:
        >>> local_confidence({"title": "Collagen structure new tricks",
        ...                   "authors": ["Jordi Bella"]},
        ...                  "Collagen structure: new tricks ... Jordi Bella")
        (0.75, 'text_layer')
    """
    title_found = title_is_on_page(merged.get("title"), title_page_text)
    author_found = first_author_is_on_page(merged.get("authors"), title_page_text)

    if title_found and has_confident_text_layer_doi(merged):
        return CONFIDENCE_DOI_AND_TITLE, SOURCE_TEXT_LAYER
    if title_found and author_found:
        return CONFIDENCE_TITLE_AND_AUTHOR, SOURCE_TEXT_LAYER
    return CONFIDENCE_UNCONFIRMED, SOURCE_VLM_EXTRACTION


def has_confident_text_layer_doi(merged: Dict[str, Any]) -> bool:
    """Whether the DOI was read from the text layer beside a label or link."""
    if not merged.get("doi") or merged.get("doi_source") != TEXT_LAYER_DOI_SOURCE:
        return False
    try:
        return float(merged.get("doi_confidence") or 0.0) >= CONFIDENT_DOI
    except (TypeError, ValueError):
        return False


def title_is_on_page(title: Any, page_text: str) -> bool:
    """
    Whether the title's words are printed on the title page.

    Compared word by word, after folding case and accents and dropping
    punctuation, because a title is often broken over lines, hyphenated, or
    set in capitals on the page.
    """
    title_words = _words(title)
    if len(title_words) < MIN_TITLE_WORDS:
        return False
    page_words = set(_words(page_text))
    found = sum(1 for word in title_words if word in page_words)
    return found / len(title_words) >= TITLE_WORDS_ON_PAGE


def first_author_is_on_page(authors: Any, page_text: str) -> bool:
    """Whether the first author's surname is printed on the title page."""
    if not isinstance(authors, (list, tuple)) or not authors:
        return False
    name_words = _words(str(authors[0]).split(",")[0] if "," in str(authors[0])
                        else str(authors[0]))
    if not name_words:
        return False
    # "Last, First" was cut at the comma above; otherwise the surname is the
    # last word of "First Last".
    surname = name_words[-1]
    return len(surname) > 1 and surname in set(_words(page_text))


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _words(text: Any) -> list:
    """Lower-cased, accent-free words of a text, punctuation dropped."""
    if not isinstance(text, str):
        return []
    folded = "".join(char for char in unicodedata.normalize("NFKD", text.lower())
                     if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", folded)
