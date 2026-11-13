#!/usr/bin/env python3
"""
Front-matter identifier scan — look past page 1 for DOI, ISBN and ISSN.

For biology researchers: a journal article prints its DOI on the first page,
so reading page 1 was enough. A *book* does not. A Springer volume puts its
ISBN, ISSN and chapter DOI on the **copyright page**, which is typically page
4-6, after a half-title, a series page and a title page. Scanning only page 1
of such a PDF finds nothing at all, and the document arrives at the pipeline
looking like an unidentifiable article.

This module reads the text layer of the first few pages (default 10) and
returns the best identifier of each kind, together with the page it came from.
It is pure CPU work — regex over text that PdfBackend already hands us — and
it makes no routing decision. Deciding what *kind* of document this is comes
later (see plan.md, "Known gap"); this scan only gathers the evidence.

Two rules keep the deeper scan from doing harm:

1. **Page 1 wins.** If page 1 yields a DOI, that is the document's DOI, at the
   confidence page 1 earned. Only when page 1 is silent do we look deeper.
2. **Beyond page 1, only *labelled* DOIs count** (``doi: 10.…`` or a doi.org
   URL, confidence >= 0.95). A bare ``10.xxxx/yyyy`` on page 3 of an article
   is far more likely to be a *citation of someone else's paper* than this
   document's own identifier, and adopting it would silently mis-attribute the
   whole record. A missed DOI is recoverable downstream; a wrong one is not —
   the same reasoning as the line-wrap repair in `metadata_extractor`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from .metadata_extractor import MetadataExtractor
from ..utils.contents_leaders import contents_page_text

# How many pages of front matter to read when nothing says otherwise. Ten
# covers the deepest copyright page seen in the sample corpus (p5 of a
# Springer volume) with room to spare, and costs only text-layer reads.
DEFAULT_FRONT_MATTER_PAGES = 10

# Below this, a DOI found after page 1 is treated as a citation, not as this
# document's own identifier. See rule 2 in the module docstring.
MIN_DEEP_DOI_CONFIDENCE = 0.95


@dataclass
class FrontMatterIdentifiers:
    """
    What the first few pages declare about the document's identity.

    Every identifier carries the 1-based page it was found on, so a later
    classifier can tell "ISSN on the series page" from "ISSN in a reference".

    Attributes:
        doi / doi_confidence / doi_page: the document's DOI, its regex
            confidence (0.98 labelled, 0.95 URL, 0.85 bare) and its page.
        isbn / isbn_confidence / isbn_page: ISBN-10 or ISBN-13, digits only.
        issn / issn_page: ISSN in its canonical ``NNNN-NNNX`` form.
        pages_scanned: how many pages were actually read.
        page_texts: text layer per 1-based page, so a caller (classification)
            can reuse it instead of re-opening the PDF.
    """
    doi: Optional[str] = None
    doi_confidence: float = 0.0
    doi_page: Optional[int] = None

    isbn: Optional[str] = None
    isbn_confidence: float = 0.0
    isbn_page: Optional[int] = None

    issn: Optional[str] = None
    issn_page: Optional[int] = None

    pages_scanned: int = 0
    page_texts: Dict[int, str] = field(default_factory=dict)

    def identifier_pages(self) -> Dict[str, int]:
        """The page each identifier came from, for provenance in the profile."""
        found = {"doi": self.doi_page, "isbn": self.isbn_page, "issn": self.issn_page}
        return {kind: page for kind, page in found.items() if page is not None}


def scan_front_matter(doc, max_pages: int = DEFAULT_FRONT_MATTER_PAGES,
                      page_1_text: Optional[str] = None) -> FrontMatterIdentifiers:
    """
    Read the first `max_pages` pages and collect DOI, ISBN and ISSN.

    Args:
        doc: an open `PdfDocument` from the PdfBackend seam.
        max_pages: how many leading pages to read (clamped to the document).
        page_1_text: page 1's text if the caller already has it, so a PDF is
            not read twice for the same page.

    Returns:
        A `FrontMatterIdentifiers` holding the best of each identifier and the
        page it came from. All fields are None/0.0 when nothing was found.

    Example:
        >>> with backend.open(path) as doc:            # doctest: +SKIP
        ...     ids = scan_front_matter(doc)
        ...     ids.isbn, ids.isbn_page
        ('9783030809461', 5)
    """
    found = FrontMatterIdentifiers()
    extractor = MetadataExtractor()

    found.page_texts = _read_front_matter_text(doc, max_pages, page_1_text)
    found.pages_scanned = len(found.page_texts)

    for page_number in sorted(found.page_texts):
        text = found.page_texts[page_number]
        if not text:
            continue
        _take_doi(found, extractor, text, page_number)
        _take_isbn(found, extractor, text, page_number)
        _take_issn(found, text, page_number)

    return found


# --- helpers (after the main function, per CLAUDE.md) -----------------------

# An ISSN is eight characters printed as 'NNNN-NNNC', where the check
# character C may be an X. The LABEL IS REQUIRED, and that is not fussiness:
# the bare shape matches far too much real page furniture. Measured over the
# sample corpus, an unlabelled pattern read a year range ("2014-2015") and the
# leading half of an ORCID ("0000-0003") as ISSNs. Since a downstream
# classifier will weigh "has an ISSN" as evidence, a false one is worse than
# none — so we only accept an ISSN the document explicitly calls one.
ISSN_PATTERN = re.compile(
    r'\b(?:e-|p-|E-|P-)?ISSN(?:-L)?[:\s]*([0-9]{4}[-\s]?[0-9]{3}[0-9Xx])\b',
    re.IGNORECASE)


def _read_front_matter_text(doc, max_pages: int,
                            page_1_text: Optional[str]) -> Dict[int, str]:
    """Text layer of the leading pages, keyed by 1-based page number."""
    limit = max(1, min(max_pages, doc.page_count))
    texts: Dict[int, str] = {}

    for index in range(limit):
        if index == 0 and page_1_text is not None:
            texts[1] = page_1_text
            continue
        try:
            page = doc.page(index)
            texts[index + 1] = _contents_or_own_text(page)
        except Exception:
            # A single unreadable page must not abort the scan; the remaining
            # pages may still carry the identifiers.
            texts[index + 1] = ""

    return texts


def _contents_or_own_text(page) -> str:
    """
    A contents page's geometrically rebuilt text; any other page's own text.

    The rebuild writes out the dot leaders that pypdfium2 5.x no longer
    extracts (see `utils/contents_leaders.py`), so the classifier and the
    contents parser read the same text under PDFium old and new. It returns
    None on every page without at least three leader lines, so those keep
    their text byte for byte. Measured on all 485 corpus pages under 4.30
    and 5.15.2: it fires on the Springer book's two contents pages and
    nowhere else. Cost: median ~10 ms per page, ~0.1-0.2 s per document.
    """
    rebuilt = contents_page_text(page)
    if rebuilt is not None:
        return rebuilt
    return page.text() or ""


def _take_doi(found: FrontMatterIdentifiers, extractor: MetadataExtractor,
              text: str, page_number: int) -> None:
    """Record a DOI, honouring page-1 precedence and the deep-scan floor."""
    if found.doi is not None:
        return

    doi, confidence = extractor.extract_doi_from_text(text)
    if not doi:
        return
    if page_number > 1 and confidence < MIN_DEEP_DOI_CONFIDENCE:
        # A bare DOI this deep is almost certainly a citation. Skip it.
        return

    found.doi, found.doi_confidence, found.doi_page = doi, confidence, page_number


def _take_isbn(found: FrontMatterIdentifiers, extractor: MetadataExtractor,
               text: str, page_number: int) -> None:
    """Record the first ISBN seen. Books print it on the copyright page."""
    if found.isbn is not None:
        return

    isbn, confidence = extractor.extract_isbn_from_text(text)
    if isbn:
        found.isbn, found.isbn_confidence, found.isbn_page = isbn, confidence, page_number


def _take_issn(found: FrontMatterIdentifiers, text: str, page_number: int) -> None:
    """
    Record the first ISSN seen.

    An ISSN is NOT proof of a journal article: a book *series* (Springer's
    "Advances in Experimental Medicine and Biology", say) carries one too.
    That false equivalence is exactly what makes V5's classifier call book
    front matter an article, so this scan records the fact and its page and
    leaves the interpretation to whoever classifies.
    """
    if found.issn is not None:
        return

    issn = _match_issn(text)
    if issn:
        found.issn, found.issn_page = issn, page_number


def _match_issn(text: str) -> Optional[str]:
    """Return a normalised 'NNNN-NNNX' ISSN, or None if the page has no labelled one."""
    labelled = ISSN_PATTERN.search(text)
    if not labelled:
        return None

    digits = labelled.group(1).replace("-", "").replace(" ", "").upper()
    if len(digits) == 8 and digits[:7].isdigit():
        return f"{digits[:4]}-{digits[4:]}"

    return None
