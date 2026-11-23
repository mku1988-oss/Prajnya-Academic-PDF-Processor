"""
Metadata Extractor - Fast Metadata Extraction
Uses PyMuPDF + V3-Validated Regex Patterns

Extracts:
- ISBN (10-digit and 13-digit formats)
- DOI (multiple pattern variants)
- Chapter numbers (Arabic and Roman numerals)

All extraction uses regex patterns from V3 with 95%+ success rates.

Blueprint Reference: Section 1.3
V3 Reference: book_metadata_extractor.py lines 300-425
"""

# .\src\processing\extractors\metadata_extractor.py

import re
from typing import Union

from ..utils.regex_patterns import roman_to_arabic


class MetadataExtractor:
    """
    Fast metadata extraction using PyMuPDF and regex patterns.
    
    All patterns are V3-validated:
        - ISBN: 95% success rate on 80+ documents
        - DOI: 98% success rate on 100+ documents
        - Chapter numbers: 88% success rate on 30+ chapters
    
    No VLM needed - pure regex for speed!
    """
    
    def __init__(self):
        """
        Initialize metadata extractor and pre-compile regex patterns for performance.
        
        Performance Optimization:
            Pre-compiling patterns avoids re-compilation overhead on every call.
            Critical for batch processing (1000+ documents).
        """
        # ISBN Patterns (pre-compiled for performance)
        self.labeled_isbn_pattern = re.compile(
            r'(?:ISBN-13|ISBN-10|e-ISBN|ISBN)[:\s]+([0-9]{1,5}[-\s]?[0-9]{1,7}[-\s]?[0-9]{1,7}[-\s]?[0-9]{1,7}[-\s]?[0-9]{1})',
            re.IGNORECASE
        )
        self.isbn13_pattern = re.compile(
            r'\b(97[89][-\s]?[0-9]{1,5}[-\s]?[0-9]{1,7}[-\s]?[0-9]{1,7}[-\s]?[0-9]{1,7}[-\s]?[0-9]{1})\b'
        )
        self.isbn10_pattern = re.compile(
            r'\b([0-9]{1}[-\s]?[0-9]{3,5}[-\s]?[0-9]{3,5}[-\s]?[0-9X]{1})\b'
        )
        
        # DOI Patterns (pre-compiled for performance)
        self.labeled_doi_pattern = re.compile(
            r'doi[:\s]+([0-9]{2}\.[0-9]{4,}/[^\s]+)',
            re.IGNORECASE
        )
        self.url_doi_pattern = re.compile(
            r'(?:https?://)?(?:dx\.)?doi\.org/([0-9]{2}\.[0-9]{4,}/[^\s]+)',
            re.IGNORECASE
        )
        self.bare_doi_pattern = re.compile(
            r'\b(10\.[0-9]{4,}/[-._;()/:A-Za-z0-9]+)\b'
        )

        # Line-wrapped DOI repair (see _join_wrapped_dois). Narrow journals set
        # the DOI in a thin margin column and break it after the '/'.
        self.wrapped_doi_pattern = re.compile(
            r'(10\.[0-9]{4,}[-._;()/:A-Za-z0-9]*/)[ \t]*\r?\n[ \t]*(?=[-._;()/:A-Za-z0-9])'
        )

        # Chapter Number Patterns (pre-compiled for performance)
        self.arabic_chapter_pattern = re.compile(
            r'chapter[\s:]+([0-9]{1,2})\b',
            re.IGNORECASE
        )
        self.roman_chapter_pattern = re.compile(
            r'chapter[\s:]+(X{0,3}I{0,3}X{0,3}V?I{0,3})\b',
            re.IGNORECASE
        )
    
    def extract_isbn_from_text(self, text: str) -> tuple[Union[str, None], float]:
        """
        Extract ISBN from text using V3-validated patterns with confidence scoring.
        
        Supports:
            - ISBN-13: "ISBN-13: 978-0-123-45678-9"
            - ISBN-10: "ISBN-10: 0-12-345678-9"
            - e-ISBN: "e-ISBN: 978-0-123-45678-9"
        
        Validation:
            - Removes hyphens and spaces
            - Checks length (10 or 13 digits)
        
        Confidence Scoring (V3 Hybrid Approach):
            - 0.95: Labeled ISBN ("ISBN-13:", "ISBN-10:", "e-ISBN:")
            - 0.90: ISBN on cover page (within first 3 pages)
            - 0.85: ISBN in body text
            - 0.00: Not found
        
        Args:
            text: Text to search for ISBN
        
        Returns:
            (isbn, confidence): Tuple of ISBN string (or None) and confidence score (0.0-1.0)
        
        V3 Reference: book_metadata_extractor.py lines 320-340
        V3 Testing: 95% success rate on 80+ books/chapters
        
        Example:
            >>> extract_isbn_from_text("ISBN-13: 978-0-262-03384-8")
            ('978-0-262-03384-8', 0.95)
        """
        # Pattern 1: Labeled ISBN (highest confidence - 0.95)
        # Matches: "ISBN-13: 978...", "ISBN-10: 0...", "e-ISBN: ..."
        labeled_match = self.labeled_isbn_pattern.search(text)
        if labeled_match:
            # Clean ISBN: remove hyphens and spaces
            isbn = labeled_match.group(1).replace('-', '').replace(' ', '').strip()
            # Validate length (10 or 13 digits)
            if len(isbn) in (10, 13) and isbn.isdigit():
                return (isbn, 0.95)
        
        # Pattern 2: Bare ISBN (lower confidence - 0.85)
        # An unlabelled run of 10 or 13 digits is only an ISBN if its CHECK
        # DIGIT is right. Without that test any ten digits counted: Springer
        # Nature PDFs carry a hidden "1234567890():,;" string on every page,
        # which made a Nature Communications article score as a BOOK (S1
        # confirmation, session 33). Every candidate is tried, not just the
        # first, so a stray number before the real ISBN does not hide it.
        for match in self.isbn13_pattern.finditer(text):
            isbn = match.group(1).replace('-', '').replace(' ', '').strip()
            if is_valid_isbn(isbn):
                return (isbn, 0.85)

        # A bare ISBN-10 must ALSO be printed in its four groups ("0-262-03384-4").
        # Ten plain digits are a DOI suffix ("pnas.2220167120") or a font-test
        # string ("Vol.:(0123456789)"); one separator is a page range in a
        # reference list ("28523-28526"). All three were read as ISBNs.
        for match in self.isbn10_pattern.finditer(text):
            raw = match.group(1)
            isbn = raw.replace('-', '').replace(' ', '').strip().upper()
            if len(re.findall(r'[-\s]', raw)) == 3 and is_valid_isbn(isbn):
                return (isbn, 0.85)
        
        # No ISBN found
        return (None, 0.0)
    
    def extract_doi_from_text(self, text: str) -> tuple[Union[str, None], float]:
        """
        Extract DOI from text using V3-validated patterns with confidence scoring.
        
        Supports:
            - Labeled: "doi: 10.1234/abcd.5678"
            - URL: "https://doi.org/10.1234/abcd.5678"
            - Direct: "10.1234/abcd.5678"
        
        Cleanup:
            - Removes trailing punctuation (., ;, etc.)
        
        Confidence Scoring (V3 Hybrid Approach):
            - 0.98: Labeled DOI ("doi:", "DOI:") - Highest confidence
            - 0.95: URL format ("https://doi.org/", "http://dx.doi.org/")
            - 0.90: DOI in header/footer region
            - 0.85: Bare DOI in body text
            - 0.00: Not found
        
        Args:
            text: Text to search for DOI
        
        Returns:
            (doi, confidence): Tuple of DOI string (or None) and confidence score (0.0-1.0)
        
        V3 Reference: book_metadata_extractor.py lines 350-370, pdf_processing.py lines 330-450
        V3 Testing: 98% success rate on 100+ articles
        
        Example:
            >>> extract_doi_from_text("doi: 10.1038/s41578-020-00251-2")
            ('10.1038/s41578-020-00251-2', 0.98)
        """
        # Step 0: repair DOIs broken across a line break (see _join_wrapped_dois)
        text = self._join_wrapped_dois(text)

        # Pattern 1: Labeled DOI (highest confidence - 0.98)
        # Matches: "doi: 10.xxxx", "DOI: 10.xxxx", "doi:10.xxxx"
        labeled_match = self.labeled_doi_pattern.search(text)
        if labeled_match:
            doi = labeled_match.group(1).strip().rstrip('.,;:)]}>')
            return (self._whole_doi(doi, text), 0.98)

        # Pattern 2: URL DOI (high confidence - 0.95)
        # Matches: "https://doi.org/10.xxxx", "http://dx.doi.org/10.xxxx"
        url_match = self.url_doi_pattern.search(text)
        if url_match:
            doi = url_match.group(1).strip().rstrip('.,;:)]}>')
            return (self._whole_doi(doi, text), 0.95)

        # Pattern 3: Bare DOI (lower confidence - 0.85)
        # Matches: "10.1234/abcd" anywhere in text
        # V3 core pattern: 10.dddd/[allowed chars]
        bare_match = self.bare_doi_pattern.search(text)
        if bare_match:
            doi = bare_match.group(1).strip().rstrip('.,;:)]}>')
            return (self._whole_doi(doi, text), 0.85)
        
        # No DOI found
        return (None, 0.0)

    def _join_wrapped_dois(self, text: str) -> str:
        """
        Re-join a DOI that the journal's typesetter broke across two lines.

        For biology researchers: some journals (Nature Reviews, for one) print
        the DOI down a narrow column in the bottom-left margin, so it wraps:

            https://doi.org/10.1038/
            s41578-020-00251-2

        Every DOI pattern here needs at least one character after the '/', so
        the wrapped form matches nothing and the paper looks like it has no
        DOI at all — which costs a VLM call to recover something that was
        sitting in the text layer the whole time.

        We only repair the UNAMBIGUOUS case: a line break falling immediately
        after the '/' of a `10.dddd/` prefix, where the next line resumes with
        a DOI-legal character. A break elsewhere (mid-suffix, or at a hyphen)
        is NOT repaired, because a real hyphen and a hyphenation hyphen are
        indistinguishable here and joining the wrong pair would invent a DOI
        that does not exist. A missed DOI is recoverable downstream; a
        fabricated one is not.

        Args:
            text: raw text-layer content of a page

        Returns:
            The same text with wrapped DOIs re-joined.

        Example:
            >>> MetadataExtractor()._join_wrapped_dois("doi.org/10.1038/\\ns41578-1")
            'doi.org/10.1038/s41578-1'
        """
        return self.wrapped_doi_pattern.sub(r'\1', text)

    def _whole_doi(self, doi: str, text: str) -> str:
        """
        The longer DOI the page prints whole, when the one found was cut short.

        For biology researchers: PNAS prints "doi:10.1073/pnas." at the end of
        a line in its supplement link, with "2220167120" on the next line, and
        prints "https://doi.org/10.1073/pnas.2220167120" whole in its footer.
        The labelled pattern is tried first and found "10.1073/pnas" (ledger
        T33). When another DOI on the same text begins with the one found and
        continues it with ".", "-" or "_", that whole one is the DOI. Nothing
        is joined across lines: only a DOI printed whole is ever used.

        Args:
            doi: the DOI a pattern found, trailing punctuation removed.
            text: the text it was found in.

        Returns:
            The longest DOI in `text` that continues `doi`, or `doi` itself.
        """
        longer = [candidate.rstrip('.,;:)]}>') for candidate in self.bare_doi_pattern.findall(text)]
        longer = [candidate for candidate in longer
                  if len(candidate) > len(doi) and candidate.startswith(doi)
                  and candidate[len(doi)] in '.-_']
        return max(longer, key=len) if longer else doi

    def extract_chapter_number(self, text: str) -> tuple[Union[int, None], float]:
        """
        Extract chapter number from text with confidence scoring.
        
        Handles:
            - Arabic numerals: "Chapter 5" → 5
            - Roman numerals: "Chapter V" → 5
            - Uppercase/lowercase variations
        
        Patterns:
            - "Chapter 5"
            - "CHAPTER V"
            - "Chapter III"
        
        Confidence Scoring:
            - 0.95: Pattern like "Chapter 5" or "Chapter V" is found
            - 0.00: Not found
        
        Note:
            Context-based confidence adjustments (e.g., based on page number)
            should be handled by the calling classifier or orchestrator.
            This function only detects the presence of chapter patterns.
        
        Args:
            text: Text to search (typically from first page)
        
        Returns:
            (chapter_number, confidence): Tuple of chapter number (or None) and confidence score
        
        V3 Reference: book_metadata_extractor.py lines 400-425
        V3 Testing: 88% success rate on 30+ chapters
        
        Example:
            >>> extract_chapter_number("Chapter III: Introduction")
            (3, 0.95)
        """
        # Pattern 1: "Chapter" keyword with Arabic numeral (confidence 0.95)
        arabic_match = self.arabic_chapter_pattern.search(text)
        if arabic_match:
            chapter_num = int(arabic_match.group(1))
            return (chapter_num, 0.95)
        
        # Pattern 2: "Chapter" keyword with Roman numeral (confidence 0.95)
        roman_match = self.roman_chapter_pattern.search(text)
        if roman_match:
            roman_str = roman_match.group(1).upper()
            # Convert roman to arabic using util function
            chapter_num = roman_to_arabic(roman_str)
            if chapter_num:
                return (chapter_num, 0.95)
        
        # No chapter number found
        return (None, 0.0)


# --- helpers (after the class, per CLAUDE.md) --------------------------------

def is_valid_isbn(digits: str) -> bool:
    """
    Whether a string of 10 or 13 characters is a real ISBN (its check digit is right).

    For biology researchers: the last character of an ISBN is computed from the
    others, like a DOI's checksum would be if it had one. A random run of ten
    digits passes this test only about one time in eleven, so it separates a
    real ISBN from a page number, a phone number or a font-test string.

        ISBN-10: sum of digit x weight (weights 10 down to 1) divisible by 11;
                 the last character may be "X", meaning 10.
        ISBN-13: digits weighted 1, 3, 1, 3, ... sum divisible by 10.

    Example:
        >>> is_valid_isbn("0262033844"), is_valid_isbn("1234567890")
        (True, False)
    """
    if len(digits) == 13 and digits.isdigit():
        total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits))
        return total % 10 == 0
    if len(digits) == 10 and digits[:9].isdigit() and (digits[9].isdigit() or digits[9] == "X"):
        values = [int(d) for d in digits[:9]] + [10 if digits[9] == "X" else int(digits[9])]
        return sum(v * w for v, w in zip(values, range(10, 0, -1))) % 11 == 0
    return False
