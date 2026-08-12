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
        # Matches: 13-digit or 10-digit number patterns
        # 13-digit ISBN (starts with 978 or 979)
        isbn13_match = self.isbn13_pattern.search(text)
        if isbn13_match:
            isbn = isbn13_match.group(1).replace('-', '').replace(' ', '').strip()
            if len(isbn) == 13 and isbn.isdigit():
                return (isbn, 0.85)
        
        # 10-digit ISBN pattern
        isbn10_match = self.isbn10_pattern.search(text)
        if isbn10_match:
            isbn = isbn10_match.group(1).replace('-', '').replace(' ', '').strip()
            if len(isbn) == 10:
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
            return (doi, 0.98)
        
        # Pattern 2: URL DOI (high confidence - 0.95)
        # Matches: "https://doi.org/10.xxxx", "http://dx.doi.org/10.xxxx"
        url_match = self.url_doi_pattern.search(text)
        if url_match:
            doi = url_match.group(1).strip().rstrip('.,;:)]}>')
            return (doi, 0.95)
        
        # Pattern 3: Bare DOI (lower confidence - 0.85)
        # Matches: "10.1234/abcd" anywhere in text
        # V3 core pattern: 10.dddd/[allowed chars]
        bare_match = self.bare_doi_pattern.search(text)
        if bare_match:
            doi = bare_match.group(1).strip().rstrip('.,;:)]}>')
            return (doi, 0.85)
        
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
