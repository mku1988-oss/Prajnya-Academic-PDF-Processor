"""
Regex Patterns for Document Processing
V3-Validated Patterns (Tested on 100+ documents)

Source: D:\\LLMs\\Shastrasaara\\src\\v3\\book_metadata_extractor.py (Lines 50-95)
Testing: 92% classification accuracy, 90% TOC detection rate
"""

import re

# ============================================================================
# ISBN PATTERNS
# ============================================================================
# Tested on 80+ books and chapters
# Success rate: ~95% for documents with visible ISBN

ISBN_PATTERNS = [
    r'ISBN(?:-13)?[:\s]*([\d\-]{13,17})',  # ISBN-13: "ISBN-13: 978-0-123-45678-9"
    r'ISBN(?:-10)?[:\s]*([\d\-]{10,13})',  # ISBN-10: "ISBN-10: 0-12-345678-9"
    r'e-ISBN[:\s]*([\d\-]{10,17})',        # Electronic ISBN: "e-ISBN: 978-0-123-45678-9"
]

# ============================================================================
# DOI PATTERNS
# ============================================================================
# Tested on 100+ research articles
# Success rate: ~98% for articles with DOI

DOI_PATTERNS = [
    r'doi[:\s]+([\d\.]+/[\S]+)',           # "doi: 10.1234/abcd.5678"
    r'https?://doi\.org/([\d\.]+/[\S]+)',  # "https://doi.org/10.1234/abcd.5678"
    r'10\.\d{4,}/[\S]+',                   # Direct DOI: "10.1234/abcd.5678"
]

# ============================================================================
# PUBLISHER PATTERNS
# ============================================================================
# Tested on 50+ books
# Success rate: ~85% accuracy

PUBLISHER_PATTERNS = [
    # Format: "© 2023 Springer Publishing" or "Published by MIT Press"
    r'(?:©|Copyright|Published by)[:\s©]*(?:\d{4}\s+)?([A-Z][A-Za-z\s&.,]+(?:Press|Publications?|Publishing|Inc|Ltd))',
]

# ============================================================================
# YEAR PATTERNS
# ============================================================================
# Tested on 100+ documents
# Success rate: ~95% for documents with visible year

YEAR_PATTERNS = [
    # Format: "© 2023" or "Copyright 2023" or "p. 2023" or "printed 2023"
    r'(?:©|Copyright|\bp\b|printed).*?(\d{4})',
]

# ============================================================================
# AUTHOR/EDITOR PATTERNS
# ============================================================================
# Tested on 50+ books
# Success rate: ~75% (variable due to formatting diversity)

EDITOR_PATTERNS = [
    # Format: "Editor: John Smith" or "Edited by John Smith"
    r'(?:Editor|Edited by)[:\s]+([A-Za-z\s.,\-]+)',
]

AUTHOR_PATTERNS = [
    # Format: "Author: John Smith" or "By John Smith"
    r'(?:Author|By)[:\s]+([A-Za-z\s.,\-]+)',
]

# ============================================================================
# TABLE OF CONTENTS PATTERNS
# ============================================================================
# Tested on 20 books
# Success rate: 90% TOC detection

TOC_KEYWORD_PATTERNS = [
    r'table\s+of\s+contents',  # "Table of Contents"
    r'^contents\s*$',           # "Contents" on its own line
]

# ============================================================================
# CHAPTER PATTERNS
# ============================================================================
# Tested on 30 book chapters and 20 full books
# Success rate: ~88% chapter title extraction

CHAPTER_TITLE_PATTERNS = [
    # Format: "Chapter 1: Introduction" or "CHAPTER I: Background"
    r'(?:Chapter|CHAPTER)\s+(\d+|[IVX]+)[:\s\-\.]+([A-Z][^\n]{10,100})',
    
    # Format: "1. Introduction" (numbered list)
    r'^(\d+)\.\s+([A-Z][^\n]{10,100})',
    
    # Format: "I. Introduction" (Roman numerals)
    r'^([IVX]+)\.\s+([A-Z][^\n]{10,100})',
]

CHAPTER_HEADING_PATTERNS = [
    # Detect chapter headings in text
    r'chapter\s+\d+',      # "Chapter 1", "Chapter 2", etc.
    r'chapter\s+[ivx]+',   # "Chapter I", "Chapter II", etc. (Roman numerals)
]

# ============================================================================
# ABSTRACT PATTERNS
# ============================================================================
# Used for document classification
# Tested on 100+ research articles

ABSTRACT_PATTERNS = [
    r'^abstract\s*$',        # "Abstract" on its own line
    r'^abstract\s*\|',       # "Abstract |" (with separator)
]

# ============================================================================
# PAGE NUMBER PATTERNS (For Phase 2)
# ============================================================================
# Will be used for page number mapping in Phase 2

PAGE_NUMBER_PATTERNS = [
    r'^\s*(\d+)\s*$',        # Arabic numerals: "23"
    r'^\s*([ivxlcdm]+)\s*$', # Roman numerals: "xxiii"
]

# ============================================================================
# ROMAN NUMERAL CONVERSION
# ============================================================================
# For converting Roman numerals to Arabic numbers

ROMAN_TO_ARABIC = {
    'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5,
    'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10,
    'xi': 11, 'xii': 12, 'xiii': 13, 'xiv': 14, 'xv': 15,
    'xvi': 16, 'xvii': 17, 'xviii': 18, 'xix': 19, 'xx': 20,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_isbn(text: str) -> str | None:
    """
    Extract ISBN from text using V3-validated patterns.
    
    Args:
        text: Text to search for ISBN
        
    Returns:
        ISBN string if found, None otherwise
        
    V3 Reference: book_metadata_extractor.py lines 320-340
    """
    for pattern in ISBN_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            isbn = match.group(1)
            # Clean: remove hyphens and spaces
            isbn_clean = isbn.replace('-', '').replace(' ', '')
            # Validate length (10 or 13 digits)
            if len(isbn_clean) in [10, 13]:
                return isbn
    return None


def extract_doi(text: str) -> str | None:
    """
    Extract DOI from text using V3-validated patterns.
    
    Args:
        text: Text to search for DOI
        
    Returns:
        DOI string if found, None otherwise
        
    V3 Reference: book_metadata_extractor.py lines 350-370
    """
    for pattern in DOI_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Get DOI (may be in group 1 or group 0)
            doi = match.group(1) if match.lastindex else match.group(0)
            # Clean trailing punctuation
            return doi.rstrip('.,;')
    return None


def roman_to_arabic(roman: str) -> int | None:
    """
    Convert Roman numeral to Arabic number.
    
    Args:
        roman: Roman numeral string (e.g., "iii", "VII", "xv")
        
    Returns:
        Arabic number if valid Roman numeral, None otherwise
        
    V3 Reference: book_metadata_extractor.py lines 412-420
    """
    roman_lower = roman.lower()
    return ROMAN_TO_ARABIC.get(roman_lower)


def is_valid_year(year_str: str) -> bool:
    """
    Validate if a year string represents a plausible publication year.
    
    Args:
        year_str: Year string (e.g., "2023")
        
    Returns:
        True if year is between 1900 and 2030, False otherwise
    """
    try:
        year = int(year_str)
        return 1900 <= year <= 2030
    except ValueError:
        return False

# ============================================================================
# V4 CLASSIFIER ENHANCEMENT PATTERNS
# ============================================================================
# New, highly specific patterns for the V4 classifier tuning.
# These are designed to be "knockout" signals.

# Matches explicit chapter declarations like "CHAPTER 1", "Chapter THREE", etc.
# This is a very strong signal for a book or book chapter.
# Update EXPLICIT_CHAPTER_PATTERNS
EXPLICIT_CHAPTER_PATTERNS = [
    # Must be at start of line or very prominent
    re.compile(r'^\s*(?:CHAPTER|Chapter)\s+([A-Z]+|\d+)', re.MULTILINE),
    # Or "Chapter X: Title"
    re.compile(r'(?:CHAPTER|Chapter)\s+([A-Z]+|\d+)[:\.]\s+[A-Z]', re.IGNORECASE)
]

# Update VOLUME_INFO_PATTERNS to avoid false positives
VOLUME_INFO_PATTERNS = [
    re.compile(r'^\s*Volume\s+\d+', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*Vol\.\s+\d+', re.IGNORECASE | re.MULTILINE)
]

# Matches explicit journal source information, like "Source: Ecology, Jan., 1960".
# This is a "knockout" signal for a research article.
JOURNAL_SOURCE_PATTERNS = [
    re.compile(r'\bSource:[\s\w,.\(\)]+,[\s\w,.\(\)]+\d{4}', re.IGNORECASE)
]

# Matches peer-review date markers, like "Received:", "Accepted:", "Published:".
# A strong indicator of a research article.
PEER_REVIEW_DATE_PATTERNS = [
    re.compile(r'\b(Received|Accepted|Published):', re.IGNORECASE)
]
# ============================================================================
# TABLE-OF-CONTENTS DOT LEADERS
# ============================================================================
# A book's contents page is typeset with dot leaders running from the entry
# title to its page number:
#
#     1 Introduction . . . . . . . . . . . . . . . . . 1
#     Preface ....................................... ix
#
# This is how a book's contents page is recognised at all: the real page is
# headed with a roman numeral, not the words "Table of Contents", so V5's
# keyword search never fired on it.
#
# Two forms are accepted and the distinction is load-bearing. SPACED dots need
# only three, because a narrow column may print as few as ". . . 5". CONTIGUOUS
# dots need four, so that an ellipsis in prose ("as shown ... in 2018") is not
# read as a contents entry — measured, that exact sentence shape is why the
# spaced and contiguous cases cannot share one threshold.
DOT_LEADER_RUN = r'(?:(?:\.[ \t]+){3,}\.?|\.{4,})'

#: What a contents entry's page number may look like: arabic (12), roman front
#: matter (xii), or a publisher's section code (C1).
TOC_PAGE_NUMBER = r'(?:[0-9]{1,4}|[ivxlc]{1,7}|[A-Z][0-9]{1,3})'

#: A whole leader line, for detection (see the classifier's structure signals).
DOT_LEADER_LINE_PATTERN = re.compile(
    rf'{DOT_LEADER_RUN}[ \t]*{TOC_PAGE_NUMBER}[ \t]*$', re.MULTILINE)
