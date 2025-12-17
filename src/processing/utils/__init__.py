#.\src\processing\utils\__init__.py
"""
Utils Package - Processing Utilities
"""

# This file correctly imports from its own sub-module, regex_patterns.py
from .regex_patterns import (
    # V3 Patterns
    ISBN_PATTERNS, DOI_PATTERNS, PUBLISHER_PATTERNS,
    YEAR_PATTERNS, EDITOR_PATTERNS, AUTHOR_PATTERNS,
    TOC_KEYWORD_PATTERNS, CHAPTER_TITLE_PATTERNS,
    CHAPTER_HEADING_PATTERNS, ABSTRACT_PATTERNS,
    extract_isbn, extract_doi, roman_to_arabic, is_valid_year,
    
    # V4 Classifier Patterns
    EXPLICIT_CHAPTER_PATTERNS,
    VOLUME_INFO_PATTERNS,
    JOURNAL_SOURCE_PATTERNS,
    PEER_REVIEW_DATE_PATTERNS
)
from .confidence_scorer import ConfidenceScorer
from .reading_order import ReadingOrderReconstructor
from .visual_clusterer import VisualRegionClusterer

# This list correctly exports all the patterns and classes for the rest of the app to use
__all__ = [
    # Pattern constants
    'ISBN_PATTERNS', 'DOI_PATTERNS', 'PUBLISHER_PATTERNS',
    'YEAR_PATTERNS', 'EDITOR_PATTERNS', 'AUTHOR_PATTERNS',
    'TOC_KEYWORD_PATTERNS', 'CHAPTER_TITLE_PATTERNS',
    'CHAPTER_HEADING_PATTERNS', 'ABSTRACT_PATTERNS',
    'EXPLICIT_CHAPTER_PATTERNS',
    'VOLUME_INFO_PATTERNS',
    'JOURNAL_SOURCE_PATTERNS',
    'PEER_REVIEW_DATE_PATTERNS',
    
    # Helper functions
    'extract_isbn', 'extract_doi', 'roman_to_arabic', 'is_valid_year',
    
    # Classes
    'ConfidenceScorer',
    'ReadingOrderReconstructor',
    'VisualRegionClusterer',
]