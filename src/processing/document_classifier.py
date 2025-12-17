# File: src/processing/document_classifier.py
# Version: 4.1 (Restored Balance)
"""
Phase 4: Hardened Document Classifier - Multi-Signal Scoring System
Classifies documents as: research_article, book_chapter, or book.
"""

import re
import logging
from typing import Dict, Tuple, Optional

try:
    import fitz
except ImportError:
    fitz = None

# Import the full set of V4 patterns
from .utils.regex_patterns import (
    TOC_KEYWORD_PATTERNS, CHAPTER_HEADING_PATTERNS, ABSTRACT_PATTERNS, 
    EXPLICIT_CHAPTER_PATTERNS, VOLUME_INFO_PATTERNS, JOURNAL_SOURCE_PATTERNS, 
    PEER_REVIEW_DATE_PATTERNS
)
from .extractors.toc_extractor import TOCExtractor # Assuming this exists from V3
from .utils.confidence_scorer import ConfidenceScorer

class DocumentClassifier:
    """
    V4.1 Balanced Document Classifier.
    Restores V3 heuristics while keeping V4's signal aggregation structure.
    """
    
    def __init__(self):
        self.toc_extractor = TOCExtractor()
        self.logger = logging.getLogger(__name__)

    def classify(self, pdf_doc: 'fitz.Document', page_1_text: Optional[str] = None) -> Tuple[Optional[str], float]:
        self.logger.info("Starting document classification (V4.1 Balanced Model)...")

        if not pdf_doc or len(pdf_doc) == 0:
            self.logger.error("Cannot classify an empty or invalid document.")
            return None, 0.0

        # Gather signals
        signals = self._analyze_text_content(pdf_doc, page_1_text)
        
        # Aggregate
        winner_type, confidence = self._aggregate_signals(signals, pdf_doc)
        
        self.logger.info(f"Final classification: '{winner_type}' with confidence {confidence:.2f}")
        return winner_type, confidence

    def _analyze_text_content(self, pdf_doc: fitz.Document, page_1_text: Optional[str] = None) -> Dict:
        """
        Gathers evidence by scanning text. Unchanged from V4, just collecting data.
        """
        signals = {
            'abstract_count': 0, 'chapter_heading_count': 0, 'contents_count': 0,
            'has_explicit_chapter': False, 'has_volume_info': False,
            'has_journal_source': False, 'has_peer_review_dates': False,
            'has_doi': False, 'has_isbn': False, 'has_issn': False
        }
        
        max_pages_to_scan = min(5, len(pdf_doc))
        
        for i in range(max_pages_to_scan):
            text = ""
            if i == 0 and page_1_text is not None:
                text = page_1_text
            else:
                try:
                    text = pdf_doc[i].get_text("text") or ""
                except Exception:
                    continue
            
            text_lower = text.lower()

            if any(re.search(p, text_lower, re.IGNORECASE) for p in ABSTRACT_PATTERNS):
                signals['abstract_count'] += 1
            
            # Count chapter headings (e.g. "Chapter 1")
            signals['chapter_heading_count'] += sum(len(re.findall(p, text_lower, re.IGNORECASE)) for p in CHAPTER_HEADING_PATTERNS)
            
            if any(re.search(p, text_lower, re.IGNORECASE) for p in TOC_KEYWORD_PATTERNS):
                signals['contents_count'] += 1
            
            if 'doi:' in text_lower or '10.' in text:
                signals['has_doi'] = True
            if 'isbn' in text_lower:
                signals['has_isbn'] = True
            if 'issn' in text_lower:
                signals['has_issn'] = True

            # V4 Patterns
            if not signals['has_explicit_chapter'] and any(p.search(text) for p in EXPLICIT_CHAPTER_PATTERNS):
                signals['has_explicit_chapter'] = True
            if not signals['has_volume_info'] and any(p.search(text) for p in VOLUME_INFO_PATTERNS):
                signals['has_volume_info'] = True
            if not signals['has_journal_source'] and any(p.search(text) for p in JOURNAL_SOURCE_PATTERNS):
                signals['has_journal_source'] = True
            if not signals['has_peer_review_dates'] and any(p.search(text) for p in PEER_REVIEW_DATE_PATTERNS):
                signals['has_peer_review_dates'] = True
                
        return signals

    def _aggregate_signals(self, signals: Dict, pdf_doc: fitz.Document) -> Tuple[str, float]:
        """
        Aggregates signals with restored V3 weighting balance.
        """
        scores = {'research_article': 0, 'book_chapter': 0, 'book': 0}
        page_count = len(pdf_doc)

        # --- 1. PAGE COUNT HEURISTICS (Restored from V3) ---
        if page_count < 5:
            scores['research_article'] += 20 # Short -> Article
        elif 5 <= page_count <= 40:
            scores['research_article'] += 10
            scores['book_chapter'] += 10     # Ambiguous
        elif page_count > 60:
            scores['book'] += 40             # Long -> Book
            scores['research_article'] -= 20 # Unlikely to be article

        # --- 2. IDENTIFIERS (Metadata) ---
        if signals['has_doi']:
            scores['research_article'] += 20
            scores['book_chapter'] += 10 # Chapters also have DOIs
        if signals['has_isbn']:
            scores['book_chapter'] += 30
            scores['book'] += 30
        if signals['has_issn']:
            scores['research_article'] += 25

        # --- 3. KEYWORDS (Content) ---
        if signals['abstract_count'] > 0:
            scores['research_article'] += 25
            scores['book_chapter'] += 5 # Chapters rarely have "Abstract" explicitly labelled
        
        if signals['has_peer_review_dates']: # Received/Accepted/Published
            scores['research_article'] += 40 # Very strong article signal

        if signals['contents_count'] > 0:
            scores['book'] += 30
        
        # --- 4. STRUCTURE & V4 PATTERNS (Tuned) ---
        
        # Journal Source info (e.g. "Source: Ecology...")
        if signals['has_journal_source']:
            scores['research_article'] += 50
        
        # Explicit Chapter (e.g. "Chapter 1")
        # Only apply if we ALSO have ISBN or low page count to confirm it's not a random reference.
        if signals['has_explicit_chapter']:
            if signals['has_isbn'] or page_count > 10:
                scores['book_chapter'] += 30
            else:
                scores['book_chapter'] += 10 # Weak signal in isolation

        # Volume Info (e.g. "Vol 12")
        if signals['has_volume_info']:
            scores['research_article'] += 5
            scores['book_chapter'] += 5

        # --- 5. LOGIC CHECKS (Tie-Breakers) ---
        
        # If it's definitely a book (TOC + High Page Count), kill article score
        if scores['book'] > 50:
            scores['research_article'] = 0
        
        # If strong article signals exist, dampen chapter score
        if scores['research_article'] > 60 and not signals['has_isbn']:
            scores['book_chapter'] -= 20

        self.logger.info(f"Final classification scores: {scores}")
        
        # --- 6. DETERMINE WINNER ---
        winner_type, winner_score = ConfidenceScorer.select_winner(scores)
        
        if winner_type is None or winner_score <= 0:
            self.logger.warning("No definitive classification. Defaulting to research_article.")
            return 'research_article', 0.5

        # Calculate confidence
        sorted_scores = sorted(scores.values(), reverse=True)
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0
        margin = winner_score - runner_up
        
        confidence = min(1.0, 0.5 + (margin / 50.0))
        
        return winner_type, round(confidence, 2)