"""
Document Orchestrator - Main Entry Point
Intelligent Document Processing Pipeline

Handles:
- Document type detection (article, chapter, book)
- Strategy-based processing
- Result aggregation

Blueprint Reference: Intelligent_Document_Orchestrator_v1_Practical_partial.md
V3 Reference: book_metadata_extractor.py
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional
import fitz  # PyMuPDF
import re
from PIL import Image
import pytesseract
import io


from .document_classifier import DocumentClassifier
from .extractors import MetadataExtractor
from .models import DocumentProfile, ProcessingResult
from .utils.image_utils import pixmap_to_pil
from .strategies import (
    ResearchArticleStrategy,
    BookChapterStrategy,
    BookStrategy
)

logger = logging.getLogger(__name__)

class DocumentOrchestrator:
    """
    Main orchestrator for intelligent document processing.
    
    Phase 1 Implementation:
        - Multi-signal classification
        - Strategy-based processing
        - Simple TOC extraction for books
    
    Architecture:
        Stage 1: First pass analysis (PyMuPDF - fast)
        Stage 2: Document classification (4 signals)
        Stage 3: Strategy dispatch and processing
        Stage 4: Result aggregation
    
    Usage:
        orchestrator = DocumentOrchestrator()
        result = orchestrator.process_document("paper.pdf")
    """
    
    def __init__(self):
        """
        Initialize the document orchestrator.
        
        Sets up:
            - Document classifier
            - Metadata extractor (single instance for performance)
            - Processing strategies
            - Logger
        """
        self.classifier = DocumentClassifier()
        self.metadata_extractor = MetadataExtractor()  # Instantiate once for all documents
        self.logger = logging.getLogger(__name__)
        
        # --- TESSERACT / OCR FALLBACK SETUP ---
        # Optional runtime config. These may be overridden by strategy YAML later.
        self.tesseract_enabled = True
        self.tesseract_cmd = None   # set if tesseract is not on PATH
        self.text_threshold = 200   # min characters in .get_text() to consider "has native text"
        self.tesseract_dpi = 300

        # Defensive initialization: detect tesseract availability
        try:
            # Only set tesseract_cmd if configured explicitly
            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            # quick smoke check
            _ = pytesseract.get_tesseract_version()
            self.is_tesseract_available = True
            self.logger.info("Tesseract detected and available for OCR fallback.")
        except Exception as e:
            self.is_tesseract_available = False
            self.logger.warning(f"Tesseract not available: {e}. OCR fallback disabled.")

        
        # Initialize strategies
        self.strategies = {
            'research_article': ResearchArticleStrategy(),
            'book_chapter': BookChapterStrategy(),
            'book': BookStrategy()
        }
    
    
    def process_document(self, pdf_path: Union[str, Path], pages_to_process: Optional[List[int]] = None, **kwargs) -> Dict:
        """
        Main entry point for document processing.
        kwargs will be passed directly to the strategy's process method.
        """
        start_time = time.time()
        errors = []
        
        self.logger.info("="*70)
        self.logger.info("DOCUMENT ORCHESTRATION START")
        self.logger.info("="*70)
        
        pdf_doc_ref = None
        profile = None # Define profile in the outer scope
        try:
            # ====================================================================
            # STAGE 1: First-pass analysis
            # ====================================================================
            self.logger.info(f"STAGE 1: First-pass analysis for {pdf_path}")
            profile = self._first_pass_analysis(pdf_path)
            # We don't need pdf_doc_ref anymore if we use `profile` in the finally block
            self.logger.info(f"  ✓ Pages: {profile.page_count}, Size: {profile.file_size} bytes")
            
            # ====================================================================
            # ====================================================================
            self.logger.info("Extracting Page-1 text with Tesseract fallback...")
            profile.page_1_text = self._get_initial_page_text(profile)
            self.logger.info(f"  ✓ Page-1 text extracted (len={len(profile.page_1_text)})")
            
            # ====================================================================
            # STAGE 2: Document classification
            # ====================================================================
            self.logger.info("STAGE 2: Document classification")
            classification_start = time.time()
            document_type, classification_confidence = self.classifier.classify(profile.pdf_doc, profile.page_1_text)

            profile.document_type = document_type
            profile.classification_confidence = classification_confidence
            
            classification_time = time.time() - classification_start
            self.logger.info(f"  ✓ Classified as: {document_type} (confidence: {classification_confidence:.2f})")
            
            # Classifier Override Heuristic
            if not profile.document_type or profile.classification_confidence < 0.70:
                if "jstor" in profile.page_1_text.lower() or "stable url" in profile.page_1_text.lower():
                    self.logger.warning(f"Low confidence, but JSTOR signal found. Overriding to 'research_article'.")
                    profile.document_type = 'research_article'
            
            # ====================================================================
            # STAGE 3: Title page detection
            # ====================================================================
            self.logger.info("STAGE 3: Title page detection")
            title_detection_start = time.time()
            if profile.document_type == 'research_article':
                self.logger.info("  → Research article detected: Using Smarter Title Page Search.")
                profile.title_page_index = self._fast_title_detection_article(profile)
            else:
                self.logger.info(f"  → {profile.document_type} detected: Using FULL detection (with VLM if needed)")
                profile.title_page_index = self._find_title_page_index(
                    profile.pdf_doc,
                    document_type=profile.document_type
                )
            
            title_detection_time = time.time() - title_detection_start
            self.logger.info(f"  ✓ Title page: Page {profile.title_page_index + 1}")
            
            # ====================================================================
            # STAGE 4: Route to appropriate strategy
            # ====================================================================
            self.logger.info(f"STAGE 4: Strategy dispatch ({profile.document_type})")
            strategy_start = time.time()
            
            strategy = self.strategies.get(profile.document_type)
            if not strategy:
                raise ValueError(f"No strategy found for document type: {profile.document_type}")
            
            self.logger.info(f"  → Executing {profile.document_type} strategy...")
            
            # PASS KWARGS TO STRATEGY
            processed_profile = strategy.process(profile, test_pages_override=pages_to_process, **kwargs)

            
            strategy_time = time.time() - strategy_start
            self.logger.info(f"  ✓ Strategy complete")
            
            # ====================================================================
            # STAGE 5: Aggregate results
            # ====================================================================
            self.logger.info("STAGE 5: Result aggregation")
            
            # The test script expects the full ProcessingResult object.
            # We no longer dismantle it here.
            final_processing_result = processed_profile.processing_result
            if final_processing_result and final_processing_result.errors:
                errors.extend(final_processing_result.errors)

            total_time = time.time() - start_time
            
            result = {
                'document_type': processed_profile.document_type,
                'classification_confidence': processed_profile.classification_confidence,
                'processing_result': final_processing_result, # <-- THIS IS THE FIX
                'processing_time': total_time,
                'errors': errors
            }

            self.logger.info("="*70)
            self.logger.info(f"✓ PROCESSING COMPLETE: {total_time:.2f}s total")
            # ... (timing logs are fine)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}", exc_info=True)
            errors.append(str(e))
            
            # Return a consistent error structure
            return {
                'document_type': None,
                'classification_confidence': 0.0,
                'processing_result': None,
                'processing_time': time.time() - start_time,
                'errors': errors
            }
        
        finally:
            # Use the `profile` object, which is guaranteed to be in scope.
            if profile and hasattr(profile, 'pdf_doc') and profile.pdf_doc:
                profile.pdf_doc.close()
                self.logger.info(f"Closed PDF document object for {pdf_path}.")
    
    def _get_initial_page_text(
        self,
        profile,
        text_threshold: int = 200,
        dpi: int = 300
    ):
        """
        Return the initial page text for classification heuristics:
          - Prefer embedded PyMuPDF text if longer than text_threshold
          - Otherwise, fall back to Tesseract OCR on the rendered first page (if available)
        """
        try:
            page = profile.pdf_doc.load_page(0)  # first page (0-indexed)
            native_text = page.get_text("text") or ""
            native_text = native_text.strip()

            if len(native_text) >= getattr(self, "text_threshold", 200):
                self.logger.debug("Using native PyMuPDF text for page 1 (length=%d).", len(native_text))
                return native_text

            # fallback to Tesseract if available
            if getattr(self, "is_tesseract_available", False):
                self.logger.info("Scanned document or low-text page detected on page 1. Running Tesseract OCR (dpi=%s).", self.tesseract_dpi)
                pix = page.get_pixmap(dpi=getattr(self, "tesseract_dpi", 300))
                pil_image = pixmap_to_pil(pix)
                ocr_text = pytesseract.image_to_string(pil_image)
                ocr_text = ocr_text.strip()
                self.logger.debug("Tesseract OCR length=%d", len(ocr_text))
                return ocr_text
            else:
                self.logger.warning("No native text on page 1 and Tesseract unavailable — returning empty string for initial page text.")
                return ""
        except Exception as e:
            self.logger.warning(f"_get_initial_page_text failed: {e}")
            return ""
    
    def _first_pass_analysis(self, pdf_path: Path) -> DocumentProfile:
        """
        Fast first-pass analysis using PyMuPDF.
        CORRECTED to ensure pdf_doc is correctly attached.
        """
        try:
            pdf_path = Path(pdf_path)
            
            # Open the PDF document using fitz
            pdf_doc = fitz.open(pdf_path)
            
            # Get basic file info
            page_count = len(pdf_doc)
            file_size = pdf_path.stat().st_size
            pdf_metadata = pdf_doc.metadata or {}
            
            # --- Perform quick regex scans ---
            # For this step, we need the text of the first page.
            # We get it once and pass it to the helper methods.
            first_page_text = ""
            if page_count > 0:
                first_page_text = pdf_doc[0].get_text("text") or ""

            doi, doi_confidence = self.metadata_extractor.extract_doi_from_text(first_page_text)
            isbn, isbn_confidence = self.metadata_extractor.extract_isbn_from_text(first_page_text) # ISBN is also often on page 1
            
            # Create the Pydantic model with all serializable data
            profile = DocumentProfile(
                pdf_path=pdf_path,
                page_count=page_count,
                file_size=file_size,
                pdf_metadata=pdf_metadata,
                doi=doi,
                doi_confidence=doi_confidence,
                isbn=isbn,
                isbn_confidence=isbn_confidence
            )
            
            # --- THIS IS THE CRITICAL LINE ---
            # Attach the live, non-serializable fitz.Document object to the profile.
            # This must be done AFTER the main Pydantic model is created.
            profile.pdf_doc = pdf_doc
            
            self.logger.info(f"First-pass analysis complete for {pdf_path.name}")
            return profile
        
        except Exception as e:
            self.logger.error(f"First-pass analysis failed for {pdf_path}: {e}", exc_info=True)
            # Ensure we close the doc if it was opened
            if 'pdf_doc' in locals() and pdf_doc:
                pdf_doc.close()
            raise # Re-raise the exception to be caught by the main process_document method
    
    def _calculate_title_score(self, page: fitz.Page, page_text: str) -> int:
        """
        Calculates a "title-likeness" score for a given page.
        Higher scores indicate a higher probability of being the title page.
        """
        score = 0
        page_text_lower = page_text.lower()

        # --- Negative Signals (Strong Penalties) ---
        if 'jstor' in page_text_lower or 'stable url' in page_text_lower:
            score -= 50  # Very strong negative signal for a repository cover
            self.logger.info(f"    Page {page.number + 1}: -50 (JSTOR cover detected)")
        if 'references' in page_text_lower and 'abstract' not in page_text_lower:
            score -= 30
            self.logger.info(f"    Page {page.number + 1}: -30 (References page)")
        if len(page_text) < 300:
            score -= 10
            self.logger.info(f"    Page {page.number + 1}: -10 (Low text content)")

        # --- Positive Signals (Bonuses) ---
        if 'abstract' in page_text_lower:
            score += 40  # The strongest positive signal
            self.logger.info(f"    Page {page.number + 1}: +40 (Abstract found)")
        if 'introduction' in page_text_lower:
            score += 15
            self.logger.info(f"    Page {page.number + 1}: +15 (Introduction found)")
        if 'keywords' in page_text_lower:
            score += 10
            self.logger.info(f"    Page {page.number + 1}: +10 (Keywords found)")

        # --- Font Prominence Score (Most Powerful Heuristic) ---
        try:
            # Get detailed text block info with font sizes
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]
            if not blocks:
                return score

            # Find the most common font size (likely body text)
            from collections import Counter
            sizes = [span['size'] for block in blocks if 'lines' in block for line in block['lines'] for span in line['spans']]
            if not sizes:
                return score
            
            dominant_size = Counter(sizes).most_common(1)[0][0]
            
            # Find the largest font size on the page that is significantly larger than the body text
            largest_title_size = 0
            for s in set(sizes):
                if s > largest_title_size and s > (dominant_size * 1.3): # Must be at least 30% larger
                    largest_title_size = s
            
            if largest_title_size > 0:
                # The score is proportional to how much larger the title font is
                font_bonus = int((largest_title_size / dominant_size) * 10)
                score += font_bonus
                self.logger.info(f"    Page {page.number + 1}: +{font_bonus} (Font prominence: {largest_title_size:.1f}pt vs {dominant_size:.1f}pt)")

        except Exception as e:
            self.logger.warning(f"Could not analyze font sizes for page {page.number + 1}: {e}")

        return score

    def _fast_title_detection_article(self, profile: 'DocumentProfile') -> int:
        """
        Phase 4: Smarter Title Page Search.
        Scans the first 3 pages and selects the one with the highest "title score".
        """
        self.logger.info("  → Smarter title detection: Scoring first 3 pages...")
        page_scores = {}

        # Use the pre-computed page_1_text for the first page
        page_scores[0] = self._calculate_title_score(profile.pdf_doc[0], profile.page_1_text)
        self.logger.info(f"    → Page 1 Score: {page_scores[0]}")

        # For subsequent pages, we can use native text
        for i in range(1, min(3, profile.page_count)):
            page = profile.pdf_doc[i]
            text = page.get_text("text") or ""
            page_scores[i] = self._calculate_title_score(page, text)
            self.logger.info(f"    → Page {i + 1} Score: {page_scores[i]}")

        # The page with the maximum score is our winner
        if not page_scores:
            self.logger.warning("Could not score any pages for title detection; defaulting to page 1.")
            return 0
            
        best_page_index = max(page_scores, key=page_scores.get)
        self.logger.info(f"    ✓ Best title page candidate: Page {best_page_index + 1} (Score: {page_scores[best_page_index]})")
        
        return best_page_index
    
    def _find_title_page_index(self, pdf_doc: fitz.Document, scan_limit: int = 5, 
                              document_type: Optional[str] = None) -> int:
        """
        Smart title page detection with VLM validation (CONDITIONAL).
        
        Two-phase approach:
        1. Heuristic scoring (FAST - < 1 second)
        2. VLM validation (SLOW - only if confidence < 0.9)
        
        This is used for books/chapters where title page location varies.
        Research articles use _fast_title_detection_article() instead.
        
        ARCHITECTURE: VLM-Agnostic with Strategy Integration
        - Gets detection config from strategy (if available)
        - Falls back to defaults for strategies without config method
        
        Args:
            pdf_doc: PyMuPDF document object
            scan_limit: Number of pages to scan (default: 5)
            document_type: Optional document type for strategy lookup
            
        Returns:
            0-indexed page number of title page
        """
        self.logger.info(f"  → Smart title detection: Scanning first {min(scan_limit, len(pdf_doc))} pages...")
        
        # ====================================================================
        # ====================================================================
        self.logger.info("    PHASE 1: Heuristic scoring...")
        heuristic_start = time.time()
        
        page_scores = self._score_pages_generic(pdf_doc, scan_limit)
        
        heuristic_time = time.time() - heuristic_start
        self.logger.info(f"    ✓ Heuristic complete in {heuristic_time:.3f}s")
        self.logger.info(f"    Scores: {[(p+1, s) for p, s in page_scores.items()]}")
        
        # Get top candidate
        top_page, top_score = max(page_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence
        confidence = self._calculate_heuristic_confidence(top_score, page_scores)
        
        self.logger.info(f"    Top candidate: Page {top_page + 1} (score: {top_score}, confidence: {confidence:.2f})")
        
        # ====================================================================
        # ====================================================================
        if confidence >= 0.9:
            self.logger.info(f"    ✓ HIGH CONFIDENCE ({confidence:.2f}) - Skipping VLM validation")
            self.logger.info(f"    ⚡ Fast path: {heuristic_time:.3f}s total")
            return top_page
        
        # Low confidence - use VLM
        self.logger.warning(f"    ⚠ LOW CONFIDENCE ({confidence:.2f}) - Using VLM validation")
        self.logger.info("    PHASE 2: VLM validation...")
        
        vlm_start = time.time()
        
        try:
            # --- GET DETECTION CONFIG FROM STRATEGY (VLM-Agnostic) ---
            detection_config = None
            if document_type and document_type in self.strategies:
                strategy = self.strategies[document_type]
                if hasattr(strategy, 'get_title_detection_config'):
                    detection_config = strategy.get_title_detection_config()
                    self.logger.info(
                        f"    → Using detection config from {document_type} strategy: "
                        f"{detection_config.get('profile')} / {detection_config.get('task')}"
                    )
                else:
                    self.logger.warning(
                        f"    ⚠ Strategy '{document_type}' has no get_title_detection_config() method"
                    )
            else:
                self.logger.warning("    ⚠ No document_type provided, using legacy defaults")
            
            # Call VLM validation with config (or None for legacy fallback)
            verdict = self._vlm_validate_title_page(pdf_doc, top_page, detection_config)
            vlm_time = time.time() - vlm_start
            
            self.logger.info(f"    ✓ VLM verdict: {verdict} (took {vlm_time:.2f}s)")
            
            if verdict == "TITLE_PAGE":
                self.logger.info(f"    ✓ VLM confirmed: Page {top_page + 1}")
                return top_page
            elif verdict == "COVER_PAGE":
                # Try second best candidate
                sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_pages) > 1:
                    second_page = sorted_pages[1][0]
                    self.logger.info(f"    ✗ VLM rejected, trying Page {second_page + 1}...")
                    return second_page
                else:
                    self.logger.warning("    No fallback page, using top candidate anyway")
                    return top_page
            else:  # UNCERTAIN
                self.logger.warning(f"    ? VLM uncertain, using top candidate: Page {top_page + 1}")
                return top_page
                
        except Exception as e:
            vlm_time = time.time() - vlm_start
            self.logger.error(f"    ✗ VLM validation failed after {vlm_time:.2f}s: {e}")
            self.logger.warning(f"    Falling back to heuristic: Page {top_page + 1}")
            return top_page
    
    def _calculate_heuristic_confidence(self, top_score: int, all_scores: dict) -> float:
        """
        Calculate confidence score for heuristic title page detection.
        
        High confidence means:
        - Top score is strongly positive (> 5)
        - Large gap between top and second (> 3 points)
        - No other pages with similar scores
        
        Args:
            top_score: Score of top candidate
            all_scores: Dict of all page scores
            
        Returns:
            Confidence between 0.0 and 1.0
        """
        if not all_scores:
            return 0.0
        
        # Sort scores descending
        sorted_scores = sorted(all_scores.values(), reverse=True)
        
        confidence = 0.0
        
        # Factor 1: Absolute score strength (0.0-0.5)
        # Score >= 8 = 0.5, Score <= 0 = 0.0
        score_strength = min(0.5, max(0.0, top_score / 16))
        confidence += score_strength
        
        # Factor 2: Gap to second place (0.0-0.5)
        if len(sorted_scores) > 1:
            gap = sorted_scores[0] - sorted_scores[1]
            # Gap >= 5 = 0.5, Gap <= 0 = 0.0
            gap_strength = min(0.5, max(0.0, gap / 10))
            confidence += gap_strength
        else:
            # Only one page scored - full gap confidence
            confidence += 0.5
        
        return min(1.0, confidence)
    
    def _score_pages_generic(self, pdf_doc: fitz.Document, scan_limit: int) -> dict:
        """
        Generic heuristic scoring using patterns that work across ALL repositories.
        
        CRITICAL: 'abstract' gets highest priority - it's unique to title pages!
        
        Args:
            pdf_doc: The PyMuPDF document object
            scan_limit: Number of pages to score
            
        Returns:
            Dict mapping page_idx to score
        """
        page_scores = {}
        
        self.logger.info(f"      Scoring {min(scan_limit, len(pdf_doc))} pages...")
        
        for i in range(min(scan_limit, len(pdf_doc))):
            page = pdf_doc.load_page(i)
            text = page.get_text("text")
            text_lower = text.lower()
            
            score = 0
            signals = []  # Track what contributed to score
            
            # ========================================================================
            # POSITIVE SIGNALS (Research Article Title Page Indicators)
            # ========================================================================
            
            # HIGHEST PRIORITY: Abstract section (unique to title pages!)
            if re.search(r'\babstract\b', text_lower):
                score += 5  # Highest weight
                signals.append("+5 (abstract)")
            
            # Bold title detection (heuristic: multiple capital letters in sequence)
            # Title pages often have prominent titles with proper capitalization
            if re.search(r'[A-Z][a-z]+(\s+[A-Z][a-z]+){3,}', text):
                score += 3
                signals.append("+3 (prominent title)")
            
            # Introduction section
            if re.search(r'\bintroduction\b', text_lower):
                score += 2
                signals.append("+2 (introduction)")
            
            # Keywords section
            if re.search(r'\bkeywords?:\s', text_lower):
                score += 2
                signals.append("+2 (keywords)")
            
            # Author affiliations (number + institution pattern)
            if re.search(r'\d+\s*[A-Z][a-z]+.*[Uu]niversity', text):
                score += 2
                signals.append("+2 (affiliations)")
            
            # Date stamps (received/accepted/published)
            if re.search(r'\b(received|accepted|published):\s*\d{4}', text_lower):
                score += 2
                signals.append("+2 (date stamps)")
            
            # ========================================================================
            # NEGATIVE SIGNALS (Cover Page / Non-Title Page Indicators)
            # ========================================================================
            
            # References section without article content
            if re.search(r'^\s*references\s*$', text_lower, re.MULTILINE) and 'abstract' not in text_lower:
                score -= 4
                signals.append("-4 (references only)")
            
            # Link-heavy page (repository covers often have many URLs)
            if text_lower.count('http') > 3:
                score -= 2
                signals.append("-2 (many URLs)")
            
            # Too short to be title page (minimal content)
            if len(text) < 300:
                score -= 3
                signals.append("-3 (too short)")
            
            # Table of contents (chapter listings)
            if re.search(r'table of contents|chapter \d+', text_lower):
                score -= 4
                signals.append("-4 (TOC)")
            
            # ========================================================================
            # TEXT LENGTH BONUS (Title pages have moderate text length)
            # ========================================================================
            if 500 < len(text) < 3000:
                score += 1
                signals.append("+1 (optimal length)")
            
            page_scores[i] = score
            
            # Log page score with signals
            if signals:
                self.logger.info(f"        Page {i + 1}: score={score:+3d} [{', '.join(signals)}]")
            else:
                self.logger.info(f"        Page {i + 1}: score={score:+3d} [no signals]")
        
        return page_scores
    
    def _detect_actual_figures(self, page: fitz.Page, text: str) -> tuple[bool, int]:
        """
        Detect ACTUAL figures on the page, not just citations in text.
        
        Distinguishes between:
        - Figure CAPTIONS: "Figure 1. Diagram of..." (actual figure present)
        - Figure CITATIONS: "as shown in Figure 1" (reference to another page)
        
        Strategy:
        1. Look for visual elements (images, drawings) - strongest indicator
        2. Look for figure CAPTIONS (line-start pattern with description)
        3. Require BOTH visual content AND captions for high confidence
        
        Args:
            page: PyMuPDF page object
            text: Page text content
            
        Returns:
            Tuple of (has_figures: bool, figure_count: int)
        """
        # Method 1: Visual element detection (most reliable)
        images = page.get_images(full=True)
        drawings = page.get_drawings()
        
        # Significant visual content indicates figures
        has_visual_content = len(images) > 0 or len(drawings) > 100  # 100+ drawings = complex diagram
        
        # Method 2: Figure CAPTIONS (not citations)
        # Look for "Figure X." or "Figure X:" at start of line followed by descriptive text
        # This pattern matches captions, not in-text citations
        caption_pattern = r'^\s*Figure\s+\d+[\.:]\s+[A-Z]'
        figure_captions = re.findall(caption_pattern, text, re.MULTILINE | re.IGNORECASE)
        
        caption_count = len(figure_captions)
        
        # Decision logic:
        # - BOTH visual content AND captions = confident "has figures"
        # - Multiple captions (2+) without visuals = likely figure page (scanned diagrams)
        # - Visual content without captions = maybe (low confidence)
        # - Neither = definitely not a figure page
        
        if has_visual_content and caption_count > 0:
            # Best case: visual elements + proper captions
            return (True, caption_count)
        elif caption_count >= 2:
            # Multiple captions suggest figure page (even if visual detection failed)
            return (True, caption_count)
        elif has_visual_content and len(images) >= 2:
            # Multiple images without captions (maybe unlabeled figures)
            return (True, len(images))
        else:
            # No strong evidence of actual figures
            return (False, 0)
    
    def _vlm_validate_title_page(self, pdf_doc: fitz.Document, page_idx: int, 
                                 detection_config: Optional[Dict] = None) -> str:
        """
        Use VLM to visually validate if a page is a title page.
        
        ARCHITECTURE: VLM-Agnostic with Strategy-Provided Config
        - Accepts detection_config from strategy (profile + task)
        - Falls back to hardcoded defaults if strategy doesn't provide config
        - No hardcoded model/task names in primary path
        
        Args:
            pdf_doc: The PyMuPDF document object
            page_idx: 0-indexed page number to validate
            detection_config: Optional dict with 'profile' and 'task' keys
                             If None, uses legacy hardcoded defaults
            
        Returns:
            "TITLE_PAGE", "COVER_PAGE", or "UNCERTAIN"
            
        Example:
            detection_config = {
                'profile': 'olmocr-2-7b',
                'task': 'title_page_detection'
            }
        """
        self.logger.info(f"      → VLM validation starting for page {page_idx + 1}...")
        vlm_start = time.time()
        
        try:
            # Import here to avoid circular dependency
            from .strategies.research_article_strategy import ResearchArticleStrategy
            from ..llm.providers.universal_vlm_provider import UniversalVLMProvider
            
            # Render page to image (reuse existing method)
            strategy = ResearchArticleStrategy()
            page = pdf_doc.load_page(page_idx)
            
            self.logger.info("      → Rendering page image...")
            render_start = time.time()
            image = strategy._render_and_pad_image(page, target_size=1024, dpi=300)
            render_time = time.time() - render_start
            self.logger.info(f"      ✓ Rendering complete in {render_time:.2f}s")
            
            # --- VLM-AGNOSTIC INITIALIZATION ---
            # Use strategy-provided config OR fall back to defaults
            if detection_config:
                model_name = detection_config.get('profile')
                task_type = detection_config.get('task')
                self.logger.info(
                    f"      → Initializing VLM from strategy config: "
                    f"{model_name} / {task_type}"
                )
            else:
                # Legacy fallback (for strategies without get_title_detection_config)
                model_name = "olmocr-2-7b"
                task_type = "title_page_detection"
                self.logger.warning(
                    f"      ⚠ No detection config provided, using legacy defaults: "
                    f"{model_name} / {task_type}"
                )
            
            # Create lightweight VLM provider for detection
            provider_start = time.time()
            detector = UniversalVLMProvider(
                model_name=model_name,
                task_type=task_type
            )
            provider_time = time.time() - provider_start
            self.logger.info(f"      ✓ Provider initialized in {provider_time:.2f}s")
            
            # Quick classification (no base_text needed - pure visual)
            self.logger.info("      → Calling VLM API...")
            api_start = time.time()
            result = detector.process_page(image, base_text=None)
            api_time = time.time() - api_start
            self.logger.info(f"      ✓ VLM API complete in {api_time:.2f}s")
            
            # Parse response (simple text format)
            response = result.get('content', '').strip().upper()
            self.logger.info(f"      → Raw VLM response: '{response[:100]}'")
            
            # Validate and normalize response
            if "TITLE_PAGE" in response or "TITLE PAGE" in response:
                verdict = "TITLE_PAGE"
            elif "COVER_PAGE" in response or "COVER_PAGE" in response:
                verdict = "COVER_PAGE"
            else:
                verdict = "UNCERTAIN"
            
            total_time = time.time() - vlm_start
            self.logger.info(f"      ✓ VLM validation complete: {verdict} (total: {total_time:.2f}s)")
            self.logger.info(f"        ├─ Render: {render_time:.2f}s")
            self.logger.info(f"        ├─ Init: {provider_time:.2f}s")
            self.logger.info(f"        └─ API: {api_time:.2f}s")
            
            return verdict
            
        except Exception as e:
            total_time = time.time() - vlm_start
            self.logger.error(f"      ✗ VLM validation failed after {total_time:.2f}s: {e}")
            return "UNCERTAIN"
    
    def _extract_doi_quick(self, pdf_doc: fitz.Document) -> tuple[Union[str, None], float]:
        """
        Quick DOI extraction from first page with confidence scoring.
        
        Uses V3-validated regex patterns to find DOI in first page text.
        This is faster than VLM extraction and sufficient for classification.
        Returns confidence score for hybrid approach decision-making.
        
        Args:
            pdf_doc: PyMuPDF document object
        
        Returns:
            (doi, confidence): Tuple of DOI string (or None) and confidence score (0.0-1.0)
        
        V3 Reference: book_metadata_extractor.py lines 350-370
        Blueprint Reference: Section 1.6
        """
        try:
            # Extract text from first page only (fast)
            first_page = pdf_doc[0]
            text = first_page.get_text()
            
            # Use instance attribute (no need to instantiate)
            doi, confidence = self.metadata_extractor.extract_doi_from_text(text)
            
            return (doi, confidence)
        except Exception as e:
            self.logger.warning(f"DOI quick extraction failed: {e}")
            return (None, 0.0)
    
    def _extract_isbn_quick(self, pdf_doc: fitz.Document) -> tuple[Union[str, None], float]:
        """
        Quick ISBN extraction from first 3 pages with confidence scoring.
        
        Uses V3-validated regex patterns to find ISBN in early pages.
        Scans first 3 pages for better coverage of publisher information.
        Returns confidence score for hybrid approach decision-making.
        
        Args:
            pdf_doc: PyMuPDF document object
        
        Returns:
            (isbn, confidence): Tuple of ISBN string (or None) and confidence score (0.0-1.0)
        
        V3 Reference: book_metadata_extractor.py lines 320-340
        Blueprint Reference: Section 1.6
        """
        try:
            # Scan first 3 pages for ISBN (common locations)
            max_pages = min(3, len(pdf_doc))
            
            best_isbn = None
            best_confidence = 0.0
            
            for page_num in range(max_pages):
                page = pdf_doc[page_num]
                text = page.get_text()
                
                # Use instance attribute (no need to instantiate)
                isbn, confidence = self.metadata_extractor.extract_isbn_from_text(text)
                
                # Keep the ISBN with highest confidence
                if isbn and confidence > best_confidence:
                    best_isbn = isbn
                    best_confidence = confidence
            
            # Adjust confidence based on page number (earlier = higher)
            if best_isbn:
                # First page gets full confidence, later pages slightly reduced
                return (best_isbn, best_confidence)
            
            return (None, 0.0)
        except Exception as e:
            self.logger.warning(f"ISBN quick extraction failed: {e}")
            return (None, 0.0)
    
    def process_batch(self, pdf_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Process multiple documents in batch.
        
        Iterates through list of PDF paths and processes each one.
        Collects all results and returns as list.
        
        Args:
            pdf_paths: List of PDF file paths
        
        Returns:
            List of result dictionaries (one per document)
        
        Blueprint Reference: Section 1.6
        """
        self.logger.info(f"Starting batch processing: {len(pdf_paths)} documents")
        results = []
        
        for idx, pdf_path in enumerate(pdf_paths, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing document {idx}/{len(pdf_paths)}: {pdf_path}")
            self.logger.info(f"{'='*60}")
            
            try:
                result = self.process_document(pdf_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_path}: {e}")
                results.append({
                    'document_type': None,
                    'classification_confidence': 0.0,
                    'metadata': {},
                    'content': {},
                    'processing_time': 0.0,
                    'errors': [str(e)]
                })
        
        self.logger.info(f"\nBatch processing complete: {len(results)} documents processed")
        return results
    
    def process_directory(
        self, 
        directory: Union[str, Path], 
        pattern: str = "*.pdf"
    ) -> List[Dict]:
        """
        Process all PDFs in a directory.
        
        Finds all PDF files matching pattern in directory
        and processes them in batch.
        
        Args:
            directory: Path to directory containing PDFs
            pattern: Glob pattern for files (default: "*.pdf")
        
        Returns:
            List of result dictionaries
        
        Example:
            results = orchestrator.process_directory(
                "sample_pdfs/",
                pattern="*.pdf"
            )
        
        Blueprint Reference: Section 1.6
        """
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.error(f"Directory not found: {directory}")
            return []
        
        if not directory.is_dir():
            self.logger.error(f"Path is not a directory: {directory}")
            return []
        
        # Find all matching PDF files
        pdf_files = list(directory.glob(pattern))
        
        if not pdf_files:
            self.logger.warning(f"No files matching '{pattern}' found in {directory}")
            return []
        
        self.logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        # Process in batch
        return self.process_batch(pdf_files)
