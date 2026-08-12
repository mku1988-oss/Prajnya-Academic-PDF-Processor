"""
Research Article Strategy - Processing for Research Articles and Reviews
Uses existing validated VLM pipeline

Processing Flow:
1. Page 1: Title page metadata extraction (DOI, title, authors) at 1288px
2. Pages 2+: Body content extraction at 1024px

Blueprint Reference: Section 1.2
Version: See src.version.PIPELINE_VERSION for current version
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING, Any
import sys
import yaml
import logging
import time
import os
import json
import pytesseract
import fitz
from PIL import Image
import io
import re

# Add parent directory to path for imports
#   See: Response&Suggestions.txt - Suggestion 3
#   Action: Create pyproject.toml/setup.py, install with 'pip install -e .'
#   Benefits: Cleaner imports, easier distribution, standard Python packaging
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Type-only imports to avoid circular dependencies while enabling static analysis
# Benefits: IDE autocomplete, MyPy support, better type checking
if TYPE_CHECKING:
    from ..models import DocumentProfile, ProcessingResult, ExtractedMetadata, ContentPage

from ..models import ProcessingResult, ExtractedMetadata, ContentPage 
from .base_strategy import DocumentProcessingStrategy
from ...llm.providers.universal_vlm_provider import UniversalVLMProvider, ModelProfile, TaskDefinition
from ...llm.providers.async_universal_vlm_provider import AsyncUniversalVLMProvider
import asyncio
import fitz
from ..utils.anchor_extractor import DocumentAnchoringExtractor
from ..utils.page_classifier import PageClassifier, PageType
from ..utils.surgical_vlm_helper import SurgicalVLMHelper
from ..utils.reading_order import ReadingOrderReconstructor
from ..utils.image_utils import pixmap_to_pil
# GLM two-phase offline path (tier overhaul). These utilities are VLM-agnostic; all model-specific
# detail (names, prompts, grounding format) is resolved from config at call time.
from ..utils.content_drop_check import check_page_for_content_drop, is_scanned_page
from ..utils.table_region_detector import GroundingSpec, parse_grounding_output
from ..utils.table_repair import repair_page_tables
from ..utils.figure_extractor import extract_and_save_figures, append_figure_anchors, safe_doc_id
from ..utils.model_swap import RouterClient
from ..ledger.semantic_ledger import SemanticLedger, LedgerRecord, generate_hash
from ..ledger.csv_backend import CsvLedgerBackend
from ..utils.zotero_service import ZoteroService, normalize_doi_for_compare
from src.processing.utils.metadata_merger import merge_metadata
from src.processing.utils.author_utils import merge_author_lists
from PIL import Image
from datetime import datetime

class ResearchArticleStrategy(DocumentProcessingStrategy):
    """
    Processing strategy for research articles and reviews.
    
    Uses Existing Infrastructure:
        - title_page_metadata_extraction.yaml (Page 1, 1288px) ✅ Tested
        - olmocr_document_processing.yaml (Pages 2+, 1024px) ✅ Tested
    
    No changes needed for Phase 1! This strategy simply orchestrates
    the existing tested VLM pipeline.
    
    Metadata Extracted:
        - Title (from Page 1)
        - Authors (from Page 1)
        - DOI (from Page 1) - Primary identifier for Zotero
        - Full text content (from all pages)
    
    Zotero Integration:
        - Primary: DOI lookup
        - Fallback: Title + Authors
    """

    # White border (px per side) added by _render_page_pypdfium. Kept as one constant so the
    # renderer and the pixel->PDF-point caption mapping can never drift apart.
    RENDER_PADDING_PX = 25

    def __init__(self):
        """
        Initialize research article strategy with registry resolution.
        
        ARCHITECTURE: VLM-Agnostic with Generic Resolution
        - All model/task references resolved from registries
        - Change ONE line in YAML -> entire pipeline updates
        - No hardcoded model names anywhere in code
        """
        self.logger = logging.getLogger(__name__)
        
        # Load strategy configuration from YAML
        self.config = self._load_strategy_config()
        
        self.ledger: Optional[SemanticLedger] = None
        self.zotero_service: Optional[ZoteroService] = None
        
        zotero_config = self.config.get('zotero', {})
        if zotero_config.get('enabled', False):
            self.logger.info("Zotero integration is enabled. Initializing services.")
            
            # Initialize the Semantic Ledger (using CSV backend for now)
            ledger_path = Path(os.getenv("SARA_LEDGER_PATH", "persistence/semantic_ledger.csv"))
            self.ledger = CsvLedgerBackend(file_path=str(ledger_path.resolve()))
            
            # Initialize the Zotero Service
            try:
                cache_dir = Path(os.getenv("SARA_CACHE_PATH", "persistence/zotero_cache"))
                self.zotero_service = ZoteroService(config=zotero_config, cache_dir=str(cache_dir.resolve()))
            except ImportError:
                self.logger.error("Failed to initialize ZoteroService due to missing libraries. Continuing without Zotero.")
                self.zotero_service = None
        else:
            self.logger.info("Zotero integration is disabled in the configuration.")
        
        self.logger.info(
            f"Loaded strategy config: {self.config['strategy_name']} "
            f"v{self.config['version']} with registry resolution"
        )
        
        # ========================================================================
        # VLM PROVIDER CONFIGURATION (API Endpoint)
        # ========================================================================
        self.api_endpoint = self.config['vlm_provider'].get('api_endpoint')
        if self.api_endpoint:
            self.logger.info(f"Using configured API Endpoint: {self.api_endpoint}")

        # ========================================================================
        # BODY PAGE PROVIDER - TIER 3 (Single-Shot with Registry Resolution)
        # ========================================================================
        body_pages_config = self.config['vlm_provider']['body_pages']
        tier3_config = body_pages_config.get('tier3_single_shot', {})
        
        # Resolve references from registry (NO hardcoded defaults)
        body_provider_config = {
            'profile': self._resolve_reference(
                tier3_config.get('profile'),  # "@profile_registry.default"
                self.config
            ),
            'task_type': self._resolve_reference(
                tier3_config.get('task'),  # "@task_registry.legacy_ocr"
                self.config
            ),
            'resolution_override': 1024  # Standard body page resolution
        }
        
        self.logger.info(
            f"Body page provider (Tier 3): profile={body_provider_config['profile']}, "
            f"task={body_provider_config['task_type']}"
        )
        
        self.body_provider, self.body_vlm_profile, self.body_resolution = self._initialize_provider_from_config(
            body_provider_config,
            api_endpoint=self.api_endpoint
        )
        
        # Detect if we are using a Cloud API (e.g., Gemini, OpenRouter) to enable the simplified
        # "Early Branching" pipeline, bypassing the complex 5-Tier local router.
        self.use_cloud_pipeline = self.body_vlm_profile.get('model_type') in ['cloud_api_gemini', 'cloud_api_openai']
        if self.use_cloud_pipeline:
            self.logger.info("Cloud API Profile Detected: Enabling Simplified Hybrid Pipeline (Lite Router)")
        
        # ========================================================================
        # TITLE PAGE CONFIG (Resolution for Two-Shot Processing)
        # ========================================================================
        title_page_config = self.config['vlm_provider']['title_page']
        self.title_resolution = title_page_config.get('resolution_override', 1288)
        
        # Store resolved two-shot task names for title page
        # These will be used by _process_title_page()
        self.title_shot1_task = self._resolve_reference(
            title_page_config.get('shot1_task'),  # "@task_registry.raw_ocr"
            self.config
        )
        self.title_shot2_task = self._resolve_reference(
            title_page_config.get('shot2_task'),  # "@task_registry.structured_title"
            self.config
        )
        
        self.logger.info(
            f"Title page two-shot: shot1={self.title_shot1_task}, "
            f"shot2={self.title_shot2_task}"
        )


        # The rest of the __init__ method remains the same.
        # Store important config values for easy access
        self.doi_threshold = self.config['metadata']['doi_confidence_threshold']
        self.isbn_threshold = self.config['metadata']['isbn_confidence_threshold']
        self.continue_on_error = self.config['processing']['continue_on_error']
        self.min_quality_score = self.config['quality']['minimum_page_score']
        
        # Initialize 3-Tier Hybrid Processing components (if enabled)
        hybrid_config = self.body_vlm_profile.get('hybrid_processing', {})
        self.hybrid_enabled = hybrid_config.get('enabled', False)
        
        if self.hybrid_enabled:
            self.logger.info("3-Tier Hybrid Processing ENABLED - Initializing components...")
            self.page_classifier = PageClassifier(self.body_vlm_profile)
            config_base_path = Path(__file__).parent.parent.parent / 'llm' / 'providers'
            self.surgical_helper = SurgicalVLMHelper(str(config_base_path))
            self.logger.info("3-Tier components initialized successfully")
        else:
            self.logger.info("3-Tier Hybrid Processing DISABLED - Using VLM-only mode")
            self.page_classifier = None
            self.surgical_helper = None
        
        self.reconstructor = ReadingOrderReconstructor()
        self.logger.info("ReadingOrderReconstructor initialized for V2.")
        
        # Load Concurrency Config
        self.concurrency_config = self.config['vlm_provider'].get('concurrency', {})
        if self.concurrency_config.get('enabled', False):
             self.logger.info(f"AsyncIO Concurrency Enabled (Limit: {self.concurrency_config.get('max_concurrent', 'Auto')})")

        # Set up the GLM two-phase offline body path (no-op unless config selects it).
        self._init_glm_twophase()

        self.logger.info("Research article strategy initialized with config-driven settings")
    
    def process(self, profile: 'DocumentProfile', test_pages_override: Optional[List[int]] = None, concurrency_enabled: bool = True) -> 'DocumentProfile':
        """
        Executes the strategy to process a research article, correctly populating
        the DocumentProfile's processing_result field.
        
        Phase 4.6 Concurrency:
        If concurrency_enabled is True, body pages are processed using a Processing Map
        and parallel CPU/GPU threads.
        """
        all_pages: List[ContentPage] = []
        bibliographic_metadata = {}
        errors = []
        self.current_pdf_path = profile.pdf_path
        try:
            if self.use_cloud_pipeline:
                self.logger.info("Executing Cloud Hybrid Pipeline (Gemini)...")
                                
                # 1. Title Page (Single-Pass)
                title_page = self._process_title_page_gemini(profile)
                all_pages.append(title_page)
                
                # 2. Extract Metadata from Single-Pass Result
                # The metadata is already parsed into title_page.metadata by _process_title_page_gemini
                if title_page.metadata and 'generated_metadata' in title_page.metadata:
                     bibliographic_metadata = title_page.metadata['generated_metadata']
                     # Normalize keys to match ExtractedMetadata/Ledger expectations
                     if 'journal_name' in bibliographic_metadata and 'journal' not in bibliographic_metadata:
                         bibliographic_metadata['journal'] = bibliographic_metadata.pop('journal_name')
                     if 'publication_year' in bibliographic_metadata and 'year' not in bibliographic_metadata:
                         bibliographic_metadata['year'] = bibliographic_metadata.pop('publication_year')
                else:
                     # Fallback to standard extraction if TOML parsing failed
                     bibliographic_metadata = self._extract_metadata_from_page1(title_page, profile.pdf_path)

                # Same enrichment as local pipeline - uses Zotero to fill missing DOI, etc.
                try:
                    page_obj = profile.pdf_doc.load_page(profile.title_page_index)
                    title_page_raw_text = page_obj.get_text("text") or ""
                except Exception:
                    title_page_raw_text = ""

                if not title_page_raw_text.strip() and title_page:
                    self.logger.info("PyMuPDF text empty (scanned doc?), using VLM extracted text as referee.")
                    title_page_raw_text = title_page.content or ""

                # Enrich and merge metadata using cache + Zotero + referee text
                bibliographic_metadata = self._enrich_metadata_with_cache_and_zotero(
                    vlm_metadata=bibliographic_metadata,
                    pdf_path=profile.pdf_path,
                    title_page_text=title_page_raw_text,
                    profile=profile
                )

                # 3. Body Pages (Lite Router)
                if test_pages_override:
                    pages_to_process = [p - 1 for p in test_pages_override if p > 1]
                else:
                    pages_to_process = list(range(1, profile.page_count))
                
                body_pages = self._process_body_pages_gemini(profile, pages_to_process)
                all_pages.extend(body_pages)

            else:
                
                self.logger.info(f"Processing page 1/{profile.page_count} (title page metadata)...")
                title_page = self._process_title_page(profile)
                all_pages.append(title_page)
    
                # Extract key metadata immediately after processing the title page
                bibliographic_metadata = self._extract_metadata_from_page1(title_page, profile.pdf_path)
                
                # This is the core integration of the new workflow
                # Get raw title-page text via PyMuPDF (fast, already available)
                try:
                    page_obj = profile.pdf_doc.load_page(profile.title_page_index)
                    title_page_raw_text = page_obj.get_text("text") or ""
                except Exception:
                    title_page_raw_text = ""
    
                # The 'title_page' object (ContentPage) contains the text extracted by VLM Shot 1
                if not title_page_raw_text.strip() and title_page:
                    self.logger.info("PyMuPDF text empty (scanned doc?), using VLM extracted text as referee.")
                    title_page_raw_text = title_page.content or ""
                
                # Enrich and merge metadata using the title page raw text as referee
                bibliographic_metadata = self._enrich_metadata_with_cache_and_zotero(
                    vlm_metadata=bibliographic_metadata,
                    pdf_path=profile.pdf_path,
                    title_page_text=title_page_raw_text,
                    profile=profile  # Pass the full profile object
                )
                
                if test_pages_override:
                    pages_to_process = [p - 1 for p in test_pages_override if p > 1]
                    self.logger.info(f"Processing specific pages from test override: {test_pages_override}")
                else:
                    pages_to_process = list(range(1, profile.page_count))
    
                if pages_to_process and self.glm_twophase_enabled:
                    # GLM two-phase offline path: Phase A reads every page with the extraction
                    # model and repairs any dropped table (co-resident, no swap); Phase B swaps
                    # once to the understanding model to describe figures.
                    self.logger.info("Body pages: GLM two-phase offline path (extraction + repair + understanding)")
                    body_pages = self._process_body_pages_glm(profile, pages_to_process)
                    all_pages.extend(body_pages)
                elif pages_to_process:
                    # Check all three conditions: parameter, hybrid enabled, AND config enabled
                    async_vlm_enabled = self.concurrency_config.get('enabled', False)

                    if concurrency_enabled and self.hybrid_enabled and async_vlm_enabled:
                        self.logger.info(f"Tier 0: Classifying {len(pages_to_process)} body pages for concurrent processing...")
                        try:
                            processing_map = self._build_processing_map(profile, pages_to_process)
                            self.logger.info(f"Processing Map: {json.dumps(processing_map, indent=2)}")

                            body_pages = self._process_body_pages_concurrent(profile, processing_map)
                            all_pages.extend(body_pages)

                        except Exception as e:
                            self.logger.error(f"Concurrent processing failed (Tier 0/concurrent error): {e}", exc_info=True)
                            self.logger.warning("Falling back to sequential processing...")
                            body_pages = self._process_body_pages(profile, pages_to_process)
                            all_pages.extend(body_pages)
                    else:
                        # Sequential Fallback
                        if not concurrency_enabled:
                            self.logger.info("Concurrency disabled manually.")
                        if not self.hybrid_enabled:
                             self.logger.info("Hybrid processing disabled, skipping Tier 0.")
                        if not async_vlm_enabled:
                             self.logger.info("AsyncIO VLM concurrency disabled in config.")

                        body_pages = self._process_body_pages(profile, pages_to_process)
                        all_pages.extend(body_pages)

        except Exception as e:
            error_message = f"Strategy execution failed: {type(e).__name__} - {e}"
            self.logger.error(error_message, exc_info=True)
            errors.append(error_message)
        
        finally:
            # Strategy does NOT manage ledger persistence
            # Runner handles all ledger operations after review
            pass
            # THIS BLOCK ALWAYS RUNS:
            # It builds the final ProcessingResult object, which is the correct
            # way to store the output according to models.py.
            self.logger.info(f"Finalizing processing. Total pages processed: {len(all_pages)}")
            
            # Sanitize bibliographic metadata so empty strings become None and authors is always a list
            bibliographic_metadata = self._sanitize_bibliographic_metadata(bibliographic_metadata)
            final_metadata = ExtractedMetadata(**bibliographic_metadata)
            processing_output = ProcessingResult(
                metadata=final_metadata,
                pages=all_pages,
                errors=errors
            )

            # Assign the complete result to the correct field in the profile
            profile.processing_result = processing_output

        self.logger.info(f"Research article processing complete.")
        return profile
    
    def get_title_detection_config(self) -> dict:
        """
        Get configuration for title page detection task.
        
        Returns profile and task for VLM-based title page validation.
        Used by orchestrator for conditional title detection.
        
        ARCHITECTURE: VLM-Agnostic with Registry Resolution
        - Resolves @profile_registry.title_detection
        - Resolves @task_registry.title_detection
        - No hardcoded model/task names
        
        Returns:
            Dict with 'profile' and 'task' keys (resolved from registry)
            
        Example:
            config = strategy.get_title_detection_config()
            # config = {
            #     'profile': 'olmocr-2-7b',
            #     'task': 'title_page_detection'
            # }
        """
        return {
            'profile': self._resolve_reference(
                '@profile_registry.title_detection',
                self.config
            ),
            'task': self._resolve_reference(
                '@task_registry.title_detection',
                self.config
            )
        }
    
    def _process_title_page(self, profile: 'DocumentProfile') -> 'ContentPage':
        """
        Processes the detected title page using the PROPER two-shot approach from OlmOCR Hybrid v2.
        
        Two-Shot Architecture (Same as body pages, but with title-specific Shot 2):
        1. Shot 1: Raw OCR extraction (temp=0.05) - Faithful transcription
        2. Deduplication: Paragraph-level cleanup
        3. Shot 2: Structured metadata extraction (temp=0.0) - Extract title, authors, DOI
        
        This mirrors the proven OlmOCR_run_hybrid_v2.py approach that successfully
        extracted metadata from legacy/Collagen_review_olmocr-2-7b_p1_hybrid.json
        """
        page_number = profile.title_page_index
        self.logger.info(f"Processing detected title page (Page {page_number + 1}) with OlmOCR two-shot architecture...")
        
        try:
            start_time = time.time()
            
            # Prepare image and anchor text
            page = profile.pdf_doc.load_page(page_number)
            # CRITICAL: Pass PDF path to rendering function for pypdfium2
            image = self._render_page_pypdfium(profile.pdf_path, page_number, target_size=self.title_resolution)
            anchor_text = DocumentAnchoringExtractor.extract_document_anchoring(
                pdf_path=profile.pdf_path,
                page_num=page_number + 1
            )

            self.logger.info("Title Page Shot 1: Raw OCR extraction (temp=0.05)...")
            shot1_start = time.time()
            
            # Load two-shot parameters from VLM profile
            two_shot_params = self.body_vlm_profile.get('two_shot_parameters', {})
            shot1_params = two_shot_params.get('shot1_ocr', {})
            
            # Create temporary provider with Shot 1 parameters (use resolved task)
            shot1_provider = self._create_two_shot_provider(
                task_type=self.title_shot1_task,  # Resolved from registry
                temperature=shot1_params.get('temperature', 0.05),
                max_tokens=shot1_params.get('max_tokens', 8192),
                repeat_penalty=shot1_params.get('repeat_penalty', 1.15),
                frequency_penalty=shot1_params.get('frequency_penalty', 0.1)
            )
            
            # Execute Shot 1 WITH anchor text for complete page extraction
            # The anchor text provides structural guidance about page content
            # without biasing toward specific regions (unlike partial text)
            shot1_result = shot1_provider.process_page(image, base_text=anchor_text)
            raw_text = shot1_result.get('content', '')
            shot1_time = time.time() - shot1_start
            
            self.logger.info(f"Title Page Shot 1 complete: {len(raw_text)} characters in {shot1_time:.2f}s")
            
            if not raw_text:
                self.logger.error("Title Page Shot 1 returned empty text - CRITICAL FAILURE")
                return ContentPage(
                    page_number=page_number + 1,
                    content="",
                    quality_score=0.0,
                    metadata={'error': 'shot1_empty_title_page', 'processing_method': 'two_shot_title_failed'}
                )
            
            self.logger.info("Title Page: Performing paragraph-level deduplication...")
            dedup_start = time.time()
            
            cleaned_text = self._deduplicate_paragraphs(raw_text)
            dedup_time = time.time() - dedup_start
            
            self.logger.info(
                f"Title Page deduplication: {len(raw_text)} -> {len(cleaned_text)} chars "
                f"({len(cleaned_text)/len(raw_text)*100:.1f}% retained) in {dedup_time:.2f}s"
            )
            
            self.logger.info("Title Page Shot 2: Structured metadata extraction (temp=0.0)...")
            shot2_start = time.time()
            
            # Load Shot 2 parameters
            shot2_params = two_shot_params.get('shot2_analysis', {})
            
            # Create temporary provider with Shot 2 parameters
            # Task is title-specific, resolved from registry
            shot2_provider = self._create_two_shot_provider(
                task_type=self.title_shot2_task,  # Resolved from registry
                temperature=shot2_params.get('temperature', 0.0),
                max_tokens=shot2_params.get('max_tokens', 8192),
                repeat_penalty=shot2_params.get('repeat_penalty', 1.15),
                frequency_penalty=shot2_params.get('frequency_penalty', 0.1)
            )
            
            # Execute Shot 2 with cleaned text
            shot2_result = shot2_provider.process_page(
                image,
                base_text=cleaned_text  # Pass cleaned text as base_text
            )
            
            shot2_time = time.time() - shot2_start
            self.logger.info(f"Title Page Shot 2 complete in {shot2_time:.2f}s")
            
            # Extract results from Shot 2
            # CRITICAL: Title page task returns ONLY metadata, no content
            # We MUST use cleaned_text from Shot 1 for the page content
            final_metadata = shot2_result.get('metadata', {})
            
            # ALWAYS use cleaned_text for title page content (Shot 2 only extracts metadata)
            final_content = cleaned_text
            
            # Calculate quality score based on final content and metadata
            # Don't use shot2_result quality_score as it's based on metadata-only task
            quality_score = 0.75  # Default baseline for title pages with valid content
            if final_content and len(final_content) > 100:
                # Higher score for substantial content
                quality_score = 0.85
            
            # Add processing metadata
            final_metadata['processing_method'] = 'two_shot_title_page'
            final_metadata['shot1_time'] = round(shot1_time, 2)
            final_metadata['shot2_time'] = round(shot2_time, 2)
            final_metadata['dedup_time'] = round(dedup_time, 2)
            final_metadata['total_time'] = round(time.time() - start_time, 2)
            final_metadata['raw_text_length'] = len(raw_text)
            final_metadata['cleaned_text_length'] = len(cleaned_text)
            final_metadata['dedup_ratio'] = round(len(cleaned_text)/len(raw_text), 3) if raw_text else 0.0
            
            self.logger.info(
                f"Title page two-shot complete: {final_metadata['total_time']}s total, "
                f"quality={quality_score:.2f}"
            )
            
            return ContentPage(
                page_number=page_number + 1,
                content=final_content,
                quality_score=quality_score,
                metadata=final_metadata
            )

        except Exception as e:
            self.logger.error(f"Title page two-shot processing failed: {e}", exc_info=True)
            return ContentPage(
                page_number=page_number + 1, 
                content="", 
                quality_score=0.0, 
                metadata={'error': str(e), 'processing_method': 'two_shot_title_failed'}
            )

    def _process_body_pages(self, profile: 'DocumentProfile', page_indices: List[int]) -> List['ContentPage']:
        """
        Processes a list of body pages by calling the single-page processor for each.

        Args:
            profile: The DocumentProfile containing the PDF.
            page_indices: A list of 0-indexed page numbers to process.

        Returns:
            A list of processed ContentPage objects for the body pages.
        """
        self.logger.info(f"Processing {len(page_indices)} body pages...")
        processed_pages = []
        
        # Note: For simplicity, this is a sequential loop. It could be parallelized
        # in the future for performance improvements.
        for page_idx in page_indices:
            # We call the existing and functional _process_single_body_page method here
            page_result = self._process_single_body_page(profile, page_idx)
            processed_pages.append(page_result)
        
        return processed_pages

    def _process_single_body_page(self, profile: 'DocumentProfile', page_idx: int) -> 'ContentPage':
        """V2 ADAPTIVE ROUTER (YAML-DRIVEN - 5-Tier Implementation)"""
        try:
            self.logger.debug(f"Processing page {page_idx + 1} with V2 Adaptive Router...")
            page = profile.pdf_doc.load_page(page_idx)

            if not self.hybrid_enabled:
                self.logger.warning(f"Page {page_idx + 1}: Hybrid disabled, using Tier 5 fallback.")
                return self._process_with_vlm(profile, page, page_idx)
            
            # 1. Extract PyMuPDF text statistics for text density calculation
            pymupdf_result = self._extract_pymupdf_stats(page)
            
            # 2. Classify the page WITH text statistics
            page_type, analysis = self.page_classifier.classify(page, page_idx, pymupdf_result=pymupdf_result)
            page_characteristics = {
                "page_type": page_type.value,
                **analysis
            }
            page_characteristics.setdefault('has_large_table', False)
            page_characteristics.setdefault('has_figures', False)
            page_characteristics.setdefault('has_mixed_content', False)
            page_characteristics.setdefault('has_dense_figures', False)

            self.logger.info(f"Page {page_idx + 1} characteristics: {page_characteristics}")

            # 3. Loop through routing rules
            routing_rules = self.config['vlm_provider']['body_pages'].get('adaptive_routing_rules', [])
            
            for rule in routing_rules:
                conditions = rule.get('conditions', {})
                
                if conditions.get('default', False):
                    self.logger.info(f"Page {page_idx + 1}: Default rule matched.")
                    return self._process_with_vlm(profile, page, page_idx)

                if self._check_conditions(page_characteristics, conditions):
                    selected_tier = rule.get('tier')
                    self.logger.info(f"Page {page_idx + 1}: Rule '{rule.get('name')}' -> {selected_tier}")
                    
                    if selected_tier == 'tier1_pymupdf':
                        return self._process_with_pymupdf(page, page_idx, analysis)
                    elif selected_tier == 'tier2_surgical_vlm':
                        return self._process_with_surgical_vlm(profile, page, page_idx, page_type, analysis)
                    elif selected_tier == 'tier3_single_shot':
                        return self._process_with_vlm(profile, page, page_idx)
                    elif selected_tier == 'tier4_two_shot':
                        return self._process_with_two_shot(profile, page, page_idx)
                    elif selected_tier == 'tier5_fallback':
                        return self._process_with_vlm(profile, page, page_idx)
            
            self.logger.error(f"Page {page_idx + 1}: No rule matched. Using fallback.")
            return self._process_with_vlm(profile, page, page_idx)

        except Exception as e:
            self.logger.error(f"Router failed page {page_idx + 1}: {e}", exc_info=True)
            return ContentPage(page_number=page_idx + 1, content="", quality_score=0.0, 
                            metadata={'error': str(e), 'processing_method': 'router_failed'})
    
    def _check_conditions(self, characteristics: Dict, conditions: Dict) -> bool:
        """Evaluate if page characteristics match rule conditions (AND/OR logic)."""
        conditions_to_check = conditions.get('all_of', [conditions])
        
        if isinstance(conditions, dict) and 'any_of' not in conditions:
            conditions_to_check = [conditions]

        for condition_group in conditions_to_check:
            if 'any_of' in condition_group:
                any_matched = False
                for sub_condition in condition_group['any_of']:
                    if all(characteristics.get(k) == v for k, v in sub_condition.items()):
                        any_matched = True
                        break
                if not any_matched:
                    return False
                continue

            if not all(characteristics.get(k) == v for k, v in condition_group.items() if k not in ['any_of']):
                return False
        
        return True
    
    def _process_with_surgical_vlm(self, profile: 'DocumentProfile', page: fitz.Page, page_idx: int,
                                   page_type: 'PageType', analysis: Dict) -> 'ContentPage':
        """Process page using surgical VLM (Tier 2) - targeted visual processing."""
        self.logger.info(f"Page {page_idx + 1}: Tier 2 Surgical VLM processing...")
        
        try:
            # 1. Extract base text blocks
            text_blocks = self.reconstructor.extract_text_blocks(page)
            self.logger.debug(f"Extracted {len(text_blocks)} text blocks")
            
            # 2. Generate surgical content for special regions
            figure_regions = self._generate_figure_descriptions(page, analysis)
            equation_regions = self._generate_equation_latex(page, analysis)
            table_regions = self._generate_table_markdowns(page, analysis)
            
            self.logger.debug(
                f"Surgical: {len(figure_regions)} figures, {len(equation_regions)} equations, "
                f"{len(table_regions)} tables"
            )

            # 3. Reconstruct in reading order
            full_content = self.reconstructor.reconstruct(
                text_blocks=text_blocks,
                figure_regions=figure_regions,
                equation_regions=equation_regions,
                table_regions=table_regions
            )
            
            return ContentPage(
                page_number=page_idx + 1,
                content=full_content,
                quality_score=0.85,
                metadata={
                    'processing_method': f'surgical_v2_{page_type.value}',
                    'text_blocks': len(text_blocks),
                    'figure_regions': len(figure_regions),
                    'equation_regions': len(equation_regions),
                    'table_regions': len(table_regions)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Surgical VLM failed page {page_idx + 1}: {e}", exc_info=True)
            self.logger.warning(f"Falling back to full VLM for page {page_idx + 1}")
            return self._process_with_vlm(profile, page, page_idx)
    


    # ============================================================================
    # ============================================================================

    def _process_title_page_gemini(self, profile: 'DocumentProfile') -> 'ContentPage':
        """
        Process title page using Single-Pass Gemini task.
        Extracts structured metadata (TOML) + Content in one shot.
        """
        page_number = profile.title_page_index
        self.logger.info(f"Gemini: Processing title page {page_number + 1}...")
        
        try:
            start_time = time.time()
            
            # 1. Render Page
            image = self._render_page_pypdfium(profile.pdf_path, page_number, target_size=self.title_resolution)
            
            # 2. Call Gemini Provider
            # Note: We use the existing 'comprehensive_single_pass_extraction' task
            # We must load it through the provider to get the prompt/schema
            task_def = self.body_provider._load_task_definition("comprehensive_single_pass_extraction")
            
            # CRITICAL: Force the model to recognize this as the title page and output metadata
            # We inject this into the prompt
            task_def.prompt += "\n\nCRITICAL OVERRIDE: This specific page IS the TITLE PAGE of the document. You MUST extract the metadata in JSON format as specified in the Metadata section."
            
            result = self.body_provider.process_page(
                page_image=image,
                task_definition=task_def
            )
            
            if not result.get('success', False):
                raise RuntimeError(f"Provider failed: {result.get('error', 'Unknown error')}")
            
            # 3. Parse Metadata (JSON Block - proven schema)
            content = result.get('content', '')
            
            # Debug: Log start of content to see if metadata is present
            self.logger.debug(f"Gemini output for Title Page (First 500 chars):\n{content[:500]}")

            metadata = {}

            # Extract JSON metadata block (proven schema from structured_analysis_title_page.yaml)
            # Handle both markdown code fence wrapped and bare JSON blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            full_block_to_remove = None

            if json_match:
                # Markdown fence wrapped JSON
                json_str = json_match.group(1)
                full_block_to_remove = json_match.group(0)
            else:
                # Fallback to bare JSON block at start of content
                json_match = re.search(r'^\s*(\{.*?\})\s*', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    full_block_to_remove = json_match.group(0)

            if json_match:
                try:
                    import json
                    parsed = json.loads(json_str)

                    # Extract the nested "metadata" object (following proven schema)
                    if 'metadata' in parsed:
                        metadata = parsed['metadata']
                    else:
                        # Fallback: assume the entire object is metadata
                        metadata = parsed

                    # Validation: Check for required fields (from structured_analysis_title_page.yaml)
                    required_fields = ['title', 'authors', 'doi', 'journal_name', 'publication_year']
                    missing = [f for f in required_fields if f not in metadata]
                    if missing:
                        self.logger.warning(f"JSON metadata missing fields: {missing}")

                    # Validation: Check for important empty values
                    if metadata.get('doi') is None:
                        self.logger.info("DOI not found in title page (this may be expected)")

                    # Remove the JSON block (with fence if present) from the content for clean markdown
                    if full_block_to_remove:
                        content = content.replace(full_block_to_remove, '').strip()

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON metadata: {e}", exc_info=True)
                except Exception as e:
                    self.logger.warning(f"Error processing metadata: {e}", exc_info=True)
            
            processing_time = time.time() - start_time
            
            return ContentPage(
                page_number=page_number + 1,
                content=content,
                quality_score=0.95, # High confidence for Gemini
                metadata={
                    'processing_method': 'gemini_single_pass',
                    'total_time': round(processing_time, 2),
                    'generated_metadata': metadata
                }
            )

        except Exception as e:
            self.logger.error(f"Gemini title page failed: {e}", exc_info=True)
            return ContentPage(
                page_number=page_number + 1,
                content="",
                quality_score=0.0,
                metadata={'error': str(e)}
            )

    def _process_body_pages_gemini(self, profile: 'DocumentProfile', page_indices: List[int]) -> List['ContentPage']:
        """
        Process body pages using Simplified 'Lite Router'.
        - Text Density > 0.9 + No Images -> PyMuPDF (Tier 1)
        - Else -> Gemini (Tier 3)
        """
        processed_pages = []
        self.logger.info(f"Gemini: Processing {len(page_indices)} body pages with Lite Router...")
        
        for page_idx in page_indices:
            try:
                page = profile.pdf_doc.load_page(page_idx)
                
                # 1. Lite Classification
                page_type = PageType.SIMPLE_VISUAL
                use_pymupdf = False
                
                if self.page_classifier:
                    pymupdf_result = self._extract_pymupdf_stats(page)
                    # We pass None for vlm_provider to disable heavy verification
                    page_type, analysis = self.page_classifier.classify(page, page_idx, pymupdf_result=pymupdf_result)
                    
                    # LOGIC: High Text Density AND No Visuals AND No Equations -> Free Tier
                    text_density = analysis.get('text_density', 0.0)
                    visual_count = analysis.get('visual_count', 0)
                    equation_count = len(analysis.get('equation_regions', []))
                    
                    # Threshold: 0.9 density (strict) and 0 visuals/tables/equations
                    if text_density >= 0.90 and visual_count == 0 and equation_count == 0:
                        use_pymupdf = True
                        self.logger.info(f"Page {page_idx+1}: Lite Router -> PyMuPDF (Density={text_density:.2f})")
                    else:
                        self.logger.info(f"Page {page_idx+1}: Lite Router -> Gemini (Density={text_density:.2f}, Visuals={visual_count}, Eqs={equation_count})")
                
                # 2. Execution
                if use_pymupdf:
                    # Tier 1
                    page_result = self._process_with_pymupdf(page, page_idx, analysis if self.page_classifier else {})
                else:
                    # Tier 3 (Gemini Single Pass)
                    # Render Page
                    start_time = time.time()
                    image = self._render_page_pypdfium(profile.pdf_path, page_idx, target_size=self.body_resolution)
                    
                    task_def = self.body_provider._load_task_definition("comprehensive_single_pass_extraction")
                    result = self.body_provider.process_page(page_image=image, task_definition=task_def)
                    
                    page_result = ContentPage(
                        page_number=page_idx + 1,
                        content=result.get('content', ''),
                        quality_score=0.90,
                        metadata={
                            'processing_method': 'gemini_single_pass',
                            'total_time': round(time.time() - start_time, 2)
                        }
                    )
                
                processed_pages.append(page_result)

            except Exception as e:
                self.logger.error(f"Gemini body page {page_idx+1} failed: {e}", exc_info=True)
                # Fallback to empty page
                processed_pages.append(ContentPage(
                    page_number=page_idx + 1,
                    content="",
                    quality_score=0.0,
                    metadata={'error': str(e)}
                ))
        
        return processed_pages

    # ============================================================================
    # ============================================================================

    def _build_processing_map(self, profile: 'DocumentProfile', page_indices: List[int]) -> Dict[str, List[int]]:
        """
        Tier 0: Pre-classify all pages to determine optimal processing tier.

        PRIORITY 3 FIX: Now caches classification results to avoid re-classification
        in GPU thread (saves 5-10 seconds per document).

        Args:
            profile: DocumentProfile containing the PDF
            page_indices: List of 0-indexed page numbers to classify

        Returns:
            Processing map: {'tier1': [indices], 'tier2': [indices], ...}
        """
        processing_map = {
            'tier1': [],
            'tier2': [],
            'tier3': [],
            'tier4': [],
            'tier5': []
        }

        # PRIORITY 3 FIX: Initialize classification cache to store analysis results
        # This avoids re-classification in GPU thread (line 906-907)
        self._classification_cache = {}

        for page_idx in page_indices:
            # We must load the page to classify it
            # This adds overhead, but saves time by routing correctly
            page = profile.pdf_doc.load_page(page_idx)

            # Extract PyMuPDF stats for text density calculation
            pymupdf_result = self._extract_pymupdf_stats(page)

            # Classify the page
            page_type, analysis = self.page_classifier.classify(
                page, page_idx, pymupdf_result=pymupdf_result
            )

            # PRIORITY 3 FIX: Cache classification results for later use
            # Stores (page_type, analysis) tuple to avoid re-classification
            self._classification_cache[page_idx] = (page_type, analysis)

            # Build page characteristics dict
            page_characteristics = {
                "page_type": page_type.value,
                **analysis
            }
            # Set defaults for missing keys
            page_characteristics.setdefault('has_large_table', False)
            page_characteristics.setdefault('has_figures', False)
            page_characteristics.setdefault('has_mixed_content', False)
            page_characteristics.setdefault('has_dense_figures', False)

            # Evaluate routing rules to determine tier
            routing_rules = self.config['vlm_provider']['body_pages'].get('adaptive_routing_rules', [])

            selected_tier = 'tier5'  # Default fallback
            tier_key = 'tier5' # Key for the map

            for rule in routing_rules:
                conditions = rule.get('conditions', {})

                if conditions.get('default', False):
                    selected_tier = rule.get('tier', 'tier5_fallback')
                    break

                if self._check_conditions(page_characteristics, conditions):
                    selected_tier = rule.get('tier', 'tier5_fallback')
                    break

            # Map selected tier string to our map keys
            if selected_tier == 'tier1_pymupdf':
                tier_key = 'tier1'
            elif selected_tier == 'tier2_surgical_vlm':
                tier_key = 'tier2'
            elif selected_tier == 'tier3_single_shot':
                tier_key = 'tier3'
            elif selected_tier == 'tier4_two_shot':
                tier_key = 'tier4'
            else:
                tier_key = 'tier5'

            # Add page index to the appropriate tier list
            processing_map[tier_key].append(page_idx)

        self.logger.info(f"Tier 0: Cached {len(self._classification_cache)} page classifications")

        return processing_map

    def _process_body_pages_concurrent(self, profile: 'DocumentProfile', processing_map: Dict[str, List[int]]) -> List['ContentPage']:
        """
        Orchestrates concurrent processing for all tiers.
        Uses asyncio to run Tier 1 (Executor) and Tiers 2-5 (Async VLM) in parallel.
        Fix 4: Pure AsyncIO Architecture Refactor (Orchestrator Entry Point).
        """
        async def _orchestrator():
            tasks = []
            loop = asyncio.get_event_loop()
            
            # 1. Tier 1: PyMuPDF (CPU Bound -> Executor)
            tier1_indices = processing_map.get('tier1', [])
            if tier1_indices:
                self.logger.info(f"Tier 1: Processing {len(tier1_indices)} pages with PyMuPDF (Executor)...")
                def _process_tier1_sync():
                    results = []
                    for idx in tier1_indices:
                        try:
                            # Re-load page in this thread
                            page = profile.pdf_doc.load_page(idx)
                            # Get cached analysis
                            analysis = {}
                            if hasattr(self, '_classification_cache') and idx in self._classification_cache:
                                _, analysis = self._classification_cache[idx]
                            
                            res = self._process_with_pymupdf(page, idx, analysis)
                            results.append(res)
                        except Exception as e:
                            self.logger.error(f"Tier 1 page {idx+1} failed: {e}")
                            results.append(ContentPage(idx+1, "", 0.0, {'error': str(e)}))
                    return results
                
                # Default executor (Thread pool)
                tasks.append(loop.run_in_executor(None, _process_tier1_sync))
            
            # 2. Tiers 2-5: Async VLM
            vlm_tiers = ['tier2', 'tier3', 'tier4', 'tier5']
            if any(processing_map.get(t) for t in vlm_tiers):
                tasks.append(self._process_vlm_layers_async(profile, processing_map))
            
            # Execute
            if not tasks:
                return []
                
            results_list = await asyncio.gather(*tasks)
            
            # Flatten
            final_pages = []
            for r in results_list:
                final_pages.extend(r)
            
            return final_pages

        # Check if loop is already running (e.g., in GUI)
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # We are in an async context (GUI).
                # Since we cannot return an awaitable here (signature is sync List[ContentPage]),
                # we are stuck if we don't change the signature.
                # However, for now we assume this is called by the CLI runner (Sync).
                # If called by GUI, this will raise RuntimeError.
                # Ideally, we should refactor 'process' to be async.
                # For this fix, we implement the CLI-safe version.
                raise RuntimeError("Cannot call sync _process_body_pages_concurrent from running loop. Use async version.")
        except RuntimeError:
            pass # No loop running

        return asyncio.run(_orchestrator())

    async def _process_vlm_layers_async(
        self,
        profile: 'DocumentProfile',
        processing_map: Dict[str, List[int]]
    ) -> List['ContentPage']:
        """
        AsyncIO orchestrator for VLM tiers (2, 3, 4, 5).
        Prepares jobs and executes them via AsyncUniversalVLMProvider.
        """
        # 1. Collect all VLM jobs
        vlm_jobs = [] # List of (image, base_text, task_def, page_idx, tier, analysis)
        
        # Iterating tiers in priority is handled by the result merging later
        # We just need to gather all work
        for tier in ['tier2', 'tier3', 'tier4', 'tier5']:
            page_indices = processing_map.get(tier, [])
            for page_idx in page_indices:
                # Prepare data for this page
                 # Get cached classification
                analysis = {}
                page_type = None
                if hasattr(self, '_classification_cache') and page_idx in self._classification_cache:
                    page_type, analysis = self._classification_cache[page_idx]
                
                # Render Image
                image = self._render_page_pypdfium(profile.pdf_path, page_idx, target_size=self.body_resolution)
                anchor_text = DocumentAnchoringExtractor.extract_document_anchoring(
                    pdf_path=profile.pdf_path,
                    page_num=page_idx + 1
                )
                
                # Determine Task and Params based on Tier
                task_def = None # Use default
                
                # Specific Tier Prep
                if tier == 'tier4':
                     # Two-shot is complex (chained calls). 
                     # AsyncUniversalVLMProvider supports single-shot batching best.
                     # For now, we will run Two-Shot in SEQUENTIAL fallback within the batch loop?
                     # OR we need full async two-shot support.
                     # Given complexity, let's process Tier 4 SEQUENTIALLY for now inside the async loop wrapper?
                     # OR, simpler: We execute Tier 2, 3, 5 concurrently. Tier 4 (Two-shot) is hard to batch 
                     # because it depends on shot 1 result.
                     pass 
                
                # For simplicity in Phase 4.9, we route Tier 3 and 5 (Single Shot) to Async Batch.
                # Tier 2 (Surgical) and 4 (Two-Shot) are complex multi-step.
                # We will handle them:
                # - Tier 3/5: Add to batch
                # - Tier 2/4: Execute individually (can still be async awaited)
                
                vlm_jobs.append({
                    'page_idx': page_idx,
                    'tier': tier,
                    'image': image,
                    'anchor': anchor_text,
                    'page_type': page_type,
                    'analysis': analysis
                })

        if not vlm_jobs:
            return []

        # 2. Initialize Async Provider
        # Reuse validated Pydantic models from existing sync provider (DRY principle)
        async_provider = AsyncUniversalVLMProvider(
            model_name=self.body_provider.model_name,
            task_type="basic_ocr", # Default, overridden per job
            model_profile=self.body_provider.model_profile,
            task_definition=self.body_provider.task_definition,
            skip_validation=True
        )
        await async_provider.initialize_async(self.concurrency_config)
        
        results_map = {} # page_idx -> ContentPage

        try:
            # 3. Prepare Tasks
            tasks = []
            
            # A. Batch Process Single-Shot Jobs (Tier 3/5)
            batch_pages = []
            batch_metadata = [] # stores (page_idx, tier) for mapping back
            
            for job in vlm_jobs:
                if job['tier'] in ['tier3', 'tier5']:
                     # Standard VLM
                     batch_pages.append((job['image'], job['anchor'], None))
                     batch_metadata.append(job)
            
            async def run_batch():
                if not batch_pages:
                    return []
                self.logger.info(f"AsyncIO: Batch processing {len(batch_pages)} pages (Tier 3/5)...")
                batch_results = await async_provider.process_pages_batch_async(batch_pages)
                
                local_results = []
                for i, res in enumerate(batch_results):
                    job = batch_metadata[i]
                    page_idx = job['page_idx']
                    
                    if res.get('success'):
                         cp = ContentPage(
                            page_number=page_idx + 1,
                            content=res.get('content', ''),
                            quality_score=res.get('quality_score', 0.0),
                            metadata=res.get('metadata', {})
                        )
                    else:
                         cp = ContentPage(
                            page_number=page_idx + 1,
                            content="",
                            quality_score=0.0,
                            metadata={'error': res.get('error', 'Async batch failed')}
                        )
                    local_results.append((page_idx, cp))
                return local_results

            tasks.append(run_batch())

            # B. Complex Jobs (Tier 2/4) - Now Async!
            for job in vlm_jobs:
                if job['tier'] == 'tier4':
                    # Fix 1: Async Two-Shot
                    tasks.append(self._process_with_two_shot_async(
                        async_provider, 
                        job['image'], 
                        job['page_idx'],
                        job['anchor']
                    ))
                elif job['tier'] == 'tier2':
                    # Fix 2: Async Surgical VLM - Executed Concurrently
                    p_idx = job['page_idx']
                    # Loading page is fast CPU op, acceptable to do in main thread or could be threaded
                    # For simplicity and thread-safety of fitz, we load it here
                    page_obj = profile.pdf_doc.load_page(p_idx)
                    
                    tasks.append(self._process_with_surgical_vlm_async(
                        async_provider,
                        profile,
                        page_obj,
                        p_idx,
                        job['page_type'],
                        job['analysis']
                    ))

            # 4. Execute All Concurrently
            self.logger.info(f"AsyncIO: Executing {len(tasks)} concurrent tasks (Batch + Complex)...")
            all_task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 5. Aggregate Results
            for res_group in all_task_results:
                if isinstance(res_group, Exception):
                    self.logger.error(f"Async task failed: {res_group}")
                    continue
                    
                # run_batch returns list of (page_idx, cp)
                if isinstance(res_group, list): 
                    for p_idx, cp in res_group:
                        results_map[p_idx] = cp
                # run_tier2_sync returns (page_idx, cp)
                elif isinstance(res_group, tuple) and len(res_group) == 2:
                    results_map[res_group[0]] = res_group[1]
                # two_shot_async returns ContentPage directly (need to extract page_idx)
                elif isinstance(res_group, ContentPage):
                    results_map[res_group.page_number - 1] = res_group
                    
        finally:
            await async_provider.close()

        return list(results_map.values())

    def _process_body_pages_concurrent(
        self,
        profile: 'DocumentProfile',
        processing_map: Dict[str, List[int]]
    ) -> List['ContentPage']:
        """
        Process body pages using AsyncIO for VLM and Thread for PyMuPDF.
        """
        from concurrent.futures import ThreadPoolExecutor
        import time

        all_results = []
        concurrent_start = time.time()

        # 1. Start CPU Thread for Tier 1
        with ThreadPoolExecutor(max_workers=1) as executor:
            cpu_future = executor.submit(
                self._process_tier1_batch,
                profile,
                processing_map['tier1']
            )
            
            # 2. Run AsyncIO for VLM Tiers
            try:
                # We use asyncio.run to execute the async orchestrator
                vlm_results = asyncio.run(self._process_vlm_layers_async(profile, processing_map))
            except Exception as e:
                self.logger.error(f"AsyncIO VLM processing failed: {e}", exc_info=True)
                raise

            # 3. Get CPU results
            tier1_results = cpu_future.result()
            
            all_results.extend(tier1_results)
            all_results.extend(vlm_results)

        concurrent_end = time.time()
        
        # Logging Metrics
        wall_clock_time = concurrent_end - concurrent_start
        self.logger.info("=" * 70)
        self.logger.info(f"ASYNCIO CONCURRENCY METRICS")
        self.logger.info(f"Wall Clock Time:    {wall_clock_time:.2f}s")
        self.logger.info(f"VLM Pages:          {len(vlm_results)}")
        self.logger.info(f"Tier 1 CPU Pages:   {len(tier1_results)}")
        self.logger.info("=" * 70)

        all_results.sort(key=lambda p: p.page_number)
        
        if hasattr(self, '_classification_cache'):
            del self._classification_cache

        return all_results

    def _process_tier1_batch(
        self,
        profile: 'DocumentProfile',
        page_indices: List[int]
    ) -> List['ContentPage']:
        """
        Batch process all Tier 1 pages using PyMuPDF.

        This runs on the CPU thread and can execute while GPU processes VLM tiers.

        THREAD SAFETY (Phase 4 Fix):
        Opens a separate document handle for this thread to avoid race conditions
        with the GPU thread. PyMuPDF documents are NOT thread-safe.
        """
        if not page_indices:
            return []

        self.logger.info(f"CPU Thread: Processing {len(page_indices)} Tier 1 pages...")

        thread_doc = fitz.open(profile.pdf_path)
        self.logger.debug(f"CPU Thread: Opened separate document handle (ptr: {id(thread_doc)})")

        results = []

        try:
            for page_idx in page_indices:
                page = thread_doc.load_page(page_idx)
                # Re-extract stats/analysis?
                # We pass empty analysis dict as we don't need it for pure text extraction output
                # (or we could re-run extract_stats if metadata needs it)
                result = self._process_with_pymupdf(page, page_idx, analysis={})
                results.append(result)

            self.logger.info(f"CPU Thread: Tier 1 batch complete ({len(results)} pages)")

        finally:
            # Clean up: Close the thread-specific document
            thread_doc.close()
            self.logger.debug(f"CPU Thread: Closed separate document handle")

        return results

    def _process_vlm_sequential(
        self,
        profile: 'DocumentProfile',
        processing_map: Dict[str, List[int]]
    ) -> List['ContentPage']:
        """
        Sequentially process all VLM-dependent tiers (2, 3, 4, 5).

        This runs on the GPU thread and maintains priority order.

        THREAD SAFETY (Phase 4 Fix):
        Opens a separate document handle for this thread to avoid race conditions
        with the CPU thread. PyMuPDF documents are NOT thread-safe.
        """
        self.logger.info("GPU Thread: Starting sequential VLM processing...")

        thread_doc = fitz.open(profile.pdf_path)
        self.logger.debug(f"GPU Thread: Opened separate document handle (ptr: {id(thread_doc)})")

        results = []

        try:
            # Process tiers in priority order: 4 (Slowest/Deepest) -> 2 (Visual) -> 3 (Single) -> 5 (Fallback)
            # Actually any order is fine for result correctness, but maybe we want 4 done first?
            # Let's stick to simple priority list.
            tier_priority = ['tier4', 'tier2', 'tier3', 'tier5']

            for tier in tier_priority:
                page_indices = processing_map.get(tier, [])
                if not page_indices:
                    continue

                self.logger.info(f"GPU Thread: Processing {len(page_indices)} {tier} pages...")

                for page_idx in page_indices:
                    page = thread_doc.load_page(page_idx)

                    # PRIORITY 3 FIX: Use cached classification instead of re-classifying
                    # Previously: Re-ran classification here (5-10s wasted per document)
                    # Now: Retrieve from cache populated in Tier 0

                    if tier == 'tier2':
                        # Get cached classification from Tier 0 (if available)
                        if hasattr(self, '_classification_cache') and page_idx in self._classification_cache:
                            page_type, analysis = self._classification_cache[page_idx]
                            self.logger.debug(f"GPU Thread: Using cached classification for page {page_idx}")
                        else:
                            # Fallback: Re-classify if cache miss (shouldn't happen, but defensive)
                            self.logger.warning(f"GPU Thread: Cache miss for page {page_idx}, re-classifying")
                            pymupdf_result = self._extract_pymupdf_stats(page)
                            page_type, analysis = self.page_classifier.classify(page, page_idx, pymupdf_result=pymupdf_result)

                        result = self._process_with_surgical_vlm(profile, page, page_idx, page_type, analysis)
                    elif tier == 'tier3':
                        result = self._process_with_vlm(profile, page, page_idx)
                    elif tier == 'tier4':
                        result = self._process_with_two_shot(profile, page, page_idx)
                    else:  # tier5
                        result = self._process_with_vlm(profile, page, page_idx)

                    results.append(result)

            self.logger.info(f"GPU Thread: Sequential VLM processing complete ({len(results)} pages)")

        finally:
            # Clean up: Close the thread-specific document
            thread_doc.close()
            self.logger.debug(f"GPU Thread: Closed separate document handle")

        return results

    # ============================================================================
    # GLM TWO-PHASE OFFLINE PATH (Tier overhaul — verify-then-repair)
    # ============================================================================
    # Phase A: the extraction model reads every body page; a PyMuPDF word-coverage gate catches
    #          any silently-dropped table; the layout model locates it and the extraction model
    #          re-reads the crop. Extraction + layout models are co-resident, so this costs NO
    #          model swap. Phase B: one swap to the understanding model to describe figures.
    #
    # VLM-AGNOSTIC: every model name is resolved from profile_registry and every model-specific
    # detail (grounding format, prompts, render DPI, coverage threshold) from the models' own
    # config. No model identity is hardcoded in this code.

    def _init_glm_twophase(self):
        """
        Configure the offline GLM two-phase body path, or leave it disabled.

        Enabled only when BOTH hold: the local llama-server is in use, and profile_registry
        defines an `extraction` role. That makes this path strictly opt-in via config — every
        existing path (cloud, offline OlmOCR tiers) is untouched when the role is absent.
        """
        self.glm_twophase_enabled = False
        self.extraction_provider = None
        self.layout_provider = None
        self.grounding_spec = None
        self.grounding_prompt = None
        self.router = None

        local_cfg = self.config['vlm_provider'].get('local_server', {})
        registry = self.config.get('profile_registry', {})
        if not local_cfg.get('enabled', False) or 'extraction' not in registry:
            self.logger.info(
                "GLM two-phase offline path DISABLED "
                "(needs local_server.enabled=true + profile_registry.extraction)"
            )
            return

        # Resolve each role -> model name from the registry (config-driven, never hardcoded).
        self.extraction_model = self._resolve_reference('@profile_registry.extraction', self.config)
        self.understanding_model = self._resolve_reference('@profile_registry.understanding', self.config)
        self.layout_model = self._resolve_reference('@profile_registry.layout_detector', self.config)

        # Extraction reader: the raw-OCR task maps (via the model's fixed_prompts) to its
        # "read the whole page" prompt.
        raw_ocr_task = self._resolve_reference('@task_registry.raw_ocr', self.config)
        self.extraction_provider, self.extraction_profile_data, _ = self._initialize_provider_from_config(
            {'profile': self.extraction_model, 'task_type': raw_ocr_task},
            api_endpoint=self.api_endpoint,
        )
        fixed_prompts = self.extraction_profile_data.get('fixed_prompts', {})
        self.extraction_dpi = self.extraction_profile_data.get('app_config', {}).get('target_dpi', 300)
        self.coverage_threshold = self.extraction_profile_data.get(
            'content_drop_check', {}).get('min_word_coverage', 0.90)

        # Prompt for re-reading a recovered table crop (table task -> the model's table prompt;
        # fall back to its page-read prompt if the model has no dedicated table prompt).
        table_task = self._resolve_reference('@task_registry.table_desc', self.config)
        self.table_crop_prompt = fixed_prompts.get(table_task) or fixed_prompts.get(raw_ocr_task)

        # Plain page-read prompt, used to read a figure caption off a SCANNED page (no text layer).
        # Born-digital captions come free from the text layer and never touch this.
        self.caption_read_prompt = fixed_prompts.get(raw_ocr_task)

        # Layout detector provider + its grounding format (both from the layout model's config).
        self.layout_provider, layout_profile_data, _ = self._initialize_provider_from_config(
            {'profile': self.layout_model, 'task_type': raw_ocr_task},
            api_endpoint=self.api_endpoint,
        )
        grounding_format = layout_profile_data.get('grounding_format')
        if not grounding_format:
            self.logger.error(
                f"Layout model '{self.layout_model}' declares no grounding_format; "
                "GLM two-phase path stays DISABLED"
            )
            return
        self.grounding_spec = GroundingSpec.from_config(grounding_format)
        self.grounding_prompt = grounding_format['prompt']

        # Router remote control: keep exactly the right models resident per phase.
        base_url = f"http://{local_cfg.get('host', 'localhost')}:{local_cfg.get('port', 8080)}"
        self.router = RouterClient(base_url, logger=self.logger)

        # Where cropped figures + the manifest are written for the multimodal RAG. Config-
        # overridable (vlm_provider.figures.output_dir); defaults under run_outputs. Stored image
        # paths are made relative to figures_path_root so the manifest carries portable paths.
        figures_cfg = self.config['vlm_provider'].get('figures', {})
        self.figures_base_dir = Path(figures_cfg.get('output_dir', 'run_outputs/figures'))
        self.figures_path_root = Path(figures_cfg.get('path_root', 'run_outputs'))

        self.glm_twophase_enabled = True
        self.logger.info(
            f"GLM two-phase offline path ENABLED: extraction={self.extraction_model}, "
            f"layout={self.layout_model}, understanding={self.understanding_model}, "
            f"coverage_threshold={self.coverage_threshold}, extraction_dpi={self.extraction_dpi}"
        )

    def _process_body_pages_glm(self, profile: 'DocumentProfile', page_indices: List[int]) -> List['ContentPage']:
        """
        Run the offline GLM body-page pipeline over the given pages — single phase, no swap.

        The extraction + layout models stay co-resident for the whole body. Each page is read by
        the extraction model, verified against the text layer, and any dropped table is repaired.
        Figures are cropped and caption-tagged from the SAME layout-grounding response (no
        understanding model, no swap back — OlmOCR is used only for the title page).

        Args:
            profile: the DocumentProfile (source PDF).
            page_indices: 0-indexed body pages to process.

        Returns:
            ContentPage list in page order.
        """
        if self.router:
            self.router.set_resident([self.extraction_model, self.layout_model])

        # Sequential per-page processing. Concurrency was investigated (asyncio + threads, and a
        # read/ground two-stage split) but gave no wall-time win on a single GPU — the extraction
        # and layout models share one device, so overlapping their calls only trades gains for
        # contention (measured: identical time to sequential, and worse when table-repair crops
        # collided with groundings). Kept simple; see the optimization report for the full analysis.
        results: Dict[int, 'ContentPage'] = {}
        for page_idx in page_indices:
            try:
                page = profile.pdf_doc.load_page(page_idx)
                results[page_idx] = self._read_body_page_glm(profile, page, page_idx)
            except Exception as error:
                self.logger.error(f"Body page {page_idx + 1} failed: {error}", exc_info=True)
                results[page_idx] = ContentPage(
                    page_number=page_idx + 1, content="", quality_score=0.0,
                    metadata={'error': str(error), 'processing_method': 'glm_failed'},
                )

        # Document-level figure manifest for the multimodal RAG (aggregated across all pages).
        self._write_figure_manifest(profile, results)

        return [results[idx] for idx in sorted(results)]


    def _read_body_page_glm(
        self, profile: 'DocumentProfile', page: fitz.Page, page_idx: int
    ) -> 'ContentPage':
        """
        Read one body page: classify -> pure-text fast path OR extraction read + verify/repair +
        figure extraction.

        Pure-text pages skip the VLM entirely (PyMuPDF native text). Every other page is read
        full-page by the extraction model, checked against the text layer, and (if a table was
        dropped) repaired; figures are cropped and caption-tagged from one shared grounding pass.
        """
        analysis: Dict = {}
        page_type = None
        if self.page_classifier is not None:
            stats = self._extract_pymupdf_stats(page)
            page_type, analysis = self.page_classifier.classify(page, page_idx, pymupdf_result=stats)

        # T1: pure-text page -> PyMuPDF native text, no VLM needed.
        if page_type is not None and page_type == PageType.TEXT_ONLY:
            self.logger.info(f"Page {page_idx + 1}: pure text -> PyMuPDF (T1)")
            return self._process_with_pymupdf(page, page_idx, analysis)

        # T2: extraction model reads the whole page.
        self.logger.info(f"Page {page_idx + 1}: extraction model full-page read")
        page_image = self._render_extraction_page(profile.pdf_path, page_idx, page)
        read = self.extraction_provider.process_page(page_image)
        glm_text = read.get('content', '') if read.get('success') else ''
        base_quality = read.get('quality_score', 0.0) or 0.0

        # ONE grounding pass per page, shared by table repair AND figure extraction. A page needs
        # it when the read dropped content (born-digital gate), when it is scanned (no text layer
        # to verify against), or when the classifier flagged visuals to crop.
        gate = check_page_for_content_drop(
            page, glm_text, min_coverage=self.coverage_threshold, logger=self.logger
        )
        has_visuals = bool(analysis.get('visual_regions'))
        grounding = self._ground_page_if_needed(page, page_image, gate, has_visuals)

        final_text, repair_meta = self._repair_page(page, glm_text, page_image, gate, grounding)

        # Figures: crop + caption-tag from the same grounding response (no description, no swap).
        figures = self._extract_page_figures(profile, page, page_image, grounding, page_idx)
        if figures:
            final_text = append_figure_anchors(final_text, figures)

        quality = max(base_quality, 0.85) if repair_meta.get('tables_recovered') else base_quality
        metadata = {
            'processing_method': repair_meta['processing_method'],
            'page_type': page_type.value if page_type is not None else None,
            'coverage': repair_meta.get('coverage'),
            'tables_recovered': repair_meta.get('tables_recovered', 0),
            'visual_element_count': len(analysis.get('visual_regions', [])),
            'figures': [f.to_dict() for f in figures],
            'figure_count': len(figures),
        }
        return ContentPage(
            page_number=page_idx + 1, content=final_text, quality_score=quality, metadata=metadata
        )

    def _ground_page_if_needed(
        self, page: fitz.Page, page_image: 'Image.Image', gate, has_visuals: bool
    ) -> Optional[str]:
        """
        Run the layout-grounding call at most ONCE per page, only when something needs it.

        Returns the raw grounding text, or None when the page needs neither table repair nor
        figure extraction (so we never pay for a grounding call we will not use).
        """
        born_digital_drop = gate.checked and gate.content_dropped
        scanned = (not gate.checked) and is_scanned_page(page)
        want_figures = (
            has_visuals and self.grounding_spec is not None and self.grounding_spec.supports_figures
        )
        if not (born_digital_drop or scanned or want_figures):
            return None
        return self._run_layout_grounding(page_image)

    def _repair_page(
        self, page: fitz.Page, glm_text: str, page_image: 'Image.Image', gate, grounding: Optional[str]
    ) -> Tuple[str, Dict]:
        """
        Repair a dropped table using the (already computed) gate + grounding — no new model call.

        Born-digital pages use the PyMuPDF word-coverage gate. Scanned pages have no trustworthy
        text layer, so the drop signal is "the layout model sees a table the read did not produce".

        Returns:
            (possibly-repaired text, metadata dict with processing_method / coverage / tables_recovered)
        """
        coverage = round(gate.coverage, 3) if gate.checked else None
        born_digital_drop = gate.checked and gate.content_dropped
        scanned_has_table = (
            (not gate.checked) and is_scanned_page(page)
            and grounding is not None and self._grounding_has_table(grounding)
        )
        if not (born_digital_drop or scanned_has_table) or grounding is None:
            return glm_text, {'processing_method': 'glm_fullpage', 'coverage': coverage, 'tables_recovered': 0}

        result = repair_page_tables(
            page_image=page_image,
            full_page_text=glm_text,
            grounding_output=grounding,
            spec=self.grounding_spec,
            read_table_crop=self._read_table_crop,
            logger=self.logger,
        )
        return result.text, {
            'processing_method': 'glm_repair' if result.repaired else 'glm_fullpage',
            'coverage': coverage,
            'tables_recovered': result.tables_recovered,
        }

    def _render_extraction_page(self, pdf_path: str, page_idx: int, page: fitz.Page) -> 'Image.Image':
        """Render a page at the extraction model's target DPI (longest edge, in pixels)."""
        longest_pts = max(page.rect.width, page.rect.height)
        target_size = int(longest_pts / 72.0 * self.extraction_dpi)
        return self._render_page_pypdfium(pdf_path, page_idx, target_size=target_size)

    def _run_layout_grounding(self, page_image: 'Image.Image') -> str:
        """Ask the layout detector for its grounding boxes (image-first; prompt from its config)."""
        return self.layout_provider.read_crop(page_image, self.grounding_prompt, image_first=True)

    def _grounding_has_table(self, grounding_output: str) -> bool:
        """True when the layout model's grounding output contains at least one table box."""
        if not grounding_output:
            return False
        detections = parse_grounding_output(grounding_output, self.grounding_spec)
        return any(d.label == self.grounding_spec.table_label for d in detections)

    def _read_table_crop(self, crop_image: 'Image.Image') -> str:
        """Re-read one cropped table with the extraction model and its configured table prompt."""
        return self.extraction_provider.read_crop(crop_image, self.table_crop_prompt)


    def _extract_page_figures(
        self, profile: 'DocumentProfile', page: fitz.Page, page_image: 'Image.Image',
        grounding: Optional[str], page_idx: int
    ) -> List['FigureAsset']:
        """
        Crop and save every figure on this page, tagging each with its caption text.

        Uses the already-computed layout grounding (no new model call). Returns [] when the page
        has no grounding, the detector lacks figure support, or the page carries no figure.
        """
        if grounding is None or self.grounding_spec is None or not self.grounding_spec.supports_figures:
            return []
        doc_id = safe_doc_id(Path(profile.pdf_path).stem)
        out_dir = self.figures_base_dir / doc_id
        read_caption = self._make_caption_reader(page, page_image)
        return extract_and_save_figures(
            page_image=page_image,
            grounding_output=grounding,
            spec=self.grounding_spec,
            out_dir=out_dir,
            page_number=page_idx + 1,
            read_caption=read_caption,
            path_root=self.figures_path_root,
            logger=self.logger,
        )

    def _make_caption_reader(self, page: fitz.Page, page_image: 'Image.Image'):
        """
        Build the caption reader for one page: box (in page-image pixels) -> caption text.

        Born-digital pages read the caption verbatim from the PyMuPDF text layer (authoritative,
        free, no fabrication). A scanned page has no usable text layer, so it falls back to the
        extraction model reading the caption crop.
        """
        def _read(box_px: Optional[Tuple[int, int, int, int]]) -> str:
            if not box_px:
                return ""
            rect = self._image_box_to_pdf_rect(box_px, page, page_image)
            text = page.get_text("text", clip=rect).strip()
            if text:
                return text
            if is_scanned_page(page) and self.caption_read_prompt:
                crop = page_image.crop(box_px)
                return (self.extraction_provider.read_crop(crop, self.caption_read_prompt) or "").strip()
            return ""
        return _read

    def _image_box_to_pdf_rect(
        self, box_px: Tuple[int, int, int, int], page: fitz.Page, page_image: 'Image.Image'
    ) -> 'fitz.Rect':
        """
        Map a box in rendered-page-image pixels back to PDF points, to clip the text layer.

        The page image is rendered by `_render_page_pypdfium` at a uniform scale with a fixed
        white border (RENDER_PADDING_PX px per side). We derive the scale from the actual content
        width so DPI rounding cannot drift, subtract the padding, and add a small point-slop so a
        caption line intersecting the box is captured.
        """
        content_width = max(1, page_image.width - 2 * self.RENDER_PADDING_PX)
        scale = content_width / page.rect.width  # pixels per PDF point (uniform on both axes)
        x0, y0, x1, y1 = box_px
        pad = self.RENDER_PADDING_PX
        slop = 4.0  # points
        return fitz.Rect(
            (x0 - pad) / scale - slop,
            (y0 - pad) / scale - slop,
            (x1 - pad) / scale + slop,
            (y1 - pad) / scale + slop,
        )

    def _write_figure_manifest(self, profile: 'DocumentProfile', results: Dict[int, 'ContentPage']) -> None:
        """
        Write a document-level figure manifest (JSON) aggregating every page's figures.

        The manifest is what the multimodal RAG consumes: one flat list of {figure_id, image_path,
        caption, page, bbox}. Nothing is written when the document has no figures.
        """
        figures = []
        for idx in sorted(results):
            figures.extend(results[idx].metadata.get('figures', []) or [])
        if not figures:
            return
        doc_id = safe_doc_id(Path(profile.pdf_path).stem)
        manifest_dir = self.figures_base_dir / doc_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / 'manifest.json'
        try:
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    {'document': doc_id, 'figure_count': len(figures), 'figures': figures},
                    handle, indent=2, ensure_ascii=False,
                )
            self.logger.info(f"Figure manifest: {len(figures)} figure(s) -> {manifest_path}")
        except Exception as error:
            self.logger.warning(f"Could not write figure manifest: {error}")

    # ============================================================================
    # 3-TIER PROCESSING METHODS (Existing)
    # ============================================================================

    def _process_with_vlm(self, profile: 'DocumentProfile', page: fitz.Page, page_idx: int) -> 'ContentPage':
        """Process page with full VLM (original method - Tier 3 fallback)."""
        image = self._render_page_pypdfium(profile.pdf_path, page_idx, target_size=self.body_resolution)
        anchor_text = DocumentAnchoringExtractor.extract_document_anchoring(
            pdf_path=profile.pdf_path,
            page_num=page_idx + 1
        )
        result = self.body_provider.process_page(image, base_text=anchor_text)
        return ContentPage(
            page_number=page_idx + 1,
            content=result.get('content', ''),
            quality_score=result.get('quality_score'),
            metadata=result.get('metadata', {})
        )
    
    def _process_with_pymupdf(self, page: fitz.Page, page_idx: int, analysis: Dict) -> 'ContentPage':
        """Process pure text page with PyMuPDF (Tier 1 - fastest)."""
        # Extract text with reading order
        text = page.get_text("text")
        return ContentPage(
            page_number=page_idx + 1,
            content=text,
            quality_score=0.95,  # High confidence for pure text
            metadata={'processing_method': 'pymupdf', 'analysis': analysis}
        )
    
    def _extract_pymupdf_stats(self, page: fitz.Page) -> Dict[str, Any]:
        """
        Extract text statistics from page using PyMuPDF for text density calculation.
        
        This provides the data needed by PageClassifier to calculate text_density
        and enable Tier 1 (PyMuPDF-only) routing for pure text pages.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Dict with keys:
                - word_count: Total number of words
                - avg_words_per_line: Average words per line
                - char_count: Total character count
                - line_count: Number of text lines
        """
        try:
            # Extract text with position info
            text = page.get_text("text")
            
            # Calculate basic statistics
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            lines = [line for line in text.split('\n') if line.strip()]
            line_count = len(lines)
            
            # Calculate average words per line
            avg_words_per_line = word_count / line_count if line_count > 0 else 0
            
            stats = {
                'word_count': word_count,
                'char_count': char_count,
                'line_count': line_count,
                'avg_words_per_line': round(avg_words_per_line, 2)
            }
            
            self.logger.debug(
                f"PyMuPDF stats: {word_count} words, {line_count} lines, "
                f"{avg_words_per_line:.1f} words/line"
            )
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Failed to extract PyMuPDF stats: {e}")
            # Return empty stats on failure
            return {
                'word_count': 0,
                'char_count': 0,
                'line_count': 0,
                'avg_words_per_line': 0.0
            }
    
    def _process_with_two_shot(self, profile: 'DocumentProfile', page: fitz.Page, page_idx: int) -> 'ContentPage':
        """
        Process complex page with two-shot approach for maximum accuracy (Tier 4).
        
        Two-Shot Architecture:
        1. Shot 1 (Raw OCR): Faithful transcription at temp=0.05
        2. Deduplication: Paragraph-level cleanup to remove repetition
        3. Shot 2 (Analysis): Structure into JSON at temp=0.0
        
        This decouples "seeing" from "thinking" for superior accuracy on:
        - Complex visual layouts (mixed tables + figures)
        - Dense multi-column text
        - Pages with equations and diagrams
        
        Expected processing time: ~90 seconds per page
        Expected quality score: 0.82+ (OlmOCR-2-7B-1025 benchmark)
        
        Args:
            profile: DocumentProfile containing the PDF
            page: PyMuPDF page object
            page_idx: Zero-indexed page number
            
        Returns:
            ContentPage with structured content and metadata
        """
        try:
            self.logger.info(f"Processing page {page_idx + 1} with TWO-SHOT method (Tier 4)...")
            start_time = time.time()
            
            # Prepare image and anchor text
            image = self._render_page_pypdfium(profile.pdf_path, page_idx, target_size=self.body_resolution)
            anchor_text = DocumentAnchoringExtractor.extract_document_anchoring(
                pdf_path=profile.pdf_path,
                page_num=page_idx + 1
            )
            
            self.logger.debug("Executing Shot 1: Raw OCR extraction (temp=0.05)...")
            shot1_start = time.time()
            
            # Load two-shot parameters from VLM profile
            two_shot_params = self.body_vlm_profile.get('two_shot_parameters', {})
            shot1_params = two_shot_params.get('shot1_ocr', {})
            
            # Create temporary provider with Shot 1 parameters (resolved from registry)
            shot1_provider = self._create_two_shot_provider(
                task_type=self._resolve_reference(
                    self.config['vlm_provider']['body_pages']['tier4_two_shot']['shot1_task'],
                    self.config
                ),
                temperature=shot1_params.get('temperature', 0.05),
                max_tokens=shot1_params.get('max_tokens', 8192),
                repeat_penalty=shot1_params.get('repeat_penalty', 1.15),
                frequency_penalty=shot1_params.get('frequency_penalty', 0.1)
            )
            
            # Execute Shot 1 WITH anchor text for complete page extraction
            # The anchor text provides structural guidance about page content
            # Prevents early stopping at visual boundaries (diagrams, figures)
            shot1_result = shot1_provider.process_page(image, base_text=anchor_text)
            raw_text = shot1_result.get('content', '')
            shot1_time = time.time() - shot1_start
            
            self.logger.debug(f"Shot 1 complete: {len(raw_text)} characters in {shot1_time:.2f}s")
            
            if not raw_text:
                self.logger.warning("Shot 1 returned empty text, aborting two-shot processing")
                return ContentPage(
                    page_number=page_idx + 1,
                    content="",
                    quality_score=0.0,
                    metadata={'error': 'shot1_empty', 'processing_method': 'two_shot_failed'}
                )
            
            self.logger.debug("Performing paragraph-level deduplication...")
            dedup_start = time.time()
            
            cleaned_text = self._deduplicate_paragraphs(raw_text)
            dedup_time = time.time() - dedup_start
            
            self.logger.debug(
                f"Deduplication complete: {len(raw_text)} -> {len(cleaned_text)} chars "
                f"({len(cleaned_text)/len(raw_text)*100:.1f}% retained) in {dedup_time:.2f}s"
            )
            
            self.logger.debug("Executing Shot 2: Structured analysis (temp=0.0)...")
            shot2_start = time.time()
            
            # Load Shot 2 parameters
            shot2_params = two_shot_params.get('shot2_analysis', {})
            
            # Create temporary provider with Shot 2 parameters (resolved from registry)
            shot2_provider = self._create_two_shot_provider(
                task_type=self._resolve_reference(
                    self.config['vlm_provider']['body_pages']['tier4_two_shot']['shot2_task'],
                    self.config
                ),
                temperature=shot2_params.get('temperature', 0.0),
                max_tokens=shot2_params.get('max_tokens', 8192),
                repeat_penalty=shot2_params.get('repeat_penalty', 1.15),
                frequency_penalty=shot2_params.get('frequency_penalty', 0.1)
            )
            
            # Inject cleaned text into Shot 2 prompt
            # The task definition has {raw_text} placeholder
            shot2_result = shot2_provider.process_page(
                image,
                base_text=cleaned_text  # Pass cleaned text as base_text
            )
            
            shot2_time = time.time() - shot2_start
            self.logger.debug(f"Shot 2 complete in {shot2_time:.2f}s")
            
            # Extract results
            # CRITICAL: Shot 2's metadata contains the structured JSON with 'natural_text' field
            # The 'content' field is what was extracted from _post_process_output
            final_metadata = shot2_result.get('metadata', {})
            
            # CRITICAL FIX: Use Shot 2's 'natural_text' field (which should be cleaned_text)
            # If Shot 2 fails to return natural_text, fall back to cleaned_text
            final_content = final_metadata.get('natural_text', cleaned_text)
            
            # If natural_text is empty but we have cleaned_text, use it
            if not final_content and cleaned_text:
                self.logger.warning("Shot 2 returned empty natural_text, using cleaned text from Shot 1")
                final_content = cleaned_text
            
            quality_score = shot2_result.get('quality_score', 0.75)
            
            # Add processing metadata
            final_metadata['processing_method'] = 'two_shot'
            final_metadata['shot1_time'] = round(shot1_time, 2)
            final_metadata['shot2_time'] = round(shot2_time, 2)
            final_metadata['dedup_time'] = round(dedup_time, 2)
            final_metadata['total_time'] = round(time.time() - start_time, 2)
            final_metadata['raw_text_length'] = len(raw_text)
            final_metadata['cleaned_text_length'] = len(cleaned_text)
            final_metadata['dedup_ratio'] = round(len(cleaned_text)/len(raw_text), 3) if raw_text else 0.0
            
            self.logger.info(
                f"Two-shot processing complete for page {page_idx + 1}: "
                f"{final_metadata['total_time']}s total, quality={quality_score:.2f}"
            )
            
            return ContentPage(
                page_number=page_idx + 1,
                content=final_content,
                quality_score=quality_score,
                metadata=final_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Two-shot processing failed for page {page_idx + 1}: {e}", exc_info=True)
            return ContentPage(
                page_number=page_idx + 1,
                content="",
                quality_score=0.0,
                metadata={'error': str(e), 'processing_method': 'two_shot_failed'}
            )

    async def _process_with_two_shot_async(
        self,
        async_provider: AsyncUniversalVLMProvider,
        image: Any, # Image.Image
        page_idx: int,
        formatted_anchor: Optional[str] = None
    ) -> 'ContentPage':
        """
        Async version of Two-Shot processing for Tier 4.
        
        Fixes Blocking Issue in Phase 4.9 (Fix 1).
        Uses proper await calls for Shot 1 and Shot 2.
        """
        try:
            self.logger.info(f"Processing page {page_idx + 1} with ASYNC TWO-SHOT method (Tier 4)...")
            start_time = time.time()
            
            # Load two-shot parameters
            two_shot_params = self.body_vlm_profile.get('two_shot_parameters', {})
            shot1_params = two_shot_params.get('shot1_ocr', {})
            shot2_params = two_shot_params.get('shot2_analysis', {})
            
            shot1_task_name = self.config['vlm_provider']['body_pages']['tier4_two_shot']['shot1_task']
            shot1_task_type = self._resolve_reference(shot1_task_name, self.config)
            
            # Use async provider with override params
            shot1_wrapper = await async_provider._process_single_page_wrapper_async(
                page_image=image,
                base_text=formatted_anchor,
                task_definition=self.body_provider._load_task_definition(shot1_task_type),
                page_num=page_idx + 1,
                api_params_override={
                    'temperature': shot1_params.get('temperature', 0.05),
                    'max_tokens': shot1_params.get('max_tokens', 8192),
                    'repeat_penalty': shot1_params.get('repeat_penalty', 1.15),
                    'frequency_penalty': shot1_params.get('frequency_penalty', 0.1)
                }
            )
            
            if not shot1_wrapper.get('success'):
                raise RuntimeError(f"Shot 1 failed: {shot1_wrapper.get('error')}")
                
            raw_text = shot1_wrapper.get('content', '')
            shot1_time = time.time() - start_time
            
            # Fast CPU op, safe to run sync in this coroutine context
            cleaned_text = self._deduplicate_paragraphs(raw_text)
            dedup_time = time.time() - (start_time + shot1_time)
            
            shot2_start_abs = time.time()
            shot2_task_name = self.config['vlm_provider']['body_pages']['tier4_two_shot']['shot2_task']
            shot2_task_type = self._resolve_reference(shot2_task_name, self.config)
            
            shot2_wrapper = await async_provider._process_single_page_wrapper_async(
                page_image=image,
                base_text=cleaned_text, # Inject cleaned text
                task_definition=self.body_provider._load_task_definition(shot2_task_type),
                page_num=page_idx + 1,
                api_params_override={
                    'temperature': shot2_params.get('temperature', 0.0),
                    'max_tokens': shot2_params.get('max_tokens', 8192),
                    'repeat_penalty': shot2_params.get('repeat_penalty', 1.15),
                    'frequency_penalty': shot2_params.get('frequency_penalty', 0.1)
                }
            )
            
            shot2_time = time.time() - shot2_start_abs
            
            # Extract Results
            final_metadata = shot2_wrapper.get('metadata', {})
            final_content = final_metadata.get('natural_text', cleaned_text)
            if not final_content and cleaned_text:
                final_content = cleaned_text
                
            quality_score = shot2_wrapper.get('quality_score', 0.0)
            
            # Add Metrics
            final_metadata['processing_method'] = 'two_shot_async'
            final_metadata['shot1_time'] = round(shot1_time, 2)
            final_metadata['shot2_time'] = round(shot2_time, 2)
            final_metadata['dedup_time'] = round(dedup_time, 2)
            final_metadata['total_time'] = round(time.time() - start_time, 2)
            
            return ContentPage(
                page_number=page_idx + 1,
                content=final_content,
                quality_score=quality_score,
                metadata=final_metadata
            )

        except Exception as e:
            self.logger.error(f"Async two-shot failed for page {page_idx + 1}: {e}")
            return ContentPage(
                page_number=page_idx + 1,
                content="",
                quality_score=0.0,
                metadata={'error': str(e), 'tier': 'tier4_async_failed'}
            )
    
    def _deduplicate_paragraphs(self, text: str) -> str:
        """
        Perform paragraph-level deduplication to remove repetitive text blocks.
        
        The OlmOCR model can sometimes generate the same paragraph multiple times
        during Shot 1. This method removes duplicates while preserving order.
        
        Algorithm:
        1. Split text by double-newlines (paragraph boundaries)
        2. Track unique paragraphs in order of first appearance
        3. Rejoin with double-newlines
        
        Args:
            text: Raw text from Shot 1
            
        Returns:
            Deduplicated text with unique paragraphs preserved in order
            
        Example:
            Input:  "Para A\n\nPara B\n\nPara A\n\nPara C"
            Output: "Para A\n\nPara B\n\nPara C"
        """
        if not text:
            return text
        
        # Split by paragraph boundaries (double newlines)
        paragraphs = text.split('\n\n')
        
        # Track seen paragraphs and preserve order
        seen = set()
        unique_paragraphs = []
        
        for para in paragraphs:
            # Normalize whitespace for comparison
            normalized = ' '.join(para.split())
            
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_paragraphs.append(para)
        
        # Rejoin with double newlines
        return '\n\n'.join(unique_paragraphs)
    
    def _create_two_shot_provider(self, task_type: str, temperature: float, 
                                   max_tokens: int, repeat_penalty: float,
                                   frequency_penalty: float) -> UniversalVLMProvider:
        """
        Create a temporary VLM provider with custom parameters for two-shot processing.
        
        This creates a provider instance with overridden API parameters specifically
        for Shot 1 or Shot 2, without modifying the base profile.
        
        Args:
            task_type: Task definition to use ('raw_ocr_extraction' or 'structured_analysis_body_page')
            temperature: Temperature setting for this shot
            max_tokens: Maximum tokens for this shot
            repeat_penalty: Repeat penalty setting
            frequency_penalty: Frequency penalty setting
            
        Returns:
            UniversalVLMProvider configured for this specific shot
        """
        # Load task definition
        task_definition_data = self._load_task_definition(task_type)
        pydantic_task = TaskDefinition(**task_definition_data)
        
        # Create modified profile with shot-specific parameters
        modified_profile_data = self.body_vlm_profile.copy()
        modified_profile_data['api_parameters'] = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'repeat_penalty': repeat_penalty,
            'frequency_penalty': frequency_penalty
        }
        
        pydantic_profile = ModelProfile(**modified_profile_data)
        
        # Create provider with modified config
        return UniversalVLMProvider(
            model_name=pydantic_profile.model_name,
            task_type=task_type,
            model_profile=pydantic_profile,
            task_definition=pydantic_task,
            skip_validation=True,  # OPTIMIZATION: Skip redundant health check for page-level providers
            api_endpoint=self.api_endpoint
        )
    
    def _generate_figure_descriptions(self, page: fitz.Page, analysis: Dict) -> List[Dict]:
        """Generates descriptions for all figure regions on a page."""
        figure_regions = []
        # Use 'visual_regions' as this is what the PageClassifier provides
        for region in analysis.get('visual_regions', []):
            try:
                bbox = region['bbox']
                clip = fitz.Rect(bbox)
                
                # Render the specific region at high DPI for clarity
                pix = page.get_pixmap(clip=clip, dpi=300)
                img_bytes = pix.tobytes("png")
                
                description = self.surgical_helper.describe_figure(self.body_provider, img_bytes)
                
                figure_regions.append({'content': description, 'bbox': bbox, 'type': 'figure'})
            except Exception as e:
                self.logger.warning(f"Failed to generate figure description for region {region.get('bbox')}: {e}")
        return figure_regions

    def _generate_equation_latex(self, page: fitz.Page, analysis: Dict) -> List[Dict]:
        """Generates LaTeX for all equation regions on a page."""
        equation_regions = []
        for region in analysis.get('equation_regions', []):
            try:
                bbox = region['bbox']
                clip = fitz.Rect(bbox)
                
                pix = page.get_pixmap(clip=clip, dpi=300)
                img_bytes = pix.tobytes("png")
                
                latex = self.surgical_helper.convert_equation_to_latex(self.body_provider, img_bytes)
                
                # The reconstructor will handle spacing, but we add the $$ wrappers here
                equation_regions.append({'content': f"$${latex}$$", 'bbox': bbox, 'type': 'equation'})
            except Exception as e:
                self.logger.warning(f"Failed to generate equation LaTeX for region {region.get('bbox')}: {e}")
        return equation_regions

    def _generate_table_markdowns(self, page: fitz.Page, analysis: Dict) -> List[Dict]:
        """Generates Markdown for all table regions on a page."""
        table_regions = []
        # Tables are also detected as 'visual_regions', so we'll need to distinguish them.
        # For now, we assume all visual regions on a 'TABLE' page are tables.
        for region in analysis.get('visual_regions', []):
            try:
                bbox = region['bbox']
                clip = fitz.Rect(bbox)
                
                pix = page.get_pixmap(clip=clip, dpi=300)
                img_bytes = pix.tobytes("png")
                
                markdown = self.surgical_helper.describe_table(self.body_provider, img_bytes)
                
                table_regions.append({'content': markdown, 'bbox': bbox, 'type': 'table'})
            except Exception as e:
                self.logger.warning(f"Failed to generate table markdown for region {region.get('bbox')}: {e}")
        return table_regions

    async def _generate_figure_descriptions_async(self, async_provider: AsyncUniversalVLMProvider, page: fitz.Page, analysis: Dict) -> List[Dict]:
        """Async version of figure description generation."""
        figure_regions = []
        for region in analysis.get('visual_regions', []):
            try:
                bbox = region['bbox']
                clip = fitz.Rect(bbox)
                pix = page.get_pixmap(clip=clip, dpi=300)
                img_bytes = pix.tobytes("png")
                
                description = await self.surgical_helper.describe_figure_async(async_provider, img_bytes)
                figure_regions.append({'content': description, 'bbox': bbox, 'type': 'figure'})
            except Exception as e:
                self.logger.warning(f"Failed to generate async figure description: {e}")
        return figure_regions

    async def _generate_equation_latex_async(self, async_provider: AsyncUniversalVLMProvider, page: fitz.Page, analysis: Dict) -> List[Dict]:
        """Async version of equation LaTeX generation."""
        equation_regions = []
        for region in analysis.get('equation_regions', []):
            try:
                bbox = region['bbox']
                clip = fitz.Rect(bbox)
                pix = page.get_pixmap(clip=clip, dpi=300)
                img_bytes = pix.tobytes("png")
                
                latex = await self.surgical_helper.convert_equation_to_latex_async(async_provider, img_bytes)
                equation_regions.append({'content': f"$${latex}$$", 'bbox': bbox, 'type': 'equation'})
            except Exception as e:
                self.logger.warning(f"Failed to generate async equation LaTeX: {e}")
        return equation_regions

    async def _generate_table_markdowns_async(self, async_provider: AsyncUniversalVLMProvider, page: fitz.Page, analysis: Dict) -> List[Dict]:
        """Async version of table markdown generation."""
        table_regions = []
        # As per sync logic, treat visual regions on TABLE pages (or just all visual regions if general)
        # Sync logic iterates visual_regions.
        for region in analysis.get('visual_regions', []):
            try:
                bbox = region['bbox']
                clip = fitz.Rect(bbox)
                pix = page.get_pixmap(clip=clip, dpi=300)
                img_bytes = pix.tobytes("png")
                
                markdown = await self.surgical_helper.describe_table_async(async_provider, img_bytes)
                table_regions.append({'content': markdown, 'bbox': bbox, 'type': 'table'})
            except Exception as e:
                self.logger.warning(f"Failed to generate async table markdown: {e}")
        return table_regions

    async def _process_with_surgical_vlm_async(
        self,
        async_provider: AsyncUniversalVLMProvider,
        profile: 'DocumentProfile',
        page: fitz.Page,
        page_idx: int,
        page_type: str,
        analysis: Dict
    ) -> 'ContentPage':
        """
        Async version of Surgical VLM processing (Tier 2).
        Fixes Fix 2 (Async Surgical VLM).
        """
        try:
            self.logger.info(f"Processing page {page_idx + 1} with ASYNC SURGICAL method (Tier 2)...")
            start_time = time.time()
            
            # 1. Extract Text Blocks (Fast CPU op)
            text_blocks = self.reconstructor.extract_text_blocks(page)

            # 2. Extract Visual Components (Async)
            # Determine what to extract based on page type
            if page_type == 'TABLE':
                # Table-only page
                table_task = self._generate_table_markdowns_async(async_provider, page, analysis)
                table_regions = await table_task
                figure_regions = []
                equation_regions = []
            else:
                # Standard pages: Figures + Equations (concurrent)
                visual_tasks = []
                visual_tasks.append(self._generate_figure_descriptions_async(async_provider, page, analysis))

                if analysis.get('equation_regions'):
                    visual_tasks.append(self._generate_equation_latex_async(async_provider, page, analysis))

                # Run concurrently
                results = await asyncio.gather(*visual_tasks)
                figure_regions = results[0] if results else []
                equation_regions = results[1] if len(results) > 1 else []
                table_regions = []

            # 3. Reconstruct Reading Order (Sync, fast CPU)
            final_content = self.reconstructor.reconstruct(
                text_blocks=text_blocks,
                figure_regions=figure_regions,
                equation_regions=equation_regions,
                table_regions=table_regions
            )
            
            total_time = time.time() - start_time

            # Calculate total visual elements
            total_visual_elements = len(figure_regions) + len(equation_regions) + len(table_regions)

            return ContentPage(
                page_number=page_idx + 1,
                content=final_content,
                quality_score=0.85, # Standard score for surgical
                metadata={
                    'processing_method': 'surgical_async',
                    'page_type': page_type,
                    'visual_element_count': total_visual_elements,
                    'text_blocks': len(text_blocks),
                    'figure_regions': len(figure_regions),
                    'equation_regions': len(equation_regions),
                    'table_regions': len(table_regions),
                    'total_time': round(total_time, 2)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Async surgical processing failed for page {page_idx + 1}: {e}")
            return ContentPage(
                page_number=page_idx + 1,
                content="", # Failure
                quality_score=0.0,
                metadata={'error': str(e), 'processing_method': 'surgical_async_failed'}
            )
    
    # ============================================================================
    # METADATA & UTILITY METHODS
    # ============================================================================
    
    def _extract_metadata_from_page1(self, page1_content: 'ContentPage', pdf_path: str) -> Dict:
        """
        Extracts key bibliographic metadata from the processed title page's
        metadata dictionary, handling the nested structure from the VLM.

        Args:
            page1_content: The processed ContentPage object for page 1.

        Returns:
            A dictionary with extracted bibliographic data (title, authors, doi, source_hash).
        """
        bibliographic_metadata = {}

        # 1. Validate structure
        if not page1_content or not isinstance(page1_content.metadata, dict):
            self.logger.warning("Title page content or its metadata is invalid, skipping extraction.")
            return bibliographic_metadata

        # 2. Extract nested VLM metadata block
        # 2. Extract nested VLM metadata block
        # Check for nested 'metadata' first (legacy/schema constraint), then fall back to flat structure
        vlm_metadata_block = page1_content.metadata.get('metadata')
        
        if not isinstance(vlm_metadata_block, dict):
            # The provider likely unwrapped it already. Use the top-level metadata dict.
            vlm_metadata_block = page1_content.metadata
            
            # Validation trace
            self.logger.debug(f"Using flat metadata structure. Keys: {list(vlm_metadata_block.keys())}")

        # 3. Extract fields (safe access)
        bibliographic_metadata['title'] = vlm_metadata_block.get('title')
        
        authors = vlm_metadata_block.get('authors')
        bibliographic_metadata['authors'] = authors if isinstance(authors, list) else []

        bibliographic_metadata['doi'] = vlm_metadata_block.get('doi')
        bibliographic_metadata['year'] = vlm_metadata_block.get('publication_year')
        bibliographic_metadata['journal'] = vlm_metadata_block.get('journal_name')

        # 4. Add stable source_hash for ledger identity (PATCH #1)
        try:
            # ALWAYS use profile.pdf_path, never self.pdf_path
            pdf_path = page1_content.metadata.get("document_path") or getattr(page1_content, "document_path", None)
            if not pdf_path:
                pdf_path = self.current_pdf_path  # NEW: stored by process()

            source_hash = generate_hash(pdf_path)
            bibliographic_metadata['source_hash'] = source_hash
            self.logger.info(f"Assigned stable source_hash from PDF: {source_hash[:12]}...")
        except Exception as e:
            self.logger.error(f"Failed to generate source_hash: {e}")
            bibliographic_metadata['source_hash'] = f"fallback_{time.time_ns()}"

        # 5. Logging (unchanged)
        if bibliographic_metadata.get('doi'):
            self.logger.info(f"Successfully extracted DOI: {bibliographic_metadata['doi']}")
        else:
            self.logger.warning("DOI was not found in the title page metadata block.")

        return bibliographic_metadata

    def _enrich_metadata_with_cache_and_zotero(
        self,
        vlm_metadata: dict,
        pdf_path: Path,
        title_page_text: str = "",
        profile: "DocumentProfile" = None  # Accept the profile
    ) -> dict:
        """
        Phase 3.6 final enrichment step:
        - Checks ledger cache
        - Queries Zotero
        - Merges metadata
        - Detects missing/suspicious fields
        - Triggers user-in-loop selective correction
        """

        self.logger.info("Enriching metadata using cache + Zotero + referee text…")

        # Prefer the DOI the orchestrator already read straight out of the PDF's text layer.
        # This must happen BEFORE the Zotero lookup below, which matches on DOI first.
        vlm_metadata = self._prefer_text_layer_doi(vlm_metadata, profile)

        document_id = generate_hash(str(pdf_path))

        # ==============================
        # 1. LEDGER CHECK
        # ==============================
        if self.ledger:
            cached_record = self.ledger.get_record(document_id)
            if cached_record:
                self.logger.info(f"Ledger cache hit for document {document_id[:12]}")
                return cached_record.model_dump()

        # ==============================
        # 2. ZOTERO LOOKUP
        # ==============================
        zotero_record = {}
        zotero_confidence = 0.0
        zotero_source = "unknown"
        if self.zotero_service:
            try:
                self.logger.info("Performing Zotero canonical lookup...")
                zr = self.zotero_service.get_canonical_metadata(vlm_metadata)
                zotero_record = zr.get("record", {})
                zotero_confidence = zr.get("confidence", 0.0)
                zotero_source = zr.get("source", "unknown")
            except Exception as e:
                self.logger.error(f"Zotero lookup failed: {e}")
                zotero_record = {}

        # ==============================
        # 3. MERGE METADATA
        # ==============================
        merged = merge_metadata(vlm_metadata, zotero_record, title_page_text)
        
        merged["validation_confidence"] = zotero_confidence
        merged["validation_source"] = zotero_source

        self.logger.info(f"Merged Metadata (pre-review): {merged}")

        # ==============================
        # 4. ANNOTATE PROVENANCE FOR RUNNER
        # ==============================
        provenance = self._annotate_provenance(merged)
        
        # CRITICAL FIX: Remove both provenance AND _provenance to prevent circular references
        # The old _provenance field may still exist from legacy code
        merged.pop("_provenance", None)
        merged.pop("provenance", None)  # Remove the one we just created too
        
        # Now add it back cleanly
        merged["provenance"] = provenance
        
        bibliographic_metadata = merged

        self.logger.debug(f"Provenance annotations: {merged.get('_provenance', {})}")

        # Strategy returns metadata to Runner
        # Runner decides when/if to write to ledger

        return merged

    def _prefer_text_layer_doi(self, vlm_metadata: dict, profile: "DocumentProfile" = None) -> dict:
        """
        Use the DOI read directly from the PDF's text layer in preference to the VLM's.

        For biology researchers: a DOI is a paper's permanent identifier, e.g.
        "10.1016/j.cellbi.2005.09.009". One wrong character and Zotero fetches the wrong
        paper, or none at all. Born-digital PDFs (produced from a word processor rather than
        scanned) store the DOI as exact, machine-readable text. DocumentOrchestrator already
        extracts it from that text layer before any model runs, scoring it by HOW it was
        found: 0.98 when printed next to the label "doi:", 0.95 as a doi.org URL, and only
        0.85 for a bare "10.xxxx/..." floating in the text -- which might be a citation to a
        DIFFERENT paper rather than this one's identifier.

        We accept that DOI only when its confidence clears `metadata.doi_confidence_threshold`
        (default 0.85). Below the bar we keep the VLM's reading, since a low-confidence regex
        hit is no better than a vision model's guess. A vision model, for its part, must
        recognise the DOI from pixels of tiny footer print and can transpose digits.

        Args:
            vlm_metadata: Metadata dict from the title-page VLM. Modified copy is returned.
            profile: DocumentProfile carrying `.doi` / `.doi_confidence` from the
                     orchestrator's first-pass regex scan. May be None (e.g. in unit tests).

        Returns:
            The metadata dict, with 'doi' backfilled or corrected from the text layer when it
            is trustworthy. Sets 'doi_source' so the ledger records which source won.
        """
        text_layer_doi = getattr(profile, 'doi', None) if profile else None
        vlm_doi = vlm_metadata.get('doi')

        # No text-layer DOI (scanned page, or none printed): keep whatever the VLM found.
        if not text_layer_doi:
            if vlm_doi:
                vlm_metadata['doi_source'] = 'vlm_extraction'
            return vlm_metadata

        confidence = getattr(profile, 'doi_confidence', 0.0) or 0.0

        # Honour the configured threshold rather than blindly trusting any regex hit.
        if confidence < self.doi_threshold:
            self.logger.info(
                f"Text-layer DOI '{text_layer_doi}' scored {confidence:.2f}, below the "
                f"{self.doi_threshold:.2f} threshold (likely a bare DOI that may belong to a "
                f"cited paper). Keeping the VLM's DOI."
            )
            if vlm_doi:
                vlm_metadata['doi_source'] = 'vlm_extraction'
            return vlm_metadata

        # A disagreement means the VLM misread a DOI that was sitting in the text layer.
        # Worth surfacing: it is the clearest signal that OCR is garbling small print.
        if vlm_doi and normalize_doi_for_compare(vlm_doi) != normalize_doi_for_compare(text_layer_doi):
            self.logger.warning(
                f"DOI mismatch - VLM read '{vlm_doi}' but the PDF text layer says "
                f"'{text_layer_doi}' (confidence={confidence:.2f}). Trusting the text layer."
            )
        elif not vlm_doi:
            self.logger.info(
                f"VLM found no DOI; recovered '{text_layer_doi}' from the PDF text layer "
                f"(confidence={confidence:.2f})."
            )

        vlm_metadata['doi'] = text_layer_doi
        vlm_metadata['doi_source'] = 'pymupdf_text_layer'
        vlm_metadata['doi_confidence'] = confidence
        return vlm_metadata

    def _render_page_pypdfium(self, pdf_path: str, page_num: int, target_size: int) -> Image.Image:
        """
        Renders a PDF page using pypdfium2 (EXACT match to OlmOCR Hybrid v2).
        
        This is the PROVEN working method from OlmOCR_run_hybrid_v2.py.
        Takes PDF path directly and renders with pypdfium2.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: 0-indexed page number
            target_size: Target size for longest dimension
            
        Returns:
            PIL Image with 25px white padding
        """
        import pypdfium2 as pdfium
        
        # Open PDF with pypdfium2
        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[page_num]
        
        # Calculate scale factor
        width_pts, height_pts = page.get_size()
        longest_dim_pts = max(width_pts, height_pts)
        
        if longest_dim_pts == 0:
            raise ValueError(f"Page {page_num + 1} has zero dimensions")
        
        scale = target_size / longest_dim_pts
        
        # Render at calculated scale
        image = page.render(scale=scale).to_pil()
        
        # Add white padding (matching OlmOCR Hybrid v2). Uses the shared class constant so the
        # caption pixel->PDF-point mapping stays in lockstep with this border.
        padding = self.RENDER_PADDING_PX
        new_size = (image.width + 2 * padding, image.height + 2 * padding)
        padded_image = Image.new("RGB", new_size, (255, 255, 255))
        padded_image.paste(image, (padding, padding))
        
        pdf.close()
        
        return padded_image

    def _render_and_pad_image(self, page: fitz.Page, target_size: int, dpi: int) -> Image.Image:
        """
        Renders a PDF page to a padded, square PIL Image using pypdfium2 (matching OlmOCR Hybrid v2).
        
        CRITICAL: Uses pypdfium2 for rendering, NOT PyMuPDF, because:
        - pypdfium2 produces correct full-page renders
        - PyMuPDF get_pixmap() was producing corrupted images (only Box 1 visible)
        - This matches the working OlmOCR_run_hybrid_v2.py implementation
        
        Args:
            page: The PyMuPDF page object (used only for page number reference)
            target_size: The final square dimension for the image (e.g., 1288 or 1024)
            dpi: IGNORED - using scale-based rendering instead
            
        Returns:
            A PIL.Image object, padded to be square with 25px white padding
        """
        import pypdfium2 as pdfium
        
        # Get page number from PyMuPDF page
        page_num = page.number
        
        # Re-open with pypdfium2 for correct rendering
        pdf = pdfium.PdfDocument(str(page.parent.name))  # page.parent is the Document
        pdfium_page = pdf[page_num]
        
        # Calculate scale to make longest dimension = target_size
        width_pts, height_pts = pdfium_page.get_size()
        longest_dim_pts = max(width_pts, height_pts)
        
        if longest_dim_pts == 0:
            raise ValueError(f"Page {page_num + 1} has zero dimensions")
        
        scale = target_size / longest_dim_pts
        
        # Render at calculated scale
        image = pdfium_page.render(scale=scale).to_pil()
        
        # Add 25px white padding (matching OlmOCR Hybrid v2, not 200px!)
        padding = 25
        new_size = (image.width + 2 * padding, image.height + 2 * padding)
        padded_image = Image.new("RGB", new_size, (255, 255, 255))
        padded_image.paste(image, (padding, padding))
        
        pdf.close()
        
        return padded_image
    
    def _resolve_author_ambiguity(
        self, 
        suspicious: List[str], 
        confirmed: List[str], 
        page: 'fitz.Page'
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Use Tesseract OCR to analyze suspicious author strings.
        Returns cleaned author list + provenance metadata.
        
        NO USER INTERACTION - only automated analysis.
        """
        provenance = {
            "tesseract_checked": False,
            "tesseract_verdict": None
        }
        
        if not suspicious:
            return confirmed, provenance
        
        suspicious_str = suspicious[0]
        self.logger.info(f"Analyzing suspicious author string: '{suspicious_str}'")
        
        bboxes = page.search_for(suspicious_str)
        if bboxes:
            bbox = bboxes[0]
            pix = page.get_pixmap(clip=bbox, dpi=300)
            pil_image = pixmap_to_pil(pix)
            
            try:
                ocr_snippet = pytesseract.image_to_string(pil_image).strip()
                provenance["tesseract_checked"] = True
                provenance["tesseract_result"] = ocr_snippet[:100]  # Store snippet
                
                # If Tesseract shows multi-line, it's a list
                if "\n" in ocr_snippet and len(ocr_snippet.splitlines()) > 1:
                    self.logger.info("Tesseract confirms multi-line author list")
                    provenance["tesseract_verdict"] = "confirmed_list"
                    return confirmed, provenance
                else:
                    provenance["tesseract_verdict"] = "uncertain"
            except Exception as e:
                self.logger.warning(f"Tesseract failed: {e}")
                provenance["tesseract_error"] = str(e)
        
        # Return confirmed list + provenance for Runner
        return confirmed, provenance
    
    def _load_strategy_config(self) -> Dict:
        """
        Load strategy configuration from YAML file.
        
        Loads research_article_strategy.yaml from config directory.
        Falls back to safe defaults if config file not found.
        
        Returns:
            Dict: Strategy configuration settings
        
        Raises:
            FileNotFoundError: If config file not found and no defaults available
        """
        config_path = Path(__file__).parent / 'config' / 'research_article_strategy.yaml'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Loaded strategy config from: {config_path}")
                return config
        
        except FileNotFoundError:
            self.logger.error(f"Strategy config not found: {config_path}")
            
            # Provide minimal fallback config (backward compatibility)
            self.logger.warning("Using fallback default configuration")
            return {
                'strategy_name': 'research_article',
                'version': '1.0-fallback',
                'vlm_provider': {
                    'default_model': 'olmocr-7b',
                    'tasks': {
                        'title_page': {
                            'task_name': 'title_page_metadata_extraction',
                            'target_resolution': 1288
                        },
                        'body_pages': {
                            'task_name': 'olmocr_document_processing',
                            'target_resolution': 1024
                        }
                    }
                },
                'metadata': {
                    'doi_confidence_threshold': 0.85,
                    'isbn_confidence_threshold': 0.85
                },
                'processing': {
                    'continue_on_error': True
                },
                'quality': {
                    'minimum_page_score': 0.3
                }
            }
        
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse strategy config: {e}")
            raise
    
    def _initialize_provider_from_config(self, provider_config: Dict, api_endpoint: Optional[str] = None) -> Tuple[UniversalVLMProvider, Dict, int]:
        """
        Initialize a UniversalVLMProvider from strategy config (DRY helper).

        This now loads all config data, creates Pydantic models, and INJECTS them
        into the UniversalVLMProvider, making the strategy's config the single source of truth.
        """
        profile_name = provider_config['profile']
        task_type = provider_config['task_type']
        
        # 1. Load the raw dictionary data from YAML files
        vlm_profile_data = self._load_vlm_profile(profile_name)
        task_definition_data = self._load_task_definition(task_type)

        # 2. Create Pydantic model instances from the loaded data
        pydantic_model_profile = ModelProfile(**vlm_profile_data)
        pydantic_task_definition = TaskDefinition(**task_definition_data)

        # 3. Determine final resolution (strategy can override VLM default)
        resolution = provider_config.get('resolution_override', vlm_profile_data.get('default_resolution', 1024))
        
        self.logger.info(
            f"Initializing provider: {pydantic_model_profile.model_name} -> {task_type} "
            f"(resolution: {resolution}px)"
        )
        
        # 4. Create the provider, INJECTING the pre-loaded config objects
        # SKIP VALIDATION: In GUI lazy launch mode, server isn't running during strategy init
        provider = UniversalVLMProvider(
            model_name=pydantic_model_profile.model_name,
            task_type=task_type,
            model_profile=pydantic_model_profile,
            task_definition=pydantic_task_definition,
            api_endpoint=api_endpoint,
            skip_validation=True  # Skip health check during init (server may not be running yet)
        )
        
        return provider, vlm_profile_data, resolution
    
    def _load_vlm_profile(self, profile_name: str) -> Dict:
        """
        Load VLM profile configuration from vlm_profiles directory.
        
        VLM profiles are self-contained configs with model-specific settings:
        - model_name, api_parameters, default_resolution, etc.
        
        Args:
            profile_name: Name of VLM profile (e.g., "olmocr-7b")
        
        Returns:
            Dict: VLM profile configuration
        
        Raises:
            FileNotFoundError: If VLM profile not found
        """
        profile_path = Path(__file__).parent / 'config' / 'vlm_profiles' / f'{profile_name}.yaml'
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile = yaml.safe_load(f)
                self.logger.info(f"Loaded VLM profile: {profile_name} ({profile.get('model_type', 'unknown type')})")
                return profile
        
        except FileNotFoundError:
            self.logger.error(f"VLM profile not found: {profile_path}")
            raise FileNotFoundError(
                f"VLM profile '{profile_name}' not found. "
                f"Expected location: {profile_path}\n"
                f"Available profiles should be in: strategies/config/vlm_profiles/"
            )
        
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse VLM profile {profile_name}: {e}")
            raise

    def _load_task_definition(self, task_type: str) -> Dict:
        """
        Load task definition from the VLM provider's config directory.
        
        Args:
            task_type: Name of the task definition (e.g., "title_page_metadata_extraction").
        
        Returns:
            Dict: Task definition configuration.
        """
        # Note: This path points to the provider's config, not the strategy's.
        # This is correct as tasks are tightly coupled to the provider.
        task_path = Path(__file__).parent.parent.parent / 'llm' / 'providers' / 'config' / 'task_definitions' / f'{task_type}.yaml'
        
        try:
            # CRITICAL: Explicitly specify UTF-8 encoding to handle special characters
            # Windows defaults to cp1252 which fails on Unicode characters
            with open(task_path, 'r', encoding='utf-8') as f:
                task_def = yaml.safe_load(f)
                self.logger.info(f"Loaded task definition: {task_type}")
                return task_def
        except FileNotFoundError:
            self.logger.error(f"Task definition file not found: {task_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse task definition {task_type}: {e}")
            raise
    
    def _resolve_reference(self, value: Any, config: dict) -> Any:
        """
        Resolve generic registry references in configuration values.
        
        Supports @-prefixed references that point to registry entries:
        - @profile_registry.default -> config['profile_registry']['default']
        - @task_registry.raw_ocr -> config['task_registry']['raw_ocr']
        
        Non-reference values (strings without @-prefix, numbers, bools, etc.)
        are returned unchanged.
        
        Args:
            value: Configuration value (may be reference or literal)
            config: Full strategy config containing registries
            
        Returns:
            Resolved value (literal string/number/bool)
            
        Raises:
            ValueError: If reference syntax is invalid or key not found
            
        Examples:
            >>> self._resolve_reference("@profile_registry.default", config)
            "olmocr-2-7b"
            
            >>> self._resolve_reference("literal-value", config)
            "literal-value"
            
            >>> self._resolve_reference(123, config)
            123
        """
        # Only strings can be references
        if not isinstance(value, str):
            return value
        
        # Only @-prefixed strings are references
        if not value.startswith('@'):
            return value
        
        # Parse reference: @profile_registry.default -> ["profile_registry", "default"]
        ref_path = value[1:].split('.')
        
        if len(ref_path) < 2:
            raise ValueError(
                f"Invalid reference syntax: '{value}'. "
                f"Expected format: @registry_name.key (e.g., @profile_registry.default)"
            )
        
        # Navigate config dict to resolve reference
        current = config
        for i, key in enumerate(ref_path):
            if not isinstance(current, dict):
                raise ValueError(
                    f"Invalid reference: '{value}'. "
                    f"Path segment '{ref_path[i-1]}' is not a dictionary."
                )
            
            if key not in current:
                available_keys = ', '.join(current.keys()) if isinstance(current, dict) else 'N/A'
                raise ValueError(
                    f"Invalid reference: '{value}'. "
                    f"Key '{key}' not found in {'.'.join(ref_path[:i])}. "
                    f"Available keys: {available_keys}"
                )
            
            current = current[key]
        
        # Verify resolved value is not another reference (no recursion)
        if isinstance(current, str) and current.startswith('@'):
            raise ValueError(
                f"Invalid reference: '{value}'. "
                f"Registry values cannot be references (no recursion allowed). "
                f"Found: {current}"
            )
        
        self.logger.debug(f"Resolved reference '{value}' -> '{current}'")
        return current
    
    def _sanitize_bibliographic_metadata(self, bib: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure optional string fields are None (not empty string) and authors is a list.
        Called immediately before Pydantic model instantiation.
        """
        sanitized = dict(bib)  # shallow copy

        # Convert empty strings to None for optional text fields to avoid Pydantic regex/validation errors
        for k in ("doi", "title", "journal", "publisher", "validation_source", "pdf_filename"):
            v = sanitized.get(k)
            if isinstance(v, str) and v.strip() == "":
                sanitized[k] = None

        # Ensure authors is always a list
        a = sanitized.get("authors")
        if a is None:
            sanitized["authors"] = []
        elif isinstance(a, str):
            # If a comma-separated string slipped through, convert it to list
            sanitized["authors"] = [s.strip() for s in a.split(",") if s.strip()]
        elif isinstance(a, (list, tuple)):
            # normalize to list of stripped strings
            sanitized["authors"] = [str(x).strip() for x in a if str(x).strip()]
        else:
            # Fallback: ensure authors is a list
            sanitized["authors"] = []

        return sanitized

    def _annotate_provenance(self, merged: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create provenance information for Runner inspection.
        
        IMPORTANT: This method does NOT modify the merged dict.
        It only RETURNS a provenance dict.
        
        Args:
            merged: Merged metadata dict from merge_metadata()
            
        Returns:
            Provenance dict (not the merged dict!)
        """
        # Create fresh provenance dict
        provenance = {}
        
        # Annotate missing critical fields
        if not merged.get("title"):
            provenance["missing_title"] = True
        
        if not merged.get("authors"):
            provenance["missing_authors"] = True
        
        if not merged.get("doi"):
            provenance["missing_doi"] = True
        
        if not merged.get("journal"):
            provenance["missing_journal"] = True
        
        # Annotate suspicious authors (from merge_metadata)
        suspicious = merged.get("suspicious_authors", [])
        if suspicious:
            provenance["suspicious_authors"] = suspicious
            # Remove from top-level (it's metadata about metadata)
            merged.pop("suspicious_authors", None)
        
        # Annotate confidence scores
        provenance["validation_confidence"] = merged.get("validation_confidence", 0.0)
        provenance["validation_source"] = merged.get("validation_source", "unknown")
        
        # Annotate Zotero match quality
        retrieval_prov = merged.get("retrieval_provenance", {})
        if retrieval_prov:
            provenance["zotero_source"] = retrieval_prov.get("source_key", "none")
        
        return provenance  # Return ONLY the provenance dict