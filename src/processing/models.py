"""
Pydantic Models for Document Processing Pipeline

Provides type-safe, validated data structures for:
- Document profiles (pipeline state container)
- Classification signals (4-signal analysis results)
- Extracted metadata (standardized output)
- Processing results (strategy outputs)

Architecture Pattern: Progressive Enrichment
    Stage 1: First-pass analysis → DocumentProfile (basic fields)
    Stage 2: Classification → Update profile with classification results
    Stage 3: Strategy processing → Update profile with processing results

All models use Pydantic for:
    ✅ Automatic validation (type checking, constraints)
    ✅ Self-documentation (Field descriptions)
    ✅ IDE support (autocomplete, type hints)
    ✅ JSON serialization (for caching in Phase 3)
"""

from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


# ============================================================================
# ENUMS FOR TYPE SAFETY
# ============================================================================

class DocumentType(str, Enum):
    """Document classification types with type safety."""
    RESEARCH_ARTICLE = "research_article"
    BOOK_CHAPTER = "book_chapter"
    BOOK = "book"


class VLMTaskType(str, Enum):
    """VLM task types for Phase 3 batch processing."""
    TITLE_PAGE_EXTRACTION = "title_page_metadata_extraction"
    OLMOCR_PROCESSING = "olmocr_document_processing"


class PageType(str, Enum):
    """Page complexity classification for 3-tier hybrid processing.
    
    Used by PageClassifier to route pages to appropriate processing tier:
    - TEXT_ONLY: Pure text pages → Tier 1 (PyMuPDF) - 99% faster
    - SIMPLE_VISUAL: Simple figures/tables → Tier 2 (OlmoCR Markdown) - 82% faster
    - COMPLEX_VISUAL: Complex diagrams/equations → Tier 3 (Surgical VLM) - Targeted precision
    """
    TEXT_ONLY = "text_only"
    SIMPLE_VISUAL = "simple_visual"
    COMPLEX_VISUAL = "complex_visual"


# ============================================================================
# CLASSIFICATION MODELS
# ============================================================================

class ClassificationSignals(BaseModel):
    """
    Container for the four classification signals.
    
    Used by DocumentClassifier to store intermediate signal results
    before aggregation into final classification.
    
    Signals:
        1. metadata: PDF metadata analysis (DOI, ISBN, ISSN detection)
        2. page_count: Page count heuristics (article vs chapter vs book)
        3. text_content: Keyword analysis (abstract, chapters, TOC, etc.)
        4. structure: Document structure (first page, TOC pages, etc.)
    """
    metadata: Dict = Field(
        default_factory=dict,
        description="Signal 1: PDF metadata identifiers (DOI, ISBN, ISSN)"
    )
    page_count: Dict = Field(
        default_factory=dict,
        description="Signal 2: Page count heuristics (article/chapter/book patterns)"
    )
    text_content: Dict = Field(
        default_factory=dict,
        description="Signal 3: Text content analysis (keywords, patterns)"
    )
    structure: Dict = Field(
        default_factory=dict,
        description="Signal 4: Document structure (layout, TOC detection)"
    )
    
    class Config:
        validate_assignment = True  # Validate on field updates


# ============================================================================
# METADATA MODELS
# ============================================================================

class ExtractedMetadata(BaseModel):
    """
    Standardized metadata output from any document type.
    
    Used by all processing strategies to return metadata in consistent format.
    Supports both articles (DOI-based) and books/chapters (ISBN-based).
    
    Validation:
        - DOI: Must match pattern "10.xxxx/..." if present
        - ISBN: Must be 10 or 13 digits if present
        - Year: Must be 4-digit year if present
    """
    # Common fields (all document types)
    title: Optional[str] = Field(
        None,
        description="Document title",
        min_length=1,
        max_length=500
    )
    authors: List[str] = Field(
        default_factory=list,
        description="List of author names"
    )
    
    # Identifiers (mutually exclusive or complementary)
    doi: Optional[str] = Field(
        None,
        pattern=r'^10\.\d{4,}/.+$',
        description="Digital Object Identifier (articles/chapters) - just the DOI part without URL"
    )
    journal: Optional[str] = Field(
        None,
        description="Journal name (articles)"
    )
    isbn: Optional[str] = Field(
        None,
        pattern=r'^\d{10}(\d{3})?$',
        description="ISBN-10 or ISBN-13 (books/chapters)"
    )
    validation_source: Optional[str] = Field(
        None, 
        description="Source of the final metadata (e.g., 'vlm_extraction', 'zotero_doi_lookup', 'fuzzy_match', 'manual_entry')"
    )
    validation_confidence: Optional[float] = Field(
        None, 
        description="Confidence score (0.0-1.0) of the metadata's accuracy based on its source."
    )
    last_updated: Optional[str] = Field(
        None, description="Timestamp of when the record was last verified or updated."
    )
    # NOTE: This is stored as 'provenance' in the model but Strategy/Runner use '_provenance' key in dicts
    # The disconnect is intentional - Pydantic fields can't start with underscore
    provenance: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Provenance dictionary (validation confidence, source, missing fields, etc.)"
    )
    
    @validator('doi', pre=True)
    def strip_doi_url(cls, v):
        """Strip URL prefix from DOI if present, keeping only the DOI identifier.
        
        Examples:
            'https://doi.org/10.1038/s41578-020-00251-2' -> '10.1038/s41578-020-00251-2'
            'http://dx.doi.org/10.1234/abcd' -> '10.1234/abcd'
            '10.1038/s41578-020-00251-2' -> '10.1038/s41578-020-00251-2' (unchanged)
        """
        if v is None:
            return v
        
        # Strip common DOI URL prefixes
        if isinstance(v, str):
            v = v.replace('https://doi.org/', '')
            v = v.replace('http://doi.org/', '')
            v = v.replace('https://dx.doi.org/', '')
            v = v.replace('http://dx.doi.org/', '')
            v = v.strip()
        
        return v
    
    # Publication info
    publisher: Optional[str] = Field(
        None,
        description="Publisher name",
        max_length=200
    )
    year: Optional[int] = Field(
        None,
        description="Publication year (4 digits, validated range)"
    )
    
    # Book-specific
    editors: List[str] = Field(
        default_factory=list,
        description="List of editor names (books)"
    )
    
    # Chapter-specific
    chapter_number: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Chapter number (book chapters)"
    )
    
    @validator('year')
    def validate_year(cls, v):
        """Validate year is in reasonable range (1400 to current_year + 1)."""
        if v is None:
            return v
        current_year = datetime.now().year
        if not (1400 <= v <= current_year + 1):
            raise ValueError(
                f"Year {v} is not in a reasonable range (1400-{current_year + 1})"
            )
        return v
    
    class Config:
        validate_assignment = True
        extra = 'allow'  # Allow strategy-specific fields


class ContentPage(BaseModel):
    """
    Single page content result from VLM processing.
    
    Represents the extracted content and metadata for one page.
    Used in ProcessingResult.pages list.
    """
    page_number: int = Field(
        ...,
        ge=1,
        description="1-based page number"
    )
    content: str = Field(
        ...,
        description="Extracted text content (markdown format)"
    )
    quality_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Quality score (0.0-1.0)"
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Page-specific metadata (language, tables, diagrams, etc.)"
    )
    
    class Config:
        validate_assignment = True


# ============================================================================
# PROCESSING RESULT MODELS
# ============================================================================

class ProcessingResult(BaseModel):
    """
    Complete strategy output for any document type.
    
    Standardized result format returned by all processing strategies.
    Contains extracted metadata, page content, and quality metrics.
    
    Usage:
        - ResearchArticleStrategy returns this
        - BookChapterStrategy returns this
        - BookStrategy returns this (with toc field populated)
    """
    # Required fields
    metadata: ExtractedMetadata = Field(
        ...,
        description="Extracted bibliographic metadata"
    )
    
    # Content (varies by document type)
    pages: List[ContentPage] = Field(
        default_factory=list,
        description="Per-page content and metadata"
    )
    
    # Book-specific (Phase 1: simple TOC only)
    toc: List[str] = Field(
        default_factory=list,
        description="Table of Contents (chapter titles for books)"
    )
    toc_pages: List[int] = Field(
        default_factory=list,
        description="Page indices containing TOC"
    )
    
    # Quality metrics
    overall_quality: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overall extraction quality score"
    )
    
    # Processing metadata
    processing_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Total processing time in seconds"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Error messages if any"
    )
    
    class Config:
        validate_assignment = True
        extra = 'allow'  # Allow strategy-specific fields


# ============================================================================
# DOCUMENT PROFILE (Main Pipeline State Container)
# ============================================================================

class DocumentProfile(BaseModel):
    """
    Progressive pipeline state container.
    
    Architecture Pattern: Progressive Enrichment
        The DocumentProfile is created after first-pass analysis with basic fields,
        then progressively enriched as it moves through the pipeline:
        
        Stage 1 (First-pass): pdf_path, page_count, file_size, quick identifiers
        Stage 2 (Classification): document_type, confidence, classification_signals
        Stage 3 (Processing): processing_result
    
    Lifecycle Management:
        The pdf_doc (fitz.Document) object is managed by the orchestrator:
        1. Orchestrator creates profile with serializable fields
        2. Orchestrator attaches pdf_doc AFTER validation
        3. Orchestrator uses pdf_doc for classification
        4. Orchestrator closes pdf_doc when done
        5. Profile can be safely serialized (pdf_doc excluded)
    
    Thread-Safety (Phase 3):
        This model is immutable during processing. Multiple threads can
        read the same profile concurrently without issues.
    """
    
    # ========================================================================
    # STAGE 1: First-Pass Analysis (Required Fields)
    # ========================================================================
    
    pdf_path: Path = Field(
        ...,
        description="Path to PDF file"
    )
    
    page_count: int = Field(
        ...,
        gt=0,
        description="Total number of pages in PDF"
    )
    
    file_size: int = Field(
        ...,
        gt=0,
        description="File size in bytes"
    )
    
    pdf_metadata: Dict = Field(
        default_factory=dict,
        description="Raw PDF metadata from PyMuPDF"
    )
    
    # NEW FIELD (Phase 3.6)
    page_1_text: Optional[str] = Field(
        None,
        description="Extracted first-page text (native or OCR). Used for classification and initial metadata."
    )
    
    # Quick identifiers (from first-pass regex extraction with confidence scoring)
    # Confidence-based hybrid approach (V3 pattern):
    #   - High confidence (≥0.85): Use regex value, skip VLM
    #   - Low confidence (<0.85): Fallback to VLM extraction
    
    doi: Optional[str] = Field(
        None,
        description="DOI found in first page (regex extraction)"
    )
    doi_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for DOI (0.0-1.0). Thresholds: 0.98=labeled, 0.95=URL, 0.90=header, 0.85=body"
    )
    
    isbn: Optional[str] = Field(
        None,
        description="ISBN found in first 3 pages (regex extraction)"
    )
    isbn_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for ISBN (0.0-1.0). Thresholds: 0.95=labeled, 0.90=cover, 0.85=early pages"
    )
    
    # It will store the 0-indexed page number of the true title page.
    # Default to 0 to maintain backward compatibility for now.
    title_page_index: int = Field(
        0, 
        description="0-indexed page number of the detected title page"
    )
    
    # ========================================================================
    # STAGE 2: Classification (Populated After classify())
    # ========================================================================
    
    document_type: Optional[DocumentType] = Field(
        None,
        description="Classified document type (type-safe enum)"
    )
    
    classification_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Classification confidence score"
    )
    
    classification_signals: Optional[ClassificationSignals] = Field(
        None,
        description="Detailed classification signal results"
    )
    
    # ========================================================================
    # STAGE 3: Processing (Populated After strategy.process())
    # ========================================================================
    
    processing_result: Optional[ProcessingResult] = Field(
        None,
        description="Complete processing output from strategy"
    )
    
    # ========================================================================
    # RUNTIME ONLY (Excluded from Serialization)
    # ========================================================================
    
    pdf_doc: Any = Field(
        None,
        exclude=True,
        description="Live PyMuPDF document object (runtime only, not serialized)"
    )
    
    pages_to_process: Optional[List[int]] = Field(
        None,
        description="Optional override to process only a specific list of 1-based page numbers. If None, all pages are processed."
    )
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    class Config:
        arbitrary_types_allowed = True  # Allows fitz.Document in pdf_doc
        validate_assignment = True      # Validate on field updates (progressive enrichment)
        
        # JSON serialization config
        json_encoders = {
            Path: str  # Serialize Path as string
        }


# ============================================================================
# ============================================================================

class BatchJob(BaseModel):
    """
    GPU queue job for Phase 3 pipelined processing.
    
    Contains all data needed for a single VLM processing task.
    Created by CPU workers, consumed by GPU worker.
    """
    profile: DocumentProfile = Field(
        ...,
        description="Document profile (without pdf_doc)"
    )
    
    pages: List[int] = Field(
        ...,
        description="Page numbers to process (1-based)"
    )
    
    task_type: VLMTaskType = Field(
        ...,
        description="VLM task type (type-safe enum)"
    )
    
    job_id: str = Field(
        ...,
        description="Unique job identifier for result matching"
    )
    
    class Config:
        validate_assignment = True



class BatchResult(BaseModel):
    """
    Result from GPU queue processing.
    
    Contains VLM output matched to original job.
    Returned by GPU worker, consumed by CPU workers.
    """
    job_id: str = Field(
        ...,
        description="Job identifier (matches BatchJob.job_id)"
    )
    
    result: ProcessingResult = Field(
        ...,
        description="VLM processing result"
    )
    
    processing_time: float = Field(
        ...,
        ge=0.0,
        description="GPU processing time in seconds"
    )
    
    class Config:
        validate_assignment = True


# ============================================================================
# ============================================================================

class JobConfig(BaseModel):
    """
    Configuration payload for a single worker process job.
    
    Passed from Main Process (UI) to Worker Process (Engine).
    Must be completely picklable.
    """
    job_id: str = Field(..., description="Unique identifier for the job (usually file hash)")
    file_path: str = Field(..., description="Absolute path to the PDF file")
    pages_to_process: Optional[List[int]] = Field(None, description="Specific pages to process (optional)")
    
    # Runtime settings
    tesseract_enabled: bool = Field(True, description="Enable Tesseract OCR fallback")
    vlm_enabled: bool = Field(True, description="Enable VLM validation")
    
    # Output config
    output_dir: Optional[str] = Field(None, description="Directory to save artifacts (if any)")
    concurrency_enabled: bool = Field(True, description="Enable concurrent CPU/GPU processing")



# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_profile_stage(profile: DocumentProfile, stage: int) -> bool:
    """
    Validate that profile has required fields for a given pipeline stage.
    
    Args:
        profile: DocumentProfile to validate
        stage: Pipeline stage (1, 2, or 3)
    
    Returns:
        True if profile has all required fields for stage, False otherwise
    
    Example:
        >>> profile = DocumentProfile(pdf_path=Path("test.pdf"), page_count=10, file_size=1000)
        >>> validate_profile_stage(profile, 1)
        True
        >>> validate_profile_stage(profile, 2)
        False  # Missing document_type
    """
    if stage == 1:
        # Stage 1: Basic fields
        return all([
            profile.pdf_path is not None,
            profile.page_count > 0,
            profile.file_size > 0
        ])
    
    elif stage == 2:
        # Stage 2: Classification complete
        return all([
            validate_profile_stage(profile, 1),
            profile.document_type is not None,
            profile.classification_confidence is not None
        ])
    
    elif stage == 3:
        # Stage 3: Processing complete
        return all([
            validate_profile_stage(profile, 2),
            profile.processing_result is not None
        ])
    
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")
