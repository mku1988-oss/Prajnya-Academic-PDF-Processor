#!/usr/bin/env python3
"""
Core data models — what the pipeline passes between its stages.

For biology researchers: these are the "specimen labels" of the system. A PDF
moves through the pipeline inside a `DocumentProfile`, which starts with basic
facts (path, page count) and is progressively enriched: classification adds
the document type, processing adds the extracted content. `ExtractedMetadata`
is the bibliographic record (title, authors, DOI…), validated so a malformed
DOI or an impossible year is rejected at the boundary instead of polluting the
library downstream.

Ported from V5's `src/processing/models.py` with three changes:
- pydantic 2 idioms (`field_validator`, `ConfigDict`) instead of the
  deprecated v1 shims.
- `pdf_doc` holds a `PdfDocument` from the `PdfBackend` seam, never a raw
  `fitz.Document` (the licence rule).
- The Phase-3 batch models (`BatchJob`, `BatchResult`, `JobConfig`) and
  `VLMTaskType` are not carried: nothing on the kept paths references them.

`PageType` (the routing enum) lives in `utils/page_classifier.py`, next to the
classifier that produces it — V5 kept a second copy here, which is exactly the
sort of drift the port removes.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentType(str, Enum):
    """Document classification types with type safety."""
    RESEARCH_ARTICLE = "research_article"
    BOOK_CHAPTER = "book_chapter"
    BOOK = "book"


# ============================================================================
# CLASSIFICATION MODELS
# ============================================================================

class ClassificationSignals(BaseModel):
    """
    Container for the four classification signals.

    Signals:
        1. metadata: PDF metadata analysis (DOI, ISBN, ISSN detection)
        2. page_count: Page count heuristics (article vs chapter vs book)
        3. text_content: Keyword analysis (abstract, chapters, TOC, etc.)
        4. structure: Document structure (first page, TOC pages, etc.)
    """
    model_config = ConfigDict(validate_assignment=True)

    metadata: Dict = Field(default_factory=dict,
                           description="Signal 1: PDF metadata identifiers (DOI, ISBN, ISSN)")
    page_count: Dict = Field(default_factory=dict,
                             description="Signal 2: Page count heuristics")
    text_content: Dict = Field(default_factory=dict,
                               description="Signal 3: Text content analysis (keywords, patterns)")
    structure: Dict = Field(default_factory=dict,
                            description="Signal 4: Document structure (layout, TOC detection)")


# ============================================================================
# METADATA MODELS
# ============================================================================

class ExtractedMetadata(BaseModel):
    """
    Standardized bibliographic metadata for any document type.

    Validation:
        - DOI: must match "10.xxxx/..." if present (URL prefixes are stripped)
        - ISBN: must be 10 or 13 digits if present
        - Year: must fall in 1400 .. current year + 1
    """
    model_config = ConfigDict(validate_assignment=True, extra="allow")

    # Common fields (all document types)
    title: Optional[str] = Field(None, min_length=1, max_length=500,
                                 description="Document title")
    authors: List[str] = Field(default_factory=list, description="List of author names")

    # Identifiers
    doi: Optional[str] = Field(
        None, pattern=r"^10\.\d{4,}/.+$",
        description="Digital Object Identifier, without any URL prefix")
    journal: Optional[str] = Field(None, description="Journal name (articles)")
    isbn: Optional[str] = Field(None, pattern=r"^\d{10}(\d{3})?$",
                                description="ISBN-10 or ISBN-13 (books/chapters)")

    validation_source: Optional[str] = Field(
        None, description="Source of the final metadata, e.g. 'vlm_extraction', "
                          "'zotero_doi_lookup', 'fuzzy_match'")
    validation_confidence: Optional[float] = Field(
        None, description="Confidence (0.0-1.0) in the metadata, based on its source")
    last_updated: Optional[str] = Field(
        None, description="Timestamp of the last verification or update")

    # Provenance tracking. Stored as 'provenance' in the model while dicts use
    # the '_provenance' key — intentional, pydantic fields cannot start with
    # an underscore.
    provenance: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Provenance dictionary (confidence, source, missing fields…)")

    # Publication info
    publisher: Optional[str] = Field(None, max_length=200, description="Publisher name")
    year: Optional[int] = Field(None, description="Publication year")

    # Book- and chapter-specific
    editors: List[str] = Field(default_factory=list, description="Editor names (books)")
    chapter_number: Optional[int] = Field(None, ge=1, le=100,
                                          description="Chapter number (book chapters)")

    @field_validator("doi", mode="before")
    @classmethod
    def strip_doi_url(cls, value):
        """
        Strip URL prefixes from a DOI, keeping only the identifier.

        'https://doi.org/10.1038/xyz' -> '10.1038/xyz'
        """
        if isinstance(value, str):
            for prefix in ("https://doi.org/", "http://doi.org/",
                           "https://dx.doi.org/", "http://dx.doi.org/"):
                value = value.replace(prefix, "")
            value = value.strip()
        return value

    @field_validator("year")
    @classmethod
    def validate_year(cls, value):
        """Reject years outside 1400 .. next year (scanner glitches, OCR noise)."""
        if value is None:
            return value
        current_year = datetime.now().year
        if not (1400 <= value <= current_year + 1):
            raise ValueError(
                f"Year {value} is not in a reasonable range (1400-{current_year + 1})")
        return value


class ContentPage(BaseModel):
    """One page's extracted content — the unit of `ProcessingResult.pages`."""
    model_config = ConfigDict(validate_assignment=True)

    page_number: int = Field(..., ge=1, description="1-based page number")
    content: str = Field(..., description="Extracted text content (markdown)")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0,
                                           description="Quality score (0.0-1.0)")
    metadata: Dict = Field(default_factory=dict,
                           description="Page-specific metadata (tables, figures, method…)")


# ============================================================================
# PROCESSING RESULT MODELS
# ============================================================================

class ProcessingResult(BaseModel):
    """
    Complete strategy output for one document: metadata + pages + quality.

    Every strategy returns one of these, whatever the document type.
    """
    model_config = ConfigDict(validate_assignment=True, extra="allow")

    metadata: ExtractedMetadata = Field(..., description="Extracted bibliographic metadata")
    pages: List[ContentPage] = Field(default_factory=list,
                                     description="Per-page content and metadata")

    # Book-specific
    toc: List[str] = Field(default_factory=list,
                           description="Table of contents (chapter titles, books only)")
    toc_pages: List[int] = Field(default_factory=list,
                                 description="Page indices containing the TOC")
    chapters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-chapter records for books: number, title, authors, "
                    "start_page_printed. In an edited volume every chapter has "
                    "DIFFERENT authors, so a passage retrieved from chapter 3 "
                    "must be cited to its own authors, not to the book's editor. "
                    "`start_page_printed` is the page number as printed in the "
                    "book, NOT an index into the PDF — mapping between the two "
                    "needs the body pages (see plan.md, book reading pipeline)")

    # Quality and bookkeeping
    overall_quality: Optional[float] = Field(None, ge=0.0, le=1.0,
                                             description="Overall extraction quality")
    processing_time: Optional[float] = Field(None, ge=0.0,
                                             description="Total processing time (seconds)")
    errors: List[str] = Field(default_factory=list, description="Error messages, if any")


# ============================================================================
# DOCUMENT PROFILE (Main Pipeline State Container)
# ============================================================================

class DocumentProfile(BaseModel):
    """
    Progressive pipeline state container.

    Created after first-pass analysis with basic facts, then enriched:

        Stage 1 (first pass):     pdf_path, page_count, quick identifiers
        Stage 2 (classification): document_type, confidence, signals
        Stage 3 (processing):     processing_result

    `pdf_doc` holds the live `PdfDocument` (from the `PdfBackend` seam) and is
    excluded from serialization; the orchestrator attaches it after validation
    and closes it when done, so a serialized profile is always safe to store.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # --- Stage 1: first-pass analysis --------------------------------------
    pdf_path: Path = Field(..., description="Path to the PDF file")
    page_count: int = Field(..., gt=0, description="Total number of pages")
    file_size: int = Field(..., gt=0, description="File size in bytes")
    pdf_metadata: Dict = Field(default_factory=dict,
                               description="Raw PDF metadata from the backend")
    page_1_text: Optional[str] = Field(
        None, description="First-page text (native or OCR), for classification "
                          "and initial metadata")

    # Quick identifiers (first-pass regex extraction with confidence scoring;
    # >= 0.85 uses the regex value directly, below that the VLM decides).
    doi: Optional[str] = Field(None, description="DOI found on the first page (regex)")
    doi_confidence: float = Field(0.0, ge=0.0, le=1.0,
                                  description="DOI confidence: 0.98 labeled, 0.95 URL, "
                                              "0.90 header, 0.85 body")
    isbn: Optional[str] = Field(None, description="ISBN found in the front matter (regex)")
    isbn_confidence: float = Field(0.0, ge=0.0, le=1.0,
                                   description="ISBN confidence: 0.95 labeled, 0.90 cover, "
                                               "0.85 early pages")
    issn: Optional[str] = Field(None, description="ISSN found in the front matter, "
                                                  "normalised as NNNN-NNNX. NOTE: a book "
                                                  "SERIES carries an ISSN too — it is not "
                                                  "evidence of a journal article")
    identifier_pages: Dict[str, int] = Field(
        default_factory=dict,
        description="1-based page each identifier was found on, e.g. {'isbn': 5}. "
                    "Books put identifiers on the copyright page, not page 1")
    front_matter_pages_scanned: int = Field(
        0, ge=0, description="How many leading pages the identifier scan read")
    title_page_index: int = Field(0, description="0-indexed page number of the "
                                                 "detected title page")

    # --- Stage 2: classification -------------------------------------------
    document_type: Optional[DocumentType] = Field(None, description="Classified type")
    classification_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    classification_signals: Optional[ClassificationSignals] = Field(None)

    # --- Stage 3: processing -----------------------------------------------
    processing_result: Optional[ProcessingResult] = Field(None)

    # --- Runtime only (excluded from serialization) ------------------------
    pdf_doc: Any = Field(None, exclude=True,
                         description="Live PdfDocument (PdfBackend seam), runtime only")
    pages_to_process: Optional[List[int]] = Field(
        None, description="Optional override: process only these 1-based pages")
