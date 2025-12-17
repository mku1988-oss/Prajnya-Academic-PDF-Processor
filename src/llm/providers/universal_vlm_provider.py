#!/usr/bin/env python3
"""
Universal VLM Provider - Model-Agnostic PDF Processing
ENHANCED WITH OLMOCR RESEARCH INTEGRATION & ROBUST ENGINEERING PRACTICES

ARCHITECTURE:
- Model-agnostic core processing pipeline
- External model profiles for optimization (Validated with Pydantic)
- Task-specific prompt libraries (Validated with Pydantic)
- Single source of truth configuration
- Official OlmOCR integration with 77.4 benchmark score
- Document-level processing vs page-by-page
- Structured YAML + markdown output support
- Testable via Dependency Injection
- Resilient with Model Availability Checks

Author: Claude 4 (Universal Provider Architecture Enhanced) & Gemini
Date: September 29, 2025
Version: 2.2.0 - Enhanced with Retry Logic & Intelligent Aggregation
"""

import logging
import time
import yaml
import json
import requests
import re
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, HttpUrl, ValidationError
import base64
import io
import pypdfium2 as pdfium
from PIL import Image
import os

# Gemini SDK import (conditional - only imported if API key present)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- REVISED: Pydantic Models for Full YAML Compatibility ---

# A flexible model for parameters: defines common ones, allows any others.
# Three-Tier Parameter Architecture
class APIParameters(BaseModel):
    """Tier 1: Parameters sent to the API endpoint (llama.cpp /v1/chat/completions).
    
    These parameters can override server defaults on a per-request basis.
    Only include parameters that are actually accepted by the API endpoint.
    """
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate (default: 100)")
    repeat_penalty: Optional[float] = Field(None, ge=0.0, description="Penalty for repeating recent tokens")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Penalty for tokens appearing anywhere in the text")
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Penalty based on token frequency")
    # Add other API-accepted parameters as needed (validated against llama.cpp docs)
    
    class Config:
        extra = 'forbid'  # Strict validation - only known API params allowed

class ServerConfig(BaseModel):
    """Tier 2: Server configuration (documentation only, not sent in API requests).
    
    These settings are configured in llama-swap run_swap_Config.yaml when starting
    the llama-server. They are documented here for reference and understanding.
    """
    threads: Optional[int] = Field(None, description="Number of CPU threads")
    ctx_size: Optional[int] = Field(None, description="Context window size")
    flash_attn: Optional[bool] = Field(None, description="Flash attention enabled")
    n_gpu_layers: Optional[int] = Field(None, description="GPU layers to offload")
    cache_type_k: Optional[str] = Field(None, description="KV cache type for keys")
    cache_type_v: Optional[str] = Field(None, description="KV cache type for values")
    default_temp: Optional[float] = Field(None, description="Server default temperature")
    default_top_p: Optional[float] = Field(None, description="Server default top_p")
    default_repeat_penalty: Optional[float] = Field(None, description="Server default repeat_penalty")
    default_min_p: Optional[float] = Field(None, description="Server default min_p")
    
    class Config:
        extra = 'allow'  # Allow other server-specific settings

class AppConfig(BaseModel):
    """Tier 3: Application-level configuration (Shastrasaara pipeline settings).
    
    These control YOUR application's behavior, not the model server.
    Examples: image preprocessing, batch sizes, OCR-specific settings.
    """
    target_dpi: Optional[int] = Field(None, description="Target DPI for image rendering")
    preferred_image_format: Optional[str] = Field(None, description="Image format (png, jpeg)")
    target_anchor_text_len: Optional[int] = Field(None, description="OCR anchor text length")
    batch_size: Optional[int] = Field(None, description="Processing batch size")
    target_image_size: Optional[int] = Field(None, description="Target square resolution for the vision model (e.g., 1288)")
    
    class Config:
        extra = 'allow'  # Allow app-specific extensions

# Legacy support - maps to api_parameters for backward compatibility
class ModelParameters(BaseModel):
    """DEPRECATED: Use api_parameters, server_config, and app_config instead.
    
    Kept for backward compatibility with existing code.
    """
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    repeat_penalty: Optional[float] = Field(None, ge=0.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    
    class Config:
        extra = 'allow'

class ModelProfile(BaseModel):
    """Three-Tier Model Profile with backward compatibility.
    
    Supports both new three-tier architecture and legacy single 'parameters' field.
    """
    model_name: str = Field(..., description="Model identifier (must match llama-swap config)")
    model_type: str = Field(..., description="Model category (e.g., 'current_vlm', 'official_specialized_ocr')")
    benchmark_score: Optional[float] = Field(None, description="Model performance score")
    status: Optional[str] = Field(None, description="Model status (e.g., 'production_ready', 'working_baseline')")
    
    # Three-Tier Architecture (NEW - Preferred)
    api_parameters: Optional[APIParameters] = Field(None, description="Tier 1: API request parameters")
    server_config: Optional[ServerConfig] = Field(None, description="Tier 2: Server configuration (documentation)")
    app_config: Optional[AppConfig] = Field(None, description="Tier 3: Application settings")
    
    # Legacy Support (DEPRECATED - for backward compatibility)
    parameters: Optional[ModelParameters] = Field(None, description="DEPRECATED: Use api_parameters instead")
    
    # Additional metadata
    performance: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    
    def get_api_params(self) -> Dict[str, Any]:
        """Get API parameters with backward compatibility.
        
        Prefers api_parameters (new) over parameters (legacy).
        Returns only non-None values.
        """
        if self.api_parameters:
            return self.api_parameters.dict(exclude_none=True)
        elif self.parameters:
            # Legacy fallback
            return self.parameters.dict(exclude_none=True)
        return {}
    
    class Config:
        # Allow extra fields for additional metadata like biology_optimizations, processing, etc.
        extra = 'allow'

class TaskDefinition(BaseModel):
    """Expanded structure for a task definition YAML to match your files."""
    task_name: str
    task_type: Optional[str] = None
    description: Optional[str] = None
    prompt: str = "Extract text from this document page accurately."
    output_format: Dict[str, Any] = {"type": "plain_text"}
    quality_targets: Dict[str, Any] = {}
    
    class Config:
        # Allow extra fields for task_metadata and other extensions
        extra = 'allow'

# --- REVISED: Models for universal_config.yaml ---

class GlobalSettings(BaseModel):
    """Global processing settings."""
    max_retries: int = 3
    enable_logging: bool = True
    log_level: str = "INFO"
    
    class Config:
        extra = 'allow'

class PerformanceSettings(BaseModel):
    """Performance configuration settings."""
    enable_caching: bool = False
    batch_processing: bool = False
    parallel_pages: bool = False
    
    class Config:
        extra = 'allow'

class QualitySettings(BaseModel):
    """Quality control settings."""
    enable_validation: bool = True
    minimum_confidence: float = 0.5
    enable_fallback: bool = True
    
    class Config:
        extra = 'allow'

class DevelopmentSettings(BaseModel):
    """Development and debugging settings."""
    debug_mode: bool = False
    save_intermediate: bool = False
    timing_enabled: bool = True
    
    class Config:
        extra = 'allow'

class Metadata(BaseModel):
    """Nested metadata model for the universal config."""
    provider_type: str
    version: str
    created: Optional[str] = None
    updated: Optional[str] = None
    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    development: DevelopmentSettings = Field(default_factory=DevelopmentSettings)

class UniversalConfig(BaseModel):
    """Universal configuration loaded from YAML, including all metadata."""
    api_endpoint: HttpUrl = "http://localhost:8080/v1/chat/completions"
    timeout_seconds: int = 900  # Increased from 300s to 15m for complex Two-Shot tasks
    default_model: str = "olmocr-7b"
    fallback_model: str = "numarkdown-8b"
    image_padding: int = 200
    conversion_dpi: int = 300
    image_format: str = "png"
    metadata: Optional[Metadata] = None  # Include the whole metadata block


class UniversalVLMProvider:
    """
    Universal VLM Provider - Works with any vision-language model
    ENHANCED with official OlmOCR integration
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "basic_ocr",
        session: Optional[requests.Session] = None,
        config: Optional[UniversalConfig] = None,
        model_profile: Optional[ModelProfile] = None,
        task_definition: Optional[TaskDefinition] = None,
        skip_validation: bool = False,
        api_endpoint: Optional[str] = None
    ):
        """
        Initialize with model and task selection.

        Can be initialized by loading its own configs OR by injecting pre-loaded
        Pydantic config objects for use in a larger pipeline.

        Args:
            model_name (str): The name of the model to use.
            task_type (str, optional): The task definition to use.
            session (Optional[requests.Session], optional): A requests.Session for API calls.
            config (Optional[UniversalConfig], optional): Pre-loaded universal config. If provided, skips file loading.
            model_profile (Optional[ModelProfile], optional): Pre-loaded model profile. If provided, skips file loading.
            task_definition (Optional[TaskDefinition], optional): Pre-loaded task definition. If provided, skips file loading.
            skip_validation (bool, optional): If True, skips the startup server health check.
            api_endpoint (Optional[str], optional): Override the API endpoint in the config.
        """
        self.model_name = model_name
        self.task_type = task_type
        
        self.logger = self._setup_logging()
        self.session = session or requests.Session()
        
        # Load configurations: Use injected objects if provided, otherwise load from files.
        self.config = config or self._load_universal_config()
        
        # Apply API endpoint override if provided
        if api_endpoint:
            self.config.api_endpoint = api_endpoint
            self.logger.info(f"Overriding API endpoint: {api_endpoint}")
            
        self.model_profile = model_profile or self._load_model_profile(model_name)
        self.task_definition = task_definition or self._load_task_definition(task_type)

        # Detect API type and initialize appropriate client
        self.api_type = self._detect_api_type()
        self.gemini_client = None  # Initialize to None, set only for Gemini models

        if self.api_type == "gemini":
            self._initialize_gemini_client()

        # Check if the model is actually available before finishing initialization
        # OPTIMIZATION: Allow skipping this for temporary/internal providers spawned per-page
        # Skip check for Gemini models (they don't have a local server to check)
        if not skip_validation and self.api_type != "gemini":
            self._check_model_availability()
        else:
            reason = "Gemini cloud API" if self.api_type == "gemini" else "skip_validation=True"
            self.logger.info(f"Skipping model availability check ({reason})")
        
        self.logger.info(f"Universal Provider initialized and validated: {self.model_profile.model_name} -> {self.task_definition.task_name}")
    
    def process_document(self, pdf_path: str, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Main processing method with ENHANCED aggregation and retry logic.

        Orchestrates the pipeline:
        1. Converts PDF pages to images.
        2. Processes each image through the selected VLM with automatic retries.
        3. Intelligently aggregates multi-page results.
        
        Version: 2.2.0 - Enhanced with retry logic and intelligent aggregation
        """
        try:
            start_time = time.time()
            self.logger.info(f"Processing document: {pdf_path}, pages: {pages or 'all'}")

            # 1. Convert PDF to a list of images
            images = self._convert_pdf_to_images(pdf_path, pages)
            
            if not images:
                raise ValueError("PDF conversion resulted in no images to process.")

            # 2. Process each image and collect results with retry logic
            all_page_results = []
            page_indices = pages if pages else list(range(1, len(images) + 1))

            for i, page_image in enumerate(images):
                page_num = page_indices[i]
                self.logger.info(f"Processing page {page_num}...")
                
                # --- INTEGRATION OF RETRY LOGIC ---
                # Call the new wrapper method instead of the original one
                raw_output = self._process_single_page_with_retry(page_image)
                
                structured_output = self._post_process_output(raw_output)
                
                # Calculate quality score for this page
                page_quality = self._calculate_page_quality_score(
                    structured_output.get('content', ''),
                    structured_output.get('metadata', {})
                )
                
                all_page_results.append({
                    "page_number": page_num,
                    **structured_output,
                    "quality_score": page_quality  # Add quality score to page results
                })

            # --- INTEGRATION OF ENHANCED AGGREGATION ---
            # Replace simple aggregation with a call to the new helper method
            aggregated_results = self._aggregate_multi_page_results(all_page_results)

            # 3. Assemble the final, standardized result
            processing_time = time.time() - start_time
            self.logger.info(f"Document processing finished in {processing_time:.2f} seconds.")
            
            result = {
                'success': True,
                'model_used': self.model_profile.model_name,
                'task_type': self.task_definition.task_name,
                'processing_time': round(processing_time, 2),
                'page_count': len(images),
                'results_by_page': all_page_results,  # Keep per-page results for detailed inspection
                'structured_content': aggregated_results['aggregated_content'],
                'metadata': aggregated_results['aggregated_metadata'],
                'quality_score': aggregated_results.get('quality_score'),  # Document-level quality
                'universal_provider_version': '2.2.0'  # Version bump for new features
            }
            return result

        except Exception as e:
            self.logger.error(f"Processing failed for document '{pdf_path}': {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def process_page(self, page_image: Image.Image, base_text: Optional[str] = None, task_definition: Optional[TaskDefinition] = None) -> Dict[str, Any]:
        """
        Public API method for processing a single, pre-rendered page image.
        This method can be enhanced with document anchoring via the 'base_text' parameter.
        Accepts optional 'task_definition' to override the default task.
        """
        try:
            self.logger.info("Processing single page via public process_page method...")
            
            # 1. Call the robust internal method, passing the optional anchoring text and task
            raw_output = self._process_single_page_with_retry(page_image, base_text=base_text, task_definition=task_definition)
            
            # 2. Post-process the raw output into structured data
            structured_output = self._post_process_output(raw_output, task_definition=task_definition)
            
            # 3. Calculate the quality score for the page
            page_quality = self._calculate_page_quality_score(
                structured_output.get('content', ''),
                structured_output.get('metadata', {})
            )
            
            # 4. Assemble the final, standardized result for one page
            return {
                "success": True,
                **structured_output,
                "quality_score": page_quality
            }

        except Exception as e:
            self.logger.error(f"process_page failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    # --- HELPER METHOD: PAGE QUALITY SCORING ---
    def _calculate_page_quality_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """
        Calculates a quality score (0.0-1.0) for a processed page.
        This version is now TASK-AWARE and checks for the correct metadata fields.
        """
        scores = {}
        
        # 1. Content Length Score (unchanged)
        scores['content_length'] = min(1.0, len(content) / 2500.0)

        # 2. Structural Marker Score (unchanged)
        markers = ['#', '##', '###', '* ', '- ', '**']
        found_markers = sum(1 for marker in markers if marker in content)
        scores['structural_markers'] = min(1.0, found_markers / 4.0)
        
        # 3. Text Quality Indicators (unchanged)
        indicators = [
            len(re.findall(r'\b\w{10,}\b', content)) < len(content) * 0.1, # Few overly long words
            content.count(' ') / (len(content) + 1) > 0.1, # Reasonable spacing
            re.search(r'[a-zA-Z]', content) is not None, # Contains letters
            len(re.findall(r'[.,;:]', content)) > 5, # Contains punctuation
        ]
        scores['text_quality'] = sum(indicators) / len(indicators)

        # --- NEW TASK-AWARE METADATA SCORING ---
        if self.task_type in ['title_page_metadata_extraction', 'structured_analysis_title_page']:
            # Check for the specific fields from the title page task
            expected_fields = ['title', 'authors', 'doi']
            present_fields = sum(1 for field in expected_fields if metadata.get(field))
            
            validity_checks = [
                isinstance(metadata.get('title'), str) and len(metadata.get('title', '')) > 5,
                isinstance(metadata.get('authors'), list) and len(metadata.get('authors', [])) > 0,
                isinstance(metadata.get('doi'), str) and '10.' in metadata.get('doi', '')
            ]
            valid_fields = sum(validity_checks)

            scores['metadata_presence'] = present_fields / len(expected_fields)
            scores['metadata_validity'] = valid_fields / len(expected_fields)
            
            # Log breakdown for title page
            self.logger.info(f"  Metadata Fields (Title): {present_fields}/{len(expected_fields)} present")
            self.logger.info(f"  Metadata Validity (Title): {valid_fields}/{len(expected_fields)} checks passed")

        else: # Default to body page scoring
            expected_fields = ['primary_language', 'is_rotation_valid', 'rotation_correction', 'is_table', 'is_diagram']
            present_fields = sum(1 for field in expected_fields if metadata.get(field) is not None)
            
            validity_checks = [
                isinstance(metadata.get('primary_language'), str),
                isinstance(metadata.get('is_rotation_valid'), bool),
                isinstance(metadata.get('rotation_correction'), int),
                isinstance(metadata.get('is_table'), bool),
                isinstance(metadata.get('is_diagram'), bool),
            ]
            valid_fields = sum(validity_checks)

            scores['metadata_presence'] = present_fields / len(expected_fields)
            scores['metadata_validity'] = valid_fields / len(expected_fields)

            # Log breakdown for body page
            self.logger.info(f"  Metadata Fields (Body): {present_fields}/{len(expected_fields)} present")
            self.logger.info(f"  Metadata Validity (Body): {valid_fields}/{len(expected_fields)} checks passed")
        # --- END OF NEW LOGIC ---

        # Final weighted score calculation (unchanged)
        weights = {'content_length': 0.3, 'structural_markers': 0.1, 'text_quality': 0.2, 'metadata_presence': 0.2, 'metadata_validity': 0.2}
        final_score = sum(scores[key] * weights[key] for key in scores)
        
        self.logger.info(f"  → Final Quality Score: {final_score:.2f} ({final_score:.1%})")
        return final_score

        self.logger.info(f"  → Final Quality Score: {final_score:.2f} ({final_score:.1%})")
        return final_score

    # --- NEW HELPER METHOD 1: RETRY LOGIC ---

    def _process_single_page_with_retry(
        self, 
        page_image: Image.Image,
        base_text: Optional[str] = None,
        max_retries: Optional[int] = None,
        task_definition: Optional[TaskDefinition] = None
    ) -> str:
        """
        Wrapper for _process_single_page that adds resilient retry logic
        with exponential backoff for transient network errors.
        
        Args:
            page_image: PIL Image to process
            max_retries: Maximum retry attempts (uses config if not specified)
        
        Returns:
            Processed text output from VLM
            
        Raises:
            requests.RequestException: After all retries exhausted
            ValueError: For non-retryable errors (malformed responses)
        """
        # Safe config access with fallback
        if max_retries is None:
            max_retries = 3  # Default
            if self.config.metadata and hasattr(self.config.metadata, 'global_settings'):
                max_retries = getattr(self.config.metadata.global_settings, 'max_retries', 3)
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Attempt to process the page
                return self._process_single_page(page_image, base_text=base_text, task_definition=task_definition)
                
            except requests.Timeout as e:
                last_exception = e
                self.logger.warning(
                    f"API timeout on attempt {attempt + 1} of {max_retries}. Error: {e}"
                )
                
            except requests.ConnectionError as e:
                last_exception = e
                self.logger.warning(
                    f"Connection error on attempt {attempt + 1} of {max_retries}. Error: {e}"
                )
                
            except requests.HTTPError as e:
                # Don't retry 4xx errors (client errors - bad request, auth, etc.)
                if e.response and 400 <= e.response.status_code < 500:
                    self.logger.error(f"Client error (non-retryable): {e}")
                    raise
                # Retry 5xx errors (server errors)
                last_exception = e
                self.logger.warning(
                    f"Server error on attempt {attempt + 1} of {max_retries}. Status: {e.response.status_code if e.response else 'N/A'}"
                )
                
            except (ValueError, KeyError) as e:
                # Malformed response - likely not transient, but give one retry
                if attempt == 0:
                    last_exception = e
                    self.logger.warning(f"Response parsing error, will retry once: {e}")
                else:
                    self.logger.error(f"Response parsing error persists: {e}")
                    raise
            
            # If we get here, the attempt failed and we should retry
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s, 8s...
                backoff_time = 2 ** attempt
                self.logger.info(f"Waiting {backoff_time} seconds before retry {attempt + 2}...")
                time.sleep(backoff_time)
            else:
                # Final attempt failed
                self.logger.error(
                    f"All {max_retries} retry attempts failed. Last error: {last_exception}"
                )
                raise last_exception
        
        # Safeguard (should never reach here)
        raise RuntimeError("Retry loop completed unexpectedly without success or exception.")

    # --- NEW HELPER METHOD 2: ENHANCED AGGREGATION ---

    def _aggregate_multi_page_results(self, all_page_results: List[Dict]) -> Dict[str, Any]:
        """
        Intelligently merges multi-page content and metadata for document-level understanding.
        
        Strategy:
        - Metadata: First-occurrence priority (first page's metadata takes precedence)
        - Content: Page-by-page concatenation with separators
        - Quality: Average across all pages with valid scores
        
        Args:
            all_page_results: List of per-page results with content and metadata
            
        Returns:
            Dict with aggregated_content, aggregated_metadata, and quality_score
        """
        if not all_page_results:
            return {
                "aggregated_content": "",
                "aggregated_metadata": {},
                "quality_score": None
            }

        # 1. Aggregate content with page separators
        aggregated_content = "\n\n---\n\n".join(
            str(p.get("content", "")) for p in all_page_results
        )

        # 2. Intelligently merge metadata (first-come, first-served)
        aggregated_metadata = {}
        metadata_sources = []  # Track which pages contributed metadata
        
        for page_result in all_page_results:
            page_num = page_result.get("page_number", "unknown")
            page_metadata = page_result.get("metadata")
            
            if isinstance(page_metadata, dict):
                for key, value in page_metadata.items():
                    # Add key only if it hasn't been seen before
                    if key not in aggregated_metadata:
                        aggregated_metadata[key] = value
                        metadata_sources.append(f"{key} from page {page_num}")
        
        # Log metadata merge summary
        if metadata_sources:
            self.logger.debug(f"Merged metadata from {len(metadata_sources)} unique fields")
        
        # 3. Calculate document-level quality score (if available)
        quality_scores = []
        for page_result in all_page_results:
            if 'quality_score' in page_result and page_result['quality_score'] is not None:
                try:
                    quality_scores.append(float(page_result['quality_score']))
                except (ValueError, TypeError):
                    continue
        
        avg_quality = None
        if quality_scores:
            avg_quality = round(sum(quality_scores) / len(quality_scores), 3)
            self.logger.info(f"Document quality score: {avg_quality} (avg of {len(quality_scores)} pages)")
        
        # 4. Analyze document structure (optional enhancement)
        document_structure = {
            'total_pages': len(all_page_results),
            'pages_with_metadata': sum(1 for p in all_page_results if p.get('metadata')),
            'pages_with_quality': len(quality_scores),
            'content_length': len(aggregated_content)
        }
        
        return {
            "aggregated_content": aggregated_content,
            "aggregated_metadata": aggregated_metadata,
            "quality_score": avg_quality,
            "document_structure": document_structure  # Additional analysis
        }

    # --- EXISTING HELPER METHODS ---

    def _convert_pdf_to_images(self, pdf_path: str, pages: Optional[List[int]]) -> List[Image.Image]:
        """
        Converts specified pages of a PDF into images with resolution based on page number.
        
        RESOLUTION STRATEGY (Based on testing October 1-4, 2025):
        - Page 1 (Title Page): 1288x1288 pixels
          Reason: Captures footer region containing DOI for bibliographic metadata.
          Testing showed DOI is cropped at 1024 resolution but visible at 1288.
        
        - Other Pages: 1024x1024 pixels (from config, typically 1024)
          Reason: Optimal balance of speed vs quality for body text extraction.
          Benchmark: 76.2% quality at 437s processing time.
        
        Official OlmOCR-7B-0825 requirement: https://huggingface.co/allenai/olmOCR-7B-0825
        "This model expects as input a single document image, rendered such that
        the longest dimension is 1288 pixels."
        
        Note: Page 1 uses higher resolution specifically for DOI extraction.
              Based on multi-configuration testing documented in memory.
        """
        # Get default resolution from config (typically 1024 for optimized processing)
        DEFAULT_SIZE = self.model_profile.app_config.target_image_size if (self.model_profile.app_config and self.model_profile.app_config.target_image_size) else 1024
        images = []
        
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            # If no pages are specified, process all pages.
            # pypdfium2 uses 0-based indexing, so we adjust.
            page_indices = [p - 1 for p in pages] if pages else range(len(pdf))
            
            for i in page_indices:
                if i >= len(pdf):
                    self.logger.warning(f"Page number {i+1} is out of range. Skipping.")
                    continue
                
                # CONDITIONAL RESOLUTION: Page 1 needs 1288 for DOI, others use optimized size
                # Based on testing: 1024 crops footer (DOI missing), 1288 captures it fully
                page_num = i + 1  # Convert 0-indexed to 1-indexed page number
                if page_num == 1:
                    TARGET_SIZE = 1288  # Higher resolution for title page footer (DOI)
                    self.logger.info(f"Using 1288x1288 resolution for Page 1 (captures DOI in footer)")
                else:
                    TARGET_SIZE = DEFAULT_SIZE  # Use optimized resolution from config
                
                # Render page at high DPI for quality
                image = pdf.get_page(i).render(scale=self.config.conversion_dpi / 72).to_pil()
                
                # Resize maintaining aspect ratio (longest edge = TARGET_SIZE)
                image.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
                
                # Create white TARGET_SIZE x TARGET_SIZE canvas
                padded_image = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (255, 255, 255))
                
                # Center paste the resized image
                paste_x = (TARGET_SIZE - image.width) // 2
                paste_y = (TARGET_SIZE - image.height) // 2
                padded_image.paste(image, (paste_x, paste_y))
                
                images.append(padded_image)
            
            self.logger.info(
                f"Successfully converted {len(images)} pages to target resolutions "
                f"(Page 1: 1288x1288 for DOI, Others: {DEFAULT_SIZE}x{DEFAULT_SIZE})"
            )
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to convert PDF '{pdf_path}': {e}", exc_info=True)
            raise

    def _load_grammar_content(self, grammar_filename: str) -> Optional[str]:
        """Loads the content of a GBNF grammar file."""
        if not grammar_filename:
            return None
        
        grammar_path = Path(__file__).parent / 'config' / 'grammars' / grammar_filename
        if not grammar_path.exists():
            self.logger.warning(f"Grammar file specified but not found: {grammar_path}")
            return None
        
        try:
            with open(grammar_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to read grammar file {grammar_path}: {e}", exc_info=True)
            return None

    def _process_single_page(self, page_image: Image.Image, base_text: Optional[str] = None, task_definition: Optional[TaskDefinition] = None) -> str:
        """
        Handles processing a single image via the /v1/chat/completions endpoint.
        This endpoint is more robust for multimodal input.

        Routes to Gemini API if api_type is "gemini".
        """
        # Determine effective task definition
        effective_task = task_definition or self.task_definition

        # Route to Gemini processing if using Gemini API
        if self.api_type == "gemini":
            return self._process_single_page_gemini(page_image, base_text, task_definition=effective_task)

        # Original llama.cpp /v1/chat/completions processing
        base64_image = self._encode_image_to_base64(page_image)
        headers = {"Content-Type": "application/json"}
        api_params = self.model_profile.get_api_params()

        prompt_text = effective_task.prompt
        # This check is for the document anchoring feature in the test file
        if base_text and '{base_text}' in prompt_text:
            prompt_text = prompt_text.format(base_text=base_text)
            self.logger.info(f"Document anchoring enabled: {len(base_text)} chars base_text")

        # --- THIS IS THE CORRECT LOGIC FOR /v1/chat/completions ---
        payload = {
            "model": self.model_profile.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{self.config.image_format};base64,{base64_image}"},
                        },
                    ],
                }
            ],
            **api_params
        }

        # === RE-ENABLED: response_format DOES WORK (confirmed via OlmOCR_run_hybrid_v2.py) ===
        # Add response_format for JSON tasks to force valid JSON output
        # This eliminates issues with:
        # - Python boolean syntax (True/False instead of true/false)
        # - Markdown code blocks (```json ... ```)
        # - Unescaped newlines in strings
        if (effective_task.output_format.get("type") == "json" and 
                "schema" in effective_task.output_format):
            payload['response_format'] = {
                "type": "json_object",
                "schema": effective_task.output_format["schema"]
            }
            self.logger.info("Requesting schema-constrained JSON via response_format")
        # === END RE-ENABLED SECTION ===

        # CRITICAL: llama.cpp build 6660+ rejects requests with BOTH response_format AND grammar.
        # This causes 400 Bad Request errors. The response_format JSON schema is sufficient
        # for structured output, and Tier 2 robust extraction handles any parsing issues.
        # Load and add GBNF grammar to the payload if specified in the task definition.
        # This overrides any server-side grammar and allows for per-task constraints.
        
        grammar_filename = getattr(effective_task, 'grammar_file', None)
        if grammar_filename:
            grammar_content = self._load_grammar_content(grammar_filename)
            if grammar_content:
                self.logger.info(f"Applying per-request GBNF grammar: {grammar_filename}")
                payload['grammar'] = grammar_content
        
        # Debug logging
        debug_payload = {k: v for k, v in payload.items() if k != 'messages' and k != 'grammar'}
        if 'grammar' in payload:
            debug_payload['grammar'] = f"<{len(payload['grammar'])} chars>"
        debug_payload['messages'] = "..." # Omit long message content from log
        self.logger.info(f"Sending API request to /v1/chat/completions with payload: {json.dumps(debug_payload, indent=2)}")

        try:
            # This line MUST use the configured endpoint.
            start_req_time = time.time()
            response = self.session.post(
                str(self.config.api_endpoint),
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds
            )
            req_duration = time.time() - start_req_time
            
            response.raise_for_status()
            response_json = response.json()
            
            raw_output = response_json['choices'][0]['message']['content']
            
            # === DEBUG LOGGING START ===
            self.logger.info("=" * 60)
            self.logger.info("RAW SERVER RESPONSE DEBUG")
            self.logger.info("=" * 60)
            self.logger.info(f"Request Duration: {req_duration:.2f}s (Timeout limit: {self.config.timeout_seconds}s)")
            self.logger.info(f"Payload Context Size: {len(base_text) if base_text else 0} chars (base_text)")
            self.logger.info(f"Response Status: {response.status_code}")
            self.logger.info(f"Response Headers: {dict(response.headers)}")
            self.logger.info(f"Raw Output Length: {len(raw_output)} characters")
            self.logger.info(f"Raw Output (first 500 chars): {raw_output[:500]}")
            self.logger.info(f"Raw Output Type: {type(raw_output)}")
            self.logger.info(f"Is Empty: {len(raw_output.strip()) == 0}")
            self.logger.info("=" * 60)
            # === DEBUG LOGGING END ===
            
            return raw_output.strip()
            
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}", exc_info=True)
            raise
        except (KeyError, IndexError) as e:
            self.logger.error(f"Failed to parse API response: {response.text}. Error: {e}")
            raise ValueError("Invalid API response format from /v1/chat/completions endpoint") from e

    def _post_process_output(self, raw_output: str, task_definition: Optional[TaskDefinition] = None) -> Dict[str, Any]:
        """
        Parses the raw string output from a model into a structured dictionary.
        Uses task_definition if provided, else self.task_definition.
        """
        effective_task = task_definition or self.task_definition
        
        # Check if this task expects plain text output (e.g., surgical descriptions)
        output_type = effective_task.output_format.get("type", "json")
        
        if output_type in ["plain_text", "text", "markdown"]:
            # Surgical tasks (figure/table descriptions) return plain text directly
            self.logger.debug(f"Task '{self.task_type}' expects plain text/markdown - skipping JSON parsing")
            return {
                "metadata": {},
                "content": raw_output,
                "raw_output": raw_output
            }
        
        # For JSON tasks, proceed with parsing
        try:
            # Clean up any markdown code block wrappers
            cleaned_output = raw_output.strip()
            
            # Handle markdown code blocks wrapping JSON
            if cleaned_output.startswith("```json") and cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[7:-3].strip()
            elif cleaned_output.startswith("```") and cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[3:-3].strip()
            
            # Sanitize Python syntax before JSON parsing
            cleaned_output = self._sanitize_json_output(cleaned_output)
            
            # Parse JSON - GBNF grammar should ensure valid JSON (with sanitization fallback)
            parsed_json = json.loads(cleaned_output)
            
            # Extract natural_text as content
            content = parsed_json.get('natural_text', '')
            
            # Extract metadata (all fields except natural_text)
            metadata = {k: v for k, v in parsed_json.items() if k != 'natural_text'}
            
            self.logger.info(f"Successfully parsed JSON output with {len(metadata)} metadata fields.")
            return {
                "metadata": metadata,
                "content": content,
                "raw_output": raw_output
            }
            
        except json.JSONDecodeError as e:
            # JSON parsing failed even after sanitization
            # Only log as ERROR if this was expected to be JSON
            if output_type == "json":
                self.logger.error(
                    f"JSON parsing failed for JSON task '{self.task_type}': {e}. "
                    f"First 200 chars: {cleaned_output[:200]}"
                )
            else:
                # Just debug log if format was ambiguous
                self.logger.debug(
                    f"JSON parsing attempted but failed for task '{self.task_type}' (output_type={output_type}): {e}"
                )
            
            # Emergency fallback to plain text
            return {
                "metadata": {},
                "content": raw_output,
                "raw_output": raw_output
            }
    
    def _sanitize_json_output(self, json_text: str) -> str:
        """
        Sanitizes common JSON syntax errors from VLM output.
        
        The VLM sometimes returns Python syntax instead of JSON syntax:
        - Python: True, False, None
        - JSON:   true, false, null
        
        This method fixes these issues before JSON parsing.
        
        Args:
            json_text: Raw JSON text from VLM (may contain Python syntax)
            
        Returns:
            Sanitized JSON text with proper JSON syntax
        """
        # Replace Python booleans with JSON booleans (word boundaries to avoid partial matches)
        json_text = re.sub(r'\bTrue\b', 'true', json_text)
        json_text = re.sub(r'\bFalse\b', 'false', json_text)
        
        # Replace Python None with JSON null
        json_text = re.sub(r'\bNone\b', 'null', json_text)
        
        return json_text
            
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encodes a PIL Image to a base64 string."""
        buffered = io.BytesIO()
        image_format = self.config.image_format.upper()
        if image_format == "JPG":
            image_format = "JPEG" # Common PIL alias
        image.save(buffered, format=image_format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _load_universal_config(self) -> UniversalConfig:
        """Load and validate universal configuration from YAML file."""
        config_path = Path(__file__).parent / 'config' / 'universal_config.yaml'
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                return UniversalConfig(**config_data)
        except (FileNotFoundError, ValidationError) as e:
            # Use a temporary logger for this error since self.logger might not exist yet
            temp_logger = logging.getLogger(f'UniversalVLM.{self.model_name}')
            temp_logger.error(f"Could not load or validate universal_config.yaml: {e}. Using default settings.")
            return UniversalConfig()
    
    def _load_model_profile(self, model_name: str) -> ModelProfile:
        """
        Load and validate model-specific optimization profile using Pydantic.
        
        ARCHITECTURE: VLM-Agnostic Direct Resolution
        - Profile filename = model_name + ".yaml"
        - No hardcoded mappings required
        - Adding new models: Just create vlm_profiles/new-model.yaml
        
        Args:
            model_name: Model identifier (e.g., "olmocr-2-7b")
            
        Returns:
            Validated ModelProfile object
            
        Raises:
            FileNotFoundError: If profile file doesn't exist
            ValidationError: If profile YAML is invalid
            
        Example:
            model_name="olmocr-2-7b" → Loads vlm_profiles/olmocr-2-7b.yaml
        """
        # Construct filename directly from model name (VLM-agnostic)
        profile_file = f"{model_name}.yaml"
        
        # Look in VLM profiles directory (strategy config location)
        # Path: src/processing/strategies/config/vlm_profiles/
        profile_path = (
            Path(__file__).parent.parent.parent /  # Up to src/
            'processing' / 'strategies' / 'config' / 'vlm_profiles' /
            profile_file
        )
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                profile = ModelProfile(**data)
                
                self.logger.info(
                    f"✓ Loaded model profile: {model_name} "
                    f"(type: {profile.model_type}, status: {profile.status})"
                )
                return profile
                
        except FileNotFoundError:
            self.logger.error(
                f"Profile not found: {profile_path}\n"
                f"To add model '{model_name}', create: vlm_profiles/{profile_file}"
            )
            raise FileNotFoundError(
                f"No VLM profile found for model: {model_name}\n"
                f"Expected location: {profile_path}\n"
                f"Solution: Create vlm_profiles/{profile_file} with model configuration"
            )
            
        except ValidationError as e:
            self.logger.error(f"Invalid profile format in {profile_path}: {e}")
            raise ValueError(f"Profile validation failed for {model_name}: {e}")
    
    def _load_task_definition(self, task_type: str) -> TaskDefinition:
        """
        Load and validate task-specific prompts and settings using Pydantic.
        
        ARCHITECTURE: VLM-Agnostic Direct Resolution
        - Task filename = task_type + ".yaml"
        - No hardcoded mappings required
        - Adding new tasks: Just create task_definitions/new-task.yaml
        
        Args:
            task_type: Task identifier (e.g., "raw_ocr_extraction")
            
        Returns:
            Validated TaskDefinition object
            
        Raises:
            FileNotFoundError: If task file doesn't exist
            ValidationError: If task YAML is invalid
            
        Example:
            task_type="raw_ocr_extraction" → Loads task_definitions/raw_ocr_extraction.yaml
        """
        # Construct filename directly from task type (VLM-agnostic)
        task_file = f"{task_type}.yaml"
        
        # Look in task definitions directory (provider config location)
        # Path: src/llm/providers/config/task_definitions/
        task_path = Path(__file__).parent / 'config' / 'task_definitions' / task_file

        try:
            with open(task_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                task = TaskDefinition(**data)
                
                self.logger.info(
                    f"✓ Loaded task definition: {task_type} "
                    f"(output: {task.output_format.get('type', 'unknown')})"
                )
                return task
                
        except FileNotFoundError:
            self.logger.error(
                f"Task not found: {task_path}\n"
                f"To add task '{task_type}', create: task_definitions/{task_file}"
            )
            raise FileNotFoundError(
                f"No task definition found for: {task_type}\n"
                f"Expected location: {task_path}\n"
                f"Solution: Create task_definitions/{task_file} with task configuration"
            )
            
        except ValidationError as e:
            self.logger.error(f"Invalid task format in {task_path}: {e}")
            raise ValidationError(f"Task validation failed for {task_type}: {e}")

    def _check_model_availability(self):
        """ENHANCED: Pings the server's dedicated /health endpoint."""
        try:
            # SOLUTION: Construct the health URL from the main endpoint's base.
            from urllib.parse import urlparse, urlunparse
            
            parsed_url = urlparse(str(self.config.api_endpoint))
            health_url = urlunparse((parsed_url.scheme, parsed_url.netloc, '/health', '', '', ''))
            
            self.logger.info(f"Pinging model health endpoint: {health_url}")
            
            # Use GET, as health endpoints are almost always GET requests.
            response = self.session.get(health_url, timeout=5)
            response.raise_for_status()
            self.logger.info(f"Model server is available at {parsed_url.netloc}.")
            
        except requests.RequestException as e:
            self.logger.warning(f"Model health check failed: {e}")
            raise ConnectionError(f"Could not connect to VLM server at {health_url}") from e

    def _detect_api_type(self) -> str:
        """
        Detect API type based on model profile configuration.

        Returns:
            "gemini" if model_type contains "gemini"
            "openai_compatible" otherwise (default for llama.cpp)
        """
        model_type = getattr(self.model_profile, 'model_type', 'local')
        if 'gemini' in model_type.lower():
            return "gemini"
        return "openai_compatible"

    def _initialize_gemini_client(self):
        """Initialize Gemini API client with API key from environment."""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai SDK not installed. Install with: pip install google-genai"
            )

        # Load API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please set it in your .env file or environment variables."
            )

        # Initialize Gemini client
        try:
            self.gemini_client = genai.Client(api_key=api_key)
            self.logger.info(f"Gemini client initialized for model: {self.model_profile.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def _process_single_page_gemini(
        self, 
        page_image: Image.Image, 
        base_text: Optional[str] = None,
        task_definition: Optional[TaskDefinition] = None
    ) -> str:
        """
        Process a single page using Gemini Vision API.

        Args:
            page_image: PIL Image object
            base_text: Optional base text for document anchoring
            task_definition: Optional task definition to override default

        Returns:
            Raw text output from Gemini API
        """
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")

        effective_task = task_definition or self.task_definition
        start_time = time.time()

        # Prepare prompt
        prompt_text = effective_task.prompt
        if base_text and '{base_text}' in prompt_text:
            prompt_text = prompt_text.format(base_text=base_text)
            self.logger.info(f"Document anchoring enabled: {len(base_text)} chars base_text")

        # Convert PIL Image to base64 for Gemini
        # Gemini accepts inline base64 images
        buffered = io.BytesIO()
        page_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Get API parameters
        api_params = self.model_profile.get_api_params()
        temperature = api_params.get('temperature', 0.4)
        max_tokens = api_params.get('max_tokens', 8192)

        # Simple rate limit retry strategy
        # 1. Try request
        # 2. On 429: Retry once immediately
        # 3. On second 429: Wait 1 minute, retry twice (assume per-minute quota)
        # 4. If works: Resume | If fails: Assume daily quota exhausted

        def make_api_request():
            """Execute Gemini API request."""
            return self.gemini_client.models.generate_content(
                model=self.model_profile.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt_text),
                            types.Part.from_bytes(
                                data=buffered.getvalue(),
                                mime_type="image/png"
                            )
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )

        def is_rate_limit_error(error: Exception) -> bool:
            """Check if error is rate limit (429) or overload (503)."""
            error_str = str(error).lower()
            return ("429" in error_str or 
                    "quota" in error_str or 
                    "rate limit" in error_str or 
                    "503" in error_str or 
                    "overloaded" in error_str)

        response = None
        last_error = None

        try:
            # Initial request attempt
            response = make_api_request()

        except Exception as e:
            if not is_rate_limit_error(e):
                # Non-rate-limit error: fail immediately
                duration = time.time() - start_time
                self.logger.error(f"Gemini API error after {duration:.2f}s: {e}")
                raise

            # Rate limit error: Retry once immediately
            self.logger.warning(f"Gemini API rate limit hit, retrying immediately...")
            last_error = e

            try:
                response = make_api_request()
                self.logger.info("Immediate retry succeeded")

            except Exception as e2:
                if not is_rate_limit_error(e2):
                    # Different error on retry: fail
                    duration = time.time() - start_time
                    self.logger.error(f"Gemini API error on retry after {duration:.2f}s: {e2}")
                    raise

                # Second 429: Assume per-minute quota exhausted
                self.logger.warning(
                    "Second rate limit error - assuming per-minute quota exhausted. "
                    "Waiting 60 seconds..."
                )
                time.sleep(60)
                last_error = e2

                # Retry twice after waiting
                for attempt in range(1, 3):
                    try:
                        self.logger.info(f"Post-wait retry attempt {attempt}/2...")
                        response = make_api_request()
                        self.logger.info(f"Post-wait retry {attempt} succeeded - resuming processing")
                        break

                    except Exception as e3:
                        if not is_rate_limit_error(e3):
                            # Different error: fail
                            duration = time.time() - start_time
                            self.logger.error(f"Gemini API error after wait, attempt {attempt}: {e3}")
                            raise

                        last_error = e3
                        if attempt == 2:
                            # Failed both post-wait retries: Assume daily quota exhausted
                            duration = time.time() - start_time
                            self.logger.error(
                                f"Rate limit persists after 60s wait and 2 retries - "
                                f"daily quota likely exhausted (duration: {duration:.2f}s)"
                            )
                            raise RuntimeError(
                                "Gemini API daily quota likely exhausted. "
                                "Rate limit persists after waiting 60s and retrying twice. "
                                f"Original error: {last_error}"
                            ) from last_error

        # Process successful response
        duration = time.time() - start_time

        if not response or not response.text:
            raise ValueError("Empty response from Gemini API")

        raw_output = response.text

        # Calculate and log token usage/cost
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            input_tokens = getattr(usage, 'prompt_token_count', 0)
            output_tokens = getattr(usage, 'candidates_token_count', 0)

            # Calculate cost
            # Calculate cost using pricing from profile (defaulting to Gemini 3 Pro rates if missing)
            pricing = self.model_profile.performance.get('pricing', {})
            price_in = pricing.get('input_cost_per_million', 2.00)
            price_out = pricing.get('output_cost_per_million', 12.00)
            
            cost_input = (input_tokens / 1_000_000) * price_in
            cost_output = (output_tokens / 1_000_000) * price_out
            total_cost = cost_input + cost_output

            self.logger.info(
                f"Gemini API call: {duration:.2f}s, "
                f"Tokens: {input_tokens} in + {output_tokens} out, "
                f"Cost: ${total_cost:.6f}"
            )

        return raw_output.strip()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for universal provider."""
        logger = logging.getLogger(f'UniversalVLM.{self.model_name}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - [%(name)s] %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


# --- Enhanced Factory Functions for Easy and Resilient Usage ---

def create_best_available_provider(task_type: str = "auto") -> UniversalVLMProvider:
    """Create the best available provider based on task and model availability."""
    logger = logging.getLogger('ProviderFactory')
    model_priority = [
        ("allenai/olmOCR-7B-0825-FP8", "olmocr_document_processing"),
        ("olmocr-7b", "basic_ocr"),
        ("numarkdown-8b", "academic_extraction")
    ]
    
    for model, default_task in model_priority:
        try:
            # If task_type is 'auto', use the model's preferred default task
            effective_task = default_task if task_type == "auto" else task_type
            logger.info(f"Attempting to initialize provider with model: {model}")
            # NEW: This will raise ConnectionError if the endpoint isn't available
            return UniversalVLMProvider(model, effective_task)
        except ConnectionError:
            logger.warning(f"Could not connect to model '{model}'. Trying next in priority list.")
            continue
    
    raise RuntimeError("No suitable and available VLM provider could be found.")


def create_olmocr_provider(task_type: str = "olmocr_document_processing") -> UniversalVLMProvider:
    """Factory for official OlmOCR provider with 77.4 benchmark."""
    return UniversalVLMProvider("allenai/olmOCR-7B-0825-FP8", task_type)


def create_current_provider(task_type: str = "basic_ocr") -> UniversalVLMProvider:
    """Factory for current working olmocr-7b provider."""
    return UniversalVLMProvider("olmocr-7b", task_type)


# --- Enhanced Backwards Compatibility Wrapper ---

class HybridOlmOCRProvider(UniversalVLMProvider):
    """Backwards compatibility wrapper for existing code."""
    
    def __init__(self, config: Union[str, Dict, None] = None):
        """
        Initialize with backwards compatible interface.
        NEW: Now intelligently parses old config format.
        """
        model_name = "olmocr-7b"
        task_type = "basic_ocr"
        
        # NEW: Smarter config parsing
        if isinstance(config, str) and Path(config).exists():
            # If config is a path to a YAML file, try to load it
            try:
                with open(config, 'r') as f:
                    old_config = yaml.safe_load(f)
                    # Example: try to map an old key to the new system
                    model_name = old_config.get('model_identifier', model_name)
            except Exception as e:
                # Use a dedicated logger for the wrapper
                logging.warning(f"Could not parse old config file '{config}': {e}. Using defaults.")
        elif isinstance(config, dict):
            model_name = config.get('model_identifier', model_name)

        # Initialize the parent UniversalVLMProvider with the determined settings
        super().__init__(model_name, task_type)
        self.logger.info("Initialized in backwards compatibility mode.")


if __name__ == "__main__":
    """Test execution for Universal Provider validation."""
    
    # Configure logging to see detailed output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] %(levelname)s - %(message)s')

    # --- DEFINE YOUR TEST PARAMETERS HERE ---
    TEST_PDF_PATH = "path/to/your/test_document.pdf" # IMPORTANT: Change this path
    
    print("\n--- [TEST 1] ---")
    print("Testing the 'Best Available Provider' factory...")
    try:
        # This will automatically find the best, running model
        provider = create_best_available_provider(task_type="auto")
        
        # Process the first page of the document
        result = provider.process_document(TEST_PDF_PATH, pages=[1])

        if result['success']:
            print(f"Successfully processed document with model: {result['model_used']}")
            print(f"Processing time: {result['processing_time']}s")
            print(f"Extracted Metadata: {result['metadata']}")
            print("\n--- Extracted Content (first 200 chars) ---")
            print(result['structured_content'][:200] + "...")
            print("------------------------------------------")
        else:
            print(f"Processing failed: {result['error']}")

    except (RuntimeError, FileNotFoundError) as e:
        print(f"Test 1 Failed: {e}")

    print("\n--- [TEST 2] ---")
    print("Explicitly testing the official OlmOCR provider...")
    try:
        # Directly create the high-quality OlmOCR provider
        olmocr_provider = create_olmocr_provider()
        
        result = olmocr_provider.process_document(TEST_PDF_PATH, pages=[1])

        if result['success']:
            print(f"Successfully processed document with model: {result['model_used']}")
            # Check for OlmOCR-specific features
            if result['metadata']:
                print("OlmOCR YAML Front Matter found and parsed!")
            else:
                print("Warning: OlmOCR ran but did not produce YAML front matter.")
            print(f"Result: {result}")
        else:
            print(f"Processing failed: {result['error']}")

    except (ConnectionError, FileNotFoundError) as e:
        print(f"Test 2 Failed: Could not connect to or initialize OlmOCR provider. Error: {e}")