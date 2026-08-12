#!/usr/bin/env python3
"""
Cloud pipeline — the simplified single-pass path for hosted APIs (Gemini).

For biology researchers: when the config selects a cloud model instead of the
local GPU, each page is read in ONE model call (cloud models are strong
enough not to need the two-shot dance), and a "lite router" still sends
pure-text pages to the free text layer so you never pay API tokens for text
the PDF already contains.

Ported from V5's `_process_title_page_gemini` (L748) and
`_process_body_pages_gemini` (L856), kept because it is reachable via config
(a profile with `model_type: cloud_api_*`). Access goes through
`DocumentCache`/`PdfPage`; task definitions load through `ConfigLoader`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ....llm.providers.schemas import TaskDefinition
from ...config.loader import ConfigLoader
from ...models import ContentPage
from ...utils.page_classifier import PageClassifier, extract_text_stats
from .rendering import DocumentCache

#: The single-pass task both cloud paths run (see config/task_definitions/).
SINGLE_PASS_TASK = "comprehensive_single_pass_extraction"

#: Appended to the title-page prompt so the model outputs the metadata block.
TITLE_PAGE_OVERRIDE = (
    "\n\nCRITICAL OVERRIDE: This specific page IS the TITLE PAGE of the "
    "document. You MUST extract the metadata in JSON format as specified in "
    "the Metadata section."
)


def process_title_page_cloud(
    cache: DocumentCache,
    title_page_index: int,
    provider,                      # a cloud-backed VlmProvider
    loader: ConfigLoader,
    resolution: int,
    logger: Optional[logging.Logger] = None,
) -> ContentPage:
    """
    Single-pass cloud title page: structured metadata + content in one shot.

    Returns a `ContentPage` whose `metadata['generated_metadata']` carries the
    parsed bibliographic block (the strategy merges it downstream).
    """
    log = logger or logging.getLogger("CloudPipeline")
    log.info("Cloud: Processing title page %d...", title_page_index + 1)

    try:
        start_time = time.time()
        image = cache.page_image(title_page_index, target_size=resolution)

        task_data = loader.task_definition(SINGLE_PASS_TASK)
        task = TaskDefinition(**task_data)
        task.prompt += TITLE_PAGE_OVERRIDE

        result = provider.process_page(page_image=image, task_definition=task)
        if not result.get("success", False):
            raise RuntimeError(f"Provider failed: {result.get('error', 'Unknown error')}")

        content = result.get("content", "")
        log.debug("Cloud output for title page (first 500 chars):\n%s", content[:500])
        content, metadata = _split_metadata_block(content, log)

        return ContentPage(
            page_number=title_page_index + 1, content=content,
            quality_score=0.95,  # high confidence for the cloud single pass
            metadata={
                "processing_method": "gemini_single_pass",
                "total_time": round(time.time() - start_time, 2),
                "generated_metadata": metadata,
            },
        )
    except Exception as error:
        log.error("Cloud title page failed: %s", error, exc_info=True)
        return ContentPage(page_number=title_page_index + 1, content="",
                           quality_score=0.0, metadata={"error": str(error)})


def process_body_pages_cloud(
    cache: DocumentCache,
    page_indices: Sequence[int],
    provider,
    loader: ConfigLoader,
    resolution: int,
    page_classifier: Optional[PageClassifier] = None,
    logger: Optional[logging.Logger] = None,
) -> List[ContentPage]:
    """
    Body pages via the "lite router": free text layer where it is safe,
    one cloud call per page everywhere else.

    A page goes to the text layer only when it is strictly boring: text
    density >= 0.90 AND no visuals AND no equations.
    """
    log = logger or logging.getLogger("CloudPipeline")
    log.info("Cloud: Processing %d body pages with Lite Router...", len(page_indices))

    task = TaskDefinition(**loader.task_definition(SINGLE_PASS_TASK))
    processed: List[ContentPage] = []

    for page_idx in page_indices:
        try:
            page = cache.page(page_idx)

            analysis: Dict[str, Any] = {}
            use_text_layer = False
            if page_classifier:
                stats = extract_text_stats(page)
                _, analysis = page_classifier.classify(page, page_idx, text_stats=stats)
                text_density = analysis.get("text_density", 0.0)
                visual_count = analysis.get("visual_count", 0)
                equation_count = len(analysis.get("equation_regions", []))
                if text_density >= 0.90 and visual_count == 0 and equation_count == 0:
                    use_text_layer = True
                    log.info("Page %d: Lite Router -> text layer (Density=%.2f)",
                             page_idx + 1, text_density)
                else:
                    log.info("Page %d: Lite Router -> cloud (Density=%.2f, "
                             "Visuals=%d, Eqs=%d)", page_idx + 1, text_density,
                             visual_count, equation_count)

            if use_text_layer:
                processed.append(ContentPage(
                    page_number=page_idx + 1, content=page.text(),
                    quality_score=0.95,
                    metadata={"processing_method": "pymupdf", "analysis": analysis},
                ))
            else:
                start_time = time.time()
                image = cache.page_image(page_idx, target_size=resolution)
                result = provider.process_page(page_image=image, task_definition=task)
                processed.append(ContentPage(
                    page_number=page_idx + 1, content=result.get("content", ""),
                    quality_score=0.90,
                    metadata={"processing_method": "gemini_single_pass",
                              "total_time": round(time.time() - start_time, 2)},
                ))
        except Exception as error:
            log.error("Cloud body page %d failed: %s", page_idx + 1, error,
                      exc_info=True)
            processed.append(ContentPage(page_number=page_idx + 1, content="",
                                         quality_score=0.0,
                                         metadata={"error": str(error)}))

    return processed


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _split_metadata_block(content: str,
                          log: logging.Logger) -> Tuple[str, Dict[str, Any]]:
    """
    Pull the JSON metadata block out of the model's output.

    Handles both a ```json fenced block and a bare JSON object at the start
    of the content. The block is removed from the returned content so the
    page text stays clean markdown. Parse failures return the content
    untouched and an empty metadata dict — a bad block must not lose the page.
    """
    metadata: Dict[str, Any] = {}

    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not match:
        match = re.search(r"^\s*(\{.*?\})\s*", content, re.DOTALL)
    if not match:
        return content, metadata

    try:
        parsed = json.loads(match.group(1))
        metadata = parsed.get("metadata", parsed) if isinstance(parsed, dict) else {}

        required = ["title", "authors", "doi", "journal_name", "publication_year"]
        missing = [f for f in required if f not in metadata]
        if missing:
            log.warning("JSON metadata missing fields: %s", missing)
        if metadata.get("doi") is None:
            log.info("DOI not found in title page (this may be expected)")

        content = content.replace(match.group(0), "").strip()
    except json.JSONDecodeError as error:
        log.warning("Failed to parse JSON metadata: %s", error, exc_info=True)
    except Exception as error:
        log.warning("Error processing metadata: %s", error, exc_info=True)

    return content, metadata
