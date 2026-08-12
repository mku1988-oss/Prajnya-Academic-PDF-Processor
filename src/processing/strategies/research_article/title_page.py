#!/usr/bin/env python3
"""
Title-page processing — the two-shot metadata read.

For biology researchers: the title page is where a paper declares its
identity (title, authors, journal, DOI), so it gets the most careful — and
most expensive — treatment. The model reads it twice:

    Shot 1 (temp≈0.05): a faithful, verbatim transcription of the page.
    Deduplication:      OCR models sometimes repeat a paragraph; drop repeats.
    Shot 2 (temp≈0.0):  structured extraction — the model is shown the page
                        *and* the cleaned transcription and asked to fill in
                        the metadata fields.

Ported from V5's `_process_title_page` (L445) and `_deduplicate_paragraphs`
(L2150).

**Stage 0 fix, session 8: shot 2 now actually receives shot 1's text.** V5's
YAML asked for `{raw_text}`, a name nothing ever substitutes, so shot 2 was
handed the literal string `{raw_text}` under a heading promising the
transcription — and re-read the metadata from pixels while the transcription
it had just paid for went unused. The fix was not the one line plan.md
expected: correcting the placeholder alone made the call *crash*, because the
prompt shows the model an example JSON structure and `str.format` treats every
literal brace as a field. Both halves are needed — see the substitution note in
`llm/providers/prompts.py`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from ...models import ContentPage
from ...events import emit
from ...utils.anchor_extractor import extract_document_anchoring
from .provider_factory import ProviderFactory
from .rendering import DocumentCache


def process_title_page(
    cache: DocumentCache,
    title_page_index: int,
    factory: ProviderFactory,
    title_profile_data: Dict[str, Any],
    shot1_task: str,
    shot2_task: str,
    resolution: int,
    logger: Optional[logging.Logger] = None,
) -> ContentPage:
    """
    Run the two-shot title-page read and return the page with its metadata.

    Args:
        cache: the open document cache (render + text access).
        title_page_index: 0-indexed page number of the detected title page.
        factory: builds the per-shot providers.
        title_profile_data: raw profile dict of the title-page model — its
            `two_shot_parameters` supply each shot's sampling settings.
        shot1_task: task name for the verbatim OCR shot (registry-resolved).
        shot2_task: task name for the structured-metadata shot.
        resolution: render size for the title page (1288px in config — chosen
            so the footer DOI survives).
        logger: optional logger.

    Returns:
        A `ContentPage` whose content is the deduplicated shot-1 text and
        whose metadata is shot 2's extraction (plus timing bookkeeping). On
        failure, an empty page with the error recorded — the caller decides
        whether a failed title page ends the document.
    """
    log = logger or logging.getLogger("TitlePage")
    emit(log, "stage", stage="title",
         message=f"Title page (page {title_page_index + 1}): two-shot extraction",
         page=title_page_index + 1)

    try:
        start_time = time.time()
        page = cache.page(title_page_index)
        image = cache.page_image(title_page_index, target_size=resolution)
        anchor_text = extract_document_anchoring(page)

        two_shot_params = title_profile_data.get("two_shot_parameters", {})

        # --- Shot 1: raw OCR (faithful transcription) ----------------------
        log.info("Title Page Shot 1: Raw OCR extraction...")
        shot1_start = time.time()
        shot1_params = two_shot_params.get("shot1_ocr", {})
        shot1_provider = factory.build_shot(
            title_profile_data, shot1_task,
            temperature=shot1_params.get("temperature", 0.05),
            max_tokens=shot1_params.get("max_tokens", 8192),
            repeat_penalty=shot1_params.get("repeat_penalty", 1.15),
            frequency_penalty=shot1_params.get("frequency_penalty", 0.1),
        )
        # The anchor text gives structural guidance about the whole page
        # without biasing the model toward specific regions.
        shot1_result = shot1_provider.process_page(image, base_text=anchor_text)
        raw_text = shot1_result.get("content", "")
        shot1_time = time.time() - shot1_start
        log.info("Title Page Shot 1 complete: %d characters in %.2fs",
                 len(raw_text), shot1_time)

        if not raw_text:
            log.error("Title Page Shot 1 returned empty text - CRITICAL FAILURE")
            return ContentPage(
                page_number=title_page_index + 1, content="", quality_score=0.0,
                metadata={"error": "shot1_empty_title_page",
                          "processing_method": "two_shot_title_failed"},
            )

        # --- Deduplication --------------------------------------------------
        dedup_start = time.time()
        cleaned_text = deduplicate_paragraphs(raw_text)
        dedup_time = time.time() - dedup_start
        log.info("Title Page deduplication: %d -> %d chars (%.1f%% retained) in %.2fs",
                 len(raw_text), len(cleaned_text),
                 len(cleaned_text) / len(raw_text) * 100, dedup_time)

        # --- Shot 2: structured metadata extraction -------------------------
        log.info("Title Page Shot 2: Structured metadata extraction...")
        shot2_start = time.time()
        shot2_params = two_shot_params.get("shot2_analysis", {})
        shot2_provider = factory.build_shot(
            title_profile_data, shot2_task,
            temperature=shot2_params.get("temperature", 0.0),
            max_tokens=shot2_params.get("max_tokens", 8192),
            repeat_penalty=shot2_params.get("repeat_penalty", 1.15),
            frequency_penalty=shot2_params.get("frequency_penalty", 0.1),
        )
        shot2_result = shot2_provider.process_page(image, base_text=cleaned_text)
        shot2_time = time.time() - shot2_start
        log.info("Title Page Shot 2 complete in %.2fs", shot2_time)

        # Shot 2's task returns ONLY metadata — the page content is always
        # the cleaned shot-1 transcription.
        final_metadata = dict(shot2_result.get("metadata", {}))
        final_content = cleaned_text

        # Quality is scored on the content we keep, not on shot 2's
        # metadata-only task (whose score would be misleading).
        quality_score = 0.85 if len(final_content) > 100 else 0.75

        final_metadata.update({
            "processing_method": "two_shot_title_page",
            "shot1_time": round(shot1_time, 2),
            "shot2_time": round(shot2_time, 2),
            "dedup_time": round(dedup_time, 2),
            "total_time": round(time.time() - start_time, 2),
            "raw_text_length": len(raw_text),
            "cleaned_text_length": len(cleaned_text),
            "dedup_ratio": round(len(cleaned_text) / len(raw_text), 3) if raw_text else 0.0,
        })
        emit(log, "stage", stage="title",
             message=(f"Title page two-shot complete: "
                      f"{final_metadata['total_time']}s total, "
                      f"quality={quality_score:.2f}"),
             page=title_page_index + 1,
             elapsed=final_metadata["total_time"], quality=quality_score)

        return ContentPage(page_number=title_page_index + 1, content=final_content,
                           quality_score=quality_score, metadata=final_metadata)

    except Exception as error:
        log.error("Title page two-shot processing failed: %s", error, exc_info=True)
        return ContentPage(
            page_number=title_page_index + 1, content="", quality_score=0.0,
            metadata={"error": str(error),
                      "processing_method": "two_shot_title_failed"},
        )


def deduplicate_paragraphs(text: str) -> str:
    """
    Remove repeated paragraphs while preserving first-appearance order.

    OCR models can emit the same paragraph twice during a long transcription.
    Paragraphs are compared with normalized whitespace so a re-wrapped repeat
    still counts as a duplicate.

    Example:
        "Para A\\n\\nPara B\\n\\nPara A" -> "Para A\\n\\nPara B"
    """
    if not text:
        return text

    seen = set()
    unique = []
    for paragraph in text.split("\n\n"):
        normalized = " ".join(paragraph.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(paragraph)
    return "\n\n".join(unique)
