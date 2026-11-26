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
from typing import Any, Dict, Optional, Tuple

from ...models import ContentPage
from ...events import emit
from ...pdf.base import PdfPage
from ...utils.anchor_extractor import extract_document_anchoring
from ...utils.content_drop_check import is_scanned_page
from ...utils.script_marking import (
    recase_capitals_journal, text_with_scripts, unicode_scripts, without_script_marks,
)
from ...utils.text_normalization import normalize_harvested_text
from .provider_factory import ProviderFactory
from .rendering import DocumentCache

#: Shot 1's text comes from the PDF's text layer on a born-digital title page
#: (T37, eval/title_text_layer/GATE.md), or from the vision model ("vlm").
SHOT1_TEXT_LAYER = "text_layer"
SHOT1_VLM = "vlm"

#: Below this many words, a text layer is too thin to stand in for shot 1
#: (a cover page with one image, a broken text layer): the model reads it.
MIN_TEXT_LAYER_WORDS = 50


def process_title_page(
    cache: DocumentCache,
    title_page_index: int,
    factory: ProviderFactory,
    title_profile_data: Dict[str, Any],
    shot1_task: str,
    shot2_task: str,
    resolution: int,
    logger: Optional[logging.Logger] = None,
    shot1_source: str = SHOT1_VLM,
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
        shot1_source: "vlm" (the model transcribes the page) or "text_layer"
            (T37: on a born-digital page, the PDF's own text stands in for
            shot 1; a scanned or near-empty page still goes to the model).

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

        # --- Shot 1: the page's full text -----------------------------------
        shot1_start = time.time()
        raw_text, used_source = _shot1_text(
            page, image, anchor_text, factory, title_profile_data, shot1_task,
            two_shot_params.get("shot1_ocr", {}), shot1_source, log)
        shot1_time = time.time() - shot1_start
        log.info("Title Page Shot 1 complete (%s): %d characters in %.2fs",
                 used_source, len(raw_text), shot1_time)

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
        if used_source == SHOT1_TEXT_LAYER:
            _tidy_text_layer_metadata(final_metadata)
        elif "title" in final_metadata:
            # T37e: the vision model's own titles carry bare LaTeX too
            # ("Ca^{2+}", "P2RY_{12}"); written in Unicode on both paths.
            final_metadata["title"] = unicode_scripts(final_metadata["title"])
        final_content = cleaned_text

        # Quality is scored on the content we keep, not on shot 2's
        # metadata-only task (whose score would be misleading).
        quality_score = 0.85 if len(final_content) > 100 else 0.75

        final_metadata.update({
            "processing_method": "two_shot_title_page",
            "shot1_source": used_source,
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


def _shot1_text(page: PdfPage, image, anchor_text: str, factory: ProviderFactory,
                title_profile_data: Dict[str, Any], shot1_task: str,
                shot1_params: Dict[str, Any], shot1_source: str,
                log: logging.Logger) -> Tuple[str, str]:
    """
    Shot 1's transcription of the title page, and where it came from.

    For biology researchers: on a born-digital PDF the publisher's own text is
    stored in the file, so reading it back costs nothing, while the vision
    model needs 10-41 s to transcribe the same page from its picture (T37).
    The model is still used when the page is a scan or its stored text is
    too thin to trust.

    Returns:
        (text, "text_layer") or (text, "vlm").
    """
    if shot1_source == SHOT1_TEXT_LAYER:
        text = _usable_text_layer(page, log)
        if text:
            log.info("Title Page Shot 1: text layer (born-digital page)")
            return text, SHOT1_TEXT_LAYER

    log.info("Title Page Shot 1: Raw OCR extraction...")
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
    return shot1_result.get("content", ""), SHOT1_VLM


def _tidy_text_layer_metadata(metadata: Dict[str, Any]) -> None:
    """
    Undo what the text layer leads shot 2 to copy (T37b, T37c), in place.

    - authors: script marks, ORCID "ID"s and symbols copied into the names
      ("Mattia Serra$^{1}$*", "MiuraID1,2☯"), see `without_script_marks`;
    - journal: a running header copied whole ("Science Advances | Research
      Article"): a journal name with " | " keeps its first part (T37c fix 2);
      one in capitals is re-cased ("SCIENCE ADVANCES"; T37d fix 2);
    - title: script marks written in Unicode ("CO$_{2}$" -> "CO₂"; T37d fix 1).
    """
    if "authors" in metadata:
        metadata["authors"] = without_script_marks(metadata["authors"])
    if "title" in metadata:
        metadata["title"] = unicode_scripts(metadata["title"])
    journal = metadata.get("journal_name")
    if isinstance(journal, str) and " | " in journal:
        journal = journal.split(" | ")[0].strip()
    if isinstance(journal, str):
        metadata["journal_name"] = recase_capitals_journal(journal)


def _usable_text_layer(page: PdfPage, log: logging.Logger) -> str:
    """
    The page's text layer, normalised as the body's text-layer route does,
    or "" when it cannot stand in for shot 1 (a scan, or < 50 words).

    T37b: superscripts and subscripts are written as `$^{..}$` / `$_{..}$`
    (`utils/script_marking.py`), so an affiliation letter is never glued to
    a surname ("Harvey-Samuela" -> "Harvey-Samuel$^{a}$").
    """
    if is_scanned_page(page):
        log.info("Title page is a scan: shot 1 is read by the model")
        return ""
    marked = text_with_scripts(page)
    if marked.aligned:
        log.info("Title page text layer: %d sub/superscript group(s) marked", marked.runs)
    else:
        log.warning("Title page text and characters do not align: its "
                    "sub/superscripts are not marked")
    text = normalize_harvested_text(marked.text)
    if len(text.split()) < MIN_TEXT_LAYER_WORDS:
        log.info("Title page text layer has %d words (< %d): shot 1 is read by "
                 "the model", len(text.split()), MIN_TEXT_LAYER_WORDS)
        return ""
    return text


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
