#!/usr/bin/env python3
"""
Metadata enrichment — ledger cache, Zotero lookup, three-way merge, provenance.

For biology researchers: the vision model's reading of a title page is a
*hypothesis* about the paper's identity. This module confirms or corrects it:
the PDF's own text layer is trusted for the DOI (exact machine text beats
pixels), Zotero supplies the canonical library record, and the merger
reconciles the three with the title-page text as referee. The result carries
provenance — *where* each fact came from — so a wrong record can be traced.

Ported from V5's `_extract_metadata_from_page1` (L2430),
`_enrich_metadata_with_cache_and_zotero` (L2492), `_prefer_text_layer_doi`
(L2576), `_sanitize_bibliographic_metadata` (L3023) and `_annotate_provenance`
(L3052), turned from strategy methods into functions taking their
dependencies explicitly.

**Not carried: `_resolve_author_ambiguity` (V5 L2735).** plan.md's
decomposition table lists it, but a repo-wide search finds zero callers in
V5 — it is dead code (and the only fitz+pytesseract user in this group).
Git history is the archive.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ....processing.ledger.semantic_ledger import SemanticLedger, generate_hash
from ...models import ContentPage, DocumentProfile
from ...utils.metadata_merger import merge_metadata
from ...utils.zotero_service import ZoteroService, normalize_doi_for_compare


def extract_metadata_from_page1(page1_content: ContentPage, pdf_path: Path | str,
                                logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Pull the bibliographic fields out of the processed title page.

    Handles both shapes the provider produces: a nested 'metadata' block
    (schema-constrained models) and a flat dict (already unwrapped).

    Args:
        page1_content: the processed title page.
        pdf_path: the source PDF — hashed into `source_hash`, the ledger's
            stable identity for this document.
        logger: optional logger.

    Returns:
        {title, authors, doi, year, journal, source_hash}; empty when the
        page's metadata is unusable.
    """
    log = logger or logging.getLogger("MetadataEnrichment")
    bibliographic: Dict[str, Any] = {}

    if not page1_content or not isinstance(page1_content.metadata, dict):
        log.warning("Title page content or its metadata is invalid, skipping extraction.")
        return bibliographic

    block = page1_content.metadata.get("metadata")
    if not isinstance(block, dict):
        block = page1_content.metadata
        log.debug("Using flat metadata structure. Keys: %s", list(block.keys()))

    bibliographic["title"] = block.get("title")
    authors = block.get("authors")
    bibliographic["authors"] = authors if isinstance(authors, list) else []
    bibliographic["doi"] = block.get("doi")
    bibliographic["year"] = block.get("publication_year")
    bibliographic["journal"] = block.get("journal_name")

    try:
        bibliographic["source_hash"] = generate_hash(str(pdf_path))
        log.info("Assigned stable source_hash from PDF: %s...",
                 bibliographic["source_hash"][:12])
    except Exception as error:
        log.error("Failed to generate source_hash: %s", error)
        bibliographic["source_hash"] = f"fallback_{time.time_ns()}"

    if bibliographic.get("doi"):
        log.info("Successfully extracted DOI: %s", bibliographic["doi"])
    else:
        log.warning("DOI was not found in the title page metadata block.")
    return bibliographic


def enrich_metadata(
    vlm_metadata: Dict[str, Any],
    pdf_path: Path | str,
    title_page_text: str = "",
    profile: Optional[DocumentProfile] = None,
    ledger: Optional[SemanticLedger] = None,
    zotero_service: Optional[ZoteroService] = None,
    doi_threshold: float = 0.85,
    use_ledger_cache: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    The final enrichment step: ledger cache -> Zotero -> merge -> provenance.

    Args:
        vlm_metadata: the title-page extraction (a hypothesis).
        pdf_path: the source PDF (hashed for the ledger lookup).
        title_page_text: referee text for the three-way merge.
        profile: carries the orchestrator's text-layer DOI + confidence.
        ledger: long-term memory; a cache hit short-circuits everything.
        use_ledger_cache: False re-derives metadata even when the ledger
            already holds a row for this document. This is what `--force`
            sets: re-processing must be able to correct a stored mistake.
        zotero_service: the canonical library, when configured.
        doi_threshold: minimum text-layer DOI confidence to trust it.
        logger: optional logger.

    Returns:
        The merged metadata dict, provenance included. The caller (the
        runner) decides when to write it to the ledger.
    """
    log = logger or logging.getLogger("MetadataEnrichment")
    log.info("Enriching metadata using cache + Zotero + referee text…")

    # The text-layer DOI must win BEFORE the Zotero lookup, which matches on
    # DOI first — otherwise a garbled VLM DOI fetches the wrong paper.
    vlm_metadata = prefer_text_layer_doi(vlm_metadata, profile,
                                         doi_threshold=doi_threshold, logger=log)

    document_id = generate_hash(str(pdf_path))

    if ledger and use_ledger_cache:
        cached = ledger.get_record(document_id)
        if cached:
            log.info("Ledger cache hit for document %s", document_id[:12])
            return cached.model_dump()
    elif ledger:
        # `--force` means redo the work, and metadata is work. Without this,
        # re-processing a document returned the row written by the FIRST run:
        # a fixed extractor could not correct a stored mistake, and the only
        # symptom was that nothing changed. It cost two full re-runs to find
        # when a Zotero mapping fix appeared to have no effect.
        log.info("Ledger cache bypassed for %s — re-deriving metadata.",
                 document_id[:12])

    zotero_record: Dict[str, Any] = {}
    zotero_confidence = 0.0
    zotero_source = "unknown"
    if zotero_service:
        try:
            log.info("Performing Zotero canonical lookup...")
            zr = zotero_service.get_canonical_metadata(vlm_metadata)
            zotero_record = zr.get("record", {})
            zotero_confidence = zr.get("confidence", 0.0)
            zotero_source = zr.get("source", "unknown")
        except Exception as error:
            log.error("Zotero lookup failed: %s", error)
            zotero_record = {}

    merged = merge_metadata(vlm_metadata, zotero_record, title_page_text)
    merged["validation_confidence"] = zotero_confidence
    merged["validation_source"] = zotero_source
    log.info("Merged Metadata (pre-review): %s", merged)

    provenance = annotate_provenance(merged)
    # Remove any stale provenance keys, then attach the fresh one — prevents
    # the circular reference V5 once shipped.
    merged.pop("_provenance", None)
    merged.pop("provenance", None)
    merged["provenance"] = provenance

    return merged


def prefer_text_layer_doi(vlm_metadata: Dict[str, Any],
                          profile: Optional[DocumentProfile] = None,
                          doi_threshold: float = 0.85,
                          logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Use the DOI read directly from the PDF's text layer over the VLM's.

    A DOI is a paper's permanent identifier; one wrong character and Zotero
    fetches the wrong paper. Born-digital PDFs store it as exact machine text,
    already extracted by the orchestrator's first pass and scored by HOW it
    was found (0.98 next to a "doi:" label, 0.95 as a doi.org URL, 0.85 bare
    in the body — which might belong to a *cited* paper). We accept it only
    above `doi_threshold`; below that, a regex hit is no better than the
    vision model's read of tiny footer print.

    Returns the metadata dict with 'doi' backfilled/corrected when the text
    layer is trustworthy, and 'doi_source' recording which source won.
    """
    log = logger or logging.getLogger("MetadataEnrichment")
    text_layer_doi = getattr(profile, "doi", None) if profile else None
    vlm_doi = vlm_metadata.get("doi")

    if not text_layer_doi:
        if vlm_doi:
            vlm_metadata["doi_source"] = "vlm_extraction"
        return vlm_metadata

    confidence = getattr(profile, "doi_confidence", 0.0) or 0.0
    if confidence < doi_threshold:
        log.info(
            "Text-layer DOI '%s' scored %.2f, below the %.2f threshold (likely a bare "
            "DOI that may belong to a cited paper). Keeping the VLM's DOI.",
            text_layer_doi, confidence, doi_threshold)
        if vlm_doi:
            vlm_metadata["doi_source"] = "vlm_extraction"
        return vlm_metadata

    if vlm_doi and normalize_doi_for_compare(vlm_doi) != normalize_doi_for_compare(text_layer_doi):
        log.warning(
            "DOI mismatch - VLM read '%s' but the PDF text layer says '%s' "
            "(confidence=%.2f). Trusting the text layer.",
            vlm_doi, text_layer_doi, confidence)
    elif not vlm_doi:
        log.info("VLM found no DOI; recovered '%s' from the PDF text layer "
                 "(confidence=%.2f).", text_layer_doi, confidence)

    vlm_metadata["doi"] = text_layer_doi
    vlm_metadata["doi_source"] = "pymupdf_text_layer"
    vlm_metadata["doi_confidence"] = confidence
    return vlm_metadata


def sanitize_bibliographic_metadata(bib: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the merged dict just before pydantic model instantiation.

    Empty strings become None (so `ExtractedMetadata`'s pattern validators
    don't reject them), and authors is always a list of stripped strings.
    """
    sanitized = dict(bib)

    for key in ("doi", "title", "journal", "publisher", "validation_source", "pdf_filename"):
        value = sanitized.get(key)
        if isinstance(value, str) and value.strip() == "":
            sanitized[key] = None

    authors = sanitized.get("authors")
    if authors is None:
        sanitized["authors"] = []
    elif isinstance(authors, str):
        sanitized["authors"] = [s.strip() for s in authors.split(",") if s.strip()]
    elif isinstance(authors, (list, tuple)):
        sanitized["authors"] = [str(a).strip() for a in authors if str(a).strip()]
    else:
        sanitized["authors"] = []

    return sanitized


def annotate_provenance(merged: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the provenance dict the runner inspects (missing fields, suspicious
    authors, confidence, Zotero match quality).

    Side effect kept from V5: `suspicious_authors` is *moved* out of the
    merged dict into the provenance — it is metadata about metadata.
    """
    provenance: Dict[str, Any] = {}

    for field in ("title", "authors", "doi", "journal"):
        if not merged.get(field):
            provenance[f"missing_{field}"] = True

    suspicious = merged.get("suspicious_authors", [])
    if suspicious:
        provenance["suspicious_authors"] = suspicious
        merged.pop("suspicious_authors", None)

    provenance["validation_confidence"] = merged.get("validation_confidence", 0.0)
    provenance["validation_source"] = merged.get("validation_source", "unknown")

    retrieval = merged.get("retrieval_provenance", {})
    if retrieval:
        provenance["zotero_source"] = retrieval.get("source_key", "none")

    return provenance
