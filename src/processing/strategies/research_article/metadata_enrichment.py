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
from ...citations.block import SOURCE_FILE_NAME, source_label
from ...citations.acceptance import (clean_author_list, complete_truncated_authors,
                                      is_truncated_author_list, surname_of)
from ...citations.level import LEVEL_LOCAL, LEVEL_ONLINE
from ...citations.local_confidence import CONFIDENCE_UNCONFIRMED, local_confidence
from ...models import ContentPage, DocumentProfile
from ...utils.metadata_merger import merge_metadata
from ...utils.zotero_service import ZoteroService, normalize_doi_for_compare

# Source labels the online chain gives a registry record it accepted
# (`citations/chain.py`: f"{service.name}_doi" / f"{service.name}_title").
REGISTRY_SOURCE_ENDINGS = ("_doi", "_title")


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
    citation_level: Optional[str] = None,
    file_name: Optional[str] = None,
    online_chain: Any = None,
    next_page_text: str = "",
) -> Dict[str, Any]:
    """
    The final enrichment step: ledger cache -> Zotero -> merge -> provenance.

    The citation level decides what happens here (see `citations/level.py`):

        local    nobody is asked. Confidence comes from how well the
                 vision model's reading and the PDF's text layer agree.
        zotero   the Zotero library is asked, exactly as before the level
                 existed.
        (none)   `citation_level=None`: the behaviour from before the level
                 existed, kept for callers that never name one.

    At every named level a record with no author, editor or title to cite
    by is marked as cited by its file name and flagged for review.

    Args:
        citation_level: the level this run uses, or None (see above).
        online_chain: the Crossref/OpenAlex chain, at level `online`.
        file_name: the PDF's file name, for the file-name fallback. Defaults
            to `pdf_path`'s name.
        vlm_metadata: the title-page extraction (a hypothesis).
        pdf_path: the source PDF (hashed for the ledger lookup).
        title_page_text: referee text for the three-way merge.
        next_page_text: the page after the title page, where a shortened
            author list is printed in full (Cell Press).
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
    log.info("Enriching metadata (citation level: %s)…", citation_level or "not named")

    document_id = generate_hash(str(pdf_path))

    if ledger and use_ledger_cache:
        cached = ledger.get_record(document_id)
        if cached and _cached_row_is_usable(cached, citation_level):
            log.info("Ledger cache hit for document %s", document_id[:12])
            return cached.model_dump()
        if cached:
            # A row written at another level answers a different question:
            # a `local` row has never been checked against Zotero, and
            # returning it at `zotero` would hide that for ever.
            log.info("Ledger row for %s was written at citation level %r, not %r: "
                     "re-deriving metadata.", document_id[:12],
                     getattr(cached, "citation_level", None), citation_level)
    elif ledger:
        # `--force` means redo the work, and metadata is work. Without this,
        # re-processing a document returned the row written by the FIRST run:
        # a fixed extractor could not correct a stored mistake, and the only
        # symptom was that nothing changed. It cost two full re-runs to find
        # when a Zotero mapping fix appeared to have no effect.
        log.info("Ledger cache bypassed for %s — re-deriving metadata.",
                 document_id[:12])

    return establish_metadata(
        vlm_metadata, title_page_text=title_page_text, profile=profile,
        zotero_service=zotero_service, doi_threshold=doi_threshold,
        citation_level=citation_level, online_chain=online_chain,
        file_name=file_name or Path(pdf_path).name, logger=log,
        next_page_text=next_page_text)


def establish_metadata(
    vlm_metadata: Dict[str, Any],
    title_page_text: str = "",
    profile: Any = None,
    zotero_service: Optional[ZoteroService] = None,
    doi_threshold: float = 0.85,
    citation_level: Optional[str] = None,
    file_name: str = "",
    logger: Optional[logging.Logger] = None,
    online_chain: Any = None,
    next_page_text: str = "",
) -> Dict[str, Any]:
    """
    Establish a document's bibliographic record from what was read of it.

    For biology researchers: this is the part of enrichment that needs
    nothing but the READINGS: the vision model's title-page extraction,
    the text-layer DOI and the title page's text. It needs no PDF on disk
    and no ledger. That is what lets `--recite` change a document's
    citation level in seconds, from its output file alone, with no GPU
    (`citations/recite.py`). A normal run reaches it through
    `enrich_metadata`, which adds the ledger cache in front.

    Args:
        vlm_metadata: the title-page extraction (a hypothesis).
        title_page_text: referee text for the three-way merge.
        next_page_text: the page after the title page, where a shortened
            author list is printed in full (Cell Press).
        profile: anything with `.doi` and `.doi_confidence` (the text
            layer's DOI and how it was found), or None.
        zotero_service: the Zotero library, at level `zotero` only.
        doi_threshold: minimum text-layer DOI confidence to trust it.
        citation_level: `local`, `zotero`, `online`, or None for the
            behaviour from before levels existed.
        file_name: the PDF's file name, for the file-name fallback.
        logger: optional logger.
        online_chain: the Crossref/OpenAlex chain (`citations/chain.py`),
            used at level `online` only.

    Returns:
        The merged metadata dict, provenance included.
    """
    log = logger or logging.getLogger("MetadataEnrichment")

    # The text-layer DOI must win BEFORE the Zotero lookup, which matches on
    # DOI first — otherwise a garbled VLM DOI fetches the wrong paper.
    vlm_metadata = prefer_text_layer_doi(vlm_metadata, profile,
                                         doi_threshold=doi_threshold, logger=log)

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

    network = None
    if citation_level == LEVEL_ONLINE and online_chain is not None:
        zotero_record, zotero_confidence, zotero_source, network = _online_lookup(
            online_chain, vlm_metadata, title_page_text, log)

    # The model sometimes returns the whole author line as ONE string. Split
    # it at every level, not only when a registry record exists: the merger's
    # "Fix-D" discards a single concatenated block on the assumption that a
    # registry list will replace it, and at `local` none does, so 4 of the 10
    # benchmark pilot papers ended with no authors at all (session 40).
    truncated = is_truncated_author_list(vlm_metadata.get("authors"))
    completed = complete_truncated_authors(vlm_metadata.get("authors"),
                                           next_page_text) if truncated else None
    if completed:
        log.info("Author list was shortened on the title page; %d authors read "
                 "from the next page's byline.", len(completed))
    vlm_metadata = {**vlm_metadata,
                    "authors": completed or clean_author_list(vlm_metadata.get("authors"))}

    merged =merge_metadata(vlm_metadata, zotero_record, title_page_text)
    if citation_level == LEVEL_ONLINE:
        prefer_registry_authors(merged, zotero_record, zotero_source, logger=log)
    merged["validation_confidence"] = zotero_confidence
    merged["validation_source"] = zotero_source
    apply_citation_level(merged, citation_level, title_page_text, file_name,
                         logger=log, network=network)
    log.info("Merged Metadata (pre-review): %s", merged)

    provenance = annotate_provenance(merged)
    if truncated and not completed:
        # The names shown are kept, but the list is known to be incomplete.
        provenance["authors_truncated"] = True
    # Remove any stale provenance keys, then attach the fresh one — prevents
    # the circular reference V5 once shipped.
    merged.pop("_provenance", None)
    merged.pop("provenance", None)
    merged["provenance"] = provenance

    return merged


def apply_citation_level(merged: Dict[str, Any], citation_level: Optional[str],
                         title_page_text: str, file_name: str,
                         logger: Optional[logging.Logger] = None,
                         network: Optional[str] = None) -> Dict[str, Any]:
    """
    Set a merged record's confidence and source according to the level.

    For biology researchers: this is where "how much do we trust this
    record, and who vouched for it?" is answered once the readings have
    been merged.

      - No level named: nothing changes (the behaviour before levels).
      - `zotero`: nothing changes either. Zotero already set the confidence
        and the source, and this level must give the same record as before.
      - `local`: nobody vouched, so the confidence comes from how well
        Prajnya's own readings agree (`citations/local_confidence.py`).
      - `online`: an accepted Crossref/OpenAlex record keeps the confidence
        and source the chain gave it. When no record was accepted, the
        record stands on Prajnya's own reading exactly as at `local`; if the
        cause was a dead network (`network == "unavailable"`) the confidence
        is also capped at the unconfirmed value so the document is flagged.
      - Any named level: a record with no author, editor or title is
        cited by the PDF's file name. Its confidence becomes 0.0 and its
        source `file_name`, which flags it for review. The file name itself
        is NOT written into the record (it is the `source_label` of the
        output's `citations` block): the title stays empty.

    Args:
        merged: the merged metadata; changed in place and returned.
        citation_level: `local`, `zotero`, `online` or None.
        title_page_text: the title page's text (referee for `local`).
        file_name: the PDF's file name.
        logger: optional logger.

    Returns:
        `merged`, carrying `citation_level` when one was named.
    """
    log = logger or logging.getLogger("MetadataEnrichment")
    if citation_level is None:
        return merged

    merged["citation_level"] = citation_level

    if citation_level == LEVEL_LOCAL:
        confidence, source = local_confidence(merged, title_page_text)
        merged["validation_confidence"] = confidence
        merged["validation_source"] = source
        log.info("Citation level local: confidence %.2f (%s).", confidence, source)

    if citation_level == LEVEL_ONLINE and merged.get("validation_source") in (None, "", "unknown"):
        confidence, source = local_confidence(merged, title_page_text)
        if network == "unavailable":
            confidence = min(confidence, CONFIDENCE_UNCONFIRMED)
            merged["network"] = "unavailable"
            log.warning("Citation level online, but no registry could be reached: "
                        "the record stands on Prajnya's own reading and is flagged.")
        merged["validation_confidence"] = confidence
        merged["validation_source"] = source

    _label, by_file_name = source_label(merged, file_name)
    if by_file_name:
        merged["validation_confidence"] = 0.0
        merged["validation_source"] = SOURCE_FILE_NAME
        log.warning("No author, editor or title could be established: %s is "
                    "cited by its file name and flagged for review.", file_name)
    return merged


def prefer_registry_authors(merged: Dict[str, Any], record: Dict[str, Any], source: str,
                            logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    At level `online`, the accepted registry record's author list is the one kept.

    For biology researchers: the three-way merge (`merge_author_lists`) takes
    the vision model's reading of the byline as its reference and lets the
    registry only confirm or add names, so a misread surname ("Hümer" for
    Crossref's "Hümmer"), a glued affiliation mark or a dropped name survived
    even when Crossref held the exact list. A registry record accepted by
    the chain (`crossref_doi`, `openalex_title`, ...) already lists every
    author in published order, with the title page's fuller spelling of a
    name the page also prints (`chain.respell_authors`); it is used as is.
    Measured on the benchmark's 50 non-scored papers (run 1): the registry
    list was exact on all 43 it covered, the merged list on 28 of 50.

    A Zotero record (level `zotero`, or `online` with Zotero preferred) is
    left to the merge, so those levels give the same record as before.

    **A registry list that is a cut-off copy of the byline is not used.**
    Crossref's record for Myllyharju and Kivirikko 2004 (Trends Genet.,
    10.1016/j.tig.2003.11.004) names only "J Myllyharju"; taken as is, it
    dropped Kivirikko (v6.8 release check, 2026-11-22). When the registry's
    surnames are the byline's first surnames, in order, and fewer, the
    registry list is truncated, and the merged reading is kept. Never seen on
    the benchmark's 50 papers, so their result is unchanged.

    The record's separate surnames come with it (`author_names`, contract
    1.4), one per author, so "Rosendahl Huber" stays one surname; [] when the
    registry gives none (OpenAlex names are single strings).

    Args:
        merged: the merged metadata; its `authors` and `author_names`
            changed in place.
        record: the record the online chain accepted ({} when none).
        source: the chain's source label for that record.
        logger: optional logger.

    Returns:
        `merged`.
    """
    authors = [a for a in (record or {}).get("authors") or [] if isinstance(a, str) and a.strip()]
    if not authors or not source.endswith(REGISTRY_SOURCE_ENDINGS) or source.startswith("zotero"):
        return merged
    if _cut_off_copy(authors, merged.get("authors") or []):
        (logger or logging.getLogger("MetadataEnrichment")).warning(
            "The %s author list (%d) is the byline's first %d names only: kept "
            "the merged reading (%d).", source, len(authors), len(authors),
            len(merged.get("authors") or []))
        return merged
    if authors != merged.get("authors"):
        (logger or logging.getLogger("MetadataEnrichment")).info(
            "Authors taken from the %s record (%d) instead of the merged reading (%d).",
            source, len(authors), len(merged.get("authors") or []))
    merged["authors"] = list(authors)
    parts = (record or {}).get("author_names")
    merged["author_names"] = ([dict(part) for part in parts]
                              if isinstance(parts, list) and len(parts) == len(authors)
                              else [])
    return merged


def _cut_off_copy(registry: list, reading: list) -> bool:
    """True when the registry names fewer authors, and they are the first
    names of the reading, in order (compared by surname)."""
    if not registry or len(registry) >= len(reading):
        return False
    first = [surname_of(name).casefold() for name in reading[:len(registry)]]
    return first == [surname_of(name).casefold() for name in registry]


def _online_lookup(chain: Any, vlm_metadata: Dict[str, Any], title_page_text: str,
                   log: logging.Logger):
    """
    Ask the online chain and unpack its answer for the merger.

    A DOI the registry returned a DIFFERENT paper for is removed from the
    reading first (acceptance rule 6): left in, the merger would copy it
    into the final record.

    Returns:
        `(record, confidence, source, network)`.
    """
    try:
        found = chain.get_canonical_metadata(vlm_metadata, title_page_text)
    except Exception as error:      # a registry problem must never end a document run
        log.error("Online citation lookup failed: %s", error)
        return {}, 0.0, "unknown", "unavailable"
    if found.get("drop_doi"):
        log.warning("DOI %s belongs to a different paper than the title page: dropped.",
                    vlm_metadata.get("doi"))
        vlm_metadata["doi"] = None
        vlm_metadata.pop("doi_source", None)
        vlm_metadata.pop("doi_confidence", None)
    return (found.get("record") or {}, found.get("confidence", 0.0),
            found.get("source", "unknown"), found.get("network"))


def _cached_row_is_usable(cached: Any, citation_level: Optional[str]) -> bool:
    """
    Whether a stored ledger row may stand in for re-deriving the metadata.

    With no level named, any row is used (the behaviour before levels).
    With one, only a row written at that same level: a row from another
    level, or from before levels were recorded, is derived again. That
    costs no GPU time; the title page has already been read by this point.
    """
    if citation_level is None:
        return True
    return getattr(cached, "citation_level", None) == citation_level


def prefer_text_layer_doi(vlm_metadata: Dict[str, Any],
                          profile: Any = None,
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

    network = merged.pop("network", None)      # metadata about metadata, like suspicious_authors
    if network:
        provenance["network"] = network

    retrieval = merged.get("retrieval_provenance", {})
    if retrieval:
        provenance["zotero_source"] = retrieval.get("source_key", "none")

    return provenance
