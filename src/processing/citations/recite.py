#!/usr/bin/env python
"""Re-cites a processed document at another citation level, with no GPU.

For biology researchers: the citation level decides who is asked to confirm
what paper a PDF is (nobody, your Zotero library, or online registries).
Changing your mind about that level should not cost you the processing all
over again, and it does not need to. Reading the pages is the expensive
part (about two and a half minutes of GPU per paper); deciding the
bibliographic record from what was read takes well under a second.

Every output file already holds what was READ from its title page:

    pages[title page].metadata    the vision model's reading
    profile.doi, doi_confidence   the DOI found in the PDF's text layer
    profile.page_1_text           the title page's text

Those are facts. The record in `metadata_final` is DERIVED from them, and
this module derives it again at the level you ask for, through the very
same function a normal run uses (`establish_metadata`). The PDF does not
need to be on disk.

WHAT IT CHANGES in the file: `metadata_final` (and its copy inside
`profile`), the `citations` block, the review status, and the contract
version stamp (the file now carries a block). The ledger row is updated to
match. WHAT IT NEVER TOUCHES: pages, text, figures, tables, crops.

THE REFERENCE LIST follows the level. At level `online` an article's list
is fetched again (from the PDF too, when it is still on disk; otherwise from
the online services alone). At any other level the block returns to
"not requested" with no references, so re-citing downward removes the list.

TWO KINDS OF DOCUMENT KEEP THEIR RECORD and only have the level recorded:

  - a record somebody CORRECTED BY HAND. A person's correction outranks
    anything derived, at every level;
  - a BOOK. Books have their own metadata path (the front matter) and look
    nothing up.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

# Page readings that failed (their metadata is not a reading of anything)
# are defined once, in `document_status`, which also decides a document's status.
from ..document_status import FAILED_METHODS
from ..utils.atomic_write import write_json_atomically
from ..review import (assess_metadata_completeness, build_ledger_record,
                      save_record)
from .block import build_citations_block
from .level import require_built, resolve_level
from .services import online_chain_for, reference_finder_for, zotero_service_for

LOG = logging.getLogger(__name__)

#: What happened to one document.
RECITED = "recited"                # the record was derived again
LEVEL_RECORDED = "level_recorded"  # record kept (hand-corrected, or a book)
SKIPPED = "skipped"                # nothing was written; `reason` says why

#: The keys `establish_metadata` decides. They are removed from the old
#: record before the new one is laid over it, so a value the new level does
#: not supply (a publisher only Zotero knew) does not linger as if it did.
DERIVED_KEYS = ("title", "authors", "year", "doi", "journal", "publisher",
                "validation_confidence", "validation_source", "provenance",
                "doi_source", "doi_confidence", "citation_level",
                "retrieval_provenance")

#: The fields a person reads to see what a re-cite changed.
REPORTED_FIELDS = ("title", "authors", "year", "doi", "journal", "publisher")


@dataclass
class ReciteResult:
    """
    What re-citing did to one document.

    Attributes:
        document: the output file's name.
        status: `recited`, `level_recorded` or `skipped`.
        reason: why, in plain words (always set for the last two).
        level_before: the level the file had (None for a contract 1.0 file).
        level_after: the level it has now.
        changed: bibliographic fields whose value changed, as
            {field: [before, after]}.
        requires_review: whether the new record is flagged for a human.
        reference_list_status: the written list's status (`found`,
            `not_available`, ...), or None when no list was requested.
        reference_count: how many references the document now holds.
    """

    document: str
    status: str
    reason: str = ""
    level_before: Optional[str] = None
    level_after: Optional[str] = None
    changed: Dict[str, List[Any]] = field(default_factory=dict)
    requires_review: bool = False
    reference_list_status: Optional[str] = None
    reference_count: int = 0


def recite_documents(json_paths: Sequence[Path], config: Dict[str, Any],
                     override: Optional[str] = None, ledger=None,
                     doi_threshold: Optional[float] = None,
                     logger: Optional[logging.Logger] = None) -> List[ReciteResult]:
    """
    Re-cite a set of processed documents at one level.

    For biology researchers: this is `--recite`. It decides the level once
    (the `--citations` flag, else `citations.level` in the config), opens
    whatever that level may contact once, and then works through the files.

    Args:
        json_paths: the output files (`<name>.v6.json`) to re-cite.
        config: the loaded strategy config.
        override: the `--citations` value, or None.
        ledger: the ledger to update, or None to leave it alone.
        doi_threshold: minimum text-layer DOI confidence; None reads
            `metadata.doi_confidence_threshold` from the config.
        logger: optional logger.

    Returns:
        One `ReciteResult` per file, in order. A file that cannot be
        re-cited is reported as skipped; it never stops the others.

    Raises:
        ValueError / NotImplementedError: the level does not exist, or is
            not built yet. Raised before any file is touched.
    """
    log = logger or LOG
    level, set_by = resolve_level(config, override)
    require_built(level)
    zotero_service = zotero_service_for(level, config, logger=log)
    online_chain = online_chain_for(level, config, zotero_service=zotero_service, logger=log)
    reference_finder = reference_finder_for(level, config, logger=log)
    if doi_threshold is None:
        doi_threshold = float((config.get("metadata") or {})
                              .get("doi_confidence_threshold", 0.85))

    results = []
    for path in json_paths:
        try:
            result = recite_document(Path(path), level, set_by, zotero_service,
                                     doi_threshold, ledger=ledger, logger=log,
                                     online_chain=online_chain,
                                     reference_finder=reference_finder)
        except Exception as error:                 # one bad file never ends the run
            result = ReciteResult(document=Path(path).name, status=SKIPPED,
                                  reason=f"{type(error).__name__}: {error}")
        log.info("%s: %s%s", result.document, result.status,
                 f" ({result.reason})" if result.reason else "")
        results.append(result)
    return results


def recite_document(json_path: Path, level: str, level_set_by: str,
                    zotero_service=None, doi_threshold: float = 0.85,
                    ledger=None, logger: Optional[logging.Logger] = None,
                    online_chain=None, reference_finder=None) -> ReciteResult:
    """
    Re-cite ONE processed document at `level`, and write it back.

    Args:
        json_path: the document's output file.
        level: the citation level to establish the record at.
        level_set_by: what decided the level (`cli`, `config`, ...).
        zotero_service: the Zotero service, at level `zotero`.
        online_chain: the Crossref/OpenAlex chain, at level `online`.
        reference_finder: the reference-list finder, at level `online`.
            Books are never asked (one list per chapter).
        doi_threshold: minimum text-layer DOI confidence to trust it.
        ledger: the ledger to update, or None.
        logger: optional logger.

    Returns:
        A `ReciteResult`. Only `recited` and `level_recorded` wrote the file.
    """
    log = logger or LOG
    name = json_path.name
    payload = _load(json_path)
    if payload is None:
        return ReciteResult(document=name, status=SKIPPED,
                            reason="the file could not be read as a document JSON")

    old = dict(payload.get("metadata_final") or {})
    level_before = (payload.get("citations") or {}).get("level")
    keep_reason = _reason_to_keep_the_record(payload)

    if keep_reason:
        new = dict(old)
        new["citation_level"] = level
        status = LEVEL_RECORDED
    else:
        reading = title_page_reading(payload)
        if reading is None:
            return ReciteResult(
                document=name, status=SKIPPED, level_before=level_before,
                reason="the file does not hold a title-page reading to derive from")
        new = derive_record(payload, reading, old, level, zotero_service,
                            doi_threshold, log, online_chain=online_chain)
        status = RECITED

    reference_data = _find_references(reference_finder, payload, new)
    assessment = _write_back(payload, json_path, new, level, level_set_by,
                             reference_data=reference_data)
    _update_ledger(ledger, payload, new, assessment, log, kept=bool(keep_reason))

    reference_list = (reference_data or {}).get("reference_list") or {}
    return ReciteResult(
        document=name, status=status, reason=keep_reason or "",
        level_before=level_before, level_after=level,
        changed={key: [old.get(key), new.get(key)] for key in REPORTED_FIELDS
                 if old.get(key) != new.get(key)},
        requires_review=assessment.requires_review,
        reference_list_status=reference_list.get("status"),
        reference_count=len((reference_data or {}).get("references") or []))


def derive_record(payload: Dict[str, Any], reading: Dict[str, Any], old: Dict[str, Any],
                  level: str, zotero_service, doi_threshold: float,
                  log: logging.Logger, online_chain=None) -> Dict[str, Any]:
    """
    Derive the bibliographic record again, exactly as a run would.

    The readings stored in the file go through `establish_metadata`, the
    same function a normal run ends in, and the result goes through the
    same model that validates a run's record. Everything the level does not
    decide (the file's hash, a chapter number, an ISBN) is carried over.
    """
    # Imported here: the strategy package pulls in the page pipelines, which
    # a re-cite has no use for until this point.
    from ..models import ExtractedMetadata
    from ..strategies.research_article.metadata_enrichment import (
        establish_metadata, sanitize_bibliographic_metadata)

    profile = payload.get("profile") or {}
    text_layer = SimpleNamespace(doi=profile.get("doi"),
                                 doi_confidence=profile.get("doi_confidence"))
    merged = establish_metadata(
        reading, title_page_text=profile.get("page_1_text") or "",
        profile=text_layer, zotero_service=zotero_service,
        doi_threshold=doi_threshold, citation_level=level,
        file_name=_file_name(payload), logger=log, online_chain=online_chain,
        next_page_text=_page_after_title_page(payload))

    carried = {key: value for key, value in old.items() if key not in DERIVED_KEYS}
    record = sanitize_bibliographic_metadata({**carried, **merged})
    return ExtractedMetadata(**record).model_dump(mode="json")


def _page_after_title_page(payload: Dict[str, Any]) -> str:
    """The stored text of the page after the title page (where a shortened
    author list is printed in full), or '' when there is none."""
    index = (payload.get("profile") or {}).get("title_page_index") or 0
    page = next((page for page in payload.get("pages") or []
                 if page.get("page_number") == index + 2), None)
    return (page or {}).get("content") or ""


def title_page_reading(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    The vision model's own reading of the title page, as stored in the file.

    Shaped as `extract_metadata_from_page1` shapes it during a run.

    Returns:
        `{title, authors, doi, year, journal}`, or None when the file holds
        no usable reading: the title page was not among the pages processed,
        its read failed, or nothing was extracted from it.
    """
    index = (payload.get("profile") or {}).get("title_page_index") or 0
    page = next((page for page in payload.get("pages") or []
                 if page.get("page_number") == index + 1), None)
    block = (page or {}).get("metadata") or {}
    if not block or block.get("processing_method") in FAILED_METHODS:
        return None
    if isinstance(block.get("metadata"), dict):          # the nested shape
        block = block["metadata"]

    authors = block.get("authors")
    reading = {"title": block.get("title"),
               "authors": authors if isinstance(authors, list) else [],
               "doi": block.get("doi"),
               "year": block.get("publication_year"),
               "journal": block.get("journal_name")}
    if not any(reading.values()):
        return None
    return reading


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _find_references(finder, payload: Dict[str, Any],
                     metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    The document's reference list from the finder, or None.

    None when there is no finder (any level but `online`) or the document is
    a book (one list per chapter, so the finder's answer would be wrong). The
    PDF is read for its own list only when it is still where the run found
    it; otherwise the list comes from the online services alone.
    """
    if finder is None:
        return None
    if (payload.get("document_info") or {}).get("document_type") == "book":
        return None
    recorded = (payload.get("profile") or {}).get("pdf_path")
    pdf_path = Path(recorded) if recorded and Path(recorded).is_file() else None
    return finder.find(metadata.get("doi"), pdf_path)


def _reason_to_keep_the_record(payload: Dict[str, Any]) -> str:
    """Why this document's record must not be derived again, or ''."""
    metadata = payload.get("metadata_final") or {}
    if metadata.get("validation_source") == "user_corrected":
        return "corrected by hand: the record is kept, only the level is recorded"
    if (payload.get("document_info") or {}).get("document_type") == "book":
        return "a book: its record comes from the front matter, only the level is recorded"
    return ""


def _write_back(payload: Dict[str, Any], json_path: Path, metadata: Dict[str, Any],
                level: str, level_set_by: str,
                reference_data: Optional[Dict[str, Any]] = None):
    """
    Put the new record into the document and write the file.

    Only the metadata, the `citations` and `mentions` blocks, the pages'
    furniture and outside-body labels and `body`, the review status and the contract stamp
    change. The file is written beside itself and then
    swapped in, so an interruption cannot leave half a document.

    `reference_data` is the finder's answer; None writes the list as "not
    requested" (so re-citing below level `online` removes an old list).

    Returns:
        The completeness assessment of the new record.
    """
    from ..document_json import (CONTRACT_VERSION, page_contents_of, status_fields,
                                 with_author_names)
    from ..utils.page_body import pages_with_body
    from .object_mentions import build_mentions_block

    info = payload.setdefault("document_info", {})
    assessment = assess_metadata_completeness(metadata, info.get("document_type"))

    # A file from an older contract gets every field the current one requires
    # (D6), since it is stamped with the current version below.
    metadata = with_author_names(metadata)
    payload["metadata_final"] = metadata
    payload["citations"] = build_citations_block(
        metadata, _file_name(payload), level=level, level_set_by=level_set_by,
        reference_data=reference_data,
        page_contents=page_contents_of(payload.get("pages") or []))
    payload["mentions"] = build_mentions_block(payload.get("pages") or [])
    payload["pages"] = pages_with_body(payload.get("pages") or [])
    payload["review_status"] = assessment.review_status
    payload["review_needed_fields"] = list(assessment.missing_fields)
    info.update(status_fields((payload.get("summary") or {}).get("errors") or [],
                              payload["pages"]))
    info["contract_version"] = CONTRACT_VERSION        # the file now carries a block
    # 1.6: a re-cited file keeps what produced its run; an older file never
    # recorded it, and null says "not recorded" rather than guessing.
    info.setdefault("provenance", None)
    # 1.7: a run from before 1.7 recorded no commit or config hash: null.
    if isinstance(info["provenance"], dict):
        info["provenance"].setdefault("source", None)
        info["provenance"].setdefault("config_sha256", None)

    result = (payload.get("profile") or {}).get("processing_result")
    if isinstance(result, dict):
        result["metadata"] = dict(metadata)
        result["citation_level"] = level
        result["citation_level_set_by"] = level_set_by

    write_json_atomically(json_path, payload)   # never half a file (D4)
    return assessment


def _update_ledger(ledger, payload: Dict[str, Any], metadata: Dict[str, Any],
                   assessment, log: logging.Logger, kept: bool = False) -> None:
    """
    Make the ledger row say what the file now says.

    A KEPT record (hand-corrected, or a book) keeps its whole row, including
    who reviewed it; only the level on it is updated. Rebuilding that row
    from the metadata would erase the fact that a person had reviewed it.
    """
    if ledger is None:
        return
    pdf_path = Path((payload.get("profile") or {}).get("pdf_path") or _file_name(payload))
    document_type = (payload.get("document_info") or {}).get("document_type")
    record = build_ledger_record(pdf_path, metadata, assessment,
                                 document_type=document_type)
    if kept:
        existing = _existing_row(ledger, record.document_id)
        if existing is None:
            return
        record = existing.model_copy(
            update={"citation_level": metadata.get("citation_level")})
    save_record(ledger, record, log)


def _existing_row(ledger, document_id: str):
    """The ledger's row for a document, or None (a lookup failure is None)."""
    try:
        return ledger.get_record(document_id)
    except Exception:
        return None


def _file_name(payload: Dict[str, Any]) -> str:
    """The PDF's file name, as the run recorded it."""
    return (payload.get("document_info") or {}).get("filename") or ""


def _load(json_path: Path) -> Optional[Dict[str, Any]]:
    """The document, or None when the file is not a document JSON."""
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or "metadata_final" not in payload:
        return None
    return payload
