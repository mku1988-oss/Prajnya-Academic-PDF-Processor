#!/usr/bin/env python3
"""
Metadata review — deciding what needs a human, and recording the answer.

For biology researchers: the pipeline reads a paper's title, authors, DOI and
journal from the page. Most of the time it gets all four. When it does not, the
document is not *wrong* — it is *incomplete*, and a library full of incomplete
records is a library you cannot cite from. So every processed document is
assessed against those four fields, flagged if anything is missing, and written
to the semantic ledger with that flag attached. A human can then correct the
flagged ones — once — and the correction is permanent.

Three things live here, none of which touch a terminal:

    assess_metadata_completeness   is this record usable as it stands?
    build_ledger_record            turn a processed document into a ledger row
    regenerate_document_json       re-write the output JSON after a correction

The *conversation* — showing a record and asking what to change — is in
`review_prompt.py`, exactly as `book_prompt.py` is kept apart from `batch.py`.
The GUI replaces the conversation and reuses everything in this file.

**One deliberate difference from V5.** V5's `regenerate_json_from_ledger` built
a fresh JSON from the ledger row alone, which set `"pages": []` — correcting a
typo in the title threw away every page of extracted content. Here regeneration
*patches* the existing file: the metadata is replaced, everything else is left
exactly as it was.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .ledger.semantic_ledger import LedgerRecord, SemanticLedger, generate_hash

LOG = logging.getLogger(__name__)

#: What a research article must have. The user's original rule, inherited from
#: V5, and still exactly right for a paper: a citation needs all four.
REQUIRED_FIELDS = ("title", "authors", "doi", "journal")

#: What each kind of document must have to be citable.
#:
#: A tuple inside the tuple is an EITHER-OR: any one of those fields satisfies
#: it. That is not laxity, it is bibliography. An edited volume is cited by its
#: editors and a monograph by its authors — demanding both would flag every
#: book ever published, which is precisely what the flat article rule did on
#: the session-9 sample run: a Springer volume holding editors, publisher, ISBN
#: and year was marked incomplete for "missing authors, journal".
#:
#: What each rule deliberately does NOT require:
#:   book         — no DOI. Plenty of books have none, and the ISBN is the
#:                  identifier that matters.
#:   book_chapter — a container (the book or series it appears in), because a
#:                  chapter citation without one cannot be resolved. On the
#:                  sample run this correctly caught a real extraction gap.
COMPLETENESS_RULES: Dict[str, Tuple[Any, ...]] = {
    "research_article": REQUIRED_FIELDS,
    "book": ("title", ("editors", "authors"), ("publisher", "isbn"), "year"),
    "book_chapter": ("title", "authors", ("journal", "publisher"), "doi"),
}

#: Used when the document type is unknown. An unclassified document is treated
#: as a paper, which is what the overwhelming majority are.
DEFAULT_RULE = REQUIRED_FIELDS

#: Below this, the merge/Zotero agreement was weak enough to want human eyes.
LOW_CONFIDENCE = 0.75


class MetadataAssessment(BaseModel):
    """What is missing from a record, and whether that needs a human."""

    is_complete: bool = Field(True, description="All four required fields present")
    missing_fields: List[str] = Field(default_factory=list)
    suspicious_items: List[str] = Field(
        default_factory=list,
        description="Present but doubtful — flagged authors, low confidence")
    confidence: float = Field(1.0, description="Validation confidence, 0.0-1.0")
    requires_review: bool = Field(False)

    @property
    def review_status(self) -> str:
        """The ledger's status word for a record in this state."""
        return "provisional" if self.requires_review else "reviewed"

    def describe(self) -> str:
        """One line for a log or a batch summary."""
        if not self.requires_review:
            return "complete"
        parts = []
        if self.missing_fields:
            parts.append("missing " + ", ".join(self.missing_fields))
        if self.suspicious_items:
            parts.append("suspect " + ", ".join(self.suspicious_items))
        return "; ".join(parts) or "needs review"


def assess_metadata_completeness(metadata: Dict[str, Any],
                                 document_type: Any = None) -> MetadataAssessment:
    """
    Decide whether a document's metadata is usable as it stands.

    For biology researchers: what a record needs depends on what it IS. A paper
    needs title, authors, DOI and journal. A book needs its title, whoever
    edited or wrote it, a publisher or ISBN, and a year — it has no journal and
    frequently no DOI. A chapter needs its own authors plus the book it sits in.
    Judging all three by the paper's rule marks every book incomplete forever,
    which teaches people to ignore the flag.

    Args:
        metadata: the merged bibliographic dict (what `enrich_metadata`
            returns, or an `ExtractedMetadata.model_dump()`).
        document_type: `DocumentType`, its string value, or None. An unknown
            type is judged as a research article — what most documents are.

    Returns:
        A `MetadataAssessment`. An unusable input — None, or not a dict —
        assesses as needing review rather than raising: a document that cannot
        be assessed is exactly the kind a human should look at.

    Example:
        >>> assess_metadata_completeness({"title": "T", "authors": ["A"]}).missing_fields
        ['doi', 'journal']
        >>> assess_metadata_completeness({"title": "T", "editors": ["E"],
        ...                               "publisher": "Springer", "year": 2021},
        ...                              "book").is_complete
        True
    """
    rule = rule_for(document_type)

    if not isinstance(metadata, dict):
        return MetadataAssessment(is_complete=False,
                                  missing_fields=[_requirement_name(r) for r in rule],
                                  confidence=0.0, requires_review=True)

    missing = [_requirement_name(requirement) for requirement in rule
               if not _requirement_met(requirement, metadata)]
    suspicious = _suspicious_items(metadata)
    confidence = _confidence_of(metadata)

    if confidence < LOW_CONFIDENCE:
        suspicious.append(f"low_confidence({confidence:.2f})")

    return MetadataAssessment(
        is_complete=not missing,
        missing_fields=missing,
        suspicious_items=suspicious,
        confidence=confidence,
        requires_review=bool(missing or suspicious),
    )


def build_ledger_record(pdf_path: Path, metadata: Dict[str, Any],
                        assessment: Optional[MetadataAssessment] = None,
                        document_type: Any = None) -> LedgerRecord:
    """
    Turn a processed document into the row the ledger stores.

    For biology researchers: this is the permanent record — one line per
    document, keyed by the file's SHA-256 so renaming the PDF does not create
    a duplicate. It carries the review flag, so the next run knows whether a
    human has already looked at this one.

    Args:
        pdf_path: the source PDF.
        metadata: the merged bibliographic dict.
        assessment: a completed assessment; computed from `metadata` if absent.
        document_type: what the document is. Stored on the record, so a later
            `--review` reassesses it by the SAME rule that flagged it — a book
            reopened months later must not suddenly be judged as a paper.

    Returns:
        A `LedgerRecord` ready for `ledger.update_record`.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    assessment = assessment or assess_metadata_completeness(metadata, document_type)

    source_hash = metadata.get("source_hash") or _hash_of(pdf_path)

    return LedgerRecord(
        document_id=source_hash,
        pdf_filename=pdf_path.name,
        doi=metadata.get("doi"),
        title=metadata.get("title"),
        authors=_as_author_list(metadata.get("authors")),
        journal=metadata.get("journal"),
        year=_as_year(metadata.get("year")),
        editors=_as_author_list(metadata.get("editors")),
        publisher=metadata.get("publisher"),
        document_type=_type_name(document_type),
        validation_confidence=assessment.confidence,
        validation_source=str(metadata.get("validation_source") or "vlm"),
        requires_review=assessment.requires_review,
        review_needed_fields=list(assessment.missing_fields),
        review_status=assessment.review_status,
        last_updated=datetime.utcnow().isoformat(),
        retrieval_provenance=_provenance_of(metadata),
        source_hash=source_hash,
    )


def save_record(ledger: Optional[SemanticLedger], record: LedgerRecord,
                logger: Optional[logging.Logger] = None) -> bool:
    """
    Write one record to the ledger and commit it.

    Args:
        ledger: the ledger, or None (nothing is written).
        record: what to store.
        logger: optional logger.

    Returns:
        Whether the record was persisted. A ledger failure is logged and
        swallowed — losing the bookkeeping row is bad, but losing the whole
        run's extracted content because a CSV was locked is worse.
    """
    log = logger or LOG
    if ledger is None:
        return False

    try:
        ledger.update_record(record)
        ledger.commit()
        return True
    except Exception as error:
        log.warning("Could not write %s to the ledger: %s", record.pdf_filename, error)
        return False


def records_needing_review(ledger: SemanticLedger) -> List[LedgerRecord]:
    """
    Every ledger row a human still needs to look at, oldest file first.

    Uses the stored `requires_review` flag when it is set, and otherwise falls
    back to inspecting the fields themselves — so rows written by V5, or by an
    older V6 that never set the flag, are still caught.
    """
    try:
        records = list(ledger.get_all_records())
    except Exception as error:
        LOG.error("Could not enumerate the ledger: %s", error)
        return []

    return [record for record in records if record_needs_review(record)]


def record_needs_review(record: LedgerRecord) -> bool:
    """
    True when a stored row is flagged, or is missing something its kind needs.

    The fallback judges the row by ITS OWN document type, so a book with
    editors and a publisher is not re-flagged for lacking a journal it was
    never going to have. Rows written before the type was stored have none, and
    are judged as articles — the strict rule, which errs towards asking.
    """
    if record.review_status == "reviewed":
        return False
    if record.requires_review:
        return True

    return not all(_requirement_met(requirement, record_as_metadata(record))
                   for requirement in rule_for(record.document_type))


def apply_corrections(record: LedgerRecord, values: Dict[str, Any],
                      reviewed_by: str = "cli") -> LedgerRecord:
    """
    Store a human's corrections on a record, and re-decide whether it is done.

    For biology researchers: this is what "I have fixed this entry" means. It
    writes the corrected fields, re-checks whether the citation is now complete,
    and marks the record `reviewed` or leaves it `provisional`.

    **Why this is here and not in the prompt module.** Two front ends now
    collect corrections — a terminal asking questions, and a graphical form —
    and they must not each decide separately what "reviewed" means. Collecting
    the values is a user-interface job and differs completely between the two;
    deciding what the values *mean* is not, and is this function. Duplicating
    it is how the two front ends start disagreeing about whether a record is
    still flagged.

    Completeness alone decides the flag, deliberately. The pipeline's own low
    confidence is what asked for a human in the first place; once one has
    looked, that doubt is answered, and re-raising it would flag the record
    forever no matter how many times it was corrected. The rule applied is the
    one for the record's OWN document type — a book corrected months later must
    not be reassessed as a paper and re-flagged for a journal it never had.

    Args:
        record: the row to correct.
        values: the corrected values, keyed by `EDITABLE_FIELDS` name. Missing
            keys clear the field, because a review that blanks a wrong title
            must be able to say so.
        reviewed_by: who did the reviewing — `"cli"`, `"gui"`, a username. It
            was hardcoded to `"cli"` when only a terminal existed.

    Returns:
        A new `LedgerRecord`; the original is not mutated.
    """
    updated = record.model_copy(update={
        "title": values.get("title"),
        "authors": _as_string_list(values.get("authors")),
        "editors": _as_string_list(values.get("editors")),
        "journal": values.get("journal"),
        "publisher": values.get("publisher"),
        "doi": values.get("doi"),
        "year": values.get("year"),
    })

    assessment = assess_metadata_completeness(record_as_metadata(updated),
                                              updated.document_type)
    return updated.model_copy(update={
        "requires_review": not assessment.is_complete,
        "review_needed_fields": list(assessment.missing_fields),
        "review_status": "reviewed" if assessment.is_complete else "provisional",
        "last_reviewed_by": reviewed_by,
    })


def record_as_metadata(record: LedgerRecord) -> Dict[str, Any]:
    """A ledger row as the flat dict the assessment and the prompt speak."""
    return {
        "title": record.title,
        "authors": list(record.authors),
        "editors": list(record.editors),
        "doi": record.doi,
        "journal": record.journal,
        "publisher": record.publisher,
        "year": record.year,
        "validation_confidence": record.validation_confidence,
        "validation_source": record.validation_source,
        "source_hash": record.source_hash,
    }


def apply_record_to_metadata(record: LedgerRecord,
                             metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overlay a corrected ledger row onto an existing metadata block.

    Only the fields a human can edit are replaced; anything else the pipeline
    produced (provenance, ISBN, publisher, page-level detail) is preserved.

    `validation_confidence` is deliberately **left alone**. It records how well
    the pipeline's sources agreed with each other, which a later hand
    correction does not retroactively change — and the parity harness reads
    that number. What a correction does change is the *source*: a record
    someone has edited says so.
    """
    patched = dict(metadata) if isinstance(metadata, dict) else {}
    patched.update({
        "title": record.title,
        "authors": list(record.authors),
        "doi": record.doi,
        "journal": record.journal,
        "year": record.year,
        "source_hash": record.source_hash,
    })
    # Books and chapters: only overwrite when the ledger actually holds a
    # value, so a correction to a paper never blanks a book field the record
    # simply does not track.
    if record.editors:
        patched["editors"] = list(record.editors)
    if record.publisher:
        patched["publisher"] = record.publisher
    if record.last_reviewed_by:
        patched["validation_source"] = "user_corrected"
    return patched


def regenerate_document_json(json_path: Path, record: LedgerRecord,
                             logger: Optional[logging.Logger] = None) -> bool:
    """
    Re-write one output JSON so it carries the corrected metadata.

    For biology researchers: after you fix a title by hand, the JSON on disk
    still holds the old one. This puts your correction into the file — and
    ONLY the metadata. Every page of extracted content, every figure and every
    table stays exactly as processed. (V5 rebuilt the file from the ledger row
    alone and lost all of it.)

    Args:
        json_path: an existing document JSON, as written by `_write_result`.
        record: the corrected ledger row.
        logger: optional logger.

    Returns:
        Whether the file was updated. A missing or unreadable file is reported
        and skipped, never overwritten with a stub.
    """
    log = logger or LOG

    if not json_path.is_file():
        log.warning("No output JSON to regenerate at %s", json_path)
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as error:
        log.error("Could not read %s (%s) — leaving it alone.", json_path, error)
        return False

    payload["metadata_final"] = apply_record_to_metadata(
        record, payload.get("metadata_final") or {})
    payload["review_status"] = record.review_status
    payload["review_needed_fields"] = list(record.review_needed_fields)
    _patch_nested_metadata(payload, record)

    try:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except Exception as error:
        log.error("Could not write %s: %s", json_path, error)
        return False

    log.info("Regenerated %s from the ledger.", json_path.name)
    return True


def regenerate_from_ledger(outputs: Dict[Path, Path], ledger: SemanticLedger,
                           logger: Optional[logging.Logger] = None) -> int:
    """
    Re-write a batch's output files after a review session.

    Args:
        outputs: {pdf path -> the JSON that was written for it}.
        ledger: where the corrections live.
        logger: optional logger.

    Returns:
        How many files were successfully updated.
    """
    log = logger or LOG
    updated = 0

    for pdf_path, json_path in outputs.items():
        record = _record_for(ledger, pdf_path, log)
        if record is None:
            continue
        if regenerate_document_json(Path(json_path), record, log):
            updated += 1

    return updated


def rule_for(document_type: Any) -> Tuple[Any, ...]:
    """
    The completeness rule for a document type.

    Args:
        document_type: a `DocumentType`, its string value, or None.

    Returns:
        The rule's requirements. An unrecognised or absent type falls back to
        the research-article rule — the strictest of the three, so an unknown
        document is over-flagged rather than waved through.
    """
    name = getattr(document_type, "value", document_type)
    return COMPLETENESS_RULES.get(str(name), DEFAULT_RULE)


def describe_rule(document_type: Any) -> str:
    """The rule in words, for a prompt or a log line."""
    return ", ".join(_requirement_name(r).replace("_or_", " or ")
                     for r in rule_for(document_type))


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _requirement_met(requirement: Any, metadata: Dict[str, Any]) -> bool:
    """One requirement, satisfied by any of its alternatives."""
    if isinstance(requirement, (tuple, list)):
        return any(_has_value(metadata.get(name)) for name in requirement)
    return _has_value(metadata.get(requirement))


def _requirement_name(requirement: Any) -> str:
    """A requirement's name for the review list, e.g. 'editors_or_authors'."""
    if isinstance(requirement, (tuple, list)):
        return "_or_".join(requirement)
    return str(requirement)


def _has_value(value: Any) -> bool:
    """Present and non-empty — an empty list of authors is not authors."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _suspicious_items(metadata: Dict[str, Any]) -> List[str]:
    """Fields that are present but which the pipeline itself doubted."""
    provenance = metadata.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    return ["authors"] if provenance.get("suspicious_authors") else []


def _confidence_of(metadata: Dict[str, Any]) -> float:
    """Validation confidence, defaulting to 0.0 when it is absent or unusable."""
    provenance = metadata.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}

    value = provenance.get("validation_confidence",
                           metadata.get("validation_confidence"))
    try:
        return float(value)
    except (TypeError, ValueError):
        # Absent confidence is not evidence of quality; treat it as unknown,
        # which flags the record rather than passing it silently.
        return 0.0


def _provenance_of(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """The provenance block, always a dict so the ledger can serialize it."""
    provenance = metadata.get("provenance")
    return provenance if isinstance(provenance, dict) else {}


def _as_string_list(value: Any) -> List[str]:
    """
    Names as a list, however the caller supplied them.

    A terminal hands back one comma-separated string; a graphical form hands
    back a list. Both are legitimate, so both are accepted here rather than
    each front end normalizing its own way.
    """
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _as_author_list(authors: Any) -> List[str]:
    """Authors as a list of strings, whatever shape they arrived in."""
    if isinstance(authors, str):
        return [part.strip() for part in authors.split(",") if part.strip()]
    if isinstance(authors, (list, tuple)):
        return [str(name).strip() for name in authors if str(name).strip()]
    return []


def _type_name(document_type: Any) -> Optional[str]:
    """A document type as the plain string the ledger stores."""
    name = getattr(document_type, "value", document_type)
    return str(name) if name else None


def _as_year(value: Any) -> Optional[int]:
    """A four-digit year, or None — the ledger would rather have nothing."""
    try:
        return int(str(value).strip()[:4])
    except (TypeError, ValueError):
        return None


def _hash_of(pdf_path: Path) -> str:
    """The file's SHA-256, or a path-derived fallback if it cannot be read."""
    try:
        return generate_hash(str(pdf_path))
    except Exception as error:
        LOG.warning("Could not hash %s (%s); using the path as identity.",
                    pdf_path.name, error)
        return f"path:{pdf_path.name}"


def _patch_nested_metadata(payload: Dict[str, Any], record: LedgerRecord) -> None:
    """
    Keep the un-flattened `profile` block in step with `metadata_final`.

    The canonical JSON carries the same metadata twice — once flattened for the
    parity harness, once nested inside the profile so nothing is lost. A
    correction that updated only one of them would leave the file
    self-contradicting.
    """
    profile = payload.get("profile")
    if not isinstance(profile, dict):
        return
    result = profile.get("processing_result")
    if not isinstance(result, dict):
        return
    result["metadata"] = apply_record_to_metadata(record,
                                                  result.get("metadata") or {})


def _record_for(ledger: SemanticLedger, pdf_path: Path,
                log: logging.Logger) -> Optional[LedgerRecord]:
    """The ledger row for a PDF, or None with a reason logged."""
    try:
        record = ledger.get_record(generate_hash(str(pdf_path)))
    except Exception as error:
        log.warning("Ledger lookup failed for %s: %s", Path(pdf_path).name, error)
        return None

    if record is None:
        log.warning("No ledger record for %s — nothing to regenerate from.",
                    Path(pdf_path).name)
    return record
