#!/usr/bin/env python3
"""
Reviewing metadata at a terminal — the conversation, and nothing else.

For biology researchers: this is the screen that shows you a record the
pipeline was not confident about and lets you fix it — title, authors, editors,
journal, publisher, DOI and year, edited one at a time, then saved.

It offers every field ANY document type can require, not just a paper's: a book
is flagged for missing editors or a publisher, so those have to be fixable
here. A field that can be demanded and not corrected is a permanent flag.

This file is deliberately thin, and separate from `review.py`, for the same
reason `book_prompt.py` is separate from `batch.py`: **asking is a capability
the caller supplies.** The CLI supplies a terminal; the GUI will supply a
dialog; a cron job supplies nothing at all and the flagged records simply wait
in the ledger until someone runs `--review`. None of the decision logic lives
here, so none of it has to be rewritten for the GUI.

Every prompt treats EOF, a closed pipe, or Ctrl-C as "stop asking" rather than
as an answer — a batch redirected from a script must never hang, and must never
have a stray newline interpreted as a correction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .ledger.semantic_ledger import LedgerRecord, SemanticLedger
from .review import (
    MetadataAssessment,
    apply_corrections,
    assess_metadata_completeness,
    record_as_metadata,
    record_needs_review,
    records_needing_review,
    save_record,
)

LOG = logging.getLogger(__name__)

#: The fields a human may edit, in the order they are shown.
#:
#: This list must cover EVERY field any completeness rule can require, or a
#: document gets flagged for something the review screen cannot show — so
#: `editors` and `publisher` are here for books and chapters, even though a
#: paper never uses them.
EDITABLE_FIELDS = ("title", "authors", "editors", "journal", "publisher",
                   "doi", "year")

#: The menu number that saves and moves on. It follows the field list, so
#: adding a field never silently reassigns the "done" key.
DONE_CHOICE = str(len(EDITABLE_FIELDS) + 1)

#: Consecutive answers that pick nothing before we conclude nobody is there.
MAX_UNRECOGNISED_ANSWERS = 5


def review_metadata(metadata: Dict[str, Any],
                    assessment: Optional[MetadataAssessment] = None,
                    input_fn=input, output_fn=print,
                    document_type: Any = None) -> Dict[str, Any]:
    """
    Show one document's metadata and let the user correct it.

    For biology researchers: used right after a document is processed, when
    something required is missing. You see what was found, fix what is wrong,
    and the correction goes into both the ledger and the output JSON.

    Args:
        metadata: the merged bibliographic dict.
        assessment: what was found lacking; recomputed if not supplied.
        input_fn / output_fn: injected so the conversation can be tested.

    Returns:
        The metadata, corrected. If nobody can answer — EOF, a closed pipe —
        the input is returned unchanged, which leaves the record flagged for
        `--review` later rather than silently accepting it.
    """
    assessment = assessment or assess_metadata_completeness(metadata, document_type)
    _show_assessment(assessment, output_fn)

    edited = _edit_loop(dict(metadata), input_fn, output_fn, document_type)
    return edited if edited is not None else metadata


def review_ledger(ledger: SemanticLedger, input_fn=input, output_fn=print,
                  logger: Optional[logging.Logger] = None) -> int:
    """
    Work through every flagged record in the ledger — V5's "Mode D".

    For biology researchers: this is `--review`. It finds every document whose
    metadata is incomplete, shows them one at a time, and saves each correction
    as you go, so stopping halfway keeps the work you have done.

    Args:
        ledger: the ledger to review.
        input_fn / output_fn: injected for testing.
        logger: optional logger.

    Returns:
        How many records were corrected and saved.
    """
    log = logger or LOG
    pending = records_needing_review(ledger)

    if not pending:
        output_fn("Nothing in the ledger needs review.")
        return 0

    output_fn("")
    output_fn(f"{len(pending)} record(s) need review.")

    corrected = 0
    for position, record in enumerate(pending, start=1):
        output_fn("")
        output_fn("=" * 72)
        output_fn(f"  [{position}/{len(pending)}] {record.pdf_filename or record.document_id}")
        output_fn("=" * 72)

        updated = review_record(record, input_fn, output_fn)
        if updated is None:
            output_fn("  Stopping here - the rest stay flagged for next time.")
            break
        if save_record(ledger, updated, log):
            corrected += 1

    output_fn("")
    output_fn(f"Review finished: {corrected} record(s) saved.")
    return corrected


def review_record(record: LedgerRecord, input_fn=input,
                  output_fn=print) -> Optional[LedgerRecord]:
    """
    Edit one ledger row.

    Args:
        record: the row to correct.
        input_fn / output_fn: injected for testing.

    Returns:
        The corrected record, or **None** when the conversation ended (EOF or
        Ctrl-C) — which the caller reads as "stop", not as "no changes".
    """
    values = {name: getattr(record, name, None) for name in EDITABLE_FIELDS}
    edited = _edit_loop(values, input_fn, output_fn, record.document_type)
    if edited is None:
        return None

    # What the corrections MEAN — whether the record is now complete, and so
    # whether it is still flagged — is decided in `review.apply_corrections`,
    # not here. This module owns the conversation; that one owns the verdict.
    # The GUI cannot reuse a numbered-menu conversation, but it must reach the
    # same verdict, and it will because it calls the same function.
    return apply_corrections(record, edited, reviewed_by="cli")


def offer_deferred_review(pending: Sequence[Path], ledger: SemanticLedger,
                          input_fn=input, output_fn=print,
                          logger: Optional[logging.Logger] = None) -> bool:
    """
    At the end of a batch, offer to review what needs it.

    For biology researchers: a batch never stops to ask questions halfway —
    that is the whole point of leaving one running. Instead everything that
    needs your attention is collected and offered here, once, at the end.

    Args:
        pending: the PDFs whose records were flagged.
        ledger: where the flagged records live.
        input_fn / output_fn: injected for testing.
        logger: optional logger.

    Returns:
        Whether a review session actually ran. Declining, or being unable to
        answer, returns False and leaves everything flagged — `--review` picks
        it up whenever there is time.
    """
    if not pending:
        return False

    output_fn("")
    output_fn("=" * 72)
    output_fn(f"  {len(pending)} document(s) need metadata review:")
    for path in pending:
        output_fn(f"    - {Path(path).name}")
    output_fn("=" * 72)

    try:
        answer = input_fn("  Review them now? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        answer = ""

    if answer not in ("y", "yes"):
        output_fn("  Skipped. Run with --review whenever you like; the output "
                  "JSON says 'provisional' until then.")
        return False

    review_ledger(ledger, input_fn, output_fn, logger)
    return True


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _show_assessment(assessment: MetadataAssessment, output_fn) -> None:
    """Say why we are asking before asking."""
    output_fn("")
    output_fn("-" * 72)
    output_fn(f"  This document's metadata needs review: {assessment.describe()}")
    output_fn("-" * 72)


def _edit_loop(values: Dict[str, Any], input_fn, output_fn,
               document_type: Any = None) -> Optional[Dict[str, Any]]:
    """
    The five-field editing menu, shared by document and ledger review.

    Returns the edited values, or None if the conversation could not continue.
    Choosing "done" while something required is still missing asks for
    confirmation — the record can be saved incomplete, but not by accident.

    A run of unrecognised answers ends the conversation rather than looping
    forever. A person mistypes once or twice; something that answers five times
    without ever picking an option is not a person, and a menu nobody can
    satisfy must not spin.
    """
    unrecognised = 0

    while True:
        _show_values(values, output_fn)

        choice = _ask(input_fn,
                      f"\n  Choose a field to edit (1-{DONE_CHOICE}): ")
        if choice is None:
            return None

        if choice == DONE_CHOICE:
            if _confirm_done(values, input_fn, output_fn, document_type):
                return values
            continue

        field = _field_for(choice)
        if field is None:
            unrecognised += 1
            if unrecognised >= MAX_UNRECOGNISED_ANSWERS:
                output_fn("  Nothing here is choosing an option - stopping.")
                return None
            output_fn(f"  Not one of the options - pick 1-{DONE_CHOICE}.")
            continue

        unrecognised = 0
        if not _edit_one(values, field, input_fn, output_fn):
            return None


def _show_values(values: Dict[str, Any], output_fn) -> None:
    """Print the current five fields, numbered."""
    output_fn("")
    output_fn("  Current values:")
    for index, name in enumerate(EDITABLE_FIELDS, start=1):
        shown = values.get(name)
        if isinstance(shown, (list, tuple)):
            shown = ", ".join(str(item) for item in shown) or None
        output_fn(f"    {index}) {name.title():8s}: {shown}")
    output_fn(f"    {DONE_CHOICE}) Done - save and continue")


def _field_for(choice: str) -> Optional[str]:
    """Menu number to field name."""
    if choice.isdigit() and 1 <= int(choice) <= len(EDITABLE_FIELDS):
        return EDITABLE_FIELDS[int(choice) - 1]
    return None


def _edit_one(values: Dict[str, Any], field: str, input_fn, output_fn) -> bool:
    """
    Ask for one field's new value. Returns False if the conversation ended.

    Blank keeps the current value for every field except DOI, where blank
    clears it — a wrong DOI fetches the wrong paper from Zotero, so being able
    to remove one is worth the inconsistency, and the prompt says so.
    """
    prompt = _prompt_for(field)
    answer = _ask(input_fn, prompt)
    if answer is None:
        return False

    if field == "doi":
        values["doi"] = answer or None
        return True

    if not answer:
        return True

    if field == "authors":
        values["authors"] = [name.strip() for name in answer.split(",") if name.strip()]
    elif field == "year":
        year = _as_year(answer)
        if year is None:
            output_fn("  That is not a four-digit year - leaving it unchanged.")
        else:
            values["year"] = year
    else:
        values[field] = answer

    return True


def _prompt_for(field: str) -> str:
    """The wording for one field's prompt."""
    if field == "authors":
        return "  New authors (comma-separated, blank keeps): "
    if field == "doi":
        return "  New DOI (blank CLEARS it): "
    if field == "year":
        return "  New year (YYYY, blank keeps): "
    return f"  New {field} (blank keeps): "


def _confirm_done(values: Dict[str, Any], input_fn, output_fn,
                  document_type: Any = None) -> bool:
    """
    Warn before saving something still incomplete.

    Judged by the document's OWN kind, like every other check here — warning
    that a book is 'missing journal' would be nonsense advice to act on.
    """
    assessment = assess_metadata_completeness(values, document_type)
    if assessment.is_complete:
        return True

    output_fn(f"  Still missing: {', '.join(assessment.missing_fields)}")
    answer = _ask(input_fn, "  Save it anyway? (y/N): ")
    return (answer or "").lower() in ("y", "yes")


def _ask(input_fn, prompt: str) -> Optional[str]:
    """One question. None means the conversation cannot continue."""
    try:
        return input_fn(prompt).strip()
    except (EOFError, KeyboardInterrupt, OSError):
        return None


def _as_year(value: Any) -> Optional[int]:
    """A four-digit year, or None."""
    text = str(value).strip()
    return int(text) if text.isdigit() and len(text) == 4 else None


def needs_review_after_save(record: LedgerRecord) -> bool:
    """Thin re-export so callers need only import from this module."""
    return record_needs_review(record)
