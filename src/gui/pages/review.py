#!/usr/bin/env python3
"""
Review Queue — the human step for a record the pipeline couldn't fully
trust on its own. S9.

For biology researchers: not every paper's metadata comes out complete.
When a title, author, DOI, or journal is missing — or a book's editors or
publisher — the pipeline flags the record rather than guessing, and this
screen is where that flag gets resolved: read what's missing, type the
correction, save it once.

**This screen has almost no logic of its own — deliberately.** Every
decision (what counts as complete for a book vs. an article vs. a
chapter, what "reviewed" means once a human has looked, how a correction
lands on both the ledger and the document JSON) lives in
`src/processing/review.py`, because a terminal (`--review`) and this GUI
form must never each decide "is this record done?" a different way. This
module only collects the correction and hands it to
`review.apply_corrections` — the same function `review_prompt.py`'s
terminal conversation calls (`docs/gui_plan.md` section 4).

**Why the document JSON path is a guess, not a lookup.** The ledger
stores only a bare filename (see `ledger.py`'s docstring on "Open PDF"
for why), but `resolve_document_output`'s own rule for the default case
(no `--out`) is `documents/<pdf's stem>.v6.json` — the exact same rule
`document_json.py` used to write it. Recomputing that one rule here,
rather than scanning every JSON for a filename match, keeps this screen
honest about what it actually knows: a record processed with a custom
`--out` has no findable JSON here, and the screen says so instead of
guessing further.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from nicegui import run, ui

from src.processing import review
from src.processing.ledger.semantic_ledger import LedgerRecord
from src.processing.paths import DEFAULT_OUTPUT_ROOT
from src.processing.runner import open_ledger

from .. import theme

logger = logging.getLogger("prajnya_studio.review")


def mount() -> None:
    ui_state: Dict[str, Any] = {"ledger": None, "records": []}
    dialog_container = ui.column()

    with ui.column().classes("w-full gap-4"):
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Review Queue").style(
                f'font-family: var(--prajnya-font-display); font-size: 1.5rem; '
                f'font-weight: 600; color: {theme.var("indigo")};'
            )
            ui.space()
            refresh_button = ui.button("Refresh", icon="refresh").props("flat")

        summary_label = ui.label("").classes("text-sm").style(
            f'color: {theme.var("text_muted")};'
        )
        list_container = ui.column().classes("w-full gap-3")
        empty_state = ui.column().classes("w-full items-center gap-2 p-12")
        with empty_state:
            ui.icon("task_alt", size="4em").classes("text-gray-300")
            ui.label("Nothing needs review").classes("text-lg").style(
                f'color: {theme.var("text_muted")};'
            )

    def render() -> None:
        records = ui_state["records"]
        summary_label.text = (f"{len(records)} record(s) need review" if records
                              else "Every processed record is complete.")
        list_container.clear()
        has_any = bool(records)
        list_container.set_visibility(has_any)
        empty_state.set_visibility(not has_any)
        with list_container:
            for record in records:
                _render_review_card(record, ui_state, dialog_container, render)

    async def load() -> None:
        ledger, records = await run.io_bound(_load_pending_sync)
        ui_state["ledger"] = ledger
        ui_state["records"] = records
        render()

    refresh_button.on_click(load)
    ui.timer(0.0, load, once=True)


# --- data loading, run off the event loop via run.io_bound -------------------

def _load_pending_sync():
    ledger = open_ledger(logger, create=True)
    records = review.records_needing_review(ledger) if ledger is not None else []
    records.sort(key=lambda r: r.last_updated or "")
    return ledger, records


def _document_json_path(record: LedgerRecord) -> Optional[Path]:
    if not record.pdf_filename:
        return None
    stem = Path(record.pdf_filename).stem
    path = DEFAULT_OUTPUT_ROOT / "documents" / f"{stem}.v6.json"
    return path if path.is_file() else None


# --- the queue list ----------------------------------------------------------

def _render_review_card(record: LedgerRecord, ui_state: Dict[str, Any],
                        dialog_container: ui.column, render) -> None:
    with ui.card().classes("w-full").style(
        f'border: 1px solid {theme.var("warning")};'
    ):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-1"):
                ui.label(record.title or record.pdf_filename or "Untitled").style(
                    f'font-weight: 600; color: {theme.var("indigo")};'
                )
                ui.label(_reason(record)).classes("text-sm").style(
                    f'color: {theme.var("text_muted")};'
                )
                ui.label(f"Rule: {review.describe_rule(record.document_type)}").classes(
                    "text-xs"
                ).style(f'color: {theme.var("gold")};')
            ui.button("Review", icon="edit",
                     on_click=lambda: _open_review_dialog(record, ui_state, dialog_container, render)
                     ).props("color=red-9")


def _reason(record: LedgerRecord) -> str:
    if record.review_needed_fields:
        return "Missing: " + ", ".join(record.review_needed_fields)
    return "Flagged for review"


# --- the correction form -----------------------------------------------------

def _open_review_dialog(record: LedgerRecord, ui_state: Dict[str, Any],
                        container: ui.column, render) -> None:
    with container:
        with ui.dialog() as dialog, ui.card().classes("w-[560px] p-6 gap-2"):
            ui.label(record.title or record.pdf_filename or "Untitled").style(
                f'font-size: 1.15rem; font-weight: 600; color: {theme.var("indigo")};'
            )
            ui.label(f"Rule: {review.describe_rule(record.document_type)}").classes(
                "text-xs"
            ).style(f'color: {theme.var("gold")};')
            ui.label(_reason(record)).classes("text-sm").style(
                f'color: {theme.var("error")};'
            )

            title_input = ui.input("Title", value=record.title or "").classes("w-full")
            authors_input = ui.textarea(
                "Authors (one per line)", value="\n".join(record.authors)
            ).classes("w-full").props("rows=3")
            editors_input = ui.textarea(
                "Editors (one per line, books only)", value="\n".join(record.editors)
            ).classes("w-full").props("rows=2")
            doi_input = ui.input("DOI", value=record.doi or "").classes("w-full")
            journal_input = ui.input("Journal", value=record.journal or "").classes("w-full")
            publisher_input = ui.input("Publisher", value=record.publisher or "").classes("w-full")
            year_input = ui.number("Year", value=record.year, min=1900, max=2100).classes("w-full")

            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Save", on_click=lambda: _save_correction(
                    record, ui_state, dialog, render,
                    title_input.value, authors_input.value, editors_input.value,
                    doi_input.value, journal_input.value, publisher_input.value,
                    year_input.value,
                ))

        dialog.open()


def _save_correction(record: LedgerRecord, ui_state: Dict[str, Any], dialog, render,
                     title: str, authors_text: str, editors_text: str, doi: str,
                     journal: str, publisher: str, year: Optional[float]) -> None:
    values = {
        "title": title.strip(),
        "authors": _lines(authors_text),
        "editors": _lines(editors_text),
        "doi": doi.strip(),
        "journal": journal.strip(),
        "publisher": publisher.strip(),
        "year": int(year) if year else None,
    }
    updated = review.apply_corrections(record, values, reviewed_by="gui")

    ledger = ui_state["ledger"]
    if not review.save_record(ledger, updated, logger):
        ui.notify("Could not save to the ledger — see the log for details.",
                 type="negative")
        return

    json_path = _document_json_path(updated)
    if json_path is not None:
        review.regenerate_document_json(json_path, updated, logger)
    else:
        ui.notify("Saved to the ledger, but no output JSON was found to update "
                 "(a custom --out location, or the document was never processed here).",
                 type="warning", timeout=6000)

    _replace_or_drop(ui_state, record, updated)
    dialog.close()
    ui.notify(f'{"Reviewed" if updated.review_status == "reviewed" else "Updated"}: '
             f'"{updated.title or updated.pdf_filename}"', type="positive")
    render()


def _lines(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def _replace_or_drop(ui_state: Dict[str, Any], original: LedgerRecord,
                     updated: LedgerRecord) -> None:
    """Keep the queue in sync without a round trip to the ledger: still
    flagged -> show the corrected version in place; resolved -> drop it."""
    if review.record_needs_review(updated):
        ui_state["records"] = [updated if r.document_id == original.document_id else r
                               for r in ui_state["records"]]
    else:
        ui_state["records"] = [r for r in ui_state["records"]
                               if r.document_id != original.document_id]
