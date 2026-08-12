#!/usr/bin/env python3
"""
Ledger — every processed document, searchable, reviewable, exportable. S8.

For biology researchers: every PDF `run_prajnya.py` (or this GUI's Library
screen) has ever processed leaves one row here — its metadata, how
confident the pipeline was, and whether it still needs a human look. This
screen is where that permanent record book becomes browsable instead of a
CSV file on disk.

**Reuses the CLI's own ledger-path resolution.** `runner.open_ledger` is
the one function that already knows the `SARA_LEDGER_PATH` override and
the default location (`docs/gui_plan.md` section 9's open issue #1 — the
`SemanticLedger` ABC lacked `get_all_records`/`delete_record` until this
session promoted both). Importing it here, rather than re-deriving the
path, is what keeps "which ledger file is this?" answered in one place.

**Sorting and pagination are Quasar's, not hand-rolled.** V5's
`ledger_master.py` (1591 lines) implemented both from scratch — a full
`sort_entries` dispatch and a page-index calculator — because it built the
table from raw `ui.row`/`ui.label` divs. `ui.table` already wraps Quasar's
`QTable`, which sorts (`sortable: True` per column) and paginates
client-side for free; the checkbox column comes from `selection="multiple"`
for the same reason. What V5 hand-rolled here, V6 configures.

**"Open PDF" is honest about what the ledger actually stores.** A record's
`pdf_filename` is just the file's *name* (`review.py` sets it from
`pdf_path.name`, not the full path — nothing in the pipeline persists
where a PDF lived on disk). `_locate_pdf` tries the name directly, then
under the folder path Settings remembers as the last batch folder, then a
recursive search under that folder — and if none of that finds a file, it
says so rather than pretending. A GUI that silently does nothing on a
missing file is worse than one that names the gap.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nicegui import run, ui

from src.processing.ledger.semantic_ledger import LedgerRecord
from src.processing.paths import DEFAULT_OUTPUT_ROOT
from src.processing.runner import open_ledger

from .. import bibtex, persistent_settings, theme

logger = logging.getLogger("prajnya_studio.ledger")

_STATUS_OPTIONS = {"all": "All", "not_reviewed": "Not reviewed",
                   "provisional": "Provisional", "reviewed": "Reviewed"}

_COLUMNS = [
    {"name": "title", "label": "Title", "field": "title", "align": "left", "sortable": True},
    {"name": "authors", "label": "Authors", "field": "authors", "align": "left", "sortable": True},
    {"name": "doi", "label": "DOI", "field": "doi", "align": "left"},
    {"name": "review_status", "label": "Status", "field": "review_status", "align": "left", "sortable": True},
    {"name": "confidence", "label": "Conf. %", "field": "confidence", "align": "right", "sortable": True},
    {"name": "last_updated", "label": "Updated", "field": "last_updated", "align": "left", "sortable": True},
    {"name": "actions", "label": "", "field": "document_id", "align": "center"},
]

#: The row's "view" button is wired through a custom `$emit`, not QTable's
#: own `row-click`. Quasar's native event is a kebab-case Vue custom event
#: ('row-click'), while nicegui's `.on()` takes the exact string it is
#: given — guessing the casing wrong fails silently (no error, the handler
#: just never fires), the same trap CLAUDE.md documents for `.props()` vs a
#: component's own binding. `$parent.$emit('view_row', props.row)` names an
#: event this module defines itself, so there is no casing to get wrong.
_VIEW_ROW_SLOT = '''
    <q-td :props="props" auto-width>
        <q-btn flat dense round icon="visibility"
               @click="$parent.$emit('view_row', props.row)" />
    </q-td>
'''


def mount() -> None:
    ui_state: Dict[str, Any] = {"ledger": None, "records": [], "by_id": {}}
    dialog_container = ui.column()

    with ui.column().classes("w-full gap-4"):
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Ledger").style(
                f'font-family: var(--prajnya-font-display); font-size: 1.5rem; '
                f'font-weight: 600; color: {theme.var("indigo")};'
            )
            ui.space()
            export_csv_button = ui.button("Export CSV", icon="download").props("flat")
            export_bib_button = ui.button("Export BibTeX", icon="menu_book").props("flat")

        with ui.row().classes("w-full items-end gap-4"):
            search_input = ui.input("Search title, author, DOI").classes("flex-1").props(
                "outlined dense clearable"
            )
            status_select = ui.select(_STATUS_OPTIONS, value="all", label="Review status").props(
                "outlined dense"
            ).classes("w-48")
            confidence_slider = ui.slider(min=0.0, max=1.0, step=0.05, value=0.0).props(
                "label-always"
            ).classes("w-48")

        summary_label = ui.label("").classes("text-sm").style(f'color: {theme.var("text_muted")};')
        batch_row = ui.row().classes("w-full items-center gap-3")

        table = ui.table(columns=_COLUMNS, rows=[], row_key="document_id",
                         selection="multiple", pagination={"rowsPerPage": 20}).classes("w-full")
        table.add_slot("body-cell-actions", _VIEW_ROW_SLOT)
        table.set_visibility(False)
        empty_state = ui.column().classes("w-full items-center gap-2 p-12")
        with empty_state:
            ui.icon("inbox", size="4em").classes("text-gray-300")
            ui.label("No entries in ledger").classes("text-lg").style(
                f'color: {theme.var("text_muted")};'
            )

    def refresh() -> None:
        records = _filtered_records(ui_state["records"], search_input.value or "",
                                    status_select.value, confidence_slider.value)
        summary_label.text = f"Showing {len(records)} of {len(ui_state['records'])} entries"
        table.rows = [_to_row(record) for record in records]
        table.selected = []
        table.update()
        _render_batch_row(batch_row, table, ui_state, refresh)
        has_any = bool(ui_state["records"])
        table.set_visibility(has_any)
        empty_state.set_visibility(not has_any)

    async def load() -> None:
        ledger, records = await run.io_bound(_load_ledger_sync)
        ui_state["ledger"] = ledger
        ui_state["records"] = records
        ui_state["by_id"] = {record.document_id: record for record in records}
        refresh()

    def on_view_row(event) -> None:
        document_id = event.args.get("document_id") if isinstance(event.args, dict) else None
        record = ui_state["by_id"].get(document_id)
        if record is not None:
            _show_record_dialog(dialog_container, record, ui_state, refresh)

    search_input.on_value_change(lambda e: refresh())
    status_select.on_value_change(lambda e: refresh())
    confidence_slider.on_value_change(lambda e: refresh())
    table.on("view_row", on_view_row)
    table.on_select(lambda e: _render_batch_row(batch_row, table, ui_state, refresh))
    export_csv_button.on_click(lambda: _export_csv(ui_state))
    export_bib_button.on_click(lambda: _export_bibtex(ui_state))

    ui.timer(0.0, load, once=True)


# --- filtering + row shaping (pure, unit-testable) ---------------------------

def _filtered_records(records: List[LedgerRecord], query: str, status: str,
                      min_confidence: float) -> List[LedgerRecord]:
    results = records
    query = query.strip().lower()
    if query:
        results = [r for r in results if (
            (r.title and query in r.title.lower())
            or (r.doi and query in r.doi.lower())
            or any(query in author.lower() for author in r.authors)
        )]
    if status == "not_reviewed":
        results = [r for r in results if r.review_status in ("not_reviewed", "provisional")]
    elif status == "reviewed":
        results = [r for r in results if r.review_status == "reviewed"]
    return [r for r in results if r.validation_confidence >= min_confidence]


def _to_row(record: LedgerRecord) -> Dict[str, Any]:
    return {
        "document_id": record.document_id,
        "title": record.title or "Untitled",
        "authors": _author_summary(record.authors),
        "doi": record.doi or "—",
        "review_status": record.review_status.replace("_", " ").title(),
        "confidence": round(record.validation_confidence * 100),
        "last_updated": (record.last_updated or "")[:10],
    }


def _author_summary(authors: List[str]) -> str:
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0]
    return f"{authors[0]} et al. ({len(authors)})"


# --- data loading, run off the event loop via run.io_bound -------------------

def _load_ledger_sync():
    ledger = open_ledger(logger, create=True)
    records = ledger.get_all_records() if ledger is not None else []
    records.sort(key=lambda r: r.last_updated or "", reverse=True)
    return ledger, records


# --- batch actions -------------------------------------------------------

def _render_batch_row(container: ui.row, table: ui.table, ui_state: Dict[str, Any],
                      refresh) -> None:
    container.clear()
    selected = table.selected or []
    if not selected:
        return
    with container:
        ui.label(f"{len(selected)} selected").classes("text-sm font-semibold").style(
            f'color: {theme.var("indigo")};'
        )
        ui.button("Mark Reviewed", icon="check_circle",
                 on_click=lambda: _batch_mark_reviewed(selected, ui_state, refresh)).props("dense")
        ui.button("Delete Selected", icon="delete", color="negative",
                 on_click=lambda: _confirm_batch_delete(selected, ui_state, refresh)).props("dense flat")


def _batch_mark_reviewed(selected: List[Dict[str, Any]], ui_state: Dict[str, Any], refresh) -> None:
    ledger = ui_state["ledger"]
    if ledger is None:
        return
    for row in selected:
        record = ui_state["by_id"].get(row["document_id"])
        if record is None:
            continue
        record.review_status = "reviewed"
        ledger.update_record(record)
    ledger.commit()
    ui.notify(f"Marked {len(selected)} entries as reviewed", type="positive")
    refresh()


def _confirm_batch_delete(selected: List[Dict[str, Any]], ui_state: Dict[str, Any], refresh) -> None:
    count = len(selected)
    with ui.dialog() as dialog, ui.card():
        ui.label(f"Delete {count} entries? This cannot be undone.").classes("text-base")
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            def confirm() -> None:
                _delete_records(selected, ui_state)
                dialog.close()
                ui.notify(f"Deleted {count} entries", type="positive")
                refresh()

            ui.button("Delete", on_click=confirm, color="negative")
    dialog.open()


def _delete_records(rows: List[Dict[str, Any]], ui_state: Dict[str, Any]) -> None:
    ledger = ui_state["ledger"]
    if ledger is None:
        return
    for row in rows:
        ledger.delete_record(row["document_id"])
    ledger.commit()


# --- the per-record dialog: view, edit, mark, delete, open PDF ---------------

def _show_record_dialog(container: ui.column, record: LedgerRecord,
                        ui_state: Dict[str, Any], refresh) -> None:
    with container:
        with ui.dialog() as dialog, ui.card().classes("w-[560px] p-6 gap-3"):
            ui.label(record.title or "Untitled").style(
                f'font-size: 1.15rem; font-weight: 600; color: {theme.var("indigo")};'
            )

            title_input = ui.input("Title", value=record.title or "").classes("w-full")
            authors_input = ui.textarea(
                "Authors (one per line)", value="\n".join(record.authors)
            ).classes("w-full").props("rows=4")
            doi_input = ui.input("DOI", value=record.doi or "").classes("w-full")
            journal_input = ui.input("Journal / Publisher",
                                     value=record.journal or record.publisher or "").classes("w-full")
            year_input = ui.number("Year", value=record.year, min=1900, max=2100).classes("w-full")
            status_radio = ui.radio(
                ["not_reviewed", "provisional", "reviewed"], value=record.review_status
            ).props("inline")

            with ui.row().classes("w-full justify-between items-center mt-2"):
                ui.button("Open PDF", icon="picture_as_pdf",
                         on_click=lambda: _open_pdf(record)).props("flat")
                ui.button("Delete", icon="delete", color="negative",
                         on_click=lambda: _delete_from_dialog(record, ui_state, dialog, refresh)
                         ).props("flat")

            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Save", on_click=lambda: _save_from_dialog(
                    record, ui_state, dialog, refresh,
                    title_input.value, authors_input.value, doi_input.value,
                    journal_input.value, year_input.value, status_radio.value,
                ))

        dialog.open()


def _save_from_dialog(record: LedgerRecord, ui_state: Dict[str, Any], dialog, refresh,
                      title: str, authors_text: str, doi: str, journal: str,
                      year: Optional[float], status: str) -> None:
    ledger = ui_state["ledger"]
    if ledger is None:
        dialog.close()
        return

    record.title = title.strip() or None
    record.authors = [line.strip() for line in authors_text.split("\n") if line.strip()]
    record.doi = doi.strip() or None
    if record.journal:
        record.journal = journal.strip() or None
    else:
        record.publisher = journal.strip() or None
    record.year = int(year) if year else None
    record.review_status = status
    record.validation_source = "user_corrected"
    record.last_updated = datetime.utcnow().isoformat()

    ledger.update_record(record)
    ledger.commit()
    dialog.close()
    ui.notify(f'Updated "{record.title}"', type="positive")
    refresh()


def _delete_from_dialog(record: LedgerRecord, ui_state: Dict[str, Any], dialog, refresh) -> None:
    ledger = ui_state["ledger"]
    if ledger is not None:
        ledger.delete_record(record.document_id)
        ledger.commit()
    dialog.close()
    ui.notify(f'Deleted "{record.title or record.document_id}"', type="positive")
    refresh()


def _open_pdf(record: LedgerRecord) -> None:
    path = _locate_pdf(record.pdf_filename)
    if path is None:
        ui.notify(f'Could not find "{record.pdf_filename}" on disk — the ledger only '
                  f'records the filename, not where it was processed from.',
                  type="warning", timeout=6000)
        return
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))                             # noqa: S606
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(path)], check=True)
        else:
            subprocess.run(["xdg-open", str(path)], check=True)
        ui.notify(f"Opening {path.name}", type="info")
    except Exception as error:                                   # pragma: no cover
        logger.warning("Failed to open PDF %s: %s", path, error)
        ui.notify(f"Failed to open PDF: {error}", type="negative")


def _locate_pdf(filename: Optional[str]) -> Optional[Path]:
    if not filename:
        return None

    direct = Path(filename)
    if direct.is_absolute() and direct.is_file():
        return direct

    settings = persistent_settings.SettingsManager.load()
    folder = settings.get("default_batch_folder")
    if not folder:
        return None

    base = Path(folder)
    candidate = base / filename
    if candidate.is_file():
        return candidate

    for match in base.rglob(filename):
        return match
    return None


# --- exports ---------------------------------------------------------------

def _export_csv(ui_state: Dict[str, Any]) -> None:
    records = ui_state["records"]
    if not records:
        ui.notify("No entries to export", type="warning")
        return
    content = build_csv(records)
    _download_and_save(content.encode("utf-8"), "csv")


def _export_bibtex(ui_state: Dict[str, Any]) -> None:
    records = ui_state["records"]
    if not records:
        ui.notify("No entries to export", type="warning")
        return
    content = bibtex.build_bibtex(records)
    _download_and_save(content.encode("utf-8"), "bib")


def build_csv(records: List[LedgerRecord]) -> str:
    """CSV export, one row per record. A pure function so it is testable
    without a running NiceGUI page."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Title", "Authors", "DOI", "Journal", "Year", "Review Status",
                     "Confidence", "Validation Source", "Last Updated", "PDF Filename"])
    for record in records:
        writer.writerow([
            record.title or "", "; ".join(record.authors), record.doi or "",
            record.journal or record.publisher or "", record.year or "",
            record.review_status.replace("_", " ").title(),
            f"{record.validation_confidence:.0%}", record.validation_source or "",
            record.last_updated or "", record.pdf_filename or "",
        ])
    return output.getvalue()


def _download_and_save(content: bytes, extension: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ledger_export_{timestamp}.{extension}"

    export_dir = DEFAULT_OUTPUT_ROOT / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / filename).write_bytes(content)

    ui.download(content, filename)
    ui.notify(f"Exported to {filename}", type="positive")
