#!/usr/bin/env python3
"""
Results viewer — the screen V5 never had.

For biology researchers: this is where a processed paper's output becomes
visible. `run_prajnya.py` and the Library/Batch screen both write one JSON
per document plus a folder of figure and table crops; until this screen
existed, reading any of that meant opening the JSON file by hand. The 11
August gate run alone produced 107 figures and 31 tables that no UI could
show.

**Where figures and tables actually live, and why this page has to compute
it.** A document's JSON is named after its own filename
(`documents/<stem>.v6.json`), but its figure and table crops live under a
**filesystem-safe** version of that same stem
(`figures/<safe_stem>/manifest.json`) — spaces and punctuation become
underscores. `src.processing.utils.figure_extractor.safe_doc_id` is the one
function that performs that sanitisation when a document is written; this
page imports it rather than re-deriving the rule, so a future change to
that rule cannot silently break the gallery without also breaking the
pipeline's own writer.

**Why a page selector instead of one long scroll.** The corpus this project
processes includes a 345-page book. Rendering every page's text into the
DOM at once would be slow for no benefit — a reader looks at one page at a
time anyway.

**Why the raw JSON tree is built lazily.** The largest document in the gate
run is ~3.8MB of JSON. Feeding that into `ui.json_editor` on every page
load, whether anyone looks at it or not, is wasted work; it is built once,
the first time that section is opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from nicegui import run, ui

from src.processing.paths import DEFAULT_OUTPUT_ROOT
from src.processing.utils.figure_extractor import safe_doc_id

from .. import theme


def mount() -> None:
    ui_state: Dict[str, Any] = {"document": None, "stem": None, "raw_built": False}

    with ui.column().classes("w-full gap-4"):
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Results").style(
                f'font-family: var(--prajnya-font-display); font-size: 1.5rem; '
                f'font-weight: 600; color: {theme.var("indigo")};'
            )
            ui.space()
            refresh_button = ui.button("Refresh", icon="refresh").props("flat")

        picker = ui.select({}, label="Document", with_input=True).classes("w-full").props(
            "outlined dense"
        )
        detail_container = ui.column().classes("w-full gap-4")

    async def load_documents() -> None:
        index = await run.io_bound(_index_documents)
        picker.set_options({entry["path"]: entry["label"] for entry in index})
        picker.update()

    async def handle_pick(value: Optional[str]) -> None:
        detail_container.clear()
        ui_state["raw_built"] = False
        if not value:
            ui_state["document"] = None
            return

        document = await run.io_bound(_load_document, Path(value))
        ui_state["document"] = document
        stem = Path(document.get("document_info", {}).get("filename", "")).stem
        ui_state["stem"] = stem or Path(value).stem.removesuffix(".v6")

        figures = await run.io_bound(_figures_for, ui_state["stem"])
        tables = await run.io_bound(_tables_for, ui_state["stem"])
        _render_document(detail_container, document, figures, tables)

    refresh_button.on_click(load_documents)
    picker.on_value_change(lambda e: handle_pick(e.value))

    # load_documents is async (it awaits run.io_bound); a bare call at mount
    # time returns an un-awaited coroutine and does nothing but warn. once=True
    # is nicegui's own idiom for "run this async callback right after mount".
    ui.timer(0.0, load_documents, once=True)


# --- data loading, run off the event loop via run.io_bound ------------------

def _documents_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / "documents"


def _index_documents() -> List[Dict[str, str]]:
    """One line per document JSON, cheap enough to read in full — these
    files are at most a few MB, and a personal library is not thousands of
    them."""
    directory = _documents_dir()
    if not directory.is_dir():
        return []

    entries = []
    for path in sorted(directory.glob("*.v6.json")):
        try:
            document = _load_document(path)
        except Exception:                                    # pragma: no cover
            entries.append({"path": str(path), "label": f"{path.name} (unreadable)"})
            continue
        title = (document.get("metadata_final") or {}).get("title") or path.stem
        kind = (document.get("document_info") or {}).get("document_type", "?")
        flag = " ⚠" if document.get("review_status") == "provisional" else ""
        entries.append({"path": str(path), "label": f"{title} — {kind}{flag}"})
    return entries


def _load_document(path: Path) -> Dict[str, Any]:
    import json

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _figures_for(stem: str) -> List[Dict[str, Any]]:
    return _manifest_for("figures", stem, "figures")


def _tables_for(stem: str) -> List[Dict[str, Any]]:
    return _manifest_for("tables", stem, "tables")


def _manifest_for(kind: str, stem: str, list_key: str) -> List[Dict[str, Any]]:
    import json

    manifest_path = DEFAULT_OUTPUT_ROOT / kind / safe_doc_id(stem) / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            return json.load(handle).get(list_key, [])
    except Exception:                                          # pragma: no cover
        return []


# --- rendering (after mount and its data loaders, per CLAUDE.md) ------------

def _render_document(container: ui.column, document: Dict[str, Any],
                     figures: List[Dict[str, Any]], tables: List[Dict[str, Any]]) -> None:
    with container:
        _render_metadata_card(document)
        _render_page_viewer(document)
        if figures:
            _render_figure_gallery(figures)
        if tables:
            _render_table_gallery(tables)
        _render_raw_json_section(document)


def _render_metadata_card(document: Dict[str, Any]) -> None:
    info = document.get("document_info") or {}
    meta = document.get("metadata_final") or {}
    summary = document.get("summary") or {}
    review_status = document.get("review_status")
    review_fields = document.get("review_needed_fields") or []

    with ui.card().classes("w-full").style(f'border: 1px solid {theme.var("border")};'):
        ui.label(meta.get("title") or info.get("filename", "Untitled")).style(
            f'font-weight: 600; font-size: 1.1rem; color: {theme.var("indigo")};'
        )
        authors = ", ".join(meta.get("authors") or []) or "unknown authors"
        journal = meta.get("journal") or meta.get("publisher") or ""
        year = meta.get("year")
        ui.label(f"{authors} — {journal} {year or ''}".strip()).classes("text-sm").style(
            f'color: {theme.var("text_muted")};'
        )

        with ui.row().classes("w-full gap-6 mt-2"):
            _stat("Type", info.get("document_type", "?"))
            _stat("Pages", summary.get("page_count", "?"))
            _stat("Figures", summary.get("figure_count", "?"))
            _stat("DOI", meta.get("doi") or "—")
            _stat("Validation", f'{meta.get("validation_source", "?")} '
                                f'({meta.get("validation_confidence", 0):.2f})')

        if review_status == "provisional" or review_fields:
            with ui.row().classes("items-center gap-2 mt-2").style(
                f'background: {theme.var("warning_tint")}; padding: 8px 12px; border-radius: 6px;'
            ):
                ui.icon("warning", color=theme.var("warning"))
                fields = ", ".join(review_fields) if review_fields else "metadata"
                ui.label(f"Needs review: {fields}").classes("text-sm")

        errors = summary.get("errors") or []
        if errors:
            with ui.row().classes("items-center gap-2 mt-1"):
                ui.icon("error", color=theme.var("error"))
                ui.label(f"{len(errors)} error(s) during processing").classes("text-sm").style(
                    f'color: {theme.var("error")};'
                )


def _stat(label: str, value: Any) -> None:
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-xs uppercase").style(
            f'color: {theme.var("gold")}; font-weight: 600;'
        )
        ui.label(str(value)).classes("text-sm")


def _render_page_viewer(document: Dict[str, Any]) -> None:
    pages = document.get("pages") or []
    if not pages:
        return

    with ui.expansion("Pages", icon="description", value=True).classes("w-full"):
        page_numbers = [page.get("page_number", i + 1) for i, page in enumerate(pages)]
        by_number = {page.get("page_number", i + 1): page for i, page in enumerate(pages)}

        content_area = ui.column().classes("w-full")

        def show_page(number: int) -> None:
            page = by_number.get(number)
            content_area.clear()
            if page is None:
                return
            with content_area:
                score = page.get("quality_score")
                if score is not None:
                    ui.label(f"Quality score: {score:.2f}").classes("text-xs").style(
                        f'color: {theme.var("text_muted")};'
                    )
                ui.markdown(page.get("content") or "*(no text)*").classes("w-full").style(
                    "max-height: 480px; overflow-y: auto;"
                )

        selector = ui.select(page_numbers, label="Page", value=page_numbers[0],
                             on_change=lambda e: show_page(e.value)).props("outlined dense")
        show_page(page_numbers[0])


def _render_figure_gallery(figures: List[Dict[str, Any]]) -> None:
    with ui.expansion(f"Figures ({len(figures)})", icon="image", value=True).classes("w-full"):
        for label_key, group in _group_by(figures, "label_key").items():
            with ui.column().classes("w-full gap-1 mb-3"):
                heading = group[0].get("printed_label") or label_key or "Figure"
                ui.label(heading).classes("text-sm font-semibold")
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    for figure in group:
                        _figure_card(figure)
                caption = group[0].get("caption")
                if caption:
                    ui.label(caption).classes("text-xs").style(
                        f'color: {theme.var("text_muted")}; max-width: 700px;'
                    )


def _figure_card(figure: Dict[str, Any]) -> None:
    with ui.column().classes("gap-1"):
        ui.image(f"/run_outputs/{figure['image_path']}").classes("rounded").style(
            "max-width: 260px; max-height: 260px; object-fit: contain; "
            f'border: 1px solid {theme.var("border")};'
        )
        ui.label(f"p{figure.get('page')} — {figure.get('figure_id')}").classes("text-xs").style(
            f'color: {theme.var("text_muted")};'
        )


def _render_table_gallery(tables: List[Dict[str, Any]]) -> None:
    with ui.expansion(f"Tables ({len(tables)})", icon="table_chart").classes("w-full"):
        for table in tables:
            with ui.column().classes("w-full gap-1 mb-4"):
                ui.label(table.get("caption") or table.get("table_id", "Table")).classes(
                    "text-sm font-semibold"
                )
                with ui.row().classes("w-full gap-4 items-start"):
                    ui.image(f"/run_outputs/{table['image_path']}").classes("rounded").style(
                        "max-width: 320px; object-fit: contain; "
                        f'border: 1px solid {theme.var("border")};'
                    )
                    grid = table.get("grid")
                    if grid:
                        ui.html(grid, sanitize=False).classes("text-xs").style(
                            "max-width: 500px; overflow-x: auto;"
                        )


def _render_raw_json_section(document: Dict[str, Any]) -> None:
    built = {"done": False}

    with ui.expansion("Raw JSON", icon="code").classes("w-full") as expansion:
        container = ui.column().classes("w-full")

    def build_once(open_now: bool) -> None:
        if not open_now or built["done"]:
            return
        built["done"] = True
        with container:
            ui.json_editor({"content": {"json": document}, "mode": "tree", "readOnly": True})

    expansion.on_value_change(lambda e: build_once(e.value))


# --- small pure helpers -------------------------------------------------

def _group_by(items: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item.get(key) or "", []).append(item)
    return groups
