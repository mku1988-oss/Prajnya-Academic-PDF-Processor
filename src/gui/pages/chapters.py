#!/usr/bin/env python3
"""
Book Chapter Map — the contents page a book strategy already builds, made
visible for the first time. S10.

For biology researchers: when the pipeline meets a book, it reads the
front matter and the table of contents on the CPU in under a second —
number, title, per-chapter authors, the page each one starts on — but by
default reads no chapter BODIES (a full 345-page volume is hours of GPU;
see session 8's book-policy work). That map has existed in
`ProcessingResult.chapters` since session 8 and has never had a screen.
This one shows it, and lets a chapter be read on demand.

**"Read this chapter" reuses `RunRequest.chapters`, not a new mechanism.**
The CLI's `--chapters N` flag already bypasses the book policy's "ask"
step entirely — `policy.resolve_book_settings` treats an explicit
`cli_chapters` as the highest-precedence override (see its docstring),
setting the policy straight to `CHAPTERS` before `resolve_decisions` ever
gets a chance to call a prompt. So a one-PDF `RunRequest(paths=[pdf],
chapters=[n])` run through the same `BatchController` Library/Batch uses
needs no book dialog at all — there is nothing to ask, because the
request already answered it. This is the same "one runner, never
duplicated" seam the whole GUI stands on, not a chapter-map-specific
shortcut.

**Locating the PDF is the same honest guess `ledger.py` makes for "Open
PDF".** The ledger and every document JSON only ever store a bare
filename (see `ledger.py`'s docstring for why), so re-running a chapter
needs the same fallback search under Settings' remembered
`default_batch_folder`. Duplicated here rather than shared, since it is
three lines and this is only the second call site — see CLAUDE.md's rule
against premature abstraction; a third call site would be the point to
extract it.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from nicegui import run, ui

from src.processing.run_request import RunRequest

from .. import components, persistent_settings, theme
from ..controller import BatchController, BatchState
from ..session import RunAlreadyActive, RunRegistry
from .library import TRANSCRIPT_LIMIT, _STATE_LABELS, _append_transcript_line
from .results import _documents_dir, _load_document


def mount() -> None:
    ui_state: Dict[str, Any] = {
        "controller": RunRegistry.get_instance().active_controller,
        "pdf_filename": None,
        "cursor": 0,
    }
    transcript: Deque[str] = deque(maxlen=TRANSCRIPT_LIMIT)

    with ui.column().classes("w-full gap-4"):
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Book Chapter Map").style(
                f'font-family: var(--prajnya-font-display); font-size: 1.5rem; '
                f'font-weight: 600; color: {theme.var("indigo")};'
            )
            ui.space()
            refresh_button = ui.button("Refresh", icon="refresh").props("flat")

        picker = ui.select({}, label="Book", with_input=True).classes("w-full").props(
            "outlined dense"
        )
        summary_label = ui.label("").classes("text-sm").style(
            f'color: {theme.var("text_muted")};'
        )
        table_container = ui.column().classes("w-full")

        with ui.row().classes("w-full items-center gap-3"):
            state_dot = components.status_pulse("idle")
            state_label = ui.label("Idle").style("font-weight: 600;")
            ui.space()
            stop_button = ui.button("Stop", icon="stop").props("color=grey flat")
            stop_button.disable()

        with ui.card().classes("w-full").style(
            f'background: {theme.var("terminal_bg")}; padding: 0; border-radius: 8px;'
        ):
            transcript_area = ui.column().classes("w-full gap-0 p-3").style(
                "max-height: 240px; overflow-y: auto; font-family: var(--prajnya-font-mono); "
                "font-size: 12px;"
            )

    def handle_read_chapter(number: int) -> None:
        pdf_path = _locate_pdf(ui_state["pdf_filename"])
        if pdf_path is None:
            ui.notify(f'Could not find "{ui_state["pdf_filename"]}" on disk — set the '
                     f'default batch folder in Settings, or process it from Library first.',
                     type="warning", timeout=6000)
            return

        # force=True is required, not optional: already_processed() skips by
        # file hash alone, with no notion of WHICH chapters were previously
        # read. A book scanned once (even for front matter only) is in the
        # ledger forever, so without --force every "Read this chapter" past
        # the first would silently no-op — confirmed live: the real gate-run
        # book was skipped as "already in the ledger (reviewed)" until this
        # was added. See session 11's --force fix for the same class of bug.
        request = RunRequest(paths=[pdf_path], chapters=[number],
                             noninteractive=True, force=True)
        controller = BatchController(request, noninteractive=True)
        try:
            controller.start()
        except RunAlreadyActive:
            ui.notify("A batch is already running — see Library/Batch for its progress.",
                     type="warning")
            controller = RunRegistry.get_instance().active_controller

        ui_state["controller"] = controller
        ui_state["cursor"] = 0
        transcript.clear()
        transcript_area.clear()
        stop_button.enable()
        ui.notify(f"Reading chapter {number}...", type="info")

    def handle_stop() -> None:
        controller = ui_state.get("controller")
        if controller is not None:
            controller.stop()
        stop_button.disable()

    async def load_documents() -> None:
        index = await run.io_bound(_index_books)
        picker.set_options({entry["path"]: entry["label"] for entry in index})
        picker.update()

    async def handle_pick(value: Optional[str]) -> None:
        table_container.clear()
        if not value:
            ui_state["pdf_filename"] = None
            summary_label.text = ""
            return

        document = await run.io_bound(_load_document, Path(value))
        info = document.get("document_info") or {}
        ui_state["pdf_filename"] = info.get("filename")
        chapters = _chapters_of(document)
        summary_label.text = (f"{len(chapters)} chapter(s)" if chapters
                              else "No chapter map recorded for this book.")
        _render_chapters_table(table_container, chapters, handle_read_chapter)

    def poll() -> None:
        controller = ui_state.get("controller")
        if controller is None:
            return

        snapshot = controller.snapshot()
        state = snapshot["state"]
        label, color = _STATE_LABELS.get(state, (state.value, theme.var("text_muted")))
        state_label.text = label
        state_label.style(f"font-weight: 600; color: {color};")

        running = state in (BatchState.PREFLIGHT, BatchState.AWAITING_BOOK_DECISION,
                            BatchState.RUNNING, BatchState.STOPPING)
        stop_button.set_enabled(running)

        events, cursor = controller.events_since(ui_state["cursor"])
        ui_state["cursor"] = cursor
        for event in events:
            _append_transcript_line(transcript, transcript_area, event)

    refresh_button.on_click(load_documents)
    picker.on_value_change(lambda e: handle_pick(e.value))
    stop_button.on_click(handle_stop)

    if ui_state["controller"] is not None:
        stop_button.enable()

    ui.timer(0.0, load_documents, once=True)
    ui.timer(0.25, poll)


# --- data loading, run off the event loop via run.io_bound -------------------

def _index_books() -> List[Dict[str, str]]:
    """Every processed document whose type is a book — book_chapter has no
    nested chapter map of its own, that's the container it came from."""
    directory = _documents_dir()
    if not directory.is_dir():
        return []

    entries = []
    for path in sorted(directory.glob("*.v6.json")):
        try:
            document = _load_document(path)
        except Exception:                                     # pragma: no cover
            continue
        info = document.get("document_info") or {}
        if info.get("document_type") != "book":
            continue
        title = (document.get("metadata_final") or {}).get("title") or path.stem
        entries.append({"path": str(path), "label": title})
    return entries


def _chapters_of(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    profile = document.get("profile") or {}
    result = profile.get("processing_result") or {}
    return result.get("chapters") or []


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


# --- rendering ----------------------------------------------------------

def _render_chapters_table(container: ui.column, chapters: List[Dict[str, Any]],
                           on_read: Callable[[int], None]) -> None:
    container.clear()
    if not chapters:
        return

    with container:
        with ui.element("div").classes("w-full overflow-x-auto"):
            with ui.column().classes("w-full min-w-max gap-0"):
                with ui.row().classes("items-center gap-2 p-3 rounded-t-lg border-b-2").style(
                    f'background: {theme.var("sidebar")}; '
                    f'border-color: {theme.var("border_gold")}; min-width: 900px;'
                ):
                    _header_cell("#", "50px")
                    _header_cell("Title", "380px")
                    _header_cell("Authors", "300px")
                    _header_cell("Starts p.", "100px")
                    _header_cell("", "120px")

                for index, chapter in enumerate(chapters):
                    _render_chapter_row(chapter, index, on_read)


def _header_cell(text: str, width: str) -> None:
    with ui.element("div").style(f"width: {width}; flex-shrink: 0;"):
        ui.label(text).style(
            f'color: {theme.var("indigo")}; font-weight: 700; font-size: 0.875rem;'
        )


def _render_chapter_row(chapter: Dict[str, Any], index: int, on_read: Callable[[int], None]) -> None:
    number = chapter.get("number")
    bg = "background: transparent;" if index % 2 == 0 else f"background: {theme.var('row_zebra')};"

    with ui.row().classes("items-center gap-2 p-3 border-b").style(
        f'{bg} border-color: {theme.var("border")}; min-width: 900px;'
    ):
        with ui.element("div").style("width: 50px; flex-shrink: 0;"):
            ui.label(str(number) if number is not None else "—").classes("text-sm")

        with ui.element("div").style("width: 380px; flex-shrink: 0;"):
            ui.label(chapter.get("title") or "Untitled").classes("text-sm text-gray-800")

        with ui.element("div").style("width: 300px; flex-shrink: 0;"):
            authors = chapter.get("authors") or []
            ui.label(", ".join(authors) or "Unknown").classes("text-xs text-gray-600")

        with ui.element("div").style("width: 100px; flex-shrink: 0;"):
            ui.label(str(chapter.get("start_page_printed") or "—")).classes("text-sm")

        with ui.element("div").style("width: 120px; flex-shrink: 0;"):
            if number is not None:
                ui.button("Read", icon="menu_book",
                         on_click=lambda n=number: on_read(n)).props("dense flat")
