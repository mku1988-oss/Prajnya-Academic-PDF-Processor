#!/usr/bin/env python3
"""
Library / Batch — pick a folder, preview it, run it, watch it live.

For biology researchers: this is the screen that replaces
`run_prajnya.py --pdf <folder>`. It does the exact same three steps in the
exact same order — classify every PDF on the CPU first (seconds, no GPU),
settle what to do with any book *before* any GPU work starts, then run the
folder unattended — because that order is what makes a batch safe to
leave alone. See `docs/gui_plan.md` section 1 for why that order is a
hard rule and not a UI preference.

**Why the folder is typed, not uploaded.** Prajnya Studio runs on the same
machine as the PDFs; V5's browser-upload flow existed because its GUI
could, in principle, run somewhere else. Typing the path the CLI already
takes keeps `--pdf <folder>` and this screen doing the identical thing,
which is the point of the whole runner seam.

**Re-attaching, not re-running.** If a batch is already active (started
from this tab, another tab, or a previous page load), mounting this page
finds it via `RunRegistry` and joins its live progress instead of
offering to start a second one — the registry is what makes that safe on
one 8GB GPU.

**S11: Settings' path-and-policy keys apply here, and only here.** They
are read once at mount (the same moment `default_batch_folder` and
`recursive_default` already were), and each only overrides its
`RunRequest` field when actually set to something non-empty — a user who
has never opened Settings gets a `RunRequest` that looks exactly as it
did before S11. `book_policy` becomes a `book_overrides` dict (the
documented per-run channel `policy.py` already has, precedence CLI >
override > YAML > default) rather than `request.chapters`, which is a
blunter override that bypasses the book dialog entirely — the wrong tool
here, since a batch may still want to ask about a book it meets, and
`chapters.py`'s "Read this chapter" (S10) is the screen that actually
wants that blunter form.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from nicegui import run, ui

from src.processing.batch import BookDecision, PreflightEntry, find_pdfs
from src.processing.book_prompt import parse_chapter_selection
from src.processing.run_request import RunRequest

from .. import components, persistent_settings, theme
from ..controller import BatchController, BatchState, BookQuestion
from ..session import RunAlreadyActive, RunRegistry

#: How many transcript lines a page keeps on screen. The controller itself
#: keeps up to MAX_EVENTS (5000); this is just what is rendered at once.
TRANSCRIPT_LIMIT = 200

_STATE_LABELS = {
    BatchState.IDLE: ("Idle", theme.var("text_muted")),
    BatchState.PREFLIGHT: ("Classifying...", theme.var("warning")),
    BatchState.AWAITING_BOOK_DECISION: ("Waiting on a book decision", theme.var("warning")),
    BatchState.RUNNING: ("Running", theme.var("success")),
    BatchState.STOPPING: ("Stopping...", theme.var("error")),
    BatchState.DONE: ("Done", theme.var("success")),
    BatchState.STOPPED: ("Stopped", theme.var("text_muted")),
    BatchState.FAILED: ("Failed", theme.var("error")),
}


def mount() -> None:
    settings = persistent_settings.SettingsManager.load()
    ui_state: Dict[str, Any] = {
        "controller": RunRegistry.get_instance().active_controller,
        "entries": [],
        "cursor": 0,
        "question_open": False,
    }
    transcript: Deque[str] = deque(maxlen=TRANSCRIPT_LIMIT)

    with ui.column().classes("w-full gap-4"):
        ui.label("Library / Batch").style(
            f'font-family: var(--prajnya-font-display); font-size: 1.5rem; '
            f'font-weight: 600; color: {theme.var("indigo")};'
        )

        with ui.row().classes("w-full items-end gap-3"):
            folder_input = ui.input(
                "Folder", value=settings.get("default_batch_folder", "")
            ).classes("flex-1").props("outlined dense")
            recursive_switch = ui.switch(
                "Recursive", value=settings.get("recursive_default", False)
            )
            noninteractive_switch = ui.switch("Unattended", value=False)
            scan_button = ui.button("Scan Folder", icon="search")

        summary_label = ui.label("").style(f'color: {theme.var("text_muted")};')
        table_container = ui.column().classes("w-full")

        with ui.row().classes("w-full items-center gap-3"):
            state_dot = components.status_pulse("idle")
            state_label = ui.label("Idle").style("font-weight: 600;")
            ui.space()
            start_button = ui.button("Start Batch", icon="play_arrow").props("color=red-9")
            start_button.disable()
            stop_button = ui.button("Stop", icon="stop").props("color=grey flat")
            stop_button.disable()

        with ui.card().classes("w-full").style(
            f'background: {theme.var("terminal_bg")}; padding: 0; border-radius: 8px;'
        ):
            transcript_area = ui.column().classes("w-full gap-0 p-3").style(
                "max-height: 320px; overflow-y: auto; font-family: var(--prajnya-font-mono); font-size: 12px;"
            )

    async def handle_scan() -> None:
        folder = Path(folder_input.value or "")
        if not folder.is_dir():
            ui.notify(f"Not a folder: {folder}", type="negative")
            return

        paths = find_pdfs(folder, recursive=recursive_switch.value)
        if not paths:
            ui.notify("No PDFs found in that folder.", type="warning")
            return

        scan_button.disable()
        summary_label.text = f"Classifying {len(paths)} PDF(s)..."
        try:
            entries = await run.io_bound(_preflight_sync, paths, recursive_switch.value)
        finally:
            scan_button.enable()

        ui_state["entries"] = entries
        ui_state["folder"] = folder
        _render_preflight_table(table_container, entries)
        summary_label.text = _summarize(entries)
        start_button.enable()

    def handle_start() -> None:
        entries: List[PreflightEntry] = ui_state.get("entries") or []
        if not entries:
            ui.notify("Nothing to run — scan a folder first.", type="warning")
            return

        # Every scanned PDF goes into the request, INCLUDING ones that
        # already failed pre-flight — filtering them out here would drop
        # them from the batch entirely, so they'd never appear in the
        # results or batch_summary.json at all. process_batch already
        # handles a pre-flight error per entry (records it, moves on); the
        # CLI's run_folder passes every found PDF through the same way.
        request_kwargs: Dict[str, Any] = dict(
            paths=[entry.path for entry in entries],
            recursive=recursive_switch.value,
            noninteractive=noninteractive_switch.value,
        )
        _apply_settings_overrides(request_kwargs, settings)
        request = RunRequest(**request_kwargs)
        controller = BatchController(request, noninteractive=noninteractive_switch.value)
        try:
            controller.start()
        except RunAlreadyActive:
            ui.notify("A batch is already running — see its progress below.",
                     type="warning")
            controller = RunRegistry.get_instance().active_controller

        ui_state["controller"] = controller
        ui_state["cursor"] = 0
        transcript.clear()
        transcript_area.clear()
        start_button.disable()
        stop_button.enable()

    def handle_stop() -> None:
        controller = ui_state.get("controller")
        if controller is not None:
            controller.stop()
        stop_button.disable()

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
        start_button.set_enabled(not running)

        events, cursor = controller.events_since(ui_state["cursor"])
        ui_state["cursor"] = cursor
        for event in events:
            _append_transcript_line(transcript, transcript_area, event)

        question: Optional[BookQuestion] = snapshot["question"]
        if question is not None and not ui_state["question_open"]:
            ui_state["question_open"] = True
            _show_book_dialog(question, controller, lambda: ui_state.update(question_open=False))

    scan_button.on_click(handle_scan)
    start_button.on_click(handle_start)
    stop_button.on_click(handle_stop)

    if ui_state["controller"] is not None:
        # Re-attaching to a run already in progress: show what it already
        # found rather than waiting for a scan that will never happen again.
        ui_state["entries"] = ui_state["controller"].entries
        if ui_state["entries"]:
            _render_preflight_table(table_container, ui_state["entries"])
            summary_label.text = _summarize(ui_state["entries"])
        start_button.disable()
        stop_button.enable()

    ui.timer(0.25, poll)


# --- helpers (after mount, per CLAUDE.md) ------------------------------------

def _apply_settings_overrides(kwargs: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """
    Adds Settings' path-and-policy keys to a `RunRequest`'s kwargs, in place.

    Only ever ADDS a key when the setting is actually non-empty — never
    overwrites with a blank — so `RunRequest`'s own dataclass defaults keep
    deciding everything for a user who has never opened Settings.
    """
    if settings.get("output_root"):
        kwargs["out_root"] = Path(settings["output_root"])
    if settings.get("ledger_path"):
        kwargs["ledger_path"] = Path(settings["ledger_path"])
    if settings.get("front_matter_pages"):
        kwargs["front_matter_pages"] = int(settings["front_matter_pages"])
    if settings.get("pdf_backend"):
        kwargs["backend"] = settings["pdf_backend"]
    if settings.get("force_default"):
        kwargs["force"] = True

    max_pages = _optional_int(settings.get("book_max_pages"))
    if max_pages is not None:
        kwargs["max_pages"] = max_pages

    overrides = _book_overrides_from_settings(settings)
    if overrides:
        kwargs["book_overrides"] = overrides


def _book_overrides_from_settings(settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The per-run `book_overrides` dict `policy.py` expects, or None when
    Settings' book policy is left at "" (inherit `processing_policy.yaml`)."""
    policy = settings.get("book_policy")
    if not policy:
        return None

    overrides: Dict[str, Any] = {"policy": policy}
    if policy == "chapters":
        overrides["chapters"] = parse_chapter_selection(settings.get("book_chapters") or "")
    return overrides


def _optional_int(value: Any) -> Optional[int]:
    """An int from a Settings field, or None for blank/unparseable — never
    0, which `RunRequest.max_pages` would read as a real (zero-page) cap."""
    text = str(value).strip() if value is not None else ""
    return int(text) if text.isdigit() else None


def _preflight_sync(paths: List[Path], recursive: bool) -> List[PreflightEntry]:
    """Runs on a worker thread via `run.io_bound` so the event loop stays free."""
    from src.processing.paths import CONFIG_ROOT
    from src.processing.run_request import RunRequest as _RunRequest
    from src.processing.runner import run_preflight

    request = _RunRequest(paths=paths, config_root=CONFIG_ROOT, recursive=recursive)
    return run_preflight(request)


def _summarize(entries: List[PreflightEntry]) -> str:
    if not entries:
        return "No PDFs."
    errors = sum(1 for entry in entries if entry.error)
    kinds: Dict[str, int] = {}
    for entry in entries:
        if entry.error:
            continue
        kind = entry.document_type.value if entry.document_type else "unknown"
        kinds[kind] = kinds.get(kind, 0) + 1
    parts = [f"{count} {kind}" for kind, count in sorted(kinds.items())]
    summary = f"{len(entries)} PDF(s): " + ", ".join(parts) if parts else f"{len(entries)} PDF(s)"
    if errors:
        summary += f" — {errors} could not be read"
    return summary


def _render_preflight_table(container: ui.column, entries: List[PreflightEntry]) -> None:
    container.clear()
    columns = [
        {"name": "name", "label": "File", "field": "name", "align": "left"},
        {"name": "type", "label": "Type", "field": "type", "align": "left"},
        {"name": "confidence", "label": "Confidence", "field": "confidence", "align": "right"},
        {"name": "pages", "label": "Pages", "field": "pages", "align": "right"},
        {"name": "flag", "label": "", "field": "flag", "align": "left"},
    ]
    rows = []
    for entry in entries:
        if entry.error:
            rows.append({"name": entry.path.name, "type": "error",
                        "confidence": "", "pages": "", "flag": entry.error[:80]})
            continue
        flag = "possibly misclassified" if entry.looks_misclassified else (
            f"{entry.chapter_count} chapters" if entry.chapter_count else "")
        rows.append({
            "name": _display_name(entry),
            "type": entry.document_type.value if entry.document_type else "unknown",
            "confidence": f"{entry.confidence:.2f}",
            "pages": entry.page_count,
            "flag": flag,
        })
    with container:
        ui.table(columns=columns, rows=rows, row_key="name").classes("w-full")


#: Longest document name any pre-flight row or dialog will print.
#: A title is extracted text, so its length is not bounded by anything —
#: see `_display_name`.
DISPLAY_NAME_LIMIT = 110


def _display_name(entry, limit: int = DISPLAY_NAME_LIMIT) -> str:
    """A document's name, short enough to put in a table cell or a heading.

    **Why a title needs truncating at all.** `entry.title` is *extracted*,
    not declared: when title detection does poorly, it can come back as the
    whole first page. One document in the sample corpus
    (`CollagenFibrilAssemblyandFunction.pdf`) returns ~2000 characters — its
    contents list, abstract, copyright line and DOI, all as one "title".
    Printed in full it made the book dialog thousands of pixels tall and
    pushed its own buttons off the bottom of the screen, so the run could
    not be answered without scrolling a wall of text.

    Falling back to the filename would not fix it: the failure is a title
    that is too long, not one that is missing.

    Args:
        entry: a `PreflightEntry`.
        limit: maximum characters before an ellipsis.

    Returns:
        The title if there is one, else the filename; truncated either way.
    """
    name = entry.title or entry.path.name
    name = " ".join(name.split())          # collapse newlines from page text
    if len(name) <= limit:
        return name
    return name[:limit - 1].rstrip() + "…"


def _append_transcript_line(transcript: Deque[str], area: ui.column, event) -> None:
    line = _format_event(event)
    if line is None:
        return
    transcript.append(line)
    with area:
        ui.label(line).style(f"color: {theme.var('terminal_text')}; white-space: pre-wrap; margin: 0;")
    # Bounded rendering: drop the oldest label once the deque itself drops
    # its oldest line, so the DOM never grows past TRANSCRIPT_LIMIT either.
    while len(area.default_slot.children) > TRANSCRIPT_LIMIT:
        area.remove(area.default_slot.children[0])


def _format_event(event) -> Optional[str]:
    if event.kind == "page" and event.page and event.page_total:
        return f"  page {event.page}/{event.page_total}: {event.message}"
    if event.message:
        return event.message
    return None


def _show_book_dialog(question: BookQuestion, controller: BatchController,
                      on_close) -> None:
    entry = question.entry
    title = _display_name(entry)

    with ui.dialog() as dialog, ui.card().classes("w-[480px]"):
        ui.label(f'"{title}" looks like a book').style(
            f'font-weight: 600; color: {theme.var("indigo")};'
        )
        ui.label(f"{entry.page_count} pages, {entry.chapter_count} chapters found.").classes(
            "text-sm"
        ).style(f'color: {theme.var("text_muted")};')

        chapters_input = ui.input("Chapters (e.g. 3 4 5), leave blank for all").props(
            "outlined dense"
        ).classes("w-full")
        chapters_input.set_visibility(False)

        def choose(kind: str) -> None:
            if kind == "front_matter":
                decision = BookDecision.safe_default()
            elif kind == "all":
                decision = BookDecision(chapters=[], reason="chosen in the GUI: every chapter")
            else:
                numbers = [int(tok) for tok in chapters_input.value.split() if tok.strip().isdigit()]
                decision = BookDecision(chapters=numbers or [],
                                       reason=f"chosen in the GUI: chapters {numbers or 'all'}")
            controller.answer_book_decision(decision)
            dialog.close()
            on_close()

        with ui.column().classes("w-full gap-2"):
            ui.button("Front matter only (fast, no GPU)",
                     on_click=lambda: choose("front_matter")).classes("w-full").props("flat")
            ui.button("Read every chapter (hours of GPU)",
                     on_click=lambda: choose("all")).classes("w-full").props("flat")

            def toggle_specific() -> None:
                chapters_input.set_visibility(True)

            ui.button("Specific chapters...", on_click=toggle_specific).classes("w-full").props("flat")
            with ui.row().classes("w-full justify-end"):
                ui.button("Use these chapters", on_click=lambda: choose("specific")).props(
                    "color=red-9"
                )

    dialog.open()
