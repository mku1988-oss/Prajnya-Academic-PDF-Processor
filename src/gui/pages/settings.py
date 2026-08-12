#!/usr/bin/env python3
"""
Settings — the full path-and-policy surface. S11.

For biology researchers: this page decides WHERE a run's output, ledger,
and defaults live — never WHAT the pipeline extracts, or which model does
the extracting. That second kind of setting is Strategy Lab's, arriving
in S12; putting a model name here would break the rule the whole project
stands on (`src/` never names a model — see CLAUDE.md's VLM-agnostic
rule) and would be one more place a future model swap has to remember to
touch.

**Every field here defaults to blank/off, and blank/off means "change
nothing."** `library.py`'s `_apply_settings_overrides` only ever ADDS a
`RunRequest` kwarg when a setting is actually non-empty — never
overwrites a default with an explicit blank — so a user who never opens
this page gets exactly the behaviour V6 always had. That is also why
there is no "Reset to defaults" button: every field already has an
"unset" state that IS the default, one click away (clear the field).

**The quick-settings modal (`components.open_settings_modal`, S5) still
exists and still covers its original four keys** (default folder,
recursive, skip-review, theme) — this page is additive, the fuller
surface `docs/gui_plan.md` section 6 describes, not a replacement.
"""

from __future__ import annotations

from typing import Any, Dict

from nicegui import ui

from src.processing.book_prompt import parse_chapter_selection

from .. import persistent_settings, theme

_BOOK_POLICY_OPTIONS = {
    "": "Inherit from processing_policy.yaml (default: ask)",
    "skip": "Skip — front matter and chapter map only",
    "ask": "Ask — show a dialog for every book",
    "all": "Read every chapter (hours of GPU on a full volume)",
    "chapters": "Read specific chapters (listed below)",
}

_PDF_BACKEND_OPTIONS = {
    "": "pypdfium2 (default, Apache-2.0/BSD-3)",
    "pymupdf": "PyMuPDF (faster, local dev only — AGPL-3.0, never ships)",
}


def mount() -> None:
    settings = persistent_settings.SettingsManager.load()

    with ui.column().classes("w-full max-w-3xl gap-6"):
        ui.label("Settings").style(
            f'font-family: var(--prajnya-font-display); font-size: 1.5rem; '
            f'font-weight: 600; color: {theme.var("indigo")};'
        )
        ui.label("Where a run's output and ledger live, and the defaults it "
                 "starts with. The model layer is Strategy Lab's.").classes(
            "text-sm"
        ).style(f'color: {theme.var("text_muted")};')

        fields: Dict[str, Any] = {}
        _section_paths(fields, settings)
        _section_book_policy(fields, settings)
        _section_processing_defaults(fields, settings)
        _section_appearance(fields, settings)

        with ui.row().classes("w-full justify-end gap-3 mt-2"):
            status_label = ui.label("").classes("text-sm").style(
                f'color: {theme.var("success")};'
            )
            ui.button("Save Settings",
                     on_click=lambda: _save(fields, status_label)).style(
                f'background: {theme.var("vermillion")} !important; '
                f'color: {theme.var("on_accent")} !important; font-weight: 600;'
            )


# --- sections (after mount, per CLAUDE.md) -----------------------------------

def _section(title: str):
    ui.label(title).classes("text-xs font-bold uppercase tracking-wide").style(
        f'color: {theme.var("gold")};'
    )
    return ui.column().classes("w-full gap-3")


def _section_paths(fields: Dict[str, Any], settings: Dict[str, Any]) -> None:
    with _section("Paths & Storage"):
        fields["default_batch_folder"] = ui.input(
            "Default Library Folder", value=settings.get("default_batch_folder", "")
        ).classes("w-full").props("outlined dense")

        fields["output_root"] = ui.input(
            "Output Root", value=settings.get("output_root", ""),
            placeholder="blank = run_outputs/ inside the repository",
        ).classes("w-full").props("outlined dense")

        fields["ledger_path"] = ui.input(
            "Ledger File", value=settings.get("ledger_path", ""),
            placeholder="blank = persistence/semantic_ledger.csv (or SARA_LEDGER_PATH)",
        ).classes("w-full").props("outlined dense")


def _section_book_policy(fields: Dict[str, Any], settings: Dict[str, Any]) -> None:
    with _section("Book Policy (Library/Batch default)"):
        ui.label("Overrides processing_policy.yaml for every batch started from "
                 "Library/Batch. Leave on \"Inherit\" to use the file as-is.").classes(
            "text-xs"
        ).style(f'color: {theme.var("text_muted")};')

        fields["book_policy"] = ui.select(
            _BOOK_POLICY_OPTIONS, value=settings.get("book_policy", ""),
            label="Policy",
        ).classes("w-full").props("outlined dense")

        chapters_input = ui.input(
            "Chapters (e.g. 3 4 5, or 3-5)", value=settings.get("book_chapters", "")
        ).classes("w-full").props("outlined dense")
        fields["book_chapters"] = chapters_input

        def sync_chapters_visibility(value: str) -> None:
            chapters_input.set_visibility(value == "chapters")

        fields["book_policy"].on_value_change(lambda e: sync_chapters_visibility(e.value))
        sync_chapters_visibility(settings.get("book_policy", ""))

        fields["book_max_pages"] = ui.input(
            "Max Body Pages per Book", value=str(settings.get("book_max_pages") or ""),
            placeholder="blank = no ceiling",
        ).classes("w-full").props("outlined dense")


def _section_processing_defaults(fields: Dict[str, Any], settings: Dict[str, Any]) -> None:
    with _section("Processing Defaults"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label("Recursive Search").style(f'color: {theme.var("text")};')
            fields["recursive_default"] = ui.switch(
                value=settings.get("recursive_default", False)
            ).props("color=red-9")

        with ui.row().classes("w-full justify-between items-center"):
            with ui.column().classes("gap-0"):
                ui.label("Force Re-process").style(f'color: {theme.var("text")};')
                ui.label("Re-derive metadata even if the ledger already has this "
                         "document").classes("text-xs").style(
                    f'color: {theme.var("text_muted")};'
                )
            fields["force_default"] = ui.switch(
                value=settings.get("force_default", False)
            ).props("color=red-9")

        fields["front_matter_pages"] = ui.number(
            "Front-Matter Scan Depth (pages)",
            value=settings.get("front_matter_pages", 10), min=1, max=50,
        ).classes("w-full").props("outlined dense")

        fields["pdf_backend"] = ui.select(
            _PDF_BACKEND_OPTIONS, value=settings.get("pdf_backend", ""),
            label="PDF Backend",
        ).classes("w-full").props("outlined dense")


def _section_appearance(fields: Dict[str, Any], settings: Dict[str, Any]) -> None:
    with _section("Appearance"):
        # Two real options now that THEME_DARK exists. Like the quick-settings
        # modal, picking one previews it immediately; Save is what persists it.
        fields["ui_theme"] = ui.select(
            list(theme.THEME_ATTRIBUTES),
            value=theme.resolve_theme(settings.get("ui_theme", theme.DEFAULT_THEME)),
            label="UI Theme",
            on_change=lambda e: ui.run_javascript(theme.apply_theme_js(e.value)),
        ).classes("w-full").props("outlined dense")


# --- saving -------------------------------------------------------------

def _save(fields: Dict[str, Any], status_label: ui.label) -> None:
    settings = persistent_settings.SettingsManager.load()
    settings["default_batch_folder"] = fields["default_batch_folder"].value.strip()
    settings["output_root"] = fields["output_root"].value.strip()
    settings["ledger_path"] = fields["ledger_path"].value.strip()
    settings["book_policy"] = fields["book_policy"].value or ""
    settings["book_chapters"] = _validated_chapters(fields["book_chapters"].value)
    settings["book_max_pages"] = _validated_int(fields["book_max_pages"].value)
    settings["recursive_default"] = bool(fields["recursive_default"].value)
    settings["force_default"] = bool(fields["force_default"].value)
    settings["front_matter_pages"] = int(fields["front_matter_pages"].value or 10)
    settings["pdf_backend"] = fields["pdf_backend"].value or ""
    settings["ui_theme"] = fields["ui_theme"].value

    persistent_settings.SettingsManager.save(settings)
    status_label.text = "Saved."
    ui.notify("Settings saved.", type="positive")


def _validated_chapters(text: str) -> str:
    """Round-trips through the same parser Library uses, so a typo is
    dropped here rather than silently reaching a batch run later."""
    numbers = parse_chapter_selection(text or "")
    return " ".join(str(n) for n in numbers)


def _validated_int(text: str) -> str:
    text = (text or "").strip()
    return text if text.isdigit() else ""
