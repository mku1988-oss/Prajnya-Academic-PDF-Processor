#!/usr/bin/env python3
"""
Strategy Lab — the model layer, editable. S12, the last screen and the
heart of the tool (the user's own standing correction, session 13 — see
CLAUDE.md; an earlier plan draft said "skip it, ship a read-only panel"
and was rejected).

For biology researchers: every other screen in this app shows you what the
pipeline already did. This one is different — it changes what the pipeline
DOES next, by editing the three real files that decide which AI model reads
which kind of page. There is no separate "Strategy Lab format": you are
looking at, and saving, the exact files the command line also reads.

**Three tabs, one per layer a model swap touches** (`docs/gui_plan.md` §4):

1. **Role Assignment** — `profile_registry` inside
   `research_article_strategy.yaml`. Six roles (`default`,
   `title_detection`, `extraction`, `layout_detector`, `figure_escalation`,
   `understanding`); each points at a profile name from tab 3, or is empty
   (an optional role, turned off). Written with a SURGICAL line edit
   (`strategy_editor.write_profile_registry_roles`) so the file's many other
   comments — the offline-roster explanation, the escalation cost/benefit
   note — are never touched.
2. **Model Presets** (`model.ini`) — the llama-server launch parameters
   (weight paths, `n-gpu-layers`, `ctx-size`, sampling) for every model
   this machine can run. Edited as raw text, not a generated form: a new
   model may need flags this file has never seen before, and a form can
   only offer flags it already knows about.
3. **VLM Profiles** — the pipeline-facing description of a model
   (`prompt_mode`, VRAM, `fixed_prompts`, capabilities). Also raw text, for
   the same reason, plus a "Duplicate as new model" flow — the plan's
   explicit answer to "point the tool at a model this codebase has never
   heard of, without editing Python."

**Every save shows a diff before writing, and every write backs up the file
first.** A full YAML/INI round-trip loses comments the writer doesn't know
to keep; the diff is how a human notices before it happens, and the backup
is the undo button if they don't. See `strategy_editor.py`'s module
docstring for the full reasoning — this file only calls it.

**What this page does NOT do:** touch `src/`. Nothing here imports a model
name into Python — it edits the same YAML/INI the pipeline already reads at
runtime. That is the whole point of the VLM-agnostic rule (CLAUDE.md): the
pipeline outlives its models, and this editor is how a human tells it about
a new one.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import yaml
from nicegui import ui

from .. import strategy_editor as se
from .. import theme

logger = logging.getLogger("prajnya_studio.strategy_lab")

#: `layout_detector` accepts this literal in addition to a profile name — it
#: selects the CPU detector (PP-DocLayoutV3) rather than a VLM.
_LAYOUT_DETECTOR_CPU_OPTION = "onnx"


def mount() -> None:
    with ui.column().classes("w-full max-w-5xl gap-6"):
        with ui.row().classes("w-full items-start justify-between"):
            with ui.column().classes("gap-1"):
                ui.label("Strategy Lab").style(
                    f'font-family: var(--prajnya-font-display); font-size: 1.5rem; '
                    f'font-weight: 600; color: {theme.var("indigo")};'
                )
                ui.label(
                    "Which model reads which kind of page — and the two files that "
                    "make a new model assignable in the first place. Every save "
                    "shows a diff first and backs up the file it touches."
                ).classes("text-sm").style(f'color: {theme.var("text_muted")};')
            ui.button("Restore All to Factory Defaults", icon="settings_backup_restore",
                     on_click=_confirm_restore_all).props("flat").style(
                f'color: {theme.var("error")};'
            )

        with ui.tabs().classes("w-full") as tabs:
            role_tab = ui.tab("Role Assignment")
            presets_tab = ui.tab("Model Presets (model.ini)")
            profiles_tab = ui.tab("VLM Profiles")

        with ui.tab_panels(tabs, value=role_tab).classes("w-full"):
            with ui.tab_panel(role_tab):
                _mount_role_assignment()
            with ui.tab_panel(presets_tab):
                _mount_model_presets()
            with ui.tab_panel(profiles_tab):
                _mount_vlm_profiles()


# =============================================================================
# Tab 1 — Role Assignment
# =============================================================================

def _mount_role_assignment() -> None:
    current_roles = se.read_profile_registry()
    ini_sections = se.model_ini_sections(se.read_model_ini())
    profile_names = se.list_vlm_profiles()
    card_vram = se.detect_gpu_vram_gb()

    selects: Dict[str, ui.select] = {}

    with ui.column().classes("w-full gap-4"):
        if card_vram:
            ui.label(f"Detected GPU: {card_vram:.1f} GB (nvidia-smi)").classes(
                "text-xs"
            ).style(f'color: {theme.var("text_muted")};')
        else:
            ui.label(
                "GPU VRAM could not be detected (nvidia-smi unavailable) — "
                "co-residency warnings are skipped without it."
            ).classes("text-xs").style(f'color: {theme.var("warning")};')

        # Built before the role rows so each select's on_value_change closure
        # can already close over it — but `.move()`d below them afterwards
        # (nicegui appends new elements to the end of their parent, so a
        # container created first renders first unless repositioned) so it
        # reads roles-first, warnings-below, right above the Save button.
        warnings_column = ui.column().classes("w-full gap-2 mt-2")

        for role in se.ROLES:
            _role_row(role, current_roles.get(role, ""), profile_names, ini_sections,
                     selects, warnings_column, card_vram)

        warnings_column.move(target_index=-1)

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Restore Defaults", on_click=_confirm_restore_registry).props("flat").style(
                f'color: {theme.var("error")};'
            )
            ui.button(
                "Save Role Assignments",
                on_click=lambda: _save_roles(selects, current_roles),
            ).style(
                f'background: {theme.var("vermillion")} !important; '
                f'color: {theme.var("on_accent")} !important; font-weight: 600;'
            )

    # Computed once up front too, not only on the first change — an already-
    # unfit pair in the CURRENT config (this session found exactly one: glm-ocr
    # + olmocr-2-7b, 10.1 GB on an 8 GB card, never declared co-resident) must
    # be visible on arrival, not hidden until a user happens to touch a select.
    _refresh_role_warnings(ini_sections, selects, warnings_column, card_vram)


def _role_row(role: str, current_value: str, profile_names: List[str],
             ini_sections: List[str], selects: Dict[str, ui.select],
             warnings_column: ui.column, card_vram: Optional[float]) -> None:
    options = list(profile_names)
    if role == "layout_detector":
        options = [_LAYOUT_DETECTOR_CPU_OPTION] + options
    if role in se.OPTIONAL_ROLES:
        options = [""] + options

    with ui.card().classes("w-full p-4").style(f'border: 1px solid {theme.var("border")};'):
        with ui.row().classes("w-full items-start gap-4 no-wrap"):
            with ui.column().classes("gap-0").style("width: 180px;"):
                ui.label(role).style(
                    f'font-weight: 600; color: {theme.var("indigo")};'
                )
                if role in se.OPTIONAL_ROLES:
                    ui.label("optional").classes("text-xs").style(
                        f'color: {theme.var("text_muted")};'
                    )
            select = ui.select(
                options, value=current_value if current_value in options else "",
            ).classes("flex-1").props("outlined dense")
            selects[role] = select

        ui.label(se.ROLE_DESCRIPTIONS[role]).classes("text-xs mt-1").style(
            f'color: {theme.var("text_muted")};'
        )

        select.on_value_change(
            lambda: _refresh_role_warnings(ini_sections, selects, warnings_column, card_vram)
        )


def _refresh_role_warnings(ini_sections: List[str], selects: Dict[str, ui.select],
                           warnings_column: ui.column, card_vram: Optional[float]) -> None:
    """Recomputes every warning from the CURRENT selector state, live —
    not just the one role that just changed, since a VRAM warning is about
    a PAIR and either side of the pair might be what moved."""
    warnings_column.clear()

    role_profiles: Dict[str, Dict[str, Any]] = {}
    messages: List[str] = []

    for a_role, select in selects.items():
        name = select.value
        if not name or (a_role == "layout_detector" and name == _LAYOUT_DETECTOR_CPU_OPTION):
            continue
        try:
            profile = se.load_vlm_profile(name)
        except FileNotFoundError:
            messages.append(f"'{a_role}': profile '{name}' could not be loaded.")
            continue
        role_profiles[a_role] = profile

        ini_error = se.validate_model_name_matches_ini(profile, ini_sections)
        if ini_error:
            messages.append(f"'{a_role}': {ini_error}")

        missing_tasks = se.validate_fixed_prompts_for_role(a_role, profile)
        if missing_tasks:
            messages.append(
                f"'{a_role}': '{name}' has prompt_mode: fixed but no fixed_prompts "
                f"entry for {', '.join(missing_tasks)} — this role will fail at "
                f"its first real call."
            )

    messages.extend(se.compute_vram_warnings(role_profiles, card_vram))

    with warnings_column:
        if not messages:
            ui.label("No conflicts detected for the current selection.").classes(
                "text-sm"
            ).style(f'color: {theme.var("success")};')
        for message in messages:
            with ui.row().classes("items-start gap-2"):
                ui.icon("warning").style(f'color: {theme.var("warning")};')
                ui.label(message).classes("text-sm").style(
                    f'color: {theme.var("text")};'
                )


def _save_roles(selects: Dict[str, ui.select], original_roles: Dict[str, str]) -> None:
    changes: Dict[str, Optional[str]] = {}
    for role, select in selects.items():
        new_value = select.value or None
        old_value = original_roles.get(role)
        if new_value != old_value:
            changes[role] = new_value

    if not changes:
        ui.notify("Nothing changed.", type="info")
        return

    diff = se.preview_profile_registry_diff(changes)
    _confirm_diff_dialog(
        title="Save role assignments?",
        diff=diff,
        on_confirm=lambda: _commit_roles(changes),
    )


def _commit_roles(changes: Dict[str, Optional[str]]) -> None:
    backup = se.write_profile_registry_roles(changes)
    ui.notify(f"Saved. Previous version backed up to {backup.name}.", type="positive")
    ui.navigate.reload()


# =============================================================================
# Tab 2 — Model Presets (model.ini)
# =============================================================================

def _mount_model_presets() -> None:
    with ui.column().classes("w-full gap-3"):
        ui.label(
            "Raw text, not a generated form — a new model may need a flag this "
            "editor has never seen. Every [section] name here is what a VLM "
            "profile's model_name must match."
        ).classes("text-sm").style(f'color: {theme.var("text_muted")};')

        text_area = ui.textarea(value=se.read_model_ini_text()).classes(
            "w-full font-mono"
        ).props("outlined rows=28")

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Restore Defaults", on_click=_confirm_restore_model_ini).props("flat").style(
                f'color: {theme.var("error")};'
            )
            ui.button(
                "Save model.ini",
                on_click=lambda: _save_model_ini(text_area),
            ).style(
                f'background: {theme.var("vermillion")} !important; '
                f'color: {theme.var("on_accent")} !important; font-weight: 600;'
            )


def _confirm_restore_model_ini() -> None:
    diff = se.preview_model_ini_restore_diff()
    if not diff.strip():
        ui.notify("model.ini already matches the factory default.", type="info")
        return

    def commit() -> None:
        backup = se.restore_model_ini_default()
        ui.notify(f"Restored. Previous version backed up to {backup.name}.", type="positive")
        ui.navigate.reload()

    _confirm_diff_dialog(title="Restore model.ini to its factory default?", diff=diff, on_confirm=commit)


def _save_model_ini(text_area: ui.textarea) -> None:
    proposed = text_area.value
    diff = se.preview_text_diff(se.read_model_ini_text(), proposed, "model.ini")

    if not diff.strip():
        ui.notify("Nothing changed.", type="info")
        return

    def commit() -> None:
        try:
            backup = se.save_model_ini_text(proposed)
        except Exception as error:  # noqa: BLE001 — invalid INI, refused before writing
            ui.notify(f"Not saved — invalid INI: {error}", type="negative")
            return
        ui.notify(f"Saved. Previous version backed up to {backup.name}.", type="positive")
        ui.navigate.reload()

    _confirm_diff_dialog(title="Save model.ini?", diff=diff, on_confirm=commit)


# =============================================================================
# Tab 3 — VLM Profiles
# =============================================================================

def _mount_vlm_profiles() -> None:
    profile_names = se.list_vlm_profiles()

    with ui.column().classes("w-full gap-3"):
        with ui.row().classes("w-full items-center gap-3"):
            picker = ui.select(profile_names, value=profile_names[0] if profile_names else None,
                              label="Profile").classes("w-64").props("outlined dense")
            # A stable, TOP-LEVEL anchor for the duplicate flow's dialogs.
            # nicegui binds each event handler to the slot it was DEFINED
            # in, so a "Duplicate" -> "Next: review" chain that builds its
            # second dialog directly inside the first dialog's own card
            # ends up a Python-tree descendant of an already-closed
            # (v-if=false) dialog — it builds with no exception, but never
            # mounts client-side (a real S12 click-through finding: the
            # debug notifications this bug was chased down with all fired
            # cleanly, right up to "dialog function returned"). Anchoring
            # both dialogs to a container captured here, at page-build
            # time, keeps them siblings of this row instead of nested
            # inside one another.
            dialog_anchor = ui.column()
            ui.button("Duplicate as new model", icon="content_copy",
                     on_click=lambda: _open_duplicate_dialog(profile_names, picker, dialog_anchor))

        editor_container = ui.column().classes("w-full")
        _render_profile_editor(editor_container, picker.value, profile_names)

        picker.on_value_change(
            lambda e: _render_profile_editor(editor_container, e.value, profile_names)
        )


def _render_profile_editor(container: ui.column, name: Optional[str],
                           profile_names: List[str]) -> None:
    container.clear()
    if not name:
        with container:
            ui.label("No profiles found in config/vlm_profiles/.").classes("text-sm")
        return

    with container:
        ini_sections = se.model_ini_sections(se.read_model_ini())
        text_area = ui.textarea(value=se.load_vlm_profile_text(name)).classes(
            "w-full font-mono"
        ).props("outlined rows=24")

        live_status = ui.label("").classes("text-sm")

        def revalidate() -> None:
            try:
                parsed = yaml.safe_load(text_area.value) or {}
            except yaml.YAMLError as error:
                live_status.text = f"Invalid YAML: {error}"
                live_status.style(f'color: {theme.var("error")};')
                return
            problems = []
            ini_error = se.validate_model_name_matches_ini(parsed, ini_sections)
            if ini_error:
                problems.append(ini_error)
            # This profile's own required tasks are unknown here (a profile
            # can serve several roles); the Role Assignment tab is where a
            # SPECIFIC role's fixed_prompts gap is checked. This tab only
            # confirms the file parses and its model_name is real.
            live_status.text = "; ".join(problems) if problems else "Parses OK."
            live_status.style(
                f'color: {theme.var("error") if problems else theme.var("success")};'
            )

        text_area.on_value_change(lambda e: revalidate())
        revalidate()

        with ui.row().classes("w-full justify-end gap-2"):
            if se.has_factory_default_profile(name):
                ui.button("Restore Defaults", on_click=lambda: _confirm_restore_profile(name)
                         ).props("flat").style(f'color: {theme.var("error")};')
            else:
                ui.label(f"{name}.yaml has no factory default (created after the "
                        f"baseline was frozen — e.g. via Duplicate).").classes(
                    "text-xs self-center"
                ).style(f'color: {theme.var("text_muted")};')
            ui.button(
                f"Save {name}.yaml",
                on_click=lambda: _save_profile(name, text_area),
            ).style(
                f'background: {theme.var("vermillion")} !important; '
                f'color: {theme.var("on_accent")} !important; font-weight: 600;'
            )


def _confirm_restore_profile(name: str) -> None:
    diff = se.preview_vlm_profile_restore_diff(name)
    if not diff.strip():
        ui.notify(f"{name}.yaml already matches its factory default.", type="info")
        return

    def commit() -> None:
        backup = se.restore_vlm_profile_default(name)
        ui.notify(f"Restored. Previous version backed up to {backup.name}.", type="positive")
        ui.navigate.reload()

    _confirm_diff_dialog(title=f"Restore {name}.yaml to its factory default?",
                         diff=diff, on_confirm=commit)


def _save_profile(name: str, text_area: ui.textarea) -> None:
    proposed = text_area.value
    try:
        yaml.safe_load(proposed)
    except yaml.YAMLError as error:
        ui.notify(f"Not saved — invalid YAML: {error}", type="negative")
        return

    diff = se.preview_text_diff(se.load_vlm_profile_text(name), proposed, f"{name}.yaml")
    if not diff.strip():
        ui.notify("Nothing changed.", type="info")
        return

    def commit() -> None:
        backup = se.save_vlm_profile_text(name, proposed)
        ui.notify(f"Saved. Previous version backed up to {backup.name}.", type="positive")
        ui.navigate.reload()

    _confirm_diff_dialog(title=f"Save {name}.yaml?", diff=diff, on_confirm=commit)


def _open_duplicate_dialog(profile_names: List[str], picker: ui.select,
                           dialog_anchor: ui.column) -> None:
    with ui.dialog() as dialog, ui.card().classes("w-[560px] p-6 gap-3"):
        ui.label("Duplicate a profile as the template for a new model").style(
            f'font-weight: 600; color: {theme.var("indigo")};'
        )
        source_select = ui.select(profile_names, value=picker.value,
                                  label="Clone from").classes("w-full").props("outlined dense")
        new_name_input = ui.input("New profile name (no extension)").classes("w-full").props(
            "outlined dense"
        )
        model_name_input = ui.input(
            "model_name (must eventually match a [section] in model.ini)"
        ).classes("w-full").props("outlined dense")

        error_label = ui.label("").classes("text-sm").style(f'color: {theme.var("error")};')

        def create() -> None:
            new_name = new_name_input.value.strip()
            model_name = model_name_input.value.strip()
            if not new_name or not model_name:
                error_label.text = "Both fields are required."
                return
            try:
                cloned = se.duplicate_vlm_profile(source_select.value, new_name, model_name)
            except (FileNotFoundError, ValueError) as error:
                error_label.text = str(error)
                return
            starter_text = se.render_profile_yaml(cloned)
            dialog.close()
            # Built explicitly inside dialog_anchor, NOT called bare: this
            # handler is bound to the slot it was DEFINED in (dialog A's own
            # card), so a bare call here would make the review dialog a
            # Python-tree child of dialog A — which just closed (v-if now
            # false client-side). It would build with no exception and
            # never appear, a real bug this session's click-through caught
            # (every debug notification fired cleanly, right up to
            # "function returned", with nothing ever rendering). The
            # explicit `with dialog_anchor:` overrides the captured slot for
            # this block, anchoring the new dialog as dialog_anchor's child
            # instead — a container that was never inside another dialog.
            with dialog_anchor:
                _open_new_profile_review_dialog(new_name, starter_text, profile_names, picker)

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Next: review", on_click=create)
    dialog.open()


def _open_new_profile_review_dialog(new_name: str, starter_text: str,
                                    profile_names: List[str], picker: ui.select) -> None:
    """The clone is NOT written yet — this dialog is where the user edits
    the starter text (at minimum, model.ini needs a matching [section] added
    on the Model Presets tab, which this profile alone cannot guarantee) and
    explicitly saves. Nothing reaches disk from the duplicate flow alone."""
    with ui.dialog() as dialog, ui.card().classes("w-[720px] p-6 gap-3"):
        ui.label(f"Review new profile: {new_name}.yaml").style(
            f'font-weight: 600; color: {theme.var("indigo")};'
        )
        ui.label(
            "Nothing is saved yet. Remember to add a matching [section] for this "
            "model_name on the Model Presets tab before assigning this profile "
            "to a role."
        ).classes("text-xs").style(f'color: {theme.var("warning")};')
        text_area = ui.textarea(value=starter_text).classes("w-full font-mono").props(
            "outlined rows=20"
        )

        def save() -> None:
            try:
                yaml.safe_load(text_area.value)
            except yaml.YAMLError as error:
                ui.notify(f"Not saved — invalid YAML: {error}", type="negative")
                return
            backup = se.save_vlm_profile_text(new_name, text_area.value)
            dialog.close()
            ui.notify(f"Created {new_name}.yaml (backup path recorded: {backup}).",
                     type="positive")
            ui.navigate.reload()

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Discard", on_click=dialog.close).props("flat")
            ui.button("Create Profile", on_click=save).style(
                f'background: {theme.var("vermillion")} !important; '
                f'color: {theme.var("on_accent")} !important; font-weight: 600;'
            )
    dialog.open()


# =============================================================================
# Factory defaults — the idiot-proof reset
# =============================================================================
#
# V5 had a "Restore Factory Settings" button backed by a pristine
# `research_article_strategy.FACTORY_DEFAULT.yaml`; every save-to-defaults
# path here is a thin UI shim over the equivalent V6 mechanism in
# `strategy_editor.py` (`config/factory_defaults/`, frozen from the S12-gate
# state). Each restore goes through the SAME `_confirm_diff_dialog` every
# other save uses — a revert is still a write, and still deserves the same
# "see it before it happens, and it's backed up either way" guarantee.

def _confirm_restore_registry() -> None:
    diff = se.preview_profile_registry_restore_diff()
    if not diff.strip():
        ui.notify("Role assignments already match the factory default.", type="info")
        return

    def commit() -> None:
        backup = se.restore_profile_registry_default()
        ui.notify(f"Restored. Previous version backed up to {backup.name}.", type="positive")
        ui.navigate.reload()

    _confirm_diff_dialog(title="Restore role assignments to their factory default?",
                         diff=diff, on_confirm=commit)


def _confirm_restore_all() -> None:
    diff = _build_restore_all_preview()
    if not diff.strip():
        ui.notify("Everything already matches the factory defaults.", type="info")
        return

    def commit() -> None:
        backups = se.restore_all_defaults()
        changed = ", ".join(sorted(backups)) or "nothing"
        ui.notify(f"Restored: {changed}.", type="positive")
        ui.navigate.reload()

    _confirm_diff_dialog(title="Restore EVERYTHING (model.ini, every VLM profile, "
                              "and role assignments) to factory defaults?",
                         diff=diff, on_confirm=commit)


def _build_restore_all_preview() -> str:
    """Concatenates every non-empty restore diff into one preview — what the
    global button's confirmation dialog shows, so a user sees the FULL blast
    radius of "Restore All" before confirming it, not just a file count."""
    sections: List[str] = []

    ini_diff = se.preview_model_ini_restore_diff()
    if ini_diff.strip():
        sections.append(ini_diff)

    for name in se.list_vlm_profiles():
        if not se.has_factory_default_profile(name):
            continue
        profile_diff = se.preview_vlm_profile_restore_diff(name)
        if profile_diff.strip():
            sections.append(profile_diff)

    registry_diff = se.preview_profile_registry_restore_diff()
    if registry_diff.strip():
        sections.append(registry_diff)

    return "\n".join(sections)


# =============================================================================
# Shared: diff-preview confirmation
# =============================================================================

def _confirm_diff_dialog(title: str, diff: str, on_confirm: Callable[[], None]) -> None:
    """Every save in this page routes through here — the one place a diff is
    shown before a write happens, so no path can silently skip the review."""
    with ui.dialog() as dialog, ui.card().classes("w-[720px] p-6 gap-3"):
        ui.label(title).style(f'font-weight: 600; color: {theme.var("indigo")};')
        with ui.scroll_area().classes("w-full").style("height: 400px;"):
            ui.markdown(f"```diff\n{diff}\n```").classes("w-full")
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            def confirm() -> None:
                dialog.close()
                on_confirm()

            ui.button("Confirm & Save", on_click=confirm, color="negative")
    dialog.open()
