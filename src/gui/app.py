#!/usr/bin/env python3
"""
Prajnya Studio (V6) — the GUI shell: theme, sidebar, routing, storage.

For biology researchers: run this file to open the graphical interface
instead of the command line. It is the same pipeline underneath —
`run_prajnya.py` and this file both end up calling `src/processing/runner.py`
(the GUI's call arrives through a subprocess worker, wired up in S6) — this
file only decides what you see and click.

**What S5 builds, and what it deliberately does not.** Every sidebar route
exists so later sessions never have to touch navigation again; every page
except Settings' quick-modal is a placeholder naming the session that fills
it in (see `pages/*.py`). No document has ever been processed through this
file — that starts in S6, once `controller.py` exists.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from nicegui import app, ui

from src.processing.paths import DEFAULT_OUTPUT_ROOT, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

from . import components, icons, persistent_settings, theme  # noqa: E402  (after load_dotenv, deliberately)
from .pages import chapters, help as help_page, ledger, library, results, review, settings, strategy_lab
from .session import RunRegistry  # noqa: F401  (imported so pages can reach it; used from S6)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prajnya_studio")

#: nicegui's app.storage.user needs a stable key to encrypt its cookie. A
#: fresh random one every process start would silently drop a user's active
#: run id on the next server restart, so this comes from .env (see the
#: comment beside SARA_GUI_STORAGE_SECRET there) rather than being generated
#: here.
_STORAGE_SECRET = os.getenv("SARA_GUI_STORAGE_SECRET")
if not _STORAGE_SECRET:
    logger.warning(
        "SARA_GUI_STORAGE_SECRET is not set — app.storage.user will use a "
        "secret generated fresh each run, so it will not survive a restart. "
        "Set it in .env for a real session."
    )
    import secrets
    _STORAGE_SECRET = secrets.token_hex(32)

app.add_static_files("/static", str(theme.ASSETS_DIR))

# The Results screen (S7) serves figure and table crops straight out of a
# run's output tree, using each manifest's image_path AS WRITTEN — those
# paths are already relative to the run root by design (see paths.py's
# anchor_output_path), so this mount is the browser-facing half of that
# same design, not a second path scheme. Created here rather than assumed
# to exist: a fresh checkout has no run_outputs/ until something is
# processed, and add_static_files requires the directory to be there first.
DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.add_static_files("/run_outputs", str(DEFAULT_OUTPUT_ROOT))

_NAV_ITEMS = [
    ("Library / Batch", icons.BATCH_COMMANDER, "/library"),
    ("Results", icons.RESULTS_VIEWER, "/results"),
    ("Ledger", icons.LEDGER_MASTER, "/ledger"),
    ("Review Queue", icons.REVIEW_QUEUE, "/review"),
    ("Chapter Map", icons.CHAPTER_MAP, "/chapters"),
    # S11 added the full Settings page (src/gui/pages/settings.py) at
    # /settings, which _PAGE_MOUNTERS already routed — but until this line
    # was added, nothing in the sidebar ever linked to it, and the
    # footer's Settings button only opens the smaller quick-settings
    # modal. The page would have rendered fine at the URL directly, but a
    # real user clicking through the sidebar would never have found it.
    ("Strategy Lab", icons.STRATEGY_LAB, "/strategy-lab"),
    ("Settings", icons.SETTINGS_CHAKRA, "/settings"),
    ("Help", icons.HELP_DIYA, "/help"),
]

_PAGE_MOUNTERS = {
    "/library": library.mount,
    "/results": results.mount,
    "/ledger": ledger.mount,
    "/review": review.mount,
    "/chapters": chapters.mount,
    "/strategy-lab": strategy_lab.mount,
    "/settings": settings.mount,
    "/help": help_page.mount,
}


def create_shell(current_route: str):
    """Builds the sidebar + content frame and returns the content container.

    Args:
        current_route: the route being rendered, so the matching sidebar
            item can be highlighted.

    Returns:
        A `ui.column` context manager — callers `with` into it and mount
        their page's content there.
    """
    ui.add_head_html(f"<style>{theme.SARASWATI_CSS}</style>")

    # The stored theme is applied here, in the head, on every page — not
    # once at startup. Each route is a full page load in this app, so a
    # theme set on one screen has to survive navigating to the next, and the
    # settings file is the only thing that carries it across.
    stored = persistent_settings.SettingsManager.load().get("ui_theme", theme.DEFAULT_THEME)
    ui.add_head_html(theme.theme_head_html(stored))

    ui.add_head_html('''
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    ''')

    _setup_keyboard_shortcuts()

    with ui.row().classes("h-screen w-full no-wrap").style("margin:0; padding:0;"):
        with ui.column().classes("gap-0 items-center").style(
            "width: 260px; "
            "height: 100vh; "
            "padding: 16px 12px; "
            f"background: {theme.var('sidebar')}; "
            f"border-right: 1px solid {theme.var('border')}; "
            "display: flex; "
            "flex-direction: column; "
            "overflow: hidden;"
        ):
            components.lotus_logo(size=120)
            ui.label("Academic PDF Processor").style(
                'font-family: "Inter", sans-serif; '
                "font-size: 11px; "
                f"color: {theme.var('text_muted')}; "
                "margin-top: 2px; "
                "margin-bottom: 16px; "
                "letter-spacing: 0.5px;"
            )

            with ui.column().classes("w-full gap-3 flex-1").style(
                f"background: {theme.var('canvas')}; "
                "border-radius: 8px; "
                "padding: 12px 8px;"
            ):
                for label, icon, route in _NAV_ITEMS:
                    components.sidebar_item(label, icon, route, active=(current_route == route))

            with ui.column().classes("w-full gap-2"):
                ui.separator().style(f"background: {theme.var('border')}; margin: 8px 0;")

                with ui.button(on_click=components.open_settings_modal).classes("w-full group").props(
                    "flat no-caps"
                ).style("padding: 0;"):
                    with ui.row().classes("w-full items-center gap-3 q-py-sm q-px-md"):
                        with ui.element("div").style(
                            "width: 24px; height: 24px; filter: grayscale(100%) opacity(0.7);"
                        ):
                            ui.html(icons.SETTINGS_CHAKRA, sanitize=False)
                        ui.label("Settings").style(
                            f"color: {theme.var('text_muted')}; font-weight: 500;"
                        )
                        ui.space()

                with ui.column().classes("gap-0 items-center").style("padding-top: 4px;"):
                    ui.label("Prajnya Studio V6").classes("text-xs").style(
                        f"color: {theme.var('text_muted')};"
                    )
                    ui.label("Copyright © Manu Unni").classes("text-xs").style(
                        f"color: {theme.var('text_muted')}; opacity: 0.7;"
                    )

        with ui.column().classes("flex-1 h-screen").style(
            f"background: {theme.var('canvas')}; "
            "overflow-y: auto; "
            "position: relative;"
        ):
            with ui.column().classes("w-full h-full p-8").style("padding-top: 32px;") as content:
                pass

    return content


def _setup_keyboard_shortcuts() -> None:
    """Global shortcuts."""

    @ui.keyboard
    def handle_keys(e):
        if not e.action.keydown:
            return
        if e.key == "l" and e.modifiers.ctrl:
            ui.navigate.to("/library")
        elif e.key == "g" and e.modifiers.ctrl:
            ui.navigate.to("/ledger")
        elif e.key == "k" and e.modifiers.ctrl:
            ui.navigate.to("/strategy-lab")
        elif e.key == "," and e.modifiers.ctrl:
            components.open_settings_modal()


def _register_route(route: str) -> None:
    """Wires one URL to the shell + that page's mount() function."""

    @ui.page(route)
    def _page(route=route):  # default arg captures the loop variable
        content = create_shell(current_route=route)
        with content:
            _PAGE_MOUNTERS[route]()


@ui.page("/")
def index():
    ui.navigate.to("/library")


for _route in _PAGE_MOUNTERS:
    _register_route(_route)


def main(host: str = "127.0.0.1", port: int = 8850, show: bool = True) -> None:
    """Start the server. The single place the run options live.

    Called by `run_prajnya_studio.py` at the repository root, which is the
    supported way to launch the GUI — this module uses relative imports, so
    `python src/gui/app.py` cannot work and never could. The `__main__`
    guard below keeps `python -m src.gui.app` working for anyone already
    used to it.

    Args:
        host: interface to bind. The default is loopback ONLY — the app
            reads and writes files anywhere on this machine (it takes a
            folder path as typed, by design), so exposing it to a network
            has to be a deliberate act, not a default.
        port: TCP port to serve on. Deliberately NOT 8080 — that is the
            local llama-server's port (see run_prajnya_studio.py's
            DEFAULT_PORT note). Running this module directly gets no
            port-in-use check; the launcher is the guarded path.
        show: open a browser tab automatically.
    """
    ui.run(
        native=False,
        title="Prajnya Studio V6",
        host=host,
        port=port,
        show=show,
        window_size=(1280, 820),
        storage_secret=_STORAGE_SECRET,
        reload=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
