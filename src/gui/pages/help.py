#!/usr/bin/env python3
"""
Help — a quick-start guide and FAQ, rewritten for V6. S11.

For biology researchers: this replaces V5's help page rather than porting
it verbatim. V5's copy described its own screens (Batch Commander,
Strategy Lab presets "Fast/Balanced/Accurate", specific OlmOCR model
names) — none of which exist in V6 the same way. Shipping that content in
V6 would answer questions about a different product. The expandable-card
layout is kept (it reads well and V5 built it correctly); every word of
content under it is new.

Strategy Lab's own FAQ entry was updated in S12, when that screen landed —
it is no longer "arriving in a later session".

Session 17 added the three things a first-time user could not get from this
page before: how to LAUNCH the app at all (`run_prajnya_studio.py`, which did
not exist until then), that the model server must be started first (step 0 —
scanning works without it, processing does not), and how to undo a bad
Strategy Lab edit. Content here is verified against the code, not remembered:
the shortcuts match `app.py`'s handler, the button names match
`strategy_lab.py`, and the port matches `run_prajnya_studio.DEFAULT_PORT`.

**There is no Support section, and no contact links.** V5's had a GitHub URL
and an email address that were placeholders, not real destinations — shipping
fabricated contact information is worse than shipping none. S11 replaced it
with a pointer to CLAUDE.md and `docs/gui_plan.md`, which was no better in a
shipped app: those are a development history and a sprint plan, written for
whoever is building this, not for someone trying to read a paper. Session 17
replaced the pointer with **actual troubleshooting** — the failures that have
really occurred on this machine (wrong interpreter, busy port, model server
not started, a model left resident making everything 3x slower, a stuck ⚠, a
bad Strategy Lab edit), each with the action that resolves it. If a future
entry cannot name a failure someone has actually hit, it does not belong here.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from nicegui import ui

from .. import theme

QUICK_START_STEPS: List[Dict[str, object]] = [
    {
        "step": 0, "title": "Start the model server", "location": "A terminal, before anything else",
        "details": [
            "Prajnya Studio reads pages with an AI model that runs on your "
            "own machine. That model server has to be running first",
            "llama-server.exe --models-preset config\\vlm_profiles\\model.ini "
            "--models-max 1",
            "Leave it running. Scanning a folder works without it; actually "
            "processing pages does not",
        ],
    },
    {
        "step": 1, "title": "Point at a folder", "location": "Library / Batch",
        "details": [
            "Type the folder's path — this app runs on the same machine as "
            "your PDFs, so there's no upload step",
            "Click \"Scan Folder\" — every PDF is classified on the CPU in "
            "about a second each, no GPU used yet",
        ],
    },
    {
        "step": 2, "title": "Review, then start", "location": "Library / Batch",
        "details": [
            "The pre-flight table shows each file's type, confidence, and "
            "page count before anything is processed",
            "A book in the folder prompts for how much of it to read — "
            "front matter only is the safe, fast default",
            "Click \"Start Batch\"; the live transcript shows what's "
            "happening as it happens",
        ],
    },
    {
        "step": 3, "title": "Read the results", "location": "Results",
        "details": [
            "Pick a processed document to see its metadata, page text, "
            "figures and tables",
            "A ⚠ next to a document means its metadata is provisional — "
            "see the Review Queue",
        ],
    },
    {
        "step": 4, "title": "Manage the library", "location": "Ledger / Review Queue",
        "details": [
            "Ledger: search, filter, and export every processed document "
            "to CSV or BibTeX",
            "Review Queue: correct anything flagged incomplete — a "
            "missing DOI, an unclear author list",
            "Chapter Map (books only): see every chapter's authors and "
            "start page, and read one chapter on demand",
        ],
    },
]

FAQ_ITEMS: List[Dict[str, str]] = [
    {
        "q": "How do I start Prajnya Studio next time?",
        "a": "From the repository folder, run  run_prajnya_studio.py  with the "
             "project's Python. It serves at http://127.0.0.1:8850 and opens a "
             "browser tab for you. Useful flags: --port to move it, --no-browser "
             "to skip opening a tab, --verbose for detailed logs. The port is "
             "8850 rather than the more usual 8080 because 8080 is the model "
             "server's — if the port you ask for is already busy, the launcher "
             "says so and stops rather than starting and quietly serving "
             "nothing.",
    },
    {
        "q": "What's the difference between this and the command line?",
        "a": "Nothing, underneath. run_prajnya.py (command line) and "
             "run_prajnya_studio.py (this interface) both call the same "
             "pipeline code, so a paper processed either way gives the same "
             "result. The command line is better for scripting and unattended "
             "batches; this is better for looking at what came out.",
    },
    {
        "q": "What file types are supported?",
        "a": "PDF only, research articles, books, and book chapters. The "
             "pipeline classifies which of the three it's looking at "
             "automatically.",
    },
    {
        "q": "How long does processing take?",
        "a": "It depends heavily on how much of the page needs the GPU. A "
             "typical 8-25 page article is a few minutes; a 345-page book "
             "read front-matter-only is under a minute, since most of its "
             "pages never reach the GPU at all. Reading every chapter of a "
             "full volume is hours, which is why that's never the default.",
    },
    {
        "q": "Where are results saved?",
        "a": "run_outputs/documents/ for each paper's JSON, run_outputs/figures/ "
             "and run_outputs/tables/ for crops, and persistence/semantic_ledger.csv "
             "for the searchable record of everything ever processed. Settings can "
             "point all three somewhere else.",
    },
    {
        "q": "Can I use a dark theme?",
        "a": "Yes — Settings, under Appearance, or the quick-settings dialog "
             "(Ctrl+,). Pick saraswati_dark and the whole app changes as you "
             "pick it, before you save; Save is what makes it stick for next "
             "time, and Cancel puts it back. Every screen is built from the "
             "same set of named colours, so both themes cover the whole app "
             "rather than most of it.",
    },
    {
        "q": "A document has a ⚠ next to it — what does that mean?",
        "a": "Its metadata is missing something a citation needs (a research "
             "article needs title, authors, DOI and journal; a book needs its "
             "editors or authors, a publisher or ISBN, and a year). Open the "
             "Review Queue to see exactly what's missing and correct it.",
    },
    {
        "q": "Can I edit extracted metadata?",
        "a": "Yes, two ways: the Ledger screen's per-row Edit dialog for any "
             "document, or the Review Queue for documents specifically flagged "
             "incomplete. Both go through the same correction logic, so neither "
             "can disagree with the other about what \"reviewed\" means.",
    },
    {
        "q": "What if a batch run fails partway through?",
        "a": "Every document that already finished keeps its result — one bad "
             "PDF never loses the rest of the batch. Stop hard-kills only the "
             "document currently running; anything already done is untouched.",
    },
    {
        "q": "Can I choose which VLM model reads my papers?",
        "a": "Yes — Strategy Lab. It edits the same three files the pipeline "
             "reads (model.ini, a VLM profile per model, and the role "
             "assignment in the strategy config), never a separate copy, and "
             "shows a diff before saving anything. Assigning a role to a model "
             "your card can't fit alongside another already-assigned model is "
             "flagged before you save, not after a slow run.",
    },
    {
        "q": "I changed something in Strategy Lab and broke it. How do I get back?",
        "a": "Every tab in Strategy Lab has a \"Restore Defaults\" button, and "
             "the page header has \"Restore All to Factory Defaults\". These "
             "revert to a frozen, known-good copy of the configuration kept in "
             "config/factory_defaults/ — not to your last edit, so you can get "
             "out even several changes deep. A restore is still a change, so it "
             "shows you the exact difference and asks before writing, and the "
             "previous version is backed up either way.",
    },
    {
        "q": "Are there keyboard shortcuts?",
        "a": "Ctrl+L jumps to Library / Batch, Ctrl+G to the Ledger, Ctrl+K to "
             "Strategy Lab, and Ctrl+, opens the quick-settings panel.",
    },
]


#: Real failures, with what each one actually means. Every entry here has
#: happened on this machine — none are hypothetical, and none send the reader
#: to a developer document to work it out for themselves.
TROUBLESHOOTING_ITEMS: List[Dict[str, str]] = [
    {
        "symptom": "\"The wrong Python is running this file\"",
        "fix": "Something other than the project's environment started the "
               "app. Activate the 'shastrasaara' conda environment and run "
               "python run_prajnya_studio.py — or, if activation cannot find "
               "it by name, use 'conda env list' to locate it and name its "
               "python.exe in full. The message shows which Python it "
               "actually got.",
    },
    {
        "symptom": "\"Port 8850 is already in use — refusing to start\"",
        "fix": "Prajnya Studio is probably already running in another window "
               "or browser tab; look there first. Otherwise start it on a "
               "different port with --port 8851.",
    },
    {
        "symptom": "A batch starts, then every document fails",
        "fix": "The model server almost certainly isn't running. Scanning a "
               "folder works without it, so the problem only appears once "
               "real work begins. Start llama-server (see step 0 of the quick "
               "start) and run the batch again — anything already finished "
               "is kept.",
    },
    {
        "symptom": "Processing is far slower than usual, but the results look fine",
        "fix": "A model left loaded on the graphics card by a previous run "
               "makes the next one spill into ordinary memory — roughly three "
               "times slower, with identical output, so nothing looks wrong "
               "except the clock. Restart the model server between long runs, "
               "and start it with --models-max 1.",
    },
    {
        "symptom": "A document shows ⚠ and won't clear",
        "fix": "Something a citation needs is genuinely missing from the PDF. "
               "Open the Review Queue and type it in — a hand-entered value is "
               "recorded as user-corrected and is trusted from then on.",
    },
    {
        "symptom": "Strategy Lab changes made things worse",
        "fix": "Use \"Restore All to Factory Defaults\" on that screen. It "
               "returns the configuration to its known-good state, not merely "
               "to your last edit, and shows you exactly what it will change "
               "before doing it.",
    },
]


# --- rendering (after the content data, per CLAUDE.md) -----------------------

def mount() -> None:
    with ui.column().classes("w-full max-w-4xl gap-4"):
        ui.label("Help").style(
            f'font-family: var(--prajnya-font-display); font-size: 1.5rem; '
            f'font-weight: 600; color: {theme.var("indigo")};'
        )
        ui.label("A quick start, and answers to common questions.").classes(
            "text-sm"
        ).style(f'color: {theme.var("text_muted")};')

        _expandable_card("📘", "Quick Start", "Four steps from folder to answers",
                         _build_quick_start)
        _expandable_card("❔", "Frequently Asked Questions",
                         "Common questions, answered for how V6 actually works",
                         _build_faq)
        _expandable_card("🔧", "Something went wrong",
                         "The failures that actually happen, and what to do", _build_troubleshooting)


def _expandable_card(icon: str, title: str, subtitle: str,
                     content_builder: Callable[[], None]) -> None:
    with ui.expansion(text="", icon=None).classes("w-full").style(
        f'background: {theme.var("card")}; '
        f'border: 1px solid {theme.var("border")}; border-radius: 8px;'
    ) as expansion:
        with expansion.add_slot("header"):
            with ui.row().classes("w-full items-center gap-3 py-2"):
                ui.label(icon).classes("text-2xl")
                with ui.column().classes("gap-0"):
                    ui.label(title).classes("text-lg font-bold").style(
                        f'color: {theme.var("indigo")};'
                    )
                    ui.label(subtitle).classes("text-sm").style(
                        f'color: {theme.var("text_muted")};'
                    )
        with ui.column().classes("w-full gap-3 pt-2 pb-4 px-2"):
            content_builder()


def _build_quick_start() -> None:
    for step in QUICK_START_STEPS:
        with ui.card().classes("w-full").style(
            f'background: {theme.var("canvas")}; '
            f'border-left: 4px solid {theme.var("indigo")}; padding: 16px;'
        ):
            with ui.row().classes("items-center gap-3 mb-2"):
                ui.label(str(step["step"])).classes(
                    "w-8 h-8 rounded-full flex items-center justify-center "
                    "text-white font-bold"
                ).style(f'background: {theme.var("indigo")};')
                with ui.column().classes("gap-0"):
                    ui.label(step["title"]).classes("font-bold").style(
                        f'color: {theme.var("text")};'
                    )
                    ui.label(step["location"]).classes("text-xs").style(
                        f'color: {theme.var("text_muted")};'
                    )
            with ui.column().classes("ml-11 gap-1"):
                for detail in step["details"]:
                    with ui.row().classes("items-start gap-2"):
                        ui.label("•").style(f'color: {theme.var("indigo")};')
                        ui.label(detail).classes("text-sm").style(
                            f'color: {theme.var("text")};'
                        )


def _build_faq() -> None:
    for item in FAQ_ITEMS:
        with ui.expansion(item["q"]).classes("w-full").style(
            f'background: {theme.var("canvas")}; border-radius: 4px;'
        ):
            ui.label(item["a"]).classes("text-sm py-2").style(
                f'color: {theme.var("text")};'
            )


def _build_troubleshooting() -> None:
    for item in TROUBLESHOOTING_ITEMS:
        with ui.column().classes("w-full gap-1"):
            ui.label(item["symptom"]).classes("text-sm font-bold").style(
                f'color: {theme.var("vermillion_dark")};'
            )
            ui.label(item["fix"]).classes("text-sm").style(
                f'color: {theme.var("text")};'
            )
