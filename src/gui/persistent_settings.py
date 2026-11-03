"""
Cross-session GUI settings — where things go, not what the pipeline extracts.

For biology researchers: this is a small JSON file on disk (not the ledger,
not the strategy YAML) that remembers your last folder and a few UI toggles
between runs of the app.

Ported from V5's gui/persistent_settings.py with two changes:
  - "PrajnyaStudioV6" is its own settings folder, so this app never reads or
    overwrites V5's live settings.json on the same machine.
  - `vlm_parallel_enabled` is dropped. V6 has no concurrent-VLM code path —
    it was tested in four configurations and every one netted zero or worse
    (see PERFORMANCE.md) — so a switch for it would only invite chasing a
    documented dead end.

**S11 added the seven path-and-policy keys `docs/gui_plan.md` section 6
calls for** — `output_root`, `ledger_path`, `book_policy`, `book_chapters`,
`book_max_pages`, `front_matter_pages`, `pdf_backend`, `force_default` —
each defaulting to its EMPTY/neutral value so a user who never opens
Settings sees no behaviour change: `library.py`'s batch run only overrides
a `RunRequest` field when the setting is actually non-empty. This is the
same rule `default_batch_folder`/`recursive_default` already followed;
S11 just gives it more fields to follow. `overrides=` (per-run, not
per-file) is the documented channel for the book-policy trio — see
`policy.py`'s own precedence docstring — so these never rewrite
`config/processing_policy.yaml`.

**Not stored here, deliberately: any model name, VLM profile, or
`--models-max`.** Settings is "where things go"; the model layer is
Strategy Lab's (S12), per the same section of the plan.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class SettingsManager:
    _observers: List[Callable] = []

    DEFAULT_SETTINGS = {
        "default_batch_folder": "",
        "recursive_default": False,
        "skip_review_default": True,
        "zotero_enabled": True,
        "default_strategy_profile": "research_article_strategy.yaml",
        "ui_theme": "saraswati_light",
        "log_level": "INFO",
        "enable_animations": True,
        "terminal_font_size": 13,
        # --- S11: the path-and-policy surface, gui_plan.md section 6 -------
        "output_root": "",          # "" = the pipeline's own default (run_outputs/)
        "ledger_path": "",          # "" = SARA_LEDGER_PATH, then the default CSV
        "book_policy": "",          # "" = inherit processing_policy.yaml's "ask"
        "book_chapters": "",        # used only when book_policy == "chapters"
        "book_max_pages": "",       # "" = no ceiling
        "front_matter_pages": 10,   # matches DEFAULT_FRONT_MATTER_PAGES
        "pdf_backend": "",          # "" = pypdfium2, the shipping default
        "force_default": False,     # re-process even if the ledger has it
    }

    @staticmethod
    def settings_path() -> Path:
        """Returns the platform-specific settings file path."""
        if sys.platform == "win32":
            base = os.environ.get("APPDATA", os.path.expanduser("~"))
            path = Path(base) / "PrajnyaStudioV6" / "settings.json"
        else:
            path = Path.home() / ".config" / "PrajnyaStudioV6" / "settings.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def load() -> Dict[str, Any]:
        """Loads settings with schema validation (falls back to defaults)."""
        path = SettingsManager.settings_path()
        if not path.exists():
            return SettingsManager.DEFAULT_SETTINGS.copy()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            valid_settings = SettingsManager.DEFAULT_SETTINGS.copy()
            valid_settings.update(data)
            return valid_settings
        except (json.JSONDecodeError, OSError):
            return SettingsManager.DEFAULT_SETTINGS.copy()

    @staticmethod
    def ledger_path() -> Optional[Path]:
        """The Settings "Ledger File", or None to use the default resolution.

        Every GUI page that opens the ledger asks this, so the batch, the
        Ledger page and the Review Queue all read the same file.
        """
        value = (SettingsManager.load().get("ledger_path") or "").strip()
        return Path(value) if value else None

    @staticmethod
    def save(settings: Dict[str, Any]) -> None:
        """Atomically saves settings and notifies registered observers."""
        path = SettingsManager.settings_path()
        tmp_path = path.with_suffix(".tmp")

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)

            os.replace(tmp_path, path)

            for callback in SettingsManager._observers:
                try:
                    callback(settings)
                except Exception:
                    pass
        except OSError as e:
            print(f"Error saving settings: {e}")

    @staticmethod
    def on_change(callback: Callable) -> None:
        """Register a callback to be fired when settings are saved."""
        SettingsManager._observers.append(callback)
