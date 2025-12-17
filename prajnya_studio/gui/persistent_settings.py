import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Callable, List

class SettingsManager:
    _observers: List[Callable] = []

    DEFAULT_SETTINGS = {
        "default_batch_folder": "",
        "recursive_default": False,
        "skip_review_default": True,
        "zotero_enabled": True,
        "default_strategy_profile": "research_article_strategy.yaml",
        "ui_theme": "saraswati_light",
        "vlm_parallel_enabled": False,
        "log_level": "INFO",
        "enable_animations": True,
        "terminal_font_size": 13
    }

    @staticmethod
    def settings_path() -> Path:
        """Returns platform-specific settings path."""
        if sys.platform == "win32":
            base = os.environ.get("APPDATA", os.path.expanduser("~"))
            path = Path(base) / "PrajnyaStudio" / "settings.json"
        else:
            path = Path.home() / ".config" / "PrajnyaStudio" / "settings.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def load() -> Dict[str, Any]:
        """Loads settings with schema validation (fallback to defaults)."""
        path = SettingsManager.settings_path()
        if not path.exists():
            return SettingsManager.DEFAULT_SETTINGS.copy()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Simple Schema Validation: Ensure required keys exist
            valid_settings = SettingsManager.DEFAULT_SETTINGS.copy()
            valid_settings.update(data)
            
            # Type safety enforcement could go here
            return valid_settings
        except (json.JSONDecodeError, OSError):
            return SettingsManager.DEFAULT_SETTINGS.copy()

    @staticmethod
    def save(settings: Dict[str, Any]) -> None:
        """Atomically saves settings."""
        path = SettingsManager.settings_path()
        tmp_path = path.with_suffix('.tmp')
        
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            os.replace(tmp_path, path)
            
            # Notify observers
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