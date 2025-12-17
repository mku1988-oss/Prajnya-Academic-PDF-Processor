# File: src/__init__.py
"""
Shastrasaara Application - Main Package Initializer.
Configures application-wide settings, primarily logging, based on the
strategy configuration file.
"""
import logging
import sys
import yaml
from pathlib import Path

# --- Configuration-Driven Logging Setup ---

# 1. Define a safe default.
log_level = "INFO"

# 2. Try to load the log level from the central strategy configuration.
try:
    # Build the path to the config file relative to this __init__.py file.
    config_path = Path(__file__).parent / "processing/strategies/config/research_article_strategy.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # Safely get the log_level, defaulting to "INFO" if keys are missing.
        log_level = config.get('pipeline', {}).get('log_level', 'INFO').upper()
    else:
        # This case is for robustness, in case the file is ever moved.
        logging.getLogger(__name__).warning(f"Could not find config file at {config_path}. Defaulting to INFO log level.")

except Exception as e:
    # If anything goes wrong (parsing error, etc.), default safely.
    logging.getLogger(__name__).error(f"Error loading log level from config: {e}. Defaulting to INFO log level.")
    log_level = "INFO"

# 3. Configure the root logger with the determined level.
logging.basicConfig(
    level=log_level,  # <-- THE FIX: Use the variable loaded from YAML.
    format='%(asctime)s - [%(name)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)

# 4. Set higher log levels for noisy libraries to keep output clean.
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('pyzotero').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.INFO)

# 5. Announce the final, effective log level.
logging.getLogger(__name__).info(f"Shastrasaara application logger initialized. Effective log level: {log_level}")