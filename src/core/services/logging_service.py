"""
Centralized logging configuration.
Ensures consistent logging across CLI and GUI.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

class LoggingService:
    """Centralized logging setup for Prajnya."""

    @staticmethod
    def setup_logging(
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        format_string: Optional[str] = None
    ) -> None:
        """
        Configure logging with consistent format.

        Args:
            level: Logging level (default: INFO)
            log_file: Optional log file path
            format_string: Optional custom format
        """
        if format_string is None:
            format_string = "%(asctime)s - [%(name)s] %(levelname)s - %(message)s"

        handlers = [logging.StreamHandler(sys.stdout)]

        if log_file:
            handlers.append(
                logging.FileHandler(log_file, mode='w', encoding='utf-8')
            )

        logging.basicConfig(
            level=level,
            format=format_string,
            handlers=handlers
        )

        # Suppress harmless Windows asyncio errors
        logging.getLogger('asyncio').setLevel(logging.CRITICAL)
