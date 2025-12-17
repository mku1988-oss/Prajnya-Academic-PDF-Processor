# prajnya/logger.py - Sprint 0 Stub
"""
Logging configuration for Prajnya Studio.
Sprint 0: Minimal implementation for GUI startup.
"""

import logging
import sys
from collections import deque
from datetime import datetime

# Global log handler
GLOBAL_LOG_HANDLER = None

class DequeLogHandler(logging.Handler):
    """
    Custom logging handler that stores logs in a deque for GUI display.

    Maintains two deques:
    - Live stream deque (cleared after each poll for terminal display)
    - Full history deque (never cleared, for export)
    """
    def __init__(self, maxlen=2000):
        super().__init__()
        self.deque = deque(maxlen=maxlen)  # For terminal streaming (gets cleared)
        self.history = deque(maxlen=10000)  # For export (never cleared, larger buffer)
        self.level = logging.INFO

    def emit(self, record):
        try:
            msg = self.format(record)
            ts = datetime.utcnow().isoformat()
            entry = {
                'ts': ts,
                'level': record.levelname,
                'msg': msg
            }
            # Add to both deques
            self.deque.append(entry)
            self.history.append(entry)
        except Exception:
            self.handleError(record)

    def get_new_entries(self):
        """Returns and clears stream entries since last call (for terminal display)."""
        items = list(self.deque)
        self.deque.clear()
        return items

    def get_history(self):
        """Returns full history without clearing (for export)."""
        return list(self.history)


def setup_logging(level=logging.INFO):
    """
    Configure Python logging with DequeLogHandler for GUI display.
    Captures logs from both GUI and backend (src.processing).
    
    Args:
        level: Logging level (default: INFO)
    """
    global GLOBAL_LOG_HANDLER
    
    # Create handler if not exists
    if GLOBAL_LOG_HANDLER is None:
        GLOBAL_LOG_HANDLER = DequeLogHandler()
    
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    
    # Add console handler for development
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console.setFormatter(formatter)
    
    # Add handlers to root
    root.addHandler(console)
    root.addHandler(GLOBAL_LOG_HANDLER)
    
    # Capture backend logs from src.processing
    backend_logger = logging.getLogger('src.processing')
    backend_logger.addHandler(GLOBAL_LOG_HANDLER)
    backend_logger.setLevel(level)
    
    logging.info("Prajnya Studio logging initialized (Sprint 2 - Backend Integration)")


def get_log_handler():
    """Returns the global log handler for GUI access."""
    return GLOBAL_LOG_HANDLER
