#!/usr/bin/env python3
"""
Transports — how a request reaches a model.

`GeminiBackend` is deliberately NOT re-exported here. Importing it pulls in the
optional `google-genai` SDK, and an offline run must never need it; the
provider imports it lazily, only when a profile actually selects it.
"""

from .base import CLOUD, GEMINI, LOCAL, VlmBackend, select_backend_name
from .local import LocalOpenAIBackend

__all__ = [
    "CLOUD", "GEMINI", "LOCAL",
    "LocalOpenAIBackend", "VlmBackend", "select_backend_name",
]
