#!/usr/bin/env python3
"""
Configuration loading for V6.

The public surface is small on purpose — everything model-specific belongs in
a YAML profile, and this package is the only thing that reads them.
"""

from .loader import (
    ENV_PATTERN,
    ConfigLoader,
    interpolate_env,
    load_config,
    resolve_reference,
    resolve_references,
)

__all__ = [
    "ENV_PATTERN",
    "ConfigLoader",
    "interpolate_env",
    "load_config",
    "resolve_reference",
    "resolve_references",
]
