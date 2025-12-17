"""
VRAM Calculator Utility
=======================

This module calculates the maximum safe number of concurrent VLM instances
that can be run based on the user's system VRAM capacity and the model's
per-instance VRAM requirement.

Design Logic
------------
- No guessing or safety margins: Trust the user's declared VRAM capacity.
- Simple division: max_concurrent = floor(user_vram / model_vram)
- Zero concurrency handling: Returns 0 if insufficient VRAM.
- Warnings: Provides clear warning messages for insufficient resources.

Usage
-----
>>> from utils.vram_calculator import calculate_max_concurrent
>>> max_instances = calculate_max_concurrent(user_vram_gb=16, model_vram_gb=7.6)
>>> print(max_instances)
2
"""

import math
from typing import Tuple, Optional

def calculate_max_concurrent(user_vram_gb: float, model_vram_gb: float) -> int:
    """
    Calculate maximum concurrent VLM instances.

    Args:
        user_vram_gb: User-declared system VRAM capacity in GB.
        model_vram_gb: VLM profile's declared VRAM requirement in GB.

    Returns:
        Maximum concurrent instances (0 if insufficient VRAM).

    Raises:
        ValueError: If model_vram_gb is <= 0.
    """
    if model_vram_gb <= 0:
        raise ValueError("Model VRAM requirement must be positive")
    
    if user_vram_gb <= 0:
        return 0 # Invalid user VRAM means no concurrency possible

    if user_vram_gb < model_vram_gb:
        return 0  # Insufficient VRAM for even one instance

    return int(user_vram_gb // model_vram_gb)

def get_concurrency_recommendation(max_concurrent: int) -> str:
    """
    Get a human-readable recommendation based on calculated concurrency.

    Args:
        max_concurrent: Calculated maximum concurrent instances.

    Returns:
        Recommendation string.
    """
    if max_concurrent == 0:
        return "Insufficient VRAM for this model. Try a smaller model or Cloud API."
    elif max_concurrent == 1:
        return "Sequential processing only (1 instance)."
    else:
        return f"Safe for {max_concurrent} concurrent instances."

def format_vram_warning(user_vram_gb: float, model_vram_gb: float) -> str:
    """
    Format a warning message for insufficient VRAM.

    Args:
        user_vram_gb: User-declared system VRAM capacity.
        model_vram_gb: VLM profile's declared VRAM requirement.

    Returns:
        Formatted warning message.
    """
    return (
        f"INSUFFICIENT VRAM: System has {user_vram_gb:.1f} GB, "
        f"but model requires {model_vram_gb:.1f} GB per instance."
    )
