#!/usr/bin/env python3
"""
Provider construction — from config names to ready `VlmProvider` objects.

For biology researchers: the strategy never says "use GLM" in code. It says
"use whatever the config's `extraction` role points at", and this module turns
that role name into a live provider. Swapping models is then a one-line config
edit, never a code change (the VLM-agnostic rule).

Ported from V5's `_initialize_provider_from_config` (L2842) and the cached
`_create_two_shot_provider` (L2193). `_load_vlm_profile` / `_load_task_definition`
(L2881/L2917) are not re-ported: their job — YAML from a directory, UTF-8,
helpful error on a missing file — is exactly what `ConfigLoader` already does,
and V5's `Path(__file__)`-relative lookups were the Stage 4 decomposition risk
plan.md warns about. The factory takes a `ConfigLoader` with an explicit root
instead.

Differences from V5, all deliberate:
- V5 built providers with `skip_validation=True` because the local server is
  not running during strategy init (lazy launch). V6's `VlmProvider` has the
  same behaviour as its default (`validate_server=False`), so nothing is
  passed.
- Registry references (`@profile_registry.default`) are resolved here, at the
  factory boundary, so callers can hand over either a reference or a literal
  profile name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ....llm.providers.provider import VlmProvider
from ....llm.providers.schemas import ModelProfile, TaskDefinition, UniversalConfig
from ...config.loader import ConfigLoader, resolve_reference

#: Fallback body-page resolution when neither the strategy nor the profile
#: names one (V5 L2861).
DEFAULT_RESOLUTION = 1024


@dataclass
class BuiltProvider:
    """A provider plus the two facts every caller of V5's helper also needed."""
    provider: VlmProvider
    profile_data: Dict[str, Any]  # the raw profile dict (for two_shot_parameters etc.)
    resolution: int               # final render resolution for this provider's pages


class ProviderFactory:
    """
    Builds providers from the strategy config's registries.

    Args:
        loader: the config loader rooted at the V6 `config/` tree.
        strategy_config: the loaded strategy YAML (registries live in it).
        logger: optional logger.
    """

    def __init__(self, loader: ConfigLoader, strategy_config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None) -> None:
        self.loader = loader
        self.config = strategy_config
        self.logger = logger or logging.getLogger("ProviderFactory")
        self.api_endpoint: Optional[str] = (
            strategy_config.get("vlm_provider", {}).get("api_endpoint")
        )

    def build(self, profile: str, task: str,
              resolution_override: Optional[int] = None) -> BuiltProvider:
        """
        Build one provider from a profile and task, resolving registry refs.

        Args:
            profile: a profile name ("glm-ocr") or reference
                ("@profile_registry.extraction").
            task: a task name ("raw_ocr_extraction") or reference
                ("@task_registry.raw_ocr").
            resolution_override: the strategy's resolution for this provider's
                pages; falls back to the profile's `default_resolution`, then
                to `DEFAULT_RESOLUTION`.

        Returns:
            The provider, the raw profile dict, and the final resolution.
        """
        profile_name = resolve_reference(profile, self.config, logger=self.logger)
        task_name = resolve_reference(task, self.config, logger=self.logger)

        profile_data = self.loader.vlm_profile(profile_name)
        task_data = self.loader.task_definition(task_name)

        resolution = (resolution_override
                      or profile_data.get("default_resolution")
                      or DEFAULT_RESOLUTION)

        self.logger.info("Initializing provider: %s -> %s (resolution: %dpx)",
                         profile_data.get("model_name", profile_name),
                         task_name, resolution)

        provider = self._make_provider(profile_data, task_data, task_name)
        return BuiltProvider(provider=provider, profile_data=profile_data,
                             resolution=resolution)

    def build_shot(self, base_profile_data: Dict[str, Any], task: str,
                   temperature: float, max_tokens: int,
                   repeat_penalty: float, frequency_penalty: float) -> VlmProvider:
        """
        Build a provider for one shot of two-shot processing.

        Two-shot runs the same model twice with different sampling parameters
        (shot 1: verbatim OCR; shot 2: structured analysis). This copies the
        base profile, replaces only `api_parameters`, and leaves the base
        profile untouched — exactly V5's `_create_two_shot_provider`.
        """
        task_name = resolve_reference(task, self.config, logger=self.logger)
        task_data = self.loader.task_definition(task_name)

        modified = dict(base_profile_data)
        modified["api_parameters"] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repeat_penalty": repeat_penalty,
            "frequency_penalty": frequency_penalty,
        }
        return self._make_provider(modified, task_data, task_name)

    # --- helpers (after the main methods, per CLAUDE.md) ------------------

    def _make_provider(self, profile_data: Dict[str, Any],
                       task_data: Dict[str, Any], task_name: str) -> VlmProvider:
        """Validate the dicts into pydantic models and construct the provider."""
        model_profile = ModelProfile(**profile_data)
        task_definition = TaskDefinition(**task_data)
        config = UniversalConfig(api_endpoint=self.api_endpoint) \
            if self.api_endpoint else None
        return VlmProvider(
            model_profile=model_profile,
            task_definition=task_definition,
            config=config,
            task_type=task_name,
            logger=self.logger,
        )
