#!/usr/bin/env python3
"""
The data contracts for VLM profiles, tasks, and provider settings.

For biology researchers: these are the shapes that YAML config files must
fit. Nothing here does any work — they are validators. Give one a dictionary
loaded from a YAML file and it either returns a checked, typed object or tells
you exactly which field is wrong, which is far better than discovering a typo
three minutes into a GPU run.

They pair with `src/processing/config/`: that module reads the file and
expands `${ENV}` placeholders; these models then say whether the result is
usable.

## Why every model allows extra fields

Almost all of these set `extra = "allow"`. That is deliberate, not laziness.
Profiles carry model-specific keys that the pipeline never reads but a human
maintainer wants recorded (benchmark notes, tuning history, biology-specific
options). Rejecting unknown keys would mean editing Python every time someone
annotates a YAML file — precisely the coupling the VLM-agnostic rule exists to
prevent. The fields spelled out below are the ones the code actually depends
on; everything else rides along untouched.

Ported from V5's `universal_vlm_provider.py` (lines 49-237) with field
definitions unchanged. Only the Pydantic v1 `.dict()` call in `get_api_params`
is modernized to v2's `.model_dump()`, and `class Config` becomes
`model_config` — V5 pinned Pydantic v2 but still used the v1 spellings, which
emit deprecation warnings on every call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# The three-tier parameter architecture
#
# A model's settings fall into three groups that go to three different places.
# Keeping them apart is what stops someone from putting `n_gpu_layers` in an
# API request (where it is silently ignored) or `temperature` in the server
# launch config (where it is overridden per request).
# ---------------------------------------------------------------------------

class APIParameters(BaseModel):
    """
    Tier 1: parameters sent with each API request.

    These override server defaults per request. Only include parameters the
    endpoint actually accepts.
    """

    model_config = ConfigDict(extra="allow")  # e.g. thinking_level, custom knobs

    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0,
                                               description="Penalty based on token frequency")


class ServerConfig(BaseModel):
    """
    Tier 2: how the model server was launched — documentation, not requests.

    These are set in the llama-swap config when starting llama-server. They are
    recorded here so a profile is a complete description of a model's setup,
    but nothing in this package sends them anywhere.
    """

    model_config = ConfigDict(extra="allow")

    threads: Optional[int] = Field(None, description="Number of CPU threads")
    ctx_size: Optional[int] = Field(None, description="Context window size")
    flash_attn: Optional[bool] = Field(None, description="Flash attention enabled")
    n_gpu_layers: Optional[int] = Field(None, description="GPU layers to offload")
    cache_type_k: Optional[str] = Field(None, description="KV cache type for keys")
    cache_type_v: Optional[str] = Field(None, description="KV cache type for values")
    default_temp: Optional[float] = Field(None, description="Server default temperature")
    default_top_p: Optional[float] = Field(None, description="Server default top_p")
    default_repeat_penalty: Optional[float] = Field(None, description="Server default repeat_penalty")
    default_min_p: Optional[float] = Field(None, description="Server default min_p")


class AppConfig(BaseModel):
    """
    Tier 3: how *our* pipeline treats this model — image sizing, batching.

    These control Shastrasaara's behaviour, not the model server's.
    """

    model_config = ConfigDict(extra="allow")

    target_dpi: Optional[int] = Field(None, description="Target DPI for image rendering")
    preferred_image_format: Optional[str] = Field(None, description="Image format (png, jpeg)")
    target_anchor_text_len: Optional[int] = Field(None, description="OCR anchor text length")
    batch_size: Optional[int] = Field(None, description="Processing batch size")
    target_image_size: Optional[int] = Field(
        None, description="Target longest edge for the vision model, in pixels (e.g. 1288)"
    )


class ModelParameters(BaseModel):
    """
    DEPRECATED single-tier parameters, kept so older profiles still load.

    `ModelProfile.get_api_params` falls back to this when `api_parameters` is
    absent. New profiles should use the three-tier fields above.
    """

    model_config = ConfigDict(extra="allow")

    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    repeat_penalty: Optional[float] = Field(None, ge=0.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0)


class CloudConfig(BaseModel):
    """
    Settings for a hosted API (OpenRouter, OpenAI, Gemini).

    `api_key_env` names an environment VARIABLE rather than holding a key, so
    a profile can be committed to the repository safely.
    """

    api_endpoint: str = Field(..., description="Full API endpoint URL")
    api_key_env: str = Field(..., description="Environment variable name holding the API key")
    timeout: int = Field(120, description="API timeout in seconds")


# ---------------------------------------------------------------------------
# Profiles and tasks
# ---------------------------------------------------------------------------

class ModelProfile(BaseModel):
    """
    Everything specific to one model, in one place.

    This is the heart of the VLM-agnostic rule: no model name, prompt, or
    vendor quirk belongs in `src/` — it belongs in a profile YAML that becomes
    one of these. Swapping models is then a config edit.

    The `prompt_mode` / `fixed_prompts` / `ineligible_tasks` trio is how
    recognition-only models coexist with instruction-following ones. See
    `prompts.py` for what they mean in practice.
    """

    model_config = ConfigDict(extra="allow")  # profiles carry human-facing notes

    model_name: str = Field(..., description="Model identifier (must match the llama-swap config)")
    model_type: str = Field(..., description="Model category, e.g. 'official_specialized_ocr'")
    benchmark_score: Optional[float] = Field(None, description="Model performance score")
    status: Optional[str] = Field(None, description="e.g. 'production_ready', 'working_baseline'")

    # Three-tier architecture (preferred)
    api_parameters: Optional[APIParameters] = Field(None, description="Tier 1: request parameters")
    server_config: Optional[ServerConfig] = Field(None, description="Tier 2: server setup (docs only)")
    app_config: Optional[AppConfig] = Field(None, description="Tier 3: pipeline settings")
    cloud_config: Optional[CloudConfig] = Field(None, description="Hosted-API settings, if any")

    # Legacy single-tier parameters (deprecated; see get_api_params)
    parameters: Optional[ModelParameters] = Field(None, description="DEPRECATED: use api_parameters")

    # Recognition-only model support. Absent `prompt_mode`, behaviour is
    # unchanged for every profile written before this existed.
    prompt_mode: Optional[str] = Field(None, description="'fixed' for recognition-only models")
    fixed_prompts: Dict[str, str] = Field(default_factory=dict,
                                          description="task identifier -> fixed prompt string")
    ineligible_tasks: List[str] = Field(default_factory=list,
                                        description="Tasks this model cannot perform")

    performance: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")

    def get_api_params(self) -> Dict[str, Any]:
        """
        The parameters to send with a request, newest schema winning.

        Prefers `api_parameters`; falls back to the deprecated `parameters`.
        Drops None values so an unset field means "let the server decide"
        rather than "send null".
        """
        if self.api_parameters:
            return self.api_parameters.model_dump(exclude_none=True)
        if self.parameters:
            return self.parameters.model_dump(exclude_none=True)
        return {}

    def is_fixed_prompt(self) -> bool:
        """
        True when this model ignores instructions and honours only fixed prompts.

        Lives on the profile rather than in the provider because it is a fact
        about the model, and both the prompt builder and the message-ordering
        logic need to ask it.
        """
        return self.prompt_mode == "fixed"


class TaskDefinition(BaseModel):
    """
    One job a model can be asked to do — the prompt and the expected output.

    Tasks are separate from profiles so the same task ("read this page") can
    run on any model, and the same model can perform many tasks.
    """

    model_config = ConfigDict(extra="allow")  # task_metadata and friends

    task_name: str
    task_type: Optional[str] = None
    description: Optional[str] = None
    prompt: str = "Extract text from this document page accurately."
    output_format: Dict[str, Any] = Field(default_factory=lambda: {"type": "plain_text"})
    quality_targets: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider-wide settings (universal_config.yaml)
# ---------------------------------------------------------------------------

class GlobalSettings(BaseModel):
    """Global processing settings."""

    model_config = ConfigDict(extra="allow")

    max_retries: int = 3
    enable_logging: bool = True
    log_level: str = "INFO"


class PerformanceSettings(BaseModel):
    """
    Performance switches.

    `parallel_pages` stays False: concurrency was measured in four
    configurations on this hardware and every one netted zero or worse,
    because wall time tracks total GPU compute on a single card. See CLAUDE.md.
    """

    model_config = ConfigDict(extra="allow")

    enable_caching: bool = False
    batch_processing: bool = False
    parallel_pages: bool = False


class QualitySettings(BaseModel):
    """Quality control settings."""

    model_config = ConfigDict(extra="allow")

    enable_validation: bool = True
    minimum_confidence: float = 0.5
    enable_fallback: bool = True


class DevelopmentSettings(BaseModel):
    """Development and debugging settings."""

    model_config = ConfigDict(extra="allow")

    debug_mode: bool = False
    save_intermediate: bool = False
    timing_enabled: bool = True


class Metadata(BaseModel):
    """The metadata block of the universal config."""

    provider_type: str
    version: str
    created: Optional[str] = None
    updated: Optional[str] = None
    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    development: DevelopmentSettings = Field(default_factory=DevelopmentSettings)


class UniversalConfig(BaseModel):
    """
    Provider-wide defaults — endpoint, timeouts, image handling.

    `api_endpoint` is a plain `str`, where V5 declared it `HttpUrl`. Measured
    on pydantic 2.11, that declaration gives the field **two different types**
    depending on where its value came from:

        from the default   -> `str`     (defaults are not validated)
        from a config file -> `HttpUrl` (and `isinstance(value, str)` is False)

    The URL text is identical either way, so nothing about the endpoint
    changes — but the inconsistent type is why V5 wraps this field in `str(...)`
    at four separate call sites, and why assigning an override turns it back
    into a plain string mid-run. Declaring it `str` makes the type stable and
    those coercions unnecessary.
    """

    api_endpoint: str = "http://localhost:8080/v1/chat/completions"
    timeout_seconds: int = 900  # generous: complex two-shot tasks are slow, not hung
    default_model: str = "olmocr-7b"
    fallback_model: str = "numarkdown-8b"
    image_padding: int = 200
    conversion_dpi: int = 300
    image_format: str = "png"
    metadata: Optional[Metadata] = None
