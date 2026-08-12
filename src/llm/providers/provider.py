#!/usr/bin/env python3
"""
The VLM provider — one page in, structured content out.

For biology researchers: this is the object the rest of the pipeline holds. It
takes a rendered page and gives back the text plus whatever structured fields
the model found, hiding which model is running and how it is reached.

It is deliberately thin, because everything it needs already lives in a module
of its own:

    schemas    what a profile and a task look like
    prompts    which prompt this model should get
    backends   how the request travels (local server or Gemini)
    output_parser  raw reply -> {content, metadata}
    quality    how plausible the reading looks

What remains here is only the sequencing, plus the retry policy — which stays
at this level on purpose. Retrying is a decision about the pipeline's tolerance
for failure, not about any one transport, and putting it in a backend would
mean writing it twice and having it drift.

Ported from V5's `UniversalVLMProvider` (`__init__`, `process_page`,
`read_crop`, `_process_single_page_with_retry`). The batch `process_document`
path is not ported: nothing called it, and V6 renders pages through
`PdfBackend` rather than the provider.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from PIL import Image

from .backends.base import CLOUD, GEMINI, LOCAL, VlmBackend, select_backend_name
from .backends.local import LocalOpenAIBackend
from .output_parser import post_process_output
from .quality import calculate_page_quality_score
from .schemas import ModelProfile, TaskDefinition, UniversalConfig

#: Retries when the config does not say otherwise.
DEFAULT_MAX_RETRIES = 3

#: Environment variable holding the Gemini key, when that backend is selected.
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


class VlmProvider:
    """
    Runs one task, on one model, over one page at a time.

    Args:
        model_profile: the active model's profile. Required — V5 could load
            this itself from a hardcoded path, but V6 passes configs in
            (see `processing/config/`), so there is one loading path.
        task_definition: the task to run.
        config: provider-wide settings; defaults are used when omitted.
        task_type: the task identifier, which must match the task YAML's
            filename because profiles key `fixed_prompts` by it.
        session: a `requests.Session`, shared so connections are reused.
        backend: inject a transport directly. Used by tests, and by any caller
            that has already built one.
        grammar_root: directory holding GBNF grammar files.
        validate_server: ping the local server's health endpoint during
            construction. Off by default — providers are built before the
            server is loaded in the lazy-launch flow, where a check here would
            fail spuriously.
        logger: optional logger.
    """

    def __init__(
        self,
        model_profile: ModelProfile,
        task_definition: TaskDefinition,
        config: Optional[UniversalConfig] = None,
        task_type: Optional[str] = None,
        session: Optional[requests.Session] = None,
        backend: Optional[VlmBackend] = None,
        grammar_root: Optional[Path] = None,
        validate_server: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_profile = model_profile
        self.task_definition = task_definition
        self.task_type = task_type or task_definition.task_type or task_definition.task_name
        self.config = config or UniversalConfig()
        self.logger = logger or logging.getLogger(f"VlmProvider.{model_profile.model_name}")

        # A profile carrying cloud_config names its own endpoint, which must
        # win over the local default or every cloud request would go to
        # localhost.
        if model_profile.cloud_config:
            self.config.api_endpoint = model_profile.cloud_config.api_endpoint
            self.logger.info("Using cloud API endpoint: %s", self.config.api_endpoint)

        self.backend = backend or self._build_backend(session, grammar_root)

        if validate_server and hasattr(self.backend, "check_availability"):
            self.backend.check_availability()

        self.logger.info("Provider ready: %s -> %s",
                         model_profile.model_name, task_definition.task_name)

    def process_page(
        self,
        page_image: Image.Image,
        base_text: Optional[str] = None,
        task_definition: Optional[TaskDefinition] = None,
    ) -> Dict[str, Any]:
        """
        Read one page and return its content, metadata, and a quality score.

        Args:
            page_image: the rendered page.
            base_text: the page's digital text layer, used for anchoring when
                the task's prompt asks for it.
            task_definition: run a different task than the provider's default.

        Returns:
            A dict with `success`, `content`, `metadata`, `quality_score`, and
            `raw_output`. On failure, `success` is False and `error` explains —
            it does not raise, because one unreadable page should not end a
            document.
        """
        effective_task = task_definition or self.task_definition

        try:
            raw_output = self._generate_with_retry(page_image, effective_task, base_text)
        except Exception as error:
            self.logger.error("Page processing failed: %s", error, exc_info=True)
            return {
                "success": False, "error": str(error), "content": "",
                "metadata": {}, "quality_score": 0.0, "raw_output": "",
            }

        parsed = post_process_output(
            raw_output, effective_task, task_type=self.task_type, logger=self.logger
        )
        quality = calculate_page_quality_score(
            parsed["content"], parsed["metadata"],
            task_type=self.task_type, logger=self.logger,
        )
        return {"success": True, "quality_score": quality, **parsed}

    def read_crop(
        self,
        crop_image: Image.Image,
        prompt: str,
        max_tokens: Optional[int] = None,
        image_first: Optional[bool] = None,
    ) -> str:
        """
        Read one cropped region with an exact prompt, returning raw text.

        This is the table-repair reader. It skips prompt resolution and JSON
        parsing entirely: the caller supplies the exact prompt (chosen from the
        active model's config, never hardcoded) and gets the model's string
        back, ready to splice into the page.

        Args:
            crop_image: the cropped region.
            prompt: the exact prompt to send.
            max_tokens: override the model's configured limit.
            image_first: force image/text ordering. None uses the model's usual
                order; a grounding read passes True.

        Returns:
            The raw output, stripped. Empty string on failure — see
            `LocalOpenAIBackend.read_crop` for why, and for the Stage 0 note.
        """
        return self.backend.read_crop(
            crop_image, prompt, max_tokens=max_tokens, image_first=image_first
        )

    # --- helpers (after the main methods, per CLAUDE.md) ------------------

    def _build_backend(
        self, session: Optional[requests.Session], grammar_root: Optional[Path]
    ) -> VlmBackend:
        """Construct the transport this profile calls for."""
        backend_name = select_backend_name(self.model_profile)

        if backend_name == GEMINI:
            from .backends.gemini import GeminiBackend  # lazy: keeps the SDK optional
            return GeminiBackend(
                self.model_profile, api_key=os.environ.get(GEMINI_API_KEY_ENV, ""),
                task_type=self.task_type, logger=self.logger,
            )

        return LocalOpenAIBackend(
            self.model_profile, self.config, task_type=self.task_type,
            session=session, cloud=(backend_name == CLOUD),
            api_key=self._cloud_api_key() if backend_name == CLOUD else None,
            grammar_root=grammar_root, logger=self.logger,
        )

    def _cloud_api_key(self) -> str:
        """
        Read the cloud key from the environment variable the profile names.

        The profile names a VARIABLE, never holds a key, so profiles stay safe
        to commit.
        """
        if not self.model_profile.cloud_config:
            raise ValueError("Cloud backend selected but the profile has no cloud_config")

        variable = self.model_profile.cloud_config.api_key_env
        api_key = os.environ.get(variable)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {variable}")
        return api_key

    def _generate_with_retry(
        self,
        page_image: Image.Image,
        task_definition: TaskDefinition,
        base_text: Optional[str],
    ) -> str:
        """
        Call the backend, retrying transient failures with exponential backoff.

        What is and is not retried is the substance here:

        - **timeouts and connection errors** — retried; the server may be
          swapping a model into VRAM, which genuinely takes time.
        - **5xx** — retried; the server's problem, possibly momentary.
        - **4xx** — NOT retried. A malformed or unauthorized request will be
          just as malformed next time, and on this pipeline each attempt costs
          real seconds.
        - **malformed responses** — retried exactly once, then raised. Usually
          a truncated reply, occasionally transient.

        Backoff doubles: 1s, 2s, 4s.
        """
        max_retries = self._max_retries()
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                return self.backend.generate(page_image, task_definition, base_text=base_text)

            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else None
                if status is not None and 400 <= status < 500:
                    self.logger.error("Client error (non-retryable): %s", error)
                    raise
                last_error = error
                self.logger.warning("Server error on attempt %d of %d. Status: %s",
                                    attempt + 1, max_retries, status or "N/A")

            except (requests.Timeout, requests.ConnectionError) as error:
                last_error = error
                self.logger.warning("Connection problem on attempt %d of %d: %s",
                                    attempt + 1, max_retries, error)

            except (ValueError, KeyError) as error:
                if attempt > 0:
                    self.logger.error("Response parsing error persists: %s", error)
                    raise
                last_error = error
                self.logger.warning("Response parsing error, will retry once: %s", error)

            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                self.logger.info("Waiting %ds before retry %d...", backoff, attempt + 2)
                time.sleep(backoff)

        self.logger.error("All %d attempts failed. Last error: %s", max_retries, last_error)
        raise last_error  # type: ignore[misc]  # unreachable while max_retries >= 1

    def _max_retries(self) -> int:
        """Retry count from the config, falling back to the default."""
        metadata = self.config.metadata
        if metadata is not None and metadata.global_settings is not None:
            return getattr(metadata.global_settings, "max_retries", DEFAULT_MAX_RETRIES)
        return DEFAULT_MAX_RETRIES
