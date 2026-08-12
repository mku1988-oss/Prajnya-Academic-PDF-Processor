#!/usr/bin/env python3
"""
The offline transport: HTTP to llama.cpp's OpenAI-compatible endpoint.

For biology researchers: this is how a page image reaches the vision model
running on this machine. The image becomes base64 text, it is wrapped in a
JSON message alongside the prompt, and posted to `localhost:8080`. The whole
V6 performance argument is about making *fewer* of these calls, so this is the
expensive operation the rest of the pipeline works to avoid.

The same code serves hosted OpenAI-compatible APIs (OpenRouter), which differ
only by needing an `Authorization` header — hence one class with a `cloud`
flag rather than two near-identical ones.

Ported from V5's `_process_single_page` (lines 802-925), `read_crop`
(418-473), `_encode_image_to_base64` (1128-1135), `_load_grammar_content`
(785-800) and `_check_model_availability` (1264-1282). Request construction is
unchanged. The verbose per-request response dump V5 logged at INFO (11 lines
per page, including full response headers) drops to DEBUG.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import requests
from PIL import Image

from ..prompts import build_message_content, resolve_prompt_text
from ..schemas import ModelProfile, TaskDefinition, UniversalConfig
from .base import LOCAL, extract_message_content

#: Headers OpenRouter asks integrators to send for attribution.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/mku1988-oss/Prajnya-Academic-PDF-Processor",
    "X-Title": "Shastrasaara V6",
}

#: How long to wait on the server health check, in seconds.
HEALTH_CHECK_TIMEOUT = 5


class LocalOpenAIBackend:
    """
    Talks to any server speaking the OpenAI chat-completions dialect.

    Args:
        model_profile: the active model's profile (supplies the model name,
            API parameters, and prompt-mode behaviour).
        config: provider-wide settings — endpoint, timeout, image format.
        task_type: the provider's task identifier, used for fixed-prompt lookup.
        session: a `requests.Session`, so connections are reused across pages.
        cloud: True for a hosted OpenAI-compatible API, which adds an
            `Authorization` header and skips schema-constrained output.
        api_key: the bearer token, required when `cloud` is True.
        grammar_root: directory holding GBNF grammar files, if any task uses one.
        logger: optional logger.
    """

    def __init__(
        self,
        model_profile: ModelProfile,
        config: UniversalConfig,
        task_type: Optional[str] = None,
        session: Optional[requests.Session] = None,
        cloud: bool = False,
        api_key: Optional[str] = None,
        grammar_root: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_profile = model_profile
        self.config = config
        self.task_type = task_type
        self.session = session or requests.Session()
        self.cloud = cloud
        self.api_key = api_key
        self.grammar_root = grammar_root
        self.logger = logger or logging.getLogger("LocalBackend")

    @property
    def name(self) -> str:
        return "openai_cloud" if self.cloud else LOCAL

    def generate(
        self,
        page_image: Image.Image,
        task_definition: TaskDefinition,
        base_text: Optional[str] = None,
    ) -> str:
        """
        Run a full-page task and return the model's raw text.

        Raises:
            requests.RequestException: on any transport failure. Retry policy
                belongs to the provider, so failures propagate unchanged.
            ValueError: if the response is not shaped as expected.
        """
        prompt_text = resolve_prompt_text(
            self.model_profile, task_definition, base_text=base_text,
            task_type=self.task_type, logger=self.logger,
        )
        payload = self._build_payload(prompt_text, page_image, task_definition)
        self._log_payload(payload, base_text)

        response = self._post(payload)
        return extract_message_content(response).strip()

    def read_crop(
        self,
        crop_image: Image.Image,
        prompt: str,
        max_tokens: Optional[int] = None,
        image_first: Optional[bool] = None,
    ) -> str:
        """
        Read one cropped region with an exact prompt.

        The caller supplies the prompt already resolved from the model's config
        (never hardcoded here), which is what keeps this model-agnostic. The
        model's own `api_parameters` — temperature, seed, sampler settings —
        are used unchanged, so per-model quirks stay in config.

        Returns:
            The model's raw output, stripped. **Empty string on failure.**

        Note:
            Swallowing the error is V5's behaviour, ported deliberately
            unchanged for parity. It is a known weakness: a dropped table read
            returns "" and looks like an empty table rather than a failure.
            `plan.md` Stage 0 replaces this with a logged retry, gated on a
            unit test that injects a 500 here. Do not "fix" it ahead of that
            stage without moving the parity baseline with it.
        """
        try:
            api_params = self.model_profile.get_api_params()
            if max_tokens is not None:
                api_params = {**api_params, "max_tokens": max_tokens}

            payload = {
                "model": self.model_profile.model_name,
                "messages": [{
                    "role": "user",
                    "content": build_message_content(
                        prompt, self._encode_image(crop_image),
                        image_format=self.config.image_format,
                        image_first=image_first,
                        model_profile=self.model_profile,
                    ),
                }],
                **api_params,
            }

            body = self._post(payload)
            if body["choices"][0].get("finish_reason") == "length":
                self.logger.warning(
                    "read_crop hit max_tokens (%r) - output may be truncated", prompt
                )
            return extract_message_content(body).strip()

        except (requests.RequestException, ValueError, KeyError, IndexError) as error:
            self.logger.error("read_crop failed for prompt %r: %s", prompt, error, exc_info=True)
            return ""

    def check_availability(self) -> None:
        """
        Ping the server's `/health` endpoint before relying on it.

        Fails fast with a clear message rather than letting the first real page
        request time out after the full (deliberately long) timeout.

        Raises:
            ConnectionError: if the server cannot be reached.
        """
        parsed = urlparse(self.config.api_endpoint)
        health_url = urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))

        self.logger.info("Pinging model health endpoint: %s", health_url)
        try:
            response = self.session.get(health_url, timeout=HEALTH_CHECK_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as error:
            self.logger.warning("Model health check failed: %s", error)
            raise ConnectionError(f"Could not connect to VLM server at {health_url}") from error

        self.logger.info("Model server is available at %s.", parsed.netloc)

    # --- helpers (after the main methods, per CLAUDE.md) ------------------

    def _build_payload(
        self,
        prompt_text: str,
        page_image: Image.Image,
        task_definition: TaskDefinition,
    ) -> Dict[str, Any]:
        """Assemble the chat-completions request body."""
        payload: Dict[str, Any] = {
            "model": self.model_profile.model_name,
            "messages": [{
                "role": "user",
                "content": build_message_content(
                    prompt_text, self._encode_image(page_image),
                    image_format=self.config.image_format,
                    model_profile=self.model_profile,
                ),
            }],
            **self.model_profile.get_api_params(),
        }
        self._apply_output_constraints(payload, task_definition)
        return payload

    def _apply_output_constraints(
        self, payload: Dict[str, Any], task_definition: TaskDefinition
    ) -> None:
        """
        Add schema-constrained JSON and/or a GBNF grammar, where applicable.

        Two constraints, and they are **mutually exclusive by necessity**:
        llama.cpp build 6660+ returns 400 Bad Request when a request carries
        BOTH `response_format` and `grammar`. V5 carried a comment saying
        exactly that, but set the two independently — so a task declaring both
        would have produced a guaranteed 400. Nothing triggered it because no
        task definition in V5 declares `grammar_file` at all, leaving the
        collision latent. The guard below makes the code do what that comment
        already said, and changes no current behaviour.

        `response_format` wins when both are possible, for the reason V5 gave:
        the JSON schema is sufficient on its own, and `output_parser` repairs
        anything that slips through.

        Cloud APIs skip `response_format` entirely — measured as slower and
        less reliable there than prompt-based JSON plus post-processing.
        """
        output_format = task_definition.output_format
        wants_json = output_format.get("type") == "json"
        schema_constrained = wants_json and "schema" in output_format and not self.cloud

        if schema_constrained:
            payload["response_format"] = {
                "type": "json_object",
                "schema": output_format["schema"],
            }
            self.logger.info("Requesting schema-constrained JSON via response_format")
        elif wants_json and self.cloud:
            self.logger.info(
                "Skipping response_format for cloud API - using prompt-based JSON extraction"
            )

        grammar_filename = getattr(task_definition, "grammar_file", None)
        if not grammar_filename:
            return

        if schema_constrained:
            self.logger.warning(
                "Task %r declares both a JSON schema and grammar %r; sending both is a "
                "guaranteed 400 from llama.cpp 6660+. Using the schema and skipping the grammar.",
                task_definition.task_name, grammar_filename,
            )
            return

        grammar_content = self._load_grammar(grammar_filename)
        if grammar_content:
            self.logger.info("Applying per-request GBNF grammar: %s", grammar_filename)
            payload["grammar"] = grammar_content

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send the request and return the decoded JSON body."""
        response = self.session.post(
            self.config.api_endpoint,
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _headers(self) -> Dict[str, str]:
        """Request headers, with cloud authentication when needed."""
        headers = {"Content-Type": "application/json"}
        if not self.cloud:
            return headers

        if not self.api_key:
            raise ValueError(f"No API key supplied for cloud backend {self.name}")
        headers["Authorization"] = f"Bearer {self.api_key}"
        if "openrouter" in self.config.api_endpoint:
            headers.update(OPENROUTER_HEADERS)
        return headers

    def _encode_image(self, image: Image.Image) -> str:
        """Encode a PIL image as base64 in the configured format."""
        buffer = io.BytesIO()
        image_format = self.config.image_format.upper()
        if image_format == "JPG":
            image_format = "JPEG"  # PIL knows this format only by its full name
        image.save(buffer, format=image_format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _load_grammar(self, grammar_filename: str) -> Optional[str]:
        """
        Read a GBNF grammar file, or return None if it cannot be found.

        A missing grammar degrades output quality but does not stop the run —
        the JSON repair in `output_parser` is the safety net — so this warns
        rather than raising.
        """
        if self.grammar_root is None:
            self.logger.warning(
                "Task requests grammar %r but no grammar_root is configured", grammar_filename
            )
            return None

        grammar_path = self.grammar_root / grammar_filename
        try:
            return grammar_path.read_text(encoding="utf-8")
        except OSError as error:
            self.logger.warning("Could not read grammar %s: %s", grammar_path, error)
            return None

    def _log_payload(self, payload: Dict[str, Any], base_text: Optional[str]) -> None:
        """
        Log the request shape without its bulk.

        Messages and grammar are elided — a base64 page image is megabytes and
        would make the log unreadable and enormous.
        """
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        summary = {key: value for key, value in payload.items()
                   if key not in ("messages", "grammar")}
        if "grammar" in payload:
            summary["grammar"] = f"<{len(payload['grammar'])} chars>"
        summary["messages"] = "..."
        self.logger.debug(
            "POST %s payload=%s base_text=%d chars",
            self.config.api_endpoint, json.dumps(summary), len(base_text or ""),
        )
