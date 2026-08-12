#!/usr/bin/env python3
"""
The hosted transport: Google's Gemini API through its own SDK.

For biology researchers: this is the cloud alternative to running a model on
your own machine. It is faster per page and needs no GPU, but every page leaves
the building and costs money — so it is opt-in, selected only when a profile
declares `model_type: gemini_cloud`.

**The SDK is imported lazily**, inside the client setup rather than at module
level. V5 imported `google.genai` on every run, offline or not. Here an offline
run never touches it, so the dependency is genuinely optional.

## Rate limiting

Free and low-tier quotas are hit constantly, and the API signals that with 429
(quota) or 503 (overloaded). V5 expressed its response as four levels of nested
try/except; the policy underneath is simply a schedule of waits between
attempts:

    attempt 1  immediately
    attempt 2  immediately        (a 429 is often transient)
    attempt 3  after 60s          (assume a per-minute quota; wait it out)
    attempt 4  immediately

If the fourth attempt is still rate-limited, the per-minute theory is wrong and
the daily quota is likely gone, so we stop with a message that says so rather
than retrying for hours. Any error that is *not* a rate limit fails
immediately — retrying a malformed request just wastes quota.

Written here as that schedule, which is behaviourally identical to V5's nesting
and considerably easier to check.

Ported from V5's `_process_single_page_gemini` (lines 1333-1523) and
`_initialize_gemini_client` (1310-1331).
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any, List, Optional

from PIL import Image

from ..prompts import resolve_prompt_text
from ..schemas import ModelProfile, TaskDefinition
from .base import GEMINI

#: Seconds to wait before each attempt. See the module docstring.
RETRY_DELAYS_SECONDS: List[int] = [0, 0, 60, 0]

#: Substrings that mark a rate-limit or overload response, lower-cased.
RATE_LIMIT_MARKERS = ("429", "quota", "rate limit", "503", "overloaded")

#: Fallback per-million-token pricing, used only for the cost line in the log
#: when a profile does not declare its own. Gemini 3 Pro rates.
DEFAULT_INPUT_COST_PER_MILLION = 2.00
DEFAULT_OUTPUT_COST_PER_MILLION = 12.00

#: Generation defaults when a profile does not specify them.
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 8192


class GeminiBackend:
    """
    Talks to Gemini's `generate_content` API.

    Args:
        model_profile: the active model's profile.
        api_key: the Gemini API key. Read from the environment by the caller,
            so this class never touches `os.environ` and stays testable.
        task_type: the provider's task identifier.
        logger: optional logger.
    """

    def __init__(
        self,
        model_profile: ModelProfile,
        api_key: str,
        task_type: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.model_profile = model_profile
        self.task_type = task_type
        self.logger = logger or logging.getLogger("GeminiBackend")
        self._types: Any = None
        self.client = client if client is not None else self._create_client(api_key)

    @property
    def name(self) -> str:
        return GEMINI

    def generate(
        self,
        page_image: Image.Image,
        task_definition: TaskDefinition,
        base_text: Optional[str] = None,
    ) -> str:
        """
        Run a full-page task against Gemini and return its raw text.

        Raises:
            RuntimeError: when rate limiting persists through the whole retry
                schedule, which usually means the daily quota is exhausted.
            ValueError: on an empty response.
        """
        prompt_text = resolve_prompt_text(
            self.model_profile, task_definition, base_text=base_text,
            task_type=self.task_type, logger=self.logger,
        )
        image_bytes = _encode_png(page_image)
        started = time.time()

        response = self._request_with_retries(prompt_text, image_bytes, started)

        if not response or not getattr(response, "text", None):
            raise ValueError("Empty response from Gemini API")

        self._log_usage(response, time.time() - started)
        return response.text.strip()

    def read_crop(
        self,
        crop_image: Image.Image,
        prompt: str,
        max_tokens: Optional[int] = None,
        image_first: Optional[bool] = None,
    ) -> str:
        """
        Read one cropped region with an exact prompt.

        Note:
            `image_first` is accepted for interface compatibility and ignored:
            it exists for recognition-only models, whose output changes with
            part ordering. Gemini is instruction-following, so ordering is not
            load-bearing here.
        """
        started = time.time()
        response = self._request_with_retries(
            prompt, _encode_png(crop_image), started, max_tokens=max_tokens,
        )
        if not response or not getattr(response, "text", None):
            self.logger.error("read_crop got an empty response for prompt %r", prompt)
            return ""
        return response.text.strip()

    # --- helpers (after the main methods, per CLAUDE.md) ------------------

    def _create_client(self, api_key: str) -> Any:
        """
        Build the SDK client, importing `google.genai` only now.

        Raises:
            ImportError: if the optional SDK is not installed.
            ValueError: if no API key was supplied.
        """
        if not api_key:
            raise ValueError(
                "No Gemini API key supplied. Set GEMINI_API_KEY in the environment "
                "or your .env file."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:  # pragma: no cover - depends on optional install
            raise ImportError(
                "google-genai SDK not installed. Install with: pip install google-genai"
            ) from error

        self._types = types
        client = genai.Client(api_key=api_key)
        self.logger.info("Gemini client initialized for model: %s", self.model_profile.model_name)
        return client

    def _request_with_retries(
        self,
        prompt_text: str,
        image_bytes: bytes,
        started: float,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Run the request, following the retry schedule for rate limits only.

        See the module docstring for the schedule and why non-rate-limit errors
        are not retried.
        """
        last_error: Optional[Exception] = None

        for attempt, delay in enumerate(RETRY_DELAYS_SECONDS, start=1):
            if delay:
                self.logger.warning(
                    "Rate limited twice - assuming a per-minute quota. Waiting %ds...", delay
                )
                time.sleep(delay)

            try:
                return self._make_request(prompt_text, image_bytes, max_tokens)
            except Exception as error:
                if not is_rate_limit_error(error):
                    self.logger.error("Gemini API error after %.2fs: %s",
                                      time.time() - started, error)
                    raise
                last_error = error
                self.logger.warning("Gemini rate limit on attempt %d of %d",
                                    attempt, len(RETRY_DELAYS_SECONDS))

        self.logger.error(
            "Rate limit persists after the full retry schedule (duration: %.2fs)",
            time.time() - started,
        )
        raise RuntimeError(
            "Gemini API daily quota likely exhausted. Rate limit persisted through "
            f"{len(RETRY_DELAYS_SECONDS)} attempts including a 60s wait. "
            f"Original error: {last_error}"
        ) from last_error

    def _make_request(
        self, prompt_text: str, image_bytes: bytes, max_tokens: Optional[int] = None
    ) -> Any:
        """Issue one `generate_content` call."""
        types = self._types
        api_params = self.model_profile.get_api_params()

        config_args: dict = {
            "temperature": api_params.get("temperature", DEFAULT_TEMPERATURE),
            "max_output_tokens": max_tokens or api_params.get("max_tokens", DEFAULT_MAX_TOKENS),
        }

        # Gemini 3 reasoning control; requires google-genai 1.51.0+.
        thinking_level = api_params.get("thinking_level")
        if thinking_level:
            self.logger.info("Enabling thinking/reasoning with level: %s", thinking_level)
            config_args["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True, thinking_level=thinking_level
            )

        return self.client.models.generate_content(
            model=self.model_profile.model_name,
            contents=[types.Content(role="user", parts=[
                types.Part.from_text(text=prompt_text),
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ])],
            config=types.GenerateContentConfig(**config_args),
        )

    def _log_usage(self, response: Any, duration: float) -> None:
        """
        Log tokens and estimated cost — the running bill for a cloud run.

        Pricing comes from the profile so a rate change is a config edit.
        """
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return

        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        pricing = self.model_profile.performance.get("pricing", {})
        cost = (
            input_tokens / 1_000_000 * pricing.get("input_cost_per_million",
                                                   DEFAULT_INPUT_COST_PER_MILLION)
            + output_tokens / 1_000_000 * pricing.get("output_cost_per_million",
                                                      DEFAULT_OUTPUT_COST_PER_MILLION)
        )
        self.logger.info(
            "Gemini API call: %.2fs, Tokens: %d in + %d out, Cost: $%.6f",
            duration, input_tokens, output_tokens, cost,
        )


def is_rate_limit_error(error: Exception) -> bool:
    """
    Is this error a rate limit or an overload, rather than a real failure?

    Matched on the message text because the SDK raises a generic error type for
    both. Crude, but the alternative is retrying genuine failures (wasting
    quota) or giving up on transient ones.
    """
    message = str(error).lower()
    return any(marker in message for marker in RATE_LIMIT_MARKERS)


def _encode_png(image: Image.Image) -> bytes:
    """PNG bytes for an image — the format Gemini receives inline."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
