#!/usr/bin/env python3
"""
The transport seam — how a request actually reaches a model.

For biology researchers: everything else in this package decides *what* to
ask a model. This decides *how the question travels*. There are two very
different routes:

- **local**: an HTTP POST to llama.cpp running on this machine, speaking the
  OpenAI chat-completions dialect. This is the offline path, and the one the
  whole V6 speed argument is about.
- **gemini**: Google's hosted API, through its own SDK, with its own retry and
  rate-limit rules.

V5 fused both into one class and chose between them with an `if self.api_type
== "gemini"` branch part-way down the request builder. Separating them here
follows the pattern already proven for `PdfBackend` and `LayoutDetector`, and
buys three concrete things:

1. **The offline path never imports `google-genai`.** In V5 the module-level
   import ran on every run, offline or not.
2. Each transport's error handling is written for its own failure modes rather
   than a union of both. Rate limiting is a Gemini concern; a model not being
   loaded into VRAM is a local one.
3. A backend can be substituted in tests without a server.

## What a backend is responsible for

Exactly one thing: given a prepared request, return the model's raw text.
Prompt selection, output parsing, quality scoring, and retry policy all live
outside — a backend that started interpreting replies would put model-specific
behaviour back into the pipeline, which is the VLM-agnostic rule this whole
package exists to hold.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from ..schemas import ModelProfile, TaskDefinition

#: Backend identifiers, as reported by `select_backend_name`.
LOCAL = "openai_local"
CLOUD = "openai_cloud"
GEMINI = "gemini"


@runtime_checkable
class VlmBackend(Protocol):
    """One way of getting a page image and a prompt to a model."""

    @property
    def name(self) -> str:
        """Short identifier used in logs and config, e.g. "openai_local"."""
        ...

    def generate(
        self,
        page_image: "Image.Image",
        task_definition: TaskDefinition,
        base_text: Optional[str] = None,
    ) -> str:
        """
        Run a full-page task and return the model's raw text.

        Raises whatever transport error occurred; the retry policy lives in the
        provider, so a backend must not swallow failures.
        """
        ...

    def read_crop(
        self,
        crop_image: "Image.Image",
        prompt: str,
        max_tokens: Optional[int] = None,
        image_first: Optional[bool] = None,
    ) -> str:
        """
        Read one cropped region with an exact prompt, returning raw text.

        Used by the table-repair path. Unlike `generate` it does no prompt
        resolution — the caller supplies the exact string, chosen from the
        active model's config — which keeps this usable with any recognition
        model.
        """
        ...


def select_backend_name(model_profile: ModelProfile) -> str:
    """
    Decide which transport a profile needs, from its `model_type`.

    Ported unchanged from V5's `_detect_api_type`. Anything not explicitly
    marked as cloud defaults to the local server, which is the right default
    for an offline-first pipeline: a mistyped `model_type` fails loudly against
    localhost rather than quietly sending a paper to a third party.
    """
    if model_profile.model_type == "gemini_cloud":
        return GEMINI
    if model_profile.model_type == "cloud_api_openai":
        return CLOUD
    return LOCAL


def extract_message_content(response_json: Dict[str, Any]) -> str:
    """
    Pull the reply text out of an OpenAI-shaped response.

    Raises:
        ValueError: if the response does not have the expected shape. Both the
            local server and the cloud APIs speak this dialect, so a failure
            here means something upstream is badly wrong and should not be
            papered over with an empty string.
    """
    try:
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            f"Invalid API response format from /v1/chat/completions: {error}"
        ) from error
