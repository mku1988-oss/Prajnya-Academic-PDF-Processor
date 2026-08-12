#!/usr/bin/env python3
"""
The VLM provider — split by responsibility rather than kept as one class.

V5 had this as a single 1654-line `UniversalVLMProvider`, plus a 358-line
async subclass. The port drops both the async subclass (pure concurrency
machinery, whose every caller is on plan.md's delete list, for a technique
measured at zero benefit on this hardware) and the unreachable batch path,
then separates what remained:

    schemas.py        the Pydantic data contracts for profiles and tasks
    prompts.py        which prompt to send, and in what order
    output_parser.py  raw model text -> {content, metadata, raw_output}
    quality.py        how plausible a page reading looks
    backends/         how the request travels (local server, or Gemini)
    provider.py       the thin object that sequences all of the above

Everything above `backends/` is PURE — no network, no server, no GPU — which is
why it is testable on its own.

`GeminiBackend` is not exported from here: importing it pulls in the optional
`google-genai` SDK, and an offline run must never need it.
"""

from .backends import LocalOpenAIBackend, VlmBackend, select_backend_name
from .output_parser import post_process_output
from .prompts import (
    build_message_content,
    lookup_fixed_prompt,
    resolve_prompt_text,
    task_identifier,
)
from .provider import VlmProvider
from .quality import calculate_page_quality_score, title_page_metadata_is_complete
from .schemas import (
    APIParameters,
    AppConfig,
    CloudConfig,
    Metadata,
    ModelParameters,
    ModelProfile,
    ServerConfig,
    TaskDefinition,
    UniversalConfig,
)

__all__ = [
    # schemas
    "APIParameters", "AppConfig", "CloudConfig", "Metadata", "ModelParameters",
    "ModelProfile", "ServerConfig", "TaskDefinition", "UniversalConfig",
    # behaviour
    "build_message_content", "calculate_page_quality_score", "lookup_fixed_prompt",
    "post_process_output", "resolve_prompt_text", "task_identifier",
    "title_page_metadata_is_complete",
    # transports and the provider
    "LocalOpenAIBackend", "VlmBackend", "VlmProvider", "select_backend_name",
]
