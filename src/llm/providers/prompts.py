#!/usr/bin/env python3
"""
Deciding what to actually send a model: which prompt, and in what order.

For biology researchers: the vision models in this pipeline fall into two
families that must be talked to differently, and getting it wrong produces
confident nonsense rather than an error.

**Instruction-following models** (OlmOCR-2, Gemini) read your request and try
to do what it says. You hand them the task's prompt.

**Recognition-only models** (GLM-OCR) do exactly one thing — transcribe what
they see — and they honour only a small set of fixed prompt strings baked in
at training time. Hand one a rich instruction like "extract the DOI and
authors as JSON" and it does not refuse. It transcribes the page and returns
that, which looks like an answer and is not. So for these models the provider
must *substitute* the model's own prompt string instead of passing ours
through, and must **refuse outright** for tasks the model cannot do.

That refusal is the important part. Silently accepting recognition output in
place of a real answer would put unflagged, wrong content into a literature
review — the one failure mode this project can least afford.

A profile opts into this with three fields (see `schemas.ModelProfile`):

    prompt_mode: fixed
    fixed_prompts:      {task identifier: the model's own prompt string}
    ineligible_tasks:   [tasks this model cannot perform at all]

Absent `prompt_mode`, behaviour is unchanged for every profile written before
this existed. This is how recognition and instruction models coexist with no
branching in the pipeline — the VLM-agnostic rule from CLAUDE.md.

Ported from V5's `universal_vlm_provider.py` (lines 927-1022). The logic is
unchanged; the methods become free functions taking the profile explicitly,
which is what lets them be tested without a provider, a server, or a GPU.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .schemas import ModelProfile, TaskDefinition

#: The placeholder a task prompt uses to request the page's digital text.
#: Anchoring a VLM with the text layer measurably improves its reading, and it
#: is free — the text is already extracted.
BASE_TEXT_PLACEHOLDER = "{base_text}"


def resolve_prompt_text(
    model_profile: ModelProfile,
    task_definition: TaskDefinition,
    base_text: Optional[str] = None,
    task_type: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Decide what prompt text this model should actually receive for this task.

    Args:
        model_profile: the active model's profile.
        task_definition: the task being run.
        base_text: the page's digital text layer, for anchoring — used only
            when the task's prompt contains `{base_text}`.
        task_type: the provider's task identifier, which wins over the task
            definition's own fields when looking up a fixed prompt.
        logger: optional logger.

    Returns:
        The prompt string to send.

    Raises:
        ValueError: for a fixed-prompt model asked to do something it has no
            prompt for, or that its profile lists as ineligible.
    """
    log = logger or logging.getLogger("Prompts")

    if model_profile.is_fixed_prompt():
        return lookup_fixed_prompt(model_profile, task_definition, task_type, logger=log)

    prompt_text = task_definition.prompt
    if base_text and BASE_TEXT_PLACEHOLDER in prompt_text:
        # `.replace`, NOT `.format`. A prompt that asks for structured output
        # shows the model an example of that structure, so it is full of
        # literal JSON braces — and `str.format` treats every one of them as a
        # field to substitute, raising KeyError on '{ "metadata": …'. The
        # title-page prompt is exactly such a prompt, which is why V5's
        # `{raw_text}` typo could never be fixed by editing the YAML alone:
        # correcting the placeholder made the call crash, so the broken name
        # stayed. `.replace` substitutes the placeholder and nothing else.
        prompt_text = prompt_text.replace(BASE_TEXT_PLACEHOLDER, base_text)
        log.info("Document anchoring enabled: %d chars base_text", len(base_text))
    return prompt_text


def build_message_content(
    prompt_text: str,
    base64_image: str,
    image_format: str = "png",
    image_first: Optional[bool] = None,
    model_profile: Optional[ModelProfile] = None,
) -> List[Dict[str, Any]]:
    """
    Order the image and the text inside the user message.

    For recognition-only models this ordering is not cosmetic: on identical
    input, image-first and text-first can return entirely different output.
    Such models appear to condition on the image, so it must arrive first, and
    that is how they were validated here. Instruction-following models keep the
    original text-first order, which is what existing profiles were tuned
    against — so the default differs by model family rather than being one
    global choice.

    Args:
        prompt_text: the text part of the message.
        base64_image: the base64-encoded page image.
        image_format: image subtype for the data URL (png, jpeg).
        image_first: force the ordering. None means derive it from the profile.
        model_profile: used only to derive the default ordering; may be omitted
            when `image_first` is given explicitly.

    Returns:
        The message `content` list, ready to post.
    """
    image_part = {
        "type": "image_url",
        "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"},
    }
    text_part = {"type": "text", "text": prompt_text}

    if image_first is None:
        image_first = model_profile is not None and model_profile.is_fixed_prompt()

    return [image_part, text_part] if image_first else [text_part, image_part]


def task_identifier(
    task_definition: TaskDefinition,
    task_type: Optional[str] = None,
) -> str:
    """
    The key a fixed prompt is looked up by.

    Task YAMLs set `task_name`, and some also set `task_type`. The provider is
    constructed with a `task_type` string matching the YAML filename, which is
    the identifier a profile's `fixed_prompts` map is written against — so that
    wins, with the task's own fields as fallbacks.
    """
    return task_type or task_definition.task_type or task_definition.task_name


def lookup_fixed_prompt(
    model_profile: ModelProfile,
    task_definition: TaskDefinition,
    task_type: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Map a task onto one of the model's fixed prompts, or refuse loudly.

    See the module docstring for why refusing matters more than it looks: a
    recognition-only model answers *every* prompt with transcription, so an
    impossible task comes back as plausible text rather than an error.

    Raises:
        ValueError: if the profile lists the task as ineligible, or has no
            fixed prompt for it. Both messages name the fix.
    """
    log = logger or logging.getLogger("Prompts")
    identifier = task_identifier(task_definition, task_type)

    if identifier in model_profile.ineligible_tasks:
        raise ValueError(
            f"Task '{identifier}' cannot run on recognition-only model "
            f"'{model_profile.model_name}'. It requires an instruction-following "
            f"model (route it to the understanding model)."
        )

    fixed_prompt = model_profile.fixed_prompts.get(identifier)
    if not fixed_prompt:
        raise ValueError(
            f"No fixed prompt defined for task '{identifier}' on model "
            f"'{model_profile.model_name}'. Add it to `fixed_prompts` in the profile, "
            f"or list it under `ineligible_tasks` if the model cannot perform it."
        )

    log.debug("Fixed-prompt model: '%s' -> %r", identifier, fixed_prompt)
    return fixed_prompt
