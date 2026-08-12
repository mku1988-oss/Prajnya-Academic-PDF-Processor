#!/usr/bin/env python3
"""
Turning a model's raw text reply into structured content plus metadata.

For biology researchers: a vision model hands back one string. Depending on
the task that string is either prose to keep as-is, or JSON describing the
page (its title, authors, DOI, whether it holds a table). This module works
out which, parses accordingly, and always returns the same three-key shape so
callers never have to branch:

    {"content": ..., "metadata": {...}, "raw_output": <the original string>}

`raw_output` is always preserved. When parsing fails we would otherwise have
thrown away the only evidence of what the model actually said, and on a
pipeline this slow, re-running a page to find out is expensive.

## Never raising is the point

A parse failure returns the raw text as `content` rather than raising. That is
deliberate: a page whose JSON came back malformed still contains readable
text, and losing the whole page because one field was quoted wrongly would be
a worse outcome than passing the text through with empty metadata. The
downstream word-coverage gate (`content_drop_check`) is what catches genuine
content loss — this layer's job is to never make things worse.

Ported from V5's `universal_vlm_provider.py` (lines 1024-1126), logic
unchanged, as free functions so they can be tested without a provider.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from .schemas import TaskDefinition

#: Output types that are returned verbatim, with no JSON parsing attempted.
PLAIN_TEXT_TYPES = ("plain_text", "text", "markdown")

#: The JSON key models put the page's readable text under.
CONTENT_KEY = "natural_text"

#: Python literals a model sometimes emits instead of their JSON spellings.
PYTHON_TO_JSON_LITERALS = {"True": "true", "False": "false", "None": "null"}

#: Matches either a complete JSON string literal (including its escapes) OR a
#: bare Python literal. Alternation order matters: putting the string pattern
#: first means a literal *inside* quotes is consumed as part of the string and
#: never considered for replacement. See `_sanitize_json_output`.
_STRING_OR_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"|\bTrue\b|\bFalse\b|\bNone\b')


def post_process_output(
    raw_output: str,
    task_definition: TaskDefinition,
    task_type: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Parse a model's raw reply into `{content, metadata, raw_output}`.

    Args:
        raw_output: exactly what the model returned.
        task_definition: supplies `output_format.type`, which decides whether
            JSON parsing is attempted at all.
        task_type: task identifier, for log messages only.
        logger: optional logger.

    Returns:
        Always the three-key dict. Never raises — see the module docstring.
    """
    log = logger or logging.getLogger("OutputParser")
    output_type = task_definition.output_format.get("type", "json")

    # Plain-text tasks (figure descriptions, markdown reads) are returned as-is.
    if output_type in PLAIN_TEXT_TYPES:
        log.debug("Task '%s' expects plain text/markdown - skipping JSON parsing", task_type)
        return _plain_result(raw_output)

    cleaned_output = _sanitize_json_output(_strip_code_fence(raw_output))

    try:
        parsed = json.loads(cleaned_output)
    except json.JSONDecodeError as error:
        # An ERROR only when JSON was genuinely expected; otherwise the format
        # was ambiguous and falling back to text is the normal outcome.
        if output_type == "json":
            log.error("JSON parsing failed for JSON task '%s': %s. First 200 chars: %s",
                      task_type, error, cleaned_output[:200])
        else:
            log.debug("JSON parsing failed for task '%s' (output_type=%s): %s",
                      task_type, output_type, error)
        return _plain_result(raw_output)

    if not isinstance(parsed, dict):
        # Valid JSON, wrong shape — a bare list or string carries no metadata.
        log.warning("Parsed JSON for task '%s' is %s, not an object; treating as text",
                    task_type, type(parsed).__name__)
        return _plain_result(raw_output)

    parsed = _unwrap_nested_metadata(parsed, log)
    content = parsed.get(CONTENT_KEY, "")
    metadata = {key: item for key, item in parsed.items() if key != CONTENT_KEY}

    log.info("Successfully parsed JSON output with %d metadata fields.", len(metadata))
    return {"metadata": metadata, "content": content, "raw_output": raw_output}


# --- helpers (after the main function, per CLAUDE.md) ----------------------

def _plain_result(raw_output: str) -> Dict[str, Any]:
    """The no-metadata result shape, used for text tasks and every fallback."""
    return {"metadata": {}, "content": raw_output, "raw_output": raw_output}


def _strip_code_fence(raw_output: str) -> str:
    """
    Remove a wrapping markdown code fence.

    Models are trained to present JSON as ```json ... ```, so the fence is
    common even when the task asked for bare JSON.
    """
    cleaned = raw_output.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        return cleaned[7:-3].strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        return cleaned[3:-3].strip()
    return cleaned


def _sanitize_json_output(json_text: str) -> str:
    """
    Fix the Python-instead-of-JSON literals models routinely emit.

    `True` / `False` / `None` are Python spellings; JSON wants
    `true` / `false` / `null`. A single one of these fails the whole parse, so
    repairing them recovers otherwise-perfect output.

    **Only outside string literals** — and that qualifier is the whole point.
    V5 ran three blind `re.sub` calls across the entire document, which also
    rewrote the page's own prose: a paper reading "None of the samples showed
    activity" came out as "null of the samples showed activity". The page text
    travels in this same JSON (under `natural_text`), so that was silent
    corruption of extracted scientific content, in the fallback path meant to
    *repair* output. Word boundaries were not enough; the fix is to skip
    quoted regions entirely.
    """
    def replace(match: "re.Match") -> str:
        text = match.group(0)
        # A matched string literal is returned untouched; only a bare
        # Python literal reaches the mapping.
        return PYTHON_TO_JSON_LITERALS.get(text, text)

    return _STRING_OR_LITERAL.sub(replace, json_text)


def _unwrap_nested_metadata(parsed: Dict[str, Any], log: logging.Logger) -> Dict[str, Any]:
    """
    Flatten a `{"metadata": {...}}` wrapper, keeping the text if it sat outside.

    Strict output schemas often nest everything under `metadata`, which would
    otherwise leave the real fields one level too deep. When the page's text
    was a sibling of that wrapper rather than inside it, it is carried across
    so unwrapping never loses content.
    """
    nested = parsed.get("metadata")
    if not isinstance(nested, dict):
        return parsed

    log.debug("Unwrapping nested 'metadata' key from JSON")
    content_backup = parsed.get(CONTENT_KEY, "")
    unwrapped = dict(nested)
    if not unwrapped.get(CONTENT_KEY) and content_backup:
        unwrapped[CONTENT_KEY] = content_backup
    return unwrapped
