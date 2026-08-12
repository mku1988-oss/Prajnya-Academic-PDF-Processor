#!/usr/bin/env python3
"""
Scoring how well a page was read, from the output alone.

For biology researchers: after a model reads a page we need a number saying
"does this look like a real reading of a real page?" — without a human
checking. The score blends five cheap signals:

    content_length      0.30   did we get a page's worth of text?
    structural_markers  0.10   headings, lists — signs of structure, not soup
    text_quality        0.20   sane spacing, punctuation, no run-on garbage
    metadata_presence   0.20   are the expected fields there at all?
    metadata_validity   0.20   are they the right TYPE and shape?

The last two are task-aware: a title page is expected to yield a title,
authors, and a DOI, while a body page is expected to yield language and
layout flags. Scoring a body page against title-page fields would mark every
good read as poor, which is what "task-aware" is guarding against.

This is a *plausibility* score, not an accuracy one. It cannot tell whether
the transcription is correct — only whether it has the shape of a real
reading. Actual content loss is caught separately and far more reliably by the
word-coverage gate in `processing/utils/content_drop_check.py`, which compares
against the PDF's own text layer.

Ported from V5's `universal_vlm_provider.py` (lines 476-546). Weights,
thresholds, and field lists are unchanged. Two changes: it is a free function
taking `task_type` explicitly, and V5's two unreachable lines after the
`return` (548-549, a duplicated log-and-return) are dropped.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

#: Task identifiers that describe a title page rather than a body page.
TITLE_PAGE_TASKS = ("title_page_metadata_extraction", "structured_analysis_title_page")

#: Fields a title-page read should produce.
TITLE_PAGE_FIELDS = ("title", "authors", "doi")

#: Fields a body-page read should produce.
BODY_PAGE_FIELDS = ("primary_language", "is_rotation_valid", "rotation_correction",
                    "is_table", "is_diagram")

#: How the five signals combine. They sum to 1.0.
SCORE_WEIGHTS = {
    "content_length": 0.3,
    "structural_markers": 0.1,
    "text_quality": 0.2,
    "metadata_presence": 0.2,
    "metadata_validity": 0.2,
}

#: Characters of text that count as a full page for scoring purposes.
FULL_PAGE_CHARS = 2500.0

#: Markdown markers looked for, and how many earn a full structural score.
STRUCTURAL_MARKERS = ("#", "##", "###", "* ", "- ", "**")
MARKERS_FOR_FULL_SCORE = 4.0


def calculate_page_quality_score(
    content: str,
    metadata: Dict[str, Any],
    task_type: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Score a processed page from 0.0 (implausible) to 1.0 (looks like a good read).

    Args:
        content: the text the model produced for the page.
        metadata: the structured fields parsed alongside it.
        task_type: the task identifier, which selects the title-page or
            body-page field expectations.
        logger: optional logger.

    Returns:
        The weighted score.
    """
    log = logger or logging.getLogger("Quality")

    scores = {
        "content_length": min(1.0, len(content) / FULL_PAGE_CHARS),
        "structural_markers": _structural_marker_score(content),
        "text_quality": _text_quality_score(content),
    }

    is_title_page = task_type in TITLE_PAGE_TASKS
    if is_title_page:
        expected = TITLE_PAGE_FIELDS
        validity = _title_page_validity_checks(metadata)
        label = "Title"
    else:
        expected = BODY_PAGE_FIELDS
        validity = _body_page_validity_checks(metadata)
        label = "Body"

    # Presence differs subtly between the two: a body page's flags are
    # meaningfully False, so presence tests `is not None`, whereas a title
    # page's fields are strings and lists where empty means missing.
    if is_title_page:
        present = sum(1 for field in expected if metadata.get(field))
    else:
        present = sum(1 for field in expected if metadata.get(field) is not None)

    scores["metadata_presence"] = present / len(expected)
    scores["metadata_validity"] = sum(validity) / len(expected)

    log.info("  Metadata Fields (%s): %d/%d present", label, present, len(expected))
    log.info("  Metadata Validity (%s): %d/%d checks passed", label, sum(validity), len(expected))

    final_score = sum(scores[key] * SCORE_WEIGHTS[key] for key in scores)
    log.info("  → Final Quality Score: %.2f (%.1f%%)", final_score, final_score * 100)
    return final_score


def title_page_metadata_is_complete(metadata: Dict[str, Any]) -> bool:
    """
    Does this title-page metadata clear the completeness bar?

    This is the gate the metadata ladder in `plan.md` falls through on: each
    rung (text-layer DOI, Zotero lookup, text-only VLM, full two-shot) is tried
    until one produces metadata that satisfies this. It is exposed separately
    so the ladder and the quality score cannot drift apart — V5 had these same
    three checks written inline inside the scorer, where nothing else could
    reach them.

    Requires a title over five characters, at least one author, and a DOI
    containing "10." (every DOI begins with a "10." registrant prefix).
    """
    return all(_title_page_validity_checks(metadata))


# --- helpers (after the main functions, per CLAUDE.md) --------------------

def _structural_marker_score(content: str) -> float:
    """Fraction of the structural markers we hoped to see, capped at 1.0."""
    found = sum(1 for marker in STRUCTURAL_MARKERS if marker in content)
    return min(1.0, found / MARKERS_FOR_FULL_SCORE)


def _text_quality_score(content: str) -> float:
    """
    Four cheap sanity checks on the text itself.

    Together they catch the characteristic failure shapes: a wall of
    run-together words, missing spaces, no letters at all, or no punctuation
    (which usually means the model emitted a token soup rather than prose).
    """
    indicators = [
        len(re.findall(r"\b\w{10,}\b", content)) < len(content) * 0.1,  # few overly long words
        content.count(" ") / (len(content) + 1) > 0.1,                  # reasonable spacing
        re.search(r"[a-zA-Z]", content) is not None,                    # contains letters
        len(re.findall(r"[.,;:]", content)) > 5,                        # contains punctuation
    ]
    return sum(indicators) / len(indicators)


def _title_page_validity_checks(metadata: Dict[str, Any]) -> List[bool]:
    """Type-and-shape checks for the three title-page fields."""
    return [
        isinstance(metadata.get("title"), str) and len(metadata.get("title", "")) > 5,
        isinstance(metadata.get("authors"), list) and len(metadata.get("authors", [])) > 0,
        isinstance(metadata.get("doi"), str) and "10." in metadata.get("doi", ""),
    ]


def _body_page_validity_checks(metadata: Dict[str, Any]) -> List[bool]:
    """Type checks for the five body-page fields."""
    return [
        isinstance(metadata.get("primary_language"), str),
        isinstance(metadata.get("is_rotation_valid"), bool),
        isinstance(metadata.get("rotation_correction"), int),
        isinstance(metadata.get("is_table"), bool),
        isinstance(metadata.get("is_diagram"), bool),
    ]
