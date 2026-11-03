#!/usr/bin/env python
"""Records how each heading-shaped line is TYPESET, so its depth can be read.

For biology researchers: a paper's outline lives in its typography. "RESULTS"
is a top-level heading and "Fibril diameter" is a sub-heading, and the only
things that say so are that one is bigger, bolder, or in capitals. The words
themselves are silent. This pass measures those three properties for every
line that could be a heading.

WHY IT EXISTS

Measured on the 9-document corpus by the retrieval layer: 554 headings were
recovered from text alone, and 415 (75%) carry a number like "3.2.3.1" whose
depth is unambiguous. The remaining 139 are unnumbered -- "Introduction",
"HYDROXYPROLINE AND COLLAGEN STABILITY" -- and **nothing in the text says how
deep they are.** Only this repo can answer, because our JSON ships the text
and not the fonts.

WHY THREE FEATURES AND NOT ONE

Measured on real pages, and this is the whole reason the first attempt failed:

    Collagen structure p9   "HYDROXYPROLINE AND COLLAGEN STABILITY"
        cap height 1.07x body, weight 650 vs body 420  -> BOLD decides it
    CollagenFibrilAssembly p29   "REFERENCES"
        cap height 1.38x body, weight not recorded     -> SIZE decides it

One heading is barely larger but bold; the other is much larger with no
weight recorded at all. **No single feature finds both**, so all three are
emitted and the consumer combines them.

An earlier attempt measured size from whole-WORD boxes and had to be thrown
away: capitals have no descenders, so an ALL-CAPS heading measured SHORTER
than mixed-case prose -- backwards on exactly the headings that need help.
`StyledLine.cap_height` measures capital glyphs only, which fixes it.

WHAT THIS DELIBERATELY DOES NOT DO

It emits MEASUREMENTS, never a level. Depth is a whole-document question --
one page cannot know whether its largest line is a chapter title or a minor
sub-head -- and the consumer already assembles documents. Deciding here would
be guessing with less information than the reader has.
"""
from __future__ import annotations

import logging
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence

# A line this much taller than the page's body text is a heading on size
# alone. Set from the measurements above: "REFERENCES" is 1.38x.
SIZE_RATIO_THRESHOLD = 1.15

# Heavier than this multiple of the page's body weight counts as bold.
# Body text is ~400 and bold ~650-700, so 1.2 separates them with room.
WEIGHT_RATIO_THRESHOLD = 1.2

# A line this fully capitalised is a heading candidate whatever its size,
# because small-caps section heads are a common house style.
CAPS_FRACTION_THRESHOLD = 0.9

# Headings are short. Generous enough for a two-line title, tight enough to
# exclude prose, which is the only thing it must do.
MAX_HEADING_CHARS = 140

# Past this, a page is a table of contents or a mis-grouped column, and
# recording all of it would bloat the JSON for no gain.
MAX_CANDIDATES_PER_PAGE = 40

# Running prose, however it is set. A heading may end in a colon or a
# question mark; a full stop followed by more words means a sentence.
_SENTENCE_END = re.compile(r"[.!?]\s+\S")

# A real word, not a stray glyph. Superscripts and subscripts sit on their
# own baseline and therefore form their own "lines" -- measured on the
# corpus these produce fragments like "4R", "(yJ" and a lone "H", every one
# of which is 100% caps and would otherwise pass as a heading.
_HAS_WORD = re.compile(r"[A-Za-z]{3,}")

_LOGGER = logging.getLogger("HeadingPass")


def heading_candidates_for_page(
    page: Any, logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """Measure every heading-shaped line on one page.

    Args:
        page: a `PdfPage` from either backend.
        logger: optional logger.

    Returns:
        One record per candidate, in reading order:

            text          the line as printed
            size_ratio    cap height over the page's body cap height, or 0.0
                          where the backend cannot measure ink height
            weight_ratio  font weight over the page's body weight, or 0.0
                          where the file does not record weights
            is_bold       weight_ratio over the bold threshold
            caps_fraction share of letters that are uppercase
            bbox          [x0, y0, x1, y1]

        Empty for a scan, which has no typography to measure. **A zero ratio
        means UNKNOWN, never small** -- see `StyledLine`.
    """
    log = logger or _LOGGER
    try:
        lines = page.styled_lines()
    except Exception as error:  # pragma: no cover - defensive
        log.debug("Heading pass skipped a page (%s); its content is unaffected.",
                  error)
        return []
    if not lines:
        return []

    body_size = body_cap_height(lines)
    body_weight = body_font_weight(lines)

    candidates = []
    for line in lines:
        if not is_heading_shaped(line.text):
            continue
        record = _as_candidate(line, body_size, body_weight)
        if _looks_like_a_heading(record):
            candidates.append(record)

    if len(candidates) > MAX_CANDIDATES_PER_PAGE:
        log.debug("Page has %d heading candidates; keeping the %d strongest.",
                  len(candidates), MAX_CANDIDATES_PER_PAGE)
        candidates.sort(key=lambda item: -_strength(item))
        candidates = candidates[:MAX_CANDIDATES_PER_PAGE]
        candidates.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return candidates


def body_cap_height(lines: Sequence[Any]) -> float:
    """The page's body-text cap height, as the MEDIAN over its lines.

    Median rather than mean, because one display heading or a large figure
    caption drags a mean upwards and then every real heading measures as
    ordinary. On a normal page the overwhelming majority of lines are body
    text, so the median IS the body size.

    Returns:
        The height in page units, or 0.0 when nothing could be measured.
    """
    heights = [line.cap_height for line in lines if line.cap_height > 0]
    return statistics.median(heights) if heights else 0.0


def body_font_weight(lines: Sequence[Any]) -> float:
    """The page's body font weight, as the median of the weights recorded.

    Returns 0.0 when the file records no weights at all, which is common
    enough that callers must treat it as "unknown" rather than "light".
    """
    weights = [line.weight for line in lines if line.weight > 0]
    return statistics.median(weights) if weights else 0.0


def is_heading_shaped(text: str) -> bool:
    """Whether a line could be a heading, judged on its text alone.

    Typography decides whether it IS one; this only rules out the lines that
    no amount of size or weight would make a heading -- running prose, page
    furniture, bare numbers, and stray super/subscript glyphs.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    if not _HAS_WORD.search(stripped):
        return False
    return not _SENTENCE_END.search(stripped)


# ---------------------------------------------------------------------------
# Helpers. Placed AFTER the main functions, never inside them.
# ---------------------------------------------------------------------------


def _as_candidate(line: Any, body_size: float, body_weight: float) -> Dict[str, Any]:
    """Turn one measured line into its JSON record."""
    size_ratio = line.cap_height / body_size if body_size and line.cap_height else 0.0
    weight_ratio = line.weight / body_weight if body_weight and line.weight else 0.0
    return {
        "text": line.text.strip(),
        "size_ratio": round(size_ratio, 3),
        "weight_ratio": round(weight_ratio, 3),
        "is_bold": weight_ratio >= WEIGHT_RATIO_THRESHOLD,
        "caps_fraction": round(line.caps_fraction, 3),
        "bbox": [round(float(v), 2) for v in
                 (line.rect.x0, line.rect.y0, line.rect.x1, line.rect.y1)],
    }


def _looks_like_a_heading(record: Dict[str, Any]) -> bool:
    """Any ONE of the three signals is enough to record the line.

    Deliberately permissive: this pass RECORDS candidates and the consumer
    decides. A false candidate costs a JSON row; a missed one is not
    recoverable downstream, because the fonts never leave this repo.
    """
    return (
        record["is_bold"]
        or record["size_ratio"] >= SIZE_RATIO_THRESHOLD
        or record["caps_fraction"] >= CAPS_FRACTION_THRESHOLD
    )


def _strength(record: Dict[str, Any]) -> float:
    """How heading-like a record is, for trimming an overlong page."""
    return max(record["size_ratio"], record["weight_ratio"], record["caps_fraction"])
