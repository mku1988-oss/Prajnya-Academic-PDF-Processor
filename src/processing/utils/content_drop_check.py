#!/usr/bin/env python3
"""
Content-Drop Check — catch the text a recognition model silently lost.

For biology researchers: our offline OCR engine reads a page image and returns
its text. It is accurate on prose, but it has no internal model of page LAYOUT.
When a table has to compete with running prose on the same page, the reader
sometimes drops the table entirely — and then invents a sentence or two to
paper over the gap. Nothing errors. The numbers simply vanish.

We do not try to PREDICT which pages hold tables. That turned out to be the
hard problem: text-layer table finders are blind to the "booktabs" style used
by most journals (horizontal rules only, no vertical ones), and a predictor's
false negatives are silent data loss.

Instead we VERIFY. On a born-digital PDF the embedded text layer is a free,
exact record of every word on the page. If the reader dropped a table, the
table's words are missing from its output and word coverage against that text
layer collapses. We measure the collapse and escalate only the pages that fail.

Measured separation on our corpus (V5, unchanged by this port):

    page                       has table   coverage
    NMR p4                     yes           82.9%
    CollagenFibrilAssembly p16 yes           60.6%
    NMR p7  (prose only)       no            98.7%
    CollagenFibrilAssembly p12 no            99.2%

A threshold of 0.90 sits in a 16-point gap between the two populations.

IMPORTANT — this check only works on born-digital pages. A scanned page has no
trustworthy text layer, and substituting Tesseract's output as the reference
does NOT work: Tesseract is a worse OCR than our engine, so its own errors
("NH4CI", "AgNO9") register as words the engine "lost", and the engine is
penalised for reading the page correctly. See `has_reliable_text_layer()`.

Ported from V5's `content_drop_check.py`. One deliberate change: V5 filtered
rotated marginalia using PyMuPDF's per-line writing direction; the `PdfBackend`
seam does not expose direction, so the reference text is now rebuilt from word
boxes (`group_words_into_lines`), which drops rotated words by their box shape
instead. The coverage math is untouched.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.processing.pdf.base import PdfPage, group_words_into_lines, line_text

# Alphabetic tokens of 2+ characters. We ignore digits and punctuation here
# because the reader legitimately adds LaTeX markup ($, \mathrm, _{}) that
# never existed in the text layer.
WORD_PATTERN = re.compile(r"[A-Za-z]{2,}")

# Journals hyphenate across line breaks ("measure-\nments"). The text layer
# preserves the break; the reader reads the printed word whole. Rejoin before
# comparing, or the reader looks wrong.
HYPHEN_BREAK_PATTERN = re.compile(r"-\s*\n\s*")

# A page needs some real text before coverage means anything. A figure-only
# page with six words would swing wildly on a single miss.
MIN_GROUND_TRUTH_WORDS = 30

# Below this many characters, the "text layer" is probably just a stray header
# rather than a real digital layer.
MIN_TEXT_LAYER_CHARS = 100

# A scanned page is a photograph of paper: one raster image covering
# essentially the whole page. Born-digital pages embed figures, which are much
# smaller. Measured on our corpus: scanned pages 0.95 of page area;
# born-digital pages 0.00-0.22.
SCANNED_PAGE_IMAGE_COVERAGE = 0.80


@dataclass
class CoverageResult:
    """
    Outcome of comparing one page's OCR output against its digital text layer.

    Attributes:
        coverage: fraction of text-layer words found in the OCR output (0.0-1.0)
        ground_truth_words: how many words the text layer contained
        missing_words: how many of them the OCR output lacked
        checked: False when the page had no usable text layer (scanned) or too
            little text
        reason: human-readable explanation, for the ledger and the logs
        threshold: the coverage the page had to reach to pass
    """
    coverage: float
    ground_truth_words: int
    missing_words: int
    checked: bool
    reason: str
    threshold: float = 0.90

    @property
    def content_dropped(self) -> bool:
        """True only when we actually ran the check and it failed."""
        return self.checked and self.coverage < self.threshold


def is_scanned_page(page: PdfPage) -> bool:
    """
    Is this page a photograph of paper rather than a typeset digital page?

    A scan is one big raster image filling the sheet. A born-digital page may
    embed figures, but they occupy a fraction of it. Measured on our corpus:
    scanned pages have an image covering 0.95 of the page; born-digital pages
    top out at 0.22 (a half-column figure).

    Args:
        page: the page, via the PdfBackend seam

    Returns:
        True when a single image covers most of the page.
    """
    width, height = page.size
    page_area = width * height
    if page_area <= 0:
        return False

    for image in page.images():
        if image.rect.area / page_area >= SCANNED_PAGE_IMAGE_COVERAGE:
            return True
    return False


def has_reliable_text_layer(page: PdfPage) -> bool:
    """
    Can this page's embedded text be trusted as ground truth?

    Two conditions, and the second is the subtle one:

    1. There must be enough extractable text to compare against.
    2. The page must be BORN-DIGITAL. A scanned PDF often carries a text layer
       too — but that layer is the output of whatever OCR engine processed it
       years ago, and it is frequently WORSE than our engine. Our test corpus
       contains exactly this: a 1960 journal scan whose embedded layer reads
       "NH4CI", "AgNO9", "CaCI2 *6H,0". Scoring our reader against it produced
       a 43.3% coverage on a page it had read CORRECTLY — a false content-drop
       that would have escalated every page of the document.

    You cannot use a worse OCR as ground truth for a better one. If the page
    is a scan, we decline to judge it.

    Args:
        page: the page, via the PdfBackend seam

    Returns:
        True when the page has a trustworthy digital text layer.
    """
    if is_scanned_page(page):
        return False
    return len(page.text().strip()) >= MIN_TEXT_LAYER_CHARS


def extract_page_reference_text(page: PdfPage) -> str:
    """
    Pull the page's horizontal text, skipping rotated marginalia.

    Many journal scans and downloads stamp a rotated copyright notice down the
    outer margin ("Downloaded from https://... on 25/03/2024"). It is not part
    of the article, our OCR engine correctly ignores it, and counting it as
    ground truth would depress coverage on every single page.

    Lines are rebuilt geometrically from word boxes (see
    `group_words_into_lines`), which drops rotated words by their box shape —
    the seam-level replacement for V5's writing-direction filter.

    Args:
        page: the page, via the PdfBackend seam

    Returns:
        The page's body text, one line per reconstructed typeset line.
    """
    lines = group_words_into_lines(page.words())
    return "\n".join(line_text(line) for line in lines)


def tokenize_reference(text: str) -> list:
    """
    Words of the digital text layer, repaired for comparison against a VLM's
    reading.

    Two repairs, both of which exist because the TEXT LAYER is dirtier than the
    printed page:

    1. Hyphenation. "measure-\\nments" is one word on paper.
    2. Ligature loss. Many PDFs drop the fi/fl glyphs when extracted, so
       "significantly" comes out as "signi cantly". We cannot repair that here,
       so `word_coverage` credits such fragments by substring instead.

    Args:
        text: raw text-layer content

    Returns:
        Lower-cased word tokens.
    """
    dehyphenated = HYPHEN_BREAK_PATTERN.sub("", text)
    return [word.lower() for word in WORD_PATTERN.findall(dehyphenated)]


def word_coverage(reference_text: str, ocr_output: str) -> tuple:
    """
    What fraction of the page's real words survived into the OCR output?

    Compared as a multiset, order-insensitively: a VLM may reflow columns or
    move a caption, and that is not content loss. Only absence is.

    Ligature fragments left by the text layer ("signi", "cantly") are credited
    when they appear inside a word the OCR read correctly ("significantly") —
    otherwise the reader is punished for being more accurate than its own
    ground truth. The same fallback absorbs hyphen halves whose line break the
    geometric reconstruction did not preserve (e.g. across a column boundary).

    Args:
        reference_text: the page's digital text layer
        ocr_output: what the OCR engine returned for the same page

    Returns:
        (coverage 0.0-1.0, number of reference words, number of missing words)
    """
    reference_words = tokenize_reference(reference_text)
    if not reference_words:
        return 1.0, 0, 0

    output_words = [word.lower() for word in WORD_PATTERN.findall(ocr_output)]

    remaining_counts = {}
    for word in output_words:
        remaining_counts[word] = remaining_counts.get(word, 0) + 1
    output_blob = " ".join(output_words)

    found = 0
    for word in reference_words:
        if remaining_counts.get(word, 0) > 0:
            remaining_counts[word] -= 1
            found += 1
        elif len(word) >= 3 and word in output_blob:
            found += 1  # ligature fragment absorbed into a correctly-read word

    missing = len(reference_words) - found
    return found / len(reference_words), len(reference_words), missing


def check_page_for_content_drop(
    page: PdfPage,
    ocr_output: str,
    min_coverage: float = 0.90,
    logger: Optional[logging.Logger] = None,
) -> CoverageResult:
    """
    Decide whether the OCR engine silently lost content on this page.

    This is the routing mechanism for the table-repair path, not a warning. A
    page that fails is escalated: a layout detector locates the table, we crop
    it, and the OCR engine re-reads the crop, where it is accurate.

    Args:
        page: the page, via the PdfBackend seam (source of the ground truth)
        ocr_output: text the OCR engine returned for this page
        min_coverage: fraction of words that must survive (default 0.90)
        logger: optional logger

    Returns:
        A CoverageResult. Check `.content_dropped` to decide whether to
        escalate.

    Example:
        >>> result = check_page_for_content_drop(page, glm_text)
        >>> if result.content_dropped:
        ...     repair_tables_on_page(page)
    """
    log = logger or logging.getLogger("ContentDropCheck")

    if not has_reliable_text_layer(page):
        return CoverageResult(
            coverage=1.0, ground_truth_words=0, missing_words=0, checked=False,
            reason="no reliable text layer (scanned page) - cannot verify",
            threshold=min_coverage,
        )

    reference_text = extract_page_reference_text(page)
    coverage, total_words, missing = word_coverage(reference_text, ocr_output)

    if total_words < MIN_GROUND_TRUTH_WORDS:
        return CoverageResult(
            coverage=coverage, ground_truth_words=total_words, missing_words=missing,
            checked=False,
            reason=f"only {total_words} reference words - too few to judge",
            threshold=min_coverage,
        )

    result = CoverageResult(
        coverage=coverage, ground_truth_words=total_words, missing_words=missing,
        checked=True,
        reason=f"{coverage:.1%} of {total_words} words survived",
        threshold=min_coverage,
    )

    if result.content_dropped:
        log.warning(
            f"Content drop on page {page.number + 1}: {result.reason} "
            f"(threshold {min_coverage:.0%}) - escalating for table repair"
        )
    else:
        log.debug(f"Page {page.number + 1} coverage OK: {result.reason}")

    return result
