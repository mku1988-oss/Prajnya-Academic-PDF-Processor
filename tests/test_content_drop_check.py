#!/usr/bin/env python3
"""
Tests for the content-drop check — the pipeline's silent-data-loss alarm.

For biology researchers: this check is what notices that the vision model read
a page but quietly skipped its table. Getting it wrong is expensive in both
directions — too strict and every page escalates to a slow repair path, too
loose and numbers vanish from the extracted paper without a trace. So the tests
below pin down both the firing condition and, just as importantly, the three
situations where the check must decline to judge at all.

Two groups:

- **Logic tests** run everywhere, on hand-built pages (`tests/fakes.py`).
- **End-to-end tests** need the benchmark PDF and check the real separation
  measured in V5: a prose page scores far above the 0.90 threshold.
"""

from pathlib import Path

import pytest

from src.processing.pdf import get_backend
from src.processing.pdf.base import ImageBlock, Rect, Word
from src.processing.utils.content_drop_check import (
    MIN_GROUND_TRUTH_WORDS,
    CoverageResult,
    check_page_for_content_drop,
    extract_page_reference_text,
    has_reliable_text_layer,
    is_scanned_page,
    tokenize_reference,
    word_coverage,
)

from .fakes import FakePage, rotated_word, word_row
from tests.local_samples import BENCHMARK_PDF as LOCAL_BENCHMARK_PDF

BENCHMARK_PDF = LOCAL_BENCHMARK_PDF

needs_pdf = pytest.mark.skipif(
    not BENCHMARK_PDF.is_file(), reason="benchmark PDF unavailable"
)


def prose_page(word_count: int = 60, **kwargs) -> FakePage:
    """
    A page of prose laid out as rows of six words.

    Every word is DISTINCT ("wordaa", "wordab", ...). That matters: with a
    repeating vocabulary, truncating the reader's output would still leave one
    copy of every unique word behind, and the substring fallback would score
    it 100% — a fixture that could never fail the gate.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    vocabulary = [
        f"word{alphabet[i // 26]}{alphabet[i % 26]}" for i in range(word_count)
    ]
    words = []
    for row_index in range((word_count + 5) // 6):
        texts = vocabulary[row_index * 6:(row_index + 1) * 6]
        if not texts:
            break
        words.extend(word_row(texts, y=100.0 + row_index * 14.0))
    return FakePage(words=words[:word_count], text=" ".join(w.text for w in words), **kwargs)


# ---------------------------------------------------------------------------
# word_coverage — the measurement itself
# ---------------------------------------------------------------------------

def test_perfect_read_scores_full_coverage():
    reference = "the collagen fibril hydration was measured"
    coverage, total, missing = word_coverage(reference, reference)
    assert coverage == 1.0
    assert missing == 0
    assert total == 6


def test_dropped_table_collapses_coverage():
    """The failure this module exists to catch: half the page's words gone."""
    reference = "prose sentence one prose sentence two temperature relaxation amplitude fraction"
    ocr_output = "prose sentence one prose sentence two"
    coverage, _, missing = word_coverage(reference, ocr_output)
    assert coverage < 0.90
    assert missing == 4


def test_reordered_output_is_not_content_loss():
    """A VLM may reflow columns; only absence counts, not order."""
    coverage, _, missing = word_coverage(
        "alpha beta gamma delta", "delta gamma beta alpha"
    )
    assert coverage == 1.0
    assert missing == 0


def test_ligature_fragments_are_credited_by_substring():
    """
    Text layers often drop fi/fl glyphs, so "significantly" extracts as
    "signi cantly". Both halves must be credited to the correctly-read word,
    or the reader is punished for being better than its own ground truth.
    """
    coverage, _, _ = word_coverage("signi cantly increased", "significantly increased")
    assert coverage == 1.0


def test_short_repeated_words_are_counted_as_a_multiset():
    """Three "of"s in the reference need three in the output, not just one."""
    coverage, total, missing = word_coverage("of of of", "of")
    assert total == 3
    assert missing == 2
    assert coverage == pytest.approx(1 / 3)


def test_repeated_words_are_counted_as_a_multiset_at_any_length():
    """
    DECISION REVERSED 2026-11-13 (the researcher, option 1; eval/word_boxes/GATE.md
    Addendum 3). This test used to pin a deliberate leniency: any missing word
    of three or more letters was credited if it appeared anywhere in the
    output, on the reasoning that a whole vanished table would still trip the
    check. With exact word boxes that stopped being true: the benchmark page
    missing 40% of its words scored 0.903 and passed. Credit now covers only
    real ligature and line splits, so three "the"s need three.
    """
    coverage, total, missing = word_coverage("the the the", "the")
    assert total == 3
    assert missing == 2
    assert coverage == pytest.approx(1 / 3)


def test_a_word_inside_another_word_is_not_credited():
    """'for' is not present just because 'formula' is."""
    coverage, _, missing = word_coverage("for water", "formula waters")
    assert missing == 2
    assert coverage == 0.0


def test_a_ligature_dropped_at_the_start_of_a_word_is_credited():
    """'financial' extracts as 'nancial' when the fi glyph is dropped."""
    assert word_coverage("nancial report", "financial report")[0] == 1.0


def test_a_word_split_across_a_line_is_credited_by_its_two_halves():
    assert word_coverage("measure ments taken", "measurements taken")[0] == 1.0


def test_empty_reference_scores_one_rather_than_dividing_by_zero():
    assert word_coverage("", "anything at all") == (1.0, 0, 0)


def test_tokenizer_rejoins_words_hyphenated_across_a_line_break():
    assert "measurements" in tokenize_reference("measure-\nments were taken")


def test_tokenizer_ignores_digits_and_markup():
    """LaTeX the reader legitimately adds must not count as missing words."""
    tokens = tokenize_reference(r"$T_{2}$ = 47.3 ms")
    assert "ms" in tokens
    assert all(token.isalpha() for token in tokens)


# ---------------------------------------------------------------------------
# Page-level guards — when the check must decline to judge
# ---------------------------------------------------------------------------

def test_scanned_page_is_detected_by_one_full_page_image():
    page = FakePage(
        size=(612.0, 792.0),
        images=[ImageBlock(rect=Rect(5.0, 5.0, 607.0, 787.0))],
    )
    assert is_scanned_page(page)
    assert not has_reliable_text_layer(page)


def test_born_digital_figure_does_not_look_like_a_scan():
    """A half-column figure covers ~0.2 of the page; a scan covers ~0.95."""
    page = prose_page(200, images=[ImageBlock(rect=Rect(72.0, 100.0, 300.0, 450.0))])
    assert not is_scanned_page(page)


def test_a_scan_carrying_its_own_ocr_text_layer_is_still_never_judged():
    """
    The subtle one, and the reason `has_reliable_text_layer` checks for a scan
    BEFORE checking for text at all.

    A scanned PDF usually does carry a text layer — the output of whatever OCR
    engine processed it years ago. Our corpus has a 1960 journal scan whose
    layer reads "NH4CI", "AgNO9", "CaCI2 *6H,0". Scoring our reader against
    that produced 43.3% coverage on a page it had read CORRECTLY, which would
    have escalated every page of the document. You cannot use a worse OCR as
    ground truth for a better one.

    Note the page below has plenty of text: a test using an EMPTY scanned page
    would pass even with the scan guard removed, because the text-length check
    would reject it anyway.
    """
    page = prose_page(200, images=[ImageBlock(rect=Rect(0.0, 0.0, 612.0, 792.0))])
    assert is_scanned_page(page)
    assert len(page.text()) > 100, "fixture must have a text layer, or it proves nothing"

    result = check_page_for_content_drop(page, "")
    assert not result.checked
    assert not result.content_dropped
    assert "scanned" in result.reason


def test_page_with_too_little_text_is_not_judged():
    """A figure-only page would swing wildly on a single missed word."""
    page = prose_page(MIN_GROUND_TRUTH_WORDS - 10)
    result = check_page_for_content_drop(page, "")
    assert not result.checked
    assert not result.content_dropped
    assert "too few" in result.reason


def test_page_with_no_text_layer_is_not_judged():
    page = FakePage(text="Header only")
    assert not has_reliable_text_layer(page)
    assert not check_page_for_content_drop(page, "").checked


# ---------------------------------------------------------------------------
# Reference-text reconstruction
# ---------------------------------------------------------------------------

def test_reference_text_drops_rotated_marginalia():
    """
    Journals stamp "Downloaded from ..." down the outer margin. The reader
    correctly ignores it; counting it as ground truth would depress coverage
    on every page of the document.
    """
    page = FakePage(words=[
        *word_row(["collagen", "hydration", "measured"], y=100.0),
        rotated_word("Downloaded", x=580.0, y=200.0),
        rotated_word("from", x=580.0, y=260.0),
    ])
    reference = extract_page_reference_text(page)
    assert "collagen" in reference
    assert "Downloaded" not in reference


def test_reference_text_keeps_columns_on_separate_lines():
    """
    A two-column page must not fuse across the gutter, or a hyphenated word at
    a column edge would rejoin with the wrong neighbour.
    """
    left = word_row(["hydration", "of"], y=100.0, x=72.0)
    right = word_row(["collagen", "fibrils"], y=100.0, x=340.0)
    reference = extract_page_reference_text(FakePage(words=left + right))
    lines = reference.split("\n")
    assert len(lines) == 2
    assert "hydration of" in lines[0]
    assert "collagen fibrils" in lines[1]


# ---------------------------------------------------------------------------
# check_page_for_content_drop — the routing decision
# ---------------------------------------------------------------------------

def test_good_read_passes_the_gate():
    page = prose_page(60)
    result = check_page_for_content_drop(page, extract_page_reference_text(page))
    assert result.checked
    assert not result.content_dropped
    assert result.coverage == 1.0


def test_dropped_content_fails_the_gate_and_escalates():
    page = prose_page(60)
    reference = extract_page_reference_text(page)
    half = " ".join(reference.split()[:20])
    result = check_page_for_content_drop(page, half)
    assert result.checked
    assert result.content_dropped
    assert result.missing_words > 0


def test_threshold_is_configurable_per_call():
    """Stage 3 turns this gate into a validator, so the threshold must move."""
    page = prose_page(60)
    reference = extract_page_reference_text(page)
    partial = " ".join(reference.split()[:50])  # ~83% coverage

    assert check_page_for_content_drop(page, partial, min_coverage=0.90).content_dropped
    assert not check_page_for_content_drop(page, partial, min_coverage=0.50).content_dropped


def test_unchecked_result_never_reports_a_drop():
    """`content_dropped` is the routing flag; it must stay false when unjudged."""
    assert not CoverageResult(0.0, 0, 0, checked=False, reason="").content_dropped


# ---------------------------------------------------------------------------
# End to end on the benchmark paper
# ---------------------------------------------------------------------------

@needs_pdf
def test_benchmark_prose_page_is_born_digital_and_judgeable():
    with get_backend().open(BENCHMARK_PDF) as document:
        page = document.page(6)  # p7: prose only, 98.7% in V5's measurements
        assert not is_scanned_page(page)
        assert has_reliable_text_layer(page)


@needs_pdf
def test_benchmark_prose_page_scores_far_above_the_threshold():
    """
    Feeding the page's own text layer back as the "read" is the ceiling case:
    it must clear 0.90 comfortably, or the geometric line reconstruction has
    broken the reference text.
    """
    with get_backend().open(BENCHMARK_PDF) as document:
        page = document.page(6)
        result = check_page_for_content_drop(page, page.text())
        assert result.checked
        assert result.ground_truth_words > MIN_GROUND_TRUTH_WORDS
        assert result.coverage >= 0.95
        assert not result.content_dropped


@needs_pdf
def test_benchmark_page_with_a_table_removed_trips_the_gate():
    """The real firing condition: drop the last third of a page's words."""
    with get_backend().open(BENCHMARK_PDF) as document:
        page = document.page(3)  # p4: the two-table page
        words = page.text().split()
        truncated = " ".join(words[: int(len(words) * 0.6)])
        assert check_page_for_content_drop(page, truncated).content_dropped


def test_a_ligature_dropped_inside_a_word_is_credited():
    """Springer's text layer gives 'modifcations' for 'modifications' (fi glyph lost)."""
    assert word_coverage("collagen modifcations", "collagen modifications")[0] == 1.0


def test_hyphen_halves_are_credited_even_when_not_neighbours():
    """Two columns interleave in the reference, so a hyphen's halves can be apart."""
    assert word_coverage("colla left column gens", "collagens left column")[0] == 1.0


def test_pdfium_line_end_hyphen_is_rejoined():
    from src.processing.utils.content_drop_check import tokenize_reference
    assert tokenize_reference("colla\x02\ngens") == ["collagens"]
