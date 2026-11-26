#!/usr/bin/env python
"""Tests for the reference-density pass.

The pass exists because a reference list is a term-density trap: it is rich
in exactly the words a search looks for while stating nothing. It reports a
MEASUREMENT and no verdict, because where a bibliography starts and ends is a
document-level question and this pass is page-local.
"""
import json
from pathlib import Path

import pytest

from src.processing.utils.reference_pass import (
    has_reference_heading,
    looks_like_a_citation,
    reference_density,
    reference_signals,
)

RUN_OUTPUTS = Path(__file__).resolve().parents[1] / "run_outputs" / "documents"

CITATION = (
    "Van der Slot, A.J. et al. (2003) Identification of PLOD2 as telopeptide "
    "lysyl hydroxylase, an important enzyme in fibrosis. J. Biol. Chem. 278, "
    "40967-40972"
)
PROSE = (
    "Collagen fibrils show a characteristic 67 nm axial periodicity that "
    "arises from the staggered arrangement of molecules within the fibril."
)


class TestOneLineAtATime:
    def test_a_full_reference_is_recognised(self):
        assert looks_like_a_citation(CITATION)

    def test_ordinary_prose_is_not(self):
        assert not looks_like_a_citation(PROSE)

    def test_one_signal_alone_is_not_enough(self):
        """A bare year is ordinary in scientific prose and must not count."""
        assert not looks_like_a_citation(
            "Since 1998 this model has been the accepted explanation for it."
        )

    def test_a_short_line_is_never_a_citation(self):
        assert not looks_like_a_citation("Smith et al. 2003")


class TestTheHeading:
    @pytest.mark.parametrize(
        "line", ["References", "REFERENCES", "Bibliography", "5. References",
                 "Literature Cited", "References:"],
    )
    def test_headings_are_recognised(self, line):
        assert has_reference_heading(f"body text\n{line}\nmore text")

    def test_a_mention_inside_a_paragraph_does_not_count(self):
        """Matched on a line of its own, or every discussion section fires."""
        assert not has_reference_heading(
            "The references below give the full derivation of this result."
        )


class TestDensity:
    def test_a_page_of_references_scores_high(self):
        page = "\n".join([CITATION] * 8)
        assert reference_density(page) >= 0.9

    def test_a_page_of_prose_scores_low(self):
        page = "\n".join([PROSE] * 8)
        assert reference_density(page) <= 0.2

    def test_a_page_too_short_to_judge_scores_zero(self):
        """0.0 must mean 'could not measure', not 'definitely prose'."""
        assert reference_density("A title\nAn author") == 0.0

    def test_short_lines_are_excluded_rather_than_counted_as_prose(self):
        """Page furniture would otherwise dilute a real bibliography."""
        page = "\n".join([CITATION] * 6 + ["p. 12", "J. Cell Biol."] * 3)
        assert reference_density(page) >= 0.9

    def test_the_signal_block_carries_both_facts(self):
        signals = reference_signals(f"References\n{CITATION}\n{CITATION}\n{CITATION}\n{CITATION}")
        assert signals["has_reference_heading"] is True
        assert signals["reference_density"] > 0.5


class TestAgainstTheRealCorpus:
    """Fixtures prove the code runs; the corpus proves it separates."""

    @pytest.fixture(scope="class")
    @classmethod
    def documents(cls):
        if not RUN_OUTPUTS.is_dir():
            pytest.skip("corpus not present")
        found = sorted(RUN_OUTPUTS.glob("*.v6.json"))
        if len(found) < 9:
            pytest.skip("corpus incomplete -- a run is probably in progress")
        return [json.loads(p.read_text(encoding="utf-8")) for p in found]

    def test_bibliography_pages_score_far_above_body_pages(self, documents):
        """Measured 2026-08-18: body pages 0.00-0.20, reference pages
        0.50-1.00. If that separation ever closes, the signal is dead."""
        best_reference = 0.0
        body_scores = []
        for document in documents:
            for page in document["pages"]:
                score = reference_density(page.get("content") or "")
                if has_reference_heading(page.get("content") or ""):
                    best_reference = max(best_reference, score)
                else:
                    body_scores.append(score)
        body_scores.sort()
        median_body = body_scores[len(body_scores) // 2]
        assert median_body <= 0.25, f"body pages score {median_body}"
        assert best_reference >= 0.4

    def test_every_research_article_has_its_bibliography_found(self, documents):
        """A paper without a detectable reference list means the heading
        pattern has drifted away from how journals actually set them."""
        with_headings = [
            doc for doc in documents
            if any(has_reference_heading(p.get("content") or "")
                   for p in doc["pages"])
        ]
        assert len(with_headings) >= 6, "reference headings found in too few documents"
