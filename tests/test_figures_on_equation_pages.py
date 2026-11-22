#!/usr/bin/env python3
"""
A page with equations still has figures, and they must not vanish.

Found by inspecting a real run: in the 20-page *Hydration-induced* paper, every
page classified `complex_visual` reported **zero** figures. The pages plainly
have figures — the ONNX detector finds them and crops them perfectly well when
asked. It was never asked.

The mechanism was an early return, and it is worth stating precisely because
the shape recurs:

    PageClassifier.classify
      RULE 2  equations -> return COMPLEX_VISUAL     <-- returns HERE
      RULE 3  visual scan -> analysis['visual_regions'] = [...]   never ran

    body_pipeline._detect_if_needed
      has_visuals = bool(analysis['visual_regions'])  <-- [] from the initializer

So `visual_regions` was an empty list meaning "nobody looked", read by a
consumer as "there is nothing there". Layout detection was skipped, and with it
every figure on the page. Silent: the page still read fine, the run still
succeeded, and `figure_count: 0` looks like an honest answer.

**Measured cost on the one paper that exposed it: 14 figures across 8 pages.**
Across the eight-paper corpus after the fix: 0 pages where the gate skips
detection while figures exist, over 451 body pages.

The fix runs both scans before any rule returns. Rule ORDER is untouched —
equations still win, and every page's TYPE is exactly what it was — so the
routing table pinned in `test_page_classifier.py` is unaffected.
"""

import pytest

from src.processing.pdf.base import ImageBlock, Rect
from src.processing.utils.page_classifier import PageClassifier, PageType

from .fakes import FakePage, word_row


@pytest.fixture
def classifier() -> PageClassifier:
    return PageClassifier({})


def equation_page_with_a_figure() -> FakePage:
    """
    What the Hydration paper's page 5 is: equations AND an embedded figure.

    Two LaTeX rows clear the equation threshold on their own, so Rule 2 decides
    the type — which is correct. The image must still be reported.
    """
    return FakePage(
        words=[
            *word_row([r"\frac{a}{b}", r"\sqrt{c}"], y=300.0),
            *word_row([r"\int_0", r"\sum_{n}"], y=340.0),
        ],
        images=[ImageBlock(rect=Rect(72.0, 400.0, 540.0, 700.0))],
    )


class TestTheAnalysisReportsWhatIsOnThePage:
    def test_an_equation_page_still_reports_its_figure(self, classifier):
        """THE BUG. The type is right; the analysis used to be empty."""
        page_type, analysis = classifier.classify(equation_page_with_a_figure(), 3)

        assert page_type is PageType.COMPLEX_VISUAL      # unchanged, and correct
        assert analysis["visual_regions"], \
            "the figure on this page was never looked for"

    def test_the_downstream_gate_now_opens(self, classifier):
        """
        The property that actually matters: `body_pipeline` runs layout
        detection when — and only when — this is truthy.
        """
        _, analysis = classifier.classify(equation_page_with_a_figure(), 3)

        has_visuals = bool(analysis.get("visual_regions"))
        assert has_visuals is True

    def test_an_equation_page_with_no_figure_reports_none(self, classifier):
        """The fix must not invent visuals — that would detect on every page."""
        page = FakePage(words=[
            *word_row([r"\frac{a}{b}", r"\sqrt{c}"], y=300.0),
            *word_row([r"\int_0", r"\sum_{n}"], y=340.0),
        ])
        page_type, analysis = classifier.classify(page, 3)

        assert page_type is PageType.COMPLEX_VISUAL
        assert analysis["visual_regions"] == []
        assert bool(analysis.get("visual_regions")) is False

    def test_visual_count_agrees_with_the_regions(self, classifier):
        _, analysis = classifier.classify(equation_page_with_a_figure(), 3)
        assert analysis["visual_count"] == len(analysis["visual_regions"])


class TestTheRulesThemselvesAreUnCHANGED:
    """
    Running the scans earlier must not move a single page to a different type.
    The routing table in test_page_classifier.py covers the benchmark; these
    cover the rule boundaries directly.
    """

    def test_equations_still_beat_visuals(self, classifier):
        """
        Rule ORDER is the thing being preserved. A page with BOTH an equation
        and enough figures for Rule 3 must still come back COMPLEX_VISUAL —
        otherwise this "fix" silently re-routes pages.
        """
        page = FakePage(
            words=[
                *word_row([r"\frac{a}{b}", r"\sqrt{c}"], y=300.0),
                *word_row([r"\int_0", r"\sum_{n}"], y=340.0),
            ],
            images=[ImageBlock(rect=Rect(72.0, 100.0, 300.0, 300.0)),
                    ImageBlock(rect=Rect(320.0, 100.0, 540.0, 300.0)),
                    ImageBlock(rect=Rect(72.0, 400.0, 540.0, 700.0))],
        )
        assert classifier.classify(page, 3)[0] is PageType.COMPLEX_VISUAL

    def test_a_figure_page_with_no_equations_is_still_simple_visual(self, classifier):
        page = FakePage(images=[
            ImageBlock(rect=Rect(72.0, 100.0, 300.0, 300.0)),
            ImageBlock(rect=Rect(320.0, 100.0, 540.0, 300.0)),
            ImageBlock(rect=Rect(72.0, 400.0, 540.0, 700.0)),
        ])
        assert classifier.classify(page, 3)[0] is PageType.SIMPLE_VISUAL

    def test_a_dense_text_page_is_still_text_only(self, classifier):
        """The cheap path must stay cheap — this is the whole V6 thesis."""
        page = FakePage(words=word_row(["hydration", "of", "collagen", "fibrils"],
                                       y=300.0))
        dense = {"word_count": 400, "avg_words_per_line": 12, "char_count": 2500}
        page_type, _ = classifier.classify(page, 3, text_stats=dense)

        assert page_type is PageType.TEXT_ONLY

    def test_the_title_page_short_circuits_before_any_scan(self, classifier):
        """
        Page 1 returns before both scans, deliberately: the title pipeline
        handles it and nothing downstream reads this analysis. Pinned because
        it is the one early return that SHOULD stay early.
        """
        _, analysis = classifier.classify(equation_page_with_a_figure(), 0)

        assert analysis["classification_reason"] == "first_page_title"
        assert "visual_count" not in analysis
