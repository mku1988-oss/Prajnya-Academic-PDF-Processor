#!/usr/bin/env python3
"""
`p3_fig1` is where a figure sits. "Figure 2" is what the paper calls it.

For biology researchers: a figure's id encodes its position — page 3, first
figure found there — because a filename must be unique and stable. But the
body text does not cite positions, it cites numbers: "as shown in Fig. 2".
Measured on the sample corpus, **45 of 107 figures have an id whose number
disagrees with the printed one**, so nothing in the record connected the
phrase a reader searches for to the image they want.

MEASURED, on the 107 figures of the sample corpus:

    caption begins with a real label      50    <- what this module reads
    caption empty                         28
    caption present but not the caption   29    (a panel letter, or body prose
                                                 that spilled into the box)

So this closes about half the gap, and the other half is a caption-CAPTURE
problem living upstream in the detector's caption box — not a parsing one.

WHY THIS DOES NOT SCAN THE PAGE. Reading a "Figure N" line off the text layer
near the figure recovers 9 more and agrees with the caption 15/15 where both
exist. It also reproduces the trap `figure_escalation` already documents: a
figure is routinely CITED pages from where it is PRINTED. In the tendon
volume, page 20 carries "…packing is affected (Fig. 2.2). The rigid COL2
domain…" in running prose. A nearest-line rule adopts that citation as the
page's figure label and is confidently wrong — worse than no label, because
nothing downstream can tell. The caption box was already paired to its figure
geometrically; the page was not.
"""

import pytest

from src.processing.utils.figure_labels import parse_printed_label


class TestTheCommonForms:
    @pytest.mark.parametrize("caption,printed,key", [
        # Real captions from the sample corpus.
        ("Fig. 2. T1 analysis of a saturation recovery curve",
         "Fig. 2", "figure 2"),
        ("Figure 3 Helical twist in the structure of collagen",
         "Figure 3", "figure 3"),
        ("Fig. 1 | Mechanisms of hydration - induced movement.",
         "Fig. 1", "figure 1"),
        ("FIGURE 7. Collagen filament mutability",
         "FIGURE 7", "figure 7"),
    ])
    def test_the_label_is_read_and_normalized(self, caption, printed, key):
        assert parse_printed_label(caption) == (printed, key)

    def test_a_chapter_scoped_number_survives_intact(self):
        """
        An edited volume restarts numbering per chapter, so the tendon book
        prints "Fig. 2.1". Truncating that to "2" would merge fifteen
        chapters' worth of figure 1s into one.
        """
        assert parse_printed_label("Fig. 2.1 Hierarchical structure of tendon") == \
            ("Fig. 2.1", "figure 2.1")

    def test_a_roman_numeral_is_read(self):
        assert parse_printed_label("Figure IV. Apparatus")[1] == "figure iv"

    @pytest.mark.parametrize("kind", ["Scheme 1", "Chart 2", "Plate 3"])
    def test_other_printed_kinds_keep_their_own_word(self, kind):
        """A scheme is not a figure; merging them would collide numbers."""
        printed, key = parse_printed_label(f"{kind}. Something")
        assert printed == kind
        assert key.split()[0] == kind.split()[0].lower()


class TestTheKeyIsWhatMakesRetrievalWork:
    def test_every_spelling_collapses_to_one_key(self):
        """
        THE POINT OF THE KEY. A paper writes "Fig. 2" in the caption and
        "Figure 2" in the body, or the reverse. A RAG joining the two must
        not care which appeared where.
        """
        spellings = ["Fig. 2 Caption", "Fig 2 Caption", "Figure 2 Caption",
                     "FIGURE 2 Caption", "figure 2 caption", "Figs. 2 Caption"]
        keys = {parse_printed_label(text)[1] for text in spellings}

        assert keys == {"figure 2"}

    def test_the_printed_form_is_preserved_separately(self):
        """
        The key is for matching; the printed form is what a citation should
        display. "Fig. 2" must not be shown to a reader as "figure 2".
        """
        printed, key = parse_printed_label("Fig. 2 | Something")
        assert printed == "Fig. 2"
        assert key == "figure 2"


class TestOnlyTheStartCounts:
    def test_a_number_mentioned_later_is_not_this_figure(self):
        """
        A caption that ends "…compared with Figure 5" is describing a DIFFERENT
        figure. Matching anywhere in the string would mislabel this one.
        """
        caption = "Fig. 2. Stress-strain response, compared with Figure 5"
        assert parse_printed_label(caption)[1] == "figure 2"

    def test_a_caption_that_only_mentions_another_figure_gets_nothing(self):
        assert parse_printed_label("compared with Figure 5 above") == (None, None)

    def test_a_body_text_citation_is_not_a_label(self):
        """
        THE TRAP, in miniature. This is the exact shape of the tendon volume's
        page 20 text, and why the page is not scanned.
        """
        assert parse_printed_label("packing is affected (Fig. 2.2 ). The rigid "
                                   "COL2 domain") == (None, None)


class TestTheFiguresThatHaveNoLabel:
    """
    57 of 107 in the corpus. None is the honest answer — a fabricated label
    would connect the wrong image to a citation.
    """

    def test_an_empty_caption_yields_nothing(self):
        assert parse_printed_label("") == (None, None)

    def test_a_missing_caption_yields_nothing(self):
        assert parse_printed_label(None) == (None, None)

    def test_a_stray_panel_letter_is_not_a_label(self):
        """Real captured 'captions' from Hydration p7: 'd', 'f'."""
        assert parse_printed_label("d") == (None, None)
        assert parse_printed_label("f") == (None, None)

    def test_body_prose_that_spilled_into_the_box_is_not_a_label(self):
        """Real captured text from Tendon p16."""
        assert parse_printed_label(
            "cles; this collection of cells and fbers is surrounded") == (None, None)

    def test_a_panel_description_without_a_number_is_not_a_label(self):
        assert parse_printed_label("( A ) Top view of the crystal structure") == \
            (None, None)


class TestItDoesNotOverMatch:
    def test_a_word_merely_starting_with_fig_is_rejected(self):
        assert parse_printed_label("Figurative language in science") == (None, None)

    def test_a_label_needs_a_number(self):
        assert parse_printed_label("Figure showing the apparatus") == (None, None)

    def test_a_number_running_into_letters_is_not_a_label(self):
        """"Fig 2a" is a PANEL of figure 2 — handled, but not as "2a"."""
        printed, key = parse_printed_label("Fig 2a shows the detail")
        assert (printed, key) == (None, None)

    def test_whitespace_from_a_two_column_text_layer_is_collapsed(self):
        assert parse_printed_label("Fig.   2   |  Something")[0] == "Fig. 2"
