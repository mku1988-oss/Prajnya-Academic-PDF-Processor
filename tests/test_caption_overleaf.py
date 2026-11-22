#!/usr/bin/env python3
"""
Recovering the caption of a full-page figure printed one page before it.

For biology researchers: a journal running a full-page figure often prints the
caption on the NEXT page. The per-page labelling pass only reads the page a
figure sits on, so those figures arrive with no caption at all even though the
caption is in the document, one page away.

MEASURED on the 9-document sample corpus (107 figures) before this pass was
written, by reading the published JSON:

    uncaptioned figures                    13
      caption on the figure's OWN page      6   <- a different defect
      caption on an adjacent page           6   <- what this pass recovers
      no caption anywhere nearby            1

All five recoveries the rule proposes were verified before the code was
written: three by opening the crops (JRuberti p15 is a full-page Figure 10
whose panels A-E match its p16 caption; Hydration-induced p14 shows panels a-e
of Fig. 10, matching its p15 caption) and two by figure sequence (Fig 5 on p8,
Fig 6 on p9, a GAP, Fig 8 on p12 — so the unlabelled p10 crops are Fig 7,
whose caption is on p11).

THE DEFECT THIS FILE EXISTS TO PIN, found by simulating the rule against the
real corpus before any engine code was written:

    "Figure 4 shows the progression of the cell culture ..."

is running prose, and it BEGINS ITS LINE, so `find_label_lines` returns it as
a perfectly good label line. On its own page the geometric adjacency check
rejects it; across a page boundary there is no geometry left, so the first
version of the rule cheerfully attached a sentence about cell culture to a
figure it does not describe. A caption is a label followed by a TITLE; a
citation is a label followed by a VERB.
"""

import pytest

from src.processing.utils.caption_overleaf import (
    reads_like_a_caption,
    recover_overleaf_captions,
)
from src.processing.utils.figure_label_pass import LabelLine


def line(text: str) -> LabelLine:
    """A LabelLine carrying `text`, with the label parsed out of it."""
    from src.processing.utils.figure_labels import parse_printed_label

    printed, key = parse_printed_label(text)
    return LabelLine(printed=printed, key=key, text=text, rect=(0, 0, 10, 10))


def orphan(figure_id: str, page: int) -> dict:
    """A figure record with neither caption nor label — the case to recover."""
    return {"figure_id": figure_id, "page": page, "caption": "",
            "printed_label": None, "label_key": None, "panel_group_size": 1}


def captioned(figure_id: str, page: int, caption: str) -> dict:
    """A figure record the per-page pass already placed."""
    from src.processing.utils.figure_labels import parse_printed_label

    printed, key = parse_printed_label(caption)
    return {"figure_id": figure_id, "page": page, "caption": caption,
            "printed_label": printed, "label_key": key, "panel_group_size": 1}


class TestTellingACaptionFromACitation:
    """The discriminator, which is the whole safety of this pass."""

    @pytest.mark.parametrize("text", [
        "Fig. 10 | Bioinspired self-shaping materials manufactured by 4D printing",
        "Figure 10. Mechanistic models for filament assembly in early environments",
        "Figure 4 Variation of the unit twist as a function of temperature",
        "Fig. 2.2 Fibril-forming collagens and their assembly. (a) Fibrils",
        "Figure 7: Hydration-induced shape recovery of bird feathers",
    ])
    def test_a_label_followed_by_a_title_is_a_caption(self, text):
        assert reads_like_a_caption(text)

    @pytest.mark.parametrize("text", [
        "Figure 4 shows the progression of the cell culture before confluence",
        "Figure 10 depicts a stretching filament with a cell surface extension",
        "Fig. 3 illustrates the packing of collagen molecules in the fibril",
    ])
    def test_a_label_followed_by_a_verb_is_a_citation(self, text):
        """The defect that made this module's first version unsafe."""
        assert not reads_like_a_caption(text)

    def test_a_line_with_no_label_at_all_is_not_a_caption(self):
        assert not reads_like_a_caption("The progression of the cell culture")


class TestRecoveringFromTheFollowingPage:
    """The case the sample corpus actually contains."""

    def test_a_full_page_figure_adopts_the_caption_overleaf(self):
        figures = {15: [orphan("p15_fig1", 15)]}
        lines = {16: [line("Figure 10. Mechanistic models for filament assembly")]}

        assert recover_overleaf_captions(figures, lines) == 1
        assert figures[15][0]["printed_label"] == "Figure 10"
        assert figures[15][0]["label_key"] == "figure 10"
        assert "Mechanistic models" in figures[15][0]["caption"]

    def test_every_panel_on_the_page_gets_the_same_caption(self):
        """Several crops of one printed figure are panels, not competitors."""
        figures = {15: [orphan("p15_fig1", 15), orphan("p15_fig2", 15)]}
        lines = {16: [line("Figure 10. Mechanistic models for filament assembly")]}

        assert recover_overleaf_captions(figures, lines) == 2
        assert {f["label_key"] for f in figures[15]} == {"figure 10"}
        assert all(f["panel_group_size"] == 2 for f in figures[15])

    def test_the_following_page_is_preferred_over_the_preceding_one(self):
        """Measured, not assumed — see NEIGHBOUR_OFFSETS.

        Hydration-induced p14 has Fig. 9's caption on p13 and Fig. 10's on
        p15, and the crop is Fig. 10. A symmetric rule gets this wrong.
        """
        figures = {14: [orphan("p14_fig1", 14)]}
        lines = {
            13: [line("Fig. 9 | Humidity-driven supercontraction of spider silk")],
            15: [line("Fig. 10 | Bioinspired self-shaping materials by 4D printing")],
        }

        recover_overleaf_captions(figures, lines)
        assert figures[14][0]["printed_label"] == "Fig. 10"

    def test_the_preceding_page_is_used_when_there_is_nothing_after(self):
        figures = {21: [orphan("p21_fig1", 21)]}
        lines = {20: [line("Fig. 2.2 Fibril-forming collagens and their assembly")]}

        assert recover_overleaf_captions(figures, lines) == 1
        assert figures[21][0]["printed_label"] == "Fig. 2.2"


class TestWhatItRefusesToDo:
    """An unplaceable figure is left unlabelled rather than guessed at."""

    def test_it_refuses_a_citation_on_the_neighbouring_page(self):
        """The JRuberti p9 case, which the first version of the rule failed."""
        figures = {9: [orphan("p9_fig1", 9)]}
        lines = {8: [line("Figure 4 shows the progression of the cell culture")]}

        assert recover_overleaf_captions(figures, lines) == 0
        assert figures[9][0]["caption"] == ""
        assert figures[9][0]["printed_label"] is None

    def test_it_refuses_a_caption_already_claimed_elsewhere(self):
        """A printed figure is captioned once. A second claim is a citation."""
        figures = {
            5: [captioned("p5_fig1", 5, "Fig. 7 | Shape recovery of bird feathers")],
            10: [orphan("p10_fig1", 10)],
        }
        lines = {11: [line("Fig. 7 | Shape recovery of bird feathers")]}

        assert recover_overleaf_captions(figures, lines) == 0
        assert figures[10][0]["printed_label"] is None

    def test_it_refuses_when_the_neighbour_has_its_own_captioned_figure(self):
        """That page's caption plausibly belongs to that page's figure."""
        figures = {
            9: [captioned("p9_fig1", 9, "Fig. 6 | Principal mechanisms in plants")],
            10: [orphan("p10_fig1", 10)],
        }
        lines = {9: [line("Fig. 6 | Principal mechanisms in plants")]}

        assert recover_overleaf_captions(figures, lines) == 0

    def test_it_leaves_an_already_captioned_figure_alone(self):
        """The per-page geometric pairing is stronger evidence than this."""
        original = "Fig. 5 | Effect of hydration on turgor pressure in a stem"
        figures = {8: [captioned("p8_fig1", 8, original)]}
        lines = {9: [line("Fig. 6 | Principal mechanisms in plants")]}

        assert recover_overleaf_captions(figures, lines) == 0
        assert figures[8][0]["caption"] == original

    def test_it_leaves_a_figure_that_has_a_label_but_no_caption(self):
        """A label from the page it sits on outranks anything across a break."""
        figure = orphan("p7_fig1", 7)
        figure["printed_label"] = "Fig. 4"
        figure["label_key"] = "figure 4"
        figures = {7: [figure]}
        lines = {8: [line("Fig. 9 | Humidity-driven supercontraction")]}

        assert recover_overleaf_captions(figures, lines) == 0
        assert figures[7][0]["printed_label"] == "Fig. 4"

    def test_a_panel_letter_does_not_count_as_a_caption(self):
        """"d" is a panel sub-label the detector picked up, not a caption."""
        figure = orphan("p7_fig1", 7)
        figure["caption"] = "d"
        figures = {7: [figure]}
        lines = {8: [line("Fig. 4 | Hygroscopic structure in an ice plant")]}

        assert recover_overleaf_captions(figures, lines) == 1
        assert figures[7][0]["printed_label"] == "Fig. 4"

    def test_a_panel_sub_caption_still_needs_the_figures_label(self):
        """
        Hydration p10, the corpus's last unlabelled crop: panel a of Fig. 7
        came with the sentence printed under it as its "caption". Long enough
        to pass for one, so panels b and c adopted Fig. 7 and panel a did not.
        It now joins them, and keeps its own panel text.
        """
        panel_a = orphan("p10_fig1", 10)
        panel_a["caption"] = "a The shape recovery of the shaft of a bird feather"
        figures = {10: [panel_a, orphan("p10_fig2", 10), orphan("p10_fig3", 10)]}
        lines = {11: [line("Fig. 7 | Hydration-induced shape and strength recovery")]}

        assert recover_overleaf_captions(figures, lines) == 3
        assert {f["printed_label"] for f in figures[10]} == {"Fig. 7"}
        assert {f["panel_group_size"] for f in figures[10]} == {3}
        assert figures[10][0]["caption"].startswith("a The shape recovery")
        assert figures[10][1]["caption"].startswith("Fig. 7")

    @pytest.mark.parametrize("caption, is_panel", [
        ("a The shape recovery of the shaft of a bird feather", True),
        ("(b) Cross section of a fibril", True),
        ("a | Artificial skin", True),
        ("A schematic of the collagen triple helix", False),   # a real caption
        ("an overview of the setup", False),
        ("Fig. 7 | Hydration-induced shape", False),           # carries its label
    ])
    def test_what_counts_as_a_panel_sub_caption(self, caption, is_panel):
        from src.processing.utils.caption_overleaf import is_panel_caption
        assert is_panel_caption(caption) is is_panel

    def test_nothing_nearby_leaves_the_figure_untouched(self):
        figures = {61: [orphan("p61_fig1", 61)]}
        lines = {60: [], 62: []}

        assert recover_overleaf_captions(figures, lines) == 0
        assert figures[61][0]["caption"] == ""

    def test_two_different_pages_cannot_both_take_one_caption(self):
        """Uniqueness holds across the whole document, not just per page."""
        figures = {14: [orphan("p14_fig1", 14)], 16: [orphan("p16_fig1", 16)]}
        lines = {15: [line("Fig. 10 | Bioinspired self-shaping materials")]}

        assert recover_overleaf_captions(figures, lines) == 1
        taken = [f for page in figures.values() for f in page if f["printed_label"]]
        assert len(taken) == 1


class TestItCannotDamageADocumentThatIsAlreadyFine:
    """The property that matters for a released engine: it can only add."""

    def test_a_document_with_every_caption_placed_is_untouched(self):
        figures = {
            1: [captioned("p1_fig1", 1, "Fig. 1 | The hierarchy of tendon")],
            2: [captioned("p2_fig1", 2, "Fig. 2 | Fibril diameter distribution")],
        }
        before = {p: [dict(f) for f in fs] for p, fs in figures.items()}
        lines = {1: [line("Fig. 1 | The hierarchy of tendon")],
                 2: [line("Fig. 2 | Fibril diameter distribution")]}

        assert recover_overleaf_captions(figures, lines) == 0
        assert figures == before

    def test_a_document_with_no_figures_at_all_is_a_no_op(self):
        assert recover_overleaf_captions({}, {}) == 0


class TestTheOrnamentThatHidTheOverleafCaption:
    """A Nature-family ornament stopped the very case this module exists for.

    Nature sets "\u25c0" before a caption belonging to the figure on the FACING
    page. `parse_printed_label` anchors at the start of the line, so the
    ornament made the caption invisible — the mark meaning "the figure is
    overleaf" was what stopped the overleaf figure being found.

    It went unnoticed because THE TWO BACKENDS DISAGREE: PyMuPDF drops the
    ornament, pypdfium2 returns it as a word. Verified by hand on
    Hydration-induced pages 11 and 15.
    """

    def test_an_ornamented_caption_is_found(self):
        from src.processing.utils.figure_label_pass import find_label_lines

        found = find_label_lines(
            [("\u25c0 Fig. 10 | Bioinspired self-shaping materials", 0, 0, 9, 9)])
        assert [line.printed for line in found] == ["Fig. 10"]

    def test_the_ornament_is_stripped_from_the_stored_caption(self):
        from src.processing.utils.figure_label_pass import find_label_lines

        found = find_label_lines(
            [("\u25c0 Fig. 7 Hydration-induced shape recovery", 0, 0, 9, 9)])
        assert found[0].text.startswith("Fig. 7")

    @pytest.mark.parametrize("text", [
        "(Figure 7), whereas 4(S)-methylproline will favour a C-exo pucker",
        "(Fig. 3.3 ) (Kadler et al. 1996 ). End-to-end fusion would require",
        "( Figure 2 a) to allow transfer line measurements (TLMs) that",
    ])
    def test_a_parenthesised_citation_is_still_rejected(self, text):
        """MEASURED, and the reason the rule is narrow.

        Stripping any leading punctuation unblocks 21 lines across the corpus,
        of which only 2 are captions — the other 19 are citations like these.
        So "(" and "[" are pointedly not ornaments.
        """
        from src.processing.utils.figure_label_pass import find_label_lines

        assert find_label_lines([(text, 0, 0, 9, 9)]) == []
