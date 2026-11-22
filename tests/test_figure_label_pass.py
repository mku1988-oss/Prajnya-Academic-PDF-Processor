#!/usr/bin/env python3
"""
Rejoining the panels of one printed figure, and labelling them from the page.

For biology researchers: the layout detector finds *panels*, not figures. A
figure printed as "Fig. 11" with three stacked panels comes back as three
crops, and only the one touching the caption gets it — so a RAG asked for
Figure 11 returns a third of it. This pass reads the caption lines out of the
text layer and gives every crop of one printed figure the same label.

MEASURED over the sample corpus's 107 figures, against the 50 labels derived
independently from geometrically-paired caption boxes:

    reproduces those labels          48 / 50
    CONTRADICTS one                   0 / 50   <-- the property that matters
    additionally recovered           45
    still unlabelled                 12
    printed figures rejoined from
    more than one crop               12

The two it does not reproduce are LOST, not wrong — the pass declines where
the caption box succeeded. That is why the production order is caption first,
this pass second: it can only ever add.

TWO DEFECTS THIS FILE PINS, both found by that measurement and both invisible
in a unit test written from the design rather than from data:

1. **Reading order.** Words are sorted by (y, x) globally, so a bold "Fig. 1"
   set a third of a point off its caption's baseline sorts AFTER every other
   word on the line. The caption stops *beginning* with its label and the scan
   misses it. This alone lost 35 of the 50.
2. **The caption reaches inside the crop.** Figure boxes are snapped outwards
   to whitespace, which swallows the caption's first line: on the Springer
   chapter's page 3 the caption starts at y=531.3 while the crop ends at
   y=533.0. A `gap >= 0` test rejects its own caption.
"""

import pytest

from src.processing.strategies.research_article.figures import (
    _is_panel_sublabel,
    _repair_caption,
    _without_overlap,
)
from src.processing.utils.figure_label_pass import (
    LabelAssignment,
    harvest_caption_block,
    CAPTION_OVERLAP_TOLERANCE,
    MAX_GAP_FRACTION,
    assign_labels_on_page,
    build_text_lines,
    find_label_lines,
    resolve_across_document,
)

PAGE_WIDTH = 432.0
PAGE_HEIGHT = 648.0


def word(text, x0, y0, width=20.0, height=9.0):
    return (text, x0, y0, x0 + width, y0 + height)


def caption_line(label, y, x0=60.0, x1=380.0):
    """A caption line as `find_label_lines` expects to receive it."""
    return (f"{label} Something about the specimen", x0, y, x1, y + 9.0)


class TestBuildingLines:
    def test_a_label_off_its_own_baseline_still_starts_its_line(self):
        """
        DEFECT 1. "Fig." and "1" sit at y=531.3; the rest of the caption at
        531.0. Sorted by (y, x) they land at the END of the line, and the
        caption no longer begins with its label.
        """
        words = [word("Schematic", 100.0, 531.0), word("showing", 140.0, 531.0),
                 word("Fig.", 63.0, 531.3), word("1", 85.0, 531.3)]

        lines = build_text_lines(words, PAGE_WIDTH)

        assert lines[0][0].startswith("Fig. 1")

    def test_two_columns_are_not_spliced_into_one_line(self):
        """
        DEFECT 2 of the line builder. Left column prose running into a right
        column caption destroys the caption's anchoring.
        """
        # A real word space (3 pt in 9 pt text) inside the left column, and a
        # gutter of many text heights before the caption. Until v6.4.0 the
        # same-column gap here was 20 pt, over two text heights, which no
        # typesetter produces: it only passed under the old page-width rule.
        words = [word("continued", 60.0, 300.0), word("prose", 83.0, 300.0),
                 word("Fig.", 260.0, 300.0), word("3", 282.0, 300.0)]

        lines = build_text_lines(words, PAGE_WIDTH)

        assert len(lines) == 2
        assert lines[1][0].startswith("Fig. 3")

    def test_an_empty_page_yields_no_lines(self):
        assert build_text_lines([], PAGE_WIDTH) == []

    # Book p21 under pypdfium2 5.x, coordinates as measured: a Springer side
    # caption 11.4 pt from the body column (under the 1.5-heights gutter
    # rule), its tops 2.6 pt below the body line's ("Fig. 2.3") and 3.08 pt
    # below it ("FACIT ...", just past the 3 pt row tolerance).
    P21 = [word("NC", 200.0, 298.37, width=12.0, height=11.2),
           word("with", 226.6, 298.37, width=20.0, height=11.2),
           word("Fig.", 258.0, 301.0, width=13.6, height=10.5),
           word("2.3", 275.1, 301.0, width=11.6, height=10.5),
           word("FACIT", 292.6, 301.45, width=25.3, height=9.8),
           word("collagens", 321.0, 301.45, width=36.0, height=9.8)]

    def test_a_narrow_gutter_on_another_line_grid_separates_columns(self):
        """The body line must not swallow the side caption's label."""
        lines = build_text_lines(self.P21, PAGE_WIDTH)
        assert lines[0][0] == "NC with"

    def test_a_line_cut_by_the_row_tolerance_is_rejoined(self):
        """The caption's first line comes back whole, label first."""
        lines = build_text_lines(self.P21, PAGE_WIDTH)
        assert [line[0] for line in lines] == ["NC with", "Fig. 2.3 FACIT collagens"]
        assert find_label_lines(lines)[0].printed == "Fig. 2.3"

    def test_a_word_space_between_fonts_is_not_a_column(self):
        """A different font shifts a word's top by under a point; never split."""
        words = [word("Type", 60.0, 300.0), word("XII", 83.0, 300.8),
                 word("collagen", 106.0, 300.0)]
        assert [line[0] for line in build_text_lines(words, PAGE_WIDTH)] == [
            "Type XII collagen"]

    def test_the_next_line_of_a_column_is_never_rejoined(self):
        """12 pt below is another line, however close horizontally."""
        words = [word("first", 60.0, 300.0), word("line", 83.0, 300.0),
                 word("next", 105.0, 312.0)]
        assert len(build_text_lines(words, PAGE_WIDTH)) == 2


def based(text, x0, top, baseline, width=20.0, height=9.0):
    """A word WITH its baseline, as `label_words` gives them."""
    return (text, x0, top, x0 + width, top + height, baseline)


class TestBaselineRows:
    """
    With baselines, a printed line is one row whatever its fonts' ascents.
    Coordinates are the measured ones (pypdfium2 5.x), rounded.
    """

    def test_a_raised_box_on_the_same_baseline_stays_on_its_line(self):
        """Collagen structure p4: "106◦." starts 2.1 pt above its line."""
        words = [based("twist", 350.0, 390.1, 395.69), based("=−", 371.0, 389.7, 395.69),
                 based("106◦.", 387.0, 388.0, 395.69), based("This", 405.0, 390.1, 395.69)]
        assert [line[0] for line in build_text_lines(words, PAGE_WIDTH)] == [
            "twist =− 106◦. This"]

    def test_the_other_columns_row_no_longer_decides(self):
        """Collagen structure p16: a word 2.9 pt higher in the OTHER column
        used to anchor the row, cutting "(A). Hydrophobic" in two."""
        words = [based("Other", 307.0, 548.0, 554.0),
                 based("as", 60.0, 551.1, 557.2), based("in", 83.0, 551.1, 557.2),
                 based("(A).", 106.0, 550.9, 557.2), based("Hydrophobic", 129.0, 551.1, 557.2)]
        lines = [line[0] for line in build_text_lines(words, PAGE_WIDTH)]
        assert "as in (A). Hydrophobic" in lines

    def test_a_superscript_is_folded_back_into_its_line(self):
        """A bare superscript (a citation number) sits on a raised baseline."""
        words = [based("collagen", 60.0, 300.0, 307.0), based("23", 82.0, 297.0, 304.0, width=6.0),
                 based("fibrils", 90.0, 300.0, 307.0), based("assemble", 112.0, 300.0, 307.0)]
        assert [line[0] for line in build_text_lines(words, PAGE_WIDTH)] == [
            "collagen 23 fibrils assemble"]

    def test_the_next_line_is_never_folded_in_as_a_fragment(self):
        """Collagen structure Fig. 5 under 5.x: a symbol's tall box reached
        over the short last line; judged by baseline it stays a line."""
        words = [based("Interstrand", 60.0, 300.0, 307.0, width=45.0),
                 based("H···O", 108.0, 300.0, 307.0, height=16.0),
                 based("shown.", 131.0, 300.0, 307.0),
                 based("them.", 60.0, 309.0, 316.0), based("Images", 85.0, 309.0, 316.0)]
        assert [line[0] for line in build_text_lines(words, PAGE_WIDTH)] == [
            "Interstrand H···O shown.", "them. Images"]


class TestCaptionEndsAtBodyText:
    """`harvest_caption_block` with typography: stop where the body resumes."""

    LINES = [("Figure 1 Collagen fibrils", 60.0, 100.0, 300.0, 108.0),
             ("(A) Tendon in cross section.", 60.0, 109.0, 300.0, 117.0),
             ("Images prepared with PyMol.", 60.0, 118.0, 300.0, 126.0),
             ("The triple helix was first", 60.0, 127.0, 300.0, 135.0)]  # body text

    def _typography(self):
        from src.processing.pdf.base import GlyphStyle, TextStyle
        from src.processing.utils.text_style import PageTypography
        caption = [GlyphStyle(x=100.0, y=y, font="Helvetica", x_height=4.3)
                   for y in (104.0, 113.0, 122.0)]
        body = [GlyphStyle(x=100.0, y=131.0, font="Times-Roman", x_height=4.4)]
        return PageTypography(glyphs=caption + body, body=TextStyle("Times-Roman", 4.4))

    def test_the_caption_stops_before_the_body_text(self):
        label = find_label_lines(self.LINES)[0]
        caption = harvest_caption_block(label, self.LINES, self._typography())
        assert caption.endswith("Images prepared with PyMol.")

    def test_without_typography_the_geometry_alone_decides(self):
        """Same spacing, same margin: the old walk runs on (why this exists)."""
        label = find_label_lines(self.LINES)[0]
        assert harvest_caption_block(label, self.LINES).endswith("was first")

    def test_a_stray_symbol_line_is_stepped_over(self):
        """Collagen structure p2 under 5.x: an orphan "A˚" ended the caption."""
        lines = list(self.LINES[:2]) + [("A˚", 120.0, 117.5, 124.0, 125.0)] + list(self.LINES[2:])
        label = find_label_lines(lines)[0]
        caption = harvest_caption_block(label, lines, self._typography())
        assert caption.endswith("Images prepared with PyMol.")
        assert "A˚" not in caption


class TestTwoColumnCaptions:
    """Nature Communications sets long legends in two columns at the page foot
    (pilot PMC10904388); the second column must be read, a neighbouring
    caption's continuation must not (Collagen structure p19, Silberbush p5)."""

    @staticmethod
    def _typography(lines, legend_font="Legend", label_font="Legend-Bold"):
        from src.processing.pdf.base import GlyphStyle, TextStyle
        from src.processing.utils.text_style import PageTypography
        glyphs = []
        for text, x0, y0, x1, y1 in lines:
            font = label_font if text.startswith(("Fig", "Figure")) else legend_font
            if text.startswith("Body"):
                font = "Body"
            glyphs.append(GlyphStyle(x=x0 + 5.0, y=(y0 + y1) / 2, font=font,
                                     x_height=4.4 if font == "Body" else 3.9))
        return PageTypography(glyphs=glyphs, body=TextStyle("Body", 4.4))

    def test_the_second_column_is_read_after_the_first(self):
        lines = [("Fig. 5 | Associations of viral diversity", 40.0, 639.0, 284.0, 646.0),
                 ("compared. d Volcano plot of", 306.0, 639.5, 550.0, 646.5),
                 ("variables. a Effect sizes of alpha", 40.0, 649.0, 284.0, 656.0),
                 ("highlighted in red.", 306.0, 649.5, 550.0, 656.5)]
        label = find_label_lines(lines)[0]
        caption = harvest_caption_block(label, lines, self._typography(lines))
        assert caption.endswith("alpha compared. d Volcano plot of highlighted in red.")

    def test_another_captions_continuation_beside_it_is_not_read(self):
        lines = [("Figure 13 Structural impact", 306.0, 447.8, 453.0, 455.5),
                 ("Figure 12 Interruptions in the helix", 43.0, 458.9, 211.0, 466.5),
                 ("(A) Variation of the unit twist", 306.0, 463.9, 551.0, 472.0),
                 ("(A) Distribution of interruptions", 43.0, 475.1, 288.0, 482.4)]
        label = [l for l in find_label_lines(lines) if l.printed.startswith("Figure 12")][0]
        caption = harvest_caption_block(label, lines, self._typography(lines))
        assert "Variation" not in caption
        assert caption.endswith("Distribution of interruptions")

    def test_body_text_beside_a_one_column_caption_is_not_read(self):
        lines = [("Fig. 2 | A one-column caption", 40.0, 600.0, 284.0, 607.0),
                 ("Body text of the article runs here", 306.0, 600.5, 550.0, 607.5),
                 ("a The first panel shows", 40.0, 610.0, 284.0, 617.0)]
        label = find_label_lines(lines)[0]
        caption = harvest_caption_block(label, lines, self._typography(lines))
        assert "Body" not in caption

    def test_a_bare_figure_doi_line_ends_the_caption(self):
        """PLOS prints the figure's DOI as the line under its caption."""
        lines = [("Fig 1. BHRF1 induces mito-aggresomes.", 60.0, 100.0, 300.0, 108.0),
                 ("(A) HeLa cells were imaged.", 60.0, 109.0, 300.0, 117.0),
                 ("https://doi.org/10.1371/journal.ppat.1010371.g001", 60.0, 118.0, 300.0, 126.0)]
        label = find_label_lines(lines)[0]
        assert harvest_caption_block(label, lines).endswith("were imaged.")


class TestFindingCaptionLines:
    def test_a_caption_line_is_found(self):
        found = find_label_lines([caption_line("Fig. 2", 400.0)])
        assert [line.key for line in found] == ["figure 2"]

    def test_a_citation_inside_prose_is_not_a_caption(self):
        """
        THE TRAP. Real text from the tendon volume's page 20 — the figure it
        names is printed elsewhere entirely.
        """
        prose = ("packing is affected (Fig. 2.2 ). The rigid COL2 domain",
                 60.0, 400.0, 380.0, 409.0)
        assert find_label_lines([prose]) == []


class TestPanelsOfOneFigure:
    """The point of the whole pass."""

    def test_three_stacked_crops_share_the_caption_beneath_them(self):
        """
        Hydration p16 in miniature: three panels, one caption, and only the
        bottom panel is near enough to find it on its own.
        """
        boxes = [(60.0, 60.0, 380.0, 200.0),
                 (60.0, 210.0, 380.0, 350.0),
                 (60.0, 360.0, 380.0, 500.0)]
        lines = find_label_lines([caption_line("Fig. 11", 505.0)])

        result = assign_labels_on_page(boxes, lines, PAGE_HEIGHT)

        assert [r.key for r in result] == ["figure 11"] * 3
        assert [r.shared_with for r in result] == [3, 3, 3]

    def test_two_figures_on_one_page_stay_separate(self):
        """
        THE GUARD. Without refusing to propagate across a caption line, every
        figure on a page would collapse into the last one.
        """
        boxes = [(60.0, 60.0, 380.0, 200.0), (60.0, 300.0, 380.0, 440.0)]
        lines = find_label_lines([caption_line("Fig. 1", 205.0),
                                  caption_line("Fig. 2", 445.0)])

        result = assign_labels_on_page(boxes, lines, PAGE_HEIGHT)

        assert [r.key for r in result] == ["figure 1", "figure 2"]
        assert [r.shared_with for r in result] == [1, 1]

    def test_a_crop_in_another_column_does_not_join_the_group(self):
        boxes = [(60.0, 60.0, 200.0, 200.0), (240.0, 60.0, 380.0, 200.0)]
        lines = find_label_lines([caption_line("Fig. 5", 205.0, x0=60.0, x1=200.0)])

        result = assign_labels_on_page(boxes, lines, PAGE_HEIGHT)

        assert result[0].key == "figure 5"
        assert result[1].key is None


class TestAdjacency:
    def test_a_caption_reaching_inside_the_crop_is_still_its_caption(self):
        """
        DEFECT 2, directly: the measured −1.7pt overlap from the Springer
        chapter. This is the single condition that lost 35 of 50.
        """
        box = (108.7, 337.4, 355.7, 533.0)
        lines = find_label_lines([caption_line("Fig. 1", 531.3, x0=63.4)])

        assert assign_labels_on_page([box], lines, PAGE_HEIGHT)[0].key == "figure 1"

    def test_a_caption_far_down_the_page_is_adopted_when_nothing_disputes_it(self):
        """
        Adjacency is no longer absolute, and that is deliberate.

        MEASURED on the 9-document corpus: four figures have their caption
        printed well beyond `MAX_GAP_FRACTION`, for reasons that say nothing
        about the caption being wrong -- a panel the detector read as text sits
        between them (`Collagen structure` p5, a 344pt gap), or the caption's
        first line is a narrow bold heading offset from the one panel that was
        cropped (`Collagen structure` p7, zero horizontal overlap).

        `_adopt_orphan_captions` pairs those, and the A/B replay over all 107
        figures showed it adds exactly those four and changes NOTHING else --
        no existing pairing moved. The guards that make that true are the two
        tests below.
        """
        box = (60.0, 60.0, 380.0, 150.0)
        far = PAGE_HEIGHT * MAX_GAP_FRACTION + 200.0
        lines = find_label_lines([caption_line("Fig. 9", far)])

        assert assign_labels_on_page([box], lines, PAGE_HEIGHT)[0].key == "figure 9"

    def test_a_sentence_that_merely_opens_with_a_label_is_never_adopted(self):
        """
        THE TRAP, in the form the relaxed rule exposes it.

        `JRuberti` page 9 carries "Figure 6 and Video S5 show a clear example
        of flow aggregation" -- body prose that begins with a figure label at
        the start of its line, so anchoring alone does not reject it, and
        distance no longer does either. What rejects it is that the previous
        line of its own paragraph ends directly above it at the same margin.
        """
        box = (60.0, 60.0, 380.0, 150.0)
        far = PAGE_HEIGHT * MAX_GAP_FRACTION + 200.0
        previous_line = ("continuing a paragraph from further up the column",
                         60.0, far - 11.0, 380.0, far - 2.0)
        lines = find_label_lines([previous_line, caption_line("Fig. 9", far)])

        assert assign_labels_on_page([box], lines, PAGE_HEIGHT)[0].key is None

    def test_a_distant_caption_is_declined_when_two_crops_could_claim_it(self):
        """
        The ambiguity guard. Two unlabelled crops above one orphan caption
        could be panels of it or two separate figures, and nothing on the page
        says which -- so both stay unlabelled, per the module's standing rule
        that an unplaceable figure is never guessed at.
        """
        boxes = [(60.0, 60.0, 380.0, 150.0), (60.0, 170.0, 380.0, 260.0)]
        # Far enough that neither crop reaches the caption on its own -- so
        # this test exercises the fallback, not the geometric pass.
        far = PAGE_HEIGHT * MAX_GAP_FRACTION + 400.0
        lines = find_label_lines([caption_line("Fig. 9", far)])

        result = assign_labels_on_page(boxes, lines, PAGE_HEIGHT)

        assert [assignment.key for assignment in result] == [None, None]

    def test_a_caption_a_crop_already_claimed_is_not_handed_to_another(self):
        """
        The rule can only ADD pairings. A caption paired geometrically stays
        with the crop that paired with it, and a distant crop does not take it.
        """
        near = (60.0, 60.0, 380.0, 150.0)
        distant = (60.0, 400.0, 380.0, 500.0)
        lines = find_label_lines([caption_line("Fig. 9", 158.0)])

        result = assign_labels_on_page([near, distant], lines, PAGE_HEIGHT)

        assert result[0].key == "figure 9"
        assert result[1].key is None

    def test_a_caption_above_the_figure_is_accepted(self):
        """Some journals print the caption above the figure."""
        box = (60.0, 200.0, 380.0, 400.0)
        lines = find_label_lines([caption_line("Fig. 4", 185.0)])

        assert assign_labels_on_page([box], lines, PAGE_HEIGHT)[0].key == "figure 4"

    def test_below_wins_when_captions_sit_on_both_sides(self):
        box = (60.0, 200.0, 380.0, 400.0)
        lines = find_label_lines([caption_line("Fig. 4", 185.0),
                                  caption_line("Fig. 5", 405.0)])

        assert assign_labels_on_page([box], lines, PAGE_HEIGHT)[0].key == "figure 5"

    def test_the_tolerance_is_a_fraction_of_the_page_not_a_pixel_count(self):
        assert 0 < CAPTION_OVERLAP_TOLERANCE < MAX_GAP_FRACTION


class TestOneCaptionPerFigureAcrossTheDocument:
    def test_the_closer_claim_keeps_the_label(self):
        """
        A printed figure is captioned once but cited often. Two pages claiming
        "Figure 3" means at most one is the caption.
        """
        near = assign_labels_on_page([(60.0, 60.0, 380.0, 200.0)],
                                     find_label_lines([caption_line("Fig. 3", 205.0)]),
                                     PAGE_HEIGHT)
        far = assign_labels_on_page([(60.0, 60.0, 380.0, 200.0)],
                                    find_label_lines([caption_line("Fig. 3", 260.0)]),
                                    PAGE_HEIGHT)

        resolved = resolve_across_document({4: near, 9: far})

        assert resolved[4][0].key == "figure 3"
        assert resolved[9][0].key is None

    def test_distinct_labels_are_both_kept(self):
        one = assign_labels_on_page([(60.0, 60.0, 380.0, 200.0)],
                                    find_label_lines([caption_line("Fig. 1", 205.0)]),
                                    PAGE_HEIGHT)
        two = assign_labels_on_page([(60.0, 60.0, 380.0, 200.0)],
                                    find_label_lines([caption_line("Fig. 2", 205.0)]),
                                    PAGE_HEIGHT)

        resolved = resolve_across_document({2: one, 3: two})

        assert resolved[2][0].key == "figure 1"
        assert resolved[3][0].key == "figure 2"


class TestItDeclinesRatherThanGuesses:
    """
    Inherited from `chapter_boundaries`: a figure with no attribution is
    recoverable, a misattributed one is not.
    """

    def test_a_page_with_no_caption_lines_labels_nothing(self):
        result = assign_labels_on_page([(60.0, 60.0, 380.0, 200.0)], [], PAGE_HEIGHT)
        assert result[0].printed is None and result[0].shared_with == 0

    def test_a_page_with_no_figures_is_not_an_error(self):
        assert assign_labels_on_page([], find_label_lines(
            [caption_line("Fig. 1", 100.0)]), PAGE_HEIGHT) == []

    def test_the_recovered_caption_text_comes_back_with_the_label(self):
        """
        Where the detector gave no caption box, the caption's own line is
        still the best text we have for the RAG.
        """
        lines = find_label_lines([caption_line("Fig. 7", 205.0)])
        result = assign_labels_on_page([(60.0, 60.0, 380.0, 200.0)], lines, PAGE_HEIGHT)

        assert result[0].caption.startswith("Fig. 7 Something about")


class TestItIsActuallyWiredIn:
    """
    THIS CLASS EXISTS BECAUSE THE SUITE ONCE PASSED WITHOUT IT. The pass was
    written, unit-tested and called — but its import was missing from
    `figures.py`, and the caller's `except Exception` guard (there so a page
    with no text layer cannot cost us its figures) swallowed the NameError and
    logged it as a missing text layer. Every unit test still passed. Only
    running it against a real page found it.

    So: exercise the real entry point, not just the parts.
    """

    def _page_with_a_caption(self):
        from src.processing.pdf.base import Rect, Word
        from .fakes import FakePage

        words = [Word(text=text, rect=Rect(x0, 205.0, x0 + 26.0, 214.0))
                 for text, x0 in (("Fig.", 60.0), ("11", 90.0),
                                  ("Plant", 115.0), ("materials", 150.0))]
        return FakePage(words=words, size=(432.0, 648.0))

    def test_attach_printed_labels_labels_a_real_asset(self):
        from src.processing.strategies.research_article.figures import (
            attach_printed_labels,
        )
        from src.processing.utils.figure_extractor import FigureAsset

        class Rendered:                      # only `.width` is read, for scale
            width = 1800

        assets = [FigureAsset(figure_id="p16_fig1", image_path="a.png", caption="",
                              page=16, bbox=(250, 250, 1583, 833)),
                  FigureAsset(figure_id="p16_fig2", image_path="b.png", caption="",
                              page=16, bbox=(250, 850, 1583, 1000))]

        result = attach_printed_labels(self._page_with_a_caption(), Rendered(), assets)

        assert [a.label_key for a in result] == ["figure 11", "figure 11"]
        assert [a.panel_group_size for a in result] == [2, 2]

    def test_a_caption_the_detector_already_found_is_not_overwritten(self):
        """The paired caption box is the stronger evidence of the two."""
        from src.processing.strategies.research_article.figures import (
            attach_printed_labels,
        )
        from src.processing.utils.figure_extractor import FigureAsset

        class Rendered:
            width = 1800

        asset = FigureAsset(figure_id="p16_fig1", image_path="a.png",
                            caption="Fig. 3 The real caption", page=16,
                            bbox=(250, 250, 1583, 833),
                            printed_label="Fig. 3", label_key="figure 3")

        result = attach_printed_labels(self._page_with_a_caption(), Rendered(), [asset])

        assert result[0].label_key == "figure 3"
        assert result[0].caption == "Fig. 3 The real caption"


class TestAPanelSubLabelIsNotACaption:
    """
    A caption box holding "d" is worse than an empty one.

    MEASURED on the 9-document corpus: the three crops of the ice-plant figure
    on `Hydration-induced` page 7 are labelled "Fig. 4" correctly and captioned
    "d", "d" and "f" -- the little panel letters printed inside the images.
    The label pass had recovered the real caption for all three and discarded
    it, because the caption field was not EMPTY. Downstream nothing can tell a
    one-letter caption from a real one, so it reads as a captioned figure.
    """

    def test_a_panel_letter_is_recognised_as_junk(self):
        assert _is_panel_sublabel("d")
        assert _is_panel_sublabel("f")
        assert _is_panel_sublabel("( A )")

    def test_a_real_caption_is_never_mistaken_for_one(self):
        # The shortest genuine caption in the corpus, at 16 characters.
        assert not _is_panel_sublabel("Helical twisting")
        assert not _is_panel_sublabel("Fig. 4 | Hygroscopic structure")

    def test_a_short_string_that_is_itself_a_label_is_kept(self):
        """
        `parse_printed_label` signals "no label" as (None, None), never as
        None, so testing the returned PAIR is silently always-false and the
        predicate matches nothing. This pins the unpacking.
        """
        assert not _is_panel_sublabel("Fig. 4")

    def test_an_empty_caption_is_not_a_sub_label(self):
        """Empty is handled by its own branch; this must not claim it too."""
        assert not _is_panel_sublabel("")
        assert not _is_panel_sublabel("   ")


class _StoredFigure:
    """Just enough of a `FigureAsset` for the caption repair rule to judge."""

    def __init__(self, caption, printed_label=None, label_key=None):
        self.caption = caption
        self.printed_label = printed_label
        self.label_key = label_key


def assignment(caption, opening, printed="Fig. 2", key="figure 2"):
    return LabelAssignment(printed=printed, key=key, caption=caption,
                           shared_with=1, distance=0.0, opening=opening)


class TestHarvestingAWholeCaption:
    """
    The caption box very often starts PART WAY DOWN the caption, so the
    sentence naming the figure is missing and the caption cannot be found by
    its own subject. MEASURED: 26 of the corpus's 107 captions.
    """

    def test_the_caption_runs_on_past_its_first_line(self):
        lines = [("Fig. 2 Hierarchical structure of tendon.", 60.0, 300.0, 380.0, 309.0),
                 ("Tendons are composed primarily of collagen", 60.0, 310.0, 380.0, 319.0),
                 ("fibrils bundled into fascicles.", 60.0, 320.0, 380.0, 329.0)]
        line = find_label_lines(lines)[0]

        harvested = harvest_caption_block(line, lines)

        assert harvested.startswith("Fig. 2 Hierarchical structure of tendon.")
        assert "fascicles" in harvested

    def test_a_paragraph_break_ends_the_caption(self):
        """The body text after a caption must not be swallowed into it."""
        lines = [("Fig. 2 Hierarchical structure of tendon.", 60.0, 300.0, 380.0, 309.0),
                 ("Tendons are composed primarily of collagen", 60.0, 310.0, 380.0, 319.0),
                 ("This is the next paragraph entirely.", 60.0, 380.0, 380.0, 389.0)]
        line = find_label_lines(lines)[0]

        assert "next paragraph" not in harvest_caption_block(line, lines)

    def test_the_next_figures_caption_ends_this_one(self):
        lines = [("Fig. 2 Hierarchical structure of tendon.", 60.0, 300.0, 380.0, 309.0),
                 ("Tendons are composed primarily of collagen", 60.0, 310.0, 380.0, 319.0),
                 ("Fig. 3 Something else entirely.", 60.0, 320.0, 380.0, 329.0)]
        line = find_label_lines(lines)[0]

        assert "Something else" not in harvest_caption_block(line, lines)

    def test_another_column_does_not_join_the_caption(self):
        lines = [("Fig. 2 Hierarchical structure of tendon.", 60.0, 300.0, 200.0, 309.0),
                 ("prose from the neighbouring column", 240.0, 310.0, 380.0, 319.0)]
        line = find_label_lines(lines)[0]

        assert "neighbouring" not in harvest_caption_block(line, lines)


class TestRepairingACaptionThatLostItsOpening:
    """
    MEASURED over all 107 figures, and this is why the repair is split in two:
    replacing an existing caption with the harvested block traded 1343
    characters for 44 on the `Collagen structure` review's page 19, and
    interleaved the columns of the tendon volume's captions. Putting back only
    the missing opening changed 31 captions and shortened none.
    """

    def test_a_headless_caption_gets_its_opening_back(self):
        stored = _StoredFigure("respective overlaps and gaps in the fibril.")
        repaired = _repair_caption(stored, assignment(
            caption="Fig. 2 Fibril-forming collagens. respective overlaps and gaps",
            opening="Fig. 2 Fibril-forming collagens."))

        assert repaired == ("Fig. 2 Fibril-forming collagens. "
                            "respective overlaps and gaps in the fibril.")

    def test_a_caption_that_already_starts_at_its_label_is_left_alone(self):
        stored = _StoredFigure("Fig. 2 Fibril-forming collagens and their assembly.")

        assert _repair_caption(stored, assignment(
            caption="Fig. 2 Fibril-forming collagens and their assembly.",
            opening="Fig. 2 Fibril-forming collagens and their assembly.")) is None

    def test_an_empty_caption_takes_the_whole_block(self):
        """Nothing to protect, so the jumbling risk is worth the caption."""
        stored = _StoredFigure("")
        repaired = _repair_caption(stored, assignment(
            caption="Fig. 2 The whole caption, every line of it.",
            opening="Fig. 2 The whole caption,"))

        assert repaired == "Fig. 2 The whole caption, every line of it."

    def test_a_caption_for_a_different_figure_is_never_pasted_on(self):
        stored = _StoredFigure("some caption text", "Fig. 9", "figure 9")

        assert _repair_caption(stored, assignment(
            caption="Fig. 2 A different figure.",
            opening="Fig. 2 A different figure.")) is None


class TestCompletingATruncatedCaption:
    """
    Session 30. The detector's caption box sometimes holds only the TITLE
    (Collagen structure Fig. 7 and 10, JRuberti Fig. 2: 84, 94 and 57 characters
    against legends of 1.5 to 1.8k), while the text layer holds the legend.
    Whole-caption replacement is only safe when the stored text is the START of
    the harvested text: the book's two-column captions interleave body prose
    and must keep the detector's clean box.
    """

    LEGEND = ("Figure 7 Preferences for ring conformation in proline. "
              "(A) The endo conformation places the R1 group equatorial. "
              "(B) Electron-withdrawing groups stabilize the exo conformation.")

    def test_a_title_only_caption_is_completed(self):
        stored = _StoredFigure("Figure 7 Preferences for ring conformation in proline.",
                               "Fig. 2", "figure 2")
        repaired = _repair_caption(stored, assignment(
            caption=self.LEGEND, opening="Figure 7 Preferences for ring conformation"))

        assert repaired == self.LEGEND

    def test_line_breaks_and_case_do_not_hide_a_truncation(self):
        stored = _StoredFigure("figure 7  preferences for\nring conformation in proline.",
                               "Fig. 2", "figure 2")

        assert _repair_caption(stored, assignment(
            caption=self.LEGEND, opening="Figure 7 Preferences")) == self.LEGEND

    def test_a_stray_fragment_at_the_end_still_counts_as_truncated(self):
        """Hydration Fig. 11: the box ended with 'Helical twisting', 15 off."""
        stored = _StoredFigure(
            "Figure 7 Preferences for ring conformation in proline. Helical twisting",
            "Fig. 2", "figure 2")

        assert _repair_caption(stored, assignment(
            caption=self.LEGEND, opening="Figure 7 Preferences")) == self.LEGEND

    def test_a_caption_that_diverges_early_is_never_replaced(self):
        """The book: harvest = caption interleaved with the neighbouring column."""
        clean = ("Fig. 15.1 Fetlock of healthy and DSLD-affected horses. (a) Healthy "
                 "horse with the joint normally positioned over the ground surface "
                 "and the suspensory ligament intact.")
        interleaved = clean[:60] + " Linde-Sipman et al. 1985; Ploeg et al. 2013) " + clean[60:] * 2
        stored = _StoredFigure(clean, "Fig. 2", "figure 2")

        assert _repair_caption(stored, assignment(
            caption=interleaved, opening="Fig. 15.1 Fetlock")) is None

    def test_a_shorter_harvest_never_replaces_the_stored_caption(self):
        stored = _StoredFigure(self.LEGEND, "Fig. 2", "figure 2")

        assert _repair_caption(stored, assignment(
            caption="Figure 7 Preferences for ring conformation in proline.",
            opening="Figure 7 Preferences")) is None


class TestTheTwoColumnSeam:
    """
    A caption set in two columns is read straight across the gutter, so its
    first line arrives with the second column's words mixed in — which is
    exactly where the stored caption begins. Pasting them together unrepaired
    says everything twice.
    """

    def test_the_repeated_tail_is_cut_from_the_opening(self):
        opening = "Fig. 2.2 Fibril-forming collagens. respective overlaps and gaps"
        stored = "respective overlaps and gaps in the fibril. (b) Collagen fibers"

        assert _without_overlap(opening, stored) == "Fig. 2.2 Fibril-forming collagens."

    def test_an_opening_that_does_not_repeat_is_kept_whole(self):
        opening = "Fig. 13 Structural impact of interruptions"
        stored = "( A ) Variation of the unit twist for the collagen model peptide"

        assert _without_overlap(opening, stored) == opening

    def test_a_coincidental_few_characters_are_not_treated_as_a_repeat(self):
        opening = "Fig. 4 Something about a"
        stored = "a much longer caption body"

        assert _without_overlap(opening, stored) == opening


class TestOrphanCaptionBandsFollowColumns:
    """
    Session 30, eval/orphan_captions/GATE.md (R3b). `Collagen structure` p19:
    "Figure 13" (right column) starts 11 pt above "Figure 12" (left column).
    The last-resort pairing cut bands by caption order across the whole page,
    so Figure 12's band became the 3 pt between the two captions and its own
    crop, far above it, was never offered it.
    """

    def test_a_caption_in_the_other_column_does_not_shorten_the_band(self):
        from src.processing.utils.figure_label_pass import LabelLine
        figure_13 = LabelLine("Figure 13", "figure 13", "Figure 13 Structural impact",
                              (305.6, 447.8, 452.9, 455.5))
        figure_12 = LabelLine("Figure 12", "figure 12", "Figure 12 Interruptions",
                              (42.6, 458.9, 211.1, 466.5))
        left_panel_a = (70.8, 71.8, 279.1, 235.9)       # far above its caption
        right_figure = (312.5, 71.5, 557.5, 442.8)      # directly above Figure 13

        got = assign_labels_on_page([left_panel_a, right_figure],
                                    [figure_13, figure_12], page_height=792.0)

        assert [a.key for a in got] == ["figure 12", "figure 13"]

    def test_a_caption_in_the_same_column_still_closes_the_band(self):
        """Two stacked figures in one column: the upper crop is the upper caption's."""
        from src.processing.utils.figure_label_pass import LabelLine
        upper = LabelLine("Figure 1", "figure 1", "Figure 1 A", (42.0, 300.0, 250.0, 308.0))
        lower = LabelLine("Figure 2", "figure 2", "Figure 2 B", (42.0, 700.0, 250.0, 708.0))
        far_upper_crop = (60.0, 60.0, 260.0, 120.0)      # above Figure 1, out of reach

        got = assign_labels_on_page([far_upper_crop], [upper, lower], page_height=792.0)

        assert got[0].key == "figure 1"
