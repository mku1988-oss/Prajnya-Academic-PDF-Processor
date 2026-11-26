#!/usr/bin/env python
"""
Tests for T38b, figure/table pages read from the text layer
(`utils/text_first_page.py` and `BodyPipeline(body_read_mode=...)`).

For biology researchers: these pin the four guards that keep a page on the
vision model, the two text-removal steps, and above all that the default
setting (`vlm_full`) runs none of it, so shipping the code changes nothing
until the setting is switched.
"""
from __future__ import annotations

import logging

import pytest

from src.processing.pdf.base import PrintedChar, Rect
from src.processing.strategies.research_article import body_pipeline as bp_module
from src.processing.utils import text_first_page as tfp
from src.processing.utils.page_classifier import PageType
from tests.fakes import FakePage

LOG = logging.getLogger("test_text_first_page")


def char(text, x, *, y=100.0, height=10.0, baseline=None, font="Times-Roman", space=False):
    """One printed character, 5 pt wide, its box `height` tall above y."""
    return PrintedChar(text=text, rect=Rect(x, y - height, x + 5, y),
                       baseline=y if baseline is None else baseline, font=font,
                       space_before=space)


def word(text, x, **kw):
    first, *rest = text
    return [char(first, x, space=kw.pop("space", False), **kw)] + [
        char(c, x + 5 * (i + 1), **kw) for i, c in enumerate(rest)]


# ---------------------------------------------------------------------------
# the guards
# ---------------------------------------------------------------------------

class TestGuards:
    def test_stix_text_fonts_are_not_maths_but_stixmath_is(self):
        # Amendment 2: PNAS and Sci Adv set prose in STIX-Regular / STIX-Italic.
        assert not tfp.is_maths(char("b", 0, font="STIX-Regular"))
        assert not tfp.is_maths(char("b", 0, font="STIX-Italic"))
        assert tfp.is_maths(char("x", 0, font="STIXMath-Italic"))
        assert tfp.is_maths(char("x", 0, font="CMMI10"))

    def test_symbol_must_start_a_word(self):
        assert tfp.is_maths(char("a", 0, font="Symbol"))
        assert not tfp.is_maths(char("a", 0, font="SegoeUISymbol"))

    def test_a_maths_block_character_counts_whatever_the_font(self):
        assert tfp.is_maths(char("∑", 0))        # N-ARY SUMMATION
        assert not tfp.is_maths(char("μ", 0))     # mu, as in "5 um", is prose

    def test_broken_characters(self):
        assert tfp.is_broken("�") and tfp.is_broken("") and tfp.is_broken("\x07")
        assert not tfp.is_broken("\x02")               # the line-end hyphen
        assert not tfp.is_broken("a") and not tfp.is_broken("\n")

    def test_a_glued_superscript_is_counted(self):
        # "tRNA" then a raised, smaller "Trp" with no space before it.
        chars = word("tRNA", 0) + word("Trp", 20, height=6.0, baseline=96.0)
        assert tfp.glued_script_letters(chars) == 1    # the first letter of the run

    def test_a_raised_citation_number_is_not_a_script_letter(self):
        chars = word("cells", 0) + [char("2", 25, height=6.0, baseline=96.0)]
        assert tfp.glued_script_letters(chars) == 0

    def test_a_small_word_after_a_space_is_not_glued(self):
        chars = word("cells", 0) + word("Trp", 30, height=6.0, baseline=96.0, space=True)
        assert tfp.glued_script_letters(chars) == 0

    def test_route_order_and_the_text_layer_verdict(self, monkeypatch):
        monkeypatch.setattr(tfp, "is_scanned_page", lambda page: False)
        prose = word("Collagen", 0) + word("fibrils", 50, space=True)
        page = FakePage(text="Collagen fibrils", characters=prose)
        assert tfp.route_verdict(page)["route"] == tfp.TEXT_LAYER_ROUTE

        maths = prose + [char("x", 100 + 5 * i, font="CMMI10") for i in range(5)]
        page = FakePage(text="Collagen fibrils xxxxx", characters=maths)
        assert tfp.route_verdict(page)["route"] == "vlm:G2_maths"

        scripts = word("tRNA", 0) + word("Trp", 20, height=6.0, baseline=96.0)
        page = FakePage(text="tRNATrp", characters=scripts)
        assert tfp.route_verdict(page)["route"] == "vlm:G4_scripts"

    def test_no_text_layer_keeps_the_vision_read(self):
        assert tfp.route_verdict(FakePage(text=""))["route"] == "vlm:G1_no_text_layer"


# ---------------------------------------------------------------------------
# the text
# ---------------------------------------------------------------------------

class TestFilteredText:
    def _page(self):
        # Line 1 prose at y=100; line 2 a figure's axis label at y=300.
        chars = word("Body", 0) + word("text", 30, space=True) + word("Axis", 0, y=300.0)
        return FakePage(text="Body text\r\nAxis", characters=chars)

    def test_characters_inside_a_dropped_box_leave(self):
        page = self._page()
        everything = [Rect(0, 0, 600, 800)]
        text, dropped = tfp.filtered_text(page, [Rect(0, 280, 100, 310)], everything)
        assert text == "Body text" and dropped == 4     # emptied line removed, \r stripped

    def test_characters_no_box_covers_leave_too(self):
        page = self._page()
        text, dropped = tfp.filtered_text(page, [], [Rect(0, 80, 100, 110)])
        assert text == "Body text" and dropped == 4

    def test_the_line_end_hyphen_still_pairs_with_its_character(self):
        chars = word("degrada-tion", 0)
        page = FakePage(text="degrada\x02tion", characters=chars)
        text, dropped = tfp.filtered_text(page, [], [Rect(0, 0, 600, 800)])
        assert text == "degrada\x02tion" and dropped == 0


class TestWithoutLegends:
    LEGEND = "FIGURE 2 The fluorescence images of the aptasensor at each concentration."

    def test_a_legend_without_its_label_is_still_found(self):
        # The caption box held only "FIGURE 2"; the legend's words stayed.
        text = "Prose before.\nThe fluorescence images of the aptasensor at each\nconcentration.\nProse after."
        out, removed = tfp.without_legends(text, [{"caption": self.LEGEND}])
        assert removed == 1 and "fluorescence" not in out
        assert "Prose before." in out and "Prose after." in out

    def test_a_legend_with_its_label_goes_whole(self):
        text = "Prose.\nFIGURE 2 The fluorescence images of the aptasensor at each concentration."
        out, removed = tfp.without_legends(text, [{"caption": self.LEGEND}])
        assert removed == 1 and "FIGURE 2" not in out

    def test_a_legend_that_is_not_there_changes_nothing(self):
        text = "Only prose here."
        assert tfp.without_legends(text, [{"caption": self.LEGEND}]) == (text, 0)

    def test_a_short_legend_is_not_looked_for(self):
        text = "As in Figure 2 shows."
        assert tfp.without_legends(text, [{"caption": "Figure 2 shows."}]) == (text, 0)


# ---------------------------------------------------------------------------
# routing inside the body pipeline
# ---------------------------------------------------------------------------

class _Stop(Exception):
    """Raised by a fake where the test has seen what it needs."""


class _SimpleVisualClassifier:
    def classify(self, page, page_idx, text_stats=None):
        return PageType.SIMPLE_VISUAL, {"visual_regions": [object()]}


class _Cache:
    def __init__(self, page):
        self._page = page

    def page(self, idx):
        return self._page


class _StopProvider:
    def process_page(self, image):
        raise _Stop("extraction model called")


def _pipeline(mode, tmp_path):
    pipeline = bp_module.BodyPipeline(
        extraction_provider=_StopProvider(), extraction_model="m", layout_detector=None,
        resident_models=["m"], page_classifier=_SimpleVisualClassifier(), router=None,
        extraction_dpi=150, coverage_threshold=0.9, table_crop_prompt=None,
        caption_read_prompt=None, figures_base_dir=tmp_path, figures_path_root=tmp_path,
        logger=LOG, body_read_mode=mode)
    pipeline._detections_the_filter_hid = lambda cache, idx, analysis: None
    return pipeline


class TestBodyReadMode:
    def test_the_default_is_vlm_full_and_an_unknown_mode_is_refused(self, tmp_path):
        assert _pipeline("vlm_full", tmp_path).body_read_mode == "vlm_full"
        with pytest.raises(ValueError):
            _pipeline("text_only_please", tmp_path)

    def test_vlm_full_never_runs_the_text_first_guards(self, tmp_path, monkeypatch):
        def must_not_run(page):
            raise AssertionError("text-first guards ran under vlm_full")
        monkeypatch.setattr(bp_module, "route_verdict", must_not_run)
        pipeline = _pipeline("vlm_full", tmp_path)
        pipeline.extraction_image = None
        cache = _Cache(FakePage(text="x"))
        cache.extraction_image = lambda idx, dpi: None
        with pytest.raises(_Stop):                      # went straight to the model read
            pipeline.read_body_page(cache, 0)

    def test_text_first_routes_a_passing_page_away_from_the_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bp_module, "route_verdict",
                            lambda page: {"route": tfp.TEXT_LAYER_ROUTE})
        pipeline = _pipeline("text_first", tmp_path)
        pipeline._process_visual_page_with_text_layer = (
            lambda *args: "text-layer page")
        assert pipeline.read_body_page(_Cache(FakePage(text="x")), 0) == "text-layer page"

    def test_text_first_keeps_the_model_read_when_a_guard_fires(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bp_module, "route_verdict",
                            lambda page: {"route": "vlm:G2_maths"})
        pipeline = _pipeline("text_first", tmp_path)
        cache = _Cache(FakePage(text="x"))
        cache.extraction_image = lambda idx, dpi: None
        with pytest.raises(_Stop):
            pipeline.read_body_page(cache, 0)

    def test_the_live_config_ships_vlm_full(self):
        import yaml
        from pathlib import Path
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" /
                                 "research_article_strategy.yaml").read_text(encoding="utf-8"))
        assert config["vlm_provider"]["body_pages"]["read_mode"] == "vlm_full"


class TestWithoutLegendFragments:
    """T38c/T38d: the pieces of a legend that a caption box cut through."""

    LEGEND = ("Figure 1. Examples of collagen X-ray diffraction. (A) Fibre diffraction "
              "photograph of dry adult rat tail tendon. The vertical axis corresponds "
              "to the main direction of the fibres.")

    def test_the_panel_descriptions_left_below_the_box_are_deleted(self):
        # The box took "Figure 1. Examples of collagen X-ray diffraction."
        text = ("Body prose before the figure.\n(A) Fibre diffraction photograph of dry adult "
                "rat tail tendon. The vertical axis corresponds\nto the main direction of "
                "the fibres.\nBody prose after.")
        out, removed = tfp.without_legend_fragments(text, [{"caption": self.LEGEND}])
        assert "Fibre diffraction" not in out and "vertical axis" not in out
        assert "Body prose before the figure." in out and "Body prose after." in out
        assert len(removed) == 1 and removed[0]["words"] >= 8

    def test_a_line_end_hyphen_does_not_break_the_run(self):
        legend = "Figure 2. Collagen fibrils formed by flow-induced crystallization in the cell culture well."
        text = "Prose.\nCollagen fibrils formed by flowinduced crystallization in the cell culture well.\nMore."
        out, removed = tfp.without_legend_fragments(text, [{"caption": legend}])
        assert "crystallization" not in out and len(removed) == 1

    def test_a_short_share_of_a_long_legend_is_kept(self):
        # T38d (PMC9674220 p8): the stored legend ends with a prose line the
        # caption reader swallowed; that line is 10 of the legend's words, a
        # small share of it, so the body keeps it.
        legend = self.LEGEND + (" Further panels show the meridional and equatorial "
                                "reflections of hydrated tendon at higher angle. "
                                "and the two scenarios are explored below in turn")
        text = "Prose.\nand the two scenarios are explored below in turn with care.\nMore."
        out, removed = tfp.without_legend_fragments(text, [{"caption": legend}])
        assert out == text and removed == []

    def test_the_record_carries_the_share(self):
        text = ("(A) Fibre diffraction photograph of dry adult rat tail tendon. "
                "The vertical axis corresponds to the main direction of the fibres.")
        _, removed = tfp.without_legend_fragments(text, [{"caption": self.LEGEND}])
        assert removed and removed[0]["share"] >= tfp.MIN_FRAGMENT_SHARE

    def test_prose_sharing_fewer_than_eight_words_is_kept(self):
        # Prose that quotes 7 legend words in a row stays.
        text = "As shown, its vertical axis corresponds to the main direction here."
        out, removed = tfp.without_legend_fragments(text, [{"caption": self.LEGEND}])
        assert out == text and removed == []
