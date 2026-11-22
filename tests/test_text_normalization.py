"""
Tests for control-character repair in harvested PDF text.

The cases below are not invented: every one is taken from the nine-document
sample corpus in `run_outputs/`, where 4,635 control characters were measured
across 485 pages. The proportions in `text_normalization`'s docstring come from
the same measurement.
"""
import pytest

from src.processing.utils.text_normalization import (
    count_control_damage,
    normalize_harvested_text,
)


class TestSplitWordRepair:
    """96.7% of the damage: a word split by a line-break hyphen."""

    @pytest.mark.parametrize("damaged, expected", [
        ("measure\x02ments", "measurements"),
        ("cross\x02linking", "crosslinking"),
        ("biomo\x02lecular", "biomolecular"),
        ("at\x02tempt", "attempt"),
        ("ly\x02sozyme", "lysozyme"),
        ("com\x02pares", "compares"),
    ])
    def test_real_corpus_words_are_rejoined(self, damaged, expected):
        assert normalize_harvested_text(damaged) == expected

    @pytest.mark.parametrize("damaged, expected", [
        ("the regres\x02 sion fit", "the regression fit"),
        ("at the na\x02 tive hydration", "at the native hydration"),
        ("from native to com\x02 pletely dry", "from native to completely dry"),
        ("changes in wa\x02 ter organization", "changes in water organization"),
    ])
    def test_split_word_with_a_stray_space(self, damaged, expected):
        """0.7%: the harvest left a space where the line break had been."""
        assert normalize_harvested_text(damaged) == expected

    def test_the_repaired_word_is_findable(self):
        """The whole point: the token must match a plain search."""
        page = "difference in collagen cross\x02linking due to stress"
        assert "crosslinking" not in page
        assert "crosslinking" in normalize_harvested_text(page)


class TestRealHyphens:
    """2.5%: the break is a genuine hyphen, not a split word.

    Told apart by the capital letter after it. Getting this wrong would turn
    "Ehlers-Danlos" into "EhlersDanlos", which matches nothing a reader would
    type, and would corrupt the hyphenated author surnames that citations rest
    on.
    """

    @pytest.mark.parametrize("damaged, expected", [
        ("Ehlers\x02Danlos syndrome", "Ehlers-Danlos syndrome"),
        ("Chiquet\x02Ehrismann et al.", "Chiquet-Ehrismann et al."),
        ("Loeys\x02Dietz", "Loeys-Dietz"),
        ("Boot\x02Handford", "Boot-Handford"),
        ("Bruckner\x02Tuderman", "Bruckner-Tuderman"),
        ("tenascin\x02C", "tenascin-C"),
        ("trans\x02Golgi network", "trans-Golgi network"),
        ("Pro\x02Gly", "Pro-Gly"),
        ("Sigma\x02Aldrich", "Sigma-Aldrich"),
    ])
    def test_real_hyphens_are_restored_not_closed(self, damaged, expected):
        assert normalize_harvested_text(damaged) == expected

    def test_an_author_surname_stays_citable(self):
        cleaned = normalize_harvested_text("as shown by Chiquet\x02Ehrismann")
        assert "Chiquet-Ehrismann" in cleaned
        assert "ChiquetEhrismann" not in cleaned

    def test_lowercase_after_the_break_still_joins(self):
        """The capital rule must not disturb the 96.7% majority case."""
        assert normalize_harvested_text("cross\x02linking") == "crosslinking"


class TestLoneControlCharacters:
    """1.4%: a symbol the font encoding lost. Not recoverable, not guessed."""

    @pytest.mark.parametrize("damaged", [
        "149 \x03 1 ms",          # was "149 +/- 1 ms"
        "0.37 s \x04 1",           # was an exponent
        "18 Da/(3 \x01 mean",      # was a multiplication sign
    ])
    def test_lone_codes_become_spaces_and_leave_no_damage(self, damaged):
        cleaned = normalize_harvested_text(damaged)
        assert count_control_damage(cleaned) == 0

    def test_numbers_either_side_stay_separate(self):
        """Deleting rather than spacing would fuse "149" and "1" into "1491"."""
        assert normalize_harvested_text("149 \x03 1 ms") == "149 1 ms"

    def test_no_symbol_is_invented(self):
        """A guessed "+/-" would put a number in the record the paper never printed."""
        cleaned = normalize_harvested_text("1.62 \x03 0.16 SD g/g")
        for invented in ("+/-", "±", "-", "*", "^"):
            assert invented not in cleaned


class TestConservatism:
    """Everything the repair must leave alone."""

    def test_clean_text_is_untouched(self):
        clean = "The collagen superfamily can be divided into nine families."
        assert normalize_harvested_text(clean) == clean

    def test_line_structure_survives(self):
        text = "first line\nsecond line\r\nthird line"
        assert normalize_harvested_text(text) == text

    def test_never_joins_across_a_line_break(self):
        """Rejoining whole lines would change structure other passes read."""
        result = normalize_harvested_text("regres\x02\nsion")
        assert "\n" in result

    def test_tabs_and_newlines_are_not_damage(self):
        assert count_control_damage("a\tb\nc\r\nd") == 0

    def test_legitimate_unicode_is_left_alone(self):
        """Mojibake like "1/4" for "=" is real Unicode and appears in real text.

        Substituting it would corrupt correct pages to repair broken ones.
        """
        text = "h ¼ 1.62 g/g and χ1 = 26°, Tₘ = 13 °C"
        assert normalize_harvested_text(text) == text

    def test_latex_from_the_model_path_is_preserved(self):
        latex = r"$h_{\text{im}} = 5 \times 0.263 = 1.315$"
        assert normalize_harvested_text(latex) == latex

    @pytest.mark.parametrize("empty", [None, ""])
    def test_empty_input_needs_no_caller_guard(self, empty):
        assert normalize_harvested_text(empty) == ""
        assert count_control_damage(empty) == 0


class TestEveryCaptionWritePathNormalises:
    """A caption is stored in TWO places, and both read the text layer.

    `figure_extractor._build_asset` sets it from the detector's caption box;
    `figures.apply_label_pass` fills it in from the label line when the
    detector found none. Normalising only the first left two damaged captions
    in a full corpus run. This asserts on the source so a third write path
    cannot be added without the repair.
    """

    def test_both_caption_writers_call_the_normaliser(self):
        from pathlib import Path
        import src.processing.utils.figure_extractor as fe
        import src.processing.strategies.research_article.figures as fig

        for module in (fe, fig):
            source = Path(module.__file__).read_text(encoding="utf-8")
            assert "normalize_harvested_text" in source, (
                f"{module.__name__} stores caption text and must repair it")

    def test_the_label_pass_caption_is_repaired(self):
        """The exact case a full corpus run left behind."""
        from src.processing.strategies.research_article.figures import (
            normalize_harvested_text as n)
        assert n("Fig. 10.3 patients with mcEDS\x02 Cutaneous features") == (
            "Fig. 10.3 patients with mcEDS-Cutaneous features")


class TestTheParityFold:
    """The harness folds this damage on BOTH sides of its comparison.

    The archived baseline is V5's own output and carries ten of these
    characters itself. It must stay byte-fixed (tests/baselines/README.md), so
    the fold has to happen at comparison time rather than by regenerating it.
    """

    def test_whitespace_collapse_alone_would_not_have_caught_them(self):
        r"""`\s` does not match these, which is why the fold was needed."""
        import re
        assert re.sub(r"\s+", " ", "measure\x02ments") == "measure\x02ments"

    def test_a_damaged_and_a_repaired_caption_compare_equal(self):
        from tests.parity import normalize_text
        assert (normalize_text("Fig. 1. T 1 analysis of a satu\x02ration curve")
                == normalize_text("Fig. 1. T 1 analysis of a saturation curve"))

    def test_a_real_content_difference_still_fails(self):
        """The fold must not make the comparison blind to actual changes."""
        from tests.parity import normalize_text
        assert (normalize_text("Fig. 1. saturation recovery")
                != normalize_text("Fig. 2. saturation recovery"))


class TestIdempotence:
    def test_running_twice_changes_nothing_further(self):
        text = "measure\x02ments at 149 \x03 1 ms with cross\x02linking"
        once = normalize_harvested_text(text)
        assert normalize_harvested_text(once) == once

    def test_a_full_corpus_line_comes_out_clean(self):
        line = ("by the model. Experiments using tendon expansion measure\x02ments "
                "and relation to molecular collagen separation models")
        cleaned = normalize_harvested_text(line)
        assert count_control_damage(cleaned) == 0
        assert "measurements" in cleaned
