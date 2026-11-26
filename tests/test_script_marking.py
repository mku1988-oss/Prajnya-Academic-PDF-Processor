"""
T37b mark 4 (eval/title_text_layer/GATE_T37b.md): sub/superscripts in a text
layer are written as LaTeX, body text is untouched, and text that cannot be
aligned with its characters is returned plain.

For biology researchers: on a title page the affiliation letter after an
author's name is printed small and raised; the PDF stores it with no space,
so "Harvey-Samuel" + "a" reads "Harvey-Samuela". These tests build such lines
from literal character positions, no PDF involved.
"""

from __future__ import annotations

from src.processing.pdf.base import PrintedChar, Rect
from src.processing.utils.script_marking import (
    align, script_kinds, text_with_scripts,
)
from tests.fakes import FakePage

BODY = 10.0       # body letters: 10 pt tall, baseline at y = 100


def line(spec, start_x=72.0):
    """
    Characters from (text, kind) pairs: kind "" body, "^" raised, "_" lowered,
    "next" the start of the next line. Spaces set `space_before`.
    """
    characters, x, space = [], start_x, False
    for text, kind in spec:
        for letter in text:
            if letter == " ":
                space = True
                continue
            if kind == "^":
                height, baseline = 6.0, 96.0
            elif kind == "_":
                height, baseline = 6.0, 102.0
            elif kind == "next":
                height, baseline, x = 6.0, 112.0, start_x
            else:
                height, baseline = BODY, 100.0
            rect = Rect(x, baseline - 0.8 * height, x + 5.0, baseline + 0.2 * height)
            characters.append(PrintedChar(text=letter, rect=rect, baseline=baseline,
                                          space_before=space))
            x += 5.0
            space = False
    return characters


def page_for(spec, text):
    return FakePage(characters=line(spec), text=text)


class TestScripts:
    def test_an_affiliation_letter_is_set_apart(self):
        spec = [("Tim Harvey-Samuel", ""), ("a", "^"), (", Xuejiao Xu", ""), ("b,1", "^")]
        marked = text_with_scripts(page_for(spec, "Tim Harvey-Samuela, Xuejiao Xub,1"))
        assert marked.aligned
        assert marked.text == "Tim Harvey-Samuel$^{a}$, Xuejiao Xu$^{b,1}$"
        assert marked.runs == 2

    def test_a_subscript_is_written_as_one(self):
        marked = text_with_scripts(page_for([("CO", ""), ("2", "_"), (" uptake", "")],
                                            "CO2 uptake"))
        assert marked.text == "CO$_{2}$ uptake"

    def test_body_text_is_untouched(self):
        marked = text_with_scripts(page_for([("Collagen fibrils", "")], "Collagen fibrils"))
        assert marked.text == "Collagen fibrils"
        assert marked.runs == 0

    def test_the_next_line_in_a_smaller_font_is_not_a_script(self):
        """Affiliations below the author line: smaller, lower, back at the margin."""
        spec = [("Ruberti", ""), ("Dept", "next")]
        assert script_kinds(line(spec)) == [None] * len("RubertiDept")

    def test_a_space_ends_a_group(self):
        spec = [("Sah", ""), ("a", "^"), (" ", ""), ("c", "^")]
        marked = text_with_scripts(page_for(spec, "Saha c"))
        assert marked.text == "Sah$^{a}$ $^{c}$"


class TestAlignment:
    def test_the_line_end_hyphen_code_pairs_with_a_hyphen(self):
        characters = line([("neg-ative", "")])
        mapping, aligned = align("neg\x02\r\native", characters)
        assert aligned and len(mapping) == len(characters)

    def test_text_that_does_not_align_comes_back_plain(self):
        page = page_for([("Smith", ""), ("a", "^")], "Smith a extra")
        marked = text_with_scripts(page)
        assert not marked.aligned
        assert marked.text == "Smith a extra"
        assert marked.runs == 0


class TestNamesLoseTheirMarks:
    """Shot 2 copies the marks into the names it returns; they are removed."""

    def test_marks_are_removed_from_each_name(self):
        from src.processing.utils.script_marking import without_script_marks

        assert without_script_marks(["Mattia Serra$^{1}$", "Nirvay Sah$^a$",
                                     "L. Mahadevan$^{3,4,*}$$^*$"]) == [
            "Mattia Serra", "Nirvay Sah", "L. Mahadevan"]

    def test_a_whole_byline_in_one_string_keeps_its_commas(self):
        from src.processing.utils.script_marking import without_script_marks

        byline = "Tim Harvey-Samuel$^{a}$, Xuejiao Xu$^{b,1}$, and Luke Alphey$^{a,4,5}$"
        assert without_script_marks([byline]) == [
            "Tim Harvey-Samuel, Xuejiao Xu, and Luke Alphey"]

    def test_a_name_without_marks_is_unchanged(self):
        from src.processing.utils.script_marking import without_script_marks

        assert without_script_marks(["Jeffrey W. Ruberti"]) == ["Jeffrey W. Ruberti"]


class TestT37cFixes:
    """T37c (eval/title_text_layer/GATE_T37c.md): one test per fix."""

    def test_fix1_an_unclosed_group_and_its_glued_letter_go(self):
        from src.processing.citations.acceptance import clean_name
        from src.processing.utils.script_marking import without_script_marks

        # clean_name (run on every author afterwards) removes the "*".
        names = without_script_marks(["Samir Kumar Pal$^{4}*", "Silvia Vignolini $^{13}C",
                                      "Samuel Ojosnegros$^{}"])
        assert [clean_name(n) for n in names] == [
            "Samir Kumar Pal", "Silvia Vignolini", "Samuel Ojosnegros"]

    def test_fix2_a_running_header_leaves_the_journal(self):
        from src.processing.strategies.research_article.title_page import (
            _tidy_text_layer_metadata,
        )

        metadata = {"journal_name": "Science Advances | Research Article"}
        _tidy_text_layer_metadata(metadata)
        assert metadata["journal_name"] == "Science Advances"
        plain = {"journal_name": "Nature Communications"}
        _tidy_text_layer_metadata(plain)
        assert plain["journal_name"] == "Nature Communications"

    def test_fix3_a_raised_zero_in_another_font_is_a_prime(self):
        """PMC9295807: '2' + raised '0' in a maths font, not smaller."""
        two = PrintedChar(text="2", rect=Rect(72.0, 84.0, 79.0, 98.0), baseline=95.5,
                          font="Text", space_before=True)
        zero = PrintedChar(text="0", rect=Rect(79.0, 76.0, 84.0, 94.0), baseline=90.1,
                           font="Maths")
        hyphen = PrintedChar(text="-", rect=Rect(84.0, 84.0, 88.0, 98.0), baseline=95.5,
                             font="Text")
        marked = text_with_scripts(FakePage(characters=[two, zero, hyphen], text="20-"))
        assert marked.text == "2′-"

    def test_fix3_a_zero_in_the_same_font_stays_a_zero(self):
        one = PrintedChar(text="1", rect=Rect(72.0, 84.0, 79.0, 98.0), baseline=95.5, font="Text")
        zero = PrintedChar(text="0", rect=Rect(79.0, 84.0, 86.0, 98.0), baseline=95.5, font="Text")
        assert text_with_scripts(FakePage(characters=[one, zero], text="10")).text == "10"

    def test_fix4_orcid_ids_and_symbols_leave_the_names(self):
        from src.processing.utils.script_marking import without_script_marks

        assert without_script_marks(["Sayaka MiuraID,☯", "Qiqing TaoID",
                                     "Jiamin DengID1,2", "DAVID Smith"]) == [
            "Sayaka Miura,", "Qiqing Tao", "Jiamin Deng1,2", "DAVID Smith"]


class TestT37dFixes:
    """T37d (eval/title_text_layer/GATE_T37d.md): titles and capitals journals."""

    def test_fix1_title_scripts_become_unicode(self):
        from src.processing.utils.script_marking import unicode_scripts

        assert unicode_scripts("selective removal of CO$_{2}$ and H$_{2}$ storage") == \
            "selective removal of CO₂ and H₂ storage"
        assert unicode_scripts("g-C$_{3}$N$_{4}$/MeTMC-COP") == "g-C₃N₄/MeTMC-COP"
        assert unicode_scripts("Ca$^{2+}$ Handling") == "Ca²⁺ Handling"
        assert unicode_scripts("The m$^{6}$A RNA Modification") == "The m⁶A RNA Modification"

    def test_fix1_a_group_without_unicode_forms_becomes_plain(self):
        from src.processing.utils.script_marking import unicode_scripts

        assert unicode_scripts("myeloid and T$_{reg}$ cells") == "myeloid and Treg cells"

    def test_fix1_the_unicode_title_still_matches_the_text_layer(self):
        """The validation check folds with NFKD: C₃N₄ is found as C3N4."""
        from src.processing.citations.local_confidence import title_is_on_page

        page = "Synthesis and photocatalytic performance of g-C3N4/MeTMC-COP composite"
        assert title_is_on_page("Synthesis and photocatalytic performance of "
                                "g-C₃N₄/MeTMC-COP composite", page)

    def test_fix2_a_capitals_journal_is_recased(self):
        from src.processing.strategies.research_article.title_page import (
            _tidy_text_layer_metadata,
        )

        for given, expected in (("SCIENCE ADVANCES", "Science Advances"),
                                ("JOURNAL OF BIOLOGICAL CHEMISTRY",
                                 "Journal of Biological Chemistry"),
                                ("PNAS", "PNAS"), ("PLOS ONE", "PLOS ONE"),
                                ("Science Advances", "Science Advances")):
            metadata = {"journal_name": given}
            _tidy_text_layer_metadata(metadata)
            assert metadata["journal_name"] == expected


class TestT37eNotations:
    """T37e: every script notation the models write in a title is read."""

    def test_the_caret_pair_form_shot_2_wrote(self):
        from src.processing.utils.script_marking import unicode_scripts

        assert unicode_scripts("CD8^+^HLA-DR^+^ and CD4^+^Foxp3^+^ T cells") == \
            "CD8⁺HLA-DR⁺ and CD4⁺Foxp3⁺ T cells"

    def test_bare_latex_from_the_vision_model(self):
        from src.processing.utils.script_marking import unicode_scripts

        assert unicode_scripts("Ca^{2+} Handling") == "Ca²⁺ Handling"
        assert unicode_scripts("the P2RY_{12} receptor") == "the P2RY₁₂ receptor"
        assert unicode_scripts("myeloid and T_{reg} cells") == "myeloid and Treg cells"

    def test_a_group_without_braces_between_dollars(self):
        from src.processing.utils.script_marking import unicode_scripts

        assert unicode_scripts("CO$_2$ uptake") == "CO₂ uptake"

    def test_a_title_without_scripts_is_unchanged(self):
        from src.processing.utils.script_marking import unicode_scripts

        title = "Collagen fibril formation: a 2-step model (in vitro)"
        assert unicode_scripts(title) == title
