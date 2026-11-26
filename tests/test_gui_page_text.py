"""Tests for src/gui/page_text.py: \\( \\) and \\[ \\] formulas in Studio's page view.

Each formula is checked twice: the rewrite itself, and what Studio finally
draws, through the very markdown function `ui.markdown` uses (so a pass means
MathML on screen, not just a changed string).
"""
from nicegui.elements.markdown import prepare_content

from src.gui.page_text import prepare_page_markdown

#: the extras results.py passes to ui.markdown
EXTRAS = "fenced-code-blocks tables latex"

#: a real line of NMR p1 (v6.9.0 output, olmOCR's delimiters)
NMR_P1 = (r"defined by integral multiples \( N = 1, 4 \) and 24 times, "
          r"(2) a single water chain per cleft at \( h_2 = 0.264 \) g/g")


def _drawn(text):
    return prepare_content(prepare_page_markdown(text), EXTRAS)


class TestBracketFormulasAreDrawn:
    def test_inline_brackets_become_dollars(self):
        assert prepare_page_markdown(r"at \( h_2 = 0.264 \) g/g") == "at $h_2 = 0.264$ g/g"

    def test_display_brackets_become_double_dollars(self):
        text = "rate is\n\n\\[ R_1 = f_o R_o \\]\n\nwhere"
        assert prepare_page_markdown(text) == "rate is\n\n$$R_1 = f_o R_o$$\n\nwhere"

    def test_the_nmr_abstract_is_drawn_as_mathml(self):
        html = _drawn(NMR_P1)
        assert html.count("<math") == 2
        assert "<em>" not in html          # the underscores used to turn into italics
        assert "\\(" not in html and "( h" not in html

    def test_a_formula_wrapped_onto_a_second_line(self):
        assert prepare_page_markdown("at \\( h_2 =\n0.264 \\) g/g") == "at $h_2 = 0.264$ g/g"

    def test_a_display_formula_on_several_lines(self):
        assert "<math" in _drawn("\\[ R_1 =\nf_o R_o \\]")


class TestNothingElseChanges:
    def test_dollar_text_is_returned_unchanged(self):
        text = "where $f_i$ is the fraction\n\n$$1/T_1 = \\sum f_i$$\r\nend"
        assert prepare_page_markdown(text) is text

    def test_an_unclosed_bracket_does_not_swallow_the_next_paragraph(self):
        text = "a stray \\( opening\n\nnext paragraph \\( x \\) here"
        out = prepare_page_markdown(text)
        assert out.startswith("a stray \\( opening\n\n")
        assert out.endswith("next paragraph $x$ here")

    def test_code_is_left_as_written(self):
        text = "run `re.compile(r'\\(x\\)')` then \\( y \\)\n```\n\\[ z \\]\n```"
        out = prepare_page_markdown(text)
        assert "`re.compile(r'\\(x\\)')`" in out and "\n```\n\\[ z \\]\n```" in out
        assert "$y$" in out

    def test_an_escaped_backslash_is_not_a_formula(self):
        assert prepare_page_markdown(r"path C:\\(x\\) end") == r"path C:\\(x\\) end"

    def test_a_formula_holding_a_dollar_is_left_alone(self):
        text = r"cost \( \$5 \) each"
        assert prepare_page_markdown(text) == text

    def test_an_empty_formula_is_left_alone(self):
        assert prepare_page_markdown(r"empty \(  \) here") == r"empty \(  \) here"


class TestOneBadFormulaNoLongerBreaksThePage:
    """markdown2's latex extra raises on a formula latex2mathml cannot read;
    11 of 4,955 stored pages failed to draw because of it (session 48)."""

    PAGE = "good $h_2 = 0.264$ then bad $a_b_c$ and display\n\n$$x^2^3$$\n\nend $R_1$"

    def test_the_unguarded_extra_really_raises(self):
        import pytest
        with pytest.raises(Exception):
            prepare_content(self.PAGE, EXTRAS)

    def test_the_page_draws_with_good_formulas_as_maths_and_bad_ones_as_text(self):
        html = _drawn(self.PAGE)
        assert html.count("<math") == 2             # h_2 and R_1
        assert "<code>$a_b_c$</code>" in html and "<code>$$x^2^3$$</code>" in html

    def test_a_page_whose_formulas_all_parse_is_the_same_string(self):
        text = "where $f_i$ is\n\n$$1/T_1 = \\sum f_i$$\n\n`$code$` end"
        assert prepare_page_markdown(text) is text

    def test_a_formula_inside_code_is_not_touched(self):
        text = "see `$a_b_c$` here"
        assert prepare_page_markdown(text) is text
