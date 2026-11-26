"""Tests for src/gui/equations.py: the Results page's "Equations" list.

The cases are taken from real stored pages (eval/studio_equations/GATE.md):
each number form a reader actually wrote, and each thing that must NOT be
listed.
"""
from nicegui.elements.markdown import prepare_content

from src.gui.equations import equations_in_text, list_display_equations
from src.gui.page_text import guard_unparsable_formulas


def _numbers(text):
    return [eq["number"] for eq in equations_in_text(text, page_number=1)]


class TestWhatCountsAsADisplayEquation:
    def test_an_equation_on_its_own_line(self):
        found = equations_in_text("rate is\n\n$$1/T_1 = \\sum f_i$$\n\nwhere", page_number=3)
        assert found == [{"page": 3, "latex": "1/T_1 = \\sum f_i", "number": None}]

    def test_two_inline_formulas_touching_are_not_one(self):
        # PMC10965928 p11: "$+\\infty$$^{43,44}$"
        assert equations_in_text("ranges from $-\\infty$ to $+\\infty$$^{43,44}$. The", 1) == []

    def test_a_table_row_written_as_text_is_not_an_equation(self):
        # Tendon assembly p296: gene names in a table read as running text
        text = "McLaughlin et al. (2006)\n\n$$Fbln4^{R/R}$$ + Adulthood Aortic aneurysm Hanada et al. (2007)\n"
        assert equations_in_text(text, 1) == []

    def test_debris_before_the_number_keeps_the_equation_but_not_the_number(self):
        # PMC11043347 p11: the text layer's "ti 1Þð1Þ" after the equation
        assert _numbers("equation:\r\n$$v_i = (d_{i+1} - d_{i-1})$$ ti 1Þð1Þ\r\nOnly moving") == [None]

    def test_olmocr_display_brackets_count_too(self):
        document = {"pages": [{"page_number": 2, "content": "as\n\n\\[\nD = D_0\n\\]\n\nwhere"}]}
        assert list_display_equations(document)[0]["latex"] == "D = D_0"


class TestEquationNumbers:
    def test_number_alone_on_the_next_line(self):
        assert _numbers("$$X = Y,$$\n\n(1)\n\nwhere $H$") == ["1"]

    def test_number_after_the_closing_dollars(self):
        assert _numbers("$$TP / (TP + FN)$$ (5)\r\n$$MCC = 1$$") == ["5", None]

    def test_square_brackets_and_lettered_numbers(self):
        assert _numbers("$$K_x = a.$$ [5]\n\n$$P = b$$\n\n(19a)\n") == ["5", "19a"]

    def test_mangled_brackets_from_a_pdf_font(self):
        assert _numbers("$$E = mc^2$$\r\n: ð8Þ\r\nNote that") == ["8"]

    def test_tag_inside_is_the_number_and_is_removed_from_the_latex(self):
        found = equations_in_text("$$a = b. \\tag {16}$$\n\nFor the", 1)
        assert found[0]["number"] == "16" and "tag" not in found[0]["latex"]

    def test_qquad_number_at_the_end(self):
        assert _numbers("$$a = b \\qquad (3)$$\n") == ["3"]

    def test_a_word_in_brackets_is_not_a_number(self):
        # PMC5991653 p3: "\\quad (\\text{activator})"
        assert _numbers("$$f_+ = a \\quad (\\text{activator}),$$\n") == [None]

    def test_prose_starting_with_a_bracketed_number_is_not_a_label(self):
        # NMR p1 style: "(2) the Berendsen single water chain ..."
        assert _numbers("$$h_1 = 0.0658$$\n(2) the Berendsen single water chain\n") == [None]

    def test_a_year_is_not_an_equation_number(self):
        assert _numbers("$$a = b$$\n\n(2007)\n") == [None]


class TestDrawing:
    def test_a_listed_equation_draws_as_mathml(self):
        eq = equations_in_text("$$\\frac{\\partial \\phi}{\\partial t} = M \\nabla^2 \\mu$$\n(2)\n", 1)[0]
        assert "<math" in prepare_content(f"$${eq['latex']}$$", "latex")

    def test_an_unparsable_formula_is_shown_as_its_text_not_lost(self):
        # Unguarded, markdown2's latex extra RAISES here (the panel and the
        # page view both pass formulas through the guard first).
        html = prepare_content(guard_unparsable_formulas("$$a_b_c$$"), "latex")
        assert "a_b_c" in html and "<math" not in html
