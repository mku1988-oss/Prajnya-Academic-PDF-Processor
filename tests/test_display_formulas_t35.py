"""
T35: display equations on text-layer pages are read from their crop.

For biology researchers: these check, on made-up pages, that an equation's
garbled text-layer run is swapped for the model's LaTeX, that no word
outside the equation's box is touched, and that a failed or impossible
read leaves the page exactly as it was. Nothing here needs a model.
"""
from PIL import Image

from src.processing.layout.base import Detection
from src.processing.pdf.base import Rect, Word
from src.processing.utils.display_formulas import (
    clean_latex, formula_boxes, run_of_words, with_display_formulas)
from tests.fakes import FakePage

TEXT = ("The two domains are coupled via\r\n"
        "qtmd ¼ Dm r\r\n2\r\nmmd þRd; (1)\r\n"
        "where Rd is the reaction term.")
BOX = Rect(100, 200, 400, 240)          # the equation's box, in points


def equation_page(text=TEXT) -> FakePage:
    """A page whose equation words sit inside BOX and prose words outside it."""
    inside = ["qtmd", "¼", "Dm", "r", "2", "mmd", "þRd;"]
    outside_above = ["The", "two", "domains", "are", "coupled", "via"]
    outside_below = ["(1)", "where", "Rd", "is", "the", "reaction", "term."]
    words = [Word(w, Rect(60 + 30 * i, 180, 85 + 30 * i, 190)) for i, w in enumerate(outside_above)]
    words += [Word(w, Rect(110 + 35 * i, 210, 135 + 35 * i, 225)) for i, w in enumerate(inside)]
    words += [Word("(1)", Rect(450, 210, 470, 225))]           # the number, outside the box
    words += [Word(w, Rect(60 + 30 * i, 260, 85 + 30 * i, 270)) for i, w in enumerate(outside_below[1:])]
    return FakePage(words=words, text=text)


def test_the_equation_run_becomes_latex_and_the_rest_is_kept():
    page = equation_page()
    new, records = with_display_formulas(TEXT, page, [BOX], lambda box: r"\partial_t m_d = R_d")
    assert new == ("The two domains are coupled via\r\n"
                   "$$\\partial_t m_d = R_d$$ (1)\r\n"
                   "where Rd is the reaction term.")
    assert records == [{"bbox": [100, 200, 400, 240], "status": "read"}]


def test_an_empty_or_failed_read_leaves_the_text_as_it_was():
    page = equation_page()
    for answer in ("", None, "  $$  $$ "):
        new, records = with_display_formulas(TEXT, page, [BOX], lambda box: answer)
        assert new == TEXT
        assert records[0]["status"] == "empty_read"


def test_a_box_whose_words_are_not_one_run_is_not_read():
    # the text layer puts a prose word between the equation's words
    broken = TEXT.replace("Dm r", "Dm SOMETHING r")
    calls = []
    new, records = with_display_formulas(broken, equation_page(broken), [BOX],
                                         lambda box: calls.append(box) or "x")
    assert new == broken and calls == []
    assert records[0]["status"] == "not_in_text"


def test_two_boxes_over_one_run_read_it_once():
    page = equation_page()
    new, records = with_display_formulas(TEXT, page, [BOX, BOX], lambda box: "x")
    assert new.count("$$x$$") == 1
    assert [r["status"] for r in records] == ["read", "not_in_text"]


def test_the_text_clean_up_leaves_latex_alone():
    from src.processing.utils.text_normalization import normalize_harvested_text
    text = "via\r\n$$\\partial_t m_{\\text{DD}} = \\frac{a}{b} \\nabla^2$$ (1)\r\nwhere"
    assert normalize_harvested_text(text) == text


def test_run_of_words_allows_only_whitespace_between():
    assert run_of_words("a b\r\nc d", ["b", "c"]) == (2, 6)
    assert run_of_words("a b x c d", ["b", "c"]) is None
    assert run_of_words("a b", []) is None


def test_clean_latex_unwraps_and_joins_lines():
    assert clean_latex("$$\nx = 1\n$$") == "x = 1"
    assert clean_latex(r"\[ a +" + "\n" + r" b \]") == "a + b"
    assert clean_latex(None) == ""


class _Detector:
    def detect(self, image, page=None):
        return [Detection("display_formula", (100, 200, 400, 240), 0.9),
                Detection("text", (0, 0, 50, 50), 0.9)]


class _RenderedPage(FakePage):
    def render(self, *, dpi=None, max_edge=None):
        return Image.new("RGB", (1224, 1584), "white")      # twice the page's points


def test_formula_boxes_are_in_points_and_only_display_formulas():
    boxes = formula_boxes(_RenderedPage(), _Detector())
    assert boxes == [Rect(50.0, 100.0, 200.0, 120.0)]
