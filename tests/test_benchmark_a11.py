"""
The two measures amendment A11 adds to the benchmark: figure crops and formulas.

For biology researchers: these check the scoring code on made-up inputs
whose right answer is known (a synthetic page with a figure pasted at a
known place; formulas written the same way and differently), before any
real paper is scored. Nothing here needs a model, a GPU or a PDF.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from ground_truth.jats_to_common import jats_to_common          # noqa: E402
from harness.figure_locator import (crop_iou, locate_figure,     # noqa: E402
                                    score_crops, Location)
from harness.formulas import (formula_similarity, join_numbers,  # noqa: E402
                              latex_tokens, mathml_tokens, score_formulas)

MML = 'xmlns="http://www.w3.org/1998/Math/MathML"'


# --- formulas -------------------------------------------------------------------

def test_mathml_and_latex_of_the_same_formula_score_one():
    key = f'<math {MML}><mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup></math>'
    assert formula_similarity(key, "E = mc^{2}") == 1.0
    assert formula_similarity(key, "$E=mc^2$") == 1.0


def test_a_wrong_exponent_lowers_the_score():
    key = f'<math {MML}><mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup></math>'
    assert 0.5 < formula_similarity(key, "E = mc^3") < 1.0


def test_a_fraction_is_not_its_parts_side_by_side():
    assert latex_tokens(r"\frac{a+1}{b}") == ["a", "+", "1", "/", "b"]
    assert latex_tokens(r"a+1 b") != latex_tokens(r"\frac{a+1}{b}")


def test_styled_letters_and_split_numbers_read_the_same():
    bold = f'<math {MML}><msub><mi mathvariant="bold">v</mi><mi>i</mi></msub></math>'
    assert mathml_tokens(bold) == latex_tokens(r"\mathbf{v}_{i}")
    split = f'<math {MML}><mi>W</mi><mo>=</mo><mn>0</mn><mo>.</mo><mn>0192</mn></math>'
    assert formula_similarity(split, "W = 0.0192") == 1.0


def test_join_numbers_keeps_a_trailing_point_as_an_operator():
    assert join_numbers(["x", "=", "3", "."]) == ["x", "=", "3", "."]
    assert join_numbers(["1", ".", "5"]) == ["1.5"]


def test_annotations_and_invisible_operators_are_not_tokens():
    key = (f'<math {MML}><semantics><mrow><mi>x</mi><mo>&#x2062;</mo><mi>y</mi></mrow>'
           '<annotation encoding="application/x-tex">xy</annotation></semantics></math>')
    assert mathml_tokens(key) == ["x", "y"]


def test_unreadable_latex_scores_zero_and_a_missing_formula_is_none():
    key = f'<math {MML}><mi>x</mi></math>'
    assert formula_similarity(key, r"\frac{") == 0.0
    assert formula_similarity(key, None) is None


def test_score_formulas_reports_display_and_inline_apart():
    x = f'<math {MML}><mi>x</mi></math>'
    key = [{"id": "e1", "kind": "display", "mathml": x},
           {"id": "e2", "kind": "display", "mathml": x},
           {"id": "i1", "kind": "inline", "mathml": x}]
    result = score_formulas(key, {"e1": "x", "i1": "x"})
    assert result["formula_display"] == 0.5          # e2 missing scores 0
    assert result["formula_inline"] == 1.0
    assert score_formulas([], {})["formula_display"] is None


def test_equation_numbers_are_not_formula_content():
    key = f'<math {MML}><mi>x</mi><mo>=</mo><mn>1</mn></math>'
    for written in (r"x = 1 \tag{2}", r"x = 1 \quad (3)", r"x = 1 (4a)"):
        assert formula_similarity(key, written) == 1.0, written


# --- the adapters' formula reading ---------------------------------------------------

def test_math_pieces_reads_every_way_tools_write_math():
    from adapters.common import math_pieces
    text = (r"We set $x=1$ then $$E=mc^2$$ also \(a\) and \[b+c\] and "
            r'<math display="block">\frac{1}{2}</math> and <math>\mu</math>M.')
    kinds = [(k, v) for k, v in math_pieces(text) if k != "text"]
    assert kinds == [("inline", "x=1"), ("display", "E=mc^2"), ("inline", "a"),
                     ("display", "b+c"), ("display", r"\frac{1}{2}"), ("inline", r"\mu")]


def test_a_tool_formula_context_is_its_text_only():
    from adapters.common import formulas_of_pieces, math_pieces
    formulas = formulas_of_pieces(math_pieces(r"Energy $q$ is given by $$E = mc^2$$ where"))
    assert formulas[1]["context_before"] == "Energy is given by"


def test_match_formulas_pairs_by_context_in_reading_order():
    from harness.formulas import match_formulas
    key = [{"id": "e1", "context_before": "the diffusion of the membrane protein follows"},
           {"id": "e2", "context_before": "and the binding rate to the membrane is"},
           {"id": "e3", "context_before": "a sentence this tool never wrote at all here"}]
    tool = [{"text": "a", "context_before": "the diffusion of the membrane protein follows"},
            {"text": "noise", "context_before": "completely unrelated words about figures"},
            {"text": "b", "context_before": "and the binding rate to membrane is"}]
    assert match_formulas(key, tool) == {"e1": "a", "e2": "b"}
    assert match_formulas(key, None) == {}


# --- the answer key ----------------------------------------------------------------

JATS = f"""<article xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:mml="http://www.w3.org/1998/Math/MathML">
<front><journal-meta><journal-title-group><journal-title>J</journal-title></journal-title-group></journal-meta>
<article-meta><title-group><article-title>T</article-title></title-group></article-meta></front>
<body><sec>
<p>The energy is <inline-formula><mml:math><mml:mi>E</mml:mi></mml:math></inline-formula> as follows:</p>
<disp-formula id="eq1"><label>(1)</label><mml:math><mml:mi>E</mml:mi><mml:mo>=</mml:mo><mml:mn>1</mml:mn></mml:math></disp-formula>
<p>Second paragraph.</p>
<fig id="f1"><label>Figure 1</label><caption><p>A cell. <inline-formula><mml:math><mml:mi>q</mml:mi></mml:math></inline-formula></p></caption>
<graphic xlink:href="paper-g001.jpg"/></fig>
<table-wrap id="t1"><label>Table 1</label><table><tr><td><inline-formula><mml:math><mml:mi>z</mml:mi></mml:math></inline-formula></td></tr></table></table-wrap>
</sec></body></article>""".encode()


def test_key_keeps_formulas_in_body_order_with_their_place():
    formulas = jats_to_common(JATS, "P1")["formulas"]
    assert [(f["kind"], f["id"]) for f in formulas] == [("inline", "formula-1"), ("display", "eq1")]
    display = formulas[1]
    assert display["label"] == "(1)"
    assert mathml_tokens(display["mathml"]) == ["E", "=", "1"]
    assert display["paragraph_index"] == 0
    assert display["context_before"].endswith("as follows:")


def test_key_leaves_out_formulas_in_figures_and_tables():
    kinds = [f["id"] for f in jats_to_common(JATS, "P1")["formulas"]]
    assert len(kinds) == 2


def test_key_figure_keeps_its_publisher_image():
    assert jats_to_common(JATS, "P1")["figures"][0]["image"] == "paper-g001.jpg"


def test_key_figure_image_is_not_an_equation_in_its_caption():
    plos = JATS.replace(b"<caption><p>A cell.", b'<caption><p>A cell. <inline-graphic xlink:href="x"/>'
                        b'<disp-formula><graphic xlink:href="paper.e003"/></disp-formula>')
    assert jats_to_common(plos, "P1")["figures"][0]["image"] == "paper-g001.jpg"


# --- figure crops ------------------------------------------------------------------

def synthetic_page(figure: Image.Image, at: tuple, width_px: int, page_pt=(612, 792)):
    """A white page (page_pt points, rendered width_px wide) with `figure`
    pasted at `at` (pixels), plus some grey 'text' lines elsewhere."""
    scale = width_px / page_pt[0]
    page = Image.new("L", (width_px, int(page_pt[1] * scale)), 255)
    rng = np.random.default_rng(1)
    for row in range(40, page.height - 20, 22):
        line = rng.integers(60, 200, size=(6, width_px - 80), dtype=np.uint8)
        page.paste(Image.fromarray(line), (40, row))
    page.paste(figure, at)
    return page, page_pt[0], page_pt[1]


def textured_figure(size=(400, 300)) -> Image.Image:
    rng = np.random.default_rng(7)
    small = rng.integers(0, 255, size=(size[1] // 10, size[0] // 10), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.NEAREST)


def test_locator_finds_a_scaled_figure_on_the_right_page():
    publisher = textured_figure((800, 600))                 # the publisher's larger image
    printed = publisher.resize((300, 225), Image.BILINEAR)  # as printed on page 2
    blank, w, h = synthetic_page(Image.new("L", (1, 1), 255), (0, 0), 850)
    page2 = synthetic_page(printed, (120, 300), 850)
    where = locate_figure(publisher, [(blank, w, h), page2])
    assert where.page_index == 1
    scale = 850 / 612
    truth = (120 / scale, 300 / scale, 420 / scale, 525 / scale)
    assert crop_iou(truth, where.box) > 0.9
    assert where.score > 0.8      # resampled noise; the real threshold is fixed on the pilots


def test_crop_iou():
    assert crop_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert crop_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert crop_iou((0, 0, 10, 10), (0, 0, 10, 5)) == pytest.approx(0.5)


def crop_case(tool_name, crops, threshold=0.5):
    from harness import a11_measures
    key = {"figures": [{"id": "f1", "label_number": "1"}, {"id": "f2", "label_number": "2"}]}
    boxes = {"page_sizes": [[600, 800], [600, 800]],
             "figures": {"f1": {"page_index": 1, "box": [60, 80, 300, 400], "score": 0.9},
                         "f2": {"page_index": 0, "box": [0, 0, 10, 10], "score": 0.2}}}
    tool = {"tool": tool_name, "figures": [{"label_number": "1", "crops": crops}]}
    saved, a11_measures.LOCATE_THRESHOLD = a11_measures.LOCATE_THRESHOLD, threshold
    try:
        return a11_measures.crop_scores(key, tool, boxes)
    finally:
        a11_measures.LOCATE_THRESHOLD = saved


def test_crop_scores_turn_page_fractions_into_points():
    result = crop_case("mineru", [{"page_index": 1, "box": [0.1, 0.1, 0.5, 0.5], "unit": "fraction"}])
    assert result["crop_iou"] == pytest.approx(1.0)
    assert result["figures_unlocated"] == 1                 # f2's match is below the threshold


def test_crop_scores_join_the_crops_of_one_split_figure():
    halves = [{"page_index": 1, "box": [60, 80, 180, 400], "unit": "pt"},
              {"page_index": 1, "box": [180, 80, 300, 400], "unit": "pt"}]
    assert crop_case("prajnya", halves)["crop_iou"] == pytest.approx(1.0)


def test_crop_scores_are_not_measured_without_a_threshold_or_for_grobid():
    crops = [{"page_index": 1, "box": [60, 80, 300, 400], "unit": "pt"}]
    assert crop_case("prajnya", crops, threshold=None)["crop_iou"] is None
    assert crop_case("grobid", crops)["crop_iou"] is None


def test_docling_boxes_turn_to_a_top_left_origin():
    from adapters.docling import picture_crops
    picture = {"prov": [{"page_no": 2, "bbox": {"l": 50, "t": 700, "r": 250, "b": 500,
                                                 "coord_origin": "BOTTOMLEFT"}}]}
    data = {"pages": {"2": {"size": {"width": 612, "height": 792}}}}
    assert picture_crops(picture, data) == [{"page_index": 1, "box": [50.0, 92.0, 250.0, 292.0],
                                             "unit": "pt"}]


def test_prajnya_bbox_pixels_turn_to_points():
    from adapters.prajnya import crop_of
    assert crop_of(0, [25, 25, 325, 625]) == [{"page_index": 0, "box": [0.0, 0.0, 72.0, 144.0],
                                               "unit": "pt"}]


def test_score_crops_counts_a_missing_or_wrong_page_crop_as_zero():
    located = {"f1": Location(0, (0, 0, 10, 10), 0.9), "f2": Location(1, (0, 0, 10, 10), 0.9),
               "f3": Location(2, (0, 0, 10, 10), 0.9)}
    crops = {"f1": (0, (0, 0, 10, 10)), "f2": (0, (0, 0, 10, 10))}   # f2 on the wrong page, f3 none
    result = score_crops(located, crops)
    assert result["crop_iou"] == pytest.approx(1 / 3)
    assert result["crop_right"] == pytest.approx(1 / 3)
    assert score_crops({}, {})["crop_iou"] is None
