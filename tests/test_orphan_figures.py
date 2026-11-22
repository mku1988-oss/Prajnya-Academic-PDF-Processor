"""
Tests for the orphan-figure rescue (utils/orphan_figures.py, ledger T24).

For biology researchers: a hand-built page (tests/fakes.py) with a printed
caption, a line of prose above it, and a fake low-cut-off detector that returns
whatever boxes a test wants. Page and image are the same size, so a pixel is a
point and every box can be read straight off the test.
"""

from dataclasses import dataclass

from PIL import Image

from src.processing.layout.base import LABEL_IMAGE, LABEL_TABLE, Detection
from src.processing.pdf.base import Rect, Word
from src.processing.utils.orphan_figures import rescue_orphan_figures
from tests.fakes import FakePage

SIZE = (612.0, 792.0)


def word(text, x0, y0, x1, y1):
    return Word(text=text, rect=Rect(x0, y0, x1, y1), rotated=False, baseline=y1 - 2)


def line(text, x0, y0, width_per_word=40.0, height=8.0):
    """A printed line, one word box per word, left to right."""
    out, x = [], x0
    for token in text.split():
        out.append(word(token, x, y0, x + width_per_word - 6, y0 + height))
        x += width_per_word
    return out


PROSE_ABOVE = line("The pulling event creates a thin filament between two cells here", 50, 200)
CAPTION = (line("Figure 5. Colocalization and stretching of collagen and FN", 50, 500)
           + line("Shown is a thin membrane extension colocalized with collagen", 50, 511))


@dataclass
class Crop:
    bbox: tuple
    label_key: str


class FakeLowDetector:
    """Returns the boxes it was built with; counts its calls."""

    def __init__(self, *boxes):
        self.boxes = boxes
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [Detection(label=LABEL_IMAGE, box_px=b) for b in self.boxes]


def run(words, crops=(), detections=(), low=None):
    page = FakePage(words=words, size=SIZE, number=8)
    image = Image.new("RGB", (int(SIZE[0]), int(SIZE[1])), "white")
    low = low or FakeLowDetector()
    return rescue_orphan_figures(page, image, list(crops), list(detections), low), low


def test_two_low_score_panels_above_an_orphan_caption_become_one_figure():
    rescued, _ = run(PROSE_ABOVE + CAPTION,
                     low=FakeLowDetector((60, 230, 300, 480), (320, 230, 560, 480)))
    assert [d.box_px for d in rescued] == [(60, 230, 560, 480)]
    assert rescued[0].label == LABEL_IMAGE


def test_a_caption_some_crop_already_claims_calls_no_detector():
    rescued, low = run(PROSE_ABOVE + CAPTION, crops=[Crop((60, 230, 560, 480), "figure 5")],
                       low=FakeLowDetector((60, 230, 300, 480)))
    assert rescued == [] and low.calls == 0


def test_a_box_duplicating_an_existing_crop_is_refused():
    """The page's Figure 4 is already cropped; the weak box over it is not a new figure."""
    figure_4 = Crop((60, 230, 300, 480), "figure 4")
    rescued, _ = run(PROSE_ABOVE + CAPTION, crops=[figure_4],
                     low=FakeLowDetector((70, 240, 290, 470)))
    assert rescued == []


def test_a_box_over_a_table_is_refused():
    rescued, _ = run(PROSE_ABOVE + CAPTION,
                     detections=[Detection(label=LABEL_TABLE, box_px=(60, 230, 560, 480))],
                     low=FakeLowDetector((70, 240, 550, 470)))
    assert rescued == []


def test_nothing_in_the_band_adds_nothing():
    """The figure is on the facing page: the space above its caption is prose."""
    rescued, _ = run(PROSE_ABOVE + CAPTION, low=FakeLowDetector((60, 600, 300, 700)))
    assert rescued == []


def test_a_box_above_the_prose_line_is_outside_the_band():
    rescued, _ = run(PROSE_ABOVE + CAPTION, low=FakeLowDetector((60, 60, 300, 180)))
    assert rescued == []


def test_a_figure_mentioned_inside_a_paragraph_is_not_a_caption():
    """'Figure 5 shows ...' continuing the line above at the same margin is prose."""
    paragraph = (line("as the membrane retracts the filament is pulled inward and", 50, 489)
                 + line("Figure 5 shows the colocalization of collagen and FN here", 50, 500))
    rescued, low = run(paragraph, low=FakeLowDetector((60, 230, 300, 480)))
    assert rescued == [] and low.calls == 0


# --- the switch: on for articles and single chapters, off for a whole book ---

def _bare_pipeline(rescue_on):
    """A BodyPipeline with only what `_rescue_orphan_figures` touches."""
    from src.processing.strategies.research_article.body_pipeline import BodyPipeline

    class Recorder:
        """Stands in for the layout model; counts whether the rescue asked it."""
        asked = 0

        def at_threshold(self, threshold):
            Recorder.asked += 1
            return FakeLowDetector()

    pipeline = BodyPipeline.__new__(BodyPipeline)
    pipeline.layout_detector = Recorder()
    pipeline._rescue_detector = None
    pipeline.rescue_orphan_figures = rescue_on
    pipeline.logger = __import__("logging").getLogger("test")
    return pipeline, Recorder


def _call_rescue(pipeline):
    page = FakePage(words=[], size=SIZE, number=8)
    figures = ["figure"]
    out = pipeline._rescue_orphan_figures(None, page, 7, Image.new("RGB", (int(SIZE[0]), int(SIZE[1]))),
                                          [], figures)
    return out, figures


def test_the_rescue_runs_by_default():
    from src.processing.strategies.research_article.body_pipeline import BodyPipeline
    import inspect
    assert inspect.signature(BodyPipeline.__init__).parameters[
        "rescue_orphan_figures"].default is True
    pipeline, recorder = _bare_pipeline(True)
    _call_rescue(pipeline)
    assert recorder.asked == 1


def test_the_rescue_is_skipped_when_switched_off():
    pipeline, recorder = _bare_pipeline(False)
    out, figures = _call_rescue(pipeline)
    assert recorder.asked == 0
    assert out is figures


def test_the_book_reader_builds_its_pipeline_with_the_rescue_off(monkeypatch):
    """The router's whole-book pipeline is its own object, built switched off."""
    from src.processing.strategies import router
    from src.processing.strategies.research_article import body_pipeline as bp
    from src.processing.strategies.research_article import provider_factory
    from src.processing.config import loader as config_loader
    from src.processing.utils import page_classifier

    seen = {}

    class FakeLoader:
        def __init__(self, *a, **k): pass
        def strategy(self): return {}
        def vlm_profile(self, ref): return None

    monkeypatch.setattr(config_loader, "ConfigLoader", FakeLoader)
    monkeypatch.setattr(page_classifier, "PageClassifier", lambda profile: None)
    monkeypatch.setattr(provider_factory, "ProviderFactory", lambda *a, **k: None)
    monkeypatch.setattr(router, "resolve_reference", lambda ref, cfg: ref)
    monkeypatch.setattr(bp.BodyPipeline, "from_config",
                        classmethod(lambda cls, *a, **k: seen.update(k) or "pipeline"))

    assert router._book_body_pipeline("cfg", [], None) == "pipeline"
    assert seen["rescue_orphan_figures"] is False
