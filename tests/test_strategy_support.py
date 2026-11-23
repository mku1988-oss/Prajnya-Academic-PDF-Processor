#!/usr/bin/env python3
"""
Tests for the strategy support modules: rendering, figures glue, provider factory.

Rendering is tested against the real benchmark PDF (skipped when absent, like
test_pdf_backend.py). The figures glue is tested on FakePage + synthetic
images, so the geometry is explicit. The provider factory is tested against
the real shipped config tree — the same one test_config_tree.py pins.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from src.processing.config.loader import ConfigLoader
from src.processing.layout.base import LABEL_IMAGE, LABEL_IMAGE_CAPTION, Detection
from src.processing.pdf import get_backend
from src.processing.pdf.base import ImageBlock, Rect
from src.processing.strategies.research_article.figures import (
    extract_page_figures,
    image_box_to_pdf_rect,
    make_caption_reader,
    write_figure_manifest,
)
from src.processing.strategies.research_article.provider_factory import (
    DEFAULT_RESOLUTION,
    ProviderFactory,
)
from src.processing.strategies.research_article.rendering import (
    RENDER_PADDING_PX,
    DocumentCache,
    render_extraction_page,
    render_page,
)
from tests.fakes import FakePage, word_row
from tests.local_samples import BENCHMARK_PDF as LOCAL_BENCHMARK_PDF

BENCHMARK_PDF = LOCAL_BENCHMARK_PDF
CONFIG_ROOT = Path(__file__).parent.parent / "config"

needs_benchmark = pytest.mark.skipif(
    not BENCHMARK_PDF.is_file(),
    reason=f"benchmark PDF not available at {BENCHMARK_PDF}",
)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

@needs_benchmark
class TestRendering:
    def test_render_page_pads_to_target_plus_border(self):
        with DocumentCache(get_backend("pypdfium2"), BENCHMARK_PDF) as cache:
            image = render_page(cache.page(0), target_size=1024)
        # Longest edge = target + the white border on both sides.
        assert max(image.size) == 1024 + 2 * RENDER_PADDING_PX
        # The border really is white.
        assert image.getpixel((2, 2)) == (255, 255, 255)
        assert image.getpixel((image.width - 3, image.height - 3)) == (255, 255, 255)

    def test_render_extraction_page_scales_with_dpi(self):
        with DocumentCache(get_backend("pypdfium2"), BENCHMARK_PDF) as cache:
            page = cache.page(0)
            longest_pts = max(page.size)
            image = render_extraction_page(page, extraction_dpi=150)
        expected = int(longest_pts / 72.0 * 150)
        # PDFium rounds the scaled edge, so allow one pixel either way.
        assert abs(max(image.size) - (expected + 2 * RENDER_PADDING_PX)) <= 1

    def test_content_is_pasted_inside_the_border(self):
        """The page content must sit exactly `padding` in from the corner."""
        with DocumentCache(get_backend("pypdfium2"), BENCHMARK_PDF) as cache:
            page = cache.page(1)  # a page with figures, so content is unmistakable
            raw = page.render(max_edge=512)
            padded = render_page(page, target_size=512)
        pad = RENDER_PADDING_PX
        crop = padded.crop((pad, pad, pad + raw.width, pad + raw.height))
        assert list(crop.getdata()) == list(raw.getdata())

    def test_cache_memoizes_renders(self):
        with DocumentCache(get_backend("pypdfium2"), BENCHMARK_PDF) as cache:
            first = cache.page_image(0, target_size=512)
            second = cache.page_image(0, target_size=512)
            different = cache.page_image(0, target_size=256)
        assert first is second          # cache hit: the same object
        assert different is not first   # different size: a new render

    def test_closed_cache_refuses_page_access_but_keeps_images(self):
        cache = DocumentCache(get_backend("pypdfium2"), BENCHMARK_PDF)
        image = cache.page_image(0, target_size=256)
        cache.close()
        with pytest.raises(RuntimeError):
            cache.page(0)
        # Rendered images are plain PIL objects and survive the close.
        assert image.size[0] > 0 and image.size[1] > 0


def test_render_page_rejects_zero_size_page():
    page = FakePage(size=(0.0, 0.0))
    with pytest.raises(ValueError):
        render_page(page, target_size=1024)


# ---------------------------------------------------------------------------
# figures glue
# ---------------------------------------------------------------------------

def _page_image_for(page: FakePage, scale: float = 2.0) -> Image.Image:
    """A blank page image consistent with rendering.render_page's geometry."""
    width_pts, height_pts = page.size
    return Image.new(
        "RGB",
        (int(width_pts * scale) + 2 * RENDER_PADDING_PX,
         int(height_pts * scale) + 2 * RENDER_PADDING_PX),
        (255, 255, 255),
    )


class TestImageBoxToPdfRect:
    def test_round_trip_maps_pixels_back_to_points(self):
        page = FakePage(size=(600.0, 800.0))
        page_image = _page_image_for(page, scale=2.0)
        # A box at content pixels (100..300, 200..400) -> points (50..150, 100..200).
        pad = RENDER_PADDING_PX
        rect = image_box_to_pdf_rect((100 + pad, 200 + pad, 300 + pad, 400 + pad),
                                     page, page_image)
        # The 4pt slop expands each edge outward.
        assert rect.x0 == pytest.approx(50 - 4.0)
        assert rect.y0 == pytest.approx(100 - 4.0)
        assert rect.x1 == pytest.approx(150 + 4.0)
        assert rect.y1 == pytest.approx(200 + 4.0)


class TestCaptionReader:
    def test_reads_caption_from_text_layer(self):
        # A caption line at y=700pt: "Figure 1. Collagen hydration."
        words = word_row(["Figure", "1.", "Collagen", "hydration."], y=700.0, x=100.0)
        page = FakePage(words=words, size=(600.0, 800.0))
        page_image = _page_image_for(page, scale=2.0)
        read = make_caption_reader(page, page_image)
        pad = RENDER_PADDING_PX
        # Caption box in image pixels covering that line (points * 2 + padding).
        box = (180 + pad, 1380 + pad, 700 + pad, 1440 + pad)
        assert "Collagen" in read(box)

    def test_empty_box_reads_nothing(self):
        page = FakePage(size=(600.0, 800.0))
        read = make_caption_reader(page, _page_image_for(page))
        assert read(None) == ""

    def test_scanned_page_falls_back_to_model(self):
        # One raster image covering the page => is_scanned_page is True.
        page = FakePage(
            images=[ImageBlock(rect=Rect(0, 0, 600, 800))],
            size=(600.0, 800.0),
        )
        page_image = _page_image_for(page, scale=2.0)
        read = make_caption_reader(page, page_image,
                                   read_crop=lambda crop: "Model-read caption")
        assert read((100, 100, 300, 200)) == "Model-read caption"

    def test_no_model_reader_means_empty_on_scanned(self):
        page = FakePage(images=[ImageBlock(rect=Rect(0, 0, 600, 800))],
                        size=(600.0, 800.0))
        read = make_caption_reader(page, _page_image_for(page))
        assert read((100, 100, 300, 200)) == ""


class TestExtractPageFigures:
    def test_no_detections_is_no_figures(self, tmp_path):
        page = FakePage(size=(600.0, 800.0))
        image = _page_image_for(page)
        assert extract_page_figures(page, image, None, "doc.pdf", tmp_path, 0) == []
        assert extract_page_figures(page, image, [], "doc.pdf", tmp_path, 0) == []

    def test_saves_detected_figure_with_caption(self, tmp_path):
        words = word_row(["Figure", "1.", "NMR", "spectra."], y=300.0, x=60.0)
        page = FakePage(words=words, size=(600.0, 800.0))
        page_image = _page_image_for(page, scale=2.0)
        pad = RENDER_PADDING_PX
        # Draw some ink where the figure is, so snapping has something to hold.
        for x in range(200, 400):
            for y in range(200, 500):
                page_image.putpixel((x, y), (0, 0, 0))
        detections = [
            Detection(label=LABEL_IMAGE, box_px=(200, 200, 400, 500)),
            # Caption box over the y=300pt text line (points * 2 + padding).
            Detection(label=LABEL_IMAGE_CAPTION,
                      box_px=(100 + pad, 580 + pad, 600 + pad, 640 + pad)),
        ]
        assets = extract_page_figures(page, page_image, detections,
                                      "water_compartments.pdf", tmp_path, page_idx=1)
        assert len(assets) == 1
        assert Path(assets[0].image_path).name.startswith("p2_fig")
        assert (tmp_path / "water_compartments").is_dir()
        assert "NMR" in assets[0].caption


class TestManifest:
    def test_writes_manifest_json(self, tmp_path):
        figures = [{"figure_id": "p2_fig1", "image_path": "x.png",
                    "caption": "c", "page": 2, "bbox": [1, 2, 3, 4]}]
        path = write_figure_manifest("doc.pdf", tmp_path, figures)
        assert path is not None and path.name == "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["figure_count"] == 1
        assert data["figures"][0]["figure_id"] == "p2_fig1"

    def test_no_figures_writes_nothing(self, tmp_path):
        assert write_figure_manifest("doc.pdf", tmp_path, []) is None
        assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# provider factory (against the real shipped config tree)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def factory() -> ProviderFactory:
    loader = ConfigLoader(CONFIG_ROOT)
    return ProviderFactory(loader, loader.strategy())


class TestProviderFactory:
    def test_builds_from_registry_references(self, factory):
        built = factory.build("@profile_registry.default", "@task_registry.raw_ocr")
        assert built.provider.model_profile.model_name == "olmocr-2-7b"
        assert built.provider.task_definition.task_name
        assert built.provider.task_type == "raw_ocr_extraction"

    def test_builds_from_literal_names(self, factory):
        built = factory.build("glm-ocr", "raw_ocr_extraction")
        assert built.provider.model_profile.is_fixed_prompt()

    def test_resolution_override_wins(self, factory):
        built = factory.build("@profile_registry.default", "@task_registry.raw_ocr",
                              resolution_override=1288)
        assert built.resolution == 1288

    def test_resolution_falls_back_to_constant(self, factory):
        # No shipped profile carries default_resolution, so the constant wins.
        built = factory.build("@profile_registry.default", "@task_registry.raw_ocr")
        assert built.resolution == DEFAULT_RESOLUTION

    def test_profile_default_resolution_beats_the_constant(self, factory, monkeypatch):
        real = factory.loader.vlm_profile
        monkeypatch.setattr(
            factory.loader, "vlm_profile",
            lambda name: {**real(name), "default_resolution": 999},
        )
        built = factory.build("@profile_registry.default", "@task_registry.raw_ocr")
        assert built.resolution == 999

    def test_endpoint_comes_from_strategy_config(self, factory):
        built = factory.build("@profile_registry.default", "@task_registry.raw_ocr")
        assert built.provider.config.api_endpoint == \
            "http://localhost:9931/v1/chat/completions"

    def test_build_shot_overrides_parameters_without_touching_base(self, factory):
        base = factory.build("@profile_registry.default", "@task_registry.raw_ocr")
        original_params = dict(base.profile_data.get("api_parameters") or {})
        shot = factory.build_shot(base.profile_data, "@task_registry.structured_title",
                                  temperature=0.1, max_tokens=2048,
                                  repeat_penalty=1.05, frequency_penalty=0.4)
        params = shot.model_profile.get_api_params()
        assert params["temperature"] == 0.1
        assert params["max_tokens"] == 2048
        # The base profile dict is unchanged (V5 copied before modifying).
        assert dict(base.profile_data.get("api_parameters") or {}) == original_params

    def test_unknown_profile_reference_fails_loudly(self, factory):
        with pytest.raises(ValueError):
            factory.build("@profile_registry.nonexistent", "@task_registry.raw_ocr")
