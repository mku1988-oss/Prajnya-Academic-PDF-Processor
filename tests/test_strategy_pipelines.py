#!/usr/bin/env python3
"""
Tests for the four pipeline modules: title_page, metadata_enrichment,
body_pipeline, cloud_pipeline.

No model, no server, no GPU: providers are stubs returning canned results,
pages are FakePage, and the document cache is a fake that hands out blank
images. What is tested is the *orchestration* each module ports from V5 —
routing decisions, fallbacks, metadata plumbing — because that is what the
port could have broken.
"""

import logging
from pathlib import Path
from typing import Optional

import pytest
from PIL import Image

from src.processing.layout.base import LABEL_TABLE, Detection
from src.processing.models import ContentPage, DocumentProfile
from src.processing.pdf.base import ImageBlock, Rect
from src.processing.strategies.research_article.body_pipeline import BodyPipeline
from src.processing.strategies.research_article.cloud_pipeline import (
    _split_metadata_block,
    process_body_pages_cloud,
)
from src.processing.strategies.research_article.metadata_enrichment import (
    annotate_provenance,
    enrich_metadata,
    extract_metadata_from_page1,
    prefer_text_layer_doi,
    sanitize_bibliographic_metadata,
)
from src.processing.strategies.research_article.title_page import (
    deduplicate_paragraphs,
    process_title_page,
)
from src.processing.utils.content_drop_check import CoverageResult
from src.processing.utils.page_classifier import PageType
from tests.fakes import FakePage, word_row

LOG = logging.getLogger("test")


# ---------------------------------------------------------------------------
# shared fakes
# ---------------------------------------------------------------------------

class FakeCache:
    """A DocumentCache stand-in: FakePages + blank images, no file."""

    def __init__(self, pages, pdf_path="doc.pdf"):
        self._pages = pages
        self.pdf_path = pdf_path

    def page(self, index):
        return self._pages[index]

    def page_image(self, index, target_size):
        return Image.new("RGB", (target_size, int(target_size * 1.3)), (255, 255, 255))

    def extraction_image(self, index, extraction_dpi):
        return self.page_image(index, 1024)

    def body_style(self):
        """No typography in these fakes: captions end on geometry alone."""
        return None


class FakeProvider:
    """A VlmProvider stand-in returning canned results."""

    def __init__(self, page_results=None, crop_text=""):
        self._page_results = list(page_results or [])
        self._crop_text = crop_text
        self.calls = []

    def process_page(self, page_image, base_text=None, task_definition=None):
        self.calls.append({"base_text": base_text, "task": task_definition})
        if self._page_results:
            return self._page_results.pop(0)
        return {"success": True, "content": "", "metadata": {}, "quality_score": 0.5}

    def read_crop(self, crop_image, prompt, max_tokens=None, image_first=None):
        return self._crop_text


class FakeFactory:
    """A ProviderFactory stand-in whose build_shot returns queued providers."""

    def __init__(self, providers):
        self._providers = list(providers)
        self.shot_calls = []

    def build_shot(self, base_profile_data, task, **params):
        self.shot_calls.append({"task": task, **params})
        return self._providers.pop(0)


# ---------------------------------------------------------------------------
# title_page
# ---------------------------------------------------------------------------

class TestDeduplicateParagraphs:
    def test_removes_repeats_preserving_order(self):
        assert deduplicate_paragraphs("Para A\n\nPara B\n\nPara A\n\nPara C") == \
            "Para A\n\nPara B\n\nPara C"

    def test_rewrapped_repeat_still_counts_as_duplicate(self):
        assert deduplicate_paragraphs("one  two\n\none two") == "one  two"

    def test_empty_text_passes_through(self):
        assert deduplicate_paragraphs("") == ""


class TestProcessTitlePage:
    def _run(self, shot1, shot2):
        page = FakePage(words=word_row(["An", "NMR", "method"], y=100.0))
        cache = FakeCache({0: page})
        factory = FakeFactory([shot1, shot2])
        result = process_title_page(
            cache, 0, factory,
            title_profile_data={"two_shot_parameters": {
                "shot1_ocr": {"temperature": 0.05},
                "shot2_analysis": {"temperature": 0.0}}},
            shot1_task="raw_ocr_extraction",
            shot2_task="structured_analysis_title_page",
            resolution=1288, logger=LOG)
        return result, factory

    def test_content_is_shot1_text_and_metadata_is_shot2(self):
        shot1 = FakeProvider([{"success": True, "content": "Title text\n\nTitle text\n\nBody",
                               "metadata": {}, "quality_score": 0.8}])
        shot2 = FakeProvider([{"success": True, "content": "",
                               "metadata": {"title": "An NMR method", "doi": "10.1/x"},
                               "quality_score": 0.2}])
        result, factory = self._run(shot1, shot2)
        # Content: deduplicated shot-1 text, never shot 2's.
        assert result.content == "Title text\n\nBody"
        assert result.metadata["title"] == "An NMR method"
        assert result.metadata["processing_method"] == "two_shot_title_page"
        # Shot 2 saw the cleaned text as base_text.
        assert shot2.calls[0]["base_text"] == "Title text\n\nBody"
        # Shot 1 saw the anchor text.
        assert shot1.calls[0]["base_text"].startswith("[")

    def test_empty_shot1_is_a_critical_failure(self):
        shot1 = FakeProvider([{"success": True, "content": "", "metadata": {}}])
        shot2 = FakeProvider([])
        result, _ = self._run(shot1, shot2)
        assert result.quality_score == 0.0
        assert result.metadata["error"] == "shot1_empty_title_page"

    def test_provider_exception_returns_failed_page_not_raise(self):
        class Exploding:
            def process_page(self, *a, **k):
                raise RuntimeError("server down")
        result, _ = self._run(Exploding(), FakeProvider([]))
        assert result.quality_score == 0.0
        assert "server down" in result.metadata["error"]


class TestShot1FromTheTextLayer:
    """
    T37 (eval/title_text_layer/GATE.md): on a born-digital title page the
    PDF's own text stands in for shot 1; shot 2 still sees the image. Mark 4:
    a scanned page, or a thin text layer, still goes to the model.
    """

    WORDS = [f"word{i}" for i in range(60)]

    @staticmethod
    def _run(page, providers, source="text_layer"):
        factory = FakeFactory(providers)
        result = process_title_page(
            FakeCache({0: page}), 0, factory,
            title_profile_data={"two_shot_parameters": {}},
            shot1_task="raw_ocr_extraction",
            shot2_task="structured_analysis_title_page",
            resolution=1288, logger=LOG, shot1_source=source)
        return result

    @staticmethod
    def shot(content="", metadata=None):
        return FakeProvider([{"success": True, "content": content,
                              "metadata": metadata or {}, "quality_score": 0.8}])

    def test_a_born_digital_page_skips_the_model_for_shot_1(self):
        page = FakePage(words=word_row(self.WORDS, y=100.0), text=" ".join(self.WORDS))
        shot2 = self.shot(metadata={"title": "T"})
        result = self._run(page, [shot2])
        assert result.content == " ".join(self.WORDS)
        assert result.metadata["shot1_source"] == "text_layer"
        assert result.metadata["title"] == "T"
        # Shot 2 was given the text layer as its text (and the image).
        assert shot2.calls[0]["base_text"] == " ".join(self.WORDS)

    def test_a_scanned_page_still_goes_to_the_model(self):
        page = FakePage(words=word_row(self.WORDS, y=100.0), text=" ".join(self.WORDS),
                        images=[ImageBlock(rect=Rect(0.0, 0.0, 612.0, 792.0))])
        result = self._run(page, [self.shot("model text"), self.shot()])
        assert result.content == "model text"
        assert result.metadata["shot1_source"] == "vlm"

    def test_a_thin_text_layer_still_goes_to_the_model(self):
        few = self.WORDS[:10]
        page = FakePage(words=word_row(few, y=100.0), text=" ".join(few))
        result = self._run(page, [self.shot("model text"), self.shot()])
        assert result.content == "model text"
        assert result.metadata["shot1_source"] == "vlm"

    def test_vlm_is_todays_behaviour(self):
        page = FakePage(words=word_row(self.WORDS, y=100.0), text=" ".join(self.WORDS))
        result = self._run(page, [self.shot("model text"), self.shot()], source="vlm")
        assert result.content == "model text"
        assert result.metadata["shot1_source"] == "vlm"

    def test_the_setting_is_part_of_the_title_fingerprint(self):
        """A title page saved by T39 under one setting is never reused under the other."""
        import inspect

        from src.processing.strategies.research_article.strategy import ResearchArticleStrategy

        source = inspect.getsource(ResearchArticleStrategy.title_settings_key)
        assert '"shot1_source": self.title_shot1_source' in source


# ---------------------------------------------------------------------------
# metadata_enrichment
# ---------------------------------------------------------------------------

class TestPreferTextLayerDoi:
    def _profile(self, doi=None, confidence=0.0):
        return DocumentProfile(pdf_path="p.pdf", page_count=8, file_size=1,
                               doi=doi, doi_confidence=confidence)

    def test_no_text_layer_doi_keeps_vlm(self):
        meta = prefer_text_layer_doi({"doi": "10.1/vlm"}, self._profile(), logger=LOG)
        assert meta["doi"] == "10.1/vlm"
        assert meta["doi_source"] == "vlm_extraction"

    def test_low_confidence_text_doi_is_ignored(self):
        meta = prefer_text_layer_doi({"doi": "10.1/vlm"},
                                     self._profile("10.9/text", 0.5), logger=LOG)
        assert meta["doi"] == "10.1/vlm"

    def test_confident_text_doi_overrides_vlm(self):
        meta = prefer_text_layer_doi({"doi": "10.1/garbled"},
                                     self._profile("10.1016/j.cellbi.2005.09.009", 0.98),
                                     logger=LOG)
        assert meta["doi"] == "10.1016/j.cellbi.2005.09.009"
        assert meta["doi_source"] == "pymupdf_text_layer"
        assert meta["doi_confidence"] == 0.98

    def test_backfills_when_vlm_found_none(self):
        meta = prefer_text_layer_doi({}, self._profile("10.1/text", 0.95), logger=LOG)
        assert meta["doi"] == "10.1/text"


class TestSanitizeAndProvenance:
    def test_empty_strings_become_none_and_authors_normalize(self):
        out = sanitize_bibliographic_metadata(
            {"doi": "  ", "title": "T", "authors": "A. One, B. Two"})
        assert out["doi"] is None
        assert out["authors"] == ["A. One", "B. Two"]

    def test_provenance_flags_missing_and_moves_suspicious(self):
        merged = {"title": "T", "authors": [], "suspicious_authors": ["A and B and C"],
                  "validation_confidence": 0.7, "validation_source": "zotero"}
        prov = annotate_provenance(merged)
        assert prov["missing_authors"] is True
        assert prov["missing_doi"] is True
        assert "missing_title" not in prov
        assert prov["suspicious_authors"] == ["A and B and C"]
        assert "suspicious_authors" not in merged  # moved, not copied


class TestExtractMetadataFromPage1:
    def test_reads_nested_and_flat_shapes(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        nested = ContentPage(page_number=1, content="x", metadata={
            "metadata": {"title": "T", "authors": ["A"], "doi": "10.1/x",
                         "publication_year": 2006, "journal_name": "J"}})
        out = extract_metadata_from_page1(nested, pdf, logger=LOG)
        assert out["title"] == "T" and out["year"] == 2006 and out["journal"] == "J"
        assert len(out["source_hash"]) == 64  # sha256 of the file

        flat = ContentPage(page_number=1, content="x",
                           metadata={"title": "T2", "authors": ["B"]})
        out2 = extract_metadata_from_page1(flat, pdf, logger=LOG)
        assert out2["title"] == "T2"


class TestEnrichMetadata:
    def test_ledger_cache_hit_short_circuits(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")

        class FakeLedger:
            def get_record(self, document_id):
                class R:
                    @staticmethod
                    def model_dump():
                        return {"title": "Cached", "doi": "10.1/cached"}
                return R()

        out = enrich_metadata({"title": "Fresh"}, pdf, ledger=FakeLedger(), logger=LOG)
        assert out["title"] == "Cached"

    def test_no_services_still_merges_and_annotates(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        out = enrich_metadata({"title": "A perfectly reasonable article title",
                               "authors": ["A. One"],
                               "doi": "10.1016/j.cellbi.2005.09.009", "year": 2006},
                              pdf, title_page_text="A perfectly reasonable article title",
                              logger=LOG)
        assert out["doi"] == "10.1016/j.cellbi.2005.09.009"
        assert out["provenance"]["validation_source"] == "unknown"
        # An empty suspicious list stays behind (V5 behaviour); it must just
        # never carry actual suspects after annotation moved them.
        assert not out.get("suspicious_authors")


# ---------------------------------------------------------------------------
# body_pipeline
# ---------------------------------------------------------------------------

def _pipeline(extraction=None, detector=None, classifier=None, tmp_path=None):
    class NoDetector:
        name = "none"
        supports_figures = False

        def detect(self, page_image, page=None):
            return []

    return BodyPipeline(
        extraction_provider=extraction or FakeProvider(),
        extraction_model="glm-ocr",
        layout_detector=detector or NoDetector(),
        resident_models=["glm-ocr"],
        page_classifier=classifier,
        router=None,
        extraction_dpi=150,
        coverage_threshold=0.90,
        table_crop_prompt="read table",
        caption_read_prompt="read page",
        figures_base_dir=(tmp_path or "run_outputs") / "figures" if tmp_path else None
                         or "run_outputs/figures",
        figures_path_root=(tmp_path or "run_outputs"),
        logger=LOG,
    )


class TestBodyPipelineConfig:
    def test_from_config_disabled_without_local_server(self):
        config = {"vlm_provider": {"local_server": {"enabled": False}},
                  "profile_registry": {"extraction": "glm-ocr"}}
        assert BodyPipeline.from_config(config, factory=None, logger=LOG) is None

    def test_from_config_disabled_without_extraction_role(self):
        config = {"vlm_provider": {"local_server": {"enabled": True}},
                  "profile_registry": {"default": "olmocr-2-7b"}}
        assert BodyPipeline.from_config(config, factory=None, logger=LOG) is None


class TestTableCropOutputCap:
    """A dense table outgrows a page read's output cap (t11: 55 rows cut at 41)."""

    class RecordingProvider(FakeProvider):
        def __init__(self):
            super().__init__(crop_text="<table></table>")
            self.max_tokens_seen = []

        def read_crop(self, crop_image, prompt, max_tokens=None, image_first=None):
            self.max_tokens_seen.append(max_tokens)
            return self._crop_text

    def test_table_crop_reads_use_the_configured_cap(self, tmp_path):
        provider = self.RecordingProvider()
        pipeline = _pipeline(extraction=provider, tmp_path=tmp_path)
        pipeline.table_crop_max_tokens = 8192
        pipeline._read_table_crop(Image.new("RGB", (10, 10)))
        assert provider.max_tokens_seen == [8192]

    def test_no_cap_configured_keeps_the_model_default(self, tmp_path):
        provider = self.RecordingProvider()
        _pipeline(extraction=provider, tmp_path=tmp_path)._read_table_crop(Image.new("RGB", (10, 10)))
        assert provider.max_tokens_seen == [None]

    def test_the_live_glm_profile_sets_a_table_cap_above_a_page_read(self):
        import yaml
        profile = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" /
                                  "vlm_profiles" / "glm-ocr.yaml").read_text(encoding="utf-8"))
        cap = profile["app_config"]["table_crop_max_tokens"]
        assert cap > profile["api_parameters"]["max_tokens"]


class TestReadBodyPage:
    def test_text_only_page_never_calls_the_model(self, tmp_path):
        page = FakePage(words=word_row(["Pure", "prose", "page"], y=100.0))

        class TextOnlyClassifier:
            def classify(self, page, page_idx, text_stats=None):
                return PageType.TEXT_ONLY, {"visual_regions": []}

        extraction = FakeProvider()
        pipe = _pipeline(extraction=extraction, classifier=TextOnlyClassifier(),
                         tmp_path=tmp_path)
        result = pipe.read_body_page(FakeCache({2: page}), 2)
        assert result.metadata["processing_method"] == "pymupdf"
        assert result.quality_score == 0.95
        assert "Pure prose page" in result.content
        assert extraction.calls == []  # the whole point: no VLM call

    def test_text_only_page_records_its_page_type(self, tmp_path):
        """The free path used to drop page_type, so 60% of pages in a run
        could not say how they were produced. Retrieval quality differs by
        route, so anything reading this output needs to be able to tell."""
        page = FakePage(words=word_row(["Pure", "prose", "page"], y=100.0))

        class TextOnlyClassifier:
            def classify(self, page, page_idx, text_stats=None):
                return PageType.TEXT_ONLY, {"visual_regions": []}

        pipe = _pipeline(extraction=FakeProvider(),
                         classifier=TextOnlyClassifier(), tmp_path=tmp_path)
        result = pipe.read_body_page(FakeCache({2: page}), 2)
        assert result.metadata["page_type"] == PageType.TEXT_ONLY.value

    def test_text_layer_content_is_normalized(self, tmp_path):
        """A line-break hyphen stored as a control character makes the word
        unfindable by any search. The free path is where raw harvested text
        becomes stored content, so it is repaired here."""
        page = FakePage(words=word_row(["collagen", "cross\x02linking"], y=100.0))

        class TextOnlyClassifier:
            def classify(self, page, page_idx, text_stats=None):
                return PageType.TEXT_ONLY, {"visual_regions": []}

        pipe = _pipeline(extraction=FakeProvider(),
                         classifier=TextOnlyClassifier(), tmp_path=tmp_path)
        result = pipe.read_body_page(FakeCache({2: page}), 2)
        assert "crosslinking" in result.content
        assert "\x02" not in result.content

    def test_full_read_with_good_coverage_skips_detection(self, tmp_path):
        words = word_row(["alpha", "beta", "gamma", "delta"], y=100.0)
        page = FakePage(words=words)
        extraction = FakeProvider([{"success": True,
                                    "content": "alpha beta gamma delta",
                                    "metadata": {}, "quality_score": 0.8}])

        class CountingDetector:
            name = "counting"
            supports_figures = True

            def __init__(self):
                self.calls = 0

            def detect(self, page_image, page=None):
                self.calls += 1
                return []

        detector = CountingDetector()
        pipe = _pipeline(extraction=extraction, detector=detector, tmp_path=tmp_path)
        result = pipe.read_body_page(FakeCache({0: page}), 0)
        assert result.metadata["processing_method"] == "glm_fullpage"
        assert result.metadata["tables_recovered"] == 0
        assert detector.calls == 0  # nothing needed detection

    def test_dropped_table_triggers_detection_and_repair(self, tmp_path):
        # Text layer has a table's worth of words; the model returned only
        # prose. The gate needs >= 30 reference words and >= 100 chars of
        # text layer before it will judge, so build a properly wordy page.
        # The gate tokenizes alphabetic words only and needs >= 30 of them.
        prose = [f"prose{chr(97 + i)}" for i in range(24)]
        table = ["sample", "mean", "deviation", "units", "notes",
                 "collagen", "hydration", "signal", "water", "tendon",
                 "bone", "cartilage", "relaxation", "compartment", "fraction"]
        words = (word_row(prose[:12], y=80.0) + word_row(prose[12:], y=100.0)
                 + word_row(table[:5], y=300.0)
                 + word_row(table[5:10], y=320.0)
                 + word_row(table[10:], y=340.0))
        page = FakePage(words=words)
        extraction = FakeProvider(
            [{"success": True, "content": " ".join(prose),
              "metadata": {}, "quality_score": 0.6}],
            crop_text="| sample | mean |\n| collagen | hydration |")

        class TableDetector:
            name = "stub"
            supports_figures = False

            def detect(self, page_image, page=None):
                return [Detection(label=LABEL_TABLE, box_px=(50, 250, 900, 500))]

        class InkedCache(FakeCache):
            """The rendered page must have ink where the table is, or the
            repair's snap-to-whitespace collapses the detection box."""

            def extraction_image(self, index, extraction_dpi):
                image = super().extraction_image(index, extraction_dpi)
                for x in range(60, 890):
                    for y in range(260, 490):
                        image.putpixel((x, y), (0, 0, 0))
                return image

        pipe = _pipeline(extraction=extraction, detector=TableDetector(),
                         tmp_path=tmp_path)
        result = pipe.read_body_page(InkedCache({0: page}), 0)
        assert result.metadata["processing_method"] == "glm_repair"
        assert result.metadata["tables_recovered"] == 1
        assert "| sample | mean |" in result.content
        assert result.quality_score == 0.85  # repair floor

    def test_page_exception_yields_failed_page_not_crash(self, tmp_path):
        class ExplodingCache(FakeCache):
            def page(self, index):
                raise RuntimeError("corrupt page")

        pipe = _pipeline(tmp_path=tmp_path)
        pages = pipe.process_body_pages(ExplodingCache({0: None}, pdf_path="d.pdf"), [0])
        assert len(pages) == 1
        assert pages[0].quality_score == 0.0
        assert pages[0].metadata["processing_method"] == "glm_failed"

    def test_a_failed_page_keeps_its_text_layer(self, tmp_path):
        """D9: PMC10904388 p4 raised MemoryError in the classifier and was
        written EMPTY, losing both text columns. The text layer is free and
        intact, so the words survive; the page stays marked as failed (D1)."""
        class RunsOutOfMemory:
            def classify(self, page, page_idx, text_stats=None):
                raise MemoryError()

        page = FakePage(words=word_row(["vOTU", "taxonomy", "and", "function"], y=100.0))
        pipe = _pipeline(classifier=RunsOutOfMemory(), tmp_path=tmp_path)
        result = pipe.process_body_pages(FakeCache({0: page}, pdf_path="d.pdf"), [0])[0]

        assert "vOTU taxonomy" in result.content
        assert result.metadata["processing_method"] == "glm_failed"
        assert result.metadata["fallback"] == "text_layer"
        assert result.metadata["error"] == "MemoryError"     # never an empty message


class TestEveryBodyPageRecordsItsTime:
    """
    Each body page carries `page_time` (seconds), so cost can be reported per
    KIND of page. Per document it misleads: a short paper's fixed start-up
    cost made an 8-page article look slower per page than a 25-page one.
    """

    class _TextOnly:
        def classify(self, page, page_idx, text_stats=None):
            return PageType.TEXT_ONLY, {"visual_regions": []}

    def test_text_layer_page_and_model_page_both_timed(self, tmp_path):
        pages = {0: FakePage(words=word_row(["plain", "prose"], y=100.0)),
                 1: FakePage(words=word_row(["needs", "model"], y=100.0))}
        text_pipe = _pipeline(classifier=self._TextOnly(), tmp_path=tmp_path)
        model_pipe = _pipeline(tmp_path=tmp_path)  # no classifier -> model read

        text_page = text_pipe.process_body_pages(FakeCache(pages, pdf_path="d.pdf"), [0])[0]
        model_page = model_pipe.process_body_pages(FakeCache(pages, pdf_path="d.pdf"), [1])[0]

        assert text_page.metadata["processing_method"] == "pymupdf"
        assert model_page.metadata["processing_method"] != "pymupdf"
        for page in (text_page, model_page):
            assert isinstance(page.metadata["page_time"], float)
            assert page.metadata["page_time"] >= 0.0

    def test_failed_page_is_timed_too(self, tmp_path):
        class ExplodingCache(FakeCache):
            def page(self, index):
                raise RuntimeError("corrupt page")

        pipe = _pipeline(tmp_path=tmp_path)
        page = pipe.process_body_pages(ExplodingCache({0: None}, pdf_path="d.pdf"), [0])[0]
        assert page.metadata["page_time"] >= 0.0


class TestTheFreePathMustBeEarned:
    """
    A page emptied out by the text-density filter does not go free unchecked.

    The filter discards regions that are mostly covered by words, because a
    reference list's link rectangles cluster into something figure-shaped.
    That is a judgement about typesetting: a publisher who sets a sequence
    alignment as selectable text inside a figure would measure the same way.
    If that page then falls to TEXT_ONLY it is never read, never detected,
    and never escalated — the figure is gone and nothing says so.

    So the CPU detector gets one look (~600ms, no GPU) at exactly the pages
    the filter emptied, and only those.
    """

    def _classifier(self, discarded):
        class FilteredClassifier:
            def classify(self, page, page_idx, text_stats=None):
                return PageType.TEXT_ONLY, {"visual_regions": [],
                                            "text_dense_regions": discarded}
        return FilteredClassifier()

    def _figure_detector(self, label="figure"):
        class FigureDetector:
            name = "stub"
            supports_figures = True

            def __init__(self):
                self.calls = 0

            def detect(self, page_image, page=None):
                self.calls += 1
                return [Detection(label=label, box_px=(50, 50, 900, 700))]
        return FigureDetector()

    def _page(self):
        return FakePage(words=word_row(["Sequence", "alignment", "figure"], y=100.0))

    def test_a_filtered_page_the_detector_backs_up_is_read_properly(self, tmp_path):
        """The gap this closes: the figure survives instead of vanishing."""
        extraction = FakeProvider([{"success": True, "content": "read text",
                                    "metadata": {}, "quality_score": 0.8}])
        detector = self._figure_detector()
        pipe = _pipeline(extraction=extraction, detector=detector,
                         classifier=self._classifier([{"bbox": (0, 0, 10, 10)}]),
                         tmp_path=tmp_path)

        result = pipe.read_body_page(FakeCache({3: self._page()}), 3)

        assert result.metadata["processing_method"] != "pymupdf"
        assert extraction.calls, "the page must actually be read"
        assert detector.calls == 1, "detected once, and the result reused"

    def test_the_detector_agreeing_leaves_the_page_free(self, tmp_path):
        """The common case — it really was a references page. Still free."""
        class EmptyDetector:
            name = "stub"
            supports_figures = True

            def __init__(self):
                self.calls = 0

            def detect(self, page_image, page=None):
                self.calls += 1
                return []

        extraction = FakeProvider()
        detector = EmptyDetector()
        pipe = _pipeline(extraction=extraction, detector=detector,
                         classifier=self._classifier([{"bbox": (0, 0, 10, 10)}]),
                         tmp_path=tmp_path)

        result = pipe.read_body_page(FakeCache({3: self._page()}), 3)

        assert result.metadata["processing_method"] == "pymupdf"
        assert extraction.calls == []
        assert detector.calls == 1

    def test_a_page_that_filtered_nothing_is_never_checked(self, tmp_path):
        """The check costs CPU, so it fires only where the filter acted."""
        detector = self._figure_detector()
        pipe = _pipeline(detector=detector, classifier=self._classifier([]),
                         tmp_path=tmp_path)

        result = pipe.read_body_page(FakeCache({3: self._page()}), 3)

        assert result.metadata["processing_method"] == "pymupdf"
        assert detector.calls == 0

    def test_a_detected_table_also_counts_as_content_worth_reading(self, tmp_path):
        extraction = FakeProvider([{"success": True, "content": "read text",
                                    "metadata": {}, "quality_score": 0.8}])
        pipe = _pipeline(extraction=extraction,
                         detector=self._figure_detector(label=LABEL_TABLE),
                         classifier=self._classifier([{"bbox": (0, 0, 10, 10)}]),
                         tmp_path=tmp_path)

        result = pipe.read_body_page(FakeCache({3: self._page()}), 3)
        assert result.metadata["processing_method"] != "pymupdf"

    def test_without_a_figure_capable_detector_the_page_stays_free(self, tmp_path):
        """
        Nothing to check with — and the expensive path could not produce a
        figure either, so paying for it would buy nothing.
        """
        extraction = FakeProvider()
        pipe = _pipeline(extraction=extraction,
                         classifier=self._classifier([{"bbox": (0, 0, 10, 10)}]),
                         tmp_path=tmp_path)     # _pipeline's default: no figures

        result = pipe.read_body_page(FakeCache({3: self._page()}), 3)
        assert result.metadata["processing_method"] == "pymupdf"
        assert extraction.calls == []

    def test_a_detector_failure_reads_the_page_rather_than_assuming(self, tmp_path):
        """Doubt costs seconds; a lost figure is permanent and invisible."""
        class BrokenDetector:
            name = "stub"
            supports_figures = True

            def detect(self, page_image, page=None):
                raise RuntimeError("onnx session died")

        extraction = FakeProvider([{"success": True, "content": "read text",
                                    "metadata": {}, "quality_score": 0.8}])
        pipe = _pipeline(extraction=extraction, detector=BrokenDetector(),
                         classifier=self._classifier([{"bbox": (0, 0, 10, 10)}]),
                         tmp_path=tmp_path)

        result = pipe.read_body_page(FakeCache({3: self._page()}), 3)
        assert result.metadata["processing_method"] != "pymupdf"
        assert extraction.calls, "the page must be read, not assumed empty"

    def test_a_visual_page_is_unaffected_by_any_of_this(self, tmp_path):
        """A page with regions standing takes the ordinary path, as before."""
        class VisualClassifier:
            def classify(self, page, page_idx, text_stats=None):
                return PageType.SIMPLE_VISUAL, {
                    "visual_regions": [{"bbox": (0, 0, 100, 100)}],
                    "text_dense_regions": [{"bbox": (0, 0, 10, 10)}]}

        extraction = FakeProvider([{"success": True, "content": "read text",
                                    "metadata": {}, "quality_score": 0.8}])
        detector = self._figure_detector()
        pipe = _pipeline(extraction=extraction, detector=detector,
                         classifier=VisualClassifier(), tmp_path=tmp_path)

        pipe.read_body_page(FakeCache({3: self._page()}), 3)
        assert detector.calls == 1      # the ordinary pass, not an extra one


# ---------------------------------------------------------------------------
# cloud_pipeline
# ---------------------------------------------------------------------------

class TestSplitMetadataBlock:
    def test_fenced_json_is_extracted_and_removed(self):
        content = ('```json\n{"metadata": {"title": "T", "authors": [], "doi": null, '
                   '"journal_name": "J", "publication_year": 2006}}\n```\n# Page text')
        remaining, metadata = _split_metadata_block(content, LOG)
        assert metadata["title"] == "T"
        assert remaining == "# Page text"

    def test_bare_json_at_start_is_extracted(self):
        content = '{"title": "T2"}  \nBody'
        remaining, metadata = _split_metadata_block(content, LOG)
        assert metadata["title"] == "T2"
        assert remaining == "Body"

    def test_invalid_json_leaves_content_untouched(self):
        content = "```json\n{not json}\n```\nBody"
        remaining, metadata = _split_metadata_block(content, LOG)
        assert metadata == {}
        assert remaining == content

    def test_no_block_is_a_noop(self):
        remaining, metadata = _split_metadata_block("Just text.", LOG)
        assert remaining == "Just text." and metadata == {}


class TestCloudLiteRouter:
    def test_dense_text_page_skips_the_cloud(self, tmp_path, monkeypatch):
        page = FakePage(words=word_row(["Dense", "prose"], y=100.0))

        class DenseClassifier:
            def classify(self, page, page_idx, text_stats=None):
                return PageType.TEXT_ONLY, {"text_density": 0.95, "visual_count": 0,
                                            "equation_regions": []}

        provider = FakeProvider()

        class StubLoader:
            def task_definition(self, name):
                return {"task_name": name, "prompt": "extract"}

        pages = process_body_pages_cloud(
            FakeCache({1: page}), [1], provider, StubLoader(), resolution=1024,
            page_classifier=DenseClassifier(), logger=LOG)
        assert pages[0].metadata["processing_method"] == "pymupdf"
        assert provider.calls == []

    def test_visual_page_goes_to_the_cloud(self, tmp_path):
        page = FakePage(words=word_row(["Figure", "heavy"], y=100.0))

        class VisualClassifier:
            def classify(self, page, page_idx, text_stats=None):
                return PageType.SIMPLE_VISUAL, {"text_density": 0.4, "visual_count": 2,
                                                "equation_regions": []}

        provider = FakeProvider([{"success": True, "content": "cloud read",
                                  "metadata": {}, "quality_score": 0.9}])

        class StubLoader:
            def task_definition(self, name):
                return {"task_name": name, "prompt": "extract"}

        pages = process_body_pages_cloud(
            FakeCache({1: page}), [1], provider, StubLoader(), resolution=1024,
            page_classifier=VisualClassifier(), logger=LOG)
        assert pages[0].metadata["processing_method"] == "gemini_single_pass"
        assert pages[0].content == "cloud read"
