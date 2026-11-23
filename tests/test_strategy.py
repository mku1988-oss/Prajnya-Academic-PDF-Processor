#!/usr/bin/env python3
"""
Tests for the assembled strategy and the V6 entrypoint's first pass.

Construction runs against the REAL shipped config tree and the real ONNX
layout model (skipped when absent) — this is the wiring test: every registry
reference resolves, every profile loads, the offline pipeline comes up. The
process() tests stub the model-calling functions and verify the assembly
V5's `process()` did: metadata flows title -> extraction -> enrichment ->
sanitize -> ExtractedMetadata, and a failure still yields a well-formed
result.
"""

import logging
from pathlib import Path

import pytest

from src.processing.models import ContentPage, DocumentProfile, DocumentType
from tests.local_samples import BENCHMARK_PDF as LOCAL_BENCHMARK_PDF

CONFIG_ROOT = Path(__file__).parent.parent / "config"
ONNX_MODEL = Path(__file__).parent.parent / "models" / "layout" / "pp-doclayoutv3.onnx"
BENCHMARK_PDF = LOCAL_BENCHMARK_PDF

LOG = logging.getLogger("test")

needs_model = pytest.mark.skipif(not ONNX_MODEL.is_file(),
                                 reason="ONNX layout model not fetched")
needs_benchmark = pytest.mark.skipif(not BENCHMARK_PDF.is_file(),
                                     reason="benchmark PDF not available")


@pytest.fixture()
def strategy(tmp_path, monkeypatch):
    """A fully-wired strategy with persistence redirected into tmp."""
    monkeypatch.setenv("PRAJNYA_LEDGER_PATH", str(tmp_path / "ledger.csv"))
    monkeypatch.setenv("PRAJNYA_CACHE_PATH", str(tmp_path / "zotero_cache"))
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    from src.processing.strategies.research_article.strategy import (
        ResearchArticleStrategy,
    )
    return ResearchArticleStrategy(CONFIG_ROOT, logger=LOG)


def _profile(pdf_path="paper.pdf", page_count=8) -> DocumentProfile:
    return DocumentProfile(pdf_path=pdf_path, page_count=page_count, file_size=1,
                           document_type=DocumentType.RESEARCH_ARTICLE)


@needs_model
class TestConstruction:
    def test_wires_the_offline_pipeline_from_the_real_config(self, strategy):
        assert strategy.use_cloud_pipeline is False
        assert strategy.body_pipeline is not None
        assert strategy.body_pipeline.extraction_model == "glm-ocr"
        assert strategy.body_pipeline.layout_detector.name == "onnx"
        # ONNX detector: only the extraction model needs GPU residency.
        assert strategy.body_pipeline.resident_models == ["glm-ocr"]

    def test_page_classifier_enabled_by_the_title_profile(self, strategy):
        assert strategy.page_classifier is not None

    def test_title_page_settings_resolved(self, strategy):
        assert strategy.title_resolution == 1288
        assert strategy.title_shot1_task == "raw_ocr_extraction"
        assert strategy.title_shot2_task == "structured_analysis_title_page"

    def test_unset_zotero_env_degrades_not_crashes(self, strategy):
        # ZOTERO_LIBRARY_ID unset -> V6 interpolation gives None -> the
        # service's guard raises -> the strategy degrades loudly to None.
        assert strategy.zotero_service is None
        # The strategy only READS the ledger, so one that does not exist yet
        # is not created as a side effect (the runner owns every write).
        assert strategy.ledger is None

    def test_an_existing_ledger_is_opened_at_every_level(self, tmp_path, monkeypatch):
        """The ledger used to be opened only when Zotero was switched on."""
        from src.processing.ledger.csv_backend import CsvLedgerBackend
        from src.processing.strategies.research_article.strategy import (
            ResearchArticleStrategy,
        )
        ledger_path = tmp_path / "ledger.csv"
        CsvLedgerBackend(file_path=str(ledger_path))        # creates the file
        monkeypatch.setenv("PRAJNYA_LEDGER_PATH", str(ledger_path))
        monkeypatch.setenv("PRAJNYA_CACHE_PATH", str(tmp_path / "zotero_cache"))

        local = ResearchArticleStrategy(CONFIG_ROOT, logger=LOG, citation_level="local")
        assert local.ledger is not None
        assert local.citation_level == "local"
        assert local.citation_level_set_by == "cli"

    def test_at_local_the_zotero_service_is_never_built(self, tmp_path, monkeypatch):
        """Building it contacts Zotero; `local` promises nothing leaves the machine."""
        import src.processing.strategies.research_article.strategy as strategy_module

        def refuse(*args, **kwargs):
            raise AssertionError("ZoteroService was constructed at level local")

        monkeypatch.setattr("src.processing.citations.services.ZoteroService", refuse)
        monkeypatch.setenv("PRAJNYA_LEDGER_PATH", str(tmp_path / "ledger.csv"))
        local = strategy_module.ResearchArticleStrategy(CONFIG_ROOT, logger=LOG,
                                                        citation_level="local")
        assert local.zotero_service is None

    def test_a_level_that_is_not_built_yet_is_refused_before_any_work(self, tmp_path,
                                                                      monkeypatch):
        """`online` must not run as something else while the file says `online`."""
        from src.processing.strategies.research_article.strategy import (
            ResearchArticleStrategy,
        )
        monkeypatch.setenv("PRAJNYA_LEDGER_PATH", str(tmp_path / "ledger.csv"))
        monkeypatch.setattr("src.processing.citations.level.BUILT_LEVELS", ("local", "zotero"))
        with pytest.raises(NotImplementedError, match="not built yet"):
            ResearchArticleStrategy(CONFIG_ROOT, logger=LOG, citation_level="online")

    def test_title_detection_config_resolves_from_registry(self, strategy):
        config = strategy.get_title_detection_config()
        assert config == {"profile": "olmocr-2-7b", "task": "title_page_detection"}


@needs_model
class TestProcessAssembly:
    def _stub_pipelines(self, strategy, monkeypatch, title_page, body_pages):
        import src.processing.strategies.research_article.strategy as strategy_module
        monkeypatch.setattr(strategy_module, "process_title_page",
                            lambda *a, **k: title_page)
        monkeypatch.setattr(strategy_module.DocumentCache, "__init__",
                            lambda self, backend, pdf_path, logger=None: None)
        monkeypatch.setattr(strategy_module.DocumentCache, "close", lambda self: None)
        monkeypatch.setattr(strategy_module.DocumentCache, "page",
                            lambda self, index: (_ for _ in ()).throw(RuntimeError("no file")))
        monkeypatch.setattr(strategy, "body_pipeline", body_pages)
        monkeypatch.setattr(strategy, "zotero_service", None)
        monkeypatch.setattr(strategy, "ledger", None)

    def test_metadata_flows_to_the_result(self, strategy, monkeypatch, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        title_page = ContentPage(page_number=1, content="Title text", metadata={
            "title": "An NMR method", "authors": ["A. One", "B. Two"],
            "doi": "10.1016/j.cellbi.2005.09.009",
            "publication_year": 2006, "journal_name": "Cell Biology International"})

        class StubBody:
            def process_body_pages(self, cache, indices):
                return [ContentPage(page_number=i + 1, content=f"body {i + 1}",
                                    metadata={}) for i in indices]

        self._stub_pipelines(strategy, monkeypatch, title_page, StubBody())
        profile = strategy.process(_profile(pdf, page_count=3))

        result = profile.processing_result
        assert result is not None and result.errors == []
        assert result.metadata.title == "An NMR method"
        assert result.metadata.doi == "10.1016/j.cellbi.2005.09.009"
        assert result.metadata.year == 2006
        assert [p.page_number for p in result.pages] == [1, 2, 3]

    def test_pages_override_processes_only_those_bodies(self, strategy, monkeypatch,
                                                        tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        title_page = ContentPage(page_number=1, content="T", metadata={"title": "T"})
        seen = []

        class StubBody:
            def process_body_pages(self, cache, indices):
                seen.extend(indices)
                return []

        self._stub_pipelines(strategy, monkeypatch, title_page, StubBody())
        strategy.process(_profile(pdf, page_count=8), test_pages_override=[1, 4, 7])
        assert seen == [3, 6]  # 1-based 4 and 7; page 1 is the title

    def test_failure_still_yields_wellformed_result(self, strategy, monkeypatch,
                                                    tmp_path):
        import src.processing.strategies.research_article.strategy as strategy_module
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")

        def explode(*a, **k):
            raise RuntimeError("title page exploded")

        monkeypatch.setattr(strategy_module, "process_title_page", explode)
        monkeypatch.setattr(strategy_module.DocumentCache, "__init__",
                            lambda self, backend, pdf_path, logger=None: None)
        monkeypatch.setattr(strategy_module.DocumentCache, "close", lambda self: None)
        # Without this, reserving the GPU for the title page asks the REAL
        # router to load the title model into any running llama-server
        # (found 2026-11-22: the suite left olmocr-2-7b resident, which
        # skews the next timing; see "VRAM residency" in CLAUDE.md).
        monkeypatch.setattr(strategy, "_reserve_gpu_for", lambda model: None)

        profile = strategy.process(_profile(pdf, page_count=2))
        result = profile.processing_result
        assert result is not None
        assert result.pages == []
        assert any("title page exploded" in e for e in result.errors)


@needs_benchmark
class TestEntrypointFirstPass:
    def test_first_pass_finds_the_benchmark_doi(self):
        from run_prajnya import build_document_profile
        from src.processing.pdf import get_backend

        profile = build_document_profile(BENCHMARK_PDF, get_backend())
        assert profile.page_count == 8
        assert profile.doi == "10.1016/j.cellbi.2005.09.009"
        assert profile.doi_confidence >= 0.85  # clears the trust threshold
        assert profile.page_1_text and "NMR" in profile.page_1_text


class TestGpuReservation:
    """
    The title page must own the GPU alone before it runs.

    The 8GB card cannot hold the title model (~5.5GB) and the body model
    (~3GB) together. The llama-server router auto-loads on request but does
    NOT evict first — it only sheds models once --models-max is exceeded, and
    it counts models rather than measuring VRAM. So a model left resident by
    an earlier document makes the title model spill to system RAM.

    Measured cost of getting this wrong, same PDF and page, only residency
    differing: shot 1 37.2s -> 71.8s, shot 2 11.2s -> 43.8s (2.29x overall).
    These tests pin the call, because the failure is invisible in output —
    the results are byte-identical, only slower.
    """

    class _SpyRouter:
        def __init__(self, ok=True):
            self.calls = []
            self._ok = ok

        def set_resident(self, models, timeout=180):
            self.calls.append(list(models))
            return self._ok

    class _StubBody:
        def __init__(self, router):
            self.router = router

        def process_body_pages(self, cache, indices):
            return []

    def _run(self, strategy, monkeypatch, tmp_path, router):
        import src.processing.strategies.research_article.strategy as strategy_module
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        monkeypatch.setattr(strategy_module, "process_title_page",
                            lambda *a, **k: ContentPage(page_number=1, content="T",
                                                        metadata={"title": "T"}))
        monkeypatch.setattr(strategy_module.DocumentCache, "__init__",
                            lambda self, backend, pdf_path, logger=None: None)
        monkeypatch.setattr(strategy_module.DocumentCache, "close", lambda self: None)
        monkeypatch.setattr(strategy_module.DocumentCache, "page",
                            lambda self, index: (_ for _ in ()).throw(RuntimeError("no file")))
        monkeypatch.setattr(strategy, "body_pipeline", self._StubBody(router))
        monkeypatch.setattr(strategy, "zotero_service", None)
        monkeypatch.setattr(strategy, "ledger", None)
        strategy.process(_profile(pdf, page_count=2))

    @needs_model
    def test_title_model_is_made_exclusively_resident(self, strategy, monkeypatch,
                                                      tmp_path):
        router = self._SpyRouter()
        self._run(strategy, monkeypatch, tmp_path, router)
        # Exactly one model requested — never the title model alongside another.
        assert router.calls, "the title page never reserved the GPU"
        assert router.calls[0] == [strategy.title_model]
        assert len(router.calls[0]) == 1

    @needs_model
    def test_reservation_happens_before_the_title_page_runs(self, strategy,
                                                            monkeypatch, tmp_path):
        """Order is the whole point: reserving after the read saves nothing."""
        import src.processing.strategies.research_article.strategy as strategy_module
        order = []
        router = self._SpyRouter()

        def spy_set_resident(models, timeout=180):
            order.append(("reserve", list(models)))
            return True

        router.set_resident = spy_set_resident
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")

        def spy_title(*a, **k):
            order.append(("title_read", None))
            return ContentPage(page_number=1, content="T", metadata={"title": "T"})

        monkeypatch.setattr(strategy_module, "process_title_page", spy_title)
        monkeypatch.setattr(strategy_module.DocumentCache, "__init__",
                            lambda self, backend, pdf_path, logger=None: None)
        monkeypatch.setattr(strategy_module.DocumentCache, "close", lambda self: None)
        monkeypatch.setattr(strategy_module.DocumentCache, "page",
                            lambda self, index: (_ for _ in ()).throw(RuntimeError("x")))
        monkeypatch.setattr(strategy, "body_pipeline", self._StubBody(router))
        monkeypatch.setattr(strategy, "zotero_service", None)
        monkeypatch.setattr(strategy, "ledger", None)
        strategy.process(_profile(pdf, page_count=2))

        assert order[0][0] == "reserve"
        assert order[1][0] == "title_read"

    @needs_model
    def test_a_failed_reservation_does_not_abort_the_run(self, strategy, monkeypatch,
                                                         tmp_path):
        """
        Degrade loudly, not fatally. The request would auto-load the model
        anyway — just slower — so a router hiccup must not lose the document.
        """
        router = self._SpyRouter(ok=False)
        self._run(strategy, monkeypatch, tmp_path, router)
        assert router.calls == [[strategy.title_model]]

    @needs_model
    def test_no_router_is_a_no_op(self, strategy):
        """The cloud path has no local router; reserving must not explode."""
        strategy.body_pipeline = None
        strategy._reserve_gpu_for("olmocr-2-7b")  # must not raise
