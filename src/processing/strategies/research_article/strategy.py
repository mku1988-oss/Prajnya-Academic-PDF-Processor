#!/usr/bin/env python3
"""
The research-article strategy — the slim class that assembles the modules.

For biology researchers: this is the conductor. Every instrument lives in its
own module (title_page, body_pipeline, metadata_enrichment, figures,
rendering, provider_factory, cloud_pipeline); this class only decides which
of them plays, in what order, and collects the result into the
`DocumentProfile`.

Ported from V5's `ResearchArticleStrategy.__init__` + `process()`
(research_article_strategy.py L96-443, of 3096 lines total). What V5's
`process()` also carried — the 5-tier adaptive routing, the concurrency
machinery, the tier fallbacks — is on plan.md's delete list and is not here;
the two live paths are the GLM two-phase offline pipeline and the cloud
single-pass pipeline, selected exactly as V5 selected them (the title/default
profile's `model_type` for cloud, `local_server.enabled` +
`profile_registry.extraction` for offline).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...config.loader import ConfigLoader, resolve_reference
from ...ledger.csv_backend import CsvLedgerBackend
from ...ledger.semantic_ledger import SemanticLedger
from ...models import ContentPage, DocumentProfile, ExtractedMetadata, ProcessingResult
from ...paths import DEFAULT_LEDGER_PATH, DEFAULT_ZOTERO_CACHE_PATH
from ...pdf import get_backend
from ...pdf.base import PdfBackend
from ...utils.page_classifier import PageClassifier
from ...utils.zotero_service import ZoteroService
from .body_pipeline import DEFAULT_ONNX_MODEL_PATH, BodyPipeline
from .cloud_pipeline import process_body_pages_cloud, process_title_page_cloud
from .metadata_enrichment import (
    enrich_metadata,
    extract_metadata_from_page1,
    sanitize_bibliographic_metadata,
)
from .provider_factory import ProviderFactory
from .rendering import DocumentCache
from .title_page import process_title_page


class ResearchArticleStrategy:
    """
    Processes one research article end to end.

    Args:
        config_root: the `config/` directory (strategy YAML, profiles, tasks).
        pdf_backend: page access implementation; the permissive default when
            omitted.
        onnx_model_path: the CPU layout model (see models/README.md).
        logger: optional logger.
    """

    def __init__(
        self,
        config_root: Path | str,
        pdf_backend: Optional[PdfBackend] = None,
        onnx_model_path: Path | str = DEFAULT_ONNX_MODEL_PATH,
        logger: Optional[logging.Logger] = None,
        refresh_metadata: bool = False,
        output_root: Optional[Path] = None,
    ) -> None:
        self.logger = logger or logging.getLogger("ResearchArticleStrategy")
        # Where this run's figures and tables go. None means the repository's
        # own run_outputs/, which is what every run produced before `--out`
        # could move it. See `paths.anchor_output_path`.
        self.output_root = output_root
        # Set by `--force`: ignore the ledger's stored metadata and derive it
        # again, so a corrected extractor actually changes the answer.
        self.refresh_metadata = refresh_metadata
        self.loader = ConfigLoader(config_root, logger=self.logger)
        self.config = self.loader.strategy()
        self.backend = pdf_backend or get_backend()
        self.factory = ProviderFactory(self.loader, self.config, logger=self.logger)

        self._init_services()
        self._init_title_page()
        self._init_classifier_and_pipelines(onnx_model_path)

        self.logger.info("Research article strategy initialized with config-driven settings")

    def process(self, profile: DocumentProfile,
                test_pages_override: Optional[List[int]] = None) -> DocumentProfile:
        """
        Process a document, populating `profile.processing_result`.

        Args:
            profile: the document profile (path, page count, quick identifiers).
            test_pages_override: optional list of 1-based page numbers, used
                by the parity harness to run a subset.

        Returns:
            The same profile, its `processing_result` filled in. The strategy
            never persists to the ledger — the runner decides that, after
            review.
        """
        all_pages: List[ContentPage] = []
        bibliographic: Dict[str, Any] = {}
        errors: List[str] = []

        try:
            with DocumentCache(self.backend, profile.pdf_path,
                               logger=self.logger) as cache:
                if self.use_cloud_pipeline:
                    all_pages, bibliographic = self._run_cloud(
                        cache, profile, test_pages_override)
                else:
                    all_pages, bibliographic = self._run_local(
                        cache, profile, test_pages_override)
        except Exception as error:
            message = f"Strategy execution failed: {type(error).__name__} - {error}"
            self.logger.error(message, exc_info=True)
            errors.append(message)

        # This always runs — even a failed document gets a well-formed result.
        self.logger.info("Finalizing processing. Total pages processed: %d",
                         len(all_pages))
        bibliographic = sanitize_bibliographic_metadata(bibliographic)
        profile.processing_result = ProcessingResult(
            metadata=ExtractedMetadata(**bibliographic),
            pages=all_pages,
            errors=errors,
        )
        self.logger.info("Research article processing complete.")
        return profile

    def get_title_detection_config(self) -> Dict[str, str]:
        """
        The profile + task the orchestrator uses for title-page detection,
        resolved from the registries (never hardcoded).
        """
        return {
            "profile": resolve_reference("@profile_registry.title_detection", self.config),
            "task": resolve_reference("@task_registry.title_detection", self.config),
        }

    # --- helpers (after the main methods, per CLAUDE.md) ------------------

    def _init_services(self) -> None:
        """Ledger + Zotero, exactly as V5 gated them (zotero.enabled)."""
        self.ledger: Optional[SemanticLedger] = None
        self.zotero_service: Optional[ZoteroService] = None

        zotero_config = self.config.get("zotero", {})
        if zotero_config.get("enabled", False):
            self.logger.info("Zotero integration is enabled. Initializing services.")
            # Both defaults are repository-anchored (see `paths.py`): they were
            # CWD-relative, so a run from another directory quietly built its
            # own second ledger and forgot everything the first one knew.
            ledger_path = Path(os.getenv("SARA_LEDGER_PATH", DEFAULT_LEDGER_PATH))
            self.ledger = CsvLedgerBackend(file_path=str(ledger_path.resolve()))
            try:
                cache_dir = Path(os.getenv("SARA_CACHE_PATH",
                                           DEFAULT_ZOTERO_CACHE_PATH))
                self.zotero_service = ZoteroService(config=zotero_config,
                                                    cache_dir=str(cache_dir.resolve()))
            except (ImportError, ValueError) as error:
                # ValueError is new with V6's real ${ENV} interpolation: an
                # unset ZOTERO_LIBRARY_ID reaches the service as None instead
                # of V5's truthy literal "${ZOTERO_LIBRARY_ID}", so the
                # service's own guard finally fires. Degrade loudly, not
                # fatally — a missing key must not end the document run.
                self.logger.error("ZoteroService unavailable (%s). "
                                  "Continuing without Zotero.", error)
                self.zotero_service = None
        else:
            self.logger.info("Zotero integration is disabled in the configuration.")

        self.doi_threshold = self.config["metadata"]["doi_confidence_threshold"]

    def _init_title_page(self) -> None:
        """Resolve the title-page profile, tasks, and resolution."""
        title_cfg = self.config["vlm_provider"]["title_page"]
        self.title_resolution = title_cfg.get("resolution_override", 1288)
        self.title_shot1_task = resolve_reference(title_cfg.get("shot1_task"), self.config)
        self.title_shot2_task = resolve_reference(title_cfg.get("shot2_task"), self.config)

        title_profile_name = resolve_reference(title_cfg.get("profile"), self.config)
        self.title_profile_data = self.loader.vlm_profile(title_profile_name)
        # Kept so the title page can claim the GPU alone — see _reserve_gpu_for.
        self.title_model = title_profile_name

        # Cloud pipeline detection, exactly V5's rule: the configured profile
        # names a hosted API.
        self.use_cloud_pipeline = self.title_profile_data.get("model_type") in (
            "cloud_api_gemini", "cloud_api_openai")
        if self.use_cloud_pipeline:
            self.logger.info("Cloud API profile detected: enabling the simplified "
                             "cloud pipeline")

        body_cfg = self.config["vlm_provider"].get("body_pages", {})
        self.body_resolution = body_cfg.get("resolution_override", 1024)

    def _init_classifier_and_pipelines(self, onnx_model_path: Path | str) -> None:
        """The page classifier and whichever pipeline the config selects."""
        hybrid = self.title_profile_data.get("hybrid_processing", {})
        self.page_classifier = (PageClassifier(self.title_profile_data)
                                if hybrid.get("enabled", False) else None)
        if self.page_classifier is None:
            self.logger.info("Hybrid page classification disabled in the profile")

        self.body_pipeline: Optional[BodyPipeline] = None
        self.cloud_provider = None

        if self.use_cloud_pipeline:
            built = self.factory.build(
                "@profile_registry.default", "@task_registry.legacy_ocr")
            self.cloud_provider = built.provider
            return

        self.body_pipeline = BodyPipeline.from_config(
            self.config, self.factory, page_classifier=self.page_classifier,
            onnx_model_path=onnx_model_path, logger=self.logger,
            output_root=self.output_root)
        if self.body_pipeline is None:
            raise ValueError(
                "The offline body pipeline is not configured "
                "(local_server.enabled + profile_registry.extraction), and the "
                "active profile is not a cloud API. V6 has no 5-tier fallback: "
                "fix the config rather than expecting a silent degrade.")

    def _body_page_indices(self, profile: DocumentProfile,
                           test_pages_override: Optional[List[int]]) -> List[int]:
        """0-indexed body pages: everything after page 1, or the override."""
        if test_pages_override:
            self.logger.info("Processing specific pages from test override: %s",
                             test_pages_override)
            return [p - 1 for p in test_pages_override if p > 1]
        return list(range(1, profile.page_count))

    def _title_referee_text(self, cache: DocumentCache, profile: DocumentProfile,
                            title_page: ContentPage) -> str:
        """
        The text-layer reading of the title page, used as merge referee.
        Falls back to the VLM's transcription when the page is scanned.
        """
        try:
            text = cache.page(profile.title_page_index).text() or ""
        except Exception:
            text = ""
        if not text.strip() and title_page:
            self.logger.info("Text layer empty (scanned doc?), using VLM extracted "
                             "text as referee.")
            text = title_page.content or ""
        return text

    def _enrich(self, cache: DocumentCache, profile: DocumentProfile,
                title_page: ContentPage, bibliographic: Dict[str, Any]) -> Dict[str, Any]:
        return enrich_metadata(
            vlm_metadata=bibliographic,
            pdf_path=profile.pdf_path,
            title_page_text=self._title_referee_text(cache, profile, title_page),
            profile=profile,
            ledger=self.ledger,
            zotero_service=self.zotero_service,
            doi_threshold=self.doi_threshold,
            use_ledger_cache=not self.refresh_metadata,
            logger=self.logger,
        )

    def _run_local(self, cache: DocumentCache, profile: DocumentProfile,
                   test_pages_override: Optional[List[int]]):
        """Title two-shot -> enrichment -> GLM two-phase body."""
        self._reserve_gpu_for(self.title_model)

        self.logger.info("Processing page 1/%d (title page metadata)...",
                         profile.page_count)
        title_page = process_title_page(
            cache, profile.title_page_index, self.factory, self.title_profile_data,
            self.title_shot1_task, self.title_shot2_task, self.title_resolution,
            logger=self.logger)
        pages = [title_page]

        bibliographic = extract_metadata_from_page1(title_page, profile.pdf_path,
                                                    logger=self.logger)
        bibliographic = self._enrich(cache, profile, title_page, bibliographic)

        body_indices = self._body_page_indices(profile, test_pages_override)
        if body_indices:
            self.logger.info("Body pages: GLM two-phase offline path")
            pages.extend(self.body_pipeline.process_body_pages(cache, body_indices))
        return pages, bibliographic

    def _reserve_gpu_for(self, model: Optional[str]) -> None:
        """
        Make `model` the ONLY model on the GPU before we call it.

        For biology researchers: the GPU has 8GB. The title-page model needs
        ~5.5GB and the body-page model ~3GB, so the two do not fit together.
        The llama-server router loads a model automatically when you ask for
        it, but it does NOT unload anything first — it only evicts once its
        `--models-max` count is exceeded, and it counts models rather than
        measuring VRAM. So a leftover model from an earlier document silently
        stays put, the title model half-spills into system RAM, and it decodes
        at roughly a third of its normal speed.

        Measured on the benchmark title page (same PDF, same page, same code,
        only residency differing):

            empty GPU              shot 1  37.2s   shot 2  11.2s
            body model left loaded shot 1  71.8s   shot 2  43.8s

        That is 2.29x on the title page for no reason at all. Asking for the
        model explicitly costs one HTTP call and makes the run reproducible:
        without it, a document's timing depends on which document ran before
        it. `set_resident` unloads everything else first, then polls until the
        weights are really on the GPU.

        Phase A of the body pipeline does the same thing for its own model, so
        the two never overlap. V6 never needs two models at once: with the ONNX
        layout detector, Phase A keeps exactly one.

        Args:
            model: the model id to make exclusively resident. No-op when
                falsy or when there is no local router (the cloud path).
        """
        router = self._router()
        if not router or not model:
            return
        self.logger.info("Reserving the GPU for '%s' (title page)", model)
        if not router.set_resident([model]):
            # Not fatal: the request would auto-load it anyway, just slower.
            self.logger.warning(
                "Could not make '%s' exclusively resident; the title page may "
                "run against a shared GPU and be markedly slower", model)

    def _router(self):
        """
        The body pipeline's router client, when running locally.

        Reused rather than rebuilt so there is one notion of what is resident.
        """
        pipeline = getattr(self, "body_pipeline", None)
        return getattr(pipeline, "router", None) if pipeline else None

    def _run_cloud(self, cache: DocumentCache, profile: DocumentProfile,
                   test_pages_override: Optional[List[int]]):
        """Cloud single-pass title -> enrichment -> lite-router body."""
        self.logger.info("Executing cloud pipeline...")
        title_page = process_title_page_cloud(
            cache, profile.title_page_index, self.cloud_provider, self.loader,
            self.title_resolution, logger=self.logger)
        pages = [title_page]

        generated = (title_page.metadata or {}).get("generated_metadata")
        if generated:
            bibliographic = dict(generated)
            # Normalize keys to what ExtractedMetadata / the ledger expect.
            if "journal_name" in bibliographic and "journal" not in bibliographic:
                bibliographic["journal"] = bibliographic.pop("journal_name")
            if "publication_year" in bibliographic and "year" not in bibliographic:
                bibliographic["year"] = bibliographic.pop("publication_year")
        else:
            bibliographic = extract_metadata_from_page1(title_page, profile.pdf_path,
                                                        logger=self.logger)
        bibliographic = self._enrich(cache, profile, title_page, bibliographic)

        body_indices = self._body_page_indices(profile, test_pages_override)
        if body_indices:
            pages.extend(process_body_pages_cloud(
                cache, body_indices, self.cloud_provider, self.loader,
                self.body_resolution, page_classifier=self.page_classifier,
                logger=self.logger))
        return pages, bibliographic
