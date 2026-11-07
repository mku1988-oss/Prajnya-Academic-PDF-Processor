#!/usr/bin/env python3
"""
Body-page pipeline — the GLM two-phase offline path, on the V6 seams.

For biology researchers: body pages carry the science — methods, results,
tables, figures. This pipeline reads each page as cheaply as correctness
allows: a pure-text page is read straight from the PDF's text layer (free);
any other page is read by the extraction model, then *verified* against the
text layer — if the model silently dropped a table (a known failure mode),
the layout detector finds it, the table is cropped, and the model re-reads
the crop, where it is accurate. Figures are cropped and caption-tagged from
the same detection pass.

Ported from V5's `_init_glm_twophase` (L1444), `_process_body_pages_glm`
(L1527), `_read_body_page_glm` (L1570), `_ground_page_if_needed` /
`_repair_page` (L1629-1680) and `_process_with_pymupdf` (L1820), with the
seams V6 introduced:

- Layout detection goes through `LayoutDetector` -> `List[Detection]`. The
  registry's `layout_detector` role selects the implementation: `onnx` is the
  CPU detector (Stage 2, no GPU); any profile name selects the VLM grounding
  fallback. Consumers (`table_repair`, `figure_extractor`) never see a raw
  grounding string.
- Page access goes through `PdfPage`/`DocumentCache`, never fitz.
- Model residency: only the models that are actually VLMs are loaded. With
  the ONNX detector, Phase A needs *one* resident model (the extraction
  reader) — V5 co-resided two.

Concurrency was investigated in V5 (asyncio + threads, and a read/ground
two-stage split) and gave no wall-time win on a single GPU — do not re-chase
it (see plan.md).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from ...config.loader import resolve_reference
from ...layout.base import LABEL_TABLE, Detection, LayoutDetector
from ...layout.vlm_detector import GroundingSpec, VlmGroundingDetector
from ...models import ContentPage
from ...events import emit
from ...paths import REPO_ROOT, anchor_output_path
from ...pdf.base import PdfPage
from ...utils.caption_overleaf import (
    NEIGHBOUR_OFFSETS,
    recover_overleaf_captions,
)
from ...utils.figure_label_pass import build_text_lines, find_label_lines
from ...utils.reference_pass import reference_signals
from ...utils.heading_pass import heading_candidates_for_page
from ...utils.content_drop_check import (
    CoverageResult,
    check_page_for_content_drop,
    is_scanned_page,
)
from ...utils.figure_extractor import append_figure_anchors
from ...utils.model_swap import RouterClient
from ...utils.page_classifier import PageClassifier, PageType, extract_text_stats
from ...utils.table_repair import repair_page_tables
from ...utils.text_normalization import normalize_harvested_text
from .figure_escalation import (
    FIGURE_LABELS,
    collect_suspect_pages,
    describe_escalation,
    page_needs_second_opinion,
)
from .figures import extract_page_figures, write_figure_manifest
from .tables import append_table_anchors, extract_page_tables, write_table_manifest
from .provider_factory import ProviderFactory
from .rendering import DocumentCache

#: Default location of the CPU layout model (see models/README.md).
#: The CPU layout detector's weights, anchored to the repository.
#:
#: This was CWD-relative, which meant a run launched from anywhere other than
#: `src/v6` could not find its own model and died with "layout model not
#: found" — the same defect as the figure/table output paths below, in the
#: other direction. A repository asset does not move when the user does.
DEFAULT_ONNX_MODEL_PATH = REPO_ROOT / "models" / "layout" / "pp-doclayoutv3.onnx"


class BodyPipeline:
    """
    The wired-up offline body path. Build with `from_config`, which returns
    None when the config does not enable it (no local server, or no
    `extraction` role) — that mirrors V5's strictly-opt-in behaviour.
    """

    def __init__(
        self,
        extraction_provider,             # VlmProvider: reads the pixels
        extraction_model: str,           # its model id (for the router)
        layout_detector: LayoutDetector,
        resident_models: List[str],      # what Phase A must keep on the GPU
        page_classifier: Optional[PageClassifier],
        router: Optional[RouterClient],
        extraction_dpi: int,
        coverage_threshold: float,
        table_crop_prompt: Optional[str],
        caption_read_prompt: Optional[str],
        figures_base_dir: Path,
        figures_path_root: Path,
        tables_base_dir: Optional[Path] = None,
        figure_escalation_detector: Optional[LayoutDetector] = None,
        figure_escalation_models: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        table_crop_max_tokens: Optional[int] = None,
    ) -> None:
        self.extraction_provider = extraction_provider
        # Output cap for table-crop reads; None keeps the model's own limit.
        # A dense table needs more than a page read does (see glm-ocr.yaml).
        self.table_crop_max_tokens = table_crop_max_tokens
        self.extraction_model = extraction_model
        self.layout_detector = layout_detector
        self.resident_models = resident_models
        self.page_classifier = page_classifier
        self.router = router
        self.extraction_dpi = extraction_dpi
        self.coverage_threshold = coverage_threshold
        self.table_crop_prompt = table_crop_prompt
        self.caption_read_prompt = caption_read_prompt
        self.figures_base_dir = figures_base_dir
        self.figures_path_root = figures_path_root
        # Tables sit beside figures under the same path_root, so both
        # manifests carry portable paths a RAG indexer can resolve alike.
        self.tables_base_dir = tables_base_dir or (figures_path_root / "tables")
        # The second opinion on figures: None disables it entirely, which is
        # the default until a profile names a grounding model for the role.
        self.figure_escalation_detector = figure_escalation_detector
        self.figure_escalation_models = figure_escalation_models or []
        self.logger = logger or logging.getLogger("BodyPipeline")

    # ------------------------------------------------------------------ #
    # construction from config
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        strategy_config: Dict[str, Any],
        factory: ProviderFactory,
        page_classifier: Optional[PageClassifier] = None,
        onnx_model_path: Path | str = DEFAULT_ONNX_MODEL_PATH,
        logger: Optional[logging.Logger] = None,
        output_root: Optional[Path] = None,
    ) -> Optional["BodyPipeline"]:
        """
        Wire the pipeline from the strategy config, or return None when the
        config does not opt in (needs local_server.enabled=true AND a
        profile_registry.extraction role).

        Args:
            output_root: where this run's figures and tables are written.
                None means `paths.DEFAULT_OUTPUT_ROOT` — the same absolute
                place a run from the repo root has always produced, so nothing
                moves unless `--out` says so.
        """
        log = logger or logging.getLogger("BodyPipeline")
        local_cfg = strategy_config.get("vlm_provider", {}).get("local_server", {})
        registry = strategy_config.get("profile_registry", {})
        if not local_cfg.get("enabled", False) or "extraction" not in registry:
            log.info("GLM two-phase offline path DISABLED "
                     "(needs local_server.enabled=true + profile_registry.extraction)")
            return None

        extraction_model = resolve_reference("@profile_registry.extraction", strategy_config)
        raw_ocr_task = resolve_reference("@task_registry.raw_ocr", strategy_config)

        built = factory.build(extraction_model, raw_ocr_task)
        profile_data = built.profile_data
        fixed_prompts = profile_data.get("fixed_prompts", {})
        extraction_dpi = profile_data.get("app_config", {}).get("target_dpi", 300)
        table_crop_max_tokens = profile_data.get("app_config", {}).get("table_crop_max_tokens")
        coverage_threshold = profile_data.get(
            "content_drop_check", {}).get("min_word_coverage", 0.90)

        # Table-crop re-read prompt: the model's table prompt, falling back to
        # its page-read prompt. Caption reads (scanned pages only) use the
        # page-read prompt.
        table_task = resolve_reference("@task_registry.table_desc", strategy_config)
        table_crop_prompt = fixed_prompts.get(table_task) or fixed_prompts.get(raw_ocr_task)
        caption_read_prompt = fixed_prompts.get(raw_ocr_task)

        detector, extra_resident = cls._build_layout_detector(
            strategy_config, factory, raw_ocr_task, onnx_model_path, log)
        if detector is None:
            return None

        base_url = (f"http://{local_cfg.get('host', 'localhost')}:"
                    f"{local_cfg.get('port', 8080)}")
        router = RouterClient(base_url, logger=log)

        escalation_detector, escalation_models = cls._build_escalation_detector(
            strategy_config, factory, raw_ocr_task, log)

        # Anchored to the run's root, never to the working directory. These
        # three used to default to CWD-relative "run_outputs/..." paths, so a
        # run started from another folder wrote its figures somewhere the
        # documents were not. See `paths.anchor_output_path`.
        figures_cfg = strategy_config.get("vlm_provider", {}).get("figures", {})
        figures_base_dir = anchor_output_path(figures_cfg.get("output_dir"),
                                              output_root, "figures")
        figures_path_root = anchor_output_path(figures_cfg.get("path_root"),
                                               output_root)

        tables_cfg = strategy_config.get("vlm_provider", {}).get("tables", {})
        tables_base_dir = anchor_output_path(tables_cfg.get("output_dir"),
                                             output_root, "tables")

        pipeline = cls(
            extraction_provider=built.provider,
            extraction_model=extraction_model,
            layout_detector=detector,
            resident_models=[extraction_model] + extra_resident,
            page_classifier=page_classifier,
            router=router,
            extraction_dpi=extraction_dpi,
            coverage_threshold=coverage_threshold,
            table_crop_prompt=table_crop_prompt,
            caption_read_prompt=caption_read_prompt,
            figures_base_dir=figures_base_dir,
            figures_path_root=figures_path_root,
            tables_base_dir=tables_base_dir,
            figure_escalation_detector=escalation_detector,
            figure_escalation_models=escalation_models,
            logger=log,
            table_crop_max_tokens=table_crop_max_tokens,
        )
        log.info(
            "GLM two-phase offline path ENABLED: extraction=%s, layout=%s, "
            "coverage_threshold=%s, extraction_dpi=%s",
            extraction_model, detector.name, coverage_threshold, extraction_dpi)
        return pipeline

    # ------------------------------------------------------------------ #
    # the body loop
    # ------------------------------------------------------------------ #

    def process_body_pages(self, cache: DocumentCache,
                           page_indices: Sequence[int]) -> List[ContentPage]:
        """
        Run the offline body pipeline over the given pages, sequentially.

        The needed models stay resident for the whole body. Each page is
        read, verified, repaired if a table was dropped, and its figures
        cropped from the same detection pass. A document-level figure
        manifest is written at the end for the multimodal RAG.
        """
        if self.router:
            emit(self.logger, "stage", stage="gpu_swap", message=
                 f"Reserving the GPU for {self.resident_models}",
                 models=list(self.resident_models))
            self.router.set_resident(self.resident_models)

        results: Dict[int, ContentPage] = {}
        page_total = len(page_indices)
        for position, page_idx in enumerate(page_indices, start=1):
            # The only NEW log line on the GPU-hot path. It changes no
            # extracted byte, but it is why a progress bar can exist at all.
            emit(self.logger, "page", stage="body",
                 message=f"Page {page_idx + 1} ({position}/{page_total})",
                 page=page_idx + 1, position=position, page_total=page_total)
            page_started = time.perf_counter()
            try:
                results[page_idx] = self.read_body_page(cache, page_idx)
            except Exception as error:
                self.logger.error("Body page %d failed: %s", page_idx + 1, error,
                                  exc_info=True)
                results[page_idx] = ContentPage(
                    page_number=page_idx + 1, content="", quality_score=0.0,
                    metadata={"error": str(error), "processing_method": "glm_failed"},
                )
            _record_page_time(results[page_idx], time.perf_counter() - page_started)

        self._second_opinion_on_figures(cache, results)
        self._recover_overleaf_captions(cache, results)

        all_tables = [tab for idx in sorted(results)
                      for tab in (results[idx].metadata.get("tables") or [])]
        write_table_manifest(cache.pdf_path, self.tables_base_dir, all_tables,
                             logger=self.logger)

        all_figures = [fig for idx in sorted(results)
                       for fig in (results[idx].metadata.get("figures") or [])]
        write_figure_manifest(cache.pdf_path, self.figures_base_dir, all_figures,
                              logger=self.logger)

        return [results[idx] for idx in sorted(results)]

    def read_body_page(self, cache: DocumentCache, page_idx: int) -> ContentPage:
        """
        Read one body page: classify -> text-layer fast path OR extraction
        read + verify/repair + figure extraction.
        """
        page = cache.page(page_idx)

        analysis: Dict[str, Any] = {}
        page_type = None
        if self.page_classifier is not None:
            stats = extract_text_stats(page)
            page_type, analysis = self.page_classifier.classify(
                page, page_idx, text_stats=stats)

        # A page the text-density filter emptied out has not earned the free
        # path yet — the filter may have thrown away a real figure. Checking
        # costs ~600ms of CPU; being wrong costs the figure, silently, because
        # the free path never reaches the second-opinion check either.
        # None means "the free path is honest"; a list (even an empty one)
        # means it is not.
        hidden = self._detections_the_filter_hid(cache, page_idx, analysis)

        if page_type is not None and page_type == PageType.TEXT_ONLY and hidden is None:
            self.logger.info("Page %d: pure text -> text layer (T1)", page_idx + 1)
            return self._process_with_text_layer(page, page_idx, analysis,
                                                 page_type)

        if hidden is not None:
            self.logger.info(
                "Page %d: the text-density filter hid %d real region(s) — "
                "reading it properly instead of letting it go free.",
                page_idx + 1, len(hidden))

        self.logger.info("Page %d: extraction model full-page read", page_idx + 1)
        page_image = cache.extraction_image(page_idx, self.extraction_dpi)
        read = self.extraction_provider.process_page(page_image)
        glm_text = read.get("content", "") if read.get("success") else ""
        base_quality = read.get("quality_score", 0.0) or 0.0

        # ONE detection pass per page, shared by table repair AND figure
        # extraction — and only when something will actually use it.
        gate = check_page_for_content_drop(
            page, glm_text, min_coverage=self.coverage_threshold, logger=self.logger)
        has_visuals = bool(analysis.get("visual_regions"))
        # `hidden` was detected on this same memoized image, so it IS the
        # detection pass for this page — running it again would buy nothing.
        detections = (hidden if hidden
                      else self._detect_if_needed(page, page_image, gate,
                                                  has_visuals))

        final_text, repair_meta = self._repair_page(page, glm_text, page_image,
                                                    gate, detections)

        figures = extract_page_figures(
            page, page_image, detections, cache.pdf_path, self.figures_base_dir,
            page_idx, read_crop=self._read_caption_crop,
            path_root=self.figures_path_root, logger=self.logger)
        if figures:
            final_text = append_figure_anchors(final_text, figures)

        # Tables get the same treatment as figures — crop, caption, id — so a
        # table is a retrievable object rather than prose inside a page blob.
        # Any transcription the repair path already read is carried through
        # free of charge; every OTHER table is read from its crop, so no table
        # leaves without a grid. See `_grid_for` in tables.py for why.
        tables = extract_page_tables(
            page, page_image, detections, cache.pdf_path, self.tables_base_dir,
            page_idx, recovered=repair_meta.get("recovered_tables"),
            path_root=self.figures_path_root, logger=self.logger,
            read_table_crop=self._read_table_crop)
        if tables:
            final_text = append_table_anchors(final_text, tables)

        # Repair control-character damage LAST, on the text that is about to be
        # stored. Deliberately not on `glm_text` above: that feeds
        # `check_page_for_content_drop`, and changing it would change the
        # coverage number and so the repair DECISION. Normalising only what is
        # stored keeps this a text-quality fix rather than a routing change.
        # The appended figure/table anchors are generated strings and contain
        # no control characters, so passing them through is a no-op.
        final_text = normalize_harvested_text(final_text)

        quality = max(base_quality, 0.85) if repair_meta.get("tables_recovered") \
            else base_quality
        metadata = {
            "processing_method": repair_meta["processing_method"],
            "page_type": page_type.value if page_type is not None else None,
            # Typography for the heading-shaped lines, read from the PDF's own
            # text layer. Recorded on BOTH routes because a model-read page's
            # prose comes from the model but its fonts still come from the
            # file, and the depth of an unnumbered heading is not recoverable
            # from text alone -- see heading_pass.py.
            "headings": heading_candidates_for_page(page, logger=self.logger),
            # How bibliographic this page reads. A reference list is a
            # term-density trap for retrieval -- see reference_pass.py.
            **reference_signals(final_text),
            "coverage": repair_meta.get("coverage"),
            "tables_recovered": repair_meta.get("tables_recovered", 0),
            "visual_element_count": len(analysis.get("visual_regions", [])),
            "figures": [f.to_dict() for f in figures],
            "figure_count": len(figures),
            # Recorded now, acted on after the document — see
            # `figure_escalation.py`. The verdict must be taken while the
            # classifier's analysis is in hand; reconstructing it later would
            # mean re-running the classifier.
            "needs_figure_second_opinion": (
                not figures and page_needs_second_opinion(analysis, detections,
                                                          page.size)),
            "tables": [t.to_dict() for t in tables],
            "table_count": len(tables),
        }
        return ContentPage(page_number=page_idx + 1, content=final_text,
                           quality_score=quality, metadata=metadata)

    # --- helpers (after the main methods, per CLAUDE.md) ------------------

    @staticmethod
    def _build_layout_detector(
        strategy_config: Dict[str, Any], factory: ProviderFactory,
        raw_ocr_task: str, onnx_model_path: Path | str, log: logging.Logger,
    ) -> Tuple[Optional[LayoutDetector], List[str]]:
        """
        Build the detector the registry names.

        'onnx' -> the CPU detector (no extra resident model).
        Anything else is a VLM profile name -> the grounding fallback, which
        must also be resident on the GPU during Phase A.
        """
        layout_name = resolve_reference("@profile_registry.layout_detector",
                                        strategy_config)
        if layout_name == "onnx":
            from ...layout.onnx_detector import OnnxLayoutDetector  # lazy: onnxruntime
            return OnnxLayoutDetector(onnx_model_path), []

        built = factory.build(layout_name, raw_ocr_task)
        grounding_format = built.profile_data.get("grounding_format")
        if not grounding_format:
            log.error("Layout model '%s' declares no grounding_format; "
                      "GLM two-phase path stays DISABLED", layout_name)
            return None, []
        spec = GroundingSpec.from_config(grounding_format)
        prompt = grounding_format["prompt"]
        provider = built.provider
        detector = VlmGroundingDetector(
            spec, lambda image: provider.read_crop(image, prompt, image_first=True))
        return detector, [layout_name]

    @staticmethod
    def _build_escalation_detector(
        strategy_config: Dict[str, Any], factory: ProviderFactory,
        raw_ocr_task: str, log: logging.Logger,
    ) -> Tuple[Optional[LayoutDetector], List[str]]:
        """
        Build the grounding detector used for second opinions, if configured.

        Named by the `figure_escalation` role in the profile registry, so the
        model stays in YAML and never in `src/` — the VLM-agnostic rule. Absent
        role means the feature is simply off, which is the default: it costs
        GPU time, so it must be asked for.

        Returns:
            (detector, models to make resident), or (None, []) when the role is
            unset or the named profile cannot ground. A misconfiguration here
            disables the second opinion; it never fails the run, because the
            fast path alone is exactly what V6 shipped before this existed.
        """
        registry = strategy_config.get("profile_registry", {})
        if "figure_escalation" not in registry:
            return None, []

        model_name = resolve_reference("@profile_registry.figure_escalation",
                                       strategy_config)
        if model_name == "onnx":
            log.warning("figure_escalation is set to 'onnx', the same detector "
                        "that already ran — a second opinion from the same model "
                        "is not one. Disabled.")
            return None, []

        try:
            built = factory.build(model_name, raw_ocr_task)
            grounding_format = built.profile_data.get("grounding_format")
            if not grounding_format:
                log.warning("figure_escalation model '%s' declares no "
                            "grounding_format; second opinions disabled.", model_name)
                return None, []

            spec = GroundingSpec.from_config(grounding_format)
            prompt = grounding_format["prompt"]
            provider = built.provider
            detector = VlmGroundingDetector(
                spec, lambda image: provider.read_crop(image, prompt, image_first=True))
        except Exception as error:
            log.warning("Could not build the figure_escalation model '%s' (%s); "
                        "second opinions disabled.", model_name, error)
            return None, []

        log.info("Figure second opinion ENABLED via '%s' — used only on pages "
                 "where the fast detector and the classifier disagree.", model_name)
        return detector, [model_name]

    def _process_with_text_layer(self, page: PdfPage, page_idx: int,
                                 analysis: Dict[str, Any],
                                 page_type: Optional[PageType] = None) -> ContentPage:
        """
        Pure-text page: the text layer is authoritative and free (T1).

        Two things happen here that do not happen on the model path, because
        this is where raw harvested text becomes stored content:

        1. The text is normalised. Some PDFs encode the hyphen that splits a
           word across a line break as a control character, so the word arrives
           as "measure<02>ments" and can never be found by any search. See
           `text_normalization` for the measurements behind the rule.
        2. `page_type` is recorded. It was computed to make the routing
           decision and then discarded on this path only, so 60% of pages could
           not say how they were produced. Retrieval quality differs by route,
           so anything reading this output needs to be able to tell.
        """
        return ContentPage(
            page_number=page_idx + 1,
            content=normalize_harvested_text(page.text()),
            quality_score=0.95,
            metadata={
                "processing_method": "pymupdf",
                "page_type": page_type.value if page_type is not None else None,
                "analysis": analysis,
                "headings": heading_candidates_for_page(page, logger=self.logger),
                **reference_signals(normalize_harvested_text(page.text())),
            },
        )

    def _detections_the_filter_hid(
        self, cache: DocumentCache, page_idx: int, analysis: Dict[str, Any],
    ) -> Optional[List[Detection]]:
        """
        Check the text-density filter's judgement before a page goes free.

        For biology researchers: the classifier throws away regions that are
        mostly covered by words, because a reference list draws a rectangle
        around every "CrossRef" link and a cluster of those looks exactly like
        a figure. That test works — but it is a test of *typesetting*, not of
        content, and two things measure like decoration without being it:

        - **a table**, which is dense text in ruled cells by construction;
        - a figure whose text the publisher set as selectable text rather
          than outlined vector glyphs (e.g. a sequence alignment).

        With nothing visual left, such a page can fall to TEXT_ONLY — and that
        path is never read, never detected, and never escalated, since the
        second-opinion check only looks at pages the VLM read. The content
        would be lost with nothing anywhere saying so. This is not
        hypothetical: on the sample corpus the filter discarded exactly the
        box the detector calls `table` on Silberbush p4, turning a page that
        used to yield "Table 1" into a free page that yielded nothing.

        So: when the filter discarded regions and left NONE standing, the CPU
        detector gets one look at the page (~600ms, no GPU). Measured over 476
        body pages, that is 10 pages — 9 confirmed free, 1 overturned — for
        5.3s of CPU across the whole corpus.

        Args:
            cache: the document cache, for the page and its rendered image.
            page_idx: 0-based page index.
            analysis: the classifier's analysis for this page.

        Returns:
            **None** when the free path is honest: nothing was filtered away,
            there is nothing to check with, or the detector agrees the page is
            just text.

            **A list** when it is not — the page must be read properly, and
            these are the boxes to crop from. The list is empty only when the
            detector itself failed, which still answers "not free": a wasted
            read costs seconds, a lost figure is permanent and invisible.
        """
        if analysis.get("visual_regions") or not analysis.get("text_dense_regions"):
            return None                     # nothing was filtered away

        if self.layout_detector is None or not self.layout_detector.supports_figures:
            # Without a figure-capable detector the expensive path could not
            # produce a figure either, so paying for it would buy nothing.
            return None

        try:
            page_image = cache.extraction_image(page_idx, self.extraction_dpi)
            detections = self.layout_detector.detect(page_image,
                                                     page=cache.page(page_idx))
        except Exception as error:                          # pragma: no cover
            self.logger.warning(
                "p%d: could not check the text-density filter (%s); reading the "
                "page rather than assuming it is empty.", page_idx + 1, error)
            return []                       # no boxes, but not "free" either

        wanted = set(FIGURE_LABELS) | {LABEL_TABLE}
        if not any(getattr(d, "label", None) in wanted for d in detections):
            self.logger.debug("p%d: the detector agrees, %d filtered region(s) "
                              "really were text decoration.", page_idx + 1,
                              len(analysis["text_dense_regions"]))
            return None

        return detections

    def _recover_overleaf_captions(self, cache: DocumentCache,
                                   results: Dict[int, ContentPage]) -> None:
        """
        Give a full-page figure the caption printed on the following page.

        For biology researchers: a journal running a full-page figure often
        prints its caption on the next page. The per-page labelling pass reads
        only the page a figure sits on, so those figures arrive with no
        caption at all — even though the caption is in the document, one page
        away. See `utils/caption_overleaf.py` for the safety conditions.

        This runs here and nowhere else because it is the first point at which
        one page's figures and another page's text are both in hand.

        Failure is never fatal: the figures keep whatever captions they
        already had, which is exactly what they would have had without this
        step.
        """
        try:
            figures_by_page: Dict[int, List[dict]] = {}
            label_lines_by_page: Dict[int, List] = {}

            for page_idx, content_page in results.items():
                figures = content_page.metadata.get("figures") or []
                if figures:
                    figures_by_page[page_idx + 1] = figures

            if not figures_by_page:
                return

            # Only the pages that could actually supply a caption are read,
            # rather than the whole document: the neighbours of a page whose
            # figures might need one.
            wanted = {
                page + offset
                for page in figures_by_page
                for offset in NEIGHBOUR_OFFSETS
            }
            for page_number in sorted(wanted):
                page_idx = page_number - 1
                if page_idx < 0:
                    continue
                try:
                    page = cache.page(page_idx)
                except Exception:
                    continue                    # past the end of the document
                width, _ = page.size
                words = [(w.text, w.rect.x0, w.rect.y0, w.rect.x1, w.rect.y1)
                         for w in page.words()]
                label_lines_by_page[page_number] = find_label_lines(
                    build_text_lines(words, width))

            recovered = recover_overleaf_captions(
                figures_by_page, label_lines_by_page, logger=self.logger)
            if recovered:
                self.logger.info(
                    "Overleaf caption pass: %d figure(s) recovered a caption "
                    "from an adjacent page.", recovered)
        except Exception as error:                          # pragma: no cover
            self.logger.warning(
                "Overleaf caption pass failed (%s); figures keep the captions "
                "they already had.", error)

    def _second_opinion_on_figures(self, cache: DocumentCache,
                                   results: Dict[int, ContentPage]) -> None:
        """
        Re-examine the pages where the CPU pass contradicted itself.

        For biology researchers: a handful of pages end up with the classifier
        saying "there is something visual here" and the fast detector saying
        "no figure". Those are the only pages a slower, stronger model is asked
        about — measured at ~2% of pages on the sample corpus. Everything else
        is untouched, so the cost of being thorough stays proportional to how
        often we are actually unsure.

        Done ONCE for the whole document, after every page has been read. The
        grounding model is ~4.9GB on an 8GB card, so swapping it in per page
        would evict the reading model and make every remaining page decode ~3x
        slower — see CLAUDE.md, "VRAM residency".

        Failure here is never fatal: the pages keep the figures they already
        had, which is exactly what they would have had without this step.
        """
        if self.figure_escalation_detector is None:
            return

        suspects = collect_suspect_pages(results)
        if not suspects:
            return
        self.logger.info(describe_escalation(suspects))

        try:
            if self.router:
                emit(self.logger, "stage", stage="gpu_swap", message=
                     f"Swapping to {self.figure_escalation_models} for a second "
                     f"opinion on {len(suspects)} page(s)",
                     models=list(self.figure_escalation_models))
                self.router.set_resident(self.figure_escalation_models)

            for page_idx in suspects:
                self._re_detect_figures(cache, results, page_idx)
        except Exception as error:
            self.logger.warning("Figure second opinion abandoned (%s); pages keep "
                                "what the fast detector found.", error)
        finally:
            if self.router:
                emit(self.logger, "stage", stage="gpu_swap", message=
                     f"Swapping back to {self.resident_models}",
                     models=list(self.resident_models))
                self.router.set_resident(self.resident_models)

    def _re_detect_figures(self, cache: DocumentCache, results: Dict[int, ContentPage],
                           page_idx: int) -> None:
        """Re-detect one page with the grounding model and keep anything found."""
        page = cache.page(page_idx)
        page_image = cache.extraction_image(page_idx, self.extraction_dpi)

        detections = self.figure_escalation_detector.detect(page_image, page=page)
        figures = extract_page_figures(
            page, page_image, detections, cache.pdf_path, self.figures_base_dir,
            page_idx, read_crop=self._read_caption_crop,
            path_root=self.figures_path_root, logger=self.logger)

        content_page = results[page_idx]
        content_page.metadata["figure_second_opinion"] = "found" if figures else "none"
        if not figures:
            self.logger.info("  p%d: the grounding model agrees, no figure here.",
                             page_idx + 1)
            return

        self.logger.info("  p%d: RECOVERED %d figure(s) the fast detector missed.",
                         page_idx + 1, len(figures))
        content_page.content = append_figure_anchors(content_page.content, figures)
        content_page.metadata["figures"] = [f.to_dict() for f in figures]
        content_page.metadata["figure_count"] = len(figures)

    def _detect_if_needed(self, page: PdfPage, page_image: Image.Image,
                          gate: CoverageResult,
                          has_visuals: bool) -> Optional[List[Detection]]:
        """
        Run layout detection at most ONCE per page, only when something needs
        it: a born-digital content drop, a scanned page (no text layer to
        verify against), or classifier-flagged visuals to crop.

        Returns None when nothing needs it, so we never pay for a detection
        pass we will not use.
        """
        born_digital_drop = gate.checked and gate.content_dropped
        scanned = (not gate.checked) and is_scanned_page(page)
        want_figures = has_visuals and self.layout_detector.supports_figures
        if not (born_digital_drop or scanned or want_figures):
            return None
        return self.layout_detector.detect(page_image, page=page)

    def _repair_page(self, page: PdfPage, glm_text: str, page_image: Image.Image,
                     gate: CoverageResult,
                     detections: Optional[List[Detection]]) -> Tuple[str, Dict]:
        """
        Repair a dropped table using the already-computed gate + detections.

        Born-digital pages use the word-coverage gate. Scanned pages have no
        trustworthy text layer, so the drop signal is "the detector sees a
        table the read did not produce".
        """
        coverage = round(gate.coverage, 3) if gate.checked else None
        born_digital_drop = gate.checked and gate.content_dropped
        scanned_has_table = (
            (not gate.checked) and is_scanned_page(page)
            and detections is not None
            and any(d.label == LABEL_TABLE for d in detections)
        )
        if not (born_digital_drop or scanned_has_table) or detections is None:
            return glm_text, {"processing_method": "glm_fullpage",
                              "coverage": coverage, "tables_recovered": 0}

        result = repair_page_tables(
            page_image=page_image, full_page_text=glm_text,
            detections=detections, read_table_crop=self._read_table_crop,
            logger=self.logger)
        return result.text, {
            "processing_method": "glm_repair" if result.repaired else "glm_fullpage",
            "coverage": coverage,
            "tables_recovered": result.tables_recovered,
            # Carried so table tagging can reuse the transcription the crop
            # read already produced, instead of paying for it twice.
            "recovered_tables": result.tables,
        }

    def _read_table_crop(self, crop_image: Image.Image) -> str:
        """Re-read one cropped table with the extraction model."""
        return self.extraction_provider.read_crop(
            crop_image, self.table_crop_prompt, max_tokens=self.table_crop_max_tokens)

    def _read_caption_crop(self, crop_image: Image.Image) -> str:
        """Read a caption crop (scanned pages only) with the extraction model."""
        if not self.caption_read_prompt:
            return ""
        return self.extraction_provider.read_crop(crop_image, self.caption_read_prompt)


def _record_page_time(page: ContentPage, seconds: float) -> None:
    """
    Store how long one body page took to read, in seconds, as `page_time`.

    This is what lets the README report cost per KIND of page (text-layer
    page, full-page model read, table page) instead of per document, where a
    short paper's fixed start-up cost made it look slow. It covers everything
    done for that page inside the body loop: classification, the model read,
    repair, and figure and table crops. Work done once per document (loading
    models, the title page, the end-of-document figure passes) is NOT in it:
    the title page records its own `total_time`, and the rest is the document
    total minus the per-page times.
    """
    page.metadata["page_time"] = round(seconds, 2)
