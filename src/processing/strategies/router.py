#!/usr/bin/env python3
"""
Strategy routing — matching a classified document to the pipeline that suits it.

For biology researchers: the first pass decides what kind of document a PDF is
(see `processing/classification/`); this decides who reads it. Three document
types, three strategies, one lookup — deliberately boring, because the value is
in keeping the three paths independent, not in the choosing.

Each strategy is built only when a document of its type actually arrives.
That matters on this hardware: constructing the article strategy resolves the
whole config tree and loads the ONNX layout model, which a book's front-matter
read has no use for.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

from ..config.loader import resolve_reference
from ..models import DocumentProfile, DocumentType
from ..pdf import PdfBackend


class DocumentStrategy(Protocol):
    """What every strategy must do: process one document, in place."""

    def process(self, profile: DocumentProfile,
                test_pages_override: Optional[List[int]] = None) -> DocumentProfile:
        ...


def build_strategy(document_type: Optional[DocumentType], config_root: Path | str,
                   pdf_backend: Optional[PdfBackend] = None,
                   logger: Optional[logging.Logger] = None,
                   chapters: Optional[Sequence[int]] = None,
                   max_body_pages: Optional[int] = None,
                   refresh_metadata: bool = False,
                   output_root: Optional[Path] = None) -> DocumentStrategy:
    """
    Build the strategy for a classified document.

    Args:
        document_type: the classified type. None routes to the research
            article, which is also the classifier's safe default.
        config_root: the `config/` directory.
        pdf_backend: page access implementation.
        logger: optional logger.
        chapters: books only — which chapters to read. None reads no chapter
            bodies, which is the safe default: a whole volume is hours of GPU.
        max_body_pages: books only — a ceiling on pages read.
        refresh_metadata: `--force`; re-derive metadata rather than reusing
            the ledger's stored row.
        output_root: where this run's figures and tables are written. None
            means `paths.DEFAULT_OUTPUT_ROOT`. Threaded down to `BodyPipeline`
            so that every artefact of one run lands in one place, whatever
            directory the run was launched from.

    Returns:
        A strategy exposing `process(profile, test_pages_override=None)`.

    Example:
        >>> strategy = build_strategy(profile.document_type, "config")  # doctest: +SKIP
        >>> profile = strategy.process(profile)
    """
    # Imported here rather than at module scope: constructing the article
    # strategy pulls in the config tree and the layout model, and a book run
    # should not pay for that.
    if document_type is DocumentType.BOOK:
        from .book import BookStrategy

        return BookStrategy(pdf_backend=pdf_backend, logger=logger,
                            body_pipeline=_book_body_pipeline(config_root, chapters,
                                                              logger, output_root),
                            chapters_to_read=chapters,
                            max_body_pages=max_body_pages)

    if document_type is DocumentType.BOOK_CHAPTER:
        from .book_chapter import BookChapterStrategy

        return BookChapterStrategy(config_root, pdf_backend=pdf_backend, logger=logger,
                                   refresh_metadata=refresh_metadata,
                                   output_root=output_root)

    from .research_article.strategy import ResearchArticleStrategy

    return ResearchArticleStrategy(config_root, pdf_backend=pdf_backend, logger=logger,
                                   refresh_metadata=refresh_metadata,
                                   output_root=output_root)


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def _book_body_pipeline(config_root: Path | str, chapters: Optional[Sequence[int]],
                        logger: Optional[logging.Logger],
                        output_root: Optional[Path] = None):
    """
    Build the page reader a book needs — but only when chapters were asked for.

    Constructing it resolves the config tree and loads the ONNX layout model,
    which a front-matter-only run has no use for. So the cost is paid only by
    a caller who actually wants chapters read.

    Returns None (and logs) if the offline pipeline is not configured, which
    leaves the book strategy on its CPU-only path rather than failing the run.
    """
    # `is None` and not falsiness: an EMPTY list means "every chapter", which
    # needs the reader more than any other request does. Testing `if not
    # chapters` here is what made `--chapters` with no numbers read nothing.
    if chapters is None:
        return None

    from ..config.loader import ConfigLoader
    from ..utils.page_classifier import PageClassifier
    from .research_article.body_pipeline import BodyPipeline
    from .research_article.provider_factory import ProviderFactory

    log = logger or logging.getLogger("router")
    loader = ConfigLoader(config_root, logger=log)
    config = loader.strategy()

    title_profile = loader.vlm_profile(
        resolve_reference("@profile_registry.title_detection", config))
    pipeline = BodyPipeline.from_config(
        config, ProviderFactory(loader, config, logger=log),
        page_classifier=PageClassifier(title_profile), logger=log,
        output_root=output_root)

    if pipeline is None:
        log.warning("Chapters were requested but the offline body pipeline is not "
                    "configured; the book will be read on the CPU path only.")
    return pipeline
