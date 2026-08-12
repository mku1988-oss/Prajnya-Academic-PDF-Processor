#!/usr/bin/env python3
"""
The book-chapter strategy — an article-shaped read, a book-shaped record.

For biology researchers: a chapter in an edited scientific volume reads like a
paper. It opens with a title, its own authors and an abstract, then runs prose,
figures and tables — the research-article pipeline handles that shape well, and
duplicating it here would mean maintaining a second copy of a benchmarked path.

What a chapter does *not* share with a paper is its identity. It belongs to a
book: it has a chapter number, it sits inside a series, and its identifiers
(ISBN, series ISSN, a book-shaped DOI) describe the volume around it.

So this strategy **composes** rather than copies: the page reading is delegated
to `ResearchArticleStrategy`, and this file owns everything about being a
chapter. Future chapter-specific work — chapter numbering from the parent
book's contents page, editors, cross-chapter references — belongs here and can
be built without touching the article path or re-running its parity gate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from ...extractors.metadata_extractor import MetadataExtractor
from ...models import DocumentProfile
from ...pdf import PdfBackend
from ..research_article.strategy import ResearchArticleStrategy


class BookChapterStrategy:
    """
    Processes one book chapter: the article pipeline, plus the book's identity.

    Args:
        config_root: the `config/` directory, as for the article strategy.
        pdf_backend: page access implementation.
        logger: optional logger.
        article_strategy: an already-built `ResearchArticleStrategy` to
            delegate to. Injected mainly for tests; built on demand otherwise.
    """

    def __init__(self, config_root: Path | str,
                 pdf_backend: Optional[PdfBackend] = None,
                 logger: Optional[logging.Logger] = None,
                 article_strategy: Optional[ResearchArticleStrategy] = None,
                 refresh_metadata: bool = False,
                 output_root: Optional[Path] = None) -> None:
        self.logger = logger or logging.getLogger("BookChapterStrategy")
        self.reader = article_strategy or ResearchArticleStrategy(
            config_root, pdf_backend=pdf_backend, logger=self.logger,
            refresh_metadata=refresh_metadata, output_root=output_root)

    def process(self, profile: DocumentProfile,
                test_pages_override: Optional[List[int]] = None) -> DocumentProfile:
        """
        Read the chapter, then stamp the book's identity onto its record.

        Args:
            profile: the document profile from the first pass.
            test_pages_override: optional 1-based page subset, passed through.

        Returns:
            The same profile with `processing_result` filled in by the article
            pipeline, its metadata extended with the chapter number and the
            parent volume's identifiers.
        """
        profile = self.reader.process(profile, test_pages_override=test_pages_override)
        self._add_chapter_identity(profile)
        return profile

    # --- helpers (after the main method, per CLAUDE.md) -------------------

    def _add_chapter_identity(self, profile: DocumentProfile) -> None:
        """Record what makes this a chapter rather than a paper."""
        result = profile.processing_result
        if result is None:
            return

        metadata = result.metadata
        chapter_number = self._chapter_number(profile)
        if chapter_number is not None:
            metadata.chapter_number = chapter_number
        if profile.isbn and not metadata.isbn:
            metadata.isbn = profile.isbn

        provenance = dict(metadata.provenance or {})
        provenance.update({
            "document_type": "book_chapter",
            "issn": profile.issn,
            "identifier_pages": dict(profile.identifier_pages),
        })
        metadata.provenance = provenance

    def _chapter_number(self, profile: DocumentProfile) -> Optional[int]:
        """
        The chapter number, from the opening page's heading.

        Publishers write it either as a numeral ("Chapter 3") or as a word
        ("CHAPTER THREE", which is how the Elsevier sample prints it), so the
        word form is translated before the shared numeral extractor runs.
        """
        text = profile.page_1_text or ""
        if not text:
            return None

        number, confidence = MetadataExtractor().extract_chapter_number(
            _numeralize_chapter_words(text))
        return number if number is not None and confidence >= 0.9 else None


#: Publishers spell chapter numbers out. Twenty covers any edited volume we
#: are likely to meet; beyond that the numeral form is universal.
_CHAPTER_WORDS = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty",
]


def _numeralize_chapter_words(text: str) -> str:
    """Rewrite 'CHAPTER THREE' as 'Chapter 3' so the numeral extractor sees it."""
    import re

    def replace(match: "re.Match") -> str:
        word = match.group("word").lower()
        return f"Chapter {_CHAPTER_WORDS.index(word) + 1}"

    pattern = re.compile(r'\bchapter\s+(?P<word>' + "|".join(_CHAPTER_WORDS) + r')\b',
                         re.IGNORECASE)
    return pattern.sub(replace, text)
