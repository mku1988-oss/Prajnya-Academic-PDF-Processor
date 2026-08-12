#!/usr/bin/env python3
"""
The book strategy — the whole-book record, plus opt-in chapter reading.

For biology researchers: a book is not a long article. Its identity is spread
over the first few pages rather than concentrated on one title page, and its
body is a sequence of independent chapters, each with its own authors. This
class handles the first half of that properly and is honest about not yet
handling the second.

**What it does today**, entirely on the CPU with no model call at all:

- reads the bibliographic record off the front matter — title, editors,
  publisher, year, series (`front_matter_record.py`);
- carries the identifiers the first pass found on the copyright page — ISBN,
  ISSN, the book DOI (`extractors/front_matter.py`);
- parses the table of contents into chapters with their titles, **their own
  authors**, and their start pages (`toc.py`);
- maps every body page to the chapter it belongs to
  (`chapter_boundaries.py`), so a passage retrieved from page 60 can be
  attributed to *that* chapter's authors rather than to the book's editor;
- returns the front-matter pages' text as content.

**Chapter bodies are read only when asked for.** Pass a `body_pipeline` and
`chapters_to_read` and the requested chapters' pages are read and attributed;
pass neither and this stays the CPU-only front-matter read above. That default
is deliberate: at the article pipeline's measured ~20s/page a 345-page volume
would be about two hours of GPU, so a whole-book read has to be a decision
somebody made rather than something that happens by omission. The reading
itself is delegated to `BodyPipeline` — see `reading.py` for why.

**Why this is a separate file at all.** V6's research-article pipeline is
tested, benchmarked, and at parity. Book processing is going to get more
sophisticated, and none of that work should require touching the article path
or re-running its parity gate. So the book path lives here, from the start,
even while it is deliberately simple. (V5's `BookStrategy` was a stub that set
`title = pdf_path.stem`; nothing from it is carried over.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ...models import (
    ContentPage,
    DocumentProfile,
    ExtractedMetadata,
    ProcessingResult,
)
from ...extractors.front_matter import DEFAULT_FRONT_MATTER_PAGES, scan_front_matter
from ...pdf import PdfBackend, get_backend
from .chapter_boundaries import detect_chapter_spans
from ..research_article.rendering import DocumentCache
from .front_matter_record import build_book_record
from .reading import attribute_pages, estimate_reading_cost, select_chapter_pages
from .toc import TocEntry, parse_toc_entries, toc_entries_to_text


class BookStrategy:
    """
    Processes a book's front matter into a bibliographic record and a chapter map.

    Args:
        pdf_backend: page access implementation; the permissive default when
            omitted.
        front_matter_pages: how many leading pages to read.
        logger: optional logger.
        body_pipeline: the article pipeline's `BodyPipeline`, which does the
            actual page reading. Omit it and no chapter body is ever read.
        chapters_to_read: chapter numbers to read. None (the default) reads no
            bodies at all; a list of numbers reads exactly those chapters; an
            EMPTY list means every chapter — which is what the CLI's
            `--chapters` with no numbers produces, and is hours of GPU on a
            full volume. None vs empty is the whole distinction here, so this
            code tests `is None` and never falsiness.
        max_body_pages: hard ceiling on pages read, as a guard against an
            accidental whole-volume request.
    """

    def __init__(self, pdf_backend: Optional[PdfBackend] = None,
                 front_matter_pages: int = DEFAULT_FRONT_MATTER_PAGES,
                 logger: Optional[logging.Logger] = None,
                 body_pipeline=None,
                 chapters_to_read: Optional[Sequence[int]] = None,
                 max_body_pages: Optional[int] = None) -> None:
        self.backend = pdf_backend or get_backend()
        self.front_matter_pages = front_matter_pages
        self.logger = logger or logging.getLogger("BookStrategy")
        self.body_pipeline = body_pipeline
        self.chapters_to_read = chapters_to_read
        self.max_body_pages = max_body_pages

    def process(self, profile: DocumentProfile,
                test_pages_override: Optional[List[int]] = None) -> DocumentProfile:
        """
        Read the book's front matter, populating `profile.processing_result`.

        Args:
            profile: the document profile from the first pass.
            test_pages_override: accepted for interface symmetry with the
                research-article strategy; chapter selection is the book
                equivalent, so this is unused.

        Returns:
            The same profile with `processing_result` filled in: the
            bibliographic record, the contents map with per-chapter authors,
            the front-matter pages, and — only if chapters were requested —
            their body pages, each stamped with its chapter's authors.
        """
        errors: List[str] = []
        metadata = ExtractedMetadata()
        pages: List[ContentPage] = []
        entries: List[TocEntry] = []
        toc_pages: List[int] = []

        chapter_map = None

        try:
            page_texts, all_texts = self._read_text_layer(profile)
            metadata = self._build_metadata(profile, page_texts)
            entries, toc_pages = self._read_contents(page_texts)
            pages = self._front_matter_pages(page_texts, toc_pages)
            chapter_map = self._detect_chapters(all_texts, entries, toc_pages)
            pages += self._read_chapter_bodies(profile, chapter_map)
        except Exception as error:
            message = f"Book processing failed: {type(error).__name__} - {error}"
            self.logger.error(message, exc_info=True)
            errors.append(message)

        profile.processing_result = ProcessingResult(
            metadata=metadata,
            pages=pages,
            toc=toc_entries_to_text(entries),
            toc_pages=toc_pages,
            errors=errors,
            chapters=[self._chapter_record(entry, chapter_map) for entry in entries],
        )
        self._log_outcome(entries, chapter_map)
        return profile

    # --- helpers (after the main method, per CLAUDE.md) -------------------

    def _read_text_layer(self, profile: DocumentProfile
                         ) -> tuple[Dict[int, str], Dict[int, str]]:
        """
        Read the book's text once, for both jobs that need it.

        Returns two views of the same read, because the two consumers count
        pages differently and conflating them causes off-by-one attribution:
        the front-matter record is 1-based (it speaks the book's own page
        numbers), while chapter boundaries are 0-based PDF indices.

        Reading the whole book is affordable — measured 0.75s for 345 pages,
        CPU only, no model — and chapter boundaries genuinely need every page:
        the running headers that carry them are spread through the body.
        """
        with self.backend.open(profile.pdf_path) as doc:
            identifiers = scan_front_matter(doc, max_pages=self.front_matter_pages,
                                            page_1_text=profile.page_1_text)
            all_texts = {index: (doc.page(index).text() or "")
                         for index in range(doc.page_count)}

        return identifiers.page_texts, all_texts

    def _detect_chapters(self, all_texts: Dict[int, str], entries: List[TocEntry],
                         toc_pages: List[int]):
        """
        Map every body page to its chapter — the RAG attribution problem.

        The body starts after the last contents page, so nothing in the front
        matter is ever attributed to a chapter.

        Watch the numbering here: `toc_pages` are 1-based book page numbers,
        while `detect_chapter_spans` works in 0-based PDF indices. The last
        contents page at 1-based `n` is 0-based `n - 1`, so the body begins at
        0-based `n` — which is why `max(toc_pages)` is used unchanged and not
        offset. Getting this wrong shifts every chapter by one page.
        """
        if not entries:
            return None

        first_body_index = max(toc_pages) if toc_pages else self.front_matter_pages
        return detect_chapter_spans(all_texts, entries,
                                    first_body_index=first_body_index)

    def _read_chapter_bodies(self, profile: DocumentProfile,
                             chapter_map) -> List[ContentPage]:
        """
        Read the chapters asked for — and nothing unless they were asked for.

        Reading is opt-in because the difference between one chapter and a
        whole volume is hours of GPU time (see `reading.py`). With no
        `body_pipeline` or no `chapters_to_read`, this returns nothing and the
        strategy stays the CPU-only front-matter read it has always been.

        The page reading itself is delegated to the article pipeline's
        `BodyPipeline`, which already routes each page to the text layer or
        the VLM and is covered by the parity harness. What this adds is the
        selection and the attribution.
        """
        if self.body_pipeline is None or self.chapters_to_read is None:
            return []
        if chapter_map is None or not chapter_map.spans:
            self.logger.warning(
                "Chapters were requested but no chapter map was built; "
                "nothing can be read without knowing where chapters are.")
            return []

        page_indices = select_chapter_pages(chapter_map, self.chapters_to_read,
                                            max_pages=self.max_body_pages)
        if not page_indices:
            return []

        cost = estimate_reading_cost(page_indices)
        self.logger.info("Reading %d pages (worst case ~%.1f minutes; most book "
                         "pages never reach the GPU)",
                         cost["pages"], cost["worst_case_minutes"])

        with DocumentCache(self.backend, profile.pdf_path,
                           logger=self.logger) as cache:
            pages = self.body_pipeline.process_body_pages(cache, page_indices)

        return attribute_pages(pages, chapter_map)

    def _log_outcome(self, entries: List[TocEntry], chapter_map) -> None:
        """Say plainly what was mapped and what was not read."""
        numbered = sum(1 for entry in entries if entry.number is not None)
        attributed = sum(1 for entry in entries if entry.authors)

        if chapter_map is not None and chapter_map.spans:
            self.logger.info(
                "Chapter map: %d chapters located across the body, %.0f%% of body "
                "pages identified by a running header (%s).",
                len(chapter_map.spans), 100 * chapter_map.direct_coverage,
                "confident" if chapter_map.is_confident
                else "LOW CONFIDENCE - attribution rests on interpolation")
        else:
            self.logger.warning("No chapter map: page-level attribution unavailable.")

        summary = ("Book read: %d contents entries (%d numbered chapters, "
                   "%d with their own authors).")
        if self.chapters_to_read is None:
            self.logger.warning(
                summary + " Chapter BODIES were NOT read — pass chapters_to_read "
                "and a body_pipeline to read them.", len(entries), numbered, attributed)
        else:
            requested = (list(self.chapters_to_read) if self.chapters_to_read
                         else "EVERY chapter")
            self.logger.info(summary + " Chapters read: %s.",
                             len(entries), numbered, attributed, requested)

    def _build_metadata(self, profile: DocumentProfile,
                        page_texts: Dict[int, str]) -> ExtractedMetadata:
        """
        Combine the front-matter record with the identifiers already found.

        The identifiers come from the first pass rather than being re-derived,
        so the page each was found on (`profile.identifier_pages`) stays the
        single source of truth.
        """
        copyright_page = profile.identifier_pages.get("isbn")
        record = build_book_record(page_texts, copyright_page)

        fields = record.as_metadata()
        if profile.isbn:
            fields["isbn"] = profile.isbn
        if profile.doi:
            fields["doi"] = profile.doi

        fields["validation_source"] = "front_matter_text_layer"
        fields["validation_confidence"] = 0.9 if record.title else 0.5
        fields["provenance"] = {
            "series": record.series,
            "issn": profile.issn,
            "title_page": record.title_page,
            "copyright_page": record.copyright_page,
            "identifier_pages": dict(profile.identifier_pages),
            "note": "front matter only; chapter bodies not extracted",
        }
        return ExtractedMetadata(**fields)

    def _read_contents(self, page_texts: Dict[int, str]
                       ) -> tuple[List[TocEntry], List[int]]:
        """Parse every contents page found in the front matter."""
        entries: List[TocEntry] = []
        toc_pages: List[int] = []

        for page_number, text in sorted(page_texts.items()):
            if not text:
                continue
            found = parse_toc_entries(text)
            if found:
                entries.extend(found)
                toc_pages.append(page_number)

        return entries, toc_pages

    @staticmethod
    def _chapter_record(entry: TocEntry, chapter_map=None) -> Dict:
        """
        One chapter as the retrieval index will want it.

        For biology researchers: in an edited volume each chapter is written by
        different people, so a passage quoted from chapter 3 must be cited to
        *its* authors, not to the book's editor. This record carries that
        attribution alongside the chapter's own title and its start page.

        `start_page_printed` is the page number **as printed in the book**;
        `page_range` is the 0-based PDF index span the retrieval index needs.
        The two differ by the front matter's length, and the difference
        *drifts* — which is why the range is read off the running headers
        rather than computed (see `chapter_boundaries.py`).
        """
        record = {
            "number": entry.number,
            "title": entry.title,
            "authors": list(entry.authors),
            "start_page_printed": entry.start_page,
            "page_range": None,
            "page_range_confidence": None,
        }

        span = _span_for(chapter_map, entry.number)
        if span is not None:
            record["page_range"] = [span.start_index, span.end_index]
            record["page_range_confidence"] = span.confidence

        return record

    def _front_matter_pages(self, page_texts: Dict[int, str],
                            toc_pages: List[int]) -> List[ContentPage]:
        """The front-matter pages as content, straight from the text layer."""
        pages = []
        for page_number, text in sorted(page_texts.items()):
            if not text.strip():
                continue
            pages.append(ContentPage(
                page_number=page_number,
                content=text,
                metadata={
                    "source": "text_layer",
                    "role": "contents" if page_number in toc_pages else "front_matter",
                },
            ))
        return pages


# --- module helper (after the class, per CLAUDE.md) -------------------------

def _span_for(chapter_map, number):
    """The detected span for a chapter number, or None if it has none."""
    if chapter_map is None or number is None:
        return None
    for span in chapter_map.spans:
        if span.number == number:
            return span
    return None
