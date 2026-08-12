#!/usr/bin/env python3
"""
Page Classifier — decide, cheaply and on the CPU, how a page should be read.

For biology researchers: not every page of a paper needs the vision model. A
pure-prose page can be read straight from the PDF's embedded text layer in
milliseconds; a page with figures or dense equations needs the (expensive) VLM.
This classifier looks at each page's text, images, and vector drawings and
routes it:

- TEXT_ONLY: pure text → read from the text layer, no VLM call at all
- SIMPLE_VISUAL: figures/tables present → VLM read
- COMPLEX_VISUAL: equation-heavy → VLM read (same path today; kept distinct
  because the signals are different and Stage 3 may treat them differently)

Uses multi-signal analysis: text density, visual element detection (embedded
images + clustered vector drawings), and LaTeX/equation detection (4 methods).

Ported from V5's `page_classifier.py`. Deliberate changes:
- All page access goes through the `PdfBackend` seam — no `fitz`.
- `PageType` now lives here (V5's strategy already imported it from this
  module; the rest of V5's `models.py` is not needed yet).
- `_verify_low_confidence_regions` (VLM verification of equation candidates)
  is dropped: it was on plan.md's DELETE list — a dead path no caller ever
  activated.
- Equation detection works on lines rebuilt from word boxes rather than
  PyMuPDF text *blocks* (the seam has no block concept). Methods 1 and 2 count
  patterns per line instead of per block. Method 3 needed care: V5's
  "the block holds one line" test was really an ISOLATION test, and dropping
  it made every centered table cell score as an equation — see the comment at
  method 3. Thresholds and confidences are otherwise unchanged.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.processing.pdf.base import (
    PdfPage,
    Rect,
    group_words_into_lines,
    line_rect,
    line_text,
)
from src.processing.utils.visual_clusterer import VisualRegionClusterer


#: Shortest centered line that can plausibly be a display equation. Table
#: cells are frequently one or two characters ("1", "47"); a real equation
#: carries at least an operand, an operator, and a result.
MIN_EQUATION_LINE_CHARS = 5


class PageType(str, Enum):
    """
    Page complexity classification for tiered processing.

    - TEXT_ONLY: pure text pages → text-layer read (no VLM)
    - SIMPLE_VISUAL: simple figures/tables → VLM read
    - COMPLEX_VISUAL: complex diagrams/equations → VLM read
    """
    TEXT_ONLY = "text_only"
    SIMPLE_VISUAL = "simple_visual"
    COMPLEX_VISUAL = "complex_visual"


class PageClassifier:
    """
    Classifies PDF pages by complexity for intelligent routing.

    Uses confidence-based multi-signal analysis to determine whether a page
    can be processed quickly from the text layer or requires the VLM.

    Classification Rules (identical to V5):
    1. Page 0 (index) → title page, handled specially by the strategy
    2. High equation confidence → COMPLEX_VISUAL
    3. Many visual elements → SIMPLE_VISUAL
    4. High text density + no visuals → TEXT_ONLY
    5. Default → SIMPLE_VISUAL (safer fallback)

    Attributes:
        confidence_threshold: minimum text density for TEXT_ONLY (default 0.85)
        visual_threshold: number of visuals to classify as SIMPLE_VISUAL
            (default 2)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize page classifier with configuration.

        Args:
            config: configuration dict with a 'hybrid_processing' section
                containing 'page_classifier' settings (same YAML shape as V5's
                VLM profiles).
        """
        self.logger = logging.getLogger("PageClassifier")

        hybrid_config = config.get('hybrid_processing', {})
        classifier_config = hybrid_config.get('page_classifier', {})

        # Core thresholds
        self.confidence_threshold = classifier_config.get('confidence_threshold', 0.85)
        self.visual_threshold = classifier_config.get('visual_threshold', 2)
        # A candidate visual region covered by MORE than this fraction of word
        # boxes is text with rectangles drawn on it, not a figure. See
        # `_is_text_dense`: measured populations are 0-8.3% (figures) against
        # 51.3-71.5% (link decorations), so this sits in an empty band.
        self.text_coverage_max = classifier_config.get('text_coverage_max', 0.30)

        # Equation detection thresholds
        eq_config = classifier_config.get('equation', {})
        self.eq_total_conf_threshold = eq_config.get('total_confidence_threshold', 1.5)
        self.eq_high_conf_count = eq_config.get('high_confidence_count', 2)
        self.eq_high_conf_min = eq_config.get('high_confidence_min', 0.85)

        # Vector-drawing clustering (see visual_clusterer.py). V5's DPI knobs
        # were a no-op on the live path; eps is in PDF points here.
        cluster_config = hybrid_config.get('visual_clustering', {})
        self.visual_clusterer = VisualRegionClusterer(
            eps=float(cluster_config.get('dbscan_eps', 50)),
            min_samples=int(cluster_config.get('dbscan_min_samples', 2)),
        )

        self.logger.info(f"PageClassifier initialized (confidence={self.confidence_threshold})")

    def classify(
        self,
        page: PdfPage,
        page_idx: int,
        text_stats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[PageType, Dict[str, Any]]:
        """
        Classify a page by complexity.

        Args:
            page: the page, via the PdfBackend seam
            page_idx: 0-based page index
            text_stats: optional pre-computed text statistics (see
                `extract_text_stats`); without them Rule 4 (TEXT_ONLY) cannot
                fire. V5 called this parameter `pymupdf_result`.

        Returns:
            Tuple of (PageType, analysis_dict).

            **The analysis always reports what is ON the page, whichever rule
            decided the type.** `visual_regions` in particular is filled in
            before any rule can return, because a caller reading it cannot
            otherwise tell "there are no figures here" from "nobody looked".

            That distinction was not academic: the equation rule used to return
            before the visual scan ran, so every equation-bearing page reported
            zero visual regions — and `body_pipeline` skips layout detection
            when a page has no visuals. On one 20-page paper that silently cost
            **14 figures across 8 pages**, all of which the detector finds and
            crops perfectly well once it is asked.

        Example:
            >>> page_type, analysis = classifier.classify(page, 3, extract_text_stats(page))
            >>> print(page_type)  # PageType.TEXT_ONLY
            >>> print(analysis['classification_reason'])  # 'high text density (0.92)'
        """
        analysis = {
            'page_index': page_idx,
            'equation_regions': [],
            'visual_regions': [],
            'text_density': 0.0,
            'classification_reason': '',
        }

        # RULE 1: First page is always the title page
        if page_idx == 0:
            analysis['classification_reason'] = 'first_page_title'
            self.logger.debug("Page 1 classified as TITLE_PAGE (always)")
            return PageType.TEXT_ONLY, analysis  # title page handled as text

        # Both scans run BEFORE any rule returns. The rule ORDER below is
        # unchanged — equations still win, and the page type is exactly what it
        # always was — but the analysis is now complete whichever rule fires,
        # so a downstream consumer is never handed an empty list that merely
        # means "not looked". Both scans are CPU-only and cost ~1ms a page.
        equation_regions = self._detect_equation_regions(page)
        visual_regions, text_dense_regions = self._scan_visuals(page)
        analysis['equation_regions'] = equation_regions
        analysis['visual_regions'] = visual_regions
        analysis['visual_count'] = len(visual_regions)
        # Regions that LOOKED visual but were discarded as text with boxes
        # drawn on it. Reported separately because discarding is a judgement:
        # a caller about to send this page down the free path can check that
        # judgement with a real detector before trusting it. See
        # `body_pipeline._detections_the_filter_hid`.
        analysis['text_dense_regions'] = text_dense_regions

        # RULE 2: Detect equations (high priority - complex content)

        total_confidence = sum(eq['confidence'] for eq in equation_regions)
        high_confidence_count = sum(
            1 for eq in equation_regions
            if eq['confidence'] >= self.eq_high_conf_min
        )

        analysis['equation_total_confidence'] = total_confidence
        analysis['equation_high_confidence_count'] = high_confidence_count

        if (total_confidence > self.eq_total_conf_threshold or
                high_confidence_count >= self.eq_high_conf_count):

            analysis['classification_reason'] = (
                f'{len(equation_regions)} equations '
                f'(conf={total_confidence:.2f}, high_conf={high_confidence_count})'
            )
            self.logger.debug(
                f"Page {page_idx+1} classified as COMPLEX_VISUAL: {analysis['classification_reason']}"
            )
            return PageType.COMPLEX_VISUAL, analysis

        # RULE 3: Detect visual elements (figures/tables)
        if len(visual_regions) >= self.visual_threshold:
            analysis['classification_reason'] = f'{len(visual_regions)} visual elements detected'
            self.logger.debug(
                f"Page {page_idx+1} classified as SIMPLE_VISUAL ({len(visual_regions)} regions)"
            )
            return PageType.SIMPLE_VISUAL, analysis

        # RULE 4: Check text density for TEXT_ONLY classification
        if text_stats:
            text_density = self._calculate_text_density(text_stats)
            analysis['text_density'] = text_density

            # High text density + no visuals + no equations = TEXT_ONLY
            if (text_density >= self.confidence_threshold and
                    len(visual_regions) == 0 and
                    len(equation_regions) == 0):

                analysis['classification_reason'] = f'high text density ({text_density:.2f})'
                self.logger.debug(f"Page {page_idx+1} classified as TEXT_ONLY")
                return PageType.TEXT_ONLY, analysis

        # RULE 5: Default to SIMPLE_VISUAL (safer fallback - uses VLM)
        analysis['classification_reason'] = 'mixed_content_fallback'
        self.logger.debug(f"Page {page_idx+1} classified as SIMPLE_VISUAL (mixed content)")
        return PageType.SIMPLE_VISUAL, analysis

    # --- helpers (after the main function, per CLAUDE.md) ------------------

    def _detect_equation_regions(self, page: PdfPage) -> List[Dict[str, Any]]:
        """
        Detect mathematical equations using a multi-method approach.

        Methods (confidences unchanged from V5):
        1. LaTeX syntax in the text layer (high confidence)
        2. Unicode math symbols (medium confidence)
        3. Centered short single lines (low confidence)
        4. Small embedded equation-shaped images (high confidence)

        Args:
            page: the page, via the PdfBackend seam

        Returns:
            List of equation region dicts with keys:
                - bbox: (x0, y0, x1, y1)
                - confidence: 0.0-1.0
                - method: detection method name
                - text: extracted text (if available)
        """
        equations: List[Dict[str, Any]] = []

        lines = group_words_into_lines(page.words())
        line_rects = [line_rect(line) for line in lines]
        page_width, _ = page.size

        latex_patterns = [
            r'\\frac\{',
            r'\\int_',
            r'\\sum_',
            r'\\prod_',
            r'\\sqrt\{',
            r'\\alpha|\\beta|\\gamma|\\delta',
            r'\^[{].*?[}]',  # Superscripts
            r'_[{].*?[}]',   # Subscripts
        ]
        math_unicode_ranges = [
            ('\u2200', '\u22FF'),  # Mathematical Operators
            ('\u2A00', '\u2AFF'),  # Supplemental Math Operators
            ('\u27C0', '\u27EF'),  # Miscellaneous Math Symbols-A
            ('\u2980', '\u29FF'),  # Miscellaneous Math Symbols-B
        ]

        for line_index, line in enumerate(lines):
            text = line_text(line)
            rect = line_rects[line_index]
            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)

            # Method 1: LaTeX syntax
            latex_match_count = sum(
                1 for pattern in latex_patterns if re.search(pattern, text)
            )
            if latex_match_count >= 2:  # at least 2 LaTeX patterns
                equations.append({
                    'bbox': bbox,
                    'confidence': min(0.95, 0.70 + latex_match_count * 0.05),
                    'method': 'latex_syntax',
                    'text': text[:100],
                })

            # Method 2: Unicode math symbols
            math_char_count = sum(
                1 for char in text
                for start, end in math_unicode_ranges
                if start <= char <= end
            )
            if math_char_count >= 3:  # at least 3 math symbols
                equations.append({
                    'bbox': bbox,
                    'confidence': min(0.85, 0.60 + math_char_count * 0.05),
                    'method': 'unicode_math',
                    'text': text[:100],
                })

            # Method 3: an isolated, genuinely centered, short line.
            #
            # The weakest signal (0.40) and the one that needed real care to
            # port. Two guards beyond V5's page-center test, both measured
            # against the benchmark's two-table page:
            #
            # - ISOLATION. V5 required a whole PyMuPDF text BLOCK to hold one
            #   line; blocks are paragraphs, so that was really "this line
            #   stands alone". Dropped, every short centered cell of a table
            #   scores: p4 produced 32 false equations (12.8 total against a
            #   1.5 threshold) and was misrouted to COMPLEX_VISUAL before the
            #   visual rule could run.
            # - COLUMN ALIGNMENT. The page-center test alone is unsound on a
            #   two-column journal: this page is 595pt wide, so the ±20% band
            #   spans 178-416pt, and the RIGHT COLUMN begins at x=312 — inside
            #   it. Every short right-column line therefore looked "centered".
            #   A line starting at a column's left margin is left-aligned, not
            #   centered, whatever its midpoint says.
            line_center = (rect.x0 + rect.x1) / 2
            is_centered = abs(line_center - page_width / 2) < page_width * 0.2
            is_short = len(text.strip()) < 80
            is_substantial = len(text.strip()) >= MIN_EQUATION_LINE_CHARS
            if (is_centered and is_short and is_substantial
                    and not _starts_at_a_column_margin(rect, line_rects)
                    and _is_alone_on_its_line(line_index, line_rects)):
                equations.append({
                    'bbox': bbox,
                    'confidence': 0.40,  # low confidence
                    'method': 'centered_text',
                    'text': text.strip(),
                })

        # Method 4: Embedded equation images — small, wide, centered rasters.
        for index, image in enumerate(page.images()):
            rect = image.rect
            aspect_ratio = rect.width / rect.height if rect.height > 0 else 0

            is_small = rect.width < page_width * 0.5 and rect.height < 100
            is_horizontal = 0.5 < aspect_ratio < 10

            if is_small and is_horizontal:
                equations.append({
                    'bbox': (rect.x0, rect.y0, rect.x1, rect.y1),
                    'confidence': 0.75,
                    'method': 'embedded_image',
                    'text': f'[equation_image_{index}]',
                })

        return equations

    def _detect_visual_regions(self, page: PdfPage) -> List[Dict[str, Any]]:
        """The visual regions this page really has (see `_scan_visuals`)."""
        visuals, _discarded_as_text = self._scan_visuals(page)
        return visuals

    def _scan_visuals(
        self, page: PdfPage,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Detect visual elements (images and clustered vector drawings).

        Smart filtering (thresholds unchanged from V5, in PDF points):
        - headers/footers (top/bottom 10% of page) are ignored
        - small decorative elements (< 5000 pt² area) are ignored
        - thin lines (aspect ratio > 10:1 or < 1:10) are ignored
        - tiny images (< 50 pt width or height) are ignored

        Returns:
            (visual regions, regions discarded as text decoration).

            The second list is deliberately NOT thrown away. Every other
            filter here rejects on geometry — a hairline, a page-number-sized
            speck — and cannot plausibly be a figure. `_is_text_dense` rejects
            on a *measurement* of the publisher's typesetting, and a publisher
            who sets a sequence alignment as real text inside a figure would
            measure like decoration. Handing those regions back lets the caller
            spend ~600ms of CPU checking, instead of losing the figure in
            silence.
        """
        visuals: List[Dict[str, Any]] = []
        discarded_as_text: List[Dict[str, Any]] = []

        page_width, page_height = page.size
        header_zone = page_height * 0.10  # top 10%
        footer_zone = page_height * 0.90  # bottom 10%

        MIN_BBOX_AREA = 5000   # points²
        MIN_IMAGE_SIZE = 50    # minimum width or height in points
        MAX_ASPECT_RATIO = 10.0  # filter out thin lines

        # --- Step 1: embedded raster images -------------------------------
        images = page.images()
        filtered_count = 0
        for image in images:
            rect = image.rect
            if self._is_filtered(rect, header_zone, footer_zone,
                                 MIN_BBOX_AREA, MAX_ASPECT_RATIO,
                                 min_size=MIN_IMAGE_SIZE):
                filtered_count += 1
                continue
            visuals.append({
                'bbox': (rect.x0, rect.y0, rect.x1, rect.y1),
                'type': 'image',
                'size': (rect.width, rect.height),
            })

        # --- Step 2: clustered vector drawings ----------------------------
        drawing_rects = [d.rect for d in page.drawings()]
        clusters = self.visual_clusterer.cluster_drawings(drawing_rects)

        drawing_filtered_count = 0
        for cluster in clusters:
            rect = cluster.rect
            if self._is_filtered(rect, header_zone, footer_zone,
                                 MIN_BBOX_AREA, MAX_ASPECT_RATIO):
                drawing_filtered_count += 1
                continue
            if self._is_text_dense(page, rect):
                drawing_filtered_count += 1
                discarded_as_text.append({
                    'bbox': (rect.x0, rect.y0, rect.x1, rect.y1),
                    'type': 'text_dense_cluster',
                    'component_count': cluster.component_count,
                })
                continue
            visuals.append({
                'bbox': (rect.x0, rect.y0, rect.x1, rect.y1),
                'type': 'drawing_cluster',
                'component_count': cluster.component_count,
            })

        self.logger.info(
            f"Visual detection: {len(images)} images ({filtered_count} filtered), "
            f"{len(clusters)} drawing clusters ({drawing_filtered_count} filtered, "
            f"{len(discarded_as_text)} as text decoration) → "
            f"{len(visuals)} valid visual regions"
        )
        return visuals, discarded_as_text

    def _is_text_dense(self, page: PdfPage, rect: Rect) -> bool:
        """
        True when a candidate region is really just text with boxes drawn on it.

        For biology researchers: a reference list draws a little rectangle
        around every "CrossRef" and "PubMed" link. Cluster those and you get
        something shaped exactly like a figure — which sent seven bibliography
        pages of one paper to the GPU, and then to a second model to check.

        The test is DENSITY, not presence, because real figures contain text:
        panel letters, axis labels, scale bars, legends. Measured over labelled
        regions from the sample corpus, the two populations do not overlap —
        they are not even close:

            decoration (32 link clusters)   51.3% - 71.5% covered by words
            real figures (43 regions)        0.0% -  8.3%

        A figure's own labels cover a few percent of it; a link box sits
        directly on body text. `text_coverage_max` is set in the wide empty
        band between them, so the threshold is not delicate.

        Args:
            page: the page, for its word boxes.
            rect: the candidate region.

        Returns:
            Whether to discard this region as text decoration. Any failure to
            measure answers False — keeping a doubtful region costs a VLM read,
            while dropping a real figure loses it silently.
        """
        area = abs((rect.x1 - rect.x0) * (rect.y1 - rect.y0))
        if area <= 0:
            return False

        try:
            words = page.words()
        except Exception:                                     # pragma: no cover
            return False

        covered = 0.0
        for word in words:
            box = word.rect
            x0, y0 = max(rect.x0, box.x0), max(rect.y0, box.y0)
            x1, y1 = min(rect.x1, box.x1), min(rect.y1, box.y1)
            if x1 > x0 and y1 > y0:
                covered += (x1 - x0) * (y1 - y0)
                if covered / area > self.text_coverage_max:
                    return True           # decided; no need to keep counting

        return False

    def _is_filtered(
        self,
        rect: Rect,
        header_zone: float,
        footer_zone: float,
        min_area: float,
        max_aspect: float,
        min_size: Optional[float] = None,
    ) -> bool:
        """Apply the shared visual-region filters. True means: discard."""
        center_y = (rect.y0 + rect.y1) / 2
        if center_y < header_zone or center_y > footer_zone:
            self.logger.debug(f"Filtered visual in header/footer zone: y={center_y:.0f}")
            return True
        if min_size is not None and (rect.width < min_size or rect.height < min_size):
            self.logger.debug(f"Filtered tiny visual: {rect.width:.0f}x{rect.height:.0f}")
            return True
        if rect.area < min_area:
            self.logger.debug(f"Filtered small visual: area={rect.area:.0f}pt²")
            return True
        aspect_ratio = rect.width / rect.height if rect.height > 0 else 0
        if aspect_ratio > max_aspect or aspect_ratio < (1 / max_aspect):
            self.logger.debug(f"Filtered thin-line visual: aspect={aspect_ratio:.2f}")
            return True
        return False

    def _calculate_text_density(self, text_stats: Dict[str, Any]) -> float:
        """
        Calculate text density score (0.0-1.0).

        High density indicates substantial, well-structured text suitable for
        a text-layer read. Low density indicates sparse text (captions,
        labels) requiring the VLM.

        Args:
            text_stats: text statistics dict with keys:
                - word_count: number of words
                - avg_words_per_line: average words per line
                - char_count: total characters

        Returns:
            Density score from 0.0 (empty) to 1.0 (dense text)
        """
        word_count = text_stats.get('word_count', 0)
        avg_words_per_line = text_stats.get('avg_words_per_line', 0)
        char_count = text_stats.get('char_count', 0)

        # Component scores
        word_score = min(1.0, word_count / 300)        # 300+ words = max score
        line_score = min(1.0, avg_words_per_line / 10)  # 10+ words/line = max
        char_score = min(1.0, char_count / 2000)        # 2000+ chars = max

        # Weighted combination
        return word_score * 0.4 + line_score * 0.3 + char_score * 0.3


#: How close a line's left edge must be to a shared margin to count as
#: left-aligned. Justification is typically exact, so this only absorbs
#: sub-point rounding in the backends' word boxes.
COLUMN_MARGIN_TOLERANCE_PT = 3.0

#: How many lines must share a left edge before it counts as a column margin
#: rather than a coincidence.
MIN_LINES_PER_COLUMN = 3


def _starts_at_a_column_margin(rect: Rect, line_rects: List[Rect]) -> bool:
    """
    Does this line begin at a left margin shared by many other lines?

    That is the signature of ordinary justified body text, including section
    headings — it is left-aligned, not centered, no matter where its midpoint
    falls. The margins are discovered from the page itself rather than assumed,
    so it works for one-column and two-column layouts alike: any left edge that
    at least `MIN_LINES_PER_COLUMN` lines share is a column margin.
    """
    shared_edges = {}
    for other in line_rects:
        key = round(other.x0 / COLUMN_MARGIN_TOLERANCE_PT)
        shared_edges[key] = shared_edges.get(key, 0) + 1

    return any(
        count >= MIN_LINES_PER_COLUMN
        and abs(rect.x0 - key * COLUMN_MARGIN_TOLERANCE_PT) <= COLUMN_MARGIN_TOLERANCE_PT
        for key, count in shared_edges.items()
    )


def _is_alone_on_its_line(index: int, line_rects: List[Rect]) -> bool:
    """
    Does this line stand alone at its height, with no neighbour beside it?

    A display equation occupies its own line across the column. A table row,
    by contrast, is several short lines sharing one height — the line splitter
    cuts a row at its column gaps. Testing for a vertically-overlapping
    neighbour therefore separates the two cleanly, and is the geometric
    equivalent of V5's "the text block holds a single line" condition.
    """
    target = line_rects[index]
    for other_index, other in enumerate(line_rects):
        if other_index == index:
            continue
        overlap = min(target.y1, other.y1) - max(target.y0, other.y0)
        if overlap > 0.5 * min(target.height, other.height):
            return False
    return True


def extract_text_stats(page: PdfPage) -> Dict[str, Any]:
    """
    Text statistics for the density score — the classifier's Rule 4 input.

    This is V5's `_extract_pymupdf_stats`, moved next to its only consumer and
    renamed because it is no longer PyMuPDF-specific: it works on whatever the
    active `PdfBackend` reports.

    Args:
        page: the page, via the PdfBackend seam

    Returns:
        Dict with keys:
            - word_count: total number of words
            - avg_words_per_line: average words per line
            - char_count: total character count
            - line_count: number of non-blank text lines
    """
    try:
        text = page.text()
        words = text.split()
        lines = [line for line in text.split('\n') if line.strip()]

        word_count = len(words)
        line_count = len(lines)
        avg_words_per_line = word_count / line_count if line_count > 0 else 0

        return {
            'word_count': word_count,
            'char_count': len(text),
            'line_count': line_count,
            'avg_words_per_line': round(avg_words_per_line, 2),
        }
    except Exception as error:
        logging.getLogger("PageClassifier").warning(
            f"Failed to extract text stats: {error}"
        )
        return {
            'word_count': 0, 'char_count': 0,
            'line_count': 0, 'avg_words_per_line': 0.0,
        }
