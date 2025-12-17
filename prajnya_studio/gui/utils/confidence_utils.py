"""
Confidence Utils - Phase 6.6c
Utility functions for deriving per-field confidence scores.

Purpose:
    Derives per-field confidence scores from overall validation_confidence.
    This is a heuristic-based workaround until true per-field confidence
    tracking is implemented in the backend (Phase 7).

Background:
    The backend (ExtractedMetadata) provides:
    - metadata.validation_confidence (float 0.0-1.0) - Overall confidence
    - metadata.validation_source (str) - Source of validation

    It does NOT provide per-field confidence scores like:
    - metadata.title_confidence (NOT AVAILABLE)
    - metadata.authors_confidence (NOT AVAILABLE)

    For the Metadata Confidence card in Batch Commander UI, we need
    per-field scores. This module provides heuristic derivation.

Usage:
    from prajnya_studio.gui.utils.confidence_utils import derive_field_confidence

    metadata = result['processing_result'].metadata
    field_conf = derive_field_confidence(metadata)
    # Returns: {'title': 0.90, 'authors': 0.81, 'abstract': 0.71, ...}
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def derive_field_confidence(metadata) -> Dict[str, float]:
    """
    Derive per-field confidence from overall validation confidence.

    IMPORTANT: This is a heuristic-based approximation, NOT true per-field
    confidence. The backend does not track extraction confidence per field.

    Heuristic Strategy:
        - Start with overall validation_confidence as baseline
        - Apply field-specific multipliers based on field importance
        - Handle missing fields (0.0 confidence)
        - Special handling for DOI when validation_source is 'zotero_doi_lookup'

    Field Multipliers (applied to overall confidence):
        - title: 0.95 (usually highest confidence if present)
        - authors: 0.85 (slightly lower than title)
        - abstract: 0.75 (moderate confidence, often extracted separately)
        - publication: 0.60 (lower confidence, harder to extract)
        - doi: Special case (0.95 if validated via DOI lookup, else 0.80)

    Args:
        metadata: ExtractedMetadata object from processing result

    Returns:
        Dict with keys: 'title', 'authors', 'abstract', 'publication', 'doi'
        Values are floats in range 0.0-1.0

    Example:
        >>> metadata = ExtractedMetadata(
        ...     title='Research Article',
        ...     authors=['Smith, J.'],
        ...     doi='10.1234/example',
        ...     validation_confidence=0.95,
        ...     validation_source='zotero_doi_lookup'
        ... )
        >>> derive_field_confidence(metadata)
        {'title': 0.90, 'authors': 0.81, 'abstract': 0.0, 'publication': 0.0, 'doi': 0.95}
    """
    # Get overall confidence (default to 0.0 if not set)
    overall = metadata.validation_confidence or 0.0
    validation_source = metadata.validation_source or 'unknown'

    # Initialize confidence dict
    confidence = {}

    # Title: Usually highest confidence if present
    confidence['title'] = overall * 0.95 if metadata.title else 0.0

    # Authors: Slightly lower than title
    confidence['authors'] = overall * 0.85 if metadata.authors else 0.0

    # Abstract: Often extracted separately, use moderate confidence
    # Note: Not all metadata models have 'abstract' field
    if hasattr(metadata, 'abstract') and metadata.abstract:
        confidence['abstract'] = overall * 0.75
    else:
        confidence['abstract'] = 0.0

    # Publication/Publisher: Lower confidence (harder to extract)
    confidence['publication'] = overall * 0.60 if metadata.publisher else 0.0

    # DOI: Special handling based on validation source
    if metadata.doi:
        if validation_source == 'zotero_doi_lookup':
            # DOI was validated via Zotero lookup → very high confidence
            confidence['doi'] = 0.95
        else:
            # DOI extracted but not validated → moderate confidence
            confidence['doi'] = overall * 0.80
    else:
        # DOI missing
        confidence['doi'] = 0.0

    logger.debug(
        f"Derived field confidence (overall={overall:.2f}): "
        f"title={confidence['title']:.2f}, "
        f"authors={confidence['authors']:.2f}, "
        f"doi={confidence['doi']:.2f}"
    )

    return confidence


def get_confidence_level(score: float) -> str:
    """
    Get confidence level label from score.

    Thresholds:
        - High: ≥ 0.85
        - Medium: 0.65 - 0.84
        - Low: 0.01 - 0.64
        - Missing: 0.0

    Args:
        score: Confidence score (0.0-1.0)

    Returns:
        Level string: 'High', 'Medium', 'Low', or 'Missing'
    """
    if score == 0.0:
        return 'Missing'
    elif score >= 0.85:
        return 'High'
    elif score >= 0.65:
        return 'Medium'
    else:
        return 'Low'


def get_confidence_color(score: float, theme: Dict) -> str:
    """
    Get color for confidence score based on level.

    Uses theme colors for consistency with UI design.

    Args:
        score: Confidence score (0.0-1.0)
        theme: Theme dict with color definitions

    Returns:
        Color hex string (e.g., '#10b981')
    """
    level = get_confidence_level(score)

    color_map = {
        'High': theme.get('success', '#10b981'),      # Forest Green
        'Medium': theme.get('warning', '#f59e0b'),    # Amber
        'Low': theme.get('error', '#ef4444'),         # Error Red
        'Missing': theme.get('text_muted', '#9ca3af') # Gray muted
    }

    return color_map.get(level, theme.get('text_muted', '#9ca3af'))


def format_confidence_display(score: float) -> str:
    """
    Format confidence score for UI display.

    Args:
        score: Confidence score (0.0-1.0)

    Returns:
        Formatted string (e.g., 'High (95%)', 'Medium (72%)', 'Missing')
    """
    level = get_confidence_level(score)

    if score == 0.0:
        return 'Missing'
    else:
        percentage = int(score * 100)
        return f"{level} ({percentage}%)"
