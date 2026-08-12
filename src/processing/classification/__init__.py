"""Document classification — deciding what kind of document a PDF is."""

from .document_classifier import (
    ClassificationOutcome,
    classify_document,
    gather_classification_signals,
)

__all__ = [
    "ClassificationOutcome",
    "classify_document",
    "gather_classification_signals",
]
