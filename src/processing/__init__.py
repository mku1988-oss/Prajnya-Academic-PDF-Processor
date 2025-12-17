"""
The `processing` package.

This package contains the core logic for the document processing pipeline,
including the orchestrator, classifier, and processing strategies.

The main entry point for this package is the DocumentOrchestrator.
"""

# Export only the main, high-level classes from this package.
from .document_orchestrator import DocumentOrchestrator
from .document_classifier import DocumentClassifier

# Define what is exported when someone does "from src.processing import *"
__all__ = [
    'DocumentOrchestrator',
    'DocumentClassifier',
]