"""
Centralized ledger operations service.
Single source of truth for metadata-to-ledger mapping.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import os

from src.processing.ledger.csv_backend import CsvLedgerBackend
from src.processing.ledger.semantic_ledger import LedgerRecord, generate_hash, SemanticLedger
from src.processing.models import ProcessingResult

logger = logging.getLogger(__name__)

class LedgerService:
    """
    Centralized ledger management service.
    Ensures consistent field mapping across CLI and GUI.
    """

    @staticmethod
    def get_ledger_instance(
        ledger_path: Optional[Path] = None
    ) -> SemanticLedger:
        """
        Get ledger instance (singleton pattern for GUI, fresh for CLI).

        Args:
            ledger_path: Optional custom path. Defaults to SARA_LEDGER_PATH env or persistence/semantic_ledger.csv

        Returns:
            SemanticLedger instance
        """
        if ledger_path is None:
            # Check env var first
            env_path = os.getenv("SARA_LEDGER_PATH")
            if env_path:
                ledger_path = Path(env_path)
            else:
                ledger_path = Path("persistence/semantic_ledger.csv")

        return CsvLedgerBackend(str(ledger_path.resolve()))

    @staticmethod
    def create_ledger_record(
        processing_result: ProcessingResult,
        file_path: str,
        **kwargs
    ) -> LedgerRecord:
        """
        Create LedgerRecord from ProcessingResult with consistent field mapping.

        SINGLE SOURCE OF TRUTH for metadata field mapping.

        Args:
            processing_result: Pydantic ProcessingResult from orchestrator
            file_path: Path to the PDF file
            **kwargs: Optional overrides (validation_source, requires_review, etc.)

        Returns:
            LedgerRecord ready to save
        """
        metadata = processing_result.metadata
        if not metadata:
            raise ValueError(f"No metadata in processing result for {file_path}")

        doc_hash = generate_hash(file_path)

        # DEFENSIVE FIELD MAPPING - SINGLE SOURCE OF TRUTH
        # Try journal first, fallback to publisher if journal is None/empty
        journal = metadata.journal
        if not journal:
            journal = metadata.publisher

        # Handle authors (can be list or None)
        authors = metadata.authors if metadata.authors else []

        # Create record with consistent mapping
        record = LedgerRecord(
            document_id=doc_hash,
            pdf_filename=Path(file_path).name,
            doi=metadata.doi,
            title=metadata.title,
            authors=authors,
            journal=journal,  # ✅ CORRECT MAPPING with fallback
            year=metadata.year,
            validation_confidence=metadata.validation_confidence or 0.0,
            validation_source=kwargs.get('validation_source', metadata.validation_source or 'batch_processing'),
            manual_entry=False,
            requires_review=kwargs.get('requires_review', False),
            review_needed_fields=kwargs.get('review_needed_fields', []),
            review_status=kwargs.get('review_status', 'not_reviewed'),
            last_updated=datetime.now().isoformat(),
            retrieval_provenance=kwargs.get('retrieval_provenance', {
                'processed_via': 'automated',
                'processing_timestamp': datetime.now().isoformat()
            }),
            source_hash=doc_hash
        )

        return record

    @staticmethod
    def save_processing_result(
        processing_result: ProcessingResult,
        file_path: str,
        ledger: SemanticLedger,
        **kwargs
    ) -> LedgerRecord:
        """
        Save processing result to ledger with automatic commit.

        Args:
            processing_result: Result from DocumentOrchestrator
            file_path: Path to PDF
            ledger: Ledger instance to save to
            **kwargs: Optional overrides

        Returns:
            Saved LedgerRecord
        """
        record = LedgerService.create_ledger_record(
            processing_result,
            file_path,
            **kwargs
        )

        ledger.update_record(record)
        
        # Ensure commit is called if supported by backend (e.g. CsvLedgerBackend)
        if hasattr(ledger, 'commit'):
            ledger.commit()
            
        logger.info(f"💾 Saved to ledger: {Path(file_path).name} (doc_id: {record.document_id[:8]}...)")

        return record
