"""
Batch Ledger - Phase 6.6b
Data structure for tracking batch processing results.

Purpose:
    Stores batch processing results with unique BatchID:
    - BatchID generation (YYYYMMDD_HHMMSS_XXX format)
    - Entry tracking (pending/processing/success/error)
    - Table formatting for NiceGUI display
    - Export to JSON and CSV

Architecture:
    BatchProcessor → BatchLedger (add/update entries) → UI Table Display

Usage:
    ledger = BatchLedger()
    ledger.add_entry(file_path, 'pending')
    ledger.update_entry_status(file_path, 'success')
    rows = ledger.get_table_rows()
    ledger.export_json('output/batch_20251205_143022_ABC.json')
"""

import json
import csv
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BatchLedger:
    """
    Ledger for tracking batch processing results.

    Generates unique BatchID and maintains list of file processing records
    with status, metadata, confidence scores, and provenance flags.

    Attributes:
        batch_id: Unique batch identifier (YYYYMMDD_HHMMSS_XXX)
        entries: List of ledger entry dicts
        created_at: Timestamp when ledger was created
    """

    def __init__(self, batch_id: Optional[str] = None):
        """
        Initialize BatchLedger.

        Args:
            batch_id: Optional custom batch ID (auto-generates if None)
        """
        self.batch_id: str = batch_id or self._generate_batch_id()
        self.entries: List[Dict[str, Any]] = []
        self.created_at: datetime = datetime.now()

        logger.info(f"📊 BatchLedger initialized: {self.batch_id}")

    @staticmethod
    def _generate_batch_id() -> str:
        """
        Generate unique batch ID.

        Format: YYYYMMDD_HHMMSS_XXX
        Example: 20251205_143022_ABC

        Returns:
            Unique batch ID string
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
        batch_id = f"{timestamp}_{random_suffix}"
        logger.debug(f"Generated BatchID: {batch_id}")
        return batch_id

    def add_entry(
        self,
        file_path: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: Optional[Dict[str, float]] = None,
        provenance: str = '-',
        error: Optional[str] = None
    ) -> None:
        """
        Add new ledger entry for a file.

        Args:
            file_path: Absolute path to PDF file
            status: File status ('pending', 'processing', 'success', 'error')
            metadata: Extracted metadata dict (title, authors, doi, etc.)
            confidence: Per-field confidence scores dict
            provenance: Provenance flag ('-' normal, '△' issues detected)
            error: Error message if status='error'
        """
        entry = {
            'file_name': Path(file_path).name,
            'file_path': file_path,
            'status': status,
            'metadata': metadata or {},
            'confidence': confidence or {},
            'provenance': provenance,
            'error': error,
            'processed_at': datetime.now().isoformat()
        }

        self.entries.append(entry)
        logger.debug(f"Added ledger entry: {entry['file_name']} [{status}]")

    def update_entry_status(
        self,
        file_path: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: Optional[Dict[str, float]] = None,
        provenance: Optional[str] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update existing ledger entry.

        Finds entry by file_path and updates fields.

        Args:
            file_path: Path to identify entry
            status: New status
            metadata: Optional updated metadata
            confidence: Optional updated confidence scores
            provenance: Optional updated provenance flag
            error: Optional error message

        Returns:
            True if entry found and updated, False otherwise
        """
        for entry in self.entries:
            if entry['file_path'] == file_path:
                # Update fields
                entry['status'] = status
                entry['processed_at'] = datetime.now().isoformat()

                if metadata is not None:
                    entry['metadata'] = metadata
                if confidence is not None:
                    entry['confidence'] = confidence
                if provenance is not None:
                    entry['provenance'] = provenance
                if error is not None:
                    entry['error'] = error

                logger.debug(f"Updated ledger entry: {entry['file_name']} → {status}")
                return True

        logger.warning(f"Entry not found for update: {Path(file_path).name}")
        return False

    def get_entry(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get ledger entry by file path.

        Args:
            file_path: Path to identify entry

        Returns:
            Entry dict or None if not found
        """
        for entry in self.entries:
            if entry['file_path'] == file_path:
                return entry
        return None

    def get_table_rows(self) -> List[Dict[str, str]]:
        """
        Get formatted rows for NiceGUI table display.

        Formats entries for display with:
        - Truncated filename (max 25 chars)
        - Status icon (✓ OK, △ ERR, ↻ RUN, ○ WAIT)
        - Provenance flag

        Returns:
            List of row dicts with keys: 'filename', 'status', 'prov'
        """
        rows = []
        for entry in self.entries:
            # Map status to icon
            status_icon = {
                'success': '✓ OK',
                'error': '△ ERR',
                'processing': '↻ RUN',
                'pending': '○ WAIT'
            }.get(entry['status'], '? UNK')

            rows.append({
                'filename': self._truncate_filename(entry['file_name']),
                'status': status_icon,
                'prov': entry['provenance']
            })

        return rows

    @staticmethod
    def _truncate_filename(filename: str, max_length: int = 25) -> str:
        """
        Truncate filename for display.

        Keeps file extension and truncates middle if needed.
        Example: 'very_long_research_article.pdf' → 'very_long_...article.pdf'

        Args:
            filename: Original filename
            max_length: Maximum display length

        Returns:
            Truncated filename
        """
        if len(filename) <= max_length:
            return filename

        # Keep extension
        name, ext = Path(filename).stem, Path(filename).suffix

        # Calculate truncation
        available = max_length - len(ext) - 3  # -3 for '...'
        if available <= 0:
            return filename[:max_length]

        keep_start = available // 2
        keep_end = available - keep_start

        return f"{name[:keep_start]}...{name[-keep_end:]}{ext}"

    def export_json(self, output_path: str) -> None:
        """
        Export ledger to JSON file.

        Saves complete ledger with all entries and metadata.

        Args:
            output_path: Path to output JSON file
        """
        try:
            data = {
                'batch_id': self.batch_id,
                'created_at': self.created_at.isoformat(),
                'total_files': len(self.entries),
                'success_count': sum(1 for e in self.entries if e['status'] == 'success'),
                'error_count': sum(1 for e in self.entries if e['status'] == 'error'),
                'entries': self.entries
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Ledger exported to JSON: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export JSON: {e}", exc_info=True)
            raise

    def export_csv(self, output_path: str) -> None:
        """
        Export ledger to CSV file.

        Saves simplified table format for spreadsheet viewing.

        Args:
            output_path: Path to output CSV file
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    'Batch ID', self.batch_id,
                    'Created', self.created_at.isoformat()
                ])
                writer.writerow([])  # Blank line

                # Column headers
                writer.writerow([
                    'Filename', 'Status', 'Title', 'Authors', 'DOI',
                    'Publisher', 'Year', 'Confidence', 'Provenance', 'Error'
                ])

                # Data rows
                for entry in self.entries:
                    metadata = entry.get('metadata', {})
                    confidence = entry.get('confidence', {})

                    # Format authors list
                    authors_str = ', '.join(metadata.get('authors', []))

                    # Average confidence
                    avg_confidence = (
                        sum(confidence.values()) / len(confidence)
                        if confidence else 0.0
                    )

                    writer.writerow([
                        entry['file_name'],
                        entry['status'],
                        metadata.get('title', ''),
                        authors_str,
                        metadata.get('doi', ''),
                        metadata.get('publisher', ''),
                        metadata.get('year', ''),
                        f"{avg_confidence:.2f}",
                        entry['provenance'],
                        entry.get('error', '')
                    ])

            logger.info(f"✅ Ledger exported to CSV: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}", exc_info=True)
            raise

    def get_summary(self) -> Dict[str, Any]:
        """
        Get batch summary statistics.

        Returns:
            Dict with keys:
                - 'batch_id': str
                - 'total_files': int
                - 'success_count': int
                - 'error_count': int
                - 'pending_count': int
                - 'processing_count': int
                - 'created_at': str (ISO format)
        """
        return {
            'batch_id': self.batch_id,
            'total_files': len(self.entries),
            'success_count': sum(1 for e in self.entries if e['status'] == 'success'),
            'error_count': sum(1 for e in self.entries if e['status'] == 'error'),
            'pending_count': sum(1 for e in self.entries if e['status'] == 'pending'),
            'processing_count': sum(1 for e in self.entries if e['status'] == 'processing'),
            'created_at': self.created_at.isoformat()
        }
