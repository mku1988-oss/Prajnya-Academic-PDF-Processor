# File: src/processing/ledger/csv_backend.py

import csv
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
import json
import shutil

from .semantic_ledger import SemanticLedger, LedgerRecord

class CsvLedgerBackend(SemanticLedger):
    """
    CSV-based implementation of the SemanticLedger.
    This serves as the persistent, human-readable ground-truth store.
    """
    def __init__(self, file_path: str):
        self.logger = logging.getLogger(__name__)
        self.file_path = file_path
        self.fieldnames = list(LedgerRecord.model_fields.keys())
        self._records: Dict[str, LedgerRecord] = {}
        
        self._initialize_storage()

    def commit(self):
        """Writes all records to disk, fulfilling the abstract interface."""
        temp_path = f"{self.file_path}.tmp"
        try:
            with open(temp_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

                for record in self._records.values():
                    record_dict = record.model_dump()
                    # Ensure metadata_flag exists
                    if "metadata_flag" not in record_dict:
                        record_dict["metadata_flag"] = None
                    if "pdf_filename" not in record_dict:
                        record_dict["pdf_filename"] = ""

                    # JSON-serialize complex fields
                    record_dict["authors"] = json.dumps(record_dict.get("authors", []))
                    record_dict["semantic_fingerprint"] = json.dumps(record_dict.get("semantic_fingerprint", []))
                    record_dict["chunk_index_map"] = json.dumps(record_dict.get("chunk_index_map", []))
                    record_dict["retrieval_provenance"] = json.dumps(record_dict.get("retrieval_provenance", {}))

                    # year normalization
                    if record_dict.get("year") is None:
                        record_dict["year"] = ""

                    writer.writerow(record_dict)

            # Atomic replace
            shutil.move(temp_path, self.file_path)
        except Exception as e:
            self.logger.error(f"Failed to commit ledger: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _initialize_storage(self):
        """Load existing data from the CSV file or create it if it doesn't exist."""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            if not os.path.exists(self.file_path):
                with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.fieldnames)
                self.logger.info(f"Created new semantic ledger file at {self.file_path}")
            else:
                self._load()
        except IOError as e:
            self.logger.error(f"Failed to initialize ledger storage at {self.file_path}: {e}")

    def _load(self):
        """Loads all records from the CSV into an in-memory dictionary."""
        try:
            with open(self.file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # ================================
                        # ================================

                        # -------- retrieval_provenance --------
                        rp = row.get("retrieval_provenance")
                        if rp is None:
                            row["retrieval_provenance"] = {}
                        elif isinstance(rp, str):
                            try:
                                row["retrieval_provenance"] = json.loads(rp or "{}")
                            except Exception:
                                row["retrieval_provenance"] = {}

                        # -------- semantic_fingerprint (FULL FIX) --------
                        sf = row.get("semantic_fingerprint")

                        if sf is None or (isinstance(sf, str) and sf.strip().lower() in ("", "none", "null")):
                            row["semantic_fingerprint"] = []
                        else:
                            try:
                                parsed = json.loads(sf)
                                row["semantic_fingerprint"] = parsed if isinstance(parsed, list) else []
                            except Exception:
                                row["semantic_fingerprint"] = []

                        # -------- chunk_index_map --------
                        cmap = row.get("chunk_index_map")
                        if cmap is None:
                            row["chunk_index_map"] = []
                        elif isinstance(cmap, str):
                            try:
                                row["chunk_index_map"] = json.loads(cmap or "[]")
                            except Exception:
                                row["chunk_index_map"] = []

                        # -------- authors --------
                        authors = row.get("authors")
                        if authors is None:
                            row["authors"] = []
                        elif isinstance(authors, str):
                            try:
                                row["authors"] = json.loads(authors)
                            except Exception:
                                # Fallback simple parsing ("['a','b']")
                                row["authors"] = [
                                    a.strip().strip("'").strip('"')
                                    for a in authors.strip("[]").split(",")
                                    if a.strip()
                                ]

                        # -------- review_needed_fields (NEW) --------
                        rnf = row.get("review_needed_fields")
                        if rnf is None:
                            row["review_needed_fields"] = []
                        elif isinstance(rnf, str):
                            try:
                                row["review_needed_fields"] = json.loads(rnf or "[]")
                            except Exception:
                                row["review_needed_fields"] = []

                        # -------- year normalization --------
                        year_val = row.get("year")
                        if isinstance(year_val, str):
                            row["year"] = int(year_val) if year_val.isdigit() else None

                        # -------- sanitize blank optional strings --------
                        for opt in (
                            "doi", "title", "journal", "publisher",
                            "validation_source", "pdf_filename",
                            "metadata_flag"
                        ):
                            val = row.get(opt)
                            if isinstance(val, str) and val.strip() == "":
                                row[opt] = None

                        # ================================
                        # CONSTRUCT RECORD
                        # ================================
                        record = LedgerRecord(**row)
                        self._records[record.document_id] = record

                    except Exception as e:
                        self.logger.warning(
                            f"Skipping malformed row in ledger: {row}. Error: {e}"
                        )

            self.logger.info(
                f"Loaded {len(self._records)} records from the semantic ledger."
            )

        except IOError as e:
            self.logger.error(
                f"Failed to load ledger from {self.file_path}: {e}"
            )

    def get_record(self, document_id: str) -> Optional[LedgerRecord]:
        """Retrieve a record from the in-memory cache."""
        return self._records.get(document_id)
    
    def get_all_records(self) -> List[LedgerRecord]:
        """Return all records from the in-memory cache (for --review mode)."""
        return list(self._records.values())

    # MODIFIED: This now only updates the in-memory dictionary.
    def update_record(self, record: LedgerRecord):
        """Add or update a record in the in-memory ledger store."""
        if not record.last_updated: # Ensure last_updated is always set
            record.last_updated = datetime.utcnow().isoformat()
        self._records[record.document_id] = record
        self.logger.info(f"Updated in-memory record for document_id: {record.document_id}")

    def link_chunks(self, document_id: str, chunk_data: List[Dict[str, Any]]):
        super().link_chunks(document_id, chunk_data)