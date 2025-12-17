# File: src/processing/ledger/semantic_ledger.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import hashlib
from datetime import datetime # Add datetime import

# Define the schema for a record in our ledger
class LedgerRecord(BaseModel):
    document_id: str = Field(description="Persistent primary key (e.g., source_hash).")
    pdf_filename: str 
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[int] = None
    validation_confidence: float
    validation_source: str
    manual_entry: bool = False
    requires_review: bool = Field(default=False, description="Flagged for manual review")
    review_needed_fields: List[str] = Field(
        default_factory=list,
        description="List of field names that need review (e.g., ['authors', 'doi'])"
    )
    last_reviewed_by: Optional[str] = Field(
        default=None,
        description="Username/identifier of person who last reviewed this record"
    )
    review_status: str = Field(
        default="not_reviewed",
        description="Status: 'not_reviewed', 'reviewed', 'provisional', 'final'"
    )
    
    metadata_flag: Optional[str] = None  # DEPRECATED - keep for backward compat
    last_updated: str
    retrieval_provenance: Dict[str, Any] = Field(default_factory=dict)
    source_hash: str

    # Future-proofing fields for RAG (Phase 5+)
    content_embedding_id: Optional[str] = None
    semantic_fingerprint: Optional[List[float]] = None
    chunk_index_map: List[Dict[str, Any]] = Field(default_factory=list)
    sync_status: str = "NOT_PROCESSED"

def generate_hash(file_path: str) -> str:
    """Generates a SHA256 hash for a file's content to use as a unique ID."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

class SemanticLedger(ABC):
    """
    Abstract interface for the system's long-term memory.
    Decouples the application from the underlying storage mechanism.
    """
    @abstractmethod
    def get_record(self, document_id: str) -> Optional[LedgerRecord]:
        """Retrieve a record by its unique document ID."""
        pass

    @abstractmethod
    def update_record(self, record: LedgerRecord):
        """Add or update a record in the ledger's in-memory store."""
        pass

    # --- NEW METHOD FOR BATCH PERFORMANCE (YOUR SUGGESTION) ---
    @abstractmethod
    def commit(self):
        """Persist all in-memory changes to the backend storage."""
        pass
    
    def flush(self):
        """Alias for commit() for semantic clarity."""
        self.commit()
        
    @abstractmethod
    def link_chunks(self, document_id: str, chunk_data: List[Dict[str, Any]]):
        """Link chunk metadata to a document record (for future RAG)."""
        raise NotImplementedError("Chunk linking will be implemented in a future phase.")