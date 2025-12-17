"""
Base Strategy - Abstract Interface for Document Processing
All processing strategies inherit from this base class

CRITICAL DESIGN REQUIREMENT: Strategies MUST be stateless!

Thread-Safety & Phase 3 Pipelined Architecture:
    Our system uses a GPU queue pattern where CPU workers (parallel) prepare
    documents while a GPU worker (serial) processes VLM calls:
    
    CPU Workers (Parallel):
        Thread 1: PyMuPDF (2s) → Classify (0.5s) → Queue
        Thread 2: PyMuPDF (2s) → Classify (0.5s) → Queue  } Concurrent!
        Thread 3: PyMuPDF (2s) → Classify (0.5s) → Queue  }
    
    GPU Worker (Serial):
        VLM(A) 30s → VLM(B) 30s → VLM(C) 30s (100% busy, zero downtime)
    
    Benefit: GPU never waits for CPU prep work. Even though VLM calls are
    serial, we process batches faster by eliminating GPU idle time.

Statelessness Requirements:
    1. __init__ may create reusable resources (e.g., VLM providers)
    2. process() receives ALL needed data via DocumentProfile parameter
    3. process() returns ProcessingResult WITHOUT modifying instance variables
    4. NO state carried between process() calls
    5. Multiple threads MUST be able to call process() concurrently

Validation Pattern:
    ❌ BAD (Stateful):
        def process(self, profile):
            self.current_file = profile.pdf_path  # ← Breaks thread-safety!
            self.partial_result = {...}           # ← Shared mutable state!
    
    ✅ GOOD (Stateless):
        def process(self, profile: DocumentProfile) -> ProcessingResult:
            result = self.provider.process_document(...)  # ← Uses provider
            return ProcessingResult(...)                  # ← Returns new object

Phase 3 TODO (Thread-Safety Verification):
    Before implementing pipelined batch processing, verify:
    
    1. UniversalVLMProvider thread-safety:
       - process_document() does not modify instance state ✓ (verified)
       - requests.Session is NOT thread-safe ⚠️ (known issue)
       - Solution: Thread-local sessions or per-call session creation
    
    2. Strategy thread-safety:
       - No instance variables modified in process() ✓ (enforced by design)
       - Provider reuse is safe if provider is stateless ✓ (to be verified)
    
    3. PyMuPDF thread-safety:
       - fitz.Document objects are NOT thread-safe
       - Solution: Each thread opens its own document (current design)
    
    Reference: See your analysis in Response&Suggestions.txt
"""

from abc import ABC, abstractmethod
from ..models import DocumentProfile, ProcessingResult


class DocumentProcessingStrategy(ABC):
    """
    Abstract base class for document processing strategies.
    
    Each document type (article, chapter, book) has its own
    concrete strategy implementing this interface.
    
    Design Pattern: Strategy Pattern
        - Encapsulates processing algorithm per document type
        - Allows runtime strategy selection
        - Makes adding new document types easy
        - MUST be stateless for Phase 3 parallel processing
    """
    
    @abstractmethod
    def process(self, profile: DocumentProfile) -> ProcessingResult:
        """
        Process document and return results.
        
        CRITICAL: This method MUST be stateless! Do not modify any
        instance variables. Multiple threads may call this concurrently.
        
        Args:
            profile: Complete document profile with all classification data
                Contains:
                    - pdf_path, page_count, file_size (Stage 1)
                    - document_type, classification_confidence (Stage 2)
                    - pdf_doc (runtime, attached by orchestrator)
        
        Returns:
            ProcessingResult with:
                - metadata: ExtractedMetadata (validated)
                - pages: List[ContentPage] (VLM results)
                - toc: List[str] (for books)
                - overall_quality: float
                - errors: List[str]
        
        Thread-Safety:
            This method is called by multiple threads in Phase 3.
            Do NOT modify self.* variables. Use only local variables
            and return new objects.
        
        Blueprint Reference: Section 1.2-1.4
        """
        pass
