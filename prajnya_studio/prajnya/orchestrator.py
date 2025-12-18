# prajnya_studio/prajnya/orchestrator.py
"""  
Orchestrator Adapter - Sprint 2: Real Backend Integration
Bridges NiceGUI async interface with synchronous DocumentOrchestrator
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime

# Add parent directory to Python path for src/ imports
studio_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(studio_root))

# Import REAL backend from src/processing/
try:
    from src.processing.document_orchestrator import DocumentOrchestrator
    from src.processing.ledger.csv_backend import CsvLedgerBackend  
    from src.processing.ledger.semantic_ledger import LedgerRecord, generate_hash
    BACKEND_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Backend import failed: {e}. Using simulation mode.")
    BACKEND_AVAILABLE = False

logger = logging.getLogger(__name__)

# Singleton orchestrator instance for performance
_orchestrator_instance = None
_ledger_instance = None

def get_orchestrator() -> 'DocumentOrchestrator':
    """Get or create singleton DocumentOrchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None and BACKEND_AVAILABLE:
        _orchestrator_instance = DocumentOrchestrator()
        logger.info("Initialized real DocumentOrchestrator")
    return _orchestrator_instance

def get_ledger() -> 'CsvLedgerBackend':
    """Get or create singleton CsvLedgerBackend."""
    global _ledger_instance
    if _ledger_instance is None and BACKEND_AVAILABLE:
        ledger_path = studio_root / "persistence" / "semantic_ledger.csv"
        _ledger_instance = CsvLedgerBackend(str(ledger_path))
        logger.info(f"Initialized CsvLedgerBackend: {ledger_path}")
    return _ledger_instance

def reset_orchestrator():
    """
    Reset the orchestrator singleton to force reinitialization.

    This enables hot-reload functionality by clearing the cached instance.
    The next call to get_orchestrator() will create a fresh instance
    that loads the updated strategy configuration from YAML files.

    Sprint 3.2 Feature: Hot-reload for Strategy Lab

    Usage:
        # In Strategy Lab after saving config
        reset_orchestrator()
        orchestrator = get_orchestrator()  # Fresh instance with new config
    """
    global _orchestrator_instance

    if _orchestrator_instance is not None:
        logger.info("🔄 Resetting DocumentOrchestrator singleton for hot-reload")

        # Clean up any resources if needed
        # (Currently DocumentOrchestrator has no cleanup needed)

        # Clear the singleton
        _orchestrator_instance = None

        logger.info("✅ Orchestrator reset complete - next call will load fresh config")
    else:
        logger.debug("No orchestrator instance to reset")

async def run_batch(
    folder_path: str,
    recursive: bool,
    skip_reviewed: bool,
    job_id: str,
    files: Optional[List[str]] = None,
    on_progress: Optional[Callable[[str, str, float, Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Real batch processing using DocumentOrchestrator.
    Wraps synchronous backend in async context for non-blocking GUI.
    
    Args:
        folder_path: Directory containing PDFs
        recursive: Search subdirectories
        skip_reviewed: Skip files with reviewed status
        job_id: Unique job identifier
        files: Optional explicit file list (drag-and-drop)
        on_progress: Callback for UI updates
        
    Returns:
        Dict with 'processed' and 'failed' lists
    """
    if not BACKEND_AVAILABLE:
        logger.error("⚠️ Backend not available - check imports")
        return {"processed": [], "failed": [], "error": "Backend import failed"}
    
    logger.info(f"🚀 Starting Real Batch Job: {job_id}")
    logger.info(f"📂 Target: {folder_path} (Recursive: {recursive}, Skip Reviewed: {skip_reviewed})")
    
    orchestrator = get_orchestrator()
    ledger = get_ledger()
    
    if not orchestrator or not ledger:
        logger.error("Failed to initialize orchestrator or ledger")
        return {"processed": [], "failed": [], "error": "Initialization failed"}
    
    # Determine file list
    if files:
        target_files = [Path(f) for f in files]
        logger.info(f"Using uploaded files: {len(target_files)} PDFs")
    elif folder_path:
        p = Path(folder_path)
        if not p.exists():
            logger.error(f"Folder not found: {folder_path}")
            return {"processed": [], "failed": [], "error": "Folder not found"}
        
        if recursive:
            target_files = list(p.rglob("*.pdf"))
        else:
            target_files = list(p.glob("*.pdf"))
        logger.info(f"Found {len(target_files)} PDFs in folder")
    else:
        logger.warning("No folder path or files provided")
        return {"processed": [], "failed": [], "error": "No input provided"}
    
    total = len(target_files)
    results = {"processed": [], "failed": []}
    
    for idx, pdf_path in enumerate(target_files):
        try:
            logger.info(f"Processing {idx+1}/{total}: {pdf_path.name}")
            
            # Check ledger if skip_reviewed enabled
            if skip_reviewed:
                file_hash = generate_hash(str(pdf_path))
                record = ledger.get_record(file_hash)  # document_id IS the hash
                if record and record.review_status == "reviewed":
                    logger.info(f"⏭️  Skipping reviewed: {pdf_path.name}")
                    continue
            
            # REAL PROCESSING - Run in executor for non-blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default ThreadPoolExecutor
                orchestrator.process_document,
                str(pdf_path),
                None  # pages_to_process (None = all pages)
            )
            
            logger.info(f"✅ Successfully processed: {pdf_path.name}")
            results["processed"].append(str(pdf_path))
            
            # Update ledger FIRST (before UI callback that might fail)
            file_hash = generate_hash(str(pdf_path))
            
            # Extract metadata from result dict -> processing_result -> metadata
            processing_result = result.get('processing_result')
            metadata = processing_result.metadata if processing_result else None
            
            ledger_record = LedgerRecord(
                document_id=file_hash,
                pdf_filename=pdf_path.name,
                source_hash=file_hash,
                doi=metadata.doi if metadata else None,
                title=metadata.title if metadata else None,
                authors=metadata.authors if metadata else [],
                journal=metadata.journal if (metadata and hasattr(metadata, 'journal') and metadata.journal) else (metadata.publisher if metadata else None),
                year=metadata.year if metadata else None,
                validation_confidence=metadata.validation_confidence if metadata else 0.0,
                validation_source=metadata.validation_source if metadata else "unknown",
                last_updated=datetime.utcnow().isoformat(),
                review_status="not_reviewed",
                retrieval_provenance={
                    "document_type": result.get('document_type', 'unknown'),
                    "processing_time": result.get('processing_time', 0)
                }
            )
            ledger.update_record(ledger_record)
            
            # Progress callback (called AFTER ledger update, not critical if it fails)
            if on_progress:
                try:
                    progress = (idx + 1) / total
                    on_progress(job_id, pdf_path.name, progress, {"status": "complete"})
                except Exception as e:
                    logger.warning(f"Progress callback failed (non-critical): {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed: {pdf_path.name} - {str(e)}")
            results["failed"].append(str(pdf_path))
    
    # Commit ledger updates
    ledger.commit()
    
    logger.info(f"🏁 Batch complete. Processed: {len(results['processed'])}, Failed: {len(results['failed'])}")
    return results

def get_job_status(job_id: str):
    # Stub for future status polling
    return {"status": "unknown"}