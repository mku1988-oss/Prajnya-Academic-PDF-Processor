"""
Batch Processor - Phase 6.6a
Orchestrates batch PDF processing with progress tracking and UI callbacks.

Purpose:
    Manages the workflow of processing multiple PDFs sequentially:
    - Queue management for uploaded files
    - Async processing loop (non-blocking GUI)
    - Progress tracking and calculation
    - Status management (idle/processing/paused/completed/error)
    - UI update callbacks for real-time feedback

Architecture:
    UI → BatchProcessor → Orchestrator → Results
              ↓
         Callbacks (progress, file_start, file_complete, file_error)

Usage:
    processor = BatchProcessor(orchestrator)
    processor.add_files(['file1.pdf', 'file2.pdf'])
    processor.register_callback('on_file_complete', callback_fn)
    await processor.start_processing()
"""

import asyncio
import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import time
from datetime import datetime

# Import semantic ledger for saving processed results
from src.processing.ledger.semantic_ledger import SemanticLedger, LedgerRecord, generate_hash
from src.processing.ledger.csv_backend import CsvLedgerBackend
from src.processing.process_manager import ProcessManager
from src.processing.worker_engine import run_batch_process
from src.processing.models import JobConfig

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processing orchestration for PDF documents.

    Manages sequential file processing with async support for responsive GUI.
    Tracks progress and triggers UI callbacks at key events.

    Attributes:
        orchestrator: DocumentOrchestrator instance (from get_orchestrator())
        files_queue: List of file paths to process
        current_file_index: Index of currently processing file
        results: Dict mapping file_path -> result dict
        status: Processing status (idle/processing/paused/completed/error)
        callbacks: Dict mapping event name -> callback function
    """

    def __init__(self, orchestrator):
        """
        Initialize BatchProcessor.

        Args:
            orchestrator: DocumentOrchestrator instance from get_orchestrator()
        """
        self.orchestrator = orchestrator
        self.files_queue: List[str] = []
        self.current_file_index: int = 0
        self.results: Dict[str, Dict] = {}
        self.status: str = 'idle'  # idle, processing, paused, completed, error
        self.callbacks: Dict[str, Callable] = {}
        self.process_manager = ProcessManager()

        # Statistics
        self._start_time: Optional[float] = None
        self._current_file_start: Optional[float] = None

        # Initialize semantic ledger for saving results
        ledger_path = Path(os.getenv("SARA_LEDGER_PATH", "persistence/semantic_ledger.csv"))
        self.ledger: SemanticLedger = CsvLedgerBackend(file_path=str(ledger_path.resolve()))
        logger.info(f"Semantic ledger initialized at: {ledger_path.resolve()}")

        logger.info("BatchProcessor initialized")

    def add_files(self, file_paths: List[str]) -> None:
        """
        Add files to processing queue.

        Args:
            file_paths: List of absolute paths to PDF files
        """
        self.files_queue = file_paths.copy()
        self.current_file_index = 0
        self.results = {}
        logger.info(f"Added {len(file_paths)} files to batch queue")

    def clear_queue(self) -> None:
        """Clear the processing queue and reset state."""
        self.files_queue = []
        self.current_file_index = 0
        self.results = {}
        self.status = 'idle'
        logger.info("Batch queue cleared")

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register UI callback for batch events.

        Supported events:
            - 'on_file_start': Called when file processing starts
            - 'on_file_complete': Called when file completes successfully
            - 'on_file_error': Called when file processing fails
            - 'on_progress_update': Called periodically with progress stats
            - 'on_batch_complete': Called when entire batch finishes

        Args:
            event: Event name (see above)
            callback: Async callback function
        """
        self.callbacks[event] = callback
        logger.debug(f"Registered callback for event: {event}")

    async def start_processing(self) -> Dict[str, Any]:
        """
        Start batch processing workflow.

        Processes files sequentially from queue, triggering callbacks
        at each stage. Non-blocking via asyncio executor pattern.

        Returns:
            Dict with keys:
                - 'total_files': int
                - 'success_files': int
                - 'error_files': int
                - 'processing_time': float (seconds)
                - 'results': Dict[str, Dict]
        """
        if not self.files_queue:
            logger.warning("Cannot start processing: queue is empty")
            return self._get_summary()

        if self.status == 'processing':
            logger.warning("Batch processing already in progress")
            return self._get_summary()

        logger.info(f"🚀 Starting batch processing: {len(self.files_queue)} files")
        self.status = 'processing'
        self._start_time = time.time()

        # Process loop
        for idx, file_path in enumerate(self.files_queue):
            # Check for pause/stop
            if self.status != 'processing':
                logger.info(f"Batch processing stopped at file {idx + 1}/{len(self.files_queue)}")
                break

            self.current_file_index = idx

            try:
                # Trigger on_file_start callback
                await self._trigger_callback('on_file_start', file_path)

                # Process file (async via executor)
                self._current_file_start = time.time()
                result = await self._process_file(file_path)
                file_time = time.time() - self._current_file_start

                # Store result with processing time
                result['processing_time'] = file_time
                result['file_path'] = file_path
                self.results[file_path] = result

                # Check for errors in result
                if result.get('errors') or result.get('processing_result') is None:
                    # Orchestrator returned errors
                    error_msg = result.get('errors', ['Unknown error'])[0] if result.get('errors') else 'Processing failed'
                    logger.error(f"❌ File processing returned errors: {Path(file_path).name} - {error_msg}")

                    # Mark as error
                    result['status'] = 'error'
                    result['error_message'] = error_msg

                    # Trigger on_file_error callback
                    await self._trigger_callback('on_file_error', file_path, error_msg)
                else:
                    # Success
                    result['status'] = 'success'
                    logger.info(f"✅ Successfully processed: {Path(file_path).name} ({file_time:.2f}s)")

                    # Save to semantic ledger (in-memory)
                    self._save_to_ledger(file_path, result)

                    # Save full JSON output (Phase 6.6b Fix)
                    self._save_document_json(file_path, result)

                    # Trigger on_file_complete callback
                    await self._trigger_callback('on_file_complete', result)

                # Trigger on_progress_update callback
                progress = self.get_progress()
                await self._trigger_callback('on_progress_update', progress)

            except Exception as e:
                # Executor-level error
                logger.error(f"❌ Failed to process {Path(file_path).name}: {e}", exc_info=True)

                # Store error result
                self.results[file_path] = {
                    'file_path': file_path,
                    'status': 'error',
                    'error_message': str(e),
                    'processing_time': time.time() - self._current_file_start if self._current_file_start else 0.0
                }

                # Trigger on_file_error callback
                await self._trigger_callback('on_file_error', file_path, str(e))

                # Continue with next file (don't stop batch on single error)
                continue

        # Batch complete
        if self.status == 'processing':
            self.status = 'completed'
            logger.info(f"🏁 Batch processing complete: {self._get_success_count()}/{len(self.files_queue)} successful")

            # Commit all ledger changes to persistent storage (CSV)
            try:
                self.ledger.commit()
                logger.info(f"💾 Semantic ledger committed: {self._get_success_count()} records saved to CSV")
            except Exception as e:
                logger.error(f"Failed to commit semantic ledger: {e}", exc_info=True)

        # Trigger on_batch_complete callback
        summary = self._get_summary()
        await self._trigger_callback('on_batch_complete', summary)

        return summary

    def kill_current_job(self) -> None:
        """
        HARD KILL: Abort current job and kill the worker process.
        """
        logger.warning("⚡ HARD KILL requested! Ejecting engine...")
        
        # Kill the OS process
        if self.process_manager.is_alive():
            self.process_manager.kill()
            
        # Stop the batch loop
        self.status = 'aborted'
        
        # Get current file info
        idx = self.current_file_index
        if 0 <= idx < len(self.files_queue):
            file_path = self.files_queue[idx]
            
            # Record the kill (so we don't save garbage)
            self.results[file_path] = {
                'file_path': file_path,
                'status': 'error',
                'error_message': 'BATCH ABORTED BY USER (Hard Kill)',
                'processing_time': 0.0
            }
            
            # Save "Aborted" status to ledger? 
            # Ideally yes, but maybe just log it. 
            logger.info(f"Marked {Path(file_path).name} as aborted.")

    async def _process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process single file using Isolated Worker Process.
        
        Spawns a new worker, sends the job, and listens for results in a loop.
        Non-blocking (awaits sleep).
        
        Args:
            file_path: Absolute path to PDF file
            
        Returns:
            Dict containing processing results or error info.
        """
        logger.debug(f"Processing file (Isolated): {Path(file_path).name}")
        
        # Create Job Config
        job_id = generate_hash(file_path)
        config = JobConfig(
            job_id=job_id,
            file_path=str(file_path),
            tesseract_enabled=True # Could make this configurable via self.options
        )
        
        # Spawn Worker
        try:
            self.process_manager.start_worker(run_batch_process, config)
        except Exception as e:
            logger.error(f"Failed to spawn worker: {e}")
            return {
                'file_path': file_path,
                'status': 'error',
                'error_message': f"Process Spawn Failed: {e}",
                'processing_time': 0.0
            }
            
        # Listen Loop
        result = None
        while self.process_manager.is_alive():
            # Drain messages
            for msg in self.process_manager.get_messages():
                msg_type = msg.get('type')
                
                if msg_type == 'status':
                    # Optional: update UI detail text? 
                    pass
                    
                elif msg_type == 'log':
                    # Relay worker logs to main logger
                    log_level_str = msg.get('level', 'INFO')
                    log_msg = msg.get('msg', '')
                    
                    # Convert level name to int
                    level = getattr(logging, log_level_str.upper(), logging.INFO)
                    
                    # Log to main process logger (which writes to file and GUI)
                    # Prefix with [Worker] to distinguish
                    if not log_msg.startswith("202"): # If it doesn't have timestamp, or standard formatting
                         logger.log(level, f"[Worker] {log_msg}")
                    else:
                         # It's already formatted by worker
                         logger.log(level, log_msg)
                    
                elif msg_type == 'result':
                    # Success!
                    result = msg.get('data') # This is the Dict from orchestrator
                    # Append result metadata if missing
                    if result:
                        result['processing_time'] = msg.get('duration', 0.0)
                    break 
                    
                elif msg_type == 'error':
                    # Worker reported an error
                    error_msg = msg.get('error', 'Unknown Worker Error')
                    logger.error(f"Worker reported error: {error_msg}")
                    return {
                        'file_path': file_path,
                        'status': 'error',
                        'error_message': error_msg,
                        'processing_time': 0.0
                    }
            
            if result:
                break
            
            # Check if we were killed externally
            if self.status == 'aborted':
                logger.warning("Worker loop detected abort signal.")
                break
                
            # Non-blocking sleep
            await asyncio.sleep(0.1)
            
        # Loop exited. 
        
        # Case A: Status is aborted (Hard Kill)
        if self.status == 'aborted':
            return {
                'file_path': file_path,
                'status': 'error',
                'error_message': 'BATCH ABORTED BY USER',
                'processing_time': 0.0
            }
            
        # Case B: Result found
        if result:
            # Result received, but process might still be flushing logs/exiting.
            # Give it a moment to exit gracefully to avoid "Active Process" warning on next job.
            if self.process_manager.process:
                self.process_manager.process.join(timeout=2.0)
            return result
            
        # Case C: Process died without result (Crash)
        if not self.process_manager.is_alive() and not result:
            return {
                'file_path': file_path,
                'status': 'error',
                'error_message': 'Worker Process crashed unexpectedly (Out of Memory?)',
                'processing_time': 0.0
            }
            
        return result if result else {}

    def _save_to_ledger(self, file_path: str, result: Dict[str, Any]) -> None:
        """
        Save processing result to semantic ledger.

        Args:
            file_path: Path to the processed PDF
            result: Processing result dict from orchestrator

        Note:
            Only saves if processing was successful and metadata was extracted.
            Errors are logged but don't raise exceptions to avoid batch interruption.
        """
        try:
            processing_result = result.get('processing_result')
            if not processing_result or not hasattr(processing_result, 'metadata'):
                logger.warning(f"Skipping ledger save for {Path(file_path).name}: No metadata extracted")
                return

            metadata = processing_result.metadata
            if not metadata:
                logger.warning(f"Skipping ledger save for {Path(file_path).name}: Metadata is None")
                return

            # Generate document ID (hash of file content)
            doc_hash = generate_hash(file_path)

            # Create ledger record
            record = LedgerRecord(
                document_id=doc_hash,
                pdf_filename=Path(file_path).name,
                doi=metadata.doi,
                title=metadata.title,
                authors=metadata.authors if metadata.authors else [],
                journal=metadata.publisher,  # Map publisher to journal field
                year=metadata.year,
                validation_confidence=metadata.validation_confidence or 0.0,
                validation_source=metadata.validation_source or 'batch_processing',
                manual_entry=False,
                requires_review=False,  # Could be enhanced based on confidence thresholds
                last_updated=datetime.now().isoformat(),
                retrieval_provenance={
                    'processed_via': 'batch_commander',
                    'batch_processor_version': '1.0',
                    'processing_timestamp': datetime.now().isoformat()
                },
                source_hash=doc_hash
            )

            # Save to ledger (in-memory)
            self.ledger.update_record(record)
            logger.info(f"💾 Saved to ledger: {Path(file_path).name} (doc_id: {doc_hash[:8]}...)")

        except Exception as e:
            logger.error(f"Failed to save {Path(file_path).name} to ledger: {e}", exc_info=True)
            # Don't raise - allow batch to continue even if ledger save fails

    def _save_document_json(self, file_path: str, result: Dict[str, Any]) -> None:
        """
        Save the full document result to JSON.
        
        Persists the extraction result to run_outputs/documents/{doc_id}.json.
        """
        try:
            processing_result = result.get('processing_result')
            # Handle case where processing_result might be None or empty
            if not processing_result:
                return

            # Determine output path
            output_dir = Path("run_outputs/documents")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate doc_id hash again (for internal ID)
            doc_hash = generate_hash(file_path)
            
            # Use original filename for the JSON file (User Request)
            original_stem = Path(file_path).stem
            output_path = output_dir / f"{original_stem}.json"
            
            # Serialize
            # Handle Pydantic model vs Dict
            if hasattr(processing_result, 'model_dump'):
                data = processing_result.model_dump()
            elif hasattr(processing_result, 'dict'):
                data = processing_result.dict()
            else:
                data = processing_result 
            
            # Add wrapper info
            wrapper = {
                "document_id": doc_hash,
                "file_path": str(file_path),
                "processed_at": datetime.now().isoformat(),
                "processing_time": result.get('processing_time', 0.0),
                "result": data
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(wrapper, f, indent=2, ensure_ascii=False, default=str)
                
            logger.info(f"💾 Saved document JSON: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save document JSON for {Path(file_path).name}: {e}")

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current batch progress statistics.

        Returns:
            Dict with keys:
                - 'total_files': int - Total files in queue
                - 'processed_files': int - Files processed so far
                - 'success_files': int - Successfully processed files
                - 'error_files': int - Failed files
                - 'current_progress': float - Progress ratio (0.0-1.0) for current file bar
                - 'error_progress': float - Error ratio (0.0-1.0) for error bar
                - 'total_progress': float - Total completion ratio (0.0-1.0)
                - 'current_file': str - Path to currently processing file (or None)
                - 'current_file_name': str - Name of current file
                - 'status': str - Processing status
        """
        total_files = len(self.files_queue)
        processed_files = len(self.results)
        success_files = self._get_success_count()
        error_files = self._get_error_count()

        # Calculate progress ratios
        current_progress = (processed_files / total_files) if total_files > 0 else 0.0
        error_progress = (error_files / total_files) if total_files > 0 else 0.0
        total_progress = current_progress  # Same as current for now

        # Current file info
        current_file = self.files_queue[self.current_file_index] if self.current_file_index < total_files else None
        current_file_name = Path(current_file).name if current_file else ''

        return {
            'total_files': total_files,
            'processed_files': processed_files,
            'success_files': success_files,
            'error_files': error_files,
            'current_progress': current_progress,
            'error_progress': error_progress,
            'total_progress': total_progress,
            'current_file': current_file,
            'current_file_name': current_file_name,
            'status': self.status
        }

    def _get_success_count(self) -> int:
        """Get count of successfully processed files."""
        return sum(1 for r in self.results.values() if r.get('status') == 'success')

    def _get_error_count(self) -> int:
        """Get count of failed files."""
        return sum(1 for r in self.results.values() if r.get('status') == 'error')

    def _get_summary(self) -> Dict[str, Any]:
        """
        Get batch processing summary.

        Returns:
            Dict with keys:
                - 'total_files': int
                - 'success_files': int
                - 'error_files': int
                - 'processing_time': float (seconds)
                - 'results': Dict[str, Dict]
                - 'status': str
        """
        total_time = time.time() - self._start_time if self._start_time else 0.0

        return {
            'total_files': len(self.files_queue),
            'success_files': self._get_success_count(),
            'error_files': self._get_error_count(),
            'processing_time': total_time,
            'results': self.results,
            'status': self.status
        }

    async def _trigger_callback(self, event: str, *args) -> None:
        """
        Trigger registered callback for event.

        Args:
            event: Event name
            *args: Arguments to pass to callback
        """
        callback = self.callbacks.get(event)
        if callback:
            try:
                # All callbacks should be async
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    # Fallback for non-async callbacks (shouldn't happen)
                    callback(*args)
            except Exception as e:
                logger.error(f"Callback '{event}' failed: {e}", exc_info=True)
        else:
            logger.debug(f"No callback registered for event: {event}")
