import logging
import time
import sys
import traceback
import multiprocessing
from .models import JobConfig, ProcessingResult
from .document_orchestrator import DocumentOrchestrator

# Configure logging to capture everything inside the process
def setup_worker_logging(queue):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid double printing if any
    for h in root.handlers[:]:
        root.removeHandler(h)
        
    # Custom handler to push logs to queue
    class QueueHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                queue.put({
                    'type': 'log', 
                    'level': record.levelname, 
                    'msg': msg,
                    'timestamp': time.time()
                })
            except Exception:
                self.handleError(record)

    handler = QueueHandler()
    formatter = logging.Formatter('%(asctime)s - [%(name)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

def run_batch_process(queue: multiprocessing.Queue, config: JobConfig):
    """
    Entry point for the isolated worker process.
    
    Args:
        queue: Multiprocessing queue for sending status/results back to main process.
        config: JobConfig object containing file path and settings.
    """
    # 1. Setup Logging (Redirect to Queue)
    setup_worker_logging(queue)
    logger = logging.getLogger("WorkerEngine")
    
    logger.info(f"💥 Worker Process Started. PID: {multiprocessing.current_process().pid}")
    logger.info(f"Processing target: {config.file_path}")
    
    start_time = time.time()
    
    try:
        # 2. Initialize Orchestrator (Fresh instance in this memory space)
        # This bypasses any corrupted state in the main process
        logger.info("Initializing DocumentOrchestrator...")
        orchestrator = DocumentOrchestrator()
        
        # Apply config overrides
        orchestrator.tesseract_enabled = config.tesseract_enabled
        # orchestrator.vlm_enabled = config.vlm_enabled # If supported
        
        # 3. Notify Start
        queue.put({
            'type': 'status',
            'status': 'processing',
            'job_id': config.job_id,
            'file_path': config.file_path
        })
        
        # 4. Execute Processing
        # In Stage B, this calls the concurrent version (if enabled)
        logger.info(f"Starting document processing (Concurrency: {config.concurrency_enabled})...")
        result_dict = orchestrator.process_document(
            config.file_path, 
            pages_to_process=config.pages_to_process,
            concurrency_enabled=config.concurrency_enabled
        )
        
        processing_time = time.time() - start_time
        
        # 5. Handle Result
        # The orchestrator returns a Dict, but we might want to check for errors
        errors = result_dict.get('errors', [])
        
        if errors:
             logger.error(f"Processing completed with errors: {errors}")
             queue.put({
                 'type': 'error',
                 'job_id': config.job_id,
                 'error': "; ".join(errors)
             })
        else:
             logger.info(f"✅ Processing Successful ({processing_time:.2f}s)")
             
             # Extract the Pydantic model if it exists, otherwise use the dict
             # Note: JobConfig/ProcessManager expects picklable data. 
             # Pydantic models ARE picklable.
             
             # NEW: Auto-save JSON to output_dir
             if config.output_dir:
                 try:
                     import json
                     from pathlib import Path
                     
                     # Create documents directory
                     docs_dir = Path(config.output_dir) / "documents"
                     docs_dir.mkdir(parents=True, exist_ok=True)
                     
                     # Generate filename
                     stem = Path(config.file_path).stem
                     output_path = docs_dir / f"{stem}_output.json"
                     
                     # Get ProcessingResult object
                     proc_result = result_dict.get('processing_result')
                     
                     if proc_result:
                         # It's a Pydantic model, use model_dump_json (or json for V1)
                         # Safe generic export:
                         json_content = proc_result.model_dump_json(indent=2)
                         
                         with open(output_path, 'w', encoding='utf-8') as f:
                             f.write(json_content)
                             
                         logger.info(f"💾 Saved structured JSON to: {output_path}")
                     else:
                         logger.warning("No properties_result found to save.")
                         
                 except Exception as e:
                     logger.error(f"Failed to save JSON output: {e}")

             queue.put({
                 'type': 'result',
                 'job_id': config.job_id,
                 'data': result_dict, # This dict contains the ProcessingResult object
                 'duration': processing_time
             })
             
    except Exception as e:
        logger.critical(f"🔥 Critical Worker Failure: {e}", exc_info=True)
        queue.put({
            'type': 'error',
            'job_id': config.job_id,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        
    finally:
        logger.info("Worker Process shutting down.")
        # Ensure queue is flushed? multiprocessing queue handles this usually.
