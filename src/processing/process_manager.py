import multiprocessing
import queue
import logging
import time
import os
import signal
from typing import Optional, Any, Callable
from dataclasses import dataclass
from .models import JobConfig

logger = logging.getLogger(__name__)

class ProcessManager:
    """
    Manages the lifecycle of the isolated worker process ("The Ejectable Engine").
    
    Provides:
    - Process spawning (start_worker)
    - Safe termination (kill)
    - Communication channel (Queue)
    
    This class ensures the main UI process is never blocked by heavy processing
    and can always "pull the plug" instantly.
    """
    
    def __init__(self):
        self.process: Optional[multiprocessing.Process] = None
        self.msg_queue: Optional[multiprocessing.Queue] = None
        self.active_job_id: Optional[str] = None
        
    def start_worker(self, target_func: Callable, config: JobConfig):
        """
        Spawn a new worker process.
        
        Args:
            target_func: The function to run in the new process. 
                         Must accept (queue, config) as arguments.
            config: JobConfig object with all necessary data.
        """
        if self.is_alive():
            logger.warning("Attempted to start worker while one is already active provided. Killing old one.")
            self.kill()
            
        self.msg_queue = multiprocessing.Queue()
        self.active_job_id = config.job_id
        
        self.process = multiprocessing.Process(
            target=target_func,
            args=(self.msg_queue, config),
            name=f"ShastraWorker-{config.job_id[:8]}"
        )
        
        self.process.start()
        logger.info(f"🚀 Worker process started (PID: {self.process.pid}) for Job {config.job_id}")
        
    def kill(self):
        """
        Hard kill the worker process. ("The Ejectable Engine")
        """
        if self.process and self.process.is_alive():
            pid = self.process.pid
            logger.warning(f"⚡ FORCE KILLING Worker Process (PID: {pid})...")
            
            # Try Python native terminate first
            self.process.terminate()
            
            # Give it a tiny grace period to die
            self.process.join(timeout=0.5)
            
            if self.process.is_alive():
                 # If still alive (blocked in C++), use OS primitives
                logger.error(f"Process {pid} refused to die. Sending SIGKILL/TerminateProcess.")
                try:
                    # Windows/Linux compatible kill
                    if hasattr(os, 'kill'):
                         # Unix
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except AttributeError: 
                            # Windows: terminate() usually wraps TerminateProcess, but make sure
                            self.process.kill() 
                    else:
                        self.process.kill() # Python 3.7+
                        
                except Exception as e:
                    logger.error(f"Failed to force kill process: {e}")
            
            self.process.join(timeout=0.1)
            logger.info(f"💀 Worker Process {pid} is dead.")
            
        self._cleanup()
            
    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()
        
    def _cleanup(self):
        self.process = None
        # buffer_queue? typically we just abandon the queue or drain it.
        # Closing the queue might be good if we own it.
        if self.msg_queue:
            # self.msg_queue.close() # Can cause issues if process is writing
            self.msg_queue = None
        self.active_job_id = None

    def get_messages(self, block=False, timeout=None):
        """
        Drain messages from the queue.
        Generator that yields messages until queue is empty (or timeout).
        """
        if not self.msg_queue:
            return
            
        while True:
            try:
                if block:
                    yield self.msg_queue.get(block=True, timeout=timeout)
                    # If blocking, we typically yield one or loop? 
                    # Usually "listen" implies a loop. 
                    # For non-blocking polling (UI friendly), we do this:
                    block = False # Switch to non-blocking after first if needed, or just return
                else:
                    yield self.msg_queue.get_nowait()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error reading from worker queue: {e}")
                break
