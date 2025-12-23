# .\prajnya_studio\gui\app_state.py
import threading
from collections import deque
from typing import Dict, Any, Optional

class AppState:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        if AppState._instance is not None:
            raise Exception("AppState is a singleton!")
        
        self.settings: Dict[str, Any] = {}
        self.log_deque: deque = deque(maxlen=2000)
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.vlm_lock_state: bool = False
        self.temp_run_folder: Optional[str] = None
        self.server_manager: Optional[Any] = None  # To be initialized in studio_app.py
        self.vlm_online: bool = False
        self.vlm_latency: int = 0

    @staticmethod
    def get_instance() -> 'AppState':
        if AppState._instance is None:
            with AppState._lock:
                if AppState._instance is None:
                    AppState._instance = AppState()
        return AppState._instance

    def register_job(self, job_id: str, meta: dict):
        with self._lock:
            self.active_jobs[job_id] = meta

    def update_job(self, job_id: str, updates: dict):
        with self._lock:
            if job_id in self.active_jobs:
                self.active_jobs[job_id].update(updates)

    def get_vlm_latency(self) -> int:
        # Placeholder for latency metric
        return 45
    
    def get_pending_review_count(self) -> int:
        # Placeholder until Ledger is connected
        return 0