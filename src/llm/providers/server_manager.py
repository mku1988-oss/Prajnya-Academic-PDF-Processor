"""
ServerManager: A/B Testing Framework for VLM Server Launch.

This module provides THREE backends for auto-launching the VLM server:
- Option A: LlamaSwapBackend (proven, uses llama-swap.exe as proxy router)
- Option B: DirectModeBackend (traditional, restart server per model change)
- Option C: RouterModeBackend (NEW, native llama-server router with /models/load API)

All backends use the batch file wrapper for CUDA PATH handling.
All backends use file-based logging or inherited console (no subprocess.PIPE deadlocks).

Usage:
    # Option A: llama-swap (default, proven)
    manager = ServerManager(backend="llama_swap")

    # Option B: Direct mode (restarts server per model change)
    manager = ServerManager(backend="direct_mode")

    # Option C: Router mode (native API for model switching - NO restart!)
    manager = ServerManager(backend="router_mode")

    # Start server
    if manager.ensure_started():
        print("Server ready!")

A/B Test:
    python -m src.llm.providers.server_manager ab_test
"""

import subprocess
import time
import logging
import os
import abc
import configparser
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


# ============================================================================
# Path Resolution Utility
# ============================================================================

def resolve_path(path_str: str, project_root: Optional[Path] = None) -> Path:
    """
    Resolve a path that can be either absolute or relative to project root.

    Args:
        path_str: Path string (absolute or relative)
        project_root: Project root directory (auto-detected if None)

    Returns:
        Resolved absolute Path

    Examples:
        >>> resolve_path("src/processing/strategies/config/vlm_profiles/model.ini")
        Path("D:/LLMs/Shastrasaara/src/v4/gemini_refactor/src/processing/strategies/config/vlm_profiles/model.ini")

        >>> resolve_path("D:/LLMs/Shastrasaara/llama_cpp_bin/model.ini")
        Path("D:/LLMs/Shastrasaara/llama_cpp_bin/model.ini")
    """
    path = Path(path_str)

    # If already absolute, return as-is
    if path.is_absolute():
        return path

    # Resolve relative to project root
    if project_root is None:
        # Auto-detect project root (where this file is located: src/llm/providers/)
        # Project root is 3 levels up from this file
        project_root = Path(__file__).parent.parent.parent.parent

    return (project_root / path).resolve()


# ============================================================================
# Configuration Constants
# ============================================================================

DEFAULT_PATHS = {
    "llama_swap_exe": "D:/LLMs/Shastrasaara/llama_cpp_bin/llama-swap.exe",
    "swap_config_yaml": "D:/LLMs/Shastrasaara/llama_cpp_bin/run_swap_Config.yaml",
    "batch_file": "D:/LLMs/Shastrasaara/llama_cpp_bin/run-llama-server.bat",
    "model_ini": "D:/LLMs/Shastrasaara/llama_cpp_bin/model.ini",
    "log_dir": "run_outputs/logs",
}

DEFAULT_MODEL_PROFILE = "olmocr-2-7b"
DEFAULT_PORT = 8080
DEFAULT_HOST = "127.0.0.1"
DEFAULT_STARTUP_TIMEOUT = 60


def resolve_path(path_str: str, project_root: Optional[Path] = None) -> Path:
    """
    Resolve a path that can be either absolute or relative to project root.

    - Absolute paths: Returned as-is
    - Relative paths: Resolved from project root (auto-detected)
    """
    if not path_str:
        return Path(".")
        
    path = Path(path_str)
    if path.is_absolute():
        return path
        
    if project_root is None:
        # Auto-detect root: 4 levels up from this file 
        # (src/llm/providers/server_manager.py -> src/llm/providers -> src/llm -> src -> root)
        project_root = Path(__file__).parent.parent.parent.parent
        
    return (project_root / path).resolve()


class ServerBackend(Enum):
    """Available server backends."""
    LLAMA_SWAP = "llama_swap"
    DIRECT_MODE = "direct_mode"
    ROUTER_MODE = "router_mode"


# ============================================================================
# Abstract Base Class
# ============================================================================

class BaseServerBackend(abc.ABC):
    """Abstract base class for server backends."""

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        host: str = DEFAULT_HOST,
        log_dir: Optional[Path] = None,
        startup_timeout: int = DEFAULT_STARTUP_TIMEOUT
    ):
        self.port = port
        self.host = host
        self.log_dir = Path(log_dir) if log_dir else Path(DEFAULT_PATHS["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout = startup_timeout

        self._process: Optional[subprocess.Popen] = None
        self._stdout_handle = None
        self._stderr_handle = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Return human-readable backend name."""
        pass

    @abc.abstractmethod
    def _get_launch_command(self) -> list:
        """Return the command to launch the server."""
        pass

    @abc.abstractmethod
    def _get_log_prefix(self) -> str:
        """Return prefix for log files."""
        pass

    @abc.abstractmethod
    def _verify_prerequisites(self):
        """Verify required files exist. Raise FileNotFoundError if not."""
        pass

    def is_running(self) -> bool:
        """Check if server responds to /health."""
        if requests is None:
            logger.warning("requests module not available")
            return False

        try:
            response = requests.get(self.health_url, timeout=2)
            if response.status_code == 200:
                return True

            # Fallback: try /props endpoint
            response = requests.get(f"{self.base_url}/props", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def ensure_started(self) -> bool:
        """
        Ensure server is running. Returns True if ready.
        """
        # Check if already running
        if self.is_running():
            logger.info("[%s] Server already running at %s", self.backend_name, self.base_url)
            return True

        logger.info("[%s] Server not running, launching...", self.backend_name)

        # Verify prerequisites
        try:
            self._verify_prerequisites()
        except FileNotFoundError as e:
            logger.error("[%s] Prerequisite check failed: %s", self.backend_name, e)
            return False

        # Launch server
        if not self._launch():
            return False

        # Wait for health
        return self._wait_for_health()

    def _launch(self) -> bool:
        """Launch server process."""
        prefix = self._get_log_prefix()
        stdout_path = self.log_dir / f"{prefix}_stdout.log"
        stderr_path = self.log_dir / f"{prefix}_stderr.log"

        try:
            # Open file handles for output redirection (avoids pipe deadlocks)
            self._stdout_handle = open(stdout_path, 'w', encoding='utf-8')
            self._stderr_handle = open(stderr_path, 'w', encoding='utf-8')

            cmd = self._get_launch_command()
            env = self._get_environment()

            logger.info("[%s] Command: %s", self.backend_name, ' '.join(cmd))

            # CREATE_NEW_CONSOLE prevents GUI freeze on Windows
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NEW_CONSOLE

            self._process = subprocess.Popen(
                cmd,
                stdout=self._stdout_handle,
                stderr=self._stderr_handle,
                env=env,
                creationflags=creation_flags
            )

            logger.info("[%s] Process started (PID: %d)", self.backend_name, self._process.pid)
            return True

        except Exception as e:
            logger.error("[%s] Failed to launch: %s", self.backend_name, e)
            self._cleanup_handles()
            return False

    def _get_environment(self) -> dict:
        """Get environment for subprocess. Override in subclass if needed."""
        return os.environ.copy()

    def _wait_for_health(self) -> bool:
        """Poll /health endpoint until success or timeout."""
        start_time = time.time()
        check_interval = 1.0

        while time.time() - start_time < self.startup_timeout:
            # Check if process crashed
            if self._process and self._process.poll() is not None:
                exit_code = self._process.returncode
                logger.error("[%s] Process crashed (exit code: %d)", self.backend_name, exit_code)
                self._log_error_file()
                return False

            # Check health endpoint
            if self.is_running():
                elapsed = time.time() - start_time
                logger.info("[%s] Server ready after %.1fs", self.backend_name, elapsed)
                return True

            time.sleep(check_interval)

        logger.error("[%s] Timeout after %ds", self.backend_name, self.startup_timeout)
        self._log_error_file()
        return False

    def _log_error_file(self):
        """Log contents of stderr file for diagnostics."""
        prefix = self._get_log_prefix()
        stderr_path = self.log_dir / f"{prefix}_stderr.log"
        if stderr_path.exists():
            try:
                content = stderr_path.read_text(encoding='utf-8', errors='replace')
                if content.strip():
                    logger.error("[%s] stderr:\n%s", self.backend_name, content[-2000:])
            except Exception as e:
                logger.warning("[%s] Could not read stderr: %s", self.backend_name, e)

    def stop(self, timeout: int = 10):
        """Stop the server gracefully."""
        if self._process is None:
            return

        logger.info("[%s] Stopping server...", self.backend_name)

        try:
            self._process.terminate()
            self._process.wait(timeout=timeout)
            logger.info("[%s] Server stopped", self.backend_name)
        except subprocess.TimeoutExpired:
            logger.warning("[%s] Force killing...", self.backend_name)
            self._process.kill()
            self._process.wait()
        finally:
            self._cleanup_handles()
            self._process = None

    def _cleanup_handles(self):
        """Close file handles."""
        for handle in [self._stdout_handle, self._stderr_handle]:
            if handle:
                try:
                    handle.close()
                except:
                    pass
        self._stdout_handle = None
        self._stderr_handle = None

    def __del__(self):
        self._cleanup_handles()


# ============================================================================
# Option A: llama-swap Backend (PROVEN)
# ============================================================================

class LlamaSwapBackend(BaseServerBackend):
    """
    Option A: llama-swap backend.

    Uses llama-swap.exe as a proxy/router layer.
    Configuration via run_swap_Config.yaml (YAML format).

    This is the PROVEN approach that has worked reliably.
    """

    def __init__(
        self,
        llama_swap_exe: str = DEFAULT_PATHS["llama_swap_exe"],
        config_path: str = DEFAULT_PATHS["swap_config_yaml"],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.llama_swap_exe = Path(llama_swap_exe)
        self.config_path = Path(config_path)

    @property
    def backend_name(self) -> str:
        return "llama-swap"

    def _get_log_prefix(self) -> str:
        return "llama_swap"

    def _verify_prerequisites(self):
        if not self.llama_swap_exe.exists():
            raise FileNotFoundError(f"llama-swap.exe not found: {self.llama_swap_exe}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config YAML not found: {self.config_path}")

    def _get_launch_command(self) -> list:
        return [
            str(self.llama_swap_exe),
            "-config", str(self.config_path)
        ]


# ============================================================================
# Option B: Direct Mode Backend (EXPERIMENTAL)
# ============================================================================

class DirectModeBackend(BaseServerBackend):
    """
    Option B: Direct mode backend.

    Launches llama-server via batch file with explicit arguments parsed from model.ini.
    Uses environment variable overrides for enforcement.

    This approach was problematic before due to:
    1. Missing os import (fixed)
    2. Router mode ignoring --parallel 1 (fixed via LLAMA_ARG_N_PARALLEL)
    3. CUDA PATH issues (fixed via batch wrapper)
    """

    def __init__(
        self,
        batch_file: str = DEFAULT_PATHS["batch_file"],
        model_ini: str = DEFAULT_PATHS["model_ini"],
        model_profile: str = DEFAULT_MODEL_PROFILE,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_file = Path(batch_file)
        self.model_ini = Path(model_ini)
        self.model_profile = model_profile
        self._model_config: Optional[Dict[str, str]] = None

    @property
    def backend_name(self) -> str:
        return "direct-mode"

    def _get_log_prefix(self) -> str:
        return "direct_mode"

    def _verify_prerequisites(self):
        if not self.batch_file.exists():
            raise FileNotFoundError(f"Batch file not found: {self.batch_file}")
        if not self.model_ini.exists():
            raise FileNotFoundError(f"model.ini not found: {self.model_ini}")

        # Parse and cache model config
        self._parse_model_config()

    def _parse_model_config(self):
        """Parse model.ini and extract profile settings."""
        config = configparser.ConfigParser()
        config.read(self.model_ini)

        if self.model_profile not in config:
            raise ValueError(f"Profile '{self.model_profile}' not found in {self.model_ini}")

        self._model_config = dict(config[self.model_profile])
        logger.info("[%s] Loaded profile: %s", self.backend_name, self.model_profile)

    def _get_launch_command(self) -> list:
        """Build command with explicit arguments from model.ini."""
        cfg = self._model_config

        cmd = [str(self.batch_file)]

        # Connection params
        cmd.extend(["--port", str(self.port)])
        cmd.extend(["--host", str(self.host)])

        # Model paths
        if "model" in cfg:
            cmd.extend(["--model", cfg["model"]])
        if "mmproj" in cfg:
            cmd.extend(["--mmproj", cfg["mmproj"]])

        # Force single-slot (CRITICAL)
        cmd.extend(["--parallel", "1"])

        # Disable unified KV mode to prevent auto n_parallel=4
        cmd.append("-kvu")

        # Performance params
        for param in ["ctx-size", "ubatch-size", "batch-size", "n-gpu-layers", "threads"]:
            if param in cfg:
                cmd.extend([f"--{param}", cfg[param]])

        # Flash attention
        if cfg.get("flash-attn", "").lower() == "true":
            cmd.extend(["--flash-attn", "on"])

        # KV cache quantization
        for cache_param in ["cache-type-k", "cache-type-v"]:
            if cache_param in cfg:
                cmd.extend([f"--{cache_param}", cfg[cache_param]])

        # Sampling params
        if "temp" in cfg:
            cmd.extend(["--temp", cfg["temp"]])

        return cmd

    def _get_environment(self) -> dict:
        """
        Get environment with LLAMA_ARG_* overrides.

        These environment variables are STRONGER than CLI flags.
        This addresses the n_parallel=4 bug from post-mortem.
        """
        env = os.environ.copy()
        cfg = self._model_config

        # Force single-slot processing (prevents VRAM exhaustion)
        env["LLAMA_ARG_N_PARALLEL"] = "1"

        # Performance overrides
        env["LLAMA_ARG_CTX_SIZE"] = cfg.get("ctx-size", "16384")
        env["LLAMA_ARG_UBATCH_SIZE"] = cfg.get("ubatch-size", "2048")
        env["LLAMA_ARG_BATCH_SIZE"] = cfg.get("batch-size", "2048")

        logger.info("[%s] Env overrides: N_PARALLEL=1, CTX=%s, UBATCH=%s",
                    self.backend_name, env["LLAMA_ARG_CTX_SIZE"], env["LLAMA_ARG_UBATCH_SIZE"])

        return env


# ============================================================================
# Option C: Router Mode Backend (NEW - Native API for Model Switching)
# ============================================================================

class RouterModeBackend(BaseServerBackend):
    """
    Option C: Router mode backend using native llama-server model management API.

    Launches llama-server in router mode (without specifying a model) and uses
    the /models/load and /models/unload endpoints for dynamic model switching.

    Key advantages:
    - NO server restart needed to switch models
    - Uses presets INI file (model.ini) for per-model configuration
    - LRU eviction when --models-max limit reached

    New llama-server flags (Dec 2025):
    - --models-preset <file>  : Use presets INI file
    - --models-dir <path>     : Directory with GGUF files
    - --models-max N          : Max concurrent models (default: 4)
    """

    def __init__(
        self,
        batch_file: str = DEFAULT_PATHS["batch_file"],
        model_ini: str = DEFAULT_PATHS["model_ini"],
        models_max: int = 1,  # Only one model loaded at a time (VRAM constraint)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_file = Path(batch_file)
        self.model_ini = Path(model_ini)
        self.models_max = models_max
        self._current_model: Optional[str] = None

    @property
    def backend_name(self) -> str:
        return "router-mode"

    def _get_log_prefix(self) -> str:
        return "router_mode"

    def _verify_prerequisites(self):
        if not self.batch_file.exists():
            raise FileNotFoundError(f"Batch file not found: {self.batch_file}")
        if not self.model_ini.exists():
            raise FileNotFoundError(f"model.ini not found: {self.model_ini}")

    def _get_launch_command(self) -> list:
        """Build command for router mode (no model specified, uses presets)."""
        cmd = [str(self.batch_file)]

        # Connection params
        cmd.extend(["--port", str(self.port)])
        cmd.extend(["--host", str(self.host)])

        # Router mode flags
        cmd.extend(["--models-preset", str(self.model_ini)])
        cmd.extend(["--models-max", str(self.models_max)])

        # Performance defaults (inherited by all models)
        cmd.extend(["--parallel", "1"])
        cmd.append("-kvu")  # Disable unified KV mode

        return cmd

    def _get_environment(self) -> dict:
        """Get environment with LLAMA_ARG_* overrides."""
        env = os.environ.copy()
        env["LLAMA_ARG_N_PARALLEL"] = "1"
        return env

    # =========================================================================
    # Model Management API
    # =========================================================================

    def load_model(self, model_name: str, timeout: int = 120) -> bool:
        """
        Load a model using the /models/load API.

        Args:
            model_name: Model name as defined in model.ini (e.g., "olmocr-2-7b")
            timeout: Max wait time for model loading

        Returns:
            True if model loaded successfully
        """
        # Check if model already loaded (skip redundant load)
        if self._current_model == model_name:
            logger.info("[%s] Model '%s' already loaded, skipping", self.backend_name, model_name)
            return True

        if requests is None:
            logger.error("[%s] requests module not available", self.backend_name)
            return False

        url = f"{self.base_url}/models/load"
        payload = {"model": model_name}

        logger.info("[%s] Loading model: %s", self.backend_name, model_name)
        start = time.time()

        try:
            response = requests.post(url, json=payload, timeout=timeout)

            if response.status_code == 200:
                elapsed = time.time() - start
                logger.info("[%s] Model '%s' loaded in %.1fs", self.backend_name, model_name, elapsed)
                self._current_model = model_name
                return True
            else:
                logger.error("[%s] Failed to load model: %s - %s",
                           self.backend_name, response.status_code, response.text)
                return False

        except requests.exceptions.Timeout:
            logger.error("[%s] Timeout loading model '%s'", self.backend_name, model_name)
            return False
        except Exception as e:
            logger.error("[%s] Error loading model: %s", self.backend_name, e)
            return False

    def unload_model(self, model_name: str, timeout: int = 30) -> bool:
        """
        Unload a model using the /models/unload API.

        Args:
            model_name: Model name to unload
            timeout: Max wait time for unloading

        Returns:
            True if model unloaded successfully
        """
        if requests is None:
            logger.error("[%s] requests module not available", self.backend_name)
            return False

        url = f"{self.base_url}/models/unload"
        payload = {"model": model_name}

        logger.info("[%s] Unloading model: %s", self.backend_name, model_name)

        try:
            response = requests.post(url, json=payload, timeout=timeout)

            if response.status_code == 200:
                logger.info("[%s] Model '%s' unloaded", self.backend_name, model_name)
                if self._current_model == model_name:
                    self._current_model = None
                return True
            else:
                logger.warning("[%s] Unload response: %s - %s",
                             self.backend_name, response.status_code, response.text)
                return False

        except Exception as e:
            logger.error("[%s] Error unloading model: %s", self.backend_name, e)
            return False

    def switch_model(self, new_model: str, timeout: int = 120) -> bool:
        """
        Switch from current model to a new model.

        Unloads current model first (if any), then loads the new one.

        Args:
            new_model: Model name to switch to
            timeout: Max wait time for loading

        Returns:
            True if switch successful
        """
        # Unload current model if any
        if self._current_model and self._current_model != new_model:
            self.unload_model(self._current_model)

        # Load new model
        return self.load_model(new_model, timeout=timeout)

    def list_models(self) -> Optional[dict]:
        """
        List available models and their status.

        Returns:
            Dict with model information, or None on error
        """
        if requests is None:
            return None

        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning("[%s] Error listing models: %s", self.backend_name, e)

        return None

    @property
    def current_model(self) -> Optional[str]:
        """Return currently loaded model name."""
        return self._current_model


# ============================================================================
# Unified ServerManager
# ============================================================================

class ServerManager:
    """
    Unified server manager with configurable backend.

    Usage:
        # Default (llama-swap, proven)
        manager = ServerManager()

        # Experimental (direct mode)
        manager = ServerManager(backend="direct_mode")

        # From YAML config (recommended)
        manager = ServerManager.from_config(config_dict)

        # Start
        if manager.ensure_started():
            print("Ready!")
    """

    def __init__(
        self,
        backend: str = "llama_swap",
        port: int = DEFAULT_PORT,
        host: str = DEFAULT_HOST,
        log_dir: Optional[Path] = None,
        startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
        **backend_kwargs
    ):
        self.port = port
        self.host = host
        self.log_dir = Path(log_dir) if log_dir else Path(DEFAULT_PATHS["log_dir"])
        self.startup_timeout = startup_timeout

        common_kwargs = {
            "port": port,
            "host": host,
            "log_dir": self.log_dir,
            "startup_timeout": startup_timeout
        }
        common_kwargs.update(backend_kwargs)

        if backend in (ServerBackend.LLAMA_SWAP.value, "llama_swap"):
            self._backend = LlamaSwapBackend(**common_kwargs)
        elif backend in (ServerBackend.DIRECT_MODE.value, "direct_mode"):
            self._backend = DirectModeBackend(**common_kwargs)
        elif backend in (ServerBackend.ROUTER_MODE.value, "router_mode"):
            self._backend = RouterModeBackend(**common_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'llama_swap', 'direct_mode', or 'router_mode'")

        logger.info("ServerManager initialized: %s", self._backend.backend_name)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ServerManager':
        """
        Create ServerManager from configuration dictionary.

        Args:
            config: Configuration dict from YAML vlm_provider.local_server section

        Returns:
            Configured ServerManager instance

        Example:
            >>> config = {
            ...     "enabled": True,
            ...     "backend": "router_mode",
            ...     "host": "localhost",
            ...     "port": 8080,
            ...     "server_executable": "src/bin/llama-server",  # Relative path
            ...     "model_profiles": "src/processing/strategies/config/vlm_profiles/model.ini",
            ...     "models_max": 1
            ... }
            >>> manager = ServerManager.from_config(config)
        """
        backend = config.get("backend", "router_mode")
        port = config.get("port", DEFAULT_PORT)
        host = config.get("host", DEFAULT_HOST)
        timeout = config.get("timeout", DEFAULT_STARTUP_TIMEOUT)
        log_dir = config.get("log_directory")

        # Resolve log directory
        if log_dir:
            log_dir = resolve_path(log_dir)

        # Backend-specific kwargs
        backend_kwargs = {}

        if backend == "router_mode":
            # Server executable (cross-platform)
            if "server_executable" in config:
                backend_kwargs["batch_file"] = str(resolve_path(config["server_executable"]))

            # Model profiles (Prajnya-specific)
            if "model_profiles" in config:
                backend_kwargs["model_ini"] = str(resolve_path(config["model_profiles"]))

            # Models max
            if "models_max" in config:
                backend_kwargs["models_max"] = config["models_max"]

        elif backend == "llama_swap":
            # llama-swap specific paths
            if "llama_swap_executable" in config:
                backend_kwargs["llama_swap_exe"] = str(resolve_path(config["llama_swap_executable"]))

            if "swap_config" in config:
                backend_kwargs["config_path"] = str(resolve_path(config["swap_config"]))

        elif backend == "direct_mode":
            # Direct mode paths
            if "server_executable" in config:
                backend_kwargs["batch_file"] = str(resolve_path(config["server_executable"]))

        logger.info("Creating ServerManager from config: backend=%s", backend)

        return cls(
            backend=backend,
            port=port,
            host=host,
            log_dir=log_dir,
            startup_timeout=timeout,
            **backend_kwargs
        )

    @property
    def backend_name(self) -> str:
        return self._backend.backend_name

    @property
    def base_url(self) -> str:
        return self._backend.base_url

    def is_running(self) -> bool:
        return self._backend.is_running()

    def ensure_started(self) -> bool:
        return self._backend.ensure_started()

    def stop(self, timeout: int = 10):
        self._backend.stop(timeout)

    # =========================================================================
    # Router-Mode Specific Methods (delegate to backend if available)
    # =========================================================================

    def load_model(self, model_name: str, timeout: int = 120) -> bool:
        """Load a model (router-mode only)."""
        if hasattr(self._backend, 'load_model'):
            return self._backend.load_model(model_name, timeout)
        logger.warning("load_model not supported by %s backend", self.backend_name)
        return False

    def unload_model(self, model_name: str, timeout: int = 30) -> bool:
        """Unload a model (router-mode only)."""
        if hasattr(self._backend, 'unload_model'):
            return self._backend.unload_model(model_name, timeout)
        logger.warning("unload_model not supported by %s backend", self.backend_name)
        return False

    def switch_model(self, new_model: str, timeout: int = 120) -> bool:
        """Switch models without server restart (router-mode only)."""
        if hasattr(self._backend, 'switch_model'):
            return self._backend.switch_model(new_model, timeout)
        logger.warning("switch_model not supported by %s backend", self.backend_name)
        return False

    def list_models(self) -> Optional[dict]:
        """List available models (router-mode only)."""
        if hasattr(self._backend, 'list_models'):
            return self._backend.list_models()
        return None

    @property
    def current_model(self) -> Optional[str]:
        """Get currently loaded model (router-mode only)."""
        if hasattr(self._backend, 'current_model'):
            return self._backend.current_model
        return None


# ============================================================================
# Legacy Compatibility (for existing code that uses LocalServerManager)
# ============================================================================

class LocalServerManager:
    """
    Legacy wrapper for backward compatibility.

    Delegates to ServerManager with direct_mode backend.
    """

    def __init__(self, settings):
        """
        Initialize from strategy settings.

        Args:
            settings: Config object with vlm_provider.local_server section
        """
        # Extract config
        try:
            local_cfg = settings.vlm_provider.local_server
        except AttributeError:
            local_cfg = settings['vlm_provider']['local_server']

        def get_val(key, default=None):
            if isinstance(local_cfg, dict):
                return local_cfg.get(key, default)
            return getattr(local_cfg, key, default)

        self.enabled = get_val('enabled', False)

        if self.enabled:
            self._manager = ServerManager(
                backend="direct_mode",
                port=get_val('port', DEFAULT_PORT),
                host=get_val('host', DEFAULT_HOST),
            )
        else:
            self._manager = None

    def ensure_started(self) -> bool:
        if not self.enabled:
            logger.info("Local server auto-launch is disabled.")
            return True

        return self._manager.ensure_started() if self._manager else False


# ============================================================================
# A/B Testing Utilities
# ============================================================================

def test_backend(backend: str, timeout: int = 60, **kwargs) -> Dict[str, Any]:
    """
    Test a single backend.

    Args:
        backend: "llama_swap" or "direct_mode"
        timeout: Startup timeout in seconds
        **kwargs: Additional arguments for ServerManager (e.g. model_profile)

    Returns:
        Dict with test results
    """
    results = {
        "backend": backend,
        "success": False,
        "startup_time": None,
        "error": None
    }

    try:
        start = time.time()
        manager = ServerManager(backend=backend, startup_timeout=timeout, **kwargs)

        if manager.ensure_started():
            results["startup_time"] = round(time.time() - start, 2)
            results["success"] = True
            logger.info("[TEST] %s: SUCCESS (%.1fs)", backend, results["startup_time"])
        else:
            results["error"] = "Failed to start"
            logger.error("[TEST] %s: FAILED", backend)

    except Exception as e:
        results["error"] = str(e)
        logger.error("[TEST] %s: EXCEPTION - %s", backend, e)

    return results


def run_ab_test(wait_between: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Run A/B test comparing both backends.

    Stops one backend before starting the other.

    Args:
        wait_between: Seconds to wait between tests

    Returns:
        Dict with results for each backend
    """
    results = {}

    print("\n" + "=" * 60)
    print("A/B TEST: VLM Server Launch Backends")
    print("=" * 60)

    # Test A: llama-swap
    print("\n[A] Testing llama-swap backend...")
    results["llama_swap"] = test_backend("llama_swap")

    # Stop before next test
    try:
        manager = ServerManager(backend="llama_swap")
        manager.stop()
    except:
        pass

    print(f"\nWaiting {wait_between}s for port to free...")
    time.sleep(wait_between)

    # Test B: direct-mode
    print("\n[B] Testing direct-mode backend...")
    results["direct_mode"] = test_backend("direct_mode")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("-" * 60)
    for name, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        time_str = f"{result['startup_time']}s" if result["startup_time"] else "N/A"
        error_str = f" ({result['error']})" if result.get("error") else ""
        print(f"  {name:15s}: {status:4s}  |  Startup: {time_str:8s}{error_str}")
    print("=" * 60)

    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "ab_test":
            run_ab_test()
        elif arg in ("llama_swap", "direct_mode", "router_mode"):
            print(f"Testing backend: {arg}")
            kwargs = {}
            if arg == "direct_mode" and len(sys.argv) > 2:
                kwargs["model_profile"] = sys.argv[2]
                print(f"Using profile: {kwargs['model_profile']}")

            result = test_backend(arg, **kwargs)
            print(f"Result: {result}")
        elif arg == "check":
            # Quick health check
            manager = ServerManager(backend="llama_swap")
            status = "RUNNING" if manager.is_running() else "NOT RUNNING"
            print(f"Server status: {status}")
        else:
            print(f"Unknown argument: {arg}")
            print("Usage:")
            print("  python -m src.llm.providers.server_manager llama_swap    # Test llama-swap")
            print("  python -m src.llm.providers.server_manager direct_mode [profile]  # Test direct mode")
            print("  python -m src.llm.providers.server_manager router_mode   # Test router mode (native API)")
            print("  python -m src.llm.providers.server_manager ab_test       # Run A/B comparison")
            print("  python -m src.llm.providers.server_manager check         # Quick health check")
    else:
        print("Usage:")
        print("  python -m src.llm.providers.server_manager llama_swap    # Test llama-swap")
        print("  python -m src.llm.providers.server_manager direct_mode [profile]  # Test direct mode")
        print("  python -m src.llm.providers.server_manager router_mode   # Test router mode (native API)")
        print("  python -m src.llm.providers.server_manager ab_test       # Run A/B comparison")
        print("  python -m src.llm.providers.server_manager check         # Quick health check")
