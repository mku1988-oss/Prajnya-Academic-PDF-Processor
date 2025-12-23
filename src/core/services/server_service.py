"""
Centralized VLM Server management service.
Single source of truth for server initialization.
"""
from pathlib import Path
from typing import Optional, Tuple
import yaml
import logging
from src.llm.providers.server_manager import ServerManager

logger = logging.getLogger(__name__)

class ServerService:
    """
    Centralized server configuration and lifecycle management.
    Used by both CLI and GUI.
    """

    @staticmethod
    def load_server_config(
        config_path: Path = Path("src/processing/strategies/config/research_article_strategy.yaml")
    ) -> Tuple[Optional[dict], str]:
        """
        Load server configuration from strategy YAML.

        Returns:
            (server_config, model_to_load)
        """
        server_config = None
        model_to_load = "olmocr-2-7b"

        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                    server_config = cfg.get('vlm_provider', {}).get('local_server', {})
                    model_to_load = cfg.get('profile_registry', {}).get('default', "olmocr-2-7b")
        except Exception as e:
            logger.warning(f"Could not read strategy config: {e}")

        return server_config, model_to_load

    @staticmethod
    def create_server_manager(
        server_config: Optional[dict] = None
    ) -> ServerManager:
        """
        Create ServerManager instance from config.

        Args:
            server_config: Optional config dict. If None, will load from default location.

        Returns:
            Configured ServerManager instance
        """
        if server_config is None:
            server_config, _ = ServerService.load_server_config()

        if server_config and server_config.get('enabled', True):
            manager = ServerManager.from_config(server_config)
            logger.info(f"ServerManager from config: backend={server_config.get('backend', 'router_mode')}")
        else:
            manager = ServerManager(backend="router_mode")
            logger.info("Using default ServerManager configuration")

        return manager

    @staticmethod
    def is_cloud_model(model_name: str) -> bool:
        """Check if model is cloud-based (doesn't need local server)."""
        return any(x in model_name.lower() for x in ['gemini', 'gpt', 'claude', 'cloud'])

    @staticmethod
    def initialize_and_start(
        auto_load_model: bool = True
    ) -> Tuple[ServerManager, str]:
        """
        Complete server initialization workflow.

        Args:
            auto_load_model: Whether to load default model immediately

        Returns:
            (server_manager, model_name)
        """
        server_config, model_to_load = ServerService.load_server_config()
        server_manager = ServerService.create_server_manager(server_config)

        # Conditional start based on cloud vs local
        if ServerService.is_cloud_model(model_to_load):
            logger.info(f"Using Cloud Model ({model_to_load}). VLM Server will NOT be started.")
        else:
            if not server_manager.ensure_started():
                logger.error("Failed to start VLM server")
                raise RuntimeError("VLM server startup failed")

            if auto_load_model:
                if not server_manager.load_model(model_to_load):
                    logger.warning(f"Failed to load model '{model_to_load}'")

        return server_manager, model_to_load
