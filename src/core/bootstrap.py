"""
Application bootstrap - shared initialization logic.
"""
from typing import Tuple, TYPE_CHECKING
from dotenv import load_dotenv

from src.core.services.server_service import ServerService
from src.core.services.ledger_service import LedgerService
from src.core.services.logging_service import LoggingService
from src.processing.document_orchestrator import DocumentOrchestrator

if TYPE_CHECKING:
    from src.llm.providers.server_manager import ServerManager
    from src.processing.ledger.semantic_ledger import SemanticLedger

class PrajnyaBootstrap:
    """Shared initialization logic for CLI and GUI."""

    @staticmethod
    def initialize_environment() -> None:
        """Load environment variables (.env file) and configure console encoding."""
        load_dotenv()
        
        # Windows Console Encoding Fix
        import sys
        if sys.platform == 'win32':
             # Reconfigure stdout/stderr to utf-8 to handle special chars (✓, →, etc.)
            if hasattr(sys.stdout, 'reconfigure'):
                 try:
                    sys.stdout.reconfigure(encoding='utf-8')
                    sys.stderr.reconfigure(encoding='utf-8')
                 except Exception:
                     pass # Fallback for edge cases

    @staticmethod
    def initialize_logging(log_file: str = None) -> None:
        """Setup logging."""
        LoggingService.setup_logging(log_file=log_file)

    @staticmethod
    def initialize_core_services(
        start_server: bool = True
    ) -> Tuple[DocumentOrchestrator, 'ServerManager', 'SemanticLedger']:
        """
        Initialize all core services.

        Args:
            start_server: Whether to start VLM server immediately

        Returns:
            (orchestrator, server_manager, ledger)
        """
        # Load environment
        PrajnyaBootstrap.initialize_environment()

        # Create orchestrator
        orchestrator = DocumentOrchestrator()

        # Create server manager
        if start_server:
            server_manager, _ = ServerService.initialize_and_start()
        else:
            server_manager = ServerService.create_server_manager()

        # Create ledger
        ledger = LedgerService.get_ledger_instance()

        return orchestrator, server_manager, ledger
