"""
Book Strategy - V1 Implementation
Processes a full book by treating every page as a body page.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ..models import DocumentProfile, ProcessingResult, ContentPage

from ..models import ProcessingResult, ExtractedMetadata, ContentPage
from .base_strategy import DocumentProcessingStrategy
from .research_article_strategy import ResearchArticleStrategy # We will reuse its helpers

class BookStrategy(DocumentProcessingStrategy):
    """
    V1 processing strategy for books. This simple version reuses the
    body page processing logic from the ResearchArticleStrategy to extract
    content from every page of the book.
    """
    
    def __init__(self):
        """
        Initializes the BookStrategy.
        For V1, we will create an instance of the ResearchArticleStrategy
        to borrow its robust, already-built VLM providers and helpers.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing BookStrategy (V1 - Simple Implementation)")
        # Create a helper instance of the article strategy to use its providers
        self._article_strategy_helper = ResearchArticleStrategy()
        self.logger.info("BookStrategy initialized.")

    def process(self, profile: 'DocumentProfile', test_pages_override: Optional[List[int]] = None) -> 'DocumentProfile':
        """
        Executes the V1 strategy to process a book.
        """
        all_pages: List[ContentPage] = []
        errors = []
        
        self.logger.info(f"Starting V1 book processing for: {profile.pdf_path.name}")

        try:
            # For a book, we treat all pages as body pages.
            # We can directly call the helper method from the article strategy.
            pages_to_process = list(range(profile.page_count)) # Process all pages (0-indexed)

            self.logger.info(f"Processing all {len(pages_to_process)} pages as body content...")
            all_pages = self._article_strategy_helper._process_body_pages(profile, pages_to_process)

        except Exception as e:
            error_message = f"BookStrategy execution failed: {type(e).__name__} - {e}"
            self.logger.error(error_message, exc_info=True)
            errors.append(error_message)
        
        finally:
            # Finalize the result, even if errors occurred.
            self.logger.info(f"Finalizing book processing. Total pages processed: {len(all_pages)}")
            
            # For V1, metadata extraction is minimal. We can enhance this later.
            final_metadata = ExtractedMetadata(title=profile.pdf_path.stem) # Use filename as title for now
            
            processing_output = ProcessingResult(
                metadata=final_metadata,
                pages=all_pages,
                errors=errors
            )
            profile.processing_result = processing_output

        self.logger.info(f"Book processing complete.")
        return profile