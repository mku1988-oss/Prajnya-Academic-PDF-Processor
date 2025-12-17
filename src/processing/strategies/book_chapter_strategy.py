"""
Book Chapter Strategy - V1 Implementation
Processes a book chapter by treating every page as a body page.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
import logging

# Add necessary imports for models and base strategy
if TYPE_CHECKING:
    from ..models import DocumentProfile, ProcessingResult, ContentPage

from ..models import ProcessingResult, ExtractedMetadata, ContentPage
from .base_strategy import DocumentProcessingStrategy
# We will reuse the helpers from the fully implemented ResearchArticleStrategy
from .research_article_strategy import ResearchArticleStrategy 

class BookChapterStrategy(DocumentProcessingStrategy):
    """
    V1 processing strategy for book chapters. This simple version reuses the
    body page processing logic from the ResearchArticleStrategy to extract
    content from every page of the chapter.
    """
    
    def __init__(self):
        """
        Initializes the BookChapterStrategy.
        For V1, we will create an instance of the ResearchArticleStrategy
        to borrow its robust, already-built VLM providers and helpers.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing BookChapterStrategy (V1 - Simple Implementation)")
        # Create a helper instance of the article strategy to use its providers
        # This is an efficient way to reuse the complex VLM setup code.
        self._article_strategy_helper = ResearchArticleStrategy()
        self.logger.info("BookChapterStrategy initialized.")

    def process(self, profile: 'DocumentProfile', test_pages_override: Optional[List[int]] = None) -> 'DocumentProfile':
        """
        Executes the V1 strategy to process a book chapter.
        
        Args:
            profile: The complete DocumentProfile for the file.
            test_pages_override: Optional list of 1-based page numbers for testing.
            
        Returns:
            The updated DocumentProfile with the processing_result populated.
        """
        all_pages: List[ContentPage] = []
        errors = []
        
        self.logger.info(f"Starting V1 book chapter processing for: {profile.pdf_path.name}")

        try:
            # For a book chapter, we can treat all pages as body pages for V1.
            # We can directly call the helper method from the article strategy.
            if test_pages_override:
                # Use the override if provided
                pages_to_process = [p - 1 for p in test_pages_override]
            else:
                # Process all pages (0-indexed)
                pages_to_process = list(range(profile.page_count))

            self.logger.info(f"Processing {len(pages_to_process)} pages as body content...")
            all_pages = self._article_strategy_helper._process_body_pages(profile, pages_to_process)

        except Exception as e:
            error_message = f"BookChapterStrategy execution failed: {type(e).__name__} - {e}"
            self.logger.error(error_message, exc_info=True)
            errors.append(error_message)
        
        finally:
            # This block always runs to ensure a result is created.
            self.logger.info(f"Finalizing book chapter processing. Total pages processed: {len(all_pages)}")
            
            # For V1, metadata extraction is minimal. We can enhance this in future versions
            # by adding logic to find chapter titles, ISBNs, etc.
            final_metadata = ExtractedMetadata(title=profile.pdf_path.stem) # Use filename as a placeholder title

            processing_output = ProcessingResult(
                metadata=final_metadata,
                pages=all_pages,
                errors=errors
            )
            profile.processing_result = processing_output

        self.logger.info(f"Book chapter processing complete.")
        return profile