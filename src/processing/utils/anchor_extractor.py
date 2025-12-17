"""
Utility for extracting document anchoring text from PDFs.
This logic is replicated from the successful integration tests.
"""
from pathlib import Path
from typing import Union

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF is required for document anchoring. Please install with 'pip install PyMuPDF'")


class DocumentAnchoringExtractor:
    """
    Utility class for extracting document anchoring text from PDFs.
    
    Based on official OlmOCR research (arXiv 2502.18443), document anchoring
    provides +15-20% quality improvement by giving the VLM textual context.
    """
    
    @staticmethod
    def extract_document_anchoring(pdf_path: Union[str, Path], page_num: int, max_chars: int = 6000) -> str:
        """
        Extracts a sparse set of document anchors for high performance.

        This method uses PyMuPDF to get all text blocks on the page but only
        samples the first few words from each block. This provides a lightweight
        "skeleton" of the entire page layout.

        Args:
            pdf_path: Path to PDF file.
            page_num: Page number to extract from (1-indexed).
            max_chars: Maximum characters for the final anchor text.

        Returns:
            A sparse but comprehensive anchoring text string.
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num - 1) # Use 0-based index for fitz
            
            page_rect = page.mediabox
            page_height = float(page_rect.height)
            
            anchor_text = ""
            blocks = page.get_text("blocks")
            
            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                
                if block_type != 0: continue # Only process text blocks

                words = text.strip().replace('\n', ' ').split()
                sampled_text = ' '.join(words[:8])
                
                if not sampled_text: continue

                y_lower_left = page_height - y0
                block_anchor = f"[{int(x0)},{int(y_lower_left)}]{sampled_text}\n"
                
                if len(anchor_text) + len(block_anchor) > max_chars: break
                
                anchor_text += block_anchor
            
            doc.close()
            
            return anchor_text if anchor_text else f"Page dimensions: {page_rect.width}x{page_height}\n"

        except Exception as e:
            print(f"Warning: Document anchoring extraction failed: {e}")
            return f"Page dimensions: 612x792\n"  # Generic fallback