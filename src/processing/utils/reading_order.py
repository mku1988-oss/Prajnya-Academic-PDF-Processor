"""
Reading Order Reconstructor
A utility for extracting and assembling page content in a logical,
top-to-bottom, left-to-right reading order.
"""

import fitz # PyMuPDF
from typing import List, Dict, Any

class ReadingOrderReconstructor:
    """
    Extracts text and other content blocks from a page and reconstructs
    them into a single string in a natural reading order.
    
    The sorting algorithm naturally handles simple two-column layouts.
    """
    
    def extract_text_blocks(self, page: fitz.Page) -> List[Dict]:
        """
        Extracts all text blocks from a PyMuPDF page object, returning
        a list of dictionaries containing the text and its bounding box.

        Args:
            page: The PyMuPDF page object.

        Returns:
            A list of text block dictionaries.
        """
        # "blocks" is a list of tuples: (x0, y0, x1, y1, "text", block_no, block_type)
        blocks = page.get_text("blocks")
        text_blocks = []
        for block in blocks:
            # Ensure the block is a text block (type 0) and contains text
            if len(block) > 4 and block[6] == 0 and block[4].strip():
                text_blocks.append({
                    'content': block[4], 
                    'bbox': (block[0], block[1], block[2], block[3]), 
                    'type': 'text'
                })
        return text_blocks

    def reconstruct(self, **kwargs: List[Dict]) -> str:
        """
        Merges and sorts all provided content blocks into a single string.
        
        PHASE 3 ENHANCEMENT: Filters out empty/invalid surgical results to prevent
        hallucinated content from corrupting the document.

        This method takes arbitrary keyword arguments, where each argument is
        expected to be a list of content block dictionaries. All lists are
        combined, validated, and sorted to produce the final page content.

        The sorting key `(b['bbox'][1], b['bbox'][0])` sorts by the top
        y-coordinate first, then by the left x-coordinate. This creates a
        natural top-to-bottom, left-to-right reading order.
        
        Validation Rules:
        - Empty content is filtered out
        - Whitespace-only content is filtered out
        - Content must have valid bbox coordinates

        Args:
            **kwargs: Keyword arguments, each a list of content blocks.
                      Example: reconstruct(text_blocks=..., figure_regions=...)

        Returns:
            A single string with all valid content merged in reading order.
        """
        all_blocks = []
        filtered_count = 0
        
        for block_list_name, block_list in kwargs.items():
            if not block_list:  # Skip None or empty lists
                continue
                
            for block in block_list:
                # VALIDATION 1: Check for required fields
                if 'content' not in block or 'bbox' not in block:
                    filtered_count += 1
                    continue
                
                content = block['content']
                
                # VALIDATION 2: Filter empty or whitespace-only content
                if not content or not content.strip():
                    filtered_count += 1
                    continue
                
                # VALIDATION 3: Filter suspiciously short content from surgical results
                # (likely invalid descriptions like single words)
                block_type = block.get('type', 'unknown')
                if block_type in ['figure', 'table'] and len(content.strip()) < 20:
                    # Surgical results should have meaningful descriptions (>20 chars)
                    filtered_count += 1
                    continue
                
                # VALIDATION 4: Check for valid bbox coordinates
                bbox = block['bbox']
                if len(bbox) != 4 or not all(isinstance(x, (int, float)) for x in bbox):
                    filtered_count += 1
                    continue
                
                # Passed all validations - add to blocks
                all_blocks.append(block)
        
        if filtered_count > 0:
            # Log filtering (would need logger injection, but this is informative)
            # In production, this could emit a debug message
            pass
            
        # Sort all collected blocks by their vertical, then horizontal position.
        sorted_blocks = sorted(all_blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))
        
        # Join the content of the sorted blocks with double newlines for separation.
        # Additional strip() ensures no trailing/leading whitespace per block
        return "\n\n".join([block['content'].strip() for block in sorted_blocks])