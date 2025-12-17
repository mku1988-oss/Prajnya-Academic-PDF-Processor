"""
Page Classifier for 3-Tier Hybrid Processing

Classifies PDF pages by complexity to route to appropriate processing tier:
- TEXT_ONLY: Pure text pages → PyMuPDF (99% faster)
- SIMPLE_VISUAL: Simple figures/tables → OlmoCR Markdown (82% faster)
- COMPLEX_VISUAL: Complex diagrams/equations → Surgical VLM (Targeted precision)

Uses multi-signal analysis:
1. Text density from PyMuPDF
2. Visual element detection (images, drawings)
3. LaTeX equation detection (4 methods)
4. Confidence scoring with VLM verification

Author: Shastrasaara Development Team
Date: October 11, 2025
Version: 1.0
"""

import logging
import re
import fitz # PyMuPDF
from typing import Tuple, Dict, Any, Optional, List

# Import PageType from models
from src.processing.models import PageType
from .visual_clusterer import VisualRegionClusterer


class PageClassifier:
    """
    Classifies PDF pages by complexity for intelligent routing.
    
    Uses confidence-based multi-signal analysis to determine whether
    a page can be processed quickly with PyMuPDF or requires VLM.
    
    Classification Rules:
    1. Page 0 (index) → TITLE_PAGE (always handled specially)
    2. High equation confidence → COMPLEX_VISUAL
    3. Many visual elements → SIMPLE_VISUAL
    4. High text density + no visuals → TEXT_ONLY
    5. Default → SIMPLE_VISUAL (safer fallback)
    
    Attributes:
        config: Classification configuration dict
        confidence_threshold: Minimum confidence for TEXT_ONLY routing (default 0.85)
        visual_threshold: Number of images to classify as SIMPLE_VISUAL (default 2)
        enable_verification: Whether to use VLM to verify low-confidence equations
        logger: Logger instance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize page classifier with configuration.
        
        Args:
            config: Configuration dict with 'hybrid_processing' section
                   containing 'page_classifier' settings
        """
        self.logger = logging.getLogger("PageClassifier")
        
        # Extract classification config
        hybrid_config = config.get('hybrid_processing', {})
        classifier_config = hybrid_config.get('page_classifier', {})
        
        # Core thresholds
        self.confidence_threshold = classifier_config.get('confidence_threshold', 0.85)
        self.visual_threshold = classifier_config.get('visual_threshold', 2)
        self.enable_verification = classifier_config.get('enable_vlm_verification', False)
        
        # Equation detection thresholds
        eq_config = classifier_config.get('equation', {})
        self.eq_total_conf_threshold = eq_config.get('total_confidence_threshold', 1.5)
        self.eq_high_conf_count = eq_config.get('high_confidence_count', 2)
        self.eq_high_conf_min = eq_config.get('high_confidence_min', 0.85)
        
        # Initialize the VisualRegionClusterer from the hybrid_processing config
        hybrid_config = config.get('hybrid_processing', {})
        cluster_config = hybrid_config.get('visual_clustering', {})
        
        self.visual_clusterer = VisualRegionClusterer(
            eps=float(cluster_config.get('dbscan_eps', 50)),
            min_samples=int(cluster_config.get('dbscan_min_samples', 2)),
            reference_dpi=int(cluster_config.get('reference_dpi', 300))
        )
        self.logger.info("DPI-Aware VisualRegionClusterer initialized for V2.")
        
        self.logger.info(f"PageClassifier initialized (confidence={self.confidence_threshold})")
    
    def classify(
        self,
        page,  # fitz.Page object
        page_idx: int,
        pymupdf_result: Optional[Dict[str, Any]] = None,
        vlm_provider = None
    ) -> Tuple[PageType, Dict[str, Any]]:
        """
        Classify a page by complexity.
        
        Args:
            page: PyMuPDF page object (fitz.Page)
            page_idx: 0-based page index
            pymupdf_result: Optional pre-extracted PyMuPDF text result
            vlm_provider: Optional VLM provider for equation verification
            
        Returns:
            Tuple of (PageType, analysis_dict)
            
        Example:
            >>> page_type, analysis = classifier.classify(page, 0)
            >>> print(page_type)  # PageType.TEXT_ONLY
            >>> print(analysis['classification_reason'])  # 'high text density (0.92)'
        """
        analysis = {
            'page_index': page_idx,
            'equation_regions': [],
            'visual_regions': [],
            'text_density': 0.0,
            'classification_reason': '',
            'verification_performed': False
        }
        
        # RULE 1: First page is always title page
        if page_idx == 0:
            analysis['classification_reason'] = 'first_page_title'
            self.logger.debug(f"Page 1 classified as TITLE_PAGE (always)")
            return PageType.TEXT_ONLY, analysis  # Title page handled as text
        
        # RULE 2: Detect equations (high priority - complex content)
        equation_regions = self._detect_equation_regions(page)
        analysis['equation_regions'] = equation_regions
        
        # Calculate equation confidence
        total_confidence = sum(eq['confidence'] for eq in equation_regions)
        high_confidence_count = sum(
            1 for eq in equation_regions 
            if eq['confidence'] >= self.eq_high_conf_min
        )
        
        analysis['equation_total_confidence'] = total_confidence
        analysis['equation_high_confidence_count'] = high_confidence_count
        
        # VLM Verification for low-confidence equations (if enabled)
        if self.enable_verification and vlm_provider and equation_regions:
            verified_regions = self._verify_low_confidence_regions(
                page, equation_regions, vlm_provider
            )
            # Recalculate after verification
            total_confidence = sum(eq['confidence'] for eq in verified_regions)
            high_confidence_count = sum(
                1 for eq in verified_regions 
                if eq['confidence'] >= self.eq_high_conf_min
            )
            analysis['equation_regions'] = verified_regions
            analysis['verification_performed'] = True
        
        # Check equation thresholds
        if (total_confidence > self.eq_total_conf_threshold or 
            high_confidence_count >= self.eq_high_conf_count):
            
            analysis['classification_reason'] = (
                f'{len(equation_regions)} equations '
                f'(conf={total_confidence:.2f}, high_conf={high_confidence_count})'
            )
            self.logger.debug(
                f"Page {page_idx+1} classified as COMPLEX_VISUAL: {analysis['classification_reason']}"
            )
            return PageType.COMPLEX_VISUAL, analysis
        
        # RULE 3: Detect visual elements (figures/tables)
        visual_regions = self._detect_visual_regions(page)
        analysis['visual_count'] = len(visual_regions)
        analysis['visual_regions'] = visual_regions
        
        if len(visual_regions) >= self.visual_threshold:
            analysis['classification_reason'] = f'{len(visual_regions)} visual elements detected'
            self.logger.debug(
                f"Page {page_idx+1} classified as SIMPLE_VISUAL ({len(visual_regions)} regions)"
            )
            return PageType.SIMPLE_VISUAL, analysis
        
        # RULE 4: Check text density for TEXT_ONLY classification
        if pymupdf_result:
            text_density = self._calculate_text_density(pymupdf_result)
            analysis['text_density'] = text_density
            
            # High text density + no visuals + no equations = TEXT_ONLY
            if (text_density >= self.confidence_threshold and 
                len(visual_regions) == 0 and 
                len(equation_regions) == 0):
                
                analysis['classification_reason'] = f'high text density ({text_density:.2f})'
                self.logger.debug(f"Page {page_idx+1} classified as TEXT_ONLY")
                return PageType.TEXT_ONLY, analysis
        
        # RULE 5: Default to SIMPLE_VISUAL (safer fallback - uses VLM)
        analysis['classification_reason'] = 'mixed_content_fallback'
        self.logger.debug(f"Page {page_idx+1} classified as SIMPLE_VISUAL (mixed content)")
        return PageType.SIMPLE_VISUAL, analysis
    
    def _detect_equation_regions(self, page) -> List[Dict[str, Any]]:
        """
        Detect mathematical equations using multi-method approach.
        
        Methods:
        1. LaTeX syntax detection (high confidence)
        2. Unicode math symbols (medium confidence)
        3. Centered single-line text (low confidence)
        4. Embedded equation images (high confidence)
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of equation region dicts with keys:
                - bbox: (x0, y0, x1, y1)
                - confidence: 0.0-1.0
                - method: detection method name
                - text: extracted text (if available)
        """
        equations = []
        
        # Method 1: LaTeX syntax in text blocks
        text_blocks = page.get_text("blocks")
        latex_patterns = [
            r'\\frac\{',
            r'\\int_',
            r'\\sum_',
            r'\\prod_',
            r'\\sqrt\{',
            r'\\alpha|\\beta|\\gamma|\\delta',
            r'\^[{].*?[}]',  # Superscripts
            r'_[{].*?[}]'    # Subscripts
        ]
        
        for block in text_blocks:
            if len(block) < 7:  # Blocks have 7 elements
                continue
            bbox = block[:4]
            text = block[4]
            
            # Check for LaTeX patterns
            latex_match_count = sum(
                1 for pattern in latex_patterns 
                if re.search(pattern, text)
            )
            
            if latex_match_count >= 2:  # At least 2 LaTeX patterns
                equations.append({
                    'bbox': bbox,
                    'confidence': min(0.95, 0.70 + latex_match_count * 0.05),
                    'method': 'latex_syntax',
                    'text': text[:100]  # First 100 chars
                })
        
        # Method 2: Unicode math symbols
        math_unicode_ranges = [
            ('\u2200', '\u22FF'),  # Mathematical Operators
            ('\u2A00', '\u2AFF'),  # Supplemental Math Operators
            ('\u27C0', '\u27EF'),  # Miscellaneous Math Symbols-A
            ('\u2980', '\u29FF'),  # Miscellaneous Math Symbols-B
        ]
        
        for block in text_blocks:
            if len(block) < 7:
                continue
            bbox = block[:4]
            text = block[4]
            
            # Count math unicode characters
            math_char_count = sum(
                1 for char in text
                for start, end in math_unicode_ranges
                if start <= char <= end
            )
            
            if math_char_count >= 3:  # At least 3 math symbols
                equations.append({
                    'bbox': bbox,
                    'confidence': min(0.85, 0.60 + math_char_count * 0.05),
                    'method': 'unicode_math',
                    'text': text[:100]
                })
        
        # Method 3: Centered single-line text (low confidence)
        page_width = page.rect.width
        
        for block in text_blocks:
            if len(block) < 7:
                continue
            bbox = block[:4]
            text = block[4].strip()
            
            # Check if single line, centered, and short
            line_count = text.count('\n') + 1
            block_width = bbox[2] - bbox[0]
            block_center = (bbox[0] + bbox[2]) / 2
            page_center = page_width / 2
            
            is_centered = abs(block_center - page_center) < page_width * 0.2
            is_single_line = line_count == 1
            is_short = len(text) < 80
            
            if is_centered and is_single_line and is_short:
                equations.append({
                    'bbox': bbox,
                    'confidence': 0.40,  # Low confidence - needs verification
                    'method': 'centered_text',
                    'text': text
                })
        
        # Method 4: Embedded equation images
        images = page.get_images(full=True)
        
        for img in images:
            xref = img[0]
            try:
                bbox = page.get_image_bbox(xref)
            except (ValueError, RuntimeError) as e:
                # Skip images with invalid xref or other errors
                self.logger.debug(f"Skipping image {xref}: {e}")
                continue
            
            # Heuristic: small images centered on page might be equations
            img_width = bbox.width
            img_height = bbox.height
            aspect_ratio = img_width / img_height if img_height > 0 else 0
            
            is_small = img_width < page_width * 0.5 and img_height < 100
            is_horizontal = 0.5 < aspect_ratio < 10
            
            if is_small and is_horizontal:
                equations.append({
                    'bbox': (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                    'confidence': 0.75,
                    'method': 'embedded_image',
                    'text': f'[equation_image_{xref}]'
                })
        
        return equations
    
    def _verify_low_confidence_regions(
        self,
        page,
        regions: List[Dict[str, Any]],
        vlm_provider
    ) -> List[Dict[str, Any]]:
        """
        Use VLM to verify low-confidence equation detections.
        
        Args:
            page: PyMuPDF page object
            regions: List of detected regions
            vlm_provider: VLM provider instance
            
        Returns:
            Updated regions list with verified confidences
        """
        import fitz  # Import here to avoid circular dependency
        
        verified = []
        
        for region in regions:
            if region['confidence'] < 0.70:
                # Extract region as image and verify with VLM
                try:
                    # Get region bbox
                    bbox = region['bbox']
                    mat = page.get_pixmap(clip=bbox, matrix=fitz.Matrix(2, 2))
                    img_bytes = mat.tobytes("png")
                    
                    # Ask VLM: "Is this a mathematical equation? (yes/no)"
                    prompt = "Is this a mathematical equation or formula? Answer only 'yes' or 'no'."
                    response = vlm_provider.generate_with_image(img_bytes, prompt)
                    
                    if 'yes' in response.lower():
                        region['confidence'] = 0.90  # Upgrade confidence
                        region['method'] += '_vlm_verified'
                    else:
                        region['confidence'] = max(0.20, region['confidence'] - 0.20)
                    
                except Exception as e:
                    self.logger.warning(f"VLM verification failed: {e}")
            
            verified.append(region)
        
        return verified
    
    def _detect_visual_regions(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """
        Detects visual elements (images and clustered vector drawings) on a page.
        Uses V2 DPI-aware clustering with smart filtering to ignore:
        - Headers/footers (top/bottom 10% of page)
        - Small decorative elements (< 5000 px² area)
        - Thin lines (aspect ratio > 10:1 or < 1:10)
        - Tiny images (< 50px width or height)
        """
        visuals = []
        
        # Get page dimensions for filtering
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Define exclusion zones (header/footer regions)
        header_zone = page_height * 0.10  # Top 10%
        footer_zone = page_height * 0.90  # Bottom 10%
        
        # Minimum area threshold (filter out small decorative elements)
        MIN_BBOX_AREA = 5000  # pixels²
        MIN_IMAGE_SIZE = 50   # minimum width or height in pixels
        MAX_ASPECT_RATIO = 10.0  # filter out thin lines
        
        # --- Step 1: Detect Embedded Images with Smart Filtering ---
        images = page.get_images(full=True)
        filtered_count = 0
        
        for img_info in images:
            try:
                bbox = page.get_image_bbox(img_info)
                
                # Calculate dimensions
                img_width = bbox.width
                img_height = bbox.height
                img_area = img_width * img_height
                
                # Calculate aspect ratio (width/height)
                aspect_ratio = img_width / img_height if img_height > 0 else 0
                
                # Get center Y coordinate
                center_y = (bbox.y0 + bbox.y1) / 2
                
                # FILTER 1: Skip if in header/footer zone
                if center_y < header_zone or center_y > footer_zone:
                    self.logger.debug(f"Filtered image in header/footer zone: y={center_y:.0f}")
                    filtered_count += 1
                    continue
                
                # FILTER 2: Skip if too small
                if img_width < MIN_IMAGE_SIZE or img_height < MIN_IMAGE_SIZE:
                    self.logger.debug(f"Filtered tiny image: {img_width:.0f}x{img_height:.0f}")
                    filtered_count += 1
                    continue
                
                # FILTER 3: Skip if area too small (decorative elements)
                if img_area < MIN_BBOX_AREA:
                    self.logger.debug(f"Filtered small image: area={img_area:.0f}px²")
                    filtered_count += 1
                    continue
                
                # FILTER 4: Skip if aspect ratio suggests thin line
                if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < (1/MAX_ASPECT_RATIO):
                    self.logger.debug(f"Filtered thin line image: aspect={aspect_ratio:.2f}")
                    filtered_count += 1
                    continue
                
                # Passed all filters - add to visuals
                visuals.append({
                    'bbox': (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                    'type': 'image',
                    'size': (img_width, img_height)
                })
                
            except ValueError as e:
                self.logger.warning(f"Could not get bbox for image xref {img_info[0]}: {e}")

        # --- Step 2: Detect and Cluster Vector Drawings with Smart Filtering ---
        analysis_dpi = 300
        drawings = page.get_drawings()
        clustered_drawing_regions = self.visual_clusterer.cluster_drawings(
            drawings, 
            actual_dpi=analysis_dpi
        )
        
        # Apply same filters to drawing clusters
        drawing_filtered_count = 0
        for region in clustered_drawing_regions:
            bbox = region['bbox']  # [x0, y0, x1, y1]
            
            # Calculate dimensions
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Apply filters
            if center_y < header_zone or center_y > footer_zone:
                self.logger.debug(f"Filtered drawing in header/footer zone: y={center_y:.0f}")
                drawing_filtered_count += 1
                continue
            
            if area < MIN_BBOX_AREA:
                self.logger.debug(f"Filtered small drawing: area={area:.0f}px²")
                drawing_filtered_count += 1
                continue
            
            if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < (1/MAX_ASPECT_RATIO):
                self.logger.debug(f"Filtered thin line drawing: aspect={aspect_ratio:.2f}")
                drawing_filtered_count += 1
                continue
            
            # Passed filters - add to visuals
            visuals.append(region)
        
        # Log filtering results
        total_filtered = filtered_count + drawing_filtered_count
        self.logger.info(
            f"Visual detection: {len(images)} images ({filtered_count} filtered), "
            f"{len(clustered_drawing_regions)} drawings ({drawing_filtered_count} filtered) → "
            f"{len(visuals)} valid visual regions"
        )
        
        return visuals
    
    def _calculate_text_density(self, pymupdf_result: Dict[str, Any]) -> float:
        """
        Calculate text density score (0.0-1.0).
        
        High density indicates substantial, well-structured text suitable for PyMuPDF.
        Low density indicates sparse text (captions, labels) requiring VLM.
        
        Args:
            pymupdf_result: PyMuPDF extraction result dict with keys:
                - word_count: Number of words
                - avg_words_per_line: Average words per line
                - char_count: Total characters
                
        Returns:
            Density score from 0.0 (empty) to 1.0 (dense text)
        """
        word_count = pymupdf_result.get('word_count', 0)
        avg_words_per_line = pymupdf_result.get('avg_words_per_line', 0)
        char_count = pymupdf_result.get('char_count', 0)
        
        # Component scores
        word_score = min(1.0, word_count / 300)  # 300+ words = max score
        line_score = min(1.0, avg_words_per_line / 10)  # 10+ words/line = max
        char_score = min(1.0, char_count / 2000)  # 2000+ chars = max
        
        # Weighted combination
        density = (
            word_score * 0.4 +
            line_score * 0.3 +
            char_score * 0.3
        )
        
        return density


# Helper function for testing
def create_classifier_from_config(config_path: str) -> PageClassifier:
    """
    Create PageClassifier from YAML config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Initialized PageClassifier
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return PageClassifier(config)
