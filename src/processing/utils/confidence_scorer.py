"""
Confidence Scorer - Utilities for Calculating Classification Confidence

Provides methods for:
- Normalizing scores to 0.0-1.0 range
- Weighted averaging
- Threshold validation

Blueprint Reference: Section 1.1
"""

from typing import Dict


class ConfidenceScorer:
    """
    Utilities for confidence scoring in classification.
    
    Used by DocumentClassifier to:
        - Normalize signal scores (0-100 → 0.0-1.0)
        - Apply weighted averaging
        - Validate confidence thresholds
        - Apply fallback logic
    """
    
    @staticmethod
    def normalize_score(score: int, max_score: int = 100) -> float:
        """
        Normalize score to 0.0-1.0 range.
        
        Args:
            score: Raw score (typically 0-100)
            max_score: Maximum possible score (default 100)
        
        Returns:
            Normalized confidence (0.0-1.0)
        
        Example:
            >>> ConfidenceScorer.normalize_score(85, 100)
            0.85
        """
        # Handle invalid max_score (zero or negative)
        if max_score <= 0:
            return 0.0
        
        # Clamp score to be within 0 to max_score range (prevent negative scores)
        normalized_score = max(0, score) / max_score
        return min(normalized_score, 1.0)
    
    @staticmethod
    def weighted_average(scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Calculate weighted average of scores.
        
        Args:
            scores: Dictionary of {signal_name: score}
            weights: Dictionary of {signal_name: weight}
        
        Returns:
            Weighted average confidence
        
        Example:
            >>> scores = {'metadata': 0.9, 'page_count': 0.6, 'text': 0.8, 'structure': 0.85}
            >>> weights = {'metadata': 0.3, 'page_count': 0.2, 'text': 0.3, 'structure': 0.2}
            >>> ConfidenceScorer.weighted_average(scores, weights)
            0.805
        """
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(scores.get(key, 0.0) * weight for key, weight in weights.items())
        return weighted_sum / total_weight
    
    @staticmethod
    def apply_confidence_threshold(confidence: float, min_threshold: float = 0.5) -> float:
        """
        Apply minimum confidence threshold.
        
        If confidence below threshold, returns threshold value.
        This ensures all classifications have minimum baseline confidence.
        
        Args:
            confidence: Calculated confidence (0.0-1.0)
            min_threshold: Minimum allowed confidence (default 0.5)
        
        Returns:
            Confidence clamped to minimum threshold
        
        Example:
            >>> ConfidenceScorer.apply_confidence_threshold(0.3, 0.5)
            0.5
            >>> ConfidenceScorer.apply_confidence_threshold(0.7, 0.5)
            0.7
        """
        return max(confidence, min_threshold)
    
    @staticmethod
    def select_winner(scores: Dict[str, int]) -> tuple[str, int]:
        """
        Select document type with highest score.
        
        Args:
            scores: Dictionary of {document_type: score}
                Example: {'research_article': 85, 'book_chapter': 45, 'book': 30}
        
        Returns:
            (winner_type, winner_score)
            Returns (None, 0) if scores dictionary is empty
        
        Example:
            >>> scores = {'research_article': 85, 'book_chapter': 45, 'book': 30}
            >>> ConfidenceScorer.select_winner(scores)
            ('research_article', 85)
            >>> ConfidenceScorer.select_winner({})
            (None, 0)
        """
        # Handle empty scores dictionary to prevent ValueError
        if not scores:
            return (None, 0)
        
        winner_type = max(scores, key=scores.get)
        winner_score = scores[winner_type]
        return winner_type, winner_score
