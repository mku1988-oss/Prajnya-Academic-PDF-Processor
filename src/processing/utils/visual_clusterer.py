"""
Visual Region Clusterer
A utility for clustering vector drawings into coherent figure/table regions
using a DPI-aware DBSCAN algorithm.
"""

import logging
from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import DBSCAN

class VisualRegionClusterer:
    """
    Clusters vector drawings from a PDF page into precise figure regions.

    The key feature is its DPI-aware adjustment of the DBSCAN `eps` parameter,
    ensuring consistent clustering regardless of the document's rendering resolution.

    Args:
        eps (float): The maximum distance between two samples for one to be
                     considered as in the neighborhood of the other. This value is
                     calibrated for the `reference_dpi`.
        min_samples (int): The number of samples in a neighborhood for a point
                           to be considered as a core point.
        reference_dpi (int): The DPI for which the `eps` value was calibrated.
    """
    
    def __init__(self, eps: float = 50, min_samples: int = 2, reference_dpi: int = 300):
        self.eps = eps
        self.min_samples = min_samples
        self.reference_dpi = reference_dpi
        self.logger = logging.getLogger(__name__)

    def cluster_drawings(self, drawings: List[Dict], actual_dpi: int) -> List[Dict[str, Any]]:
        """
        Groups nearby vector drawings into logical clusters (e.g., a single figure).

        Args:
            drawings: A list of drawing dictionaries from PyMuPDF's `page.get_drawings()`.
            actual_dpi: The DPI at which the page was rendered for this analysis.

        Returns:
            A list of dictionaries, where each dictionary represents the
            bounding box of a clustered visual region.
        """
        if not drawings:
            return []
            
        # --- DPI-Aware Epsilon Adjustment ---
        # Scale the epsilon value based on the ratio of the actual DPI to the reference DPI.
        # This ensures that the physical distance for clustering remains constant.
        adjusted_eps = self.eps * (actual_dpi / self.reference_dpi)
        self.logger.debug(
            f"Clustering {len(drawings)} drawings with DPI-adjusted eps={adjusted_eps:.2f} "
            f"(Reference: {self.eps} @ {self.reference_dpi} DPI, Actual: {actual_dpi} DPI)"
        )

        # We need representative points for each drawing. The centroid is a good choice.
        points = [self._get_centroid(drawing['rect']) for drawing in drawings]
        
        if not points:
            return []

        # Perform DBSCAN clustering
        points_array = np.array(points)
        db = DBSCAN(eps=adjusted_eps, min_samples=self.min_samples).fit(points_array)
        
        labels = db.labels_
        unique_labels = set(labels)
        
        clustered_regions = []
        # Loop through each found cluster (ignoring noise, label -1)
        for label in unique_labels:
            if label == -1:
                continue

            # Find all points belonging to the current cluster
            cluster_points_indices = np.where(labels == label)[0]
            
            # Get the bounding boxes of all drawings in this cluster
            cluster_rects = [drawings[i]['rect'] for i in cluster_points_indices]
            
            # Merge the bounding boxes to create one large rectangle for the whole figure
            if cluster_rects:
                merged_bbox = self._merge_bboxes(cluster_rects)
                clustered_regions.append({
                    'bbox': merged_bbox,
                    'type': 'drawing_cluster',
                    'component_count': len(cluster_rects)
                })
                
        self.logger.debug(f"Found {len(clustered_regions)} clustered drawing regions.")
        return clustered_regions

    def _get_centroid(self, rect: List[float]) -> List[float]:
        """Calculates the center point (x, y) of a bounding box."""
        x0, y0, x1, y1 = rect
        return [(x0 + x1) / 2, (y0 + y1) / 2]

    def _merge_bboxes(self, bboxes: List[List[float]]) -> List[float]:
        """Merges a list of bounding boxes into a single encompassing bounding box."""
        min_x0 = min(b[0] for b in bboxes)
        min_y0 = min(b[1] for b in bboxes)
        max_x1 = max(b[2] for b in bboxes)
        max_y1 = max(b[3] for b in bboxes)
        return [min_x0, min_y0, max_x1, max_y1]