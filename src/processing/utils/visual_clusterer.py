#!/usr/bin/env python3
"""
Visual Region Clusterer — group scattered vector drawings into whole figures.

For biology researchers: a chart in a born-digital PDF is not stored as one
picture. It is hundreds of tiny vector strokes — every axis tick, bar, and
gridline is a separate drawing object. To decide whether a page "has a figure"
we must first gather those strokes back into coherent regions, which is a
clustering problem: strokes that sit close together belong to the same figure.

We use DBSCAN (a standard density-based clustering algorithm) on the centre
point of each stroke's bounding box. Strokes within `eps` page points of a
neighbour join its cluster; isolated strokes (a single horizontal rule, a stray
underline) are noise and are ignored — which is exactly the behaviour we want,
because `min_samples=2` means one lone stroke never counts as a figure.

Ported from V5's `visual_clusterer.py`. Two deliberate changes:
- Input is now `List[Rect]` from the `PdfBackend` seam (V5 took raw PyMuPDF
  drawing dicts).
- V5 carried a "DPI-aware epsilon adjustment" that always evaluated to
  `eps * (300/300) = eps` on the live path, so it was a no-op; it is dropped.
  `eps` is calibrated in PDF points (the seam's unit), default 50.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.cluster import DBSCAN

from src.processing.pdf.base import Rect


@dataclass(frozen=True)
class DrawingCluster:
    """
    One clustered visual region — probably a figure or a ruled table.

    Attributes:
        rect: the merged bounding box of every stroke in the cluster.
        component_count: how many strokes were merged (useful for debugging —
            a real figure merges dozens; a table's rules merge a handful).
    """

    rect: Rect
    component_count: int


class VisualRegionClusterer:
    """
    Clusters vector drawing boxes from a PDF page into figure-sized regions.

    Args:
        eps: the maximum centre-to-centre distance, in PDF points, for two
            strokes to be considered neighbours (default 50 — about 1.8 cm).
        min_samples: how many neighbours a stroke needs to seed a cluster
            (default 2, so an isolated stroke is never a "figure").
    """

    def __init__(self, eps: float = 50.0, min_samples: int = 2) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.logger = logging.getLogger(__name__)

    def cluster_drawings(self, drawing_rects: Sequence[Rect]) -> List[DrawingCluster]:
        """
        Group nearby vector drawings into logical clusters (e.g. one figure).

        Args:
            drawing_rects: bounding boxes of the page's vector drawings, from
                `PdfPage.drawings()`.

        Returns:
            One `DrawingCluster` per region found. Isolated strokes (DBSCAN
            noise) are dropped.
        """
        if not drawing_rects:
            return []

        # Each stroke is represented by its centre point for clustering.
        centres = np.array([
            [(r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2] for r in drawing_rects
        ])
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(centres).labels_

        clusters: List[DrawingCluster] = []
        for label in sorted(set(labels)):
            if label == -1:
                continue  # noise: isolated strokes are not figures
            members = [drawing_rects[i] for i in np.where(labels == label)[0]]
            clusters.append(DrawingCluster(
                rect=_merge_rects(members),
                component_count=len(members),
            ))

        self.logger.debug(
            "Clustered %d drawings into %d region(s) (eps=%.1f)",
            len(drawing_rects), len(clusters), self.eps,
        )
        return clusters


# --- helpers (after the main class, per CLAUDE.md) -------------------------

def _merge_rects(rects: Sequence[Rect]) -> Rect:
    """The smallest rectangle enclosing every rectangle in the list."""
    return Rect(
        min(r.x0 for r in rects), min(r.y0 for r in rects),
        max(r.x1 for r in rects), max(r.y1 for r in rects),
    )
