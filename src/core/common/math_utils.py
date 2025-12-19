#!/usr/bin/env python3
"""
Mathematical utilities and algorithms used across multiple modules in the sperm tracking project.
"""

import numpy as np
from typing import Tuple


def calculate_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distances between consecutive points."""
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx**2 + dy**2)


def compute_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two bboxes: (x, y, w, h)"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Intersection
    ixmin = max(x1, x2)
    iymin = max(y1, y2)
    ixmax = min(x1 + w1, x2 + w2)
    iymax = min(y1 + h1, y2 + h2)
    iw = max(0, ixmax - ixmin)
    ih = max(0, iymax - iymin)

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - (iw * ih)

    return (iw * ih) / union if union > 0 else 0.0


def compute_curvature_signature(contour: np.ndarray, n_points: int = 20) -> np.ndarray:
    """
    Compute curvature signature for a contour.
    This is a placeholder implementation - the actual function may exist in the original codebase.
    """
    if len(contour) < 3:
        return np.array([0.0] * n_points)

    # Simplified curvature calculation
    # This would need to be replaced with the actual implementation if it exists elsewhere
    contour_reshaped = contour.reshape(-1, 2)
    if len(contour_reshaped) < n_points:
        # Pad with zeros if not enough points
        signature = np.zeros(n_points)
        signature[: len(contour_reshaped)] = np.linalg.norm(
            np.diff(contour_reshaped, axis=0), axis=1
        )[:n_points]
        return signature
    else:
        # Downsample to n_points
        step = len(contour_reshaped) // n_points
        selected_points = contour_reshaped[::step][:n_points]
        signature = np.linalg.norm(np.diff(selected_points, axis=0), axis=1)
        # Pad if needed
        if len(signature) < n_points:
            signature = np.pad(
                signature, (0, n_points - len(signature)), mode="constant"
            )
        return signature[:n_points]
