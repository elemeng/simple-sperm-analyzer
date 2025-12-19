"""
Common package for sperm tracking project - provides shared utilities across all modules.

Submodules:
    io: File input/output operations
    math_utils: Mathematical algorithms and functions
    utils: General utility function
"""

from .data_structures import Detection
from .io import load_movie
from .math_utils import (
    calculate_distances,
    compute_iou,
    compute_curvature_signature,
)

__all__ = [
    "Detection",
    "load_movie",
    "calculate_distances",
    "compute_iou",
    "compute_curvature_signature",
]
