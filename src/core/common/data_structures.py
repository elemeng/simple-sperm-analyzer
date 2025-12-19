#!/usr/bin/env python3
"""
Data structures used across multiple modules in the sperm tracking project.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """Detection data structure used across multiple modules."""

    frame: int
    x: float
    y: float
    area: float
    solidity: float
    curvature_signature: Optional[np.ndarray] = None
    contour: Optional[np.ndarray] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
