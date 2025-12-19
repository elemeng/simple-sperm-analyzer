#!/usr/bin/env python3
"""
Immotile Sperm Tracker - Two-stage pipeline for separating immotile and motile sperm tracking.

This module implements an offline, deterministic algorithm to identify and separate
immotile sperm tracks before passing remaining detections to the main motile tracker.

Core Algorithm:
1. Hypothesis: All detections in first K frames are potential immotile anchors
2. Evidence collection: Find nearby detections within radius r across all frames
3. Outlier rejection: Fit polynomial and iteratively remove largest residuals
4. Decision: Confirm immotile tracks based on stability criteria
5. Removal: Remove immotile detections, pass remainder to motile tracker

Parallel Processing:
- Uses joblib Parallel for automatic parallelization
- Handles both parallel and sequential processing transparently
- Joblib automatically optimizes batch size for performance
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import logging

# Parallel processing imports
try:
    from joblib import Parallel, delayed
    import multiprocessing as mp

    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    Parallel = None
    delayed = None
    mp = None

logger = logging.getLogger(__name__)


@dataclass
class ImmotileConfig:
    """Configuration for immotile sperm detection and tracking."""

    # Hypothesis parameters
    first_k_frames: int = 3  # Number of initial frames to use as anchors
    search_radius: float = 6.0  # Pixel radius for finding nearby detections

    # Evidence collection parameters
    min_points: int = 8  # Minimum detections needed to confirm immotile track
    max_std: float = 1.5  # Maximum spatial standard deviation for stability

    # Polynomial fitting parameters
    poly_order: int = 1  # Polynomial order (0=static, 1=linear drift)
    outlier_percentile: float = 90.0  # Percentile for outlier removal
    max_iterations: int = 2  # Maximum outlier removal iterations

    # Track validation
    min_track_length: int = 6  # Minimum confirmed track length
    spatial_tolerance: float = 2.0  # Additional tolerance for spatial validation

    # Parallel processing parameters
    n_jobs: int = -1  # Number of parallel jobs (-1 = use all cores, 1 = sequential)
    batch_size: int | str = "auto"  # Auto-calculate batch size ('auto' = joblib auto)


@dataclass
class ImmotileTrack:
    """Represents a confirmed immotile sperm track."""

    tracking_id: int
    anchor_frame: int
    anchor_x: float
    anchor_y: float
    points: List[Tuple[int, float, float, int]]  # (frame, x, y, detection_index)
    spatial_std: float
    polynomial_coeffs: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def frame_range(self) -> Tuple[int, int]:
        """Return the frame range of this track."""
        if not self.points:
            return (self.anchor_frame, self.anchor_frame)
        frames = [p[0] for p in self.points]
        return (min(frames), max(frames))

    @property
    def duration(self) -> int:
        """Return track duration in frames."""
        start, end = self.frame_range
        return end - start + 1

    def get_detection_indices(self) -> Set[int]:
        """Return set of detection indices used by this track."""
        return {p[3] for p in self.points}


class ImmotileTracker:
    """
    Offline tracker for identifying immotile sperm tracks.

    This implements the two-stage pipeline where immotile tracks are
    identified and removed before motile tracking begins.
    """

    def __init__(self, config: ImmotileConfig):
        self.config = config
        self.immotile_tracks: List[ImmotileTrack] = []
        self.next_tracking_id = 0

        # Set up joblib processing (handles both parallel and sequential)
        if PARALLEL_AVAILABLE:
            if config.n_jobs == -1:
                self.n_jobs = mp.cpu_count()
            else:
                self.n_jobs = max(1, config.n_jobs)
            logger.info(
                f"ImmotileTracker initialized with joblib (n_jobs={self.n_jobs})"
            )
        else:
            self.n_jobs = 1
            logger.warning("Joblib not available, using sequential processing")

    def find_immotile_tracks(
        self, detections_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[ImmotileTrack]]:
        """
        Find immotile tracks using parallel processing with joblib.

        Args:
            detections_df: DataFrame with columns ['frame', 'x', 'y', ...]

        Returns:
            Tuple of (filtered_detections_df, immotile_tracks)
        """
        if detections_df.empty:
            return detections_df, []

        # Reset state
        self.immotile_tracks = []
        self.next_tracking_id = 0

        # Get anchors from first K frames
        anchors = self._get_anchors(detections_df)
        if anchors.empty:
            logger.info(
                "No anchors found in first %d frames", self.config.first_k_frames
            )
            return detections_df, []

        logger.info("Found %d potential immotile anchors", len(anchors))

        # Use joblib for processing (handles both parallel and sequential automatically)
        # When n_jobs=1, joblib processes sequentially without multiprocessing overhead
        logger.info(f"Processing anchors with joblib (n_jobs={self.n_jobs})")
        return self._process_anchors_with_joblib(detections_df, anchors)

    def _process_anchors_with_joblib(
        self, detections_df: pd.DataFrame, anchors: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[ImmotileTrack]]:
        """Process anchors using joblib with automatic batching and parallelization."""

        logger.info(
            "Searching for nearby detections within %.2f radius for each anchor (parallel)",
            self.config.search_radius,
        )

        # Use joblib's automatic batch size optimization
        batch_size = (
            self.config.batch_size if self.config.batch_size != "auto" else "auto"
        )
        logger.info(f"Using joblib batch_size={batch_size}")

        # Process individual anchors in parallel with joblib's automatic batching
        # This is more efficient than manual batching because joblib can optimize
        # the batch size dynamically based on processing time
        anchor_tracks = Parallel(
            n_jobs=self.n_jobs, backend="loky", verbose=0, batch_size=batch_size
        )(
            delayed(self._process_single_anchor)(anchor_idx, anchor, detections_df)
            for anchor_idx, anchor in anchors.iterrows()
        )

        # Filter out None results (anchors that didn't become tracks)
        all_tracks = [track for track in anchor_tracks if track is not None]

        # Handle conflicts and assign final tracking IDs
        self.immotile_tracks = []
        used_indices = set()

        for track in all_tracks:
            # Check for conflicts with already accepted tracks
            track_indices = track.get_detection_indices()
            if not track_indices.intersection(used_indices):
                # Assign final tracking ID
                track.tracking_id = len(self.immotile_tracks)
                self.immotile_tracks.append(track)
                used_indices.update(track_indices)

        # Remove immotile detections from dataframe
        if used_indices:
            filtered_df = detections_df.drop(index=list(used_indices))
            filtered_df = filtered_df.reset_index(drop=True)
            logger.info(
                "Parallel processing: Removed %d immotile detections, %d remaining for motile tracking",
                len(used_indices),
                len(filtered_df),
            )
        else:
            filtered_df = detections_df.copy()

        return filtered_df, self.immotile_tracks

    def _process_single_anchor(
        self, anchor_idx: int, anchor: pd.Series, detections_df: pd.DataFrame
    ) -> Optional[ImmotileTrack]:
        """
        Process a single anchor to determine if it represents an immotile track.
        This method is designed for parallel processing where each anchor is processed independently.
        """
        anchor_pos = np.array([anchor["x"], anchor["y"]])

        # Collect nearby points (excluding the anchor itself for now)
        candidate_points = []
        for idx, row in detections_df.iterrows():
            if idx == anchor_idx:  # Skip the anchor itself
                continue

            # Check spatial proximity
            pos = np.array([row["x"], row["y"]])
            distance = np.linalg.norm(pos - anchor_pos)

            if distance <= self.config.search_radius:
                candidate_points.append((row["frame"], row["x"], row["y"], idx))

        # Include anchor point itself
        candidate_points.append((anchor["frame"], anchor["x"], anchor["y"], anchor_idx))

        if len(candidate_points) < self.config.min_points:
            return None

        # Sort by frame
        candidate_points.sort(key=lambda x: x[0])

        # Robust polynomial fitting with outlier rejection
        refined_points = self._robust_polynomial_fit(candidate_points)

        if len(refined_points) < self.config.min_track_length:
            return None

        # Calculate spatial statistics
        spatial_std = self._calculate_spatial_std(refined_points)

        if spatial_std > self.config.max_std:
            return None

        # Fit final polynomial for the track
        poly_coeffs = self._fit_final_polynomial(refined_points)

        # Create immotile track with temporary ID (will be reassigned later)
        track = ImmotileTrack(
            tracking_id=-1,  # Temporary ID, will be reassigned during conflict resolution
            anchor_frame=anchor["frame"],
            anchor_x=anchor["x"],
            anchor_y=anchor["y"],
            points=refined_points,
            spatial_std=spatial_std,
            polynomial_coeffs=poly_coeffs,
        )

        return track

    def _get_anchors(self, detections_df: pd.DataFrame) -> pd.DataFrame:
        """Get potential immotile anchors from first K frames."""
        max_frame = detections_df["frame"].max()
        first_k = min(self.config.first_k_frames, max_frame + 1)
        return detections_df[detections_df["frame"] < first_k].copy()

    def _robust_polynomial_fit(
        self, points: List[Tuple[int, float, float, int]]
    ) -> List[Tuple[int, float, float, int]]:
        """
        Perform robust polynomial fitting with iterative outlier rejection.
        """
        if len(points) < self.config.min_points:
            return points

        # Extract arrays
        frames = np.array([p[0] for p in points])
        xs = np.array([p[1] for p in points])
        ys = np.array([p[2] for p in points])

        current_points = points.copy()
        current_frames = frames.copy()
        current_xs = xs.copy()
        current_ys = ys.copy()

        for _ in range(self.config.max_iterations):
            if len(current_frames) < self.config.min_points:
                break

            # Fit polynomials
            try:
                px = np.polyfit(current_frames, current_xs, self.config.poly_order)
                py = np.polyfit(current_frames, current_ys, self.config.poly_order)
            except np.linalg.LinAlgError:
                # Singular matrix, fall back to order 0
                px = np.polyfit(current_frames, current_xs, 0)
                py = np.polyfit(current_frames, current_ys, 0)

            # Calculate residuals
            xhat = np.polyval(px, current_frames)
            yhat = np.polyval(py, current_frames)
            residuals = np.hypot(current_xs - xhat, current_ys - yhat)

            # Remove outliers
            threshold = np.percentile(residuals, self.config.outlier_percentile)
            keep_mask = residuals <= threshold

            if np.all(keep_mask):
                break  # No outliers to remove

            # Update arrays
            current_frames = current_frames[keep_mask]
            current_xs = current_xs[keep_mask]
            current_ys = current_ys[keep_mask]
            current_points = [p for i, p in enumerate(current_points) if keep_mask[i]]

        return current_points

    def _calculate_spatial_std(
        self, points: List[Tuple[int, float, float, int]]
    ) -> float:
        """Calculate spatial standard deviation of points."""
        if len(points) < 2:
            return 0.0

        xs = np.array([p[1] for p in points])
        ys = np.array([p[2] for p in points])

        # Calculate centroid
        centroid_x = np.mean(xs)
        centroid_y = np.mean(ys)

        # Calculate distances from centroid
        distances = np.hypot(xs - centroid_x, ys - centroid_y)

        return float(np.std(distances))

    def _fit_final_polynomial(
        self, points: List[Tuple[int, float, float, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit final polynomial to confirmed track points."""
        frames = np.array([p[0] for p in points])
        xs = np.array([p[1] for p in points])
        ys = np.array([p[2] for p in points])

        try:
            px = np.polyfit(frames, xs, self.config.poly_order)
            py = np.polyfit(frames, ys, self.config.poly_order)
        except np.linalg.LinAlgError:
            # Fall back to order 0
            px = np.polyfit(frames, xs, 0)
            py = np.polyfit(frames, ys, 0)

        return px, py

    def get_immotile_tracks_dataframe(self) -> pd.DataFrame:
        """Convert immotile tracks to DataFrame format with confidence instead of std."""
        rows = []
        for track in self.immotile_tracks:
            # Convert spatial std to confidence (inverse relationship)
            # Lower std = higher confidence (more stable = more confident)
            confidence = max(
                0.0, min(1.0, 1.0 - (track.spatial_std / self.config.max_std))
            )

            for i, point in enumerate(track.points):
                rows.append(
                    {
                        "tracking_id": track.tracking_id,
                        "frame": point[0],
                        "x": point[1],
                        "y": point[2],
                        "accumulated_length": i + 1,
                        "track_length": len(track.points),
                        "confidence": confidence,
                        "track_type": "immotile",
                    }
                )

        return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_immotile_mining(
    detections_df: pd.DataFrame, config: Optional[ImmotileConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[ImmotileTrack]]:
    """
    Run the complete immotile mining pipeline.

    Args:
        detections_df: Input detections DataFrame
        config: ImmotileConfig instance (uses defaults if None)

    Returns:
        Tuple of (filtered_detections_df, immotile_tracks_df, immotile_tracks)
    """
    if config is None:
        config = ImmotileConfig()

    tracker = ImmotileTracker(config)

    logger.info(
        "Starting immotile mining with %d initial detections", len(detections_df)
    )

    # Find immotile tracks
    filtered_df, immotile_tracks = tracker.find_immotile_tracks(detections_df)

    # Convert tracks to DataFrame
    immotile_df = tracker.get_immotile_tracks_dataframe()

    logger.info(
        "Immotile mining complete: found %d immotile tracks, %d detections remain for motile tracking",
        len(immotile_tracks),
        len(filtered_df),
    )

    return filtered_df, immotile_df, immotile_tracks
