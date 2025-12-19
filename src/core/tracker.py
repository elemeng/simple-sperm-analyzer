#!/usr/bin/env python3
"""
Integration module for combining immotile and motile sperm tracking.

This module provides the interface to run the two-stage pipeline:
1. Immotile mining (offline, deterministic) using module `src.core.immotile_tracker`
2. Motile tracking (existing DirectionFirstTracker) using module `src.core.motile_tracker`
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass

from src.core.immotile_tracker import ImmotileTracker, ImmotileConfig, ImmotileTrack

from src.core.motile_tracker import (
    TrackerConfig,
    TrajectoryPostConfig,
    Detection,
    run_tracking_with_configs,
)

logger = logging.getLogger(__name__)


@dataclass
class TwoStagePipelineConfig:
    """Configuration for the complete two-stage tracking pipeline."""

    immotile_config: ImmotileConfig
    motile_tracker_config: TrackerConfig
    motile_post_config: TrajectoryPostConfig
    img_shape: Tuple[int, int]

    # Output options
    save_immotile_tracks: bool = True
    merge_output: bool = True  # Combine immotile and motile tracks in final output


def run_two_stage_tracking_with_config(
    detections_df: pd.DataFrame,
    pipeline_config: TwoStagePipelineConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the two-stage tracking pipeline with configuration object.
    
    Args:
        detections_df: Input detections DataFrame
        pipeline_config: Complete pipeline configuration
        
    Returns:
        Tuple of (combined_tracks_df, pipeline_summary)
    """
    logger.info("Starting two-stage tracking pipeline")
    logger.info(
        "Stage 1: Immotile mining with first %d frames, radius %.1f px",
        pipeline_config.immotile_config.first_k_frames,
        pipeline_config.immotile_config.search_radius,
    )
    
    # Stage 1: Immotile mining (now with integrated parallel processing)
    immotile_tracker = ImmotileTracker(pipeline_config.immotile_config)
    filtered_detections, immotile_tracks = immotile_tracker.find_immotile_tracks(
        detections_df
    )
    
    # Convert tracks to DataFrame
    immotile_df = immotile_tracker.get_immotile_tracks_dataframe()
    
    logger.info(
        "Stage 1 complete: Found %d immotile tracks with %d detections",
        len(immotile_tracks),
        len(immotile_df),
    )
    logger.info(
        "Stage 2: Motile tracking with %d remaining detections",
        len(filtered_detections),
    )

    # Stage 2: Motile tracking on filtered detections
    if not filtered_detections.empty:
        motile_tracks_df = run_tracking_with_configs(
            filtered_detections, 
            pipeline_config.total_frames if hasattr(pipeline_config, 'total_frames') else 0,
            pipeline_config.img_shape, 
            pipeline_config.motile_tracker_config, 
            pipeline_config.motile_post_config
        )
    else:
        motile_tracks_df = pd.DataFrame()

    # Get motile track count (handle both track_id and tracking_id columns)
    motile_track_count = 0
    if not motile_tracks_df.empty:
        if "tracking_id" in motile_tracks_df.columns:
            motile_track_count = motile_tracks_df["tracking_id"].nunique()
        elif "track_id" in motile_tracks_df.columns:
            motile_track_count = motile_tracks_df["track_id"].nunique()

    logger.info(
        "Stage 2 complete: Found %d motile tracks with %d detections",
        motile_track_count,
        len(motile_tracks_df),
    )

    # Combine results
    if immotile_df.empty and motile_tracks_df.empty:
        combined_df = pd.DataFrame()
    elif immotile_df.empty:
        combined_df = motile_tracks_df
    elif motile_tracks_df.empty:
        combined_df = immotile_df
    else:
        # Handle column naming - immotile uses tracking_id, motile might use track_id
        # First, ensure motile tracks have tracking_id column (preserve original track_id as tracking_id)
        if (
            "track_id" in motile_tracks_df.columns
            and "tracking_id" not in motile_tracks_df.columns
        ):
            motile_tracks_df["tracking_id"] = motile_tracks_df["track_id"]

        # Add track type column
        immotile_df["track_type"] = "immotile"
        motile_tracks_df["track_type"] = "motile"

        # Ensure unique tracking_id values across immotile and motile tracks
        if not immotile_df.empty and not motile_tracks_df.empty:
            # Get max tracking_id from immotile tracks
            max_immotile_id = (
                immotile_df["tracking_id"].max()
                if "tracking_id" in immotile_df.columns
                else -1
            )

            # Offset motile tracking_ids to ensure no overlap
            if "tracking_id" in motile_tracks_df.columns and max_immotile_id >= 0:
                motile_tracks_df["tracking_id"] = (
                    motile_tracks_df["tracking_id"] + max_immotile_id + 1
                )
                logger.info(
                    "Offset motile tracking IDs by %d to avoid overlap with immotile tracks",
                    max_immotile_id + 1,
                )

        # Combine dataframes
        combined_df = pd.concat([immotile_df, motile_tracks_df], ignore_index=True)

    # Create pipeline summary (handle both column naming conventions)
    motile_track_count = 0
    if not motile_tracks_df.empty:
        if "tracking_id" in motile_tracks_df.columns:
            motile_track_count = motile_tracks_df["tracking_id"].nunique()
        elif "track_id" in motile_tracks_df.columns:
            motile_track_count = motile_tracks_df["track_id"].nunique()

    summary = {
        "total_input_detections": len(detections_df),
        "immotile_tracks_found": len(immotile_tracks),
        "immotile_detections_removed": len(immotile_df),
        "motile_tracks_found": motile_track_count,
        "motile_detections": len(motile_tracks_df),
        "total_tracks": len(immotile_tracks) + motile_track_count,
        "detection_utilization": (len(immotile_df) + len(motile_tracks_df))
        / len(detections_df)
        if len(detections_df) > 0
        else 0.0,
    }

    logger.info(
        "Two-stage pipeline complete: %d total tracks, %.1f%% detection utilization",
        summary["total_tracks"],
        summary["detection_utilization"] * 100,
    )

    # Create user-friendly track IDs (starting from 1) while preserving tracking_id as internal ID
    if not combined_df.empty and "tracking_id" in combined_df.columns:
        # Use tracking_id as internal ID (preserve original values)
        # Create user-friendly track_id with consecutive numbering
        unique_tracks = sorted(combined_df["tracking_id"].unique())
        track_id_mapping = {
            old_id: new_id for new_id, old_id in enumerate(unique_tracks, start=1)
        }
        combined_df["track_id"] = combined_df["tracking_id"].map(track_id_mapping)

        logger.info(
            "Created user-friendly track IDs: %d tracks (IDs 1-%d) from internal tracking_id range %d-%d",
            len(unique_tracks),
            len(unique_tracks),
            min(unique_tracks),
            max(unique_tracks),
        )
    elif not combined_df.empty:
        # Fallback: if no tracking_id, use existing track_id or create simple numbering
        if "track_id" in combined_df.columns:
            logger.info(
                "Using existing track_id as user-friendly IDs (no internal tracking_id available)"
            )
        else:
            # Create simple track IDs if none exist
            combined_df["track_id"] = range(1, len(combined_df) + 1)
            logger.info(
                "Created simple track IDs for output (no internal tracking_id available)"
            )

    return combined_df, summary


def run_two_stage_tracking(
    detections_df: pd.DataFrame,
    total_frames: int,
    img_shape: Tuple[int, int],
    immotile_config: Optional[ImmotileConfig] = None,
    tracker_config: Optional[TrackerConfig] = None,
    post_config: Optional[TrajectoryPostConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the complete two-stage tracking pipeline.

    Args:
        detections_df: Input detections DataFrame
        total_frames: Total number of frames in video
        img_shape: Image dimensions (height, width)
        immotile_config: Configuration for immotile mining
        tracker_config: Configuration for motile tracking
        post_config: Configuration for post-processing

    Returns:
        Tuple of (combined_tracks_df, pipeline_summary)
    """

    # Use default configurations if not provided
    if immotile_config is None:
        immotile_config = ImmotileConfig()

    if tracker_config is None:
        tracker_config = TrackerConfig()

    if post_config is None:
        post_config = TrajectoryPostConfig()

    logger.info("Starting two-stage tracking pipeline")
    logger.info(
        "Stage 1: Immotile mining with first %d frames, radius %.1f px",
        immotile_config.first_k_frames,
        immotile_config.search_radius,
    )

    # Stage 1: Immotile mining
    immotile_tracker = ImmotileTracker(immotile_config)
    filtered_detections, immotile_tracks = immotile_tracker.find_immotile_tracks(
        detections_df
    )

    # Convert tracks to DataFrame
    immotile_df = immotile_tracker.get_immotile_tracks_dataframe()

    logger.info(
        "Stage 1 complete: Found %d immotile tracks with %d detections",
        len(immotile_tracks),
        len(immotile_df),
    )
    logger.info(
        "Stage 2: Motile tracking with %d remaining detections",
        len(filtered_detections),
    )

    # Stage 2: Motile tracking on filtered detections
    if not filtered_detections.empty:
        motile_tracks_df = run_tracking_with_configs(
            filtered_detections, total_frames, img_shape, tracker_config, post_config
        )
    else:
        motile_tracks_df = pd.DataFrame()

    # Get motile track count (handle both track_id and tracking_id columns)
    motile_track_count = 0
    if not motile_tracks_df.empty:
        if "tracking_id" in motile_tracks_df.columns:
            motile_track_count = motile_tracks_df["tracking_id"].nunique()
        elif "track_id" in motile_tracks_df.columns:
            motile_track_count = motile_tracks_df["track_id"].nunique()

    logger.info(
        "Stage 2 complete: Found %d motile tracks with %d detections",
        motile_track_count,
        len(motile_tracks_df),
    )

    # Combine results
    if immotile_df.empty and motile_tracks_df.empty:
        combined_df = pd.DataFrame()
    elif immotile_df.empty:
        combined_df = motile_tracks_df
    elif motile_tracks_df.empty:
        combined_df = immotile_df
    else:
        # Handle column naming - immotile uses tracking_id, motile might use track_id
        # First, ensure motile tracks have tracking_id column (preserve original track_id as tracking_id)
        if (
            "track_id" in motile_tracks_df.columns
            and "tracking_id" not in motile_tracks_df.columns
        ):
            motile_tracks_df["tracking_id"] = motile_tracks_df["track_id"]

        # Add track type column
        immotile_df["track_type"] = "immotile"
        motile_tracks_df["track_type"] = "motile"

        # Ensure unique tracking_id values across immotile and motile tracks
        if not immotile_df.empty and not motile_tracks_df.empty:
            # Get max tracking_id from immotile tracks
            max_immotile_id = (
                immotile_df["tracking_id"].max()
                if "tracking_id" in immotile_df.columns
                else -1
            )

            # Offset motile tracking_ids to ensure no overlap
            if "tracking_id" in motile_tracks_df.columns and max_immotile_id >= 0:
                motile_tracks_df["tracking_id"] = (
                    motile_tracks_df["tracking_id"] + max_immotile_id + 1
                )
                logger.info(
                    "Offset motile tracking IDs by %d to avoid overlap with immotile tracks",
                    max_immotile_id + 1,
                )

        # Combine dataframes
        combined_df = pd.concat([immotile_df, motile_tracks_df], ignore_index=True)

    # Create pipeline summary (handle both column naming conventions)
    motile_track_count = 0
    if not motile_tracks_df.empty:
        if "tracking_id" in motile_tracks_df.columns:
            motile_track_count = motile_tracks_df["tracking_id"].nunique()
        elif "track_id" in motile_tracks_df.columns:
            motile_track_count = motile_tracks_df["track_id"].nunique()

    summary = {
        "total_input_detections": len(detections_df),
        "immotile_tracks_found": len(immotile_tracks),
        "immotile_detections_removed": len(immotile_df),
        "motile_tracks_found": motile_track_count,
        "motile_detections": len(motile_tracks_df),
        "total_tracks": len(immotile_tracks) + motile_track_count,
        "detection_utilization": (len(immotile_df) + len(motile_tracks_df))
        / len(detections_df)
        if len(detections_df) > 0
        else 0.0,
    }

    logger.info(
        "Two-stage pipeline complete: %d total tracks, %.1f%% detection utilization",
        summary["total_tracks"],
        summary["detection_utilization"] * 100,
    )

    # Create user-friendly track IDs (starting from 1) while preserving tracking_id as internal ID
    if not combined_df.empty and "tracking_id" in combined_df.columns:
        # Use tracking_id as internal ID (preserve original values)
        # Create user-friendly track_id with consecutive numbering
        unique_tracks = sorted(combined_df["tracking_id"].unique())
        track_id_mapping = {
            old_id: new_id for new_id, old_id in enumerate(unique_tracks, start=1)
        }
        combined_df["track_id"] = combined_df["tracking_id"].map(track_id_mapping)

        logger.info(
            "Created user-friendly track IDs: %d tracks (IDs 1-%d) from internal tracking_id range %d-%d",
            len(unique_tracks),
            len(unique_tracks),
            min(unique_tracks),
            max(unique_tracks),
        )
    elif not combined_df.empty:
        # Fallback: if no tracking_id, use existing track_id or create simple numbering
        if "track_id" in combined_df.columns:
            logger.info(
                "Using existing track_id as user-friendly IDs (no internal tracking_id available)"
            )
        else:
            # Create simple track IDs if none exist
            combined_df["track_id"] = range(1, len(combined_df) + 1)
            logger.info(
                "Created simple track IDs for output (no internal tracking_id available)"
            )

    return combined_df, summary


def create_immotile_detection_objects(immotile_df: pd.DataFrame) -> List[Detection]:
    """Convert immotile tracks to Detection objects for visualization."""
    detections = []

    if immotile_df.empty:
        return detections

    for _, row in immotile_df.iterrows():
        detection = Detection(
            frame=int(row["frame"]),
            x=float(row["x"]),
            y=float(row["y"]),
            area=0.0,  # Not available for immotile tracks
            curvature=None,
        )
        detections.append(detection)

    return detections


def merge_track_results(
    immotile_tracks: List[ImmotileTrack],
    motile_tracks_df: pd.DataFrame,
    immotile_track_prefix: str = "IMM_",
) -> pd.DataFrame:
    """
    Merge immotile and motile track results into a unified format.

    Args:
        immotile_tracks: List of confirmed immotile tracks
        motile_tracks_df: DataFrame of motile tracking results
        immotile_track_prefix: Prefix for immotile track IDs

    Returns:
        Unified DataFrame with all tracks (minimal columns for final output)
    """

    # Convert immotile tracks to DataFrame with minimal columns
    immotile_rows = []
    for track in immotile_tracks:
        # Convert spatial std to confidence (inverse relationship)
        # Use a reasonable default max_std if not available (1.5 pixels)
        max_std = 1.5
        confidence = max(0.0, min(1.0, 1.0 - (track.spatial_std / max_std)))

        for i, point in enumerate(track.points):
            immotile_rows.append(
                {
                    "tracking_id": f"{immotile_track_prefix}{track.tracking_id}",
                    "frame": point[0],
                    "x": point[1],
                    "y": point[2],
                    "accumulated_length": i + 1,
                    "track_length": len(track.points),
                    "confidence": confidence,
                    "track_type": "immotile",
                }
            )

    immotile_df = pd.DataFrame(immotile_rows) if immotile_rows else pd.DataFrame()

    # Add track type to motile tracks (keep existing format)
    if not motile_tracks_df.empty:
        motile_tracks_df["track_type"] = "motile"

    # Combine results with consistent column structure
    if immotile_df.empty and motile_tracks_df.empty:
        return pd.DataFrame()
    elif immotile_df.empty:
        return motile_tracks_df
    elif motile_tracks_df.empty:
        return immotile_df
    else:
        # Ensure consistent column order and types (minimal columns only)
        common_columns = [
            "tracking_id",
            "frame",
            "x",
            "y",
            "accumulated_length",
            "track_length",
            "confidence",
            "track_type",
        ]

        for col in common_columns:
            if col not in immotile_df.columns:
                immotile_df[col] = np.nan
            if col not in motile_tracks_df.columns:
                motile_tracks_df[col] = np.nan

        # Combine and reorder columns (only essential columns for final output)
        combined_df = pd.concat(
            [immotile_df[common_columns], motile_tracks_df[common_columns]],
            ignore_index=True,
        )

        return combined_df


def validate_immotile_config(
    config: ImmotileConfig, img_shape: Tuple[int, int]
) -> List[str]:
    """
    Validate immotile configuration parameters.

    Args:
        config: ImmotileConfig to validate
        img_shape: Image dimensions (height, width)

    Returns:
        List of validation warnings/errors
    """

    issues = []
    height, width = img_shape

    # Check search radius
    if config.search_radius > min(width, height) / 4:
        issues.append(
            f"Search radius {config.search_radius} may be too large for image size {img_shape}"
        )

    # Check minimum points
    if config.min_points < 3:
        issues.append(
            f"Minimum points {config.min_points} is too low for reliable fitting"
        )

    # Check polynomial order
    if config.poly_order > 2:
        issues.append(
            f"Polynomial order {config.poly_order} may overfit immotile tracks"
        )

    # Check first_k_frames
    if config.first_k_frames < 1:
        issues.append("first_k_frames must be at least 1")

    # Check outlier percentile
    if config.outlier_percentile < 50 or config.outlier_percentile > 99:
        issues.append(
            f"Outlier percentile {config.outlier_percentile} should be between 50-99"
        )

    # Check spatial tolerance
    if config.max_std > config.search_radius:
        issues.append(
            f"Max spatial std {config.max_std} should not exceed search radius {config.search_radius}"
        )

    return issues
