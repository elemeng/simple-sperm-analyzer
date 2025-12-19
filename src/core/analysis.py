#!/usr/bin/env python3
"""
Sperm Motility Analysis Script (OpenCASA-Based)
================================================

Calculates kinematic parameters from sperm tracking data based on:
https://pmc.ncbi.nlm.nih.gov/articles/PMC6355034/table/pcbi.1006691.t002
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from .common import calculate_distances


def calculate_vsl(track_df: pd.DataFrame, pixel_size_um: float, fps: float) -> float:
    """Straight-line velocity (VSL)"""
    if len(track_df) < 2:
        return np.nan
    first_point = track_df.iloc[0][["x", "y"]].values
    last_point = track_df.iloc[-1][["x", "y"]].values
    distance_um = np.linalg.norm(last_point - first_point) * pixel_size_um
    time_seconds = (len(track_df) - 1) / fps
    return distance_um / time_seconds if time_seconds > 0 else np.nan


def calculate_vcl(track_df: pd.DataFrame, pixel_size_um: float, fps: float) -> float:
    """Curvilinear velocity (VCL)"""
    if len(track_df) < 2:
        return np.nan
    distances_px = calculate_distances(track_df["x"].values, track_df["y"].values)
    total_distance_um = np.sum(distances_px) * pixel_size_um
    time_seconds = (len(track_df) - 1) / fps
    return total_distance_um / time_seconds if time_seconds > 0 else np.nan


def calculate_vap(
    track_df: pd.DataFrame, pixel_size_um: float, fps: float, window_size: int = 5
) -> float:
    """Average-path velocity (VAP)"""
    if len(track_df) < window_size + 1:
        return np.nan
    x, y = track_df["x"].values, track_df["y"].values
    window = np.ones(window_size) / window_size
    q_x = np.convolve(x, window, mode="valid")
    q_y = np.convolve(y, window, mode="valid")
    distances_px = calculate_distances(q_x, q_y)
    total_distance_um = np.sum(distances_px) * pixel_size_um
    time_seconds = (len(track_df) - window_size) / fps
    return total_distance_um / time_seconds if time_seconds > 0 else np.nan


def calculate_alh(
    track_df: pd.DataFrame, pixel_size_um: float, window_size: int = 5
) -> Tuple[float, float]:
    """
    Calculate ALHmean and ALHmax.
    Simplified implementation based on OpenCASA definition using
    2× max deviation from average path.
    """
    if len(track_df) < window_size + 2:
        return np.nan, np.nan

    x, y = track_df["x"].values, track_df["y"].values
    window = np.ones(window_size) / window_size
    q_x = np.convolve(x, window, mode="valid")
    q_y = np.convolve(y, window, mode="valid")

    # Calculate midpoints of average path segments
    qm_x = (q_x[:-1] + q_x[1:]) / 2
    qm_y = (q_y[:-1] + q_y[1:]) / 2

    # Calculate distances from midpoints to original trajectory segments
    alh_values = []
    for i in range(len(qm_x)):
        start_idx = i
        end_idx = min(i + window_size, len(x) - 1)
        segment_x = x[start_idx:end_idx]
        segment_y = y[start_idx:end_idx]

        # Distance from midpoint to each point in segment
        distances = np.sqrt((segment_x - qm_x[i]) ** 2 + (segment_y - qm_y[i]) ** 2)
        alh_values.append(np.max(distances) * 2)  # ALH = 2 × max distance

    if not alh_values:
        return np.nan, np.nan

    alh_mean = np.mean(alh_values) * pixel_size_um
    alh_max = np.max(alh_values) * pixel_size_um

    return alh_mean, alh_max


def calculate_dance(vcl: float, alh_mean: float) -> float:
    """Calculate DANCE parameter (VCL × ALHmean)"""
    if np.isnan(vcl) or np.isnan(alh_mean):
        return np.nan
    return vcl * alh_mean


def calculate_mad(track_df: pd.DataFrame) -> float:
    """Calculate Mean Angular Displacement (MAD)"""
    if len(track_df) < 3:
        return np.nan

    x = track_df["x"].values
    y = track_df["y"].values

    # Calculate angles between consecutive segments
    angles = []
    for i in range(1, len(x) - 1):
        # Vector from point i-1 to i
        v1 = np.array([x[i] - x[i - 1], y[i] - y[i - 1]])
        # Vector from point i to i+1
        v2 = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])

        # Calculate angle between vectors
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            # Clamp to [-1, 1] to avoid numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(np.degrees(angle))

    return np.mean(angles) if angles else np.nan


def calculate_fractal_dimension(track_df: pd.DataFrame, pixel_size_um: float) -> float:
    """Calculate Fractal Dimension (FD)"""
    if len(track_df) < 3:
        return np.nan

    x = track_df["x"].values
    y = track_df["y"].values

    # Calculate curvilinear length (L)
    distances_px = calculate_distances(x, y)
    L = np.sum(distances_px) * pixel_size_um

    # Calculate straight-line distance (d)
    first_point = np.array([x[0], y[0]])
    last_point = np.array([x[-1], y[-1]])
    d = np.linalg.norm(last_point - first_point) * pixel_size_um

    if d <= 0 or L <= 0:
        return np.nan

    # Calculate number of curve intervals (N-1 segments)
    num_intervals = len(track_df) - 1

    # Fractal dimension: FD = log(num_intervals) / log(L/d)
    try:
        ratio = L / d
        if ratio <= 1.0:  # Straight line case
            return 1.0
        fd = np.log(num_intervals) / np.log(ratio)
        # FD should be between 1 (straight) and 2 (space-filling)
        return np.clip(fd, 1.0, 2.0)
    except (ValueError, ZeroDivisionError, OverflowError):
        import traceback

        traceback.print_exc()
        return np.nan


def calculate_bcf(
    track_df: pd.DataFrame, pixel_size_um: float, fps: float, window_size: int = 5
) -> float:
    """
    Calculate Beat-Cross Frequency (BCF) using OpenCASA standard method.

    This is a wrapper for calculate_motile_bcf for API compatibility.
    """
    return calculate_motile_bcf(track_df, pixel_size_um, fps, window_size)


def calculate_motile_bcf(
    track_df: pd.DataFrame, pixel_size_um: float, fps: float, window_size: int = 5
) -> float:
    """Calculate Beat-Cross Frequency (BCF) using average path crossing method."""
    if len(track_df) < window_size + 2:
        return np.nan

    x, y = track_df["x"].values, track_df["y"].values

    # Create average path by smoothing the trajectory
    window = np.ones(window_size) / window_size
    q_x = np.convolve(x, window, mode="valid")
    q_y = np.convolve(y, window, mode="valid")

    # Calculate the direction vector of the average path
    avg_direction = np.array([q_x[-1] - q_x[0], q_y[-1] - q_y[0]])
    if np.linalg.norm(avg_direction) == 0:
        return 0.0

    # Normalize the average direction vector
    avg_direction = avg_direction / np.linalg.norm(avg_direction)

    # Calculate perpendicular vector to average direction
    perp_direction = np.array([-avg_direction[1], avg_direction[0]])

    # Vectorized calculation of signed distances from trajectory points to average path
    p0 = np.array([q_x[0], q_y[0]])
    points = np.stack([x, y], axis=1)
    signed_distances = np.dot(points - p0, perp_direction)

    # Count zero crossings
    crossing_count = 0
    start_idx = window_size // 2
    end_idx = len(x) - window_size // 2

    for i in range(start_idx, end_idx - 1):
        if signed_distances[i] * signed_distances[i + 1] < 0:
            crossing_count += 1

    effective_frames = end_idx - start_idx
    if effective_frames <= 0:
        return np.nan

    bcf = (crossing_count * fps) / effective_frames
    return bcf


def sperm_classes(
    vcl: float,
    vsl: float,
    vap: float,
    motility_vcl_threshold: float = 20.0,
    motility_vsl_threshold: float = 4.0,
    motility_vap_threshold: float = 4.0,
) -> Dict[str, Any]:
    """
    Classify motility and progression:
    - PM (Motility): VCL >= motility_vcl_threshold
    - PROG (Progressive): PM AND (STR >= progressive_str_threshold) AND (VAP >= progressive_vap_threshold)

    Reference: OpenCASA sperm parameters table
    """
    # Motility based on VCL threshold only (OpenCASA standard)
    is_motile = vcl >= motility_vcl_threshold
    is_motile_vsl_vap = (vsl >= motility_vsl_threshold) and (
        vap >= motility_vap_threshold
    )
    # Progressive motility: motile AND meets STR and VAP criteria

    return {
        "pm_motile_vcl": is_motile,  # Motile sperm (by VCL method)
        "pm_motile_vsl_vap": is_motile_vsl_vap,  # Motile sperm (by VSL & VAP method)
    }


def analyze_tracks(
    tracking_df: pd.DataFrame,
    pixel_size_um: float = 0.5,
    fps: float = 60,
    motility_vcl_threshold: float = 20.0,
    motility_vsl_threshold: float = 4.0,
    motility_vap_threshold: float = 4.0,
    vap_window: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze tracks with full parameter calculation.

    Args:
        tracking_df: DataFrame with tracking results
        pixel_size_um: Micrometers per pixel
        fps: Frames per second
        motility_vcl_threshold: Minimum VCL for motility classification
        motility_vsl_vap_threshold: Minimum VSL & VAP for motility classification
        progressive_str_threshold: Minimum STR for progressive classification
        progressive_vap_threshold: Minimum VAP for progressive classification
        vap_window: Window size for VAP calculation

    Returns:
        Tuple of (results DataFrame, summary dictionary)
    """

    df = tracking_df.copy()

    if len(df) == 0:
        raise ValueError("No tracks found.")

    results = []

    for track_id, track_df in df.groupby("track_id"):
        track_df = track_df.sort_values("frame").reset_index(drop=True)

        if len(track_df) < vap_window + 2:
            continue

        # Calculate velocities
        vsl = calculate_vsl(track_df, pixel_size_um, fps)
        vcl = calculate_vcl(track_df, pixel_size_um, fps)
        vap = calculate_vap(track_df, pixel_size_um, fps, vap_window)

        if np.isnan([vsl, vcl, vap]).any():
            continue

        # Classify motility
        motility = sperm_classes(
            vcl,
            vsl,
            vap,
            motility_vcl_threshold,
            motility_vsl_threshold,
            motility_vap_threshold,
        )

        results.append(
            {
                "track_id": int(track_id),
                "track_length": len(track_df),
                "VSL_um_s": round(vsl, 3),
                "VCL_um_s": round(vcl, 3),
                "VAP_um_s": round(vap, 3),
                **motility,
            }
        )

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return pd.DataFrame(), {}

    # Calculate summary statistics
    total = len(results_df)
    pm_motile_vcl = results_df["pm_motile_vcl"].sum()
    pm_motile_vsl_vap = results_df["pm_motile_vsl_vap"].sum()

    summary = {
        "total_tracks": total,
        "pm_motile_vcl_tracks": pm_motile_vcl,  # Count of tracks motile by VCL method
        "pm_motile_vcl_percent": round(
            pm_motile_vcl / total * 100, 2
        ),  # Motile by VCL method
        "pm_motile_vsl_vap_tracks": pm_motile_vsl_vap,  # Count of tracks motile by VSL & VAP method
        "pm_motile_vsl_vap_percent": round(
            pm_motile_vsl_vap / total * 100, 2
        ),  # Motile by VSL & VAP method
        "immotile_percent": round(
            (total - pm_motile_vcl - pm_motile_vsl_vap) / total * 100, 2
        ),  # Non-motile = 100% - VCL motile - VSL & VAP motile
        "vsl": round(results_df["VSL_um_s"].mean(), 2),
        "vcl": round(results_df["VCL_um_s"].mean(), 2),
        "vap": round(results_df["VAP_um_s"].mean(), 2),
    }

    return results_df, summary


def generate_report(
    summary: Dict[str, Any], results_df: pd.DataFrame, params: Dict[str, float]
) -> str:
    """Generate formatted analysis report with all parameters."""
    if len(results_df) == 0:
        return "⚠️  No valid tracks found for analysis."

    def pb(pct: float, width: int = 30) -> str:
        """Generate proportional progress bar."""
        filled = int(width * pct / 100)
        return "█" * filled + "░" * (width - filled)

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           SPERM MOTILITY ANALYSIS REPORT                     ║
║           Based on Standard Motility Parameters              ║
╚══════════════════════════════════════════════════════════════╝

📊 ANALYSIS PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Pixel size:          {params["pixel_size"]} μm/pixel
  Frame rate:          {params["fps"]} fps
  VAP window (w):      {params["vap_window"]} frames
  Motility threshold (VCL): VCL ≥ {params["motility_vcl_threshold"]} μm/s
  Motility threshold (VSL & VAP): VSL ≥ {params["motility_vsl_threshold"]} μm/s & VAP ≥ {params["motility_vap_threshold"]} μm/s

📈 MOTILITY CLASSIFICATION (Standard Methods)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total tracks:             {summary["total_tracks"]}
  
 Motile sperm (VCL≥{params["motility_vcl_threshold"]}): {summary["pm_motile_vcl_percent"]:>6.2f}% {pb(summary["pm_motile_vcl_percent"])} ({summary["pm_motile_vcl_tracks"]} tracks)
 Motile sperm (VSL≥{params["motility_vsl_threshold"]} & VAP≥{params["motility_vap_threshold"]}): {summary["pm_motile_vsl_vap_percent"]:>6.2f}% {pb(summary["pm_motile_vsl_vap_percent"])} ({summary["pm_motile_vsl_vap_tracks"]} tracks)

⚡ VELOCITY PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VSL (Straight-line):    {summary["vsl"]:>6.2f} μm/s
  VCL (Curvilinear):      {summary["vcl"]:>6.2f} μm/s
  VAP (Average path):     {summary["vap"]:>6.2f} μm/s

"""

    return report
