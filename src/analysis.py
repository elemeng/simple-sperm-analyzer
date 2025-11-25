#!/usr/bin/env python3
"""
Sperm Motility Analysis Script (OpenCASA-Based)
================================================

Calculates kinematic parameters from sperm tracking data based on:
https://pmc.ncbi.nlm.nih.gov/articles/PMC6355034/table/pcbi.1006691.t002

Usage:
    python sperm_motility_analysis.py tracking.csv --pixel-size 0.5 --fps 60
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, Dict, Any


__version__ = "1.0.0"


def calculate_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distances between consecutive points."""
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx**2 + dy**2)


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
    Calculate ALHmean and ALHmax
    Simplified implementation based on OpenCASA definition
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
    # This is a simplified approximation
    alh_values = []
    for i in range(len(qm_x)):
        # Find the closest point in the original trajectory
        start_idx = i
        end_idx = min(i + window_size, len(x) - 1)
        segment_x = x[start_idx:end_idx]
        segment_y = y[start_idx:end_idx]

        # Calculate distance from midpoint to each point in segment
        distances = np.sqrt((segment_x - qm_x[i]) ** 2 + (segment_y - qm_y[i]) ** 2)
        alh_values.append(np.max(distances) * 2)  # 2 × maximum distance

    if not alh_values:
        return np.nan, np.nan

    alh_mean = np.mean(alh_values) * pixel_size_um
    alh_max = np.max(alh_values) * pixel_size_um

    return alh_mean, alh_max


def segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    """
    Check if two line segments intersect using the cross product method.
    Segments: p1-p2 and q1-q2
    """
    def cross_product(a: np.ndarray, b: np.ndarray) -> float:
        return a[0] * b[1] - a[1] * b[0]
    
    # Vector from p1 to p2
    d1 = p2 - p1
    # Vector from q1 to q2  
    d2 = q2 - q1
    # Vector from p1 to q1
    d3 = q1 - p1
    # Vector from p1 to q2
    d4 = q2 - p1
    
    # Calculate cross products
    cp1 = cross_product(d1, d3)
    cp2 = cross_product(d1, d4)
    cp3 = cross_product(d2, -d3)
    cp4 = cross_product(d2, (p1 - q1))
    
    # Check if segments straddle each other
    if cp1 * cp2 < 0 and cp3 * cp4 < 0:
        return True
    
    # Check collinear cases (segments on same line)
    if cp1 == 0 and cp2 == 0:  # Segments are collinear
        # Check if segments overlap
        def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
            return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and 
                   min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))
        
        return (on_segment(p1, p2, q1) or on_segment(p1, p2, q2) or 
                on_segment(q1, q2, p1) or on_segment(q1, q2, p2))
    
    return False


def calculate_head_vibration_frequency(
    track_df: pd.DataFrame, fps: float, position_threshold: float = 2.0
) -> float:
    """
    Calculate head vibration frequency for tethered sperm.
    
    For sperm with minimal position changes (VSL ≈ 0, VAP ≈ 0), this analyzes
    the rapid oscillations of the head position to determine beating frequency.
    
    Parameters:
    - track_df: DataFrame with x, y coordinates
    - fps: frames per second
    - position_threshold: maximum position change (pixels) to consider as vibration
    
    Returns vibration frequency in Hz
    """
    if len(track_df) < 10:  # Need sufficient data for frequency analysis
        return np.nan
    
    x = track_df["x"].values
    y = track_df["y"].values
    
    # Calculate total displacement from first point
    x_centered = x - x[0]
    y_centered = y - y[0]
    distances_from_origin = np.sqrt(x_centered**2 + y_centered**2)
    
    # Check if movement is within vibration threshold (tethered condition)
    max_displacement = np.max(distances_from_origin)
    if max_displacement > position_threshold:
        return np.nan  # Not tethered - too much movement
    
    # Calculate velocity components for vibration analysis
    dx = np.diff(x)
    dy = np.diff(y)
    velocity_magnitude = np.sqrt(dx**2 + dy**2)
    
    # Apply high-pass filter to remove slow drift and isolate rapid vibrations
    # Use simple difference filter to emphasize rapid changes
    velocity_diff = np.diff(velocity_magnitude)
    
    # Count zero crossings in velocity (indicating direction changes/beats)
    zero_crossings = 0
    for i in range(len(velocity_diff) - 1):
        if velocity_diff[i] * velocity_diff[i+1] < 0:
            zero_crossings += 1
    
    # Alternative: count peaks in velocity magnitude
    peaks = 0
    for i in range(1, len(velocity_magnitude) - 1):
        if velocity_magnitude[i] > velocity_magnitude[i-1] and velocity_magnitude[i] > velocity_magnitude[i+1]:
            peaks += 1
    
    # Use the more reliable method (peaks are usually more stable for vibration analysis)
    vibration_count = max(zero_crossings, peaks)
    
    # Calculate frequency: vibrations per second
    # Account for the fact that each full cycle has 2 zero crossings but 1 peak
    if vibration_count == zero_crossings:
        cycles = vibration_count / 2  # Each full cycle has 2 zero crossings
    else:
        cycles = vibration_count  # Each peak is one beat
    
    duration_seconds = (len(track_df) - 1) / fps
    if duration_seconds <= 0:
        return np.nan
    
    frequency = cycles / duration_seconds
    
    # Sanity check: biological beating frequency should be in reasonable range
    if frequency < 1.0 or frequency > 100.0:
        return np.nan
    
    return frequency


def calculate_bcf(
    track_df: pd.DataFrame, pixel_size_um: float, fps: float, window_size: int = 5, vsl: float = 0.0, vap: float = 0.0, 
    position_threshold: float = 2.0
) -> float:
    """
    Calculate Beat-Cross Frequency (BCF) - adapted for tethered sperm.
    
    For normal motile sperm: uses average path crossing method.
    For tethered sperm (VSL ≈ 0, VAP ≈ 0): uses head vibration frequency analysis.
    """
    # Check if this is likely a tethered sperm
    is_tethered = (vsl < 2.0) and (vap < 2.0)  # Very low net movement
    
    if is_tethered:
        # For tethered sperm, analyze head vibration frequency
        return calculate_head_vibration_frequency(track_df, fps, position_threshold)
    else:
        # For motile sperm, use traditional average path crossing method
        return calculate_motile_bcf(track_df, pixel_size_um, fps, window_size)


def calculate_motile_bcf(
    track_df: pd.DataFrame, pixel_size_um: float, fps: float, window_size: int = 5
) -> float:
    """
    Calculate Beat-Cross Frequency (BCF) for motile sperm using average path crossing method.
    """
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
    
    # Calculate signed distances from trajectory points to the average path line
    p0 = np.array([q_x[0], q_y[0]])
    signed_distances = []
    
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        dist = np.dot(p - p0, perp_direction)
        signed_distances.append(dist)
    
    # Count zero crossings
    crossing_count = 0
    start_idx = window_size // 2
    end_idx = len(x) - window_size // 2
    
    for i in range(start_idx, end_idx - 1):
        if signed_distances[i] * signed_distances[i+1] < 0:
            crossing_count += 1
    
    effective_frames = end_idx - start_idx
    if effective_frames <= 0:
        return np.nan
    
    bcf = (crossing_count * fps) / effective_frames
    return bcf


def classify_motility_v2(
    vsl: float,
    vap: float,
    str_ratio: float,
    vcl: float,
    alh_mean: float,
    alh_max: float,
    bcf: float = 0.0,
    tethered_vcl_threshold: float = 40.0,
    tethered_vsl_threshold: float = 5.0,
    tethered_alh_threshold: float = 1.5,
    tethered_bcf_threshold: float = 8.0,
    tethered_str_threshold: float = 50.0,
) -> Dict[str, Any]:
    """
    Classify motility with nested categories including tethered sperm detection:
    - Static: VAP < 5 μm/s AND not tethered
    - Tethered: High beating activity but low net displacement
    - Motile: VAP ≥ 5 μm/s AND VSL > 0 AND not tethered
      - Slow (subset): VSL < 30 AND VAP < 20
      - Progressive (subset): STR ≥ 80% AND VAP ≥ 50 μm/s
        - Hyperactivated (subset): VCL ≥ 270, LIN < 50, ALH ≥ 7 μm
    
    Tethered sperm criteria:
    - VCL > 40 μm/s (active beating)
    - VSL < 5 μm/s (minimal net movement)  
    - ALH_mean > 1.5 μm (detectable lateral movement)
    - BCF > 8 Hz (rapid beating)
    - STR < 50% (low straightness)
    
    Note: These criteria are tuned for the specific biological context and
    imaging conditions. Adjust thresholds based on your experimental setup.
    """
    # Check if sperm is tethered (beating but not moving)
    is_tethered = (
        (vcl > tethered_vcl_threshold) and           # Active beating
        (vsl < tethered_vsl_threshold) and            # Minimal net movement
        (alh_mean > tethered_alh_threshold) and       # Detectable lateral movement
        (bcf > tethered_bcf_threshold) and            # Rapid beating frequency
        (str_ratio < tethered_str_threshold)         # Low straightness
    )
    
    # Traditional motility classification, but exclude tethered sperm
    is_static = (vap < 5.0) and not is_tethered
    is_motile = (vap >= 5.0) and (vsl > 0) and not is_tethered
    is_slow = is_motile and (vsl < 30.0) and (vap < 20.0)
    is_progressive = is_motile and (str_ratio >= 80.0) and (vap >= 50.0)

    # Check hyperactivation criteria
    lin = (vsl / vcl * 100) if vcl > 0 else 0.0
    is_hyperactivated = (
        is_progressive and (vcl >= 270.0) and (lin < 50.0) and (alh_mean >= 7.0)
    )

    return {
        "is_static": is_static,
        "is_tethered": is_tethered,
        "is_motile": is_motile,
        "is_slow": is_slow,
        "is_progressive": is_progressive,
        "is_hyperactivated": is_hyperactivated,
    }


def analyze_sperm_tracks(
    csv_path: str,
    pixel_size_um: float = 0.5,
    fps: float = 60,
    min_track_length: int = 6,
    vap_window: int = 5,
    tethered_vcl_threshold: float = 40.0,
    tethered_vsl_threshold: float = 5.0,
    tethered_alh_threshold: float = 1.5,
    tethered_bcf_threshold: float = 8.0,
    tethered_str_threshold: float = 50.0,
    tethered_vibration_threshold: float = 2.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze sperm tracks with full parameter calculation
    """
    df = pd.read_csv(csv_path)
    df = df[df["state"] == "confirmed"].copy()

    if len(df) == 0:
        raise ValueError("No confirmed tracks found.")

    results = []

    for track_id, track_df in df.groupby("track_id"):
        track_df = track_df.sort_values("frame").reset_index(drop=True)

        if len(track_df) < max(min_track_length, vap_window + 2):
            continue

        # Calculate velocities
        vsl = calculate_vsl(track_df, pixel_size_um, fps)
        vcl = calculate_vcl(track_df, pixel_size_um, fps)
        vap = calculate_vap(track_df, pixel_size_um, fps, vap_window)

        if np.isnan([vsl, vcl, vap]).any():
            continue

        # Calculate ratios
        lin = (vsl / vcl * 100) if vcl > 0 else 0.0
        wob = (vap / vcl * 100) if vcl > 0 else 0.0
        str_ratio = (vsl / vap * 100) if vap > 0 else 0.0

        # Calculate ALH
        alh_mean, alh_max = calculate_alh(track_df, pixel_size_um, vap_window)
        
        # Calculate BCF (Beat-Cross Frequency) - pass VSL and VAP for tethered detection
        bcf = calculate_bcf(track_df, pixel_size_um, fps, vap_window, vsl, vap, tethered_vibration_threshold)

        # Classify motility - pass BCF and tethered thresholds for detection
        motility = classify_motility_v2(
            vsl, vap, str_ratio, vcl, alh_mean, alh_max, bcf,
            tethered_vcl_threshold, tethered_vsl_threshold, tethered_alh_threshold,
            tethered_bcf_threshold, tethered_str_threshold
        )

        results.append(
            {
                "track_id": int(track_id),
                "track_length": len(track_df),
                "VSL_um_s": round(vsl, 3),
                "VCL_um_s": round(vcl, 3),
                "VAP_um_s": round(vap, 3),
                "LIN_percent": round(lin, 2),
                "WOB_percent": round(wob, 2),
                "STR_percent": round(str_ratio, 2),
                "ALHmean_um": round(alh_mean, 3) if not np.isnan(alh_mean) else np.nan,
                "ALHmax_um": round(alh_max, 3) if not np.isnan(alh_max) else np.nan,
                "BCF_Hz": round(bcf, 3) if not np.isnan(bcf) else np.nan,
                **motility,
            }
        )

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return pd.DataFrame(), {}

    # Calculate summary statistics with nested percentages
    total = len(results_df)
    static = results_df["is_static"].sum()
    tethered = results_df["is_tethered"].sum()
    motile = results_df["is_motile"].sum()
    slow = results_df["is_slow"].sum()
    progressive = results_df["is_progressive"].sum()
    hyper = results_df["is_hyperactivated"].sum()

    summary = {
        "total_tracks": total,
        "static_tracks": static,
        "tethered_tracks": tethered,
        "motile_tracks": motile,
        "slow_tracks": slow,
        "progressive_tracks": progressive,
        "hyperactivated_tracks": hyper,
        "static_percent": round(static / total * 100, 2),
        "tethered_percent": round(tethered / total * 100, 2),
        "motile_percent": round(motile / total * 100, 2),
        "immotile_percent": round((total - motile - tethered) / total * 100, 2),
        "slow_percent_of_motile": round(slow / motile * 100, 2) if motile > 0 else 0.0,
        "progressive_percent_of_motile": round(progressive / motile * 100, 2)
        if motile > 0
        else 0.0,
        "hyperactivated_percent_of_progressive": round(hyper / progressive * 100, 2)
        if progressive > 0
        else 0.0,
        "vsl_mean": round(results_df["VSL_um_s"].mean(), 2),
        "vsl_std": round(results_df["VSL_um_s"].std(), 2),
        "vcl_mean": round(results_df["VCL_um_s"].mean(), 2),
        "vcl_std": round(results_df["VCL_um_s"].std(), 2),
        "vap_mean": round(results_df["VAP_um_s"].mean(), 2),
        "vap_std": round(results_df["VAP_um_s"].std(), 2),
        "lin_mean": round(results_df["LIN_percent"].mean(), 2),
        "lin_std": round(results_df["LIN_percent"].std(), 2),
        "wob_mean": round(results_df["WOB_percent"].mean(), 2),
        "wob_std": round(results_df["WOB_percent"].std(), 2),
        "str_mean": round(results_df["STR_percent"].mean(), 2),
        "str_std": round(results_df["STR_percent"].std(), 2),
        "alh_mean": round(results_df["ALHmean_um"].mean(), 3),
        "alh_std": round(results_df["ALHmean_um"].std(), 3),
        "alh_max": round(results_df["ALHmax_um"].max(), 3),
        "bcf_mean": round(results_df["BCF_Hz"].mean(), 3),
        "bcf_std": round(results_df["BCF_Hz"].std(), 3),
        "bcf_max": round(results_df["BCF_Hz"].max(), 3),
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

    # Calculate motile subset statistics
    motile_count = summary["motile_tracks"]
    prog_count = summary["progressive_tracks"]

    slow_pct_of_motile = summary["slow_percent_of_motile"] if motile_count > 0 else 0.0
    prog_pct_of_motile = (
        summary["progressive_percent_of_motile"] if motile_count > 0 else 0.0
    )
    hyper_pct_of_prog = (
        summary["hyperactivated_percent_of_progressive"] if prog_count > 0 else 0.0
    )

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           SPERM MOTILITY ANALYSIS REPORT                     ║
║           Based on OpenCASA Definitions                      ║
╚══════════════════════════════════════════════════════════════╝

📊 ANALYSIS PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Pixel size:          {params["pixel_size_um"]} μm/pixel
  Frame rate:          {params["fps"]} fps
  VAP window (w):      {params["vap_window"]} frames
  Min track length:    {params["min_track_length"]} frames

📈 MOTILITY CLASSIFICATION (WHO 6th Edition + Hyperactivated)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total tracks:             {summary["total_tracks"]}
  
  Static (VAP<5):           {summary["static_percent"]:>6.2f}% {pb(summary["static_percent"])} ({summary["static_tracks"]} tracks)
  Motile (VAP≥5 & VSL>0):   {summary["motile_percent"]:>6.2f}% {pb(summary["motile_percent"])} ({summary["motile_tracks"]} tracks)
  Immotile:                 {summary["immotile_percent"]:>6.2f}% {pb(summary["immotile_percent"])} ({summary["total_tracks"] - summary["motile_tracks"]} tracks)

  Of motile sperm ({motile_count} tracks):
  • Slow (VSL<30 & VAP<20):     {slow_pct_of_motile:>6.2f}% {pb(slow_pct_of_motile)} ({summary["slow_tracks"]} tracks)
  • Progressive (STR≥80 & VAP≥50): {prog_pct_of_motile:>6.2f}% {pb(prog_pct_of_motile)} ({summary["progressive_tracks"]} tracks)

  Of progressive sperm ({prog_count} tracks):
  • Hyperactivated:          {hyper_pct_of_prog:>6.2f}% {pb(hyper_pct_of_prog)} ({summary["hyperactivated_tracks"]} tracks)

📏 AMPLITUDE OF LATERAL HEAD DISPLACEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ALH mean:                  {summary["alh_mean"]:>6.3f} ± {summary["alh_std"]:>5.3f} μm
  ALH max:                   {summary["alh_max"]:>6.3f} μm
  
🌊 BEAT-CROSS FREQUENCY (BCF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BCF mean:                  {summary["bcf_mean"]:>6.3f} ± {summary["bcf_std"]:>5.3f} Hz
  BCF max:                   {summary["bcf_max"]:>6.3f} Hz

⚡ VELOCITY PARAMETERS (Mean ± SD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VSL (Straight-line):    {summary["vsl_mean"]:>6.2f} ± {summary["vsl_std"]:>5.2f} μm/s
  VCL (Curvilinear):      {summary["vcl_mean"]:>6.2f} ± {summary["vcl_std"]:>5.2f} μm/s
  VAP (Average path):     {summary["vap_mean"]:>6.2f} ± {summary["vap_std"]:>5.2f} μm/s

🎯 KINEMATIC RATIOS (Mean ± SD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LIN (Linearity):         {summary["lin_mean"]:>6.2f} ± {summary["lin_std"]:>5.2f}%
  WOB (Wobble):            {summary["wob_mean"]:>6.2f} ± {summary["wob_std"]:>5.2f}%
  STR (Straightness):      {summary["str_mean"]:>6.2f} ± {summary["str_std"]:>5.2f}%

📉 DISTRIBUTION RANGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VSL:   {results_df["VSL_um_s"].min():>6.2f} - {results_df["VSL_um_s"].max():>6.2f} μm/s
  VCL:   {results_df["VCL_um_s"].min():>6.2f} - {results_df["VCL_um_s"].max():>6.2f} μm/s
  VAP:   {results_df["VAP_um_s"].min():>6.2f} - {results_df["VAP_um_s"].max():>6.2f} μm/s
"""

    # Add warnings
    warnings = []
    if results_df["track_length"].min() < 10:
        warnings.append("⚠️  WARNING: Some tracks are very short (<10 frames)")
    if summary["total_tracks"] < 100:
        warnings.append(
            "⚠️  WARNING: Low number of tracks (<100) may affect reliability"
        )
    if results_df["ALHmean_um"].isna().all():
        warnings.append("⚠️  WARNING: ALH could not be calculated for some tracks")

    if warnings:
        report += "\n" + "\n".join(warnings) + "\n"

    return report


def create_combined_summary(all_summaries: list, output_dir: str, params: dict):
    """Create a combined summary report for all processed files."""
    if len(all_summaries) <= 1:
        return
    
    # Combine all summaries into a DataFrame
    combined_df = pd.DataFrame(all_summaries)
    
    # Calculate overall statistics
    total_files = len(all_summaries)
    total_tracks = combined_df['total_tracks'].sum()
    total_motile = combined_df['motile_tracks'].sum()
    total_progressive = combined_df['progressive_tracks'].sum()
    total_static = combined_df['static_tracks'].sum()
    total_hyper = combined_df['hyperactivated_tracks'].sum()
    
    # Calculate weighted averages
    motile_percent_weighted = (combined_df['motile_percent'] * combined_df['total_tracks']).sum() / total_tracks if total_tracks > 0 else 0.0
    progressive_percent_weighted = (combined_df['progressive_percent_of_motile'] * combined_df['motile_tracks']).sum() / total_motile if total_motile > 0 else 0.0
    
    # Calculate velocity statistics
    velocity_cols = ['vsl_mean', 'vcl_mean', 'vap_mean', 'lin_mean', 'wob_mean', 'str_mean', 'alh_mean', 'bcf_mean']
    velocity_stats = {}
    for col in velocity_cols:
        if col in combined_df.columns:
            # Weighted average based on total tracks
            weighted_avg = (combined_df[col] * combined_df['total_tracks']).sum() / total_tracks if total_tracks > 0 else 0.0
            velocity_stats[f'{col}_weighted'] = round(weighted_avg, 3)
    
    combined_summary = {
        "total_files": total_files,
        "total_tracks": total_tracks,
        "total_motile_tracks": total_motile,
        "total_progressive_tracks": total_progressive,
        "total_static_tracks": total_static,
        "total_hyperactivated_tracks": total_hyper,
        "overall_motile_percent": round(motile_percent_weighted, 2),
        "overall_progressive_percent": round(progressive_percent_weighted, 2),
        "overall_static_percent": round(total_static / total_tracks * 100, 2),
        **velocity_stats,
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": params
    }
    
    # Create combined summary report
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           COMBINED BATCH ANALYSIS SUMMARY                    ║
╚══════════════════════════════════════════════════════════════╝

📊 OVERALL STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total files processed:     {total_files}
  Total tracks analyzed:     {total_tracks}
  
  Overall motility:
  • Motile (VAP≥5):          {motile_percent_weighted:>6.2f}% ({total_motile} tracks)
  • Progressive (STR≥80):    {progressive_percent_weighted:>6.2f}% ({total_progressive} tracks)
  • Static:                  {combined_summary['overall_static_percent']:>6.2f}% ({total_static} tracks)
  
  Velocity Parameters (Weighted Average):
  • VSL:                     {velocity_stats.get('vsl_mean_weighted', 'N/A')} μm/s
  • VCL:                     {velocity_stats.get('vcl_mean_weighted', 'N/A')} μm/s
  • VAP:                     {velocity_stats.get('vap_mean_weighted', 'N/A')} μm/s

📈 KINEMATIC RATIOS (Weighted Average):
  • LIN (Linearity):         {velocity_stats.get('lin_mean_weighted', 'N/A')}%
  • WOB (Wobble):            {velocity_stats.get('wob_mean_weighted', 'N/A')}%
  • STR (Straightness):      {velocity_stats.get('str_mean_weighted', 'N/A')}%
  
🌊 BEAT-CROSS FREQUENCY (BCF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BCF (Weighted Average):    {velocity_stats.get('bcf_mean_weighted', 'N/A')} Hz

📅 Analysis completed: {combined_summary['analysis_date']}
"""
    
    # Save combined summary
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save report
    (output_path / "batch_combined_summary.txt").write_text(report)
    
    # Save detailed combined data
    combined_df.to_csv(output_path / "batch_file_summaries.csv", index=False)
    
    # Save combined summary as JSON for programmatic access
    import json
    # Convert numpy types to Python native types for JSON serialization
    json_safe_summary = {}
    for key, value in combined_summary.items():
        if isinstance(value, (np.integer, np.int64)):
            json_safe_summary[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            json_safe_summary[key] = float(value)
        elif key == "parameters":
            # Handle the parameters dict specially
            json_safe_summary[key] = {k: float(v) if isinstance(v, (np.floating, np.float64)) else int(v) if isinstance(v, (np.integer, np.int64)) else v for k, v in value.items()}
        else:
            json_safe_summary[key] = value
    
    with open(output_path / "batch_combined_summary.json", "w") as f:
        json.dump(json_safe_summary, f, indent=2)
    
    print(f"📊 Saved combined summary: {output_path}/batch_combined_summary.txt")
    print(f"📈 Saved file summaries: {output_path}/batch_file_summaries.csv")
    print(f"📄 Saved JSON summary: {output_path}/batch_combined_summary.json")


def save_outputs(
    results_df: pd.DataFrame,
    summary: Dict[str, Any],
    report: str,
    output_dir: str,
    output_prefix: str,
    params: Dict[str, Any],
):
    """Save all output files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Detailed results
    results_df.to_csv(output_path / f"{output_prefix}_detailed.csv", index=False)

    # 2. Summary statistics
    stats_cols = [
        "VSL_um_s",
        "VCL_um_s",
        "VAP_um_s",
        "LIN_percent",
        "WOB_percent",
        "STR_percent",
        "ALHmean_um",
        "ALHmax_um",
        "BCF_Hz",
    ]
    stats_df = results_df[stats_cols].describe()
    stats_df.to_csv(output_path / f"{output_prefix}_statistics.csv")

    # 3. Summary report (TXT)
    (output_path / f"{output_prefix}_summary.txt").write_text(report)

    # 4. Markdown report (MD)
    (output_path / f"{output_prefix}_report.md").write_text(report)

    # 5. Parameters
    params_df = pd.DataFrame([{"parameter": k, "value": v} for k, v in params.items()])
    params_df.to_csv(output_path / f"{output_prefix}_parameters.csv", index=False)

    print(f"📄 Saved: {output_path}/{output_prefix}_detailed.csv")
    print(f"📊 Saved: {output_path}/{output_prefix}_statistics.csv")
    print(f"📝 Saved: {output_path}/{output_prefix}_summary.txt")
    print(f"📝 Saved: {output_path}/{output_prefix}_report.md")
    print(f"⚙️  Saved: {output_path}/{output_prefix}_parameters.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Sperm motility analysis based on OpenCASA kinematic parameters"
    )

    parser.add_argument("csv_file", nargs="?", help="Path to tracking data CSV file (for single file mode)")
    parser.add_argument(
        "--input-dir",
        help="Input directory for batch processing (alternative to csv_file)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.5,
        help="Micrometers per pixel (default: 0.5)",
    )
    parser.add_argument(
        "--fps", type=float, default=60, help="Frames per second (default: 60)"
    )
    parser.add_argument(
        "--vap-window",
        type=int,
        default=5,
        help="Moving average window size for VAP (default: 5)",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=6,
        help="Minimum track length in frames (default: 6)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: same as input CSV)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: CSV filename stem)",
    )
    
    # Batch processing options
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process directories recursively for batch processing",
    )
    parser.add_argument(
        "--glob",
        default="*_confirmed_tracks.csv",
        help="Glob pattern for CSV filtering in batch mode (default: *_confirmed_tracks.csv)",
    )
    parser.add_argument(
        "--output-dir-batch",
        type=str,
        default=None,
        help="Output directory for batch processing (defaults to input directory if not specified)",
    )
    
    # Tethered sperm detection thresholds
    parser.add_argument(
        "--tethered-vcl-threshold",
        type=float,
        default=40.0,
        help="Minimum VCL um/s for tethered sperm detection (default: 40.0)",
    )
    parser.add_argument(
        "--tethered-vsl-threshold",
        type=float,
        default=5.0,
        help="Maximum VSL um/s for tethered sperm detection (default: 5.0)",
    )
    parser.add_argument(
        "--tethered-alh-threshold",
        type=float,
        default=1.5,
        help="Minimum ALH um for tethered sperm detection (default: 1.5)",
    )
    parser.add_argument(
        "--tethered-bcf-threshold",
        type=float,
        default=8.0,
        help="Minimum BCF Hz for tethered sperm detection (default: 8.0)",
    )
    parser.add_argument(
        "--tethered-str-threshold",
        type=float,
        default=50.0,
        help="Maximum STR percent for tethered sperm detection (default: 50.0)",
    )
    parser.add_argument(
        "--tethered-vibration-threshold",
        type=float,
        default=2.0,
        help="Maximum position change in pixels for vibration analysis (default: 2.0)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.csv_file and not args.input_dir:
        parser.error("Either csv_file or --input-dir must be specified")
    
    if args.csv_file and args.input_dir:
        parser.error("Cannot specify both csv_file and --input-dir")
    
    # Note: --output-dir-batch is optional, defaults to input directory if not specified

    # Determine processing mode: single file or batch
    if args.csv_file:
        # Single file mode
        return process_single_file(args)
    else:
        # Batch processing mode
        return process_batch_files(args)


def process_single_file(args):
    """Process a single CSV file."""
    # Set defaults
    if args.output_dir is None:
        args.output_dir = str(Path(args.csv_file).parent)
    if args.output_prefix is None:
        args.output_prefix = Path(args.csv_file).stem

    # Validate
    if not Path(args.csv_file).exists():
        print(f"❌ Error: File '{args.csv_file}' not found.")
        return 1
    if args.vap_window < 2:
        print("❌ Error: VAP window size must be at least 2.")
        return 1

    # Package parameters
    params = {
        "pixel_size_um": args.pixel_size,
        "fps": args.fps,
        "vap_window": args.vap_window,
        "min_track_length": args.min_track_length,
        "tethered_vcl_threshold": args.tethered_vcl_threshold,
        "tethered_vsl_threshold": args.tethered_vsl_threshold,
        "tethered_alh_threshold": args.tethered_alh_threshold,
        "tethered_bcf_threshold": args.tethered_bcf_threshold,
        "tethered_str_threshold": args.tethered_str_threshold,
        "tethered_vibration_threshold": args.tethered_vibration_threshold,
    }

    print(f"🔬 Analyzing: {args.csv_file}")
    print(f"📏 {len(pd.read_csv(args.csv_file))} data points loaded")

    # Run analysis
    results_df, summary = analyze_sperm_tracks(args.csv_file, **params)

    if len(results_df) == 0:
        print("\n❌ No valid tracks found. Check your data and parameters.")
        print("   - Ensure tracks have enough frames (min length > VAP window)")
        return 1

    # Generate and print report
    report = generate_report(summary, results_df, params)
    print(report)

    # Save outputs
    save_outputs(
        results_df, summary, report, args.output_dir, args.output_prefix, params
    )

    print(f"\n✅ Analysis complete! {summary['total_tracks']} tracks analyzed.")
    return 0


def process_batch_files(args):
    """Process multiple CSV files in batch mode."""
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"❌ Error: Input directory '{input_path}' not found.")
        return 1

    # Find CSV files
    if args.recursive:
        csv_files = list(input_path.rglob(args.glob))
    else:
        csv_files = list(input_path.glob(args.glob))
    
    if not csv_files:
        print(f"❌ Error: No files matching pattern '{args.glob}' found in {input_path}")
        return 1
    
    # Remove duplicates and sort
    csv_files = sorted(list(set(csv_files)))
    
    # Set default output directory to input directory if not specified
    output_base_dir = Path(args.output_dir_batch) if args.output_dir_batch else input_path
    
    print(f"🔍 Found {len(csv_files)} CSV files for batch analysis")
    print(f"📁 Input directory: {input_path}")
    print(f"📁 Output directory: {output_base_dir}")
    print(f"🔍 File pattern: {args.glob}")
    print(f"🔄 Recursive: {args.recursive}")
    
    # Package parameters for all files
    params = {
        "pixel_size_um": args.pixel_size,
        "fps": args.fps,
        "vap_window": args.vap_window,
        "min_track_length": args.min_track_length,
        "tethered_vcl_threshold": args.tethered_vcl_threshold,
        "tethered_vsl_threshold": args.tethered_vsl_threshold,
        "tethered_alh_threshold": args.tethered_alh_threshold,
        "tethered_bcf_threshold": args.tethered_bcf_threshold,
        "tethered_str_threshold": args.tethered_str_threshold,
        "tethered_vibration_threshold": args.tethered_vibration_threshold,
    }
    
    # Process files
    start_time = time.time()
    results = []
    all_summaries = []
    
    print(f"\n🚀 Starting batch analysis...")
    for i, csv_file in enumerate(csv_files, 1):
        try:
            print(f"\n[{i}/{len(csv_files)}] Analyzing: {csv_file.name}")
            
            # Determine output directory for this file
            if args.recursive:
                try:
                    # Preserve folder structure
                    rel_path = csv_file.relative_to(input_path)
                    rel_dir = rel_path.parent
                    output_dir = output_base_dir / rel_dir
                except ValueError:
                    output_dir = output_base_dir / csv_file.parent.name
            else:
                output_dir = output_base_dir
            
            # Determine output prefix
            output_prefix = csv_file.stem
            
            # Load and check data
            df = pd.read_csv(csv_file)
            print(f"   📏 {len(df)} data points loaded")
            
            # Run analysis
            results_df, summary = analyze_sperm_tracks(str(csv_file), **params)
            
            if len(results_df) == 0:
                print(f"   ⚠️  No valid tracks found in {csv_file.name}")
                results.append({
                    'file': str(csv_file),
                    'success': False,
                    'error': 'No valid tracks found',
                    'total_tracks': 0
                })
                continue
            
            # Generate report
            report = generate_report(summary, results_df, params)
            
            # Save outputs
            save_outputs(results_df, summary, report, str(output_dir), output_prefix, params)
            
            results.append({
                'file': str(csv_file),
                'success': True,
                'error': None,
                'total_tracks': summary['total_tracks'],
                'motile_percent': summary['motile_percent'],
                'progressive_percent': summary['progressive_percent_of_motile']
            })
            all_summaries.append(summary)
            
            print(f"   ✅ {summary['total_tracks']} tracks analyzed")
            
        except Exception as e:
            print(f"   ❌ Error processing {csv_file.name}: {e}")
            results.append({
                'file': str(csv_file),
                'success': False,
                'error': str(e),
                'total_tracks': 0
            })
    
    # Generate batch summary
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_tracks = sum(r.get('total_tracks', 0) for r in results)
    
    print(f"\n{'='*60}")
    print(f"Batch Analysis Summary")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total tracks analyzed: {total_tracks}")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Average time per file: {elapsed_time/len(results):.1f} seconds")
    
    if failed > 0:
        print(f"\nFailed files:")
        for result in results:
            if not result['success']:
                print(f"  ✗ {Path(result['file']).name}")
                if result['error']:
                    print(f"    Error: {result['error']}")
    
    # Create combined summary if multiple files were processed
    if len(all_summaries) > 1:
        create_combined_summary(all_summaries, output_base_dir, params)
    
    print(f"\n✅ Batch analysis complete!")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
