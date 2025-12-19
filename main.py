#!/usr/bin/env python3
"""
Sperm Motility Analysis Pipeline - Main entry point for detection, tracking, and analysis.

Provides a unified interface for processing sperm motility videos with configurable
parameters for each stage: detection, tracking, and motility analysis.

Usage:
    python -m src.main movie.tif -o results/
    python -m src.main movies/ -o results/ --viz-dir raw_videos/
    python -m src.main movie.tif -o results/ --params-file params.json --detect-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Core algorithm imports
try:
    from src.core.detect import detect_sperm
    from src.core.analysis import analyze_tracks, generate_report
    from src.core.visualization import (
        create_detection_overlay,
        create_tracking_overlay,
        create_overview_frame,
    )
    from src.core.common.io import load_movie
    from src.core.motile_tracker import (
        TrackerConfig,
        TrajectoryPostConfig,
        run_tracking_with_configs,
    )
    from src.core.tracker import (
        run_two_stage_tracking,
        ImmotileConfig,
    )
except ImportError:
    # fallback relative for local dev
    sys.path.append(str(Path(__file__).parent.parent))
    from src.core.detect import detect_sperm
    from src.core.analysis import analyze_tracks, generate_report
    from src.core.visualization import (
        create_detection_overlay,
        create_tracking_overlay,
        create_overview_frame,
    )
    from src.core.common.io import load_movie
    from src.core.motile_tracker import (
        TrackerConfig,
        TrajectoryPostConfig,
        run_tracking_with_configs,
    )
    from src.core.tracker import (
        run_two_stage_tracking,
        ImmotileConfig,
    )

# ---------------------- Utilities ----------------------


def setup_logging(output_dir: Path, level: int = logging.INFO) -> None:
    """Configure logging to file and stdout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"

    # Clear existing handlers
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    # Create formatters and handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def validate_input_path(
    input_path: Path, glob_pattern: Optional[str] = None, recursive: bool = False
) -> Tuple[bool, List[Path]]:
    """Return list of files to process (supports single file, directory, or glob pattern)."""
    if glob_pattern:
        # Use glob pattern to find files
        if input_path.is_dir():
            if recursive:
                files = list(input_path.rglob(glob_pattern))
            else:
                files = list(input_path.glob(glob_pattern))
            # Filter to only supported file types
            supported = {".tif", ".tiff", ".mp4", ".avi", ".mov"}
            files = [f for f in files if f.suffix.lower() in supported]
            return True, sorted(files)
        else:
            logging.error("--input-glob can only be used with a directory input path")
            return False, []
    else:
        if not input_path.exists():
            logging.error("Input path does not exist: %s", input_path)
            return False, []

        supported = {".tif", ".tiff", ".mp4", ".avi", ".mov"}
        if input_path.is_file():
            if input_path.suffix.lower() not in supported:
                logging.error("Unsupported input extension: %s", input_path.suffix)
                return False, []
            return True, [input_path]
        else:
            files = []
            for ext in supported:
                if recursive:
                    files.extend(sorted(input_path.rglob(f"*{ext}")))
                else:
                    files.extend(sorted(input_path.glob(f"*{ext}")))
            return True, files


def find_visualization_movie(
    input_movie_path: Path,
    viz_dir: Optional[Path] = None,
    cut_input_prefix: Optional[str] = None,
    cut_input_suffix: Optional[str] = None,
    cut_viz_prefix: Optional[str] = None,
    cut_viz_suffix: Optional[str] = None,
    recursive: bool = False,
    viz_glob: Optional[str] = None,
) -> Path:
    """Reuse your previous heuristic for selecting a visualization movie."""
    if not viz_dir or not viz_dir.exists():
        logging.info(
            "Visualization directory not provided or does not exist: %s", viz_dir
        )
        return input_movie_path

    logging.info(
        "Looking for visualization movie for: %s in directory: %s (recursive: %s, viz_glob: %s)",
        input_movie_path.name,
        viz_dir,
        recursive,
        viz_glob,
    )

    video_exts = {".mp4", ".avi", ".mov"}
    image_exts = {".tif", ".tiff"}

    stem = input_movie_path.stem
    original_stem = stem
    if cut_input_prefix and stem.startswith(cut_input_prefix):
        logging.debug(
            "Applying cut_input_prefix: %s -> %s", stem, stem[len(cut_input_prefix) :]
        )
        stem = stem[len(cut_input_prefix) :]
    if cut_input_suffix and stem.endswith(cut_input_suffix):
        logging.debug(
            "Applying cut_input_suffix: %s -> %s", stem, stem[: -len(cut_input_suffix)]
        )
        stem = stem[: -len(cut_input_suffix)]

    logging.debug("Final input stem after processing: %s", stem)

    # exact-match in same category
    inp_ext = input_movie_path.suffix.lower()
    is_video = inp_ext in video_exts
    is_image = inp_ext in image_exts

    # When recursive is True, first look in the same directory as the input movie
    # to avoid mismatches when processing directories with multiple related files
    input_movie_dir = input_movie_path.parent
    if recursive and input_movie_dir != viz_dir:
        # First look for matching files in the same directory as input movie
        if viz_glob:
            same_dir_files = list(input_movie_dir.rglob(viz_glob))
        else:
            same_dir_files = list(input_movie_dir.rglob("*"))

        logging.debug(
            "Checking for visualization movie in input movie directory: %s (found %d files)",
            input_movie_dir,
            len(same_dir_files),
        )

        for f in same_dir_files:
            if not f.is_file():
                continue
            # Skip the exact same file as input (avoid using input as visualization)
            if f.resolve() == input_movie_path.resolve():
                continue
            ext = f.suffix.lower()
            if is_video and ext not in video_exts:
                continue
            if is_image and ext not in image_exts:
                continue
            viz_stem = f.stem
            if cut_viz_prefix and viz_stem.startswith(cut_viz_prefix):
                viz_stem = viz_stem[len(cut_viz_prefix) :]
            if cut_viz_suffix and viz_stem.endswith(cut_viz_suffix):
                viz_stem = viz_stem[: -len(cut_viz_suffix)]

            if viz_stem == stem:
                logging.info(
                    "Visualization movie matched in same directory as input: %s -> %s",
                    input_movie_path.name,
                    f.name,
                )
                return f

    # Get files based on recursive flag and optional glob pattern
    if viz_glob:
        if recursive:
            all_files = list(viz_dir.rglob(viz_glob))
        else:
            all_files = list(viz_dir.glob(viz_glob))
        logging.debug(
            "Found %d files in viz_dir matching glob '%s'", len(all_files), viz_glob
        )
    else:
        if recursive:
            all_files = list(viz_dir.rglob("*"))
        else:
            all_files = list(viz_dir.iterdir())
        logging.debug("Found %d files in viz_dir", len(all_files))

    for f in all_files:
        if not f.is_file():
            continue
        # Skip the exact same file as input (avoid using input as visualization)
        if f.resolve() == input_movie_path.resolve():
            logging.debug("Skipping input file itself: %s", f.name)
            continue
        ext = f.suffix.lower()
        if is_video and ext not in video_exts:
            continue
        if is_image and ext not in image_exts:
            continue
        viz_stem = f.stem
        if cut_viz_prefix and viz_stem.startswith(cut_viz_prefix):
            logging.debug(
                "For file %s, applying cut_viz_prefix: %s -> %s",
                f.name,
                viz_stem,
                viz_stem[len(cut_viz_prefix) :],
            )
            viz_stem = viz_stem[len(cut_viz_prefix) :]
        if cut_viz_suffix and viz_stem.endswith(cut_viz_suffix):
            logging.debug(
                "For file %s, applying cut_viz_suffix: %s -> %s",
                f.name,
                viz_stem,
                viz_stem[: -len(cut_viz_suffix)],
            )
            viz_stem = viz_stem[: -len(cut_viz_suffix)]

        logging.debug(
            "Comparing processed stems: input=%s vs viz=%s (from %s)",
            stem,
            viz_stem,
            f.name,
        )

        if viz_stem == stem:
            logging.info(
                "Visualization movie matched: %s -> %s", input_movie_path.name, f.name
            )
            return f

    logging.warning(
        "No matching visualization movie found for %s in directory %s",
        input_movie_path.name,
        viz_dir,
    )
    return input_movie_path


def safe_load_csv(path: Path, description: str) -> Optional[pd.DataFrame]:
    if path.exists():
        logging.info("Loading %s: %s", description, path.name)
        return pd.read_csv(path)
    logging.error("Missing %s: %s", description, path.name)
    return None


# ---------------------- Parameter parsing helpers ----------------------


def create_default_params() -> Dict[str, Dict[str, Any]]:
    """Return default params."""
    return {
        "detection": {
            "min_area": 20,
            "max_area": 45,
            "min_aspect": 1.2,
            "max_aspect": 3.0,
            "min_solidity": 0.65,
            "threshold": 10,
            "blur_radius": 0.5,
            "visualize": True,
            "color": "red",
            "point_radius": 5,
            "fps": 30,
        },
        # Note: tracker config will be mapped into TrackerConfig dataclass
        "tracking": {
            "overlay": {
                "trail_length": 10,
                "track_width": 2,
                "track_color": "red",
                "point_radius": 2,
                "point_color": "red",
                "fps": 30,
                "overview_frame": -2,
                "overview_track_width": 1.0,
            },
            # Advanced multi-factor matching parameters
            "max_distance": 80.0,
            "max_age": 5,
            "min_hits": 3,
            "min_track_length": 10,
            "appearance_thresh": 0.85,
            "appearance_verify_dist": 50.0,
            "use_kalman": False,
            "weight_distance": 0.3,
            "weight_direction": 0.4,
            "weight_speed": 0.2,
            "weight_morphology": 0.1,
            "sigma_distance": 40.0,
            "sigma_angle": 45.0,  # degrees, will be converted to radians
            "sigma_speed": 80.0,
            "angle_hard_cut": 120.0,  # degrees, will be converted to radians
            "assignment_mode": "greedy",  # default assignment strategy
            "history_len": 2,  # frames for direction/speed calculation
        },
        "analysis": {
            "pixel_size": 0.5,
            "fps": 60,
            "vap_window": 5,
            "motility_vcl_threshold": 20.0,
            "motility_vsl_threshold": 4.0,
            "motility_vap_threshold": 4.0,
        },
    }


def parse_detection_params(
    args: argparse.Namespace, defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Return detection parameters with CLI overrides."""
    params = dict(defaults)

    # Mapping of CLI argument names to parameter names
    param_mappings = {
        "det_min_area": "min_area",
        "det_max_area": "max_area",
        "det_min_aspect": "min_aspect",
        "det_max_aspect": "max_aspect",
        "det_min_solidity": "min_solidity",
        "det_threshold": "threshold",
        "det_blur_radius": "blur_radius",
        "det_point_radius": "point_radius",
        "det_point_color": "color",
        "det_fps": "fps",
    }

    # Apply CLI overrides
    for cli_arg, param_name in param_mappings.items():
        if hasattr(args, cli_arg) and getattr(args, cli_arg) is not None:
            params[param_name] = getattr(args, cli_arg)

    return params


def parse_immotile_config(
    args: argparse.Namespace, params: Dict[str, Any] = None
) -> ImmotileConfig:
    """Return immotile tracking configuration with CLI overrides."""
    config = ImmotileConfig()

    # If params dict is provided, apply values from JSON config first
    if params and "immotile_tracking" in params:
        immotile_params = params["immotile_tracking"]

        # Apply values from JSON config file
        for param_name, attr_name in [
            ("use_immotile_mining", None),  # Special handling for boolean flag
            ("first_k_frames", "first_k_frames"),
            ("search_radius", "search_radius"),
            ("min_points", "min_points"),
            ("max_std", "max_std"),
            ("poly_order", "poly_order"),
            ("min_track_length", "min_track_length"),
            ("outlier_percentile", "outlier_percentile"),
            ("max_iterations", "max_iterations"),
            ("spatial_tolerance", "spatial_tolerance"),
        ]:
            if param_name in immotile_params and attr_name is not None:
                setattr(config, attr_name, immotile_params[param_name])

    # Apply CLI overrides
    cli_mappings = {
        "imm_first_k_frames": "first_k_frames",
        "imm_search_radius": "search_radius",
        "imm_min_points": "min_points",
        "imm_max_std": "max_std",
        "imm_poly_order": "poly_order",
        "imm_min_track_length": "min_track_length",
        "imm_n_jobs": "n_jobs",
    }

    for cli_arg, attr_name in cli_mappings.items():
        if hasattr(args, cli_arg) and getattr(args, cli_arg) is not None:
            setattr(config, attr_name, getattr(args, cli_arg))

    # Handle batch_size separately since it can be string or int
    if hasattr(args, "imm_batch_size") and args.imm_batch_size is not None:
        config.batch_size = args.imm_batch_size

    return config


def parse_tracker_config(
    args: argparse.Namespace, params: Dict[str, Any] = None
) -> TrackerConfig:
    """Return tracker configuration with CLI overrides."""
    config = TrackerConfig()

    # If params dict is provided, apply values from JSON config first
    if params:
        # Apply values from JSON config file
        for param_name, attr_name in [
            ("max_distance", "max_distance"),
            ("max_age", "max_age"),
            ("min_hits", "min_hits"),
            ("min_track_length", "min_track_length"),
            ("appearance_thresh", "appearance_thresh"),
            ("appearance_verify_dist", "appearance_verify_dist"),
            ("use_kalman", "use_kf"),
            ("weight_distance", "w_dist"),
            ("weight_direction", "w_dir"),
            ("weight_speed", "w_speed"),
            ("weight_morphology", "w_morph"),
            ("sigma_distance", "sigma_distance"),
            ("sigma_angle", "sigma_theta"),  # Convert degrees to radians
            ("sigma_speed", "sigma_speed"),
            (
                "angle_hard_cut",
                "theta_hard",
            ),  # Convert degrees to radians in the parameter handling
            ("gamma_tau", "gamma_tau"),
            ("use_tau", "use_tau"),
            ("assignment_mode", "assignment_mode"),
            ("history_len", "history_len"),
            ("min_edge_frames", "min_edge_frames"),
            ("edge_spawn_threshold", "edge_spawn_threshold"),
        ]:
            if param_name in params:
                value = params[param_name]
                # Convert angle parameters from degrees to radians
                if attr_name == "sigma_theta" and value is not None:
                    import numpy as np

                    value = np.deg2rad(value)
                elif attr_name == "theta_hard" and value is not None:
                    import numpy as np

                    value = np.deg2rad(value)
                setattr(config, attr_name, value)

    # Core tracking parameters from CLI args
    tracking_params = {
        "trk_max_distance": "max_distance",
        "trk_max_age": "max_age",
        "trk_min_hits": "min_hits",
        "trk_min_track_length": "min_track_length",
        "trk_appearance_thresh": "appearance_thresh",
        "trk_appearance_verify_dist": "appearance_verify_dist",
        "trk_use_kalman": "use_kf",
        "trk_weight_distance": "weight_distance",
        "trk_weight_direction": "weight_direction",
        "trk_weight_speed": "weight_speed",
        "trk_weight_morphology": "weight_morphology",
        "trk_sigma_distance": "sigma_distance",
        "trk_sigma_angle": "sigma_angle",
        "trk_sigma_speed": "sigma_speed",
        "trk_angle_hard_cut": "angle_hard_cut",
        "trk_gamma_tau": "gamma_tau",
        "trk_use_tau": "use_tau",
        "trk_assignment_mode": "assignment_mode",
        "trk_history_len": "history_len",
        "trk_min_edge_frames": "min_edge_frames",
        "trk_edge_spawn_threshold": "edge_spawn_threshold",
    }

    # Visualization parameters from CLI args
    viz_params = {
        "viz_trail_length": "trail_length",
        "viz_track_width": "track_width",
        "viz_overview_frame": "overview_frame",
        "viz_overview_track_width": "overview_track_width",
        "viz_point_radius": "point_radius",
        "viz_point_color": "point_color",
        "viz_track_color": "track_color",
    }

    # Apply CLI overrides second (CLI takes precedence over JSON config)
    for cli_arg, attr_name in {**tracking_params, **viz_params}.items():
        if hasattr(args, cli_arg) and getattr(args, cli_arg) is not None:
            value = getattr(args, cli_arg)
            # Convert angle parameters from degrees to radians
            if attr_name == "sigma_theta" and value is not None:
                import numpy as np

                value = np.deg2rad(value)
            elif attr_name == "theta_hard" and value is not None:
                import numpy as np

                value = np.deg2rad(value)
            setattr(config, attr_name, value)

    return config


def parse_post_config(
    params: dict, args: argparse.Namespace = None
) -> TrajectoryPostConfig:
    """Parse trajectory post-processing configuration from parameters and CLI args."""
    config = TrajectoryPostConfig()

    # Map parameter names from JSON/config file
    param_mappings = {
        "min_track_length": "min_track_length",
        "min_confidence": "min_confidence",
    }

    # Set values from parameters file first
    for param_name, attr_name in param_mappings.items():
        if param_name in params.get("tracking", {}):
            value = params["tracking"][param_name]
            setattr(config, attr_name, value)

    # Override with CLI arguments if provided
    if args:
        # Handle other direct parameter mappings
        cli_mappings = {
            "trk_min_track_length": "min_track_length",
            "trk_min_confidence": "min_confidence",
        }

        for cli_arg, attr_name in cli_mappings.items():
            if hasattr(args, cli_arg) and getattr(args, cli_arg) is not None:
                setattr(config, attr_name, getattr(args, cli_arg))

    return config


def parse_viz_params(
    args: argparse.Namespace, defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Return visualization parameters with CLI overrides."""
    params = dict(defaults)

    param_mappings = {
        "viz_trail_length": "trail_length",
        "viz_track_width": "track_width",
        "viz_overview_frame": "overview_frame",
        "viz_overview_track_width": "overview_track_width",
        "viz_point_radius": "point_radius",
        "viz_point_color": "point_color",
        "viz_track_color": "track_color",
    }

    for cli_arg, param_name in param_mappings.items():
        if hasattr(args, cli_arg) and getattr(args, cli_arg) is not None:
            params[param_name] = getattr(args, cli_arg)

    return params


def parse_detection_viz_params(
    args: argparse.Namespace, defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Return detection visualization parameters with CLI overrides."""
    params = dict(defaults)

    param_mappings = {
        "det_point_radius": "point_radius",
        "det_point_color": "color",
        "det_fps": "fps",
    }

    for cli_arg, param_name in param_mappings.items():
        if hasattr(args, cli_arg) and getattr(args, cli_arg) is not None:
            params[param_name] = getattr(args, cli_arg)

    return params


def load_configuration(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Load configuration from defaults and optional JSON params file."""
    params = create_default_params()

    if args.params_file and args.params_file.exists():
        with open(args.params_file, "r") as fh:
            loaded = json.load(fh)
            # Merge top-level keys detection/tracking/analysis/immotile_tracking if present
            for key in ("detection", "tracking", "analysis", "immotile_tracking"):
                if key in loaded:
                    if key not in params:
                        params[key] = {}
                    params[key].update(loaded[key])

    return params


def parse_analysis_params(
    args: argparse.Namespace, defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Return analysis parameters with CLI overrides."""
    params = dict(defaults)

    param_mappings = {
        "pixel_size": "pixel_size",
        "fps": "fps",
        "vap_window": "vap_window",
        "ana_motility_vcl_threshold": "motility_vcl_threshold",
        "ana_motility_vsl_threshold": "motility_vsl_threshold",
        "ana_motility_vap_threshold": "motility_vap_threshold",
    }

    for cli_arg, param_name in param_mappings.items():
        if hasattr(args, cli_arg) and getattr(args, cli_arg) is not None:
            params[param_name] = getattr(args, cli_arg)

    return params


# ---------------------- Main pipeline ----------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sperm Motility Analysis Pipeline (clean)")
    p.add_argument("input_path", type=Path, help="Input file or directory")
    p.add_argument(
        "-o", "--output-dir", required=True, type=Path, help="Base output directory"
    )
    p.add_argument("--params-file", type=Path, help="Optional JSON params file")
    p.add_argument(
        "--viz-dir", type=Path, help="Directory with high-quality visualization movies"
    )
    p.add_argument("--cut-input-prefix", type=str)
    p.add_argument("--cut-input-suffix", type=str)
    p.add_argument("--cut-viz-prefix", type=str)
    p.add_argument("--cut-viz-suffix", type=str)
    p.add_argument(
        "--input-glob",
        type=str,
        help="Glob pattern for input files (e.g., '*.tif', '*.mp4')",
    )
    p.add_argument(
        "--viz-glob",
        type=str,
        help="Glob pattern for visualization files (e.g., '*.avi', '*original.avi')",
    )
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search input directory and viz-dir for files",
    )

    # ==================== Detection Parameters ====================
    detection_group = p.add_argument_group("Detection Parameters")
    detection_group.add_argument(
        "--det-min-area",
        type=int,
        default=None,
        help="Minimum detection area (pixels^2)",
    )
    detection_group.add_argument(
        "--det-max-area",
        type=int,
        default=None,
        help="Maximum detection area (pixels^2)",
    )
    detection_group.add_argument(
        "--det-min-aspect", type=float, default=None, help="Minimum aspect ratio"
    )
    detection_group.add_argument(
        "--det-max-aspect", type=float, default=None, help="Maximum aspect ratio"
    )
    detection_group.add_argument(
        "--det-min-solidity",
        type=float,
        default=None,
        help="Minimum solidity threshold (0-1)",
    )
    detection_group.add_argument(
        "--det-threshold",
        type=int,
        default=None,
        help="Threshold for binarization movie before detection (0-255)",
    )
    detection_group.add_argument(
        "--det-blur-radius",
        type=float,
        default=None,
        help="Blur radius for preprocessing (pixel)",
    )
    # Detection visualization parameters
    detection_group.add_argument(
        "--det-point-radius",
        type=int,
        default=None,
        help="Point radius for detection visualization (pixel)",
    )
    detection_group.add_argument(
        "--det-point-color",
        type=str,
        default=None,
        help="Point color for detection visualization (e.g., 'red', 'green', 'blue')",
    )
    detection_group.add_argument(
        "--det-fps",
        type=int,
        default=None,
        help="FPS for detection visualization",
    )

    # ==================== Tracking Parameters ====================
    tracking_group = p.add_argument_group("Tracking Parameters")
    # --- Track initialization and lifetime ---
    tracking_group.add_argument(
        "--trk-max-age",
        type=int,
        default=None,
        help="Maximum number of frames a track can be missing before deletion",
    )
    tracking_group.add_argument(
        "--trk-min-hits",
        type=int,
        default=None,
        help="Minimum number of confirmed detections needed to validate a track",
    )
    tracking_group.add_argument(
        "--trk-min-track-length",
        type=int,
        default=None,
        help="Minimum number of frames in a track to be kept in results",
    )
    tracking_group.add_argument(
        "--trk-min-confidence",
        type=float,
        default=None,
        help="Minimum confidence score for tracks to be included (0-1, default 0.3) - higher values require more reliable tracks",
    )

    # --- Search space and geometric constraints ---
    tracking_group.add_argument(
        "--trk-max-distance",
        type=int,
        default=None,
        help="Maximum distance to search for matching sperm across frames (pixels)",
    )
    tracking_group.add_argument(
        "--trk-angle-hard-cut",
        type=float,
        default=None,
        help="Maximum angle change allowed for track continuation (degrees, default 120) - tracks with sharper turns are broken",
    )

    # --- Association/matching weights and parameters ---
    tracking_group.add_argument(
        "--trk-weight-distance",
        type=float,
        default=None,
        help="Importance of spatial proximity in track matching (0-1, default 0.3)",
    )
    tracking_group.add_argument(
        "--trk-weight-direction",
        type=float,
        default=None,
        help="Importance of movement direction in track matching (0-1, default 0.4)",
    )
    tracking_group.add_argument(
        "--trk-weight-speed",
        type=float,
        default=None,
        help="Importance of movement speed in track matching (0-1, default 0.2)",
    )
    tracking_group.add_argument(
        "--trk-weight-morphology",
        type=float,
        default=None,
        help="Importance of shape similarity in track matching (0-1, default 0.1)",
    )
    tracking_group.add_argument(
        "--trk-sigma-distance",
        type=float,
        default=None,
        help="Distance weight spread for matching (pixels, default max_distance/2) - higher values make distance less critical",
    )
    tracking_group.add_argument(
        "--trk-sigma-angle",
        type=float,
        default=None,
        help="Direction weight spread for matching (degrees, default 45) - higher values allow more directional changes",
    )
    tracking_group.add_argument(
        "--trk-sigma-speed",
        type=float,
        default=None,
        help="Speed weight spread for matching (pixels/frame, default max_distance) - higher values allow more speed variation",
    )
    tracking_group.add_argument(
        "--trk-appearance-thresh",
        type=float,
        default=None,
        help="Minimum similarity threshold for shape matching (0-1, default 0.85)",
    )
    tracking_group.add_argument(
        "--trk-appearance-verify-dist",
        type=float,
        default=None,
        help="Maximum distance to verify appearance similarity (pixels, default 50)",
    )

    # --- Track prediction and filtering ---
    tracking_group.add_argument(
        "--trk-gamma-tau",
        type=float,
        default=None,
        help="Maximum distance for predictive tracking based on movement direction (default 1.2) - higher values predict further ahead",
    )
    tracking_group.add_argument(
        "--trk-use-tau",
        action="store_true",
        default=True,
        help="Enable predictive tracking based on movement direction (default: True)",
    )
    tracking_group.add_argument(
        "--trk-use-kalman",
        action="store_true",
        default=None,
        help="Use Kalman filtering for tracking (default: False - meaning Kalman filtering is disabled)",
    )
    tracking_group.add_argument(
        "--trk-assignment-mode",
        type=str,
        choices=["hungarian", "greedy"],
        default=None,
        help="Assignment strategy: 'hungarian' (globally optimal) or 'greedy' (per-track best, better for dense scenes, default)",
    )
    tracking_group.add_argument(
        "--trk-history-len",
        type=int,
        default=None,
        help="Number of frames to use for direction/speed calculation (default 2, range 1-5)",
    )

    # --- Edge and spawn filtering ---
    tracking_group.add_argument(
        "--trk-min-edge-frames",
        type=int,
        default=None,
        help="Minimum number of frames after which to apply edge-based spawn filtering (default 3)",
    )
    tracking_group.add_argument(
        "--trk-edge-spawn-threshold",
        type=float,
        default=None,
        help="Fraction of image dimension to define edge region for new track spawning (default 0.1 = 10%% from edge)",
    )

    # ==================== Visualization Parameters ====================
    viz_group = p.add_argument_group("Visualization Parameters")
    viz_group.add_argument(
        "--viz-trail-length",
        type=int,
        default=None,
        help="Length of track trail in visualization (frames)",
    )
    viz_group.add_argument(
        "--viz-track-width",
        type=int,
        default=None,
        help="Width of track lines in visualization (pixel)",
    )
    viz_group.add_argument(
        "--viz-overview-frame",
        type=int,
        default=None,
        help="Frame to use for overview image",
    )
    viz_group.add_argument(
        "--viz-overview-track-width",
        type=float,
        default=None,
        help="Track width in overview image (pixel)",
    )
    viz_group.add_argument(
        "--viz-point-radius",
        type=int,
        default=None,
        help="Point radius for visualization (pixel)",
    )
    viz_group.add_argument(
        "--viz-point-color",
        type=str,
        default=None,
        help="Point color for visualization (e.g., 'red', 'green', 'blue')",
    )
    viz_group.add_argument(
        "--viz-track-color",
        type=str,
        default=None,
        help="Track color for visualization (e.g., 'red', 'green', 'blue')",
    )

    # ==================== Immotile Tracking Parameters ====================
    immotile_group = p.add_argument_group("Immotile Tracking Parameters")
    immotile_group.add_argument(
        "--use-immotile-mining",
        action="store_true",
        help="Enable two-stage tracking with immotile sperm separation",
    )
    immotile_group.add_argument(
        "--imm-first-k-frames",
        type=int,
        default=None,
        help="Number of initial frames to use as immotile anchors (default: 3)",
    )
    immotile_group.add_argument(
        "--imm-search-radius",
        type=float,
        default=None,
        help="Search radius for immotile detection proximity (pixels, default: 6.0)",
    )
    immotile_group.add_argument(
        "--imm-min-points",
        type=int,
        default=None,
        help="Minimum detections needed to confirm immotile track (default: 8)",
    )
    immotile_group.add_argument(
        "--imm-max-std",
        type=float,
        default=None,
        help="Maximum spatial standard deviation for immotile stability (pixels, default: 1.5)",
    )
    immotile_group.add_argument(
        "--imm-poly-order",
        type=int,
        default=None,
        help="Polynomial order for immotile drift fitting (0=static, 1=linear, default: 1)",
    )
    immotile_group.add_argument(
        "--imm-min-track-length",
        type=int,
        default=None,
        help="Minimum confirmed immotile track length (default: 6)",
    )
    # Parallel processing options
    immotile_group.add_argument(
        "--imm-n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for immotile processing (-1=all cores, default: -1)",
    )
    immotile_group.add_argument(
        "--imm-batch-size",
        type=str,
        default=None,
        help="Batch size for parallel processing ('auto'=joblib auto, int=custom, default: 'auto')",
    )

    # ==================== Analysis Parameters ====================
    analysis_group = p.add_argument_group("Analysis Parameters")
    analysis_group.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in micrometers (um/pixel)",
    )
    analysis_group.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Recording frames per second (fps), for speed calculation",
    )
    analysis_group.add_argument(
        "--vap-window", type=int, default=None, help="VAP smoothing window (frames)"
    )
    analysis_group.add_argument(
        "--ana-motility-vcl-threshold",
        type=float,
        default=None,
        help="VCL motility threshold (um/frame)",
    )
    analysis_group.add_argument(
        "--ana-motility-vsl-threshold",
        type=float,
        default=None,
        help="VSL motility threshold (um/s)",
    )
    analysis_group.add_argument(
        "--ana-motility-vap-threshold",
        type=float,
        default=None,
        help="VAP motility threshold (um/s)",
    )

    # ==================== Control Flags ====================
    control_group = p.add_argument_group("Control Flags")
    control_group.add_argument(
        "--detect-only", action="store_true", help="Run detection only"
    )
    control_group.add_argument(
        "--tracking-only",
        action="store_true",
        help="Run tracking only (requires existing detections)",
    )
    control_group.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run analysis only (requires existing tracks)",
    )
    control_group.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    return p


def process_single_movie(
    in_file: Path,
    args: argparse.Namespace,
    params: Dict[str, Dict[str, Any]],
    log_level: int,
    relative_path: Optional[Path] = None,
) -> bool:
    """Process a single movie file through the complete pipeline."""
    t0 = time.time()
    movie_name = in_file.stem
    if relative_path:
        # Preserve relative directory structure
        out_dir = args.output_dir / relative_path.parent / movie_name
    else:
        out_dir = args.output_dir / movie_name

    setup_logging(out_dir, level=log_level)
    logging.info("Processing %s", in_file)

    # Extract parameters
    det_params = parse_detection_params(args, params["detection"])
    tracker_cfg = parse_tracker_config(args, params["tracking"])
    analysis_params = parse_analysis_params(args, params["analysis"])
    viz_params = parse_viz_params(args, params["tracking"]["overlay"])
    det_viz_params = parse_detection_viz_params(args, params["detection"])

    # Find visualization movie
    viz_movie = find_visualization_movie(
        in_file,
        args.viz_dir,
        args.cut_input_prefix,
        args.cut_input_suffix,
        args.cut_viz_prefix,
        args.cut_viz_suffix,
        args.recursive,
        args.viz_glob,
    )

    try:
        # --- Detection Phase ---
        detections_df = run_detection_phase(
            in_file,
            out_dir,
            movie_name,
            det_params,
            det_viz_params,
            viz_movie,
            params,
            False,  # skip_detection - removed, always run detection unless --detect-only
        )
        if detections_df is None:
            return False

        # Get frame info
        total_frames, img_shape = get_frame_info(in_file, detections_df, params)

        # Parse post-processing config
        post_cfg = parse_post_config(params, args)

        # --- Tracking Phase ---
        tracks_df = run_tracking_phase(
            detections_df,
            total_frames,
            img_shape,
            out_dir,
            movie_name,
            tracker_cfg,
            post_cfg,
            viz_params,
            viz_movie,
            params,
            args,
            False,  # skip_tracking - removed, always run tracking unless --tracking-only
        )
        if tracks_df is None:
            return False

        # --- Analysis Phase ---
        run_analysis_phase(tracks_df, out_dir, movie_name, analysis_params)

        # --- Summary ---
        write_pipeline_summary(
            out_dir, movie_name, t0, det_params, tracker_cfg, analysis_params
        )
        logging.info("Processing complete: %s (%.2fs)", movie_name, time.time() - t0)
        return True

    except Exception as e:
        logging.error("Processing failed for %s: %s", movie_name, e)
        logging.error("Detailed error information:", exc_info=True)
        return False


def run_detection_phase(
    in_file: Path,
    out_dir: Path,
    movie_name: str,
    det_params: Dict[str, Any],
    det_viz_params: Dict[str, Any],
    viz_movie: Path,
    params: Dict[str, Dict[str, Any]],
    skip_detection: bool = False,
) -> Optional[pd.DataFrame]:
    """Run detection phase and return detections DataFrame."""
    coords_csv = out_dir / f"{movie_name}_det_coords.csv"

    if skip_detection:
        if not Path(coords_csv).exists():
            logging.error(
                "Missing detections and detection skipped — cannot run tracking."
            )
            return None

        detections_df = safe_load_csv(coords_csv, "detections")
        if detections_df is None:
            return None
        return detections_df

    # Run detection
    stack = load_movie(in_file)
    total_frames = stack.shape[0] if stack.ndim == 3 else 1
    img_shape = (stack.shape[-2], stack.shape[-1])

    logging.info("Frames: %d, Img shape: %s", total_frames, img_shape)
    logging.info("Starting detection...")

    detections_df = detect_sperm(
        stack,
        min_area=det_params["min_area"],
        max_area=det_params["max_area"],
        min_aspect=det_params["min_aspect"],
        max_aspect=det_params["max_aspect"],
        min_solidity=det_params["min_solidity"],
        threshold=det_params["threshold"],
        blur_radius=det_params["blur_radius"],
    )

    detections_df.to_csv(coords_csv, index=False)
    if not detections_df.empty:
        frames_count = detections_df["frame"].nunique()
    else:
        frames_count = 0
    logging.info(
        "Detection saved: %s (frames=%d, dets=%d)",
        coords_csv.name,
        frames_count,
        len(detections_df),
    )

    # Create detection overlay if enabled
    if params["detection"].get("visualize", True):
        try:
            create_detection_overlay(
                str(coords_csv),
                str(viz_movie),
                str(out_dir / f"{movie_name}_det_overlay.avi"),
                color=det_viz_params.get("color", "red"),
                radius=det_viz_params.get("point_radius", 5),
                fps=det_viz_params.get("fps", 30),
            )
        except Exception as e:
            logging.warning("Detection overlay failed: %s", e)
            logging.error("Detection overlay error details:", exc_info=True)

    return detections_df


def get_frame_info(
    in_file: Path,
    detections_df: Optional[pd.DataFrame],
    params: Dict[str, Dict[str, Any]],
) -> Tuple[int, Tuple[int, int]]:
    """Extract frame count and image shape information."""
    if detections_df is not None:
        if not detections_df.empty:
            total_frames = int(detections_df["frame"].max() + 1)
        else:
            total_frames = 0
        img_shape = (
            int(params["detection"].get("img_h", 512)),
            int(params["detection"].get("img_w", 512)),
        )
    else:
        stack = load_movie(in_file)
        total_frames = stack.shape[0] if stack.ndim == 3 else 1
        img_shape = (stack.shape[-2], stack.shape[-1])

    return total_frames, img_shape


def run_tracking_phase(
    detections_df: Optional[pd.DataFrame],
    total_frames: int,
    img_shape: Tuple[int, int],
    out_dir: Path,
    movie_name: str,
    tracker_cfg: TrackerConfig,
    post_cfg: TrajectoryPostConfig,
    viz_params: Dict[str, Any],
    viz_movie: Path,
    params: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    skip_tracking: bool = False,
) -> Optional[pd.DataFrame]:
    """Run tracking phase and return tracks DataFrame."""
    tracks_csv = out_dir / f"{movie_name}_trk_tracks.csv"

    if skip_tracking:
        if not Path(tracks_csv).exists():
            logging.error("Missing tracks and tracking skipped — cannot run analysis.")
            return None

        tracks_df = safe_load_csv(tracks_csv, "tracks")
        if tracks_df is None:
            return None
        return tracks_df

    if detections_df is None:
        return None

    logging.info("Starting tracking...")
    logging.debug(
        f"Post config - min_track_length: {post_cfg.min_track_length}, min_hits: {post_cfg.min_hits}"
    )

    # Check if two-stage tracking is enabled
    if (hasattr(args, "use_immotile_mining") and args.use_immotile_mining) or (
        params
        and "immotile_tracking" in params
        and params["immotile_tracking"].get("use_immotile_mining", False)
    ):
        logging.info("Two-stage tracking enabled: Stage 1 - Immotile mining")

        # Parse immotile configuration from JSON and CLI args
        immotile_config = parse_immotile_config(args, params)

        # Run two-stage tracking
        tracks_df, pipeline_summary = run_two_stage_tracking(
            detections_df,
            total_frames,
            img_shape,
            immotile_config,
            tracker_cfg,
            post_cfg,
        )

        # Log summary
        logging.info("Two-stage tracking summary:")
        for key, value in pipeline_summary.items():
            logging.info(f"  {key}: {value}")

        # Save immotile analysis if tracks were found
        # if 'immotile_tracks_found' in pipeline_summary and pipeline_summary['immotile_tracks_found'] > 0:
        #    immotile_analysis = analyze_immotile_tracks([])  # This will be populated from the summary
        #    logging.info("Immotile track analysis: %s", immotile_analysis)

    else:
        # Standard single-stage tracking
        logging.info("Standard single-stage tracking")
        tracks_df = run_tracking_with_configs(
            detections_df,
            total_frames,
            img_shape,
            tracker_cfg,
            post_cfg,
        )

    # Apply min_track_length filtering
    if not tracks_df.empty:
        original_tracks = tracks_df["tracking_id"].nunique()
        # Filter tracks by minimum length
        track_lengths = tracks_df.groupby("tracking_id").size()
        valid_tracks = track_lengths[track_lengths >= post_cfg.min_track_length].index
        tracks_df = tracks_df[tracks_df["tracking_id"].isin(valid_tracks)].copy()
        filtered_tracks = tracks_df["tracking_id"].nunique()
        if original_tracks != filtered_tracks:
            logging.info(
                "Filtered tracks by min_track_length=%d: %d -> %d tracks",
                post_cfg.min_track_length,
                original_tracks,
                filtered_tracks,
            )

    tracks_df.to_csv(tracks_csv, index=False)
    if not tracks_df.empty:
        tracks_count = tracks_df["tracking_id"].nunique()
    else:
        tracks_count = 0
    logging.info(
        "Tracking saved: %s (tracks=%d, rows=%d)",
        tracks_csv.name,
        tracks_count,
        len(tracks_df),
    )
    # Create tracking visualizations if enabled
    if params["tracking"].get("visualize", True):
        create_tracking_visualizations(
            tracks_df, out_dir, movie_name, viz_params, viz_movie
        )
    return tracks_df


def create_tracking_visualizations(
    tracks_df: pd.DataFrame,
    out_dir: Path,
    movie_name: str,
    viz_params: Dict[str, Any],
    viz_movie: Path,
) -> None:
    """Create tracking overlay and overview visualizations."""
    try:
        create_tracking_overlay(
            str(out_dir / f"{movie_name}_trk_tracks.csv"),
            str(viz_movie),
            str(out_dir / f"{movie_name}_trk_overlay.avi"),
            trail_length=viz_params.get("trail_length", 10),
            fps=viz_params.get("fps", 30),
            point_radius=viz_params.get("point_radius", 2),
            point_color=viz_params.get("point_color", "red"),
            track_width=viz_params.get("track_width", 2),
            track_color=viz_params.get("track_color", "red"),
        )
    except Exception as e:
        logging.warning("Tracking overlay failed: %s", e)
        logging.error("Tracking overlay error details:", exc_info=True)

    try:
        create_overview_frame(
            str(out_dir / f"{movie_name}_trk_tracks.csv"),
            str(viz_movie),
            str(out_dir / f"{movie_name}_trk_overview.png"),
            frame=viz_params.get("overview_frame", -2),
            track_color=viz_params.get("track_color", "green"),
            overview_track_width=viz_params.get("overview_track_width", 1.0),
            point_radius=viz_params.get("point_radius", 2),
            point_color=viz_params.get("point_color", "green"),
        )
    except Exception as e:
        logging.warning("Overview image failed: %s", e)
        logging.error("Overview image error details:", exc_info=True)


def run_analysis_phase(
    tracks_df: pd.DataFrame,
    out_dir: Path,
    movie_name: str,
    analysis_params: Dict[str, Any],
) -> None:
    """Run analysis phase and save results."""
    logging.info("Starting analysis...")

    try:
        results = analyze_tracks(
            tracks_df,
            pixel_size_um=analysis_params["pixel_size"],
            fps=analysis_params["fps"],
            motility_vcl_threshold=analysis_params["motility_vcl_threshold"],
            motility_vsl_threshold=analysis_params["motility_vsl_threshold"],
            motility_vap_threshold=analysis_params["motility_vap_threshold"],
            vap_window=analysis_params["vap_window"],
        )
        results_df, summary_stats = results

        # Save results and report
        analysis_file = out_dir / f"{movie_name}_ana_motility.csv"
        results_df.to_csv(analysis_file, index=False)

        report_file = out_dir / f"{movie_name}_ana_report.txt"
        report_content = generate_report(summary_stats, results_df, analysis_params)
        with open(report_file, "w") as fh:
            fh.write(report_content)

        logging.info("Analysis saved: %s", analysis_file.name)

    except Exception as e:
        logging.error("Analysis failed: %s", e)
        logging.error("Detailed analysis error:", exc_info=True)


def write_pipeline_summary(
    out_dir: Path,
    movie_name: str,
    start_time: float,
    det_params: Dict[str, Any],
    tracker_cfg: TrackerConfig,
    analysis_params: Dict[str, Any],
) -> None:
    """Write pipeline summary JSON file."""
    summary = {
        "movie": movie_name,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": time.time() - start_time,
        "params": {
            "detection": det_params,
            "tracking": tracker_cfg.__dict__,
            "analysis": analysis_params,
        },
    }
    with open(out_dir / f"{movie_name}_pipeline_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


def main():
    """Main entry point for the sperm motility analysis pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load configuration
    params = load_configuration(args)
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Validate input
    is_valid, input_files = validate_input_path(
        args.input_path, args.input_glob, args.recursive
    )
    if not is_valid or not input_files:
        print("No input files found or invalid path.")
        sys.exit(1)

    # Process each movie
    success_count = 0
    for in_file in input_files:
        # Calculate relative path to preserve directory structure
        relative_path = None
        if args.input_path.is_dir():
            try:
                relative_path = in_file.relative_to(args.input_path)
            except ValueError:
                # If in_file is not within args.input_path, use just the filename
                relative_path = Path(in_file.name)
        else:
            # If input_path is a file, don't use relative path (backward compatibility)
            relative_path = None

        if process_single_movie(in_file, args, params, log_level, relative_path):
            success_count += 1

    # Final summary
    logging.info(
        "Pipeline completed: %d/%d files processed successfully",
        success_count,
        len(input_files),
    )
    if success_count < len(input_files):
        sys.exit(1)


if __name__ == "__main__":
    main()
