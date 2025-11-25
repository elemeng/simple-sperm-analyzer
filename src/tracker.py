#!/usr/bin/env python3
"""
Sperm Trajectory Tracking CLI  –  REFINED VERSION
Only *reliable* (confirmed + length-filtered) tracks are exported / visualised.
"""

import argparse
import pandas as pd
import numpy as np
import cv2
import tifffile as tiff
import sys
import json
import os
import fnmatch
from pathlib import Path

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment

try:
    from core import compute_curvature_signature, load_tif_stack
except ImportError:
    sys.exit("Error: core.py not found.")

try:
    from filterpy.kalman import KalmanFilter

    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    print("Warning: filterpy not available – install with:  pip install filterpy")


# ---------- data classes ----------
@dataclass
class Detection:
    frame: int
    x: float
    y: float
    area: float
    solidity: float
    curvature_signature: Optional[np.ndarray] = None
    contour: Optional[np.ndarray] = None


@dataclass
class TrackState:
    track_id: int
    frame: int
    x: float
    y: float
    vx: float
    vy: float
    state: str
    missing_count: int
    track_length: int
    hits: int


# ---------- single track ----------
class SpermTrack:
    def __init__(self, track_id: int, first: Detection, use_kalman: bool):
        self.id = track_id
        self.history: List[Detection] = [first]
        self.features = {
            "mean_curvature": first.curvature_signature.copy()
            if first.curvature_signature is not None
            else None,
            "mean_area": first.area,
        }
        self.missing_count = 0
        self.hits = 1
        self.min_hits = 3  # overwritten by tracker
        self.use_kalman = use_kalman
        self.is_eliminated = False
        if use_kalman and FILTERPY_AVAILABLE:
            self.kf = self._init_kalman(first.x, first.y)
        else:
            self.last_pos = np.array([first.x, first.y], float)

    # --------------- internal helpers ---------------
    def _init_kalman(self, x, y):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([x, y, 0.0, 0.0])
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.R *= 10.0
        kf.P[2:, 2:] *= 1000.0
        kf.Q[2:, 2:] *= 0.01
        return kf

    def predict(self) -> Tuple[float, float, float]:
        if self.use_kalman and FILTERPY_AVAILABLE:
            self.kf.predict()
            pos = self.kf.x[:2]
            unc = np.trace(self.kf.P[:2, :2]) ** 0.5
        else:
            pos = self.last_pos
            unc = 60.0
        return float(pos[0]), float(pos[1]), unc

    def update(self, det: Optional[Detection]):
        if det:
            if self.use_kalman and FILTERPY_AVAILABLE:
                self.kf.update([det.x, det.y])
            else:
                self.last_pos[:] = [det.x, det.y]
            self.history.append(det)
            self.missing_count = 0
            self.hits += 1
            # exponential update of appearance
            if (
                self.features["mean_curvature"] is not None
                and det.curvature_signature is not None
            ):
                self.features["mean_curvature"] = (
                    0.9 * self.features["mean_curvature"]
                    + 0.1 * det.curvature_signature
                )
            self.features["mean_area"] = (
                0.9 * self.features["mean_area"] + 0.1 * det.area
            )
        else:
            self.missing_count += 1

    def is_dead(self, max_age: int) -> bool:
        return self.missing_count > max_age

    def eliminate(self):
        self.is_eliminated = True

    def get_state(self) -> Dict:
        if not self.history:
            return {}
        latest = self.history[-1]
        state = "confirmed" if self.hits >= self.min_hits else "tentative"
        vx = vy = 0.0
        if self.use_kalman and FILTERPY_AVAILABLE:
            vx, vy = self.kf.x[2], self.kf.x[3]
        return TrackState(
            self.id,
            latest.frame,
            latest.x,
            latest.y,
            vx,
            vy,
            state,
            self.missing_count,
            len(self.history),
            self.hits,
        ).__dict__


# ---------- tracker ----------
class SpermTracker:
    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        max_distance: float = 80.0,
        appearance_thresh: float = 0.85,
        use_kalman: bool = True,
        appearance_verify_dist: float = 50.0,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.appearance_thresh = appearance_thresh
        self.use_kalman = use_kalman
        self.app_verify_dist = appearance_verify_dist
        self.tracks: List[SpermTrack] = []
        self.next_id = 0

    # --------------- main update ---------------
    def update(self, dets: List[Detection], frame: int) -> List[SpermTrack]:
        # predict
        preds = [(t.predict()[0], t.predict()[1]) for t in self.tracks]
        uncs = [t.predict()[2] for t in self.tracks]

        # cost matrix
        cost = self._cost_matrix(preds, uncs, dets)
        if cost.size:
            trk_idx, det_idx = linear_sum_assignment(cost)
        else:
            trk_idx = det_idx = []

        unmatched_trk = set(range(len(self.tracks)))
        unmatched_det = set(range(len(dets)))

        # assignments
        for t, d in zip(trk_idx, det_idx):
            if cost[t, d] < self.max_distance:
                self.tracks[t].update(dets[d])
                unmatched_trk.discard(t)
                unmatched_det.discard(d)

        # missing / dead
        for t in unmatched_trk:
            self.tracks[t].update(None)
            if self.tracks[t].is_dead(self.max_age):
                self.tracks[t].eliminate()

        # new tracks
        for d in unmatched_det:
            tr = SpermTrack(self.next_id, dets[d], self.use_kalman)
            tr.min_hits = self.min_hits
            self.tracks.append(tr)
            self.next_id += 1

        # keep active only
        self.tracks = [t for t in self.tracks if not t.is_eliminated]
        return self.tracks

    # --------------- cost + appearance ---------------
    def _cost_matrix(self, poss: List[Tuple], uncs: List[float], dets: List[Detection]):
        n_trk, n_det = len(poss), len(dets)
        C = np.full((n_trk, n_det), 1e6, float)
        for i, ((tx, ty), unc) in enumerate(zip(poss, uncs)):
            for j, d in enumerate(dets):
                dist = np.hypot(tx - d.x, ty - d.y)
                if dist > self.max_distance:
                    continue
                score = dist / (unc + 1e-6)
                if (
                    dist > self.app_verify_dist
                    and self.tracks[i].features["mean_curvature"] is not None
                    and d.curvature_signature is not None
                ):
                    if not self._appearance_match(self.tracks[i].features, d):
                        continue
                C[i, j] = score
        return C

    def _appearance_match(self, feat: Dict, det: Detection) -> bool:
        corr = np.corrcoef(feat["mean_curvature"], det.curvature_signature)[0, 1]
        return corr > self.appearance_thresh


# ---------- IO helpers ----------
def load_detections(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_pickle(path) if path.suffix == ".pkl" else pd.read_csv(path)
    req = ["frame", "x", "y", "area", "solidity"]
    if missing := set(req) - set(df.columns):
        raise ValueError(f"Missing {missing}")
    if "contour" in df.columns:
        df["contour"] = df["contour"].apply(
            lambda s: np.array(
                [list(map(int, p.split(","))) for p in s.split(";")]
            ).reshape(-1, 1, 2)
            if pd.notna(s)
            else None
        )
        df["curvature_signature"] = df["contour"].apply(
            lambda c: compute_curvature_signature(c) if c is not None else None
        )
    else:
        df["curvature_signature"] = None
    return df


# ---------- tracking pipeline ----------
def track_movie(
    df: pd.DataFrame,
    use_kalman: bool,
    max_age: int,
    min_hits: int,
    max_dist: float,
    app_thresh: float,
) -> pd.DataFrame:
    frames = sorted(df.frame.unique())
    frame_dets = {
        f: [
            Detection(
                row.frame,
                row.x,
                row.y,
                row.area,
                row.solidity,
                row.curvature_signature,
                row.get("contour"),
            )
            for _, row in df[df.frame == f].iterrows()
        ]
        for f in frames
    }

    tracker = SpermTracker(max_age, min_hits, max_dist, app_thresh, use_kalman)
    rows = []
    for f in frames:
        for det in frame_dets[f]:
            if det.curvature_signature is None and det.contour is not None:
                det.curvature_signature = compute_curvature_signature(det.contour)
        for trk in tracker.update(frame_dets[f], f):
            rows.append(trk.get_state())
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=list(TrackState.__annotations__.keys()))
    )


# ---------- export ----------
def save_summary(tracks: pd.DataFrame, out_dir: Path, in_file: str, params: dict):
    with open(out_dir / f"{Path(in_file).stem}_tracking_log.txt", "w") as fh:
        fh.write("Sperm Tracking Summary\n")
        fh.write("=====================\n")
        fh.write(f"Input: {in_file}\n")
        fh.write(f"Date: {pd.Timestamp.now()}\n\n")
        fh.write("Parameters:\n")
        fh.write(json.dumps(params, indent=2))
        fh.write("\n\n")
        if tracks.empty:
            fh.write("No tracks found.\n")
            return
        fh.write(f"Total frames: {tracks.frame.nunique()}\n")
        fh.write(f"Total confirmed tracks: {tracks.track_id.nunique()}\n")
        fh.write(f"Total detections linked: {len(tracks)}\n")
        lens = tracks.groupby("track_id").size()
        fh.write(f"Mean track length: {lens.mean():.1f}\n")
        fh.write(f"Tracks >30 frames: {(lens > 30).sum()}\n")


# ---------- visualization ----------

# ----------  global colour table  ----------
COLOR_MAP = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
}


def _resolve_color(color: str | tuple) -> tuple:
    """Return BGR tuple for string key or pass-through tuple."""
    if isinstance(color, str):
        return COLOR_MAP.get(color.lower(), (0, 0, 255))
    return color  # already a tuple


# ----------  make_movie  ----------
def make_movie(
    movie_dir: str,
    tracks: pd.DataFrame,
    tiff_path: Path,
    trail_length: int = 10,
    fps: int = 30,
    point_radius: int = 4,
    point_color: str | tuple = "red",
    track_width: int = 2,
    track_color: str | tuple = "red",
):
    print("Creating trajectory movie (TIFF + MP4) …")

    # --- open stack ---
    stack = load_tif_stack(movie_dir)
    if stack.ndim == 2:
        stack = stack[None, ...]
    T, H, W = stack.shape[:3]

    # --- resolve colours once ---
    pt_color = _resolve_color(point_color)
    tr_color = _resolve_color(track_color)

    by_frame = {f: g for f, g in tracks.groupby("frame")}

    # --- writers ---
    tiff_w = tiff.TiffWriter(tiff_path, bigtiff=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    mp4_path = str(tiff_path.with_suffix(".mp4"))
    writer = cv2.VideoWriter(mp4_path, fourcc, float(fps), (W, H))

    # --- frame loop ---
    for idx, img in enumerate(stack):
        # 1. RGB canvas
        if img.ndim == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            vis = img[..., :3]
        vis = np.ascontiguousarray(vis, dtype=np.uint8)

        # 2. draw trails (fade) ---------------- TRACK COLOR
        if idx in by_frame:
            for _, row in by_frame[idx].iterrows():
                tid = int(row.track_id)
                hist = tracks[tracks.track_id == tid]
                pts = (
                    hist[hist.frame <= idx]
                    .tail(trail_length)[["x", "y"]]
                    .astype(np.int32)
                    .values
                )
                for i in range(len(pts) - 1):
                    alpha = (i + 1) / len(pts)
                    thick = max(1, int(alpha * track_width))
                    cv2.line(
                        vis,
                        tuple(pts[i]),
                        tuple(pts[i + 1]),
                        tuple(int(c * alpha) for c in tr_color),
                        thick,
                    )

                # 3. current spot ---------------- POINT COLOR / RADIUS
                x, y = int(row.x), int(row.y)
                cv2.circle(
                    vis, (x, y), radius=point_radius, color=pt_color, thickness=-1
                )
                cv2.putText(
                    vis,
                    f"{tid}",
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # 4. write outputs
        tiff_w.write(vis)
        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # --- finalise ---
    writer.release()
    tiff_w.close()
    print(f"TIFF saved: {tiff_path}")
    print(f"MP4  saved: {mp4_path}")


# ----------  extra export  ----------
def save_overview_frame(
    tracks: pd.DataFrame,
    movie_dir: str,
    out_path: Path,
    frame: int = 0,
    track_color: str | tuple = "green",
    overview_track_width: float = 1.0,
    point_radius: int = 1,
    point_color: str | tuple = "green",
    dpi: int = 300,
):
    """
    Plot every reliable track on a *single* frame.
    The whole future trajectory is overlaid on that frame.
    frame: frame number to plot, or -2 to use last frame
    """
    import matplotlib.pyplot as plt
    import cv2

    # resolve colours
    tr_color = (
        np.array(_resolve_color(track_color), dtype=float)[::-1] / 255.0
    )  # RGB 0-1
    pt_color = (
        np.array(_resolve_color(point_color), dtype=float)[::-1] / 255.0
    )  # RGB 0-1

    # load the chosen frame
    stack = load_tif_stack(movie_dir)
    if stack.ndim == 2:
        stack = stack[None, ...]
    
    # Ensure frame number is valid
    if frame >= len(stack):
        frame = len(stack) - 1  # Use last frame if requested frame is beyond available frames
    elif frame < 0:
        frame = 0  # Use first frame if negative number provided
    
    img = stack[frame]
    h, w = img.shape[:2]

    # prepare RGB image for matplotlib
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = img[..., :3]

    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    plt.imshow(rgb, cmap="gray")
    plt.axis("off")

    # ---- draw ----
    if frame == len(stack) - 1:
        # For last frame: show complete trajectories (all frames)
        for tid, g in tracks.groupby("track_id"):
            # Show the complete trajectory
            complete_traj = g.sort_values("frame")
            plt.plot(
                complete_traj.x,
                complete_traj.y,
                color=tr_color,
                lw=overview_track_width,
                alpha=0.8,
            )
            # start dot
            start = complete_traj.iloc[0]
            plt.scatter(
                start.x, start.y, s=point_radius**2, c=[pt_color], edgecolors="none"
            )
    else:
        # For other frames: show trajectory from this frame onward (original behavior)
        for tid, g in tracks[tracks.frame >= frame].groupby("track_id"):
            # full chain from *this* frame onward
            fut = g[g.frame >= frame].sort_values("frame")
            plt.plot(
                fut.x,
                fut.y,
                color=tr_color,
                lw=overview_track_width,  # ← use this instead of track_width
                alpha=0.8,
            )
            # start dot
            start = fut.iloc[0]
            plt.scatter(
                start.x, start.y, s=point_radius**2, c=[pt_color], edgecolors="none"
            )

    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Overview frame saved: {out_path}")


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(
        description="Reliable sperm tracking – confirmed tracks only"
    )
    p.add_argument(
        "detections_file", nargs="?", help="CSV or PKL file (for single file mode)"
    )
    p.add_argument(
        "--input_dir",
        help="Input directory for batch processing (alternative to detections_file)",
    )
    p.add_argument(
        "--output_dir", help="Output directory for results (required for batch mode)"
    )

    # Batch processing options
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process directories recursively. Default: False",
    )
    p.add_argument(
        "--glob",
        default="*_coords.csv",
        help="Glob pattern for CSV filtering (default: *_coords.csv)",
    )

    # File name matching parameters
    p.add_argument(
        "--cut_suffix_movie",
        type=int,
        default=4,
        help="Characters to cut from end of movie filename for matching (default: 4)",
    )
    p.add_argument(
        "--cut_suffix_csv",
        type=int,
        default=11,
        help="Characters to cut from end of CSV filename for matching (default: 11 = '_coords.csv')",
    )
    p.add_argument(
        "--cut_prefix_movie",
        type=int,
        default=0,
        help="Characters to cut from start of movie filename for matching (default: 0)",
    )
    p.add_argument(
        "--cut_prefix_csv",
        type=int,
        default=0,
        help="Characters to cut from start of CSV filename for matching (default: 0)",
    )

    # Movie path can now be a folder for batch processing
    p.add_argument(
        "--movie_dir",
        help="Path to movie file or folder containing movies for visualization",
    )
    # tracking
    p.add_argument(
        "--no-kalman", action="store_true", help="disable Kalman. Default: False"
    )
    p.add_argument(
        "--max-age", type=int, default=5, help="max age (frames), Default: 5"
    )
    p.add_argument(
        "--min-hits", type=int, default=3, help="confirm threshold, Default: 3"
    )
    p.add_argument(
        "--max-search-distance",
        type=int,
        default=80,
        help="max search distance (pixels), Default: 80",
    )
    p.add_argument(
        "--appearance-thresh",
        type=float,
        default=0.85,
        help="appearance threshold, Default: 0.85",
    )
    # filtering
    p.add_argument(
        "--min-track-length",
        type=int,
        default=10,
        help="minimum complete trajectory length (frames) for export/visualisation. Default 10",
    )
    # visualisation
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--movie-path", help="required when --visualize")
    p.add_argument(
        "--point-radius", type=int, default=2, help="point radius (pixels), Default: 2"
    )
    p.add_argument("--point-color", default="red", help="point colour, Default: red")
    p.add_argument("--track-color", default="red", help="track colour, Default: red")
    p.add_argument(
        "--trail-length",
        type=int,
        default=10,
        help="track plot length (frames), Default: 10",
    )
    p.add_argument(
        "--track-width", type=int, default=2, help="track width (pixels), Default: 2"
    )
    p.add_argument("--fps", type=int, default=30, help="MP4 frame rate, Default: 30")
    p.add_argument(
        "--overview-frame",
        type=int,
        default=-2,
        help="frame to plot all reliable tracks on (paper figure); "
        "set to -1 to skip; set to -2 to use last frame (default: -2)",
    )
    p.add_argument(
        "--overview-track-width",
        type=float,
        default=0.1,
        help="track line width (matplotlib pts) for overview figure.  Default 0.1",
    )

    args = p.parse_args()

    # Validate arguments
    if not args.detections_file and not args.input_dir:
        p.error("Either detections_file or --input_dir must be specified")

    if args.detections_file and args.input_dir:
        p.error("Cannot specify both detections_file and --input_dir")

    if args.input_dir and not args.output_dir:
        p.error("--output_dir is required when using --input_dir")

    # Determine processing mode: single file or batch
    if args.detections_file:
        # Single file mode - output_dir is optional, defaults to CSV location
        if not args.output_dir:
            args.output_dir = "auto"  # Will use CSV file's directory
        return process_single_file(args)
    else:
        # Batch processing mode
        return process_batch_files(args)


def extract_base_name(filename: str, cut_prefix: int, cut_suffix: int) -> str:
    """Extract base name by cutting prefix and suffix characters from filename only"""
    # Get just the filename part, removing any directory path
    basename = filename.rsplit("/")[-1]

    if cut_prefix > 0:
        basename = basename[cut_prefix:]
    if cut_suffix > 0:
        basename = basename[:-cut_suffix]
    return basename


def find_matching_movie(csv_file: Path, movie_folder: Path, args) -> Optional[Path]:
    """Find matching movie file for a given CSV file"""
    if not movie_folder.exists():
        return None

    # Extract base name from CSV file name (with extension)
    csv_name = csv_file.name
    csv_base = extract_base_name(csv_name, args.cut_prefix_csv, args.cut_suffix_csv)

    # Look for matching movie files
    movie_extensions = [".tif", ".TIF", ".tiff", ".TIFF"]

    if movie_folder.is_dir():
        # Search in folder
        for movie_file in movie_folder.iterdir():
            if movie_file.suffix in movie_extensions:
                movie_name = movie_file.name
                movie_base = extract_base_name(
                    movie_name, args.cut_prefix_movie, args.cut_suffix_movie
                )
                if movie_base == csv_base:
                    return movie_file
    else:
        # Single movie file - check if it matches
        if movie_folder.suffix in movie_extensions:
            movie_name = movie_folder.name
            movie_base = extract_base_name(
                movie_name, args.cut_prefix_movie, args.cut_suffix_movie
            )
            if movie_base == csv_base:
                return movie_folder

    return None


def process_single_file(
    args, csv_file: Optional[Path] = None, movie_file: Optional[Path] = None, preserve_structure: bool = False
):
    """Process a single CSV file"""
    if csv_file is None:
        csv_file = Path(args.detections_file)

    if not csv_file.exists():
        print(f"Error: CSV file {csv_file} not found")
        return 1

    # Determine output directory
    if args.output_dir == "auto":
        # Single file mode: use CSV file's directory
        out_dir = csv_file.parent
    elif preserve_structure and args.input_dir:
        # Batch mode with structure preservation: maintain relative path structure
        input_path = Path(args.input_dir)
        try:
            # Get relative path from input_dir to csv_file
            rel_path = csv_file.relative_to(input_path)
            # Remove the filename to get just the directory structure
            rel_dir = rel_path.parent
            # Create output directory with same structure
            out_dir = Path(args.output_dir) / rel_dir
        except ValueError:
            # If csv_file is not relative to input_dir, use parent directory
            out_dir = Path(args.output_dir) / csv_file.parent.name
    else:
        # Batch mode without structure preservation: flat output
        out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_detections(csv_file)
        print(
            f"Loaded {len(df)} detections across {df.frame.nunique()} frames from {csv_file}"
        )

        tracks = track_movie(
            df,
            use_kalman=not args.no_kalman,
            max_age=args.max_age,
            min_hits=args.min_hits,
            max_dist=args.max_search_distance,
            app_thresh=args.appearance_thresh,
        )

        # ---- keep only confirmed AND long enough ----
        confirmed_ids = tracks[tracks.state == "confirmed"].track_id.unique()
        # export/visualise the FULL track (tentative phase + confirmed phase)
        tracks = tracks[tracks.track_id.isin(confirmed_ids)]

        # ---- export ----
        stem = csv_file.stem
        if not tracks.empty:
            tracks.to_csv(out_dir / f"{stem}_confirmed_tracks.csv", index=False)
            print(
                f"Confirmed tracks (≥{args.min_track_length} frames) saved for {csv_file.name}"
            )
        else:
            print(f"No tracks passed the reliability filter for {csv_file.name}")

        params = {
            k: getattr(args, k)
            for k in [
                "no_kalman",
                "max_age",
                "min_hits",
                "max_search_distance",
                "appearance_thresh",
                "min_track_length",
            ]
        }
        save_summary(tracks, out_dir, str(csv_file), params)

        # ---- visualization ----
        if args.visualize:
            # Find matching movie file
            movie_dir = None
            if movie_file:
                movie_dir = movie_file
            elif args.movie_dir:
                movie_folder = Path(args.movie_dir)
                movie_dir = find_matching_movie(csv_file, movie_folder, args)

            if movie_dir and movie_dir.exists():
                tiff_out = out_dir / f"{stem}_trajectories.tif"
                make_movie(
                    str(movie_dir),
                    tracks,
                    tiff_out,
                    trail_length=args.trail_length,
                    fps=args.fps,
                    point_radius=args.point_radius,
                    point_color=args.point_color,
                    track_width=args.track_width,
                    track_color=args.track_color,
                )
                # ---- overview figure ----
                ov_frame = args.overview_frame
                if ov_frame == -1:
                    # Skip overview frame
                    pass
                else:
                    # Use specified frame, or last frame if -2 (default)
                    if ov_frame == -2:
                        # Use last frame
                        frame_num = len(tracks.frame.unique()) - 1
                    else:
                        # Use specified frame number
                        frame_num = ov_frame
                    
                    ov_path = out_dir / f"{stem}_overview_frame{ov_frame}.png"
                    save_overview_frame(
                        tracks,
                        str(movie_dir),
                        ov_path,
                        frame=frame_num,
                        track_color=args.track_color,
                        overview_track_width=args.overview_track_width,
                        point_radius=args.point_radius,
                        point_color=args.point_color,
                        dpi=300,
                    )
            else:
                print(
                    f"Warning: No matching movie file found for visualization of {csv_file.name}"
                )
                if args.movie_dir:
                    print(f"  Searched in: {args.movie_dir}")

        return 0

    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        return 1


def process_batch_files(args):
    """Process multiple CSV files in batch mode"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    # Find CSV files in input_dir, not output_dir
    input_path = Path(args.input_dir)
    if args.recursive:
        csv_files = list(input_path.rglob(args.glob))
    else:
        csv_files = list(input_path.glob(args.glob))

    # Also check for PKL files if glob pattern allows
    if "*.csv" in args.glob:
        pkl_pattern = args.glob.replace("*.csv", "*.pkl")
        if args.recursive:
            csv_files.extend(list(input_path.rglob(pkl_pattern)))
        else:
            csv_files.extend(list(input_path.glob(pkl_pattern)))

    # If no files found and we're in recursive mode, try searching all subdirectories
    if not csv_files and args.recursive:
        print(f"No files found in {input_path}, searching all subdirectories...")
        csv_files = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if fnmatch.fnmatch(file, args.glob):
                    csv_files.append(Path(root) / file)
                elif "*.csv" in args.glob and file.endswith(".pkl"):
                    csv_files.append(Path(root) / file)

    if not csv_files:
        print(
            f"Error: No files matching pattern {args.glob} found in {input_path}"
        )
        return 1

    # Remove duplicates and sort
    csv_files = sorted(list(set(csv_files)))

    print(f"Found {len(csv_files)} CSV files for batch tracking")
    print(f"Input directory: {input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"File pattern: {args.glob}")
    print(f"Recursive: {args.recursive}")

    # Prepare movie folder if provided
    movie_folder = None
    if args.movie_dir:
        movie_folder = Path(args.movie_dir)
        if movie_folder.exists() and movie_folder.is_dir():
            print(f"Movie folder: {movie_folder}")
        else:
            print(f"Single movie file: {movie_folder}")

    # Process files
    start_time = time.time()
    results = []

    print(f"\nStarting batch processing...")
    for i, csv_file in enumerate(csv_files, 1):
        try:
            # Find matching movie file if visualization is enabled
            movie_file = None
            if args.visualize and movie_folder:
                movie_file = find_matching_movie(csv_file, movie_folder, args)
                if movie_file:
                    print(
                        f"[{i}/{len(csv_files)}] Found matching movie: {movie_file.name}"
                    )
                else:
                    print(
                        f"[{i}/{len(csv_files)}] No matching movie found for {csv_file.name}"
                    )

            result = process_single_file(args, csv_file, movie_file, preserve_structure=True)
            results.append(
                {"file": str(csv_file), "success": result == 0, "error": None}
            )
            status = "✓" if result == 0 else "✗"
            print(f"[{i}/{len(csv_files)}] {status} {csv_file.name}")

        except Exception as e:
            results.append({"file": str(csv_file), "success": False, "error": str(e)})
            print(f"[{i}/{len(csv_files)}] ✗ {csv_file.name} - Error: {e}")

    # Generate batch summary
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\n{'=' * 60}")
    print(f"Batch Tracking Summary")
    print(f"{'=' * 60}")
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Average time per file: {elapsed_time / len(results):.1f} seconds")

    if failed > 0:
        print(f"\nFailed files:")
        for result in results:
            if not result["success"]:
                print(f"  ✗ {Path(result['file']).name}")
                if result["error"]:
                    print(f"    Error: {result['error']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    main()
