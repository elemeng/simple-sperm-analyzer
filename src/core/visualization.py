# visualizer.py
from __future__ import annotations
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union
import matplotlib.pyplot as plt

from .common.io import load_movie

Color = Union[str, Tuple[int, int, int]]


class Visualizer:
    """
    Unified API for sperm-tracking visualisations.
    One instance can be re-used for any number of movies / CSVs or DataFrames.
    """

    # ---------- class-level constants ----------
    COLOR_MAP = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "white": (255, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "gray": (128, 128, 128),
        "black": (0, 0, 0),
    }

    # ---------- helpers ----------
    @staticmethod
    def _resolve_color(color: Color) -> Tuple[int, int, int]:
        if isinstance(color, str):
            return Visualizer.COLOR_MAP.get(color.lower(), Visualizer.COLOR_MAP["red"])
        if isinstance(color, tuple) and len(color) == 3:
            return color
        raise ValueError(f"Invalid color: {color}")

    @staticmethod
    def _validate_columns(
        df: pd.DataFrame, required: List[str], name: str = "CSV"
    ) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    @staticmethod
    def ensure_rgb(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return np.squeeze(frame, axis=2)
        if frame.ndim == 3 and frame.shape[2] >= 3:
            return frame[..., :3]
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    # ---------- low-level painters ----------
    @staticmethod
    def add_markers_to_frame(
        frame: np.ndarray,
        coords: List[Tuple[float, float]],
        color: Tuple[int, int, int],
        radius: int = 3,
    ) -> np.ndarray:
        if frame.ndim != 2:
            raise ValueError("Input frame must be grayscale (2D)")
        out = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        for x, y in coords:
            cv2.circle(out, (int(x), int(y)), radius, color, -1)
        return out

    # ---------- high-level visualisations ----------
    def draw_detections(
        self,
        detection_csv: str | Path | pd.DataFrame,  # Accept DataFrame as well
        movie_path: str | Path,
        output_path: str | Path,
        color: Color = "red",
        radius: int = 5,
        fps: int = 30,
    ) -> Path:
        """
        Create AVI with detection dots overlaid (using MJPG/JPEG encoding).
        Can accept either CSV path or DataFrame directly.
        Optimized for performance.
        """
        if isinstance(detection_csv, pd.DataFrame):
            df = detection_csv
        else:
            try:
                df = pd.read_csv(detection_csv)
            except pd.errors.EmptyDataError:
                # If CSV is empty, create an empty DataFrame with the required columns
                df = pd.DataFrame(columns=["frame", "x", "y"])

        # Check if DataFrame has required columns, even if empty
        if not df.empty:
            self._validate_columns(df, ["frame", "x", "y"], "Detection CSV")

        stack = load_movie(movie_path)

        # Normalize stack shape -> (T, H, W)
        if stack.ndim == 2:
            stack = stack[None, ...]
        elif stack.ndim == 4 and stack.shape[-1] == 1:
            stack = stack.squeeze(-1)
        if stack.ndim != 3:
            raise ValueError(f"Unexpected movie shape: {stack.shape}")

        T, H, W = stack.shape
        color_bgr = self._resolve_color(color)
        out = Path(output_path)
        writer = cv2.VideoWriter(
            str(out),
            cv2.VideoWriter_fourcc(*"MJPG"),  # Use MJPG (Motion JPEG) codec for AVI
            fps,
            (W, H),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {out}")

        # Pre-group detections by frame for faster access
        by_frame = {f: g[["x", "y"]].values for f, g in df.groupby("frame")}

        # Pre-allocate arrays to avoid repeated allocations
        for f_idx in range(T):
            coords = by_frame.get(f_idx, [])
            if len(coords) > 0:
                # Create RGB frame from grayscale
                frame = stack[f_idx]
                if frame.ndim == 2:
                    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    vis = frame[..., :3].astype(np.uint8)

                # Vectorized circle drawing using numpy operations
                for x, y in coords:
                    cv2.circle(vis, (int(x), int(y)), int(radius), color_bgr, -1)
            else:
                # Frame has no detections, just ensure RGB format
                frame = stack[f_idx]
                if frame.ndim == 2:
                    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    vis = frame[..., :3].astype(np.uint8)

            writer.write(
                vis if vis.shape[-1] == 3 else cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            )
        writer.release()
        if not out.exists():
            raise RuntimeError(f"AVI not created at {out}")
        return out

    def draw_tracks(
        self,
        tracks_csv: str | Path | pd.DataFrame,  # Accept DataFrame as well
        movie_path: str | Path,
        output_path: str | Path,
        *,
        trail_length: int = 10,
        fps: int = 30,
        point_radius: int = 4,
        point_color: Color = "red",
        track_width: int = 2,
        track_color: Color = "red",
    ) -> Path:
        """
        Create AVI with tracking tails + IDs (using MJPG/JPEG encoding).
        Can accept either CSV path or DataFrame directly.
        Optimized for performance.
        """
        if isinstance(tracks_csv, pd.DataFrame):
            trk = tracks_csv
        else:
            try:
                trk = pd.read_csv(tracks_csv)
            except pd.errors.EmptyDataError:
                # If CSV is empty, create an empty DataFrame with the required columns
                trk = pd.DataFrame(columns=["track_id", "frame", "x", "y"])

        # Validate columns, but don't require 'state' column
        required_cols = ["track_id", "frame", "x", "y"]
        if not trk.empty and "state" in trk.columns:
            required_cols.append("state")
            # Only filter by state if the column exists
            trk = trk[trk["state"] == "confirmed"].copy()
        else:
            trk = trk.copy()

        self._validate_columns(trk, required_cols, "Tracking CSV")

        # Pre-compute sorted tracks by track_id for efficiency
        # Convert to numpy arrays for faster access
        by_track = {}
        for tid, g in trk.groupby("track_id"):
            sorted_g = g.sort_values("frame")
            by_track[tid] = {
                "frames": sorted_g["frame"].values,
                "x": sorted_g["x"].values,
                "y": sorted_g["y"].values,
            }

        # Pre-group tracks by frame for faster access
        by_frame = {f: g for f, g in trk.groupby("frame")}

        stack = load_movie(movie_path)
        if stack.ndim == 2:
            stack = stack[None, ...]
        if stack.ndim == 4 and stack.shape[-1] == 1:
            stack = stack.squeeze(-1)
        if stack.ndim == 3:
            T, H, W = stack.shape
        else:
            raise ValueError(f"Unexpected stack shape: {stack.shape}")

        # Convert BGR colors to RGB for drawing on RGB canvas
        pt_color_bgr = self._resolve_color(point_color)
        tr_color_bgr = self._resolve_color(track_color)
        # Convert BGR to RGB: (B, G, R) -> (R, G, B)
        pt_color_rgb = (pt_color_bgr[2], pt_color_bgr[1], pt_color_bgr[0])
        tr_color_rgb = (tr_color_bgr[2], tr_color_bgr[1], tr_color_bgr[0])

        out = Path(output_path)
        writer = cv2.VideoWriter(
            str(out),
            cv2.VideoWriter_fourcc(*"MJPG"),  # Use MJPG (Motion JPEG) codec for AVI
            float(fps),
            (W, H),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {out}")

        # Pre-allocate array for efficiency
        for idx, img in enumerate(stack):
            # Convert frame to RGB only once
            vis = self.ensure_rgb(img).astype(np.uint8)

            if idx in by_frame:
                # Process all tracks in this frame
                frame_tracks = by_frame[idx]
                for _, row in frame_tracks.iterrows():
                    tid = int(row.track_id)
                    track_data = by_track.get(tid)
                    if track_data is None:
                        continue

                    # Get trajectory points up to current frame
                    frame_indices = track_data["frames"] <= idx
                    if not np.any(frame_indices):
                        continue

                    x_vals = track_data["x"][frame_indices]
                    y_vals = track_data["y"][frame_indices]

                    # Get the last trail_length points
                    pts_x = x_vals[-trail_length:]
                    pts_y = y_vals[-trail_length:]

                    if len(pts_x) > 1:
                        pts = np.column_stack(
                            [pts_x.astype(np.int32), pts_y.astype(np.int32)]
                        )

                        # Draw track trail with optimized line drawing
                        num_points = len(pts)
                        for i in range(num_points - 1):
                            alpha = (i + 1) / num_points
                            thick = max(1, int(alpha * track_width))
                            # Apply alpha to RGB color
                            faded_color = tuple(int(c * alpha) for c in tr_color_rgb)
                            cv2.line(
                                vis,
                                tuple(pts[i]),
                                tuple(pts[i + 1]),
                                faded_color,
                                thick,
                            )

                    # Draw point
                    x, y = int(row.x), int(row.y)
                    cv2.circle(vis, (x, y), int(point_radius), pt_color_rgb, -1)
                    # Draw point ID label
                    # cv2.putText(
                    #    vis,
                    #    f"{tid}",
                    #    (x + 6, y - 6),
                    #    cv2.FONT_HERSHEY_SIMPLEX,
                    #    0.5,
                    #    (255, 255, 255),
                    #    1,
                    # )
            # Convert RGB to BGR for OpenCV VideoWriter
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            writer.write(vis_bgr)
        writer.release()
        if not out.exists():
            raise RuntimeError(f"AVI not created at {out}")
        return out

    def draw_overview(
        self,
        tracks_csv: str | Path | pd.DataFrame,  # Accept DataFrame as well
        movie_path: str | Path,
        output_path: str | Path,
        *,
        frame: int = 0,
        track_color: Color = "green",
        overview_track_width: float = 1.0,
        point_radius: int = 1,
        point_color: Color = "green",
        dpi: int = 300,
    ) -> Path:
        """
        Create PNG overview with all trajectories.
        Can accept either CSV path or DataFrame directly.
        Optimized for performance.
        """
        if isinstance(tracks_csv, pd.DataFrame):
            trk = tracks_csv
        else:
            try:
                trk = pd.read_csv(tracks_csv)
            except pd.errors.EmptyDataError:
                # If CSV is empty, create an empty DataFrame with the required columns
                trk = pd.DataFrame(columns=["track_id", "frame", "x", "y"])

        # Handle state column if it exists, otherwise use all tracks
        if not trk.empty and "state" in trk.columns:
            trk = trk[trk["state"] == "confirmed"].copy()
        else:
            trk = trk.copy()

        if not trk.empty:
            self._validate_columns(trk, ["track_id", "frame", "x", "y"], "Tracking CSV")

        tr_color = np.array(self._resolve_color(track_color))[::-1] / 255.0
        pt_color = np.array(self._resolve_color(point_color))[::-1] / 255.0

        stack = load_movie(movie_path)
        if stack.ndim == 2:
            stack = stack[None, ...]
        frame = np.clip(frame, 0, len(stack) - 1)
        img = stack[frame]
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img[..., :3]

        out = Path(output_path)

        # Use a more efficient approach for matplotlib
        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax.imshow(rgb, cmap="gray")
        ax.axis("off")

        # Show ALL trajectories, not just those starting at or after the specified frame
        # This ensures we see the complete picture of all tracks

        # Process all trajectories at once instead of one-by-one
        for tid, g in trk.groupby("track_id"):
            traj = g.sort_values("frame")
            ax.plot(
                traj.x,
                traj.y,
                color=tr_color,
                lw=overview_track_width,
                alpha=0.8,
            )
            # Mark the start point of each track
            start = traj.iloc[0]
            ax.scatter(
                start.x,
                start.y,
                s=point_radius,  # Use point_radius directly as size
                c=[pt_color],
                edgecolors="white",
                linewidth=0.5,
                alpha=0.9,
            )

        plt.tight_layout(pad=0)
        plt.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)  # Close the specific figure to free memory
        return out


# Backwards compatibility functions for legacy API
def create_detection_overlay(
    detection_csv_path: str,
    movie_path: str,
    output_path: str,
    color: Color = "red",
    radius: int = 5,
    fps: int = 30,
) -> Path:
    """
    Legacy function for backwards compatibility.
    Creates a visualization overlay for detection results (outputs AVI with MJPG/JPEG encoding).
    """
    visualizer = Visualizer()
    return visualizer.draw_detections(
        detection_csv=detection_csv_path,
        movie_path=movie_path,
        output_path=output_path,
        color=color,
        radius=radius,
        fps=fps,
    )


def create_tracking_overlay(
    tracks_csv_path: str,
    movie_path: str,
    output_path: str,
    trail_length: int = 10,
    fps: int = 30,
    point_radius: int = 4,
    point_color: Color = "red",
    track_width: int = 2,
    track_color: Color = "red",
) -> Path:
    """
    Legacy function for backwards compatibility.
    Creates a visualization overlay for tracking results (outputs AVI with MJPG/JPEG encoding).
    """
    visualizer = Visualizer()
    return visualizer.draw_tracks(
        tracks_csv=tracks_csv_path,
        movie_path=movie_path,
        output_path=output_path,
        trail_length=trail_length,
        fps=fps,
        point_radius=point_radius,
        point_color=point_color,
        track_width=track_width,
        track_color=track_color,
    )


def create_overview_frame(
    tracks_csv_path: str,
    movie_path: str,
    output_path: str,
    frame: int = 0,
    track_color: Color = "green",
    overview_track_width: float = 1.0,
    point_radius: int = 1,
    point_color: Color = "green",
    dpi: int = 300,
) -> Path:
    """
    Legacy function for backwards compatibility.
    Creates an overview frame showing all trajectories.
    """
    visualizer = Visualizer()
    return visualizer.draw_overview(
        tracks_csv=tracks_csv_path,
        movie_path=movie_path,
        output_path=output_path,
        frame=frame,
        track_color=track_color,
        overview_track_width=overview_track_width,
        point_radius=point_radius,
        point_color=point_color,
        dpi=dpi,
    )
