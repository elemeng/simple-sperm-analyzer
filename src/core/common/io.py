#!/usr/bin/env python3
"""
File I/O operations used across multiple modules in the sperm tracking project.
"""

import numpy as np
import cv2
import tifffile as tiff
from pathlib import Path


def load_movie(path: str) -> np.ndarray:
    """Return 3-D array (T, Y, X) even for single-frame files.
    Input: supports TIFF and video files (MP4/AVI/MOV)."""
    path_obj = Path(path)

    # Handle video files (for overlay only)
    if path_obj.suffix.lower() in [".mp4", ".avi", ".mov"]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[Video Info] {path_obj.name} - FPS: {fps:.1f}, Frames: {frame_count}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to grayscale for consistency
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()

        if not frames:
            raise ValueError(f"No frames could be read from video file: {path}")
        if len(frames) != frame_count:
            print(f"[Warning] Only read {len(frames)}/{frame_count} frames from video")

        return np.array(frames, dtype=np.uint8)

    # Handle TIFF files (for detection and overlay)
    else:
        # Only support TIFF files
        if path_obj.suffix.lower() not in [".tif", ".tiff"]:
            raise ValueError(
                f"Input must be TIFF file (.tif/.tiff), got: {path_obj.suffix}"
            )

        img = tiff.imread(path)
        if img.ndim == 2:
            img = img[None, ...]

        # Ensure it's grayscale
        if img.ndim == 3 and img.shape[2] in [3, 4]:
            raise ValueError(
                f"Input must be grayscale TIFF, got color image with {img.shape[2]} channels"
            )

        return img.astype(np.uint8)
