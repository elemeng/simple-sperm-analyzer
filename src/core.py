import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile as tiff
from typing import List, Tuple, Optional, Dict

from datetime import datetime


def load_tif_stack(file_path: str) -> np.ndarray:
    """Load binary TIF image stack"""
    return tiff.imread(file_path)


def detect_sperm_coordinates(
    image: np.ndarray,
    min_area: int = 20,
    max_area: int = 45,
    min_aspect: float = 1.2,
    max_aspect: float = 3.0,
    min_solidity: float = 0.65,
    threshold: int = 10,
    blur_radius: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Detect sperm coordinates using enhanced contour analysis for ~30px mouse sperm heads.
    Returns only (x, y) coordinates for backward compatibility.
    """
    # Get enhanced detections but return only coordinates
    enhanced_detections = detect_sperm_coordinates_enhanced(
        image,
        min_area,
        max_area,
        min_aspect,
        max_aspect,
        min_solidity,
        threshold,
        blur_radius,
    )
    return [(det["x"], det["y"]) for det in enhanced_detections]


def detect_sperm_coordinates_enhanced(
    image: np.ndarray,
    min_area: int = 20,
    max_area: int = 45,
    min_aspect: float = 1.2,
    max_aspect: float = 3.0,
    min_solidity: float = 0.65,
    threshold: int = 10,
    blur_radius: float = 0.5,
) -> List[Dict]:
    """
    Enhanced version that returns detailed metrics for each detection including parameters.
    """
    # Parameter validation with proper exceptions (not assertions)
    if not (min_area < max_area):
        raise ValueError(f"min_area ({min_area}) must be < max_area ({max_area})")
    if not (min_aspect < max_aspect):
        raise ValueError(
            f"min_aspect ({min_aspect}) must be < max_aspect ({max_aspect})"
        )
    if blur_radius < 0:
        raise ValueError(f"blur_radius ({blur_radius}) must be non-negative")

    # Store original parameters for inclusion in results
    detection_params = {
        "min_area": min_area,
        "max_area": max_area,
        "min_aspect": min_aspect,
        "max_aspect": max_aspect,
        "min_solidity": min_solidity,
        "threshold": threshold,
        "blur_radius": blur_radius,
    }

    # Apply Gaussian blur to reduce noise while preserving small details
    if blur_radius > 0:
        kernel_size = int(blur_radius * 2 + 1) | 1  # Force odd number for GaussianBlur
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Ensure binary image
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for contour in contours:
        # Check area first (cheap operation) before expensive shape analysis
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        # Shape analysis
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)

        # Aspect ratio filtering for elongated but compact shapes
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue

        # Solidity analysis for hook shape detection (expensive operation, do last)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = area / hull_area
        if solidity < min_solidity:
            continue

        # Calculate center of mass (centroid) for the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Create enhanced detection record with all parameters and metrics
            detection = {
                "x": cx,
                "y": cy,
                "area": area,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "contour": contour,
                "centroid": (cx, cy),
                # Include all detection parameters for traceability
                **detection_params,
            }
            detections.append(detection)

    return detections


def serialize_contour(contour: np.ndarray) -> str:
    """
    Serialize contour to a compact string format for CSV storage.
    Returns format: "x1,y1;x2,y2;x3,y3;..."
    """
    if contour is None or len(contour) == 0:
        return ""

    # Convert contour to list of (x,y) coordinates
    points = []
    for point in contour:
        x, y = point[0]  # contour points are in format [[x,y]]
        points.append(f"{x},{y}")

    return ";".join(points)


def deserialize_contour(contour_str: str) -> np.ndarray:
    """
    Deserialize contour from string format back to numpy array.
    Input format: "x1,y1;x2,y2;x3,y3;..."
    Returns numpy array of shape (n_points, 1, 2)
    """
    if not contour_str:
        return np.array([])

    try:
        points = []
        for point_str in contour_str.split(";"):
            if point_str:
                x, y = map(float, point_str.split(","))
                points.append([[x, y]])  # Format expected by OpenCV

        return np.array(points, dtype=np.int32)
    except (ValueError, IndexError):
        return np.array([])


def compute_curvature_signature(contour: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """
    Compute rotation-invariant curvature signature from contour.
    Returns n_bins radial distances from centroid, normalized by median.
    """
    if contour is None or len(contour) < 5:
        return np.zeros(n_bins)

    # Calculate centroid
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return np.zeros(n_bins)

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Calculate radial distances
    distances = np.sqrt((contour[:, 0, 0] - cx) ** 2 + (contour[:, 0, 1] - cy) ** 2)

    # Resample to n_bins (rotation-invariant)
    if len(distances) < n_bins:
        x_old = np.linspace(0, 1, len(distances))
        x_new = np.linspace(0, 1, n_bins)
        signature = np.interp(x_new, x_old, distances)
    else:
        idx = np.linspace(0, len(distances) - 1, n_bins, dtype=int)
        signature = distances[idx]

    # Normalize by median for scale invariance
    signature /= np.median(signature) + 1e-6
    return signature


def get_marker_color(color_name: str) -> Tuple[int, int, int]:
    """Convert color name to BGR tuple"""
    colors = {
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
    return colors.get(color_name.lower(), (0, 0, 255))  # Default to red


def draw_marker(
    img: np.ndarray,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    size: int,
    shape: str = "circle",
) -> None:
    """Draw marker of specified shape, size and color"""
    if shape == "circle":
        cv2.circle(img, (x, y), size, color, -1)
    elif shape == "cross":
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, size * 2, 1)
    elif shape == "square":
        half_size = size // 2
        cv2.rectangle(
            img,
            (x - half_size, y - half_size),
            (x + half_size, y + half_size),
            color,
            -1,
        )
    elif shape == "plus":
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_TILTED_CROSS, size * 2, 1)
    elif shape == "diamond":
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_DIAMOND, size * 2, 1)
    else:
        cv2.circle(img, (x, y), size, color, -1)  # Default to circle


def add_markers_to_frame(
    frame: np.ndarray,
    coordinates: List[Tuple[float, float]],
    color: str = "red",
    size: int = 3,
    shape: str = "circle",
) -> np.ndarray:
    """Add markers to frame and return marked frame for 8-bit grayscale input"""
    # Input is 8-bit grayscale TIF, convert to RGB for visualization
    if len(frame.shape) == 2:
        marked_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        # Fallback for unexpected input format
        marked_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Get color
    marker_color = get_marker_color(color)

    # Draw markers at each coordinate
    for x, y in coordinates:
        draw_marker(marked_frame, int(x), int(y), marker_color, size, shape)

    return marked_frame


def create_marked_tif_stack(
    stack: np.ndarray,
    all_coordinates: List[List[Tuple[float, float]]],
    color: str = "red",
    size: int = 3,
    shape: str = "circle",
    output_path: str = None,
) -> None:
    """Create TIF stack with markers overlaid using streaming to avoid memory accumulation"""
    if not output_path:
        print("Warning: output_path not provided, skipping marked stack generation")
        return

    # Process and save frames one at a time to avoid memory accumulation
    for frame_idx, (frame, coords) in enumerate(zip(stack, all_coordinates)):
        marked_frame = add_markers_to_frame(frame, coords, color, size, shape)

        if frame_idx == 0:
            # First frame - create new file
            tiff.imwrite(output_path, marked_frame, bigtiff=True)
        else:
            # Subsequent frames - append to existing file
            with tiff.TiffWriter(output_path, append=True) as writer:
                writer.write(marked_frame)


def save_detection_summary(
    coordinates_df: pd.DataFrame,
    input_path: str,
    output_dir: str,
    params: Dict,
    total_frames: int,
    total_detected: int,
) -> None:
    """Save detection summary and parameters"""
    input_stem = Path(input_path).stem
    summary_path = Path(output_dir) / f"{input_stem}_detection_log.txt"

    with open(summary_path, "w") as f:
        f.write("Mouse Sperm Detection Summary\n")
        f.write("=============================\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Detection Parameters:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\n")

        f.write("Results:\n")
        f.write(f"  Total frames processed: {total_frames}\n")
        f.write(f"  Total sperm detected: {total_detected}\n")
        f.write(f"  Average sperm per frame: {total_detected / total_frames:.1f}\n")
        f.write(f"  Frames with detections: {coordinates_df['frame'].nunique()}\n")

        if len(coordinates_df) > 0:
            f.write("\nDetection Statistics:\n")
            frame_counts = coordinates_df.groupby("frame").size()
            f.write(f"  Min sperm per frame: {frame_counts.min()}\n")
            f.write(f"  Max sperm per frame: {frame_counts.max()}\n")
            f.write(f"  Std dev: {frame_counts.std():.1f}\n")

    print(f"Saved detection summary: {summary_path}")


def parse_frame_range(frames_str: str, total_frames: int) -> List[int]:
    """Parse frame range string like '0,5,10' or '0-20' or '0-10,15,20-30'"""
    frames = []

    if not frames_str:
        return list(range(total_frames))

    for part in frames_str.split(","):
        if "-" in part:
            start, end = part.split("-")
            start = int(start.strip())
            end = int(end.strip())
            frames.extend(range(start, min(end + 1, total_frames)))
        else:
            frame = int(part.strip())
            if 0 <= frame < total_frames:
                frames.append(frame)

    return sorted(list(set(frames)))  # Remove duplicates and sort


def process_sperm_stack(
    tif_path: str,
    output_dir: str,
    min_area: int = 20,
    max_area: int = 45,
    min_aspect: float = 1.2,
    max_aspect: float = 3.0,
    min_solidity: float = 0.65,
    threshold: int = 10,
    blur_radius: float = 0.5,
    marker_color: Optional[str] = None,
    marker_size: int = 3,
    marker_shape: str = "circle",
    frames: Optional[str] = None,
    debug: bool = False,
    overlay_movie: Optional[str] = None,
) -> pd.DataFrame:
    """
    Process TIF stack and extract sperm coordinates for each frame with enhanced parameters.
    """
    stack = load_tif_stack(tif_path)
    all_coordinates = []

    # Handle both single frame and multi-frame stacks
    if len(stack.shape) == 2:
        stack = np.expand_dims(stack, axis=0)

    total_frames = len(stack)
    frame_indices = parse_frame_range(frames, total_frames)

    if debug:
        print(
            f"Processing {len(frame_indices)} frames out of {total_frames} total frames"
        )

    # Process specified frames
    all_frame_coords = []
    for frame_idx in frame_indices:
        frame = stack[frame_idx]
        # Use enhanced detection to get detailed metrics
        enhanced_detections = detect_sperm_coordinates_enhanced(
            frame,
            min_area,
            max_area,
            min_aspect,
            max_aspect,
            min_solidity,
            threshold,
            blur_radius,
        )
        all_frame_coords.append(enhanced_detections)

        for det_idx, detection in enumerate(enhanced_detections):
            all_coordinates.append(
                {
                    "frame": frame_idx,
                    "sperm_id": det_idx,
                    "x": detection["x"],
                    "y": detection["y"],
                    "area": detection["area"],
                    "aspect_ratio": detection["aspect_ratio"],
                    "solidity": detection["solidity"],
                    # Include serialized contour data
                    "contour": serialize_contour(detection["contour"]),
                    # Include all detection parameters for complete traceability
                    "threshold": detection["threshold"],
                    "blur_radius": detection["blur_radius"],
                    "min_area": detection["min_area"],
                    "max_area": detection["max_area"],
                    "min_aspect": detection["min_aspect"],
                    "max_aspect": detection["max_aspect"],
                    "min_solidity": detection["min_solidity"],
                }
            )

    df = pd.DataFrame(all_coordinates)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save coordinates
    input_stem = Path(tif_path).stem
    coords_path = output_path / f"{input_stem}_coords.csv"
    df.to_csv(coords_path, index=False)
    print(f"Coordinates saved to: {coords_path}")

    # Create marked TIF stack if markers requested
    if marker_color:
        marked_path = output_path / f"{input_stem}_marked.tif"

        # Use overlay movie if provided, otherwise use processed stack
        if overlay_movie:
            if Path(overlay_movie).exists():
                overlay_stack = load_tif_stack(overlay_movie)
                if len(overlay_stack.shape) == 2:
                    overlay_stack = np.expand_dims(overlay_stack, axis=0)

                # Ensure overlay has same number of frames as processed
                if len(overlay_stack) >= len(frame_indices):
                    overlay_frames = overlay_stack[frame_indices]
                    overlay_stem = Path(overlay_movie).stem
                    marked_path = (
                        output_path / f"{input_stem}_on_{overlay_stem}_marked.tif"
                    )
                    print(f"Using overlay movie: {overlay_movie}")
                else:
                    print(
                        f"Warning: Overlay movie has fewer frames ({len(overlay_stack)}) than processed ({len(frame_indices)})"
                    )
                    overlay_frames = stack[frame_indices]
            else:
                print(
                    f"Warning: Overlay movie {overlay_movie} not found, using processed stack"
                )
                overlay_frames = stack[frame_indices]
        else:
            overlay_frames = stack[frame_indices]

        # Convert enhanced detections to coordinate format for marking
        coord_list = [
            [(det["x"], det["y"]) for det in frame_dets]
            for frame_dets in all_frame_coords
        ]
        create_marked_tif_stack(
            overlay_frames,
            coord_list,
            marker_color,
            marker_size,
            marker_shape,
            str(marked_path),
        )
        print(f"Marked TIF stack saved to: {marked_path}")

    # Save detection summary
    params = {
        "min_area": min_area,
        "max_area": max_area,
        "min_aspect": min_aspect,
        "max_aspect": max_aspect,
        "min_solidity": min_solidity,
        "threshold": threshold,
        "blur_radius": blur_radius,
        "marker_color": marker_color,
        "marker_size": marker_size,
        "marker_shape": marker_shape,
    }
    save_detection_summary(
        df, tif_path, output_dir, params, len(frame_indices), len(df)
    )

    # Create summary visualization (first frame with detections)
    if len(df) > 0:
        first_frame_with_detections = df["frame"].min()
        frame_idx = (
            frame_indices.index(first_frame_with_detections)
            if first_frame_with_detections in frame_indices
            else 0
        )

        # Use overlay movie for summary if available, otherwise use processed stack
        if overlay_movie and Path(overlay_movie).exists():
            overlay_stack = load_tif_stack(overlay_movie)
            if len(overlay_stack.shape) == 2:
                overlay_stack = np.expand_dims(overlay_stack, axis=0)
            if len(overlay_stack) > frame_idx:
                summary_frame_raw = overlay_stack[frame_indices[frame_idx]]
            else:
                summary_frame_raw = stack[frame_indices[frame_idx]]
        else:
            summary_frame_raw = stack[frame_indices[frame_idx]]

        summary_frame = add_markers_to_frame(
            summary_frame_raw,
            [(det["x"], det["y"]) for det in all_frame_coords[frame_idx]],
            marker_color or "red",
            marker_size,
            marker_shape or "circle",
        )

        # Add text info
        cv2.putText(
            summary_frame,
            f"Frame {first_frame_with_detections}: {len(all_frame_coords[frame_idx])} sperm detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Update summary filename if using overlay
        if overlay_movie and Path(overlay_movie).exists():
            overlay_stem = Path(overlay_movie).stem
            summary_path = output_path / f"{input_stem}_on_{overlay_stem}_summary.png"
        else:
            summary_path = output_path / f"{input_stem}_summary.png"
        cv2.imwrite(str(summary_path), summary_frame)
        print(f"Summary visualization saved to: {summary_path}")

    return df
