#!/usr/bin/env python3
"""
Mouse-sperm-head detector that processes single images or image stacks and returns detection results as DataFrame.
"""

import cv2
import numpy as np
import pandas as pd


def _is_image_binarized(img: np.ndarray, tolerance: int = 1) -> bool:
    """
    Check if an image is already binarized by examining its unique pixel values.
    
    Args:
        img: Input image as numpy array
        tolerance: Tolerance for considering values as binary (default 1 to account for minor variations)
    
    Returns:
        True if image appears to be binarized, False otherwise
    """
    # Get unique pixel values and sort them
    unique_values = np.unique(img)
    
    # If there are only 1 or 2 unique values, it's likely binary
    if len(unique_values) <= 2:
        return True
    
    # If there are more values, check if they're close to binary values (0 and 255)
    # Allow some tolerance for minor variations that might occur due to compression or processing
    binary_like_values = []
    for val in unique_values:
        # Convert to int to avoid numpy overflow warnings
        val_int = int(val)
        if abs(val_int - 0) <= tolerance or abs(val_int - 255) <= tolerance:
            binary_like_values.append(val)
    
    # If most values are close to 0 or 255, consider it binary
    return len(binary_like_values) >= len(unique_values) * 0.95  # 95% threshold


def detect_sperm(
    img: np.ndarray,  # Can be single image or stack
    min_area: int = 20,
    max_area: int = 45,
    min_aspect: float = 1.2,
    max_aspect: float = 3.0,
    min_solidity: float = 0.65,
    threshold: int = 10,
    blur_radius: float = 0.5,
) -> pd.DataFrame:
    """
    Detect sperm in a grayscale or binary image or stack using morphological criteria.
    Automatically handles both single images and image stacks.
    Automatically detects if image is already binarized and skips thresholding if so.

    Args:
        img: Input grayscale or binary image (H, W) or image stack (T, H, W)
        min_area: Minimum area threshold for detection
        max_area: Maximum area threshold for detection
        min_aspect: Minimum aspect ratio threshold
        max_aspect: Maximum aspect ratio threshold
        min_solidity: Minimum solidity threshold
        threshold: Binary threshold value (only used if image is not already binarized)
        blur_radius: Gaussian blur radius (0 to disable)

    Returns:
        DataFrame with detection results (frame, x, y, area, aspect_ratio, solidity)
    """
    # Handle single frame vs stack
    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)

    total_frames = img.shape[0]
    rows = []

    for f_idx in range(total_frames):
        frame_img = img[f_idx]

        # Validate input image
        if not isinstance(frame_img, np.ndarray):
            raise TypeError("Input image must be a numpy array")
        if frame_img.ndim != 2:
            raise ValueError(f"Each frame must be 2D, got shape {frame_img.shape}")

        if min_area >= max_area:
            raise ValueError("min_area must be < max_area")
        if min_aspect >= max_aspect:
            raise ValueError("min_aspect must be < max_aspect")
        if blur_radius < 0:
            raise ValueError("blur_radius must be ≥ 0")

        # Apply Gaussian blur if radius is greater than 0
        if blur_radius > 0:
            k = int(blur_radius * 2 + 1) | 1  # Ensure kernel size is odd
            frame_img = cv2.GaussianBlur(frame_img, (k, k), 0)

        # Check if image is already binarized and skip thresholding if it is
        if _is_image_binarized(frame_img):
            # Use the image as-is (convert to binary format if needed)
            # Ensure that any values > 0 are set to 255 for proper contour detection
            binary = np.where(frame_img > 0, 255, 0).astype(np.uint8)
        else:
            # Apply binary threshold for grayscale images
            _, binary = cv2.threshold(frame_img, threshold, 255, cv2.THRESH_BINARY)
            
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Precompute frequently used values
        area_range = (min_area, max_area)
        aspect_range = (min_aspect, max_aspect)

        # Process each contour
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Early exit if outside area range
            if area < area_range[0] or area > area_range[1]:
                continue

            # Calculate aspect ratio - try/except in case of numerical issues
            try:
                (width, height), _ = cv2.minAreaRect(cnt)[1:]
                if width == 0 or height == 0:
                    continue
                aspect = max(width, height) / min(width, height)
            except cv2.error:
                # Skip if minAreaRect fails (e.g., due to invalid contour)
                import traceback
                traceback.print_exc()
                continue

            # Early exit if outside aspect range
            if aspect < aspect_range[0] or aspect > aspect_range[1]:
                continue

            # Calculate solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:  # Avoid division by zero
                continue
            solidity = area / hull_area
            # Early exit if solidity is too low
            if solidity < min_solidity:
                continue

            # Calculate centroid
            moments = cv2.moments(cnt)
            if moments["m00"]:
                # Calculate centroid with floating point precision
                cx_float = moments["m10"] / moments["m00"]
                cy_float = moments["m01"] / moments["m00"]
                # Round to nearest integer for pixel coordinates
                cx = int(round(cx_float))
                cy = int(round(cy_float))

                rows.append(
                    {
                        "frame": f_idx,
                        "x": cx,
                        "y": cy,
                        "area": area,
                        "aspect_ratio": aspect,
                        "solidity": solidity,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    return df
