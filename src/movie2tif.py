#!/usr/bin/env python3
"""
Video to TIFF Stack Converter
Batch converts video files to 8-bit grayscale multi-page TIFF stacks.

Usage:
    python video2tiff.py -i /path/to/videos -o /path/to/output
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

def convert_frame_to_8bit(frame):
    """Convert frame to 8-bit grayscale."""
    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Convert to grayscale if it's a color image
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        return frame

def video_to_multipage_tiff(video_path, output_path, start_frame=0, max_frames=None):
    """Convert video to single multi-page TIFF."""
    try:
        import imageio.v3 as iio
    except ImportError:
        print("Error: Install imageio for multi-page TIFF support: pip install imageio")
        sys.exit(1)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = total_frames if max_frames is None else min(start_frame + max_frames, total_frames)
    frames_to_process = end_frame - start_frame
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in tqdm(range(frames_to_process), desc=f"Reading {video_path.name}", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(convert_frame_to_8bit(frame))
    
    cap.release()
    iio.imwrite(str(output_path), frames, extension='.tif')
    return len(frames)

def find_videos(input_dir, extensions=['.mp4', '.avi', '.mov', '.mkv'], recursive=False):
    """Find all video files in directory."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"Input is not a directory: {input_dir}")
    
    pattern = "**/*" if recursive else "*"
    videos = []
    for ext in extensions:
        videos.extend(input_path.glob(f"{pattern}{ext}"))
        videos.extend(input_path.glob(f"{pattern}{ext.upper()}"))
    
    return sorted(set(videos))

def main():
    parser = argparse.ArgumentParser(
        description="Batch convert videos to 8-bit grayscale TIFF stacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all MP4s in folder to multi-page TIFFs
  python video2tiff.py -i ./videos -o ./tiffs

  # Save TIFFs in same location as source videos
  python video2tiff.py -i ./videos --same-location

  # Dry run to preview what would be converted
  python video2tiff.py -i ./videos -o ./tiffs --dry-run

  # Convert recursively from subdirectories
  python video2tiff.py -i ./data -o ./output --recursive

  # Convert first 100 frames of each video only
  python video2tiff.py -i ./videos -o ./tiffs --max-frames 100
        """
    )
    
    parser.add_argument("-i", "--input", required=True, help="Input folder containing videos")
    parser.add_argument("-o", "--output", help="Output folder for TIFF stacks (not needed with --same-location)")
    
    parser.add_argument("-e", "--extensions", default="mp4,avi,mov,mkv",
                        help="Comma-separated video extensions (default: mp4,avi,mov,mkv)")
    
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Process subdirectories recursively")
    
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Starting frame number (default: 0)")
    
    parser.add_argument("--max-frames", type=int,
                        help="Maximum frames to convert per video")
    
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    
    parser.add_argument("--same-location", action="store_true",
                        help="Save TIFF files in the same location as source videos")
    
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be converted without creating files")
    
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.output and not args.same_location:
        parser.error("Either --output or --same-location must be specified")
    
    # Setup
    input_dir = Path(args.input)
    
    # Only create output directory if not in same-location mode and not dry-run
    if not args.same_location and not args.dry_run:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find videos
    extensions = [f".{ext.strip()}" for ext in args.extensions.split(',')]
    try:
        videos = find_videos(input_dir, extensions, args.recursive)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if not videos:
        print(f"No videos found in {input_dir} with extensions {extensions}")
        sys.exit(0)
    
    print(f"Found {len(videos)} video(s) to process")
    
    # Dry run: just show what would be converted
    if args.dry_run:
        print("\nDRY RUN - No files will be created")
        for video_path in videos:
            if args.same_location:
                output_path = video_path.parent / f"{video_path.stem}.tif"
            else:
                output_path = output_dir / f"{video_path.stem}.tif"
            
            if args.overwrite or not output_path.exists():
                print(f"Would convert: {video_path} -> {output_path}")
            else:
                print(f"Would skip: {video_path} (output exists)")
        return
    
    # Process each video
    total_converted = 0
    for video_path in videos:
        try:
            # Determine output path
            if args.same_location:
                output_path = video_path.parent / f"{video_path.stem}.tif"
            else:
                output_path = output_dir / f"{video_path.stem}.tif"
            
            # Check for existing output
            if not args.overwrite and output_path.exists():
                print(f"Skipping {video_path.name} (output exists, use --overwrite to force)")
                continue
            
            # Convert to multi-page TIFF
            frames = video_to_multipage_tiff(
                video_path, output_path,
                start_frame=args.start_frame,
                max_frames=args.max_frames
            )
            
            total_converted += 1
            print(f"✓ Converted {video_path.name} ({frames} frames)")
            
        except Exception as e:
            print(f"✗ Error processing {video_path.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
    
    print(f"\nCompleted! Successfully converted {total_converted}/{len(videos)} videos")

if __name__ == "__main__":
    main()