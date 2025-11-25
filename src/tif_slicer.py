#!/usr/bin/env python3
"""
TIFF Stack Slicer - Extract frames from multi-page TIFF files
Usage: python tiff_slicer.py [options] <input> -s START -e END
"""

import sys
from pathlib import Path
import argparse
from typing import List, Optional
import numpy as np

# Required packages: tifffile, tqdm
try:
    import tifffile as tiff
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install tifffile tqdm")
    sys.exit(1)


def slice_tiff_stack(
    input_path: Path,
    start_frame: int,
    end_frame: int,
    output_folder: Optional[Path] = None,
    prefix: str = "fraction",
    verbose: bool = False,
    base_input_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Slice a multi-page TIFF file and save the selected frames.

    Args:
        input_path: Path to input TIFF file
        start_frame: Starting frame index (0-based)
        end_frame: Ending frame index (inclusive)
        output_folder: Output directory (defaults to input file's directory)
        prefix: Prefix for output filename
        verbose: Print detailed processing info

    Returns:
        Path to output file if successful, None otherwise
    """
    try:
        # Read the TIFF stack
        with tiff.TiffFile(str(input_path)) as tif:
            # Get total number of pages
            total_frames = len(tif.pages)

            # Validate frame range
            if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
                print(
                    f"Error: Invalid frame range [{start_frame}-{end_frame}] for '{input_path}'"
                )
                print(f"       File has {total_frames} frames (0-{total_frames - 1})")
                return None

            # Read selected frames
            frames = []
            for i in range(start_frame, end_frame + 1):
                frames.append(tif.pages[i].asarray())

        # Convert to numpy array
        frames = np.stack(frames)

        # Prepare output path with folder structure preservation
        if output_folder is None:
            # Default: save in same directory as input file
            output_folder = input_path.parent
        else:
            # Preserve folder structure relative to base_input_path
            if base_input_path and input_path.is_relative_to(base_input_path):
                # Calculate relative path from base input directory
                relative_path = input_path.relative_to(base_input_path)
                # Remove the filename to get just the directory structure
                if relative_path.parent != Path('.'):
                    output_folder = output_folder / relative_path.parent
                else:
                    # File is directly in base_input_path
                    pass  # output_folder stays as specified
            else:
                # Fallback: create output directory structure based on input file's parent
                output_folder = output_folder / input_path.parent.name
            
            # Create the output directory structure
            output_folder.mkdir(parents=True, exist_ok=True)

        # Generate output filename: prefix_originalname.tif
        original_name = input_path.stem
        output_name = f"{prefix}_{original_name}.tif"
        output_path = output_folder / output_name

        # Write the sliced stack
        tiff.imwrite(str(output_path), frames)

        if verbose:
            print(f"✓ Processed: {input_path.name}")
            print(
                f"  → Extracted frames {start_frame}-{end_frame} ({len(frames)} frames)"
            )
            print(f"  → Saved: {output_path}")

        return output_path

    except Exception as e:
        print(f"Error processing '{input_path}': {str(e)}")
        return None


def find_tiff_files(input_path: Path, recursive: bool = False, glob_pattern: str = "*.tif*") -> List[Path]:
    """Find all TIFF files in the given path using specified glob pattern."""
    if input_path.is_file():
        return [input_path]

    # Use the provided glob pattern
    if recursive:
        pattern = f"**/{glob_pattern}"
    else:
        pattern = glob_pattern
    
    tiff_files = list(input_path.glob(pattern))

    # If the pattern doesn't include .tiff, also check for it
    if "*.tiff" not in glob_pattern and "*.*" in glob_pattern:
        if recursive:
            pattern_tiff = f"**/*.tiff"
        else:
            pattern_tiff = "*.tiff"
        tiff_files.extend(list(input_path.glob(pattern_tiff)))

    return sorted(tiff_files)


def main():
    parser = argparse.ArgumentParser(
        description="Slice TIFF stacks by frame range. Process single file or batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Single file: extract frames 5-10
  python tiff_slicer.py image.tif -s 5 -e 10
  
  # Batch process folder (non-recursive): extract frames 0-50
  python tiff_slicer.py ./data/ -s 0 -e 50
  
  # Batch process recursively with custom prefix and output folder (preserves structure)
  # Input: test/aaa/123.tif, test/bbb/456.tif → Output: output/aaa/trimmed_123.tif, output/bbb/trimmed_456.tif
  python tiff_slicer.py ./data/ -s 10 -e 99 -r -p "trimmed" -o ./output/
  
  # Process only specific TIFF files with custom glob pattern
  python tiff_slicer.py ./data/ -s 10 -e 99 -r --glob "*enhanced*.tif" -o ./output/
  
  # Same as above but with verbose output
  python tiff_slicer.py ./data/ -s 10 -e 99 -r --glob "*enhanced*.tif" -o ./output/ -v""",
    )

    # Input
    parser.add_argument("input", help="Input TIFF file or folder")

    # Frame range (required)
    parser.add_argument(
        "-s", "--start", type=int, required=True, help="Start frame (0-based)"
    )
    parser.add_argument(
        "-e", "--end", type=int, required=True, help="End frame (inclusive)"
    )

    # Batch processing
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Process folders recursively"
    )
    parser.add_argument(
        "--glob", 
        default="*.tif*", 
        help="Glob pattern for TIFF files (default: '*.tif*') - supports *.tif, *.tiff, or custom patterns"
    )

    # Output options
    parser.add_argument(
        "-o", "--output", type=Path, help="Output folder (default: same as input). When specified, preserves input folder structure."
    )
    parser.add_argument(
        "-p",
        "--prefix",
        default="fraction",
        help="Output filename prefix (default: 'fraction')",
    )

    # Other
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Resolve input path
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)

    # Find files to process
    files_to_process = find_tiff_files(input_path, args.recursive, args.glob)

    if not files_to_process:
        if input_path.is_dir():
            print(f"No TIFF files found in '{input_path}'")
        else:
            print(f"'{input_path}' is not a valid TIFF file")
        sys.exit(1)

    # Process files
    if args.verbose:
        print(f"Found {len(files_to_process)} TIFF file(s) to process\n")

    success_count = 0

    # Use progress bar for batch processing
    for file_path in tqdm(files_to_process, desc="Processing", unit="file"):
        result = slice_tiff_stack(
            input_path=file_path,
            start_frame=args.start,
            end_frame=args.end,
            output_folder=args.output,
            prefix=args.prefix,
            verbose=args.verbose,
            base_input_path=input_path if input_path.is_dir() else None,
        )
        if result:
            success_count += 1

    # Summary
    if args.verbose or len(files_to_process) > 1:
        print(
            f"\n✓ Successfully processed {success_count}/{len(files_to_process)} files"
        )


if __name__ == "__main__":
    main()
