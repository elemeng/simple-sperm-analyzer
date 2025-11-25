import argparse
from pathlib import Path
from typing import Optional

from core import process_sperm_stack


def main():
    parser = argparse.ArgumentParser(
        description="Mouse sperm head detection from binary TIF images"
    )

    # GUI mode
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI mode for interactive parameter tuning",
    )

    # Input options (for CLI mode)
    parser.add_argument(
        "input_file", nargs="?", help="Input TIF file path (required for CLI mode)"
    )
    parser.add_argument(
        "--input_dir",
        help="Input directory for batch processing (alternative to input_file)",
    )
    parser.add_argument(
        "--output_dir", help="Output directory for all results (required for CLI mode)"
    )

    # Detection parameters
    parser.add_argument(
        "--min-area", type=int, default=20, help="Minimum area in pixels (default: 20)"
    )
    parser.add_argument(
        "--max-area", type=int, default=45, help="Maximum area in pixels (default: 45)"
    )
    parser.add_argument(
        "--min-aspect",
        type=float,
        default=1.2,
        help="Minimum aspect ratio (default: 1.2)",
    )
    parser.add_argument(
        "--max-aspect",
        type=float,
        default=3.0,
        help="Maximum aspect ratio (default: 3.0)",
    )
    parser.add_argument(
        "--min-solidity",
        type=float,
        default=0.65,
        help="Minimum solidity for hook shape (default: 0.65)",
    )
    parser.add_argument(
        "--threshold", type=int, default=10, help="Binary threshold value (default: 10)"
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=0.5,
        help="Gaussian blur radius (default: 0.5)",
    )

    # Marker visualization options
    parser.add_argument(
        "--marker-color",
        choices=["red", "blue", "green", "yellow", "white", "cyan", "magenta"],
        help="Marker color for visualization",
    )
    parser.add_argument(
        "--marker-size", type=int, default=3, help="Marker size in pixels (default: 3)"
    )
    parser.add_argument(
        "--marker-shape",
        choices=["circle", "cross", "square", "plus", "diamond"],
        default="circle",
        help="Marker shape (default: circle)",
    )

    # Processing options
    parser.add_argument(
        "--frames", help='Specific frames to process (e.g., "0,5,10" or "0-20")'
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show detailed detection info"
    )
    parser.add_argument(
        "--overlay-movie",
        help="Path to original/unprocessed movie for overlay visualization",
    )

    # Batch processing options
    parser.add_argument(
        "--file-pattern",
        default="*.tif",
        help="File pattern for batch processing (default: *.tif)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process directories recursively for batch processing",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel processes for batch processing (default: 1)",
    )
    
    # Overlay movie matching for batch processing
    parser.add_argument(
        "--overlay-movie-dir",
        type=Path,
        help="Directory containing overlay movies for batch processing",
    )
    parser.add_argument(
        "--cut-overlay-movie-prefix",
        type=int,
        default=0,
        help="Cut N characters from beginning of input filename for overlay matching (default: 0)",
    )
    parser.add_argument(
        "--cut-overlay-movie-suffix",
        type=int,
        default=0,
        help="Cut N characters from end of input filename (before extension) for overlay matching (default: 0)",
    )

    args = parser.parse_args()

    # Launch GUI mode if requested
    if args.gui:
        try:
            from gui import main as gui_main

            gui_main()
        except ImportError as e:
            print(f"Error: PyQt6 not available. Install with: pip install PyQt6")
            print(f"Details: {e}")
            return 1
        return 0

    # CLI mode - check required arguments
    if not args.gui and not args.output_dir:
        parser.print_help()
        print("\nError: --gui mode OR --output_dir is required")
        return 1

    # Determine processing mode: single file, batch, or GUI
    if args.gui:
        # GUI mode handled above
        pass
    elif args.input_dir:
        # Batch processing mode
        return process_batch(args, parser)
    elif args.input_file:
        # Single file processing mode
        return process_single_file(args, parser)
    else:
        parser.print_help()
        print("\nError: Either --gui, --input_dir, or input_file must be specified")
        return 1


def process_single_file(args, parser):
    """Process a single TIF file"""
    # Check input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} not found")
        return 1

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing {args.input_file}...")
    print(f"Output directory: {output_path}")

    try:
        df = process_sperm_stack(
            args.input_file,
            output_dir=args.output_dir,
            min_area=args.min_area,
            max_area=args.max_area,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            min_solidity=args.min_solidity,
            threshold=args.threshold,
            blur_radius=args.blur_radius,
            marker_color=args.marker_color,
            marker_size=args.marker_size,
            marker_shape=args.marker_shape,
            frames=args.frames,
            debug=args.debug,
            overlay_movie=args.overlay_movie,
        )

        print(f"\nDetection complete!")
        print(
            f"Found {len(df)} sperm coordinates across {df['frame'].nunique()} frames"
        )

        if len(df) > 0:
            print(f"Average sperm per frame: {len(df) / df['frame'].nunique():.1f}")
            print(f"\nFirst few detections:")
            print(df.head())

        return 0

    except Exception as e:
        print(f"Error processing file: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


def find_overlay_movie(input_file: Path, args) -> Optional[str]:
    """Find the appropriate overlay movie for a given input file."""
    
    # If no overlay movie options provided, use the input file itself
    if not args.overlay_movie and not args.overlay_movie_dir:
        return str(input_file)
    
    # If explicit overlay movie path is provided, use it
    if args.overlay_movie:
        # Check if it's a directory or file pattern
        overlay_path = Path(args.overlay_movie)
        if overlay_path.is_dir():
            # It's a directory, treat it like --overlay-movie-dir
            return find_overlay_movie_in_dir(input_file, overlay_path, args)
        elif overlay_path.exists():
            # It's a specific file, use it
            return str(overlay_path)
        else:
            # It might be a pattern, try glob matching
            return find_overlay_movie_by_pattern(input_file, overlay_path, args)
    
    # If overlay movie directory is provided
    if args.overlay_movie_dir:
        return find_overlay_movie_in_dir(input_file, args.overlay_movie_dir, args)
    
    # Fallback: use input file itself
    return str(input_file)


def find_overlay_movie_in_dir(input_file: Path, overlay_dir: Path, args) -> Optional[str]:
    """Find overlay movie in directory using prefix/suffix cutting."""
    
    # Get input filename without path
    input_name = input_file.name
    
    # Apply prefix cutting
    if args.cut_overlay_movie_prefix > 0:
        input_name = input_name[args.cut_overlay_movie_prefix:]
    
    # Apply suffix cutting (before extension)
    if args.cut_overlay_movie_suffix > 0:
        name_without_ext = input_name.rsplit('.', 1)[0]
        extension = input_name.rsplit('.', 1)[1] if '.' in input_name else ''
        if len(name_without_ext) > args.cut_overlay_movie_suffix:
            name_without_ext = name_without_ext[:-args.cut_overlay_movie_suffix]
            input_name = name_without_ext + ('.' + extension if extension else '')
    
    # Look for matching files in overlay directory
    if overlay_dir.is_dir():
        # Try exact match first
        exact_match = overlay_dir / input_name
        if exact_match.exists():
            return str(exact_match)
        
        # Try with common TIFF extensions
        for ext in ['.tif', '.TIF', '.tiff', '.TIFF']:
            tiff_match = overlay_dir / (input_name.rsplit('.', 1)[0] + ext)
            if tiff_match.exists():
                return str(tiff_match)
        
        # Try glob pattern matching
        for ext in ['.tif*', '.TIF*', '.tiff*', '.TIFF*']:
            pattern = input_name.rsplit('.', 1)[0] + ext
            matches = list(overlay_dir.glob(pattern))
            if matches:
                return str(matches[0])
    
    # If no match found, fall back to using input file itself
    print(f"  Warning: No overlay movie found for {input_file.name} using pattern {input_name}")
    print(f"  Using input file as overlay movie")
    return str(input_file)


def find_overlay_movie_by_pattern(input_file: Path, pattern_path: Path, args) -> Optional[str]:
    """Find overlay movie using glob pattern matching."""
    
    # If pattern contains wildcards, try to match
    if '*' in str(pattern_path) or '?' in str(pattern_path):
        parent_dir = pattern_path.parent
        pattern = pattern_path.name
        
        if parent_dir.is_dir():
            matches = list(parent_dir.glob(pattern))
            if matches:
                return str(matches[0])
    
    # Fallback
    return str(input_file)


def process_batch(args, parser):
    """Process multiple TIF files in batch mode"""
    from pathlib import Path
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} not found")
        return 1
    
    # Find all TIF files
    if args.recursive:
        tif_files = list(input_dir.rglob(args.file_pattern))
    else:
        tif_files = list(input_dir.glob(args.file_pattern))
    
    # Also check for .TIF (uppercase) pattern
    if args.file_pattern == "*.tif":
        if args.recursive:
            tif_files.extend(list(input_dir.rglob("*.TIF")))
        else:
            tif_files.extend(list(input_dir.glob("*.TIF")))
    
    if not tif_files:
        print(f"Error: No files matching pattern {args.file_pattern} found in {input_dir}")
        return 1
    
    # Remove duplicates and sort
    tif_files = sorted(list(set(tif_files)))
    
    print(f"Found {len(tif_files)} TIF files for batch processing")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processing mode: {'Recursive' if args.recursive else 'Non-recursive'}")
    print(f"Parallel processes: {args.parallel}")
    
    # Show overlay movie matching info if applicable
    if args.overlay_movie_dir or args.overlay_movie:
        print(f"Overlay movie matching: prefix_cut={args.cut_overlay_movie_prefix}, suffix_cut={args.cut_overlay_movie_suffix}")
        if args.overlay_movie_dir:
            print(f"Overlay movie directory: {args.overlay_movie_dir}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare processing tasks
    processing_tasks = []
    for tif_file in tif_files:
        # Create output subdirectory for this file
        rel_path = tif_file.relative_to(input_dir)
        file_output_dir = output_path / rel_path.parent / tif_file.stem
        
        # Find appropriate overlay movie for this input file
        overlay_movie = find_overlay_movie(tif_file, args)
        
        processing_tasks.append({
            'input_file': str(tif_file),
            'output_dir': str(file_output_dir),
            'overlay_movie': overlay_movie,
            'args': args
        })
    
    # Process files
    start_time = time.time()
    results = []
    
    if args.parallel > 1:
        # Parallel processing
        print(f"\nStarting parallel processing with {args.parallel} workers...")
        
        # Show overlay movie info for first few tasks (to avoid too much output)
        if args.overlay_movie_dir or args.overlay_movie:
            print("Overlay movie matching (showing first 3 matches):")
            for i, task in enumerate(processing_tasks[:3], 1):
                input_file = Path(task['input_file'])
                overlay_file = Path(task['overlay_movie'])
                if input_file != overlay_file:
                    print(f"  {input_file.name} -> {overlay_file.name}")
            if len(processing_tasks) > 3:
                print(f"  ... and {len(processing_tasks) - 3} more matches")
        
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(process_single_file_wrapper, task): task 
                for task in processing_tasks
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    result = future.result()
                    results.append({
                        'file': task['input_file'],
                        'success': result == 0,
                        'error': None
                    })
                    status = "✓" if result == 0 else "✗"
                    print(f"[{i}/{len(processing_tasks)}] {status} {task['input_file']}")
                except Exception as e:
                    results.append({
                        'file': task['input_file'],
                        'success': False,
                        'error': str(e)
                    })
                    print(f"[{i}/{len(processing_tasks)}] ✗ {task['input_file']} - Error: {e}")
    else:
        # Sequential processing
        print(f"\nStarting sequential processing...")
        for i, task in enumerate(processing_tasks, 1):
            # Show overlay movie info if different from input
            input_file = Path(task['input_file'])
            overlay_file = Path(task['overlay_movie'])
            if input_file != overlay_file:
                print(f"  Using overlay: {input_file.name} -> {overlay_file.name}")
            
            try:
                result = process_single_file_wrapper(task)
                results.append({
                    'file': task['input_file'],
                    'success': result == 0,
                    'error': None
                })
                status = "✓" if result == 0 else "✗"
                print(f"[{i}/{len(processing_tasks)}] {status} {task['input_file']}")
            except Exception as e:
                results.append({
                    'file': task['input_file'],
                    'success': False,
                    'error': str(e)
                })
                print(f"[{i}/{len(processing_tasks)}] ✗ {task['input_file']} - Error: {e}")
    
    # Generate batch summary
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Summary")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Average time per file: {elapsed_time/len(results):.1f} seconds")
    
    if failed > 0:
        print(f"\nFailed files:")
        for result in results:
            if not result['success']:
                print(f"  ✗ {result['file']}")
                if result['error']:
                    print(f"    Error: {result['error']}")
    
    # Save batch summary to file
    summary_path = output_path / "batch_processing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Mouse Sperm Detection - Batch Processing Summary\n")
        f.write("="*60 + "\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Output directory: {output_path}\n")
        f.write(f"File pattern: {args.file_pattern}\n")
        f.write(f"Recursive: {args.recursive}\n")
        f.write(f"Parallel processes: {args.parallel}\n")
        f.write(f"Processing date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total files processed: {len(results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Total time: {elapsed_time:.1f} seconds\n")
        f.write(f"Average time per file: {elapsed_time/len(results):.1f} seconds\n")
        
        if failed > 0:
            f.write(f"\nFailed files:\n")
            for result in results:
                if not result['success']:
                    f.write(f"  ✗ {result['file']}\n")
                    if result['error']:
                        f.write(f"    Error: {result['error']}\n")
    
    print(f"\nDetailed summary saved to: {summary_path}")
    
    return 0 if failed == 0 else 1


def process_single_file_wrapper(task):
    """Wrapper function for parallel processing"""
    args = task['args']
    
    # Create a simple args-like object for process_single_file
    class ArgsWrapper:
        def __init__(self, input_file, output_dir, overlay_movie, args):
            self.input_file = input_file
            self.output_dir = output_dir
            self.overlay_movie = overlay_movie
            # Copy all other attributes from original args
            for attr in dir(args):
                if not attr.startswith('_') and attr not in ['input_file', 'output_dir', 'input_dir', 'overlay_movie']:
                    setattr(self, attr, getattr(args, attr))
    
    wrapped_args = ArgsWrapper(task['input_file'], task['output_dir'], task['overlay_movie'], args)
    
    return process_single_file(wrapped_args, None)


if __name__ == "__main__":
    main()
