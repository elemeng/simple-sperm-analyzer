#!/usr/bin/env python3
"""
Parallel Sperm Movie Enhancement – single file OR batch folder
Frame-range selection + preview still supported.
Added: Contrast stretching with min/max or auto-percentile.
"""

import argparse
import multiprocessing
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm

warnings.filterwarnings("ignore")


def apply_contrast_stretch(
    img: np.ndarray, min_in: Optional[int], max_in: Optional[int], auto_contrast: bool
) -> np.ndarray:
    """Apply contrast stretching to float32 image."""
    if not (auto_contrast or (min_in is not None and max_in is not None)):
        return img

    # Determine contrast range
    if auto_contrast:
        # Use percentiles for robust auto-contrast
        min_val = np.percentile(img, 1)
        max_val = np.percentile(img, 99)
    else:
        min_val = float(min_in)
        max_val = float(max_in)

    if max_val <= min_val:
        return img

    # Apply contrast stretching
    scale = 255.0 / (max_val - min_val)
    stretched = np.clip((img - min_val) * scale, 0, 255)
    return stretched


# ---------------------------------------------------------------------------
#  single-frame processor
# ---------------------------------------------------------------------------
def process_frame(
    frame_info: Tuple[int, np.ndarray],
    upscale: float,
    clahe_clip: float,
    sharpen: float,
    denoise: float,
    contrast_min: Optional[int],
    contrast_max: Optional[int],
    auto_contrast: bool,
) -> Tuple[int, np.ndarray]:
    idx, img = frame_info
    # if RGB → convert to gray first
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)

    # Apply contrast stretching
    img = apply_contrast_stretch(img, contrast_min, contrast_max, auto_contrast)

    # 1. upscale
    if upscale > 1.0:
        h, w = img.shape
        img = cv2.resize(
            img, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC
        )

    # 2. CLAHE
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        img_u8 = clahe.apply(img_u8)

    # 3. unsharp
    if sharpen > 0:
        blurred = cv2.GaussianBlur(img_u8, (5, 5), 0)
        img_u8 = cv2.addWeighted(img_u8, 1 + sharpen, blurred, -sharpen, 0)

    # 4. optional denoise
    if denoise > 0:
        img_u8 = cv2.bilateralFilter(
            img_u8, d=5, sigmaColor=denoise, sigmaSpace=denoise
        )

    return idx, img_u8  # ← already 8-bit grayscale


# ---------------------------------------------------------------------------
#  enhance ONE movie
# ---------------------------------------------------------------------------
def enhance_movie(
    in_path: Path,
    out_path: Path,
    start_frame: int,
    end_frame: Optional[int],
    upscale: float,
    clahe_clip: float,
    sharpen: float,
    denoise: float,
    contrast_min: Optional[int],
    contrast_max: Optional[int],
    auto_contrast: bool,
    save_preview: bool,
    workers: int,
) -> None:
    """Process a single TIFF stack."""
    print(f"\nLoading: {in_path.name}")
    stack = tiff.imread(in_path)
    if stack.ndim == 2:
        stack = stack[None, ...]  # ensure 3-D

    total = len(stack)
    end_frame = end_frame if end_frame is not None else total
    if not (0 <= start_frame < end_frame <= total):
        raise ValueError(f"Frame range {start_frame}:{end_frame} invalid (0-{total})")
    stack = stack[start_frame:end_frame]
    num_frames = len(stack)

    print(f"Processing frames {start_frame}:{end_frame}  ({num_frames} frames)")

    # preview (frame 0 of the selected range)
    if save_preview:
        preview_path = out_path.with_suffix(".png")
        _, preview = process_frame(
            (0, stack[0]),
            upscale,
            clahe_clip,
            sharpen,
            denoise,
            contrast_min,
            contrast_max,
            auto_contrast,
        )
        cv2.imwrite(str(preview_path), preview)
        print(f"  Preview saved: {preview_path.name}")

    # parallel processing
    jobs = [(i, stack[i]) for i in range(num_frames)]
    out_frames = [None] * num_frames

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(
                process_frame,
                j,
                upscale,
                clahe_clip,
                sharpen,
                denoise,
                contrast_min,
                contrast_max,
                auto_contrast,
            ): j[0]
            for j in jobs
        }
        for fut in tqdm(
            as_completed(futures), total=num_frames, desc="Frames", unit="fr"
        ):
            idx, frame = fut.result()
            out_frames[idx] = frame

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving: {out_path.name}")
    tiff.imwrite(out_path, out_frames, dtype=np.uint8, bigtiff=True)
    print("✓ Complete!")


# ---------------------------------------------------------------------------
#  find TIFF files
# ---------------------------------------------------------------------------
def find_tiffs(root: Path, recursive: bool, glob_pat: str) -> List[Path]:
    pat = f"**/{glob_pat}" if recursive else glob_pat
    return sorted(root.glob(pat))


# ---------------------------------------------------------------------------
#  argument parser
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel sperm-movie enhancement – single file or batch folder",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # single file (old style)
  python preprocess.py movie.tif out.tif --save-preview

  # batch folder (non-recursive) → Enhanced_*.tif
  python preprocess.py --input ./stacks/ --save-preview --glob "*.tif" --prefix "Enhanced"

  # batch recursive → ./results/…
  python preprocess.py -i ./stacks/ -r -o ./results/ -p "Clean" --clahe 3 --sharpen 2
  
  # contrast stretching with manual min/max
  python preprocess.py movie.tif out.tif --contrast-min 88 --contrast-max 167 --save-preview
  
  # contrast stretching with auto-percentile
  python preprocess.py movie.tif out.tif --auto-contrast --save-preview
""",
    )
    # positional (optional when --input used)
    p.add_argument("input", help="Input TIFF file or folder (use --input for folder)")
    p.add_argument(
        "output",
        nargs="?",
        help="Output TIFF file or folder (optional when --input used)",
    )

    # batch flags
    p.add_argument(
        "-i",
        "--input",
        action="store_true",
        help="Treat first positional arg as folder (batch mode)",
    )
    p.add_argument("-r", "--recursive", action="store_true", help="Scan sub-folders")
    p.add_argument(
        "--glob", default="*.tif*", help="File pattern for batch mode (default: *.tif*)"
    )
    p.add_argument(
        "--prefix",
        default="Enhanced",
        help="Prefix for output files (default: Enhanced)",
    )

    # output directory (batch only)
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory (batch).  Default: same as input",
    )

    # processing params
    p.add_argument(
        "--start-frame", type=int, default=0, help="First frame to process (default: 0)"
    )
    p.add_argument("--end-frame", type=int, help="Last frame to process (exclusive)")
    p.add_argument(
        "--upscale",
        type=float,
        default=2.0,
        help="Resolution multiplier 1.0-4.0 (default: 2.0)",
    )
    p.add_argument(
        "--clahe", type=float, default=2.0, help="CLAHE clip 0=off, 1-5 (default: 2.0)"
    )
    p.add_argument(
        "--sharpen", type=float, default=1.5, help="Unsharp strength 0-3 (default: 1.5)"
    )
    p.add_argument(
        "--denoise",
        type=float,
        default=0.0,
        help="Bilateral sigma 0=off, 1-15 (default: 0)",
    )
    p.add_argument(
        "--contrast-min",
        type=int,
        help="Minimum input pixel value for contrast stretching (auto if not set)",
    )
    p.add_argument(
        "--contrast-max",
        type=int,
        help="Maximum input pixel value for contrast stretching (auto if not set)",
    )
    p.add_argument(
        "--auto-contrast",
        action="store_true",
        help="Automatically determine contrast range from 1st-99th percentile",
    )
    p.add_argument(
        "--workers", type=int, default=multiprocessing.cpu_count(), help="CPU workers"
    )
    p.add_argument(
        "--save-preview",
        action="store_true",
        help="Export first processed frame as PNG",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ----  determine mode  ----
    in_path = Path(args.input)
    if args.input or in_path.is_dir():  # BATCH
        root = in_path
        out_dir = args.output_dir or root
        out_dir.mkdir(parents=True, exist_ok=True)
        tiffs = find_tiffs(root, args.recursive, args.glob)
        if not tiffs:
            print("No TIFF files found.")
            sys.exit(1)
        print(f"Batch mode: {len(tiffs)} file(s)  ->  {out_dir}")
        for src in tqdm(tiffs, desc="Files", unit="file"):
            # Preserve folder structure relative to root input directory
            if src.is_relative_to(root):
                relative_path = src.relative_to(root)
                if relative_path.parent != Path("."):
                    dst_dir = out_dir / relative_path.parent
                else:
                    dst_dir = out_dir
            else:
                dst_dir = out_dir / src.parent.name

            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{args.prefix}_{src.name}"

            enhance_movie(
                src,
                dst,
                args.start_frame,
                args.end_frame,
                args.upscale,
                args.clahe,
                args.sharpen,
                args.denoise,
                args.contrast_min,
                args.contrast_max,
                args.auto_contrast,
                args.save_preview,
                args.workers,
            )

    else:  # SINGLE
        if args.output is None:
            print("Output file required for single-file mode.")
            sys.exit(1)
        enhance_movie(
            in_path,
            Path(args.output),
            args.start_frame,
            args.end_frame,
            args.upscale,
            args.clahe,
            args.sharpen,
            args.denoise,
            args.contrast_min,
            args.contrast_max,
            args.auto_contrast,
            args.save_preview,
            args.workers,
        )


if __name__ == "__main__":
    main()
