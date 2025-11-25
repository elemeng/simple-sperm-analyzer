# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Simple Sperm Analyzer is a Python-based tool for mouse sperm head detection, tracking, and motility analysis from TIF image stacks. The project uses computer vision techniques with OpenCV and provides a command-line interface.

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

- **Core Detection Pipeline** (`core.py`): Contains the main sperm detection algorithm using contour analysis with parameters for area, aspect ratio, and solidity filtering
- **Main Entry Point** (`main.py`): CLI interface supporting single-file processing, batch processing with parallel execution, and parameter customization
- **Tracking System** (`tracker.py`): Multi-object tracking using Kalman filters and curvature signatures for sperm trajectory analysis
- **Analysis Modules**: Statistical analysis of motility parameters (VSL, VCL, LIN) following OpenCASA standards

Key architectural patterns:
- Detection parameters are centralized and passed through the pipeline
- Contour-based detection with shape analysis for sperm head identification
- Support for overlay movies to visualize detection results on original/unprocessed data
- Batch processing with parallel execution capabilities

## Development Commands

### Installation and Setup
```bash
# Install dependencies using uv (preferred)
uv sync

# Or using pip
pip install -e .
```

### Running the Application
```bash
# CLI single file processing
python src/main.py input.tif --output_dir results/ --min-area 20 --max-area 45

# Batch processing
python src/main.py --input_dir data/ --output_dir results/ --pattern "*.tif" --parallel 4

# With overlay movie for visualization
python src/main.py input.tif --output_dir results/ --overlay-movie original_movie.tif
```

### Development Workflow
```bash
# Run a specific analysis script
python src/analysis.py tracking.csv --pixel-size 0.5 --fps 60

# Create plots
python src/plot.py
python src/plot_bonferroni.py

# Convert movies to TIF
python src/movie2tif.py input.mp4 output.tif

# Preprocess images
python src/preprocess.py input.tif output.tif
```

## Key Parameters

Detection parameters in `core.py`:
- `min_area`/`max_area`: Size filtering (default: 20-45 pixels)
- `min_aspect`/`max_aspect`: Elongation filtering (default: 1.2-3.0)
- `min_solidity`: Hook shape detection (default: 0.65)
- `threshold`: Binary threshold (default: 10)
- `blur_radius`: Gaussian blur for noise reduction (default: 0.5)

## Data Formats

- **Input**: TIF image stacks (8-bit grayscale)
- **Output**: CSV files with coordinates and detection metrics
- **Visualization**: Marked TIF stacks and PNG summaries
- **Tracking**: CSV with trajectory data for motility analysis

## Dependencies

Core dependencies defined in `pyproject.toml`:
- OpenCV (`opencv-python`) for image processing
- NumPy, Pandas for data handling
- Matplotlib, Seaborn for plotting
- FilterPy for Kalman filtering
- Tifffile for TIFF I/O
- ImageIO for video processing

## Testing

No formal test suite is currently implemented. Manual testing should include:
- Parameter validation in CLI
- Batch processing with various file patterns
- Overlay movie matching functionality
- Export formats (CSV, video, images)