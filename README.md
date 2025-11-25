# Simple Sperm Analyzer

A Python-based tool for mouse sperm head detection, tracking, and motility analysis from TIF image stacks. The project uses computer vision techniques with OpenCV and provides a command-line interface for automated sperm analysis following OpenCASA standards.

## Features

- **Automated Sperm Detection**: Contour-based detection with shape analysis for sperm head identification
- **Multi-Object Tracking**: Kalman filter-based tracking for sperm trajectory analysis
- **Motility Analysis**: Statistical analysis of motility parameters (VSL, VCL, LIN) following OpenCASA standards
- **Batch Processing**: Parallel processing of multiple TIF files
- **Visualization**: Overlay movies and marked images for result verification
- **Flexible Parameters**: Customizable detection parameters for different experimental conditions

## Installation

### Using uv (recommended)

```bash
uv sync
```

### Using pip

```bash
pip install -e .
```

## Usage

### Single File Processing

```bash
python src/main.py input.tif --output_dir results/ --min-area 20 --max-area 45
```

### Batch Processing

```bash
python src/main.py --input_dir data/ --output_dir results/ --pattern "*.tif" --parallel 4
```

### With Overlay Movie for Visualization

```bash
python src/main.py input.tif --output_dir results/ --overlay-movie original_movie.tif
```

### Analysis and Visualization

```bash
# Run motility analysis
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

Detection parameters (customizable via CLI):

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

## Architecture

The codebase follows a modular architecture:

- **Core Detection Pipeline** (`core.py`): Main sperm detection algorithm using contour analysis
- **Main Entry Point** (`main.py`): CLI interface with single/batch processing support
- **Tracking System** (`tracker.py`): Multi-object tracking using Kalman filters
- **Analysis Modules**: Statistical analysis of motility parameters

## Dependencies

Core dependencies (defined in `pyproject.toml`):

- OpenCV (`opencv-python`) for image processing
- NumPy, Pandas for data handling
- Matplotlib, Seaborn for plotting
- FilterPy for Kalman filtering
- Tifffile for TIFF I/O
- ImageIO for video processing

## License

MIT License

Copyright (c) 2025 Simple Sperm Analyzer
