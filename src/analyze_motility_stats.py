#!/usr/bin/env python3
"""
Motility Statistics Analyzer - Extract and analyze motility percentages from analysis subdirectories

This tool parses the motility statistics from sperm analysis results and computes
mean ± SD for Motile (VAP≥5 & VSL>0) and Progressive (STR≥80 & VAP≥50) percentages.
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import sys


def parse_motility_stats(file_path: Path) -> Dict[str, float]:
    """Parse motility statistics from a summary or report file."""
    stats = {'motile': None, 'progressive': None}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for Motile percentage pattern: "Motile (VAP≥5 & VSL>0):    88.48%"
            motile_match = re.search(r'Motile\s*\(VAP≥5\s*&\s*VSL\u003e0\):\s*(\d+(?:\.\d+)?)%', content)
            if motile_match:
                stats['motile'] = float(motile_match.group(1))
            
            # Look for Progressive percentage pattern: "Progressive (STR≥80 & VAP≥50):  23.67%"
            progressive_match = re.search(r'Progressive\s*\(STR≥80\s*&\s*VAP≥50\):\s*(\d+(?:\.\d+)?)%', content)
            if progressive_match:
                stats['progressive'] = float(progressive_match.group(1))
                
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        
    return stats


def find_analysis_files(root_dir: Path, recursive: bool = True) -> List[Path]:
    """Find analysis result files in subdirectories."""
    files = []
    
    # Look for different types of analysis files
    patterns = [
        '*_coords_confirmed_tracks_summary.txt',
        '*_coords_confirmed_tracks_report.md', 
        '*_coords_confirmed_tracks_statistics.csv',
        'batch_combined_summary.txt',
        'batch_combined_summary.json'
    ]
    
    for pattern in patterns:
        if recursive:
            found_files = list(root_dir.rglob(pattern))
        else:
            found_files = list(root_dir.glob(pattern))
        files.extend(found_files)
    
    return sorted(list(set(files)))  # Remove duplicates


def extract_motility_data(files: List[Path]) -> Tuple[List[float], List[float], List[str]]:
    """Extract motility percentages from analysis files."""
    motile_percentages = []
    progressive_percentages = []
    file_names = []
    
    for file_path in files:
        stats = parse_motility_stats(file_path)
        
        if stats['motile'] is not None:
            motile_percentages.append(stats['motile'])
            file_names.append(file_path.parent.name)  # Use parent directory name
            
            if stats['progressive'] is not None:
                progressive_percentages.append(stats['progressive'])
            else:
                progressive_percentages.append(np.nan)
    
    return motile_percentages, progressive_percentages, file_names


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean, SD, and other statistics."""
    if not values:
        return {'mean': np.nan, 'sd': np.nan, 'n': 0, 'min': np.nan, 'max': np.nan}
    
    values_array = np.array(values)
    values_array = values_array[~np.isnan(values_array)]  # Remove NaN values
    
    if len(values_array) == 0:
        return {'mean': np.nan, 'sd': np.nan, 'n': 0, 'min': np.nan, 'max': np.nan}
    
    return {
        'mean': np.mean(values_array),
        'sd': np.std(values_array, ddof=1),  # Sample standard deviation
        'n': len(values_array),
        'min': np.min(values_array),
        'max': np.max(values_array)
    }


def print_results(motile_stats: Dict[str, float], progressive_stats: Dict[str, float], 
                 file_names: List[str], verbose: bool = False):
    """Print analysis results."""
    
    print("=" * 60)
    print("MOTILITY STATISTICS ANALYSIS")
    print("=" * 60)
    print(f"Files analyzed: {len(file_names)}")
    
    if verbose and file_names:
        print("\nSample directories analyzed:")
        for i, name in enumerate(file_names[:5]):
            print(f"  {i+1}. {name}")
        if len(file_names) > 5:
            print(f"  ... and {len(file_names) - 5} more")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Motile statistics
    if not np.isnan(motile_stats['mean']):
        print(f"\n🎯 MOTILE (VAP≥5 & VSL>0):")
        print(f"   Mean: {motile_stats['mean']:.2f}%")
        print(f"   SD:   {motile_stats['sd']:.2f}%")
        print(f"   Range: {motile_stats['min']:.2f}% - {motile_stats['max']:.2f}%")
        print(f"   n = {motile_stats['n']}")
    else:
        print(f"\n🎯 MOTILE (VAP≥5 & VSL>0): No valid data found")
    
    # Progressive statistics  
    if not np.isnan(progressive_stats['mean']):
        print(f"\n🚀 PROGRESSIVE (STR≥80 & VAP≥50):")
        print(f"   Mean: {progressive_stats['mean']:.2f}%")
        print(f"   SD:   {progressive_stats['sd']:.2f}%")
        print(f"   Range: {progressive_stats['min']:.2f}% - {progressive_stats['max']:.2f}%")
        print(f"   n = {progressive_stats['n']}")
    else:
        print(f"\n🚀 PROGRESSIVE (STR≥80 & VAP≥50): No valid data found")
    
    print("\n" + "=" * 60)


def save_results_to_file(motile_stats: Dict[str, float], progressive_stats: Dict[str, float],
                        file_names: List[str], output_file: Path):
    """Save results to a text file."""
    with open(output_file, 'w') as f:
        f.write("Motility Statistics Analysis Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Analysis date: {np.datetime64('now')}\n")
        f.write(f"Files analyzed: {len(file_names)}\n\n")
        
        f.write("RESULTS\n")
        f.write("=" * 60 + "\n")
        
        if not np.isnan(motile_stats['mean']):
            f.write(f"MOTILE (VAP≥5 & VSL>0):\n")
            f.write(f"  Mean ± SD: {motile_stats['mean']:.2f} ± {motile_stats['sd']:.2f}%\n")
            f.write(f"  Range: {motile_stats['min']:.2f}% - {motile_stats['max']:.2f}%\n")
            f.write(f"  n = {motile_stats['n']}\n\n")
        
        if not np.isnan(progressive_stats['mean']):
            f.write(f"PROGRESSIVE (STR≥80 & VAP≥50):\n")
            f.write(f"  Mean ± SD: {progressive_stats['mean']:.2f} ± {progressive_stats['sd']:.2f}%\n")
            f.write(f"  Range: {progressive_stats['min']:.2f}% - {progressive_stats['max']:.2f}%\n")
            f.write(f"  n = {progressive_stats['n']}\n\n")
        
        if file_names:
            f.write("Sample directories analyzed:\n")
            for i, name in enumerate(file_names[:10]):
                f.write(f"  {i+1}. {name}\n")
            if len(file_names) > 10:
                f.write(f"  ... and {len(file_names) - 10} more\n")
    
    print(f"\nResults saved to: {output_file}")


def save_results_to_csv(motile_stats: Dict[str, float], progressive_stats: Dict[str, float],
                       file_names: List[str], output_file: Path):
    """Save results to a CSV file."""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Parameter', 'Mean (%)', 'SD (%)', 'Min (%)', 'Max (%)', 'n'])
        
        # Motile data
        if not np.isnan(motile_stats['mean']):
            writer.writerow([
                'Motile (VAP≥5 & VSL>0)',
                f"{motile_stats['mean']:.2f}",
                f"{motile_stats['sd']:.2f}",
                f"{motile_stats['min']:.2f}",
                f"{motile_stats['max']:.2f}",
                motile_stats['n']
            ])
        
        # Progressive data
        if not np.isnan(progressive_stats['mean']):
            writer.writerow([
                'Progressive (STR≥80 & VAP≥50)',
                f"{progressive_stats['mean']:.2f}",
                f"{progressive_stats['sd']:.2f}",
                f"{progressive_stats['min']:.2f}",
                f"{progressive_stats['max']:.2f}",
                progressive_stats['n']
            ])
    
    print(f"CSV results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze motility statistics from sperm analysis subdirectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze current directory recursively
  python analyze_motility_stats.py
  
  # Analyze specific directory
  python analyze_motility_stats.py /path/to/analysis/results
  
  # Non-recursive analysis
  python analyze_motility_stats.py /path/to/analysis --no-recursive
  
  # Save results to custom file
  python analyze_motility_stats.py /path/to/analysis --output custom_report.txt
  
  # Verbose output showing all analyzed files
  python analyze_motility_stats.py /path/to/analysis --verbose
        """
    )
    
    parser.add_argument(
        'analysis_dir',
        nargs='?',
        default='.',
        help='Directory containing analysis results (default: current directory)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Only analyze files in the specified directory, not subdirectories'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Save results to specified file (optional, default: auto-save to analysis_dir)'
    )
    
    parser.add_argument(
        '--no-auto-save',
        action='store_true',
        help='Disable automatic saving of results to motility_report.txt and motility_report.csv'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information about analyzed files'
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    analysis_dir = Path(args.analysis_dir)
    
    if not analysis_dir.exists():
        print(f"Error: Directory '{analysis_dir}' does not exist")
        sys.exit(1)
    
    print(f"Analyzing motility statistics in: {analysis_dir}")
    print(f"Mode: {'Recursive' if not args.no_recursive else 'Non-recursive'}")
    print("Searching for analysis files...")
    
    # Find analysis files
    files = find_analysis_files(analysis_dir, recursive=not args.no_recursive)
    
    if not files:
        print("Error: No analysis files found")
        print("Looking for files matching patterns:")
        for pattern in ['*_coords_confirmed_tracks_summary.txt', '*_coords_confirmed_tracks_report.md', 
                       '*_coords_confirmed_tracks_statistics.csv', 'batch_combined_summary.txt']:
            print(f"  - {pattern}")
        sys.exit(1)
    
    print(f"Found {len(files)} analysis files")
    
    # Extract motility data
    motile_percentages, progressive_percentages, file_names = extract_motility_data(files)
    
    if not motile_percentages and not progressive_percentages:
        print("Error: No valid motility statistics found in the analysis files")
        print("Make sure files contain patterns like:")
        print("  'Motile (VAP≥5 & VSL>0):    88.48%'")
        print("  'Progressive (STR≥80 & VAP≥50):  23.67%'")
        sys.exit(1)
    
    # Compute statistics
    motile_stats = compute_statistics(motile_percentages)
    progressive_stats = compute_statistics(progressive_percentages)
    
    # Print results
    print_results(motile_stats, progressive_stats, file_names, args.verbose)
    
    # Auto-save results by default (unless disabled)
    if not args.no_auto_save:
        # Default filenames in the analysis directory
        txt_report = analysis_dir / "motility_report.txt"
        csv_report = analysis_dir / "motility_report.csv"
        
        save_results_to_file(motile_stats, progressive_stats, file_names, txt_report)
        save_results_to_csv(motile_stats, progressive_stats, file_names, csv_report)
    
    # Also save to custom location if requested
    if args.output:
        save_results_to_file(motile_stats, progressive_stats, file_names, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())