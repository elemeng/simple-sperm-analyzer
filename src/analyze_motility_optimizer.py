#!/usr/bin/env python3
"""
CLI tool to analyze motility results and find optimal parameters.
Finds results with Maximum, Minimum, and Best Mean (minimum SD) of motile sperm ratio.
"""

import os
import re
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import sys
from io import StringIO


def extract_motile_percentage(summary_file: Path) -> Optional[float]:
    """Extract motile sperm percentage from summary file."""
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
        
        # Look for motile percentage pattern
        motile_match = re.search(r'Motile\s*\(VAP≥5\s*&\s*VSL>\)\s*:\s*(\d+\.?\d*)%', content)
        if motile_match:
            return float(motile_match.group(1))
        
        # Alternative pattern
        motile_match = re.search(r'Motile.*?(\d+\.?\d*)%', content)
        if motile_match:
            return float(motile_match.group(1))
            
        return None
    except Exception as e:
        print(f"Error reading {summary_file}: {e}")
        return None


def collect_motility_data(analysis_dir: Path) -> Dict[str, float]:
    """Collect motility percentages from all analysis folders."""
    motility_data = {}
    
    # Find all analysis subdirectories
    for item in analysis_dir.iterdir():
        # Check for both Enhanced_fraction_ and Enhanced_ patterns
        if item.is_dir() and (item.name.startswith('Enhanced_fraction_') or item.name.startswith('Enhanced_')):
            summary_file = item / f"{item.name}_coords_confirmed_tracks_summary.txt"
            
            if summary_file.exists():
                motile_pct = extract_motile_percentage(summary_file)
                if motile_pct is not None:
                    motility_data[item.name] = motile_pct
    
    return motility_data


def find_optimal_results(motility_data: Dict[str, float]) -> Dict[str, Dict]:
    """Find results with Maximum, Minimum, and Best Mean (minimum SD) motile percentage."""
    if not motility_data:
        return {}
    
    # Convert to list for easier manipulation
    values = list(motility_data.values())
    names = list(motility_data.keys())
    
    # Find maximum and minimum
    max_idx = np.argmax(values)
    min_idx = np.argmin(values)
    
    # For "best mean" - find the value closest to overall mean with minimum deviation
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Calculate distance from mean normalized by standard deviation
    distances_from_mean = np.abs(np.array(values) - mean_val) / std_val
    best_mean_idx = np.argmin(distances_from_mean)
    
    results = {
        'maximum': {
            'name': names[max_idx],
            'value': values[max_idx],
            'type': 'Maximum motile percentage'
        },
        'minimum': {
            'name': names[min_idx], 
            'value': values[min_idx],
            'type': 'Minimum motile percentage'
        },
        'best_mean': {
            'name': names[best_mean_idx],
            'value': values[best_mean_idx],
            'type': f'Best mean (closest to overall mean {mean_val:.2f}%)'
        }
    }
    
    # Add overall statistics
    results['overall'] = {
        'mean': mean_val,
        'std': std_val,
        'min_overall': min(values),
        'max_overall': max(values),
        'count': len(values)
    }
    
    return results


def plot_motility_histogram(motility_data: Dict[str, float], optimal_results: Dict, output_path: Path):
    """Create histogram plot with optimal results highlighted."""
    values = list(motility_data.values())
    mean_val = optimal_results['overall']['mean']
    std_val = optimal_results['overall']['std']
    
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    n, bins, patches = plt.hist(values, bins='auto', alpha=0.7, color='lightblue', edgecolor='black')
    
    # Add mean and standard deviation lines
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}%')
    plt.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2, label=f'Mean + 1 SD: {mean_val + std_val:.2f}%')
    plt.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2, label=f'Mean - 1 SD: {mean_val - std_val:.2f}%')
    
    # Highlight optimal results
    colors = {'maximum': 'green', 'minimum': 'red', 'best_mean': 'purple'}
    for key, result in optimal_results.items():
        if key != 'overall':
            plt.axvline(result['value'], color=colors[key], linestyle='-', linewidth=3, 
                       label=f"{result['type']}: {result['value']:.2f}%")
    
    plt.xlabel('Motile Sperm Percentage (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Motile Sperm Percentage Across Analysis Results\nwith Optimal Results Highlighted', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics text box
    stats_text = f"""Overall Statistics:
Mean: {mean_val:.2f}%
Std Dev: {std_val:.2f}%
Range: {min(values):.2f}% - {max(values):.2f}%
Count: {len(values)} analyses"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_combinations(motility_data: Dict[str, float], top_n: int = 10) -> List[Dict]:
    """Analyze all combinations of 3 results and calculate their statistics."""
    if len(motility_data) < 3:
        return []
    
    combinations_data = []
    items = list(motility_data.items())
    
    # Generate all combinations of 3
    for combo in combinations(items, 3):
        names = [item[0] for item in combo]
        values = [item[1] for item in combo]
        
        combo_mean = np.mean(values)
        combo_std = np.std(values)
        combo_cv = combo_std / combo_mean * 100  # Coefficient of variation
        
        combinations_data.append({
            'combination': names,
            'values': values,
            'mean': combo_mean,
            'std': combo_std,
            'cv': combo_cv,
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values)
        })
    
    # Sort by different criteria
    sorted_by_lowest_std = sorted(combinations_data, key=lambda x: x['std'])
    sorted_by_lowest_cv = sorted(combinations_data, key=lambda x: x['cv'])
    
    return {
        'lowest_std': sorted_by_lowest_std[:top_n],
        'lowest_cv': sorted_by_lowest_cv[:top_n],
        'all_combinations': combinations_data
    }


def print_combinations_analysis(combinations_results: Dict):
    """Print detailed analysis of 3-result combinations."""
    print(f"\n🔬 COMBINATIONS OF 3 RESULTS ANALYSIS:")
    print("="*80)
    
    # Best combinations by lowest standard deviation
    print(f"\n📊 TOP 5 COMBINATIONS WITH LOWEST STANDARD DEVIATION:")
    print("   (Most consistent results - Best reproducibility)")
    for i, combo in enumerate(combinations_results['lowest_std'][:5], 1):
        print(f"\n   #{i}: {combo['combination']}")
        print(f"      Mean: {combo['mean']:.2f}% ± {combo['std']:.2f}%")
        print(f"      Range: {combo['min']:.2f}% - {combo['max']:.2f}% (span: {combo['range']:.2f}%)")
        print(f"      CV: {combo['cv']:.2f}%")
    
    # Best combinations by lowest coefficient of variation
    print(f"\n📈 TOP 5 COMBINATIONS WITH LOWEST COEFFICIENT OF VARIATION:")
    print("   (Most stable relative to their mean)")
    for i, combo in enumerate(combinations_results['lowest_cv'][:5], 1):
        print(f"\n   #{i}: {combo['combination']}")
        print(f"      Mean: {combo['mean']:.2f}% ± {combo['std']:.2f}%")
        print(f"      Range: {combo['min']:.2f}% - {combo['max']:.2f}% (span: {combo['range']:.2f}%)")
        print(f"      CV: {combo['cv']:.2f}%")


def print_results(optimal_results: Dict, motility_data: Dict[str, float], combinations_results: Dict = None):
    """Print detailed results to console."""
    print("\n" + "="*80)
    print("MOTILITY ANALYSIS OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total analyses: {optimal_results['overall']['count']}")
    print(f"   Mean motile %: {optimal_results['overall']['mean']:.2f}%")
    print(f"   Standard deviation: {optimal_results['overall']['std']:.2f}%")
    print(f"   Range: {optimal_results['overall']['min_overall']:.2f}% - {optimal_results['overall']['max_overall']:.2f}%")
    
    print(f"\n🎯 OPTIMAL INDIVIDUAL RESULTS:")
    for key, result in optimal_results.items():
        if key != 'overall':
            print(f"\n   {result['type'].upper()}:")
            print(f"   📁 Folder: {result['name']}")
            print(f"   📊 Value: {result['value']:.2f}%")
            print(f"   📈 Deviation from mean: {abs(result['value'] - optimal_results['overall']['mean']):.2f}%")
    
    if combinations_results:
        print_combinations_analysis(combinations_results)
    
    print(f"\n📋 ALL RESULTS (sorted by motile %):")
    sorted_results = sorted(motility_data.items(), key=lambda x: x[1])
    for name, value in sorted_results:
        print(f"   {name}: {value:.2f}%")
    
    print("\n" + "="*80)


def save_output_to_markdown(output_content: str, output_path: Path):
    """Save the console output to a markdown file with proper formatting."""
    # Convert the console output to markdown format
    md_content = f"""# Motility Analysis Optimization Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Summary

This report contains the analysis of motility results with the following objectives:
- Find Maximum, Minimum, and Best Mean (minimum SD) individual results
- Analyze all combinations of 3 results for optimal consistency
- Provide statistical insights for parameter optimization

---

## Console Output

```
{output_content}
```

---

## Key Findings

### Individual Results
- **Maximum motile percentage**: Found in the analysis with highest motility rate
- **Minimum motile percentage**: Found in the analysis with lowest motility rate  
- **Best mean**: Result closest to the overall mean (most representative)

### Combination Analysis
- All possible combinations of 3 results were analyzed
- Ranked by lowest standard deviation (most consistent)
- Ranked by lowest coefficient of variation (most stable relative to mean)

### Statistical Metrics
- **Mean ± SD**: Average motility percentage with standard deviation
- **CV (Coefficient of Variation)**: Relative variability measure (std/mean × 100)
- **Range**: Min to max values in each combination

---

*This report was generated automatically by the Motility Analysis Optimizer Tool*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return output_path


class OutputCapture:
    """Context manager to capture stdout output."""
    def __init__(self):
        self.output = StringIO()
        self.original_stdout = None
        
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self.output
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        
    def get_output(self):
        return self.output.getvalue()


def main():
    parser = argparse.ArgumentParser(description='Analyze motility results and find optimal parameters')
    parser.add_argument('analysis_dir', type=Path, help='Path to analysis results directory')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for plots and results (default: same as input directory)')
    
    # Save options - enabled by default, can be disabled with negative flags
    parser.add_argument('--no-save-data', action='store_false', dest='save_data', default=True,
                       help='Disable saving detailed motility data as CSV')
    parser.add_argument('--no-save-markdown', action='store_false', dest='save_markdown', default=True,
                       help='Disable saving console output to markdown file')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.analysis_dir.exists():
        print(f"Error: Analysis directory '{args.analysis_dir}' does not exist")
        return 1
    
    # Set default output directory to input directory if not specified
    if args.output_dir is None:
        args.output_dir = args.analysis_dir
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Capture output if markdown save is requested
    if args.save_markdown:
        with OutputCapture() as capture:
            print(f"🔍 Analyzing motility results in: {args.analysis_dir}")
            print(f"📁 Output directory: {args.output_dir}")
            
            # Collect motility data
            motility_data = collect_motility_data(args.analysis_dir)
            
            if not motility_data:
                print("❌ No motility data found in the analysis directory")
                print("   Make sure the directory contains analysis subfolders with summary files")
                return 1
            
            print(f"✅ Found {len(motility_data)} analysis results")
            
            # Find optimal results
            optimal_results = find_optimal_results(motility_data)
            
            # Analyze combinations of 3 results
            combinations_results = analyze_combinations(motility_data, top_n=5)
            
            # Print results
            print_results(optimal_results, motility_data, combinations_results)
            
            # Create and save plot
            plot_path = args.output_dir / 'motility_distribution.png'
            plot_motility_histogram(motility_data, optimal_results, plot_path)
            print(f"📊 Plot saved to: {plot_path}")
            
            # Save detailed data if requested
            if args.save_data:
                csv_path = args.output_dir / 'motility_data.csv'
                df = pd.DataFrame(list(motility_data.items()), columns=['Analysis_Folder', 'Motile_Percentage'])
                df = df.sort_values('Motile_Percentage')
                df.to_csv(csv_path, index=False)
                print(f"📄 Detailed data saved to: {csv_path}")
            
            # Save optimal results summary
            summary_path = args.output_dir / 'optimal_results.json'
            output_data = {
                'individual_results': optimal_results,
                'combinations_analysis': combinations_results
            }
            with open(summary_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"📝 Summary saved to: {summary_path}")
        
        # Save captured output to markdown
        captured_output = capture.get_output()
        markdown_path = args.output_dir / 'motility_analysis_report.md'
        save_output_to_markdown(captured_output, markdown_path)
        print(f"\n📄 Report saved to: {markdown_path}")
        
        # Also print the output to console
        print(captured_output)
        
    else:
        # Normal execution without capture
        print(f"🔍 Analyzing motility results in: {args.analysis_dir}")
        print(f"📁 Output directory: {args.output_dir}")
        
        # Collect motility data
        motility_data = collect_motility_data(args.analysis_dir)
        
        if not motility_data:
            print("❌ No motility data found in the analysis directory")
            print("   Make sure the directory contains analysis subfolders with summary files")
            return 1
        
        print(f"✅ Found {len(motility_data)} analysis results")
        
        # Find optimal results
        optimal_results = find_optimal_results(motility_data)
        
        # Analyze combinations of 3 results
        combinations_results = analyze_combinations(motility_data, top_n=5)
        
        # Print results
        print_results(optimal_results, motility_data, combinations_results)
        
        # Create and save plot
        plot_path = args.output_dir / 'motility_distribution.png'
        plot_motility_histogram(motility_data, optimal_results, plot_path)
        print(f"📊 Plot saved to: {plot_path}")
        
        # Save detailed data if requested
        if args.save_data:
            csv_path = args.output_dir / 'motility_data.csv'
            df = pd.DataFrame(list(motility_data.items()), columns=['Analysis_Folder', 'Motile_Percentage'])
            df = df.sort_values('Motile_Percentage')
            df.to_csv(csv_path, index=False)
            print(f"📄 Detailed data saved to: {csv_path}")
        
        # Save optimal results summary
        summary_path = args.output_dir / 'optimal_results.json'
        output_data = {
            'individual_results': optimal_results,
            'combinations_analysis': combinations_results
        }
        with open(summary_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"📝 Summary saved to: {summary_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())