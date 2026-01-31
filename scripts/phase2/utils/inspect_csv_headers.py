#!/usr/bin/env python3
"""
Inspect CSV Headers - Show structure of all experiment files
"""

import os
import glob
from pathlib import Path

def inspect_experiment_files(data_dir='~/generative-ai-workload-modeling/data/raw/phase1'):
    """Show structure of all CSV files in experiment directories."""
    
    data_path = Path(data_dir).expanduser()
    
    # Get all experiment directories
    exp_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    print("="*80)
    print("CSV FILE STRUCTURE INSPECTION")
    print("="*80)
    print(f"\nData directory: {data_path}")
    print(f"Found {len(exp_dirs)} experiment directories\n")
    
    for exp_dir in exp_dirs:
        csv_files = sorted(glob.glob(str(exp_dir / "*.csv")))
        
        if not csv_files:
            continue
            
        print("\n" + "="*80)
        print(f"📁 {exp_dir.name}")
        print("="*80)
        print(f"Number of CSV files: {len(csv_files)}\n")
        
        # Group files by metric type
        metrics = {}
        for csv_file in csv_files:
            filename = Path(csv_file).name
            # Extract metric name (remove workload_rN_ prefix and timestamp suffix)
            parts = filename.split('_')
            
            # Find where the metric name starts (after workload_rN_)
            metric_name = None
            for i, part in enumerate(parts):
                if part.startswith('r') and part[1:].isdigit():
                    # This is the replica number, metric starts after this
                    metric_parts = parts[i+1:]
                    # Remove timestamp and .csv
                    metric_name = '_'.join(metric_parts[:-1])
                    break
            
            if metric_name:
                metrics[metric_name] = csv_file
        
        print(f"Metrics found ({len(metrics)}):")
        for i, metric in enumerate(sorted(metrics.keys()), 1):
            print(f"  {i:2d}. {metric}")
        
        # Show first CSV file in detail
        if csv_files:
            print(f"\n Example file: {Path(csv_files[0]).name}")
            print("-"*80)
            
            try:
                with open(csv_files[0], 'r') as f:
                    header = f.readline().strip()
                    first_row = f.readline().strip()
                    second_row = f.readline().strip()
                    
                    columns = header.split(',')
                    print(f"Columns ({len(columns)}):")
                    for i, col in enumerate(columns, 1):
                        print(f"  {i:2d}. {col}")
                    
                    print(f"\nFirst data row:")
                    print(f"  {first_row[:120]}...")
                    
                    # Count total rows
                    with open(csv_files[0], 'r') as f2:
                        total_lines = sum(1 for _ in f2)
                    
                    print(f"\nFile statistics:")
                    print(f"  Total lines: {total_lines} (1 header + {total_lines-1} data rows)")
                    
                    # Check if has pod column
                    if 'pod' in columns:
                        print(f"  ✓ Has 'pod' column (multi-pod data)")
                    else:
                        print(f"  ✗ No 'pod' column (system-level data)")
                    
            except Exception as e:
                print(f"  Error reading file: {e}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total experiment directories: {len(exp_dirs)}")
    print("\nExpected structure per experiment:")
    print("  • 15 CSV files (one per metric)")
    print("  • Each CSV: timestamp, value, pod (if multi-pod), metadata")
    print("  • Multi-pod CSVs: same timestamp appears N times (once per pod)")
    print("  • System-level CSVs: one value per timestamp (shared across pods)")

if __name__ == "__main__":
    inspect_experiment_files()