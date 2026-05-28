#!/usr/bin/env python3
"""
Merge fixed_padding_roi_crop_NNN_results/mAp.csv files into fixed_padding_results.csv

Usage:
    python merge_csv.py                                           # Process all models
    python merge_csv.py --target-model-dir voc-roissd             # Process specific model by name
    python merge_csv.py --target-model-dir /path/to/model         # Process specific model by full path
"""

import os
import csv
import re
import argparse
from pathlib import Path


def merge_padding_results(model_folder):
    """
    Merge all fixed_padding_roi_crop_NNN_results/mAp.csv files into fixed_padding_results.csv
    
    Args:
        model_folder: Path to the model folder (e.g., trained_models/voc-roissd)
    
    Returns:
        Number of rows merged, or None if no results found
    """
    pattern = r'fixed_padding_roi_crop_(?:yolo_)?(\d+)_results'
    results = []
    
    # Find all matching folders
    for folder_name in sorted(os.listdir(model_folder)):
        match = re.match(pattern, folder_name)
        if not match:
            continue
        
        crop_px = int(match.group(1))
        csv_path = os.path.join(model_folder, folder_name, 'mAp.csv')
        
        if os.path.exists(csv_path):
            # Read the CSV and add crop_px column
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['crop_px'] = crop_px
                    results.append(row)
            print(f"  ✓ Added {folder_name} (crop_px={crop_px})")
        else:
            print(f"  ✗ Missing: {csv_path}")
    
    # Write merged CSV with crop_px as first column
    if results:
        results.sort(key=lambda row: int(row['crop_px']))
        output_path = os.path.join(model_folder, 'fixed_padding_results.csv')
        fieldnames = ['crop_px'] + [k for k in results[0].keys() if k != 'crop_px']
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"  → Created: {output_path} ({len(results)} rows)\n")
        return len(results)
    else:
        print(f"  → No results found\n")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Merge fixed_padding_roi_crop_*_results/mAp.csv files into fixed_padding_results.csv'
    )
    parser.add_argument(
        '--target-model-dir',
        type=str,
        default=None,
        help='Specific model directory to process (model name under trained_models or full path)',
    )
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models')
    
    if not os.path.exists(base_dir):
        print(f"Error: trained_models directory not found at {base_dir}")
        return
    
    # Check if model name/path was provided
    if args.target_model_dir:
        model_spec = args.target_model_dir
        
        # Check if it's a full path
        if os.path.isdir(model_spec):
            model_folder = model_spec
            model_name = os.path.basename(model_spec)
        else:
            # Treat as model name
            model_folder = os.path.join(base_dir, model_spec)
            model_name = model_spec
        
        if not os.path.isdir(model_folder):
            print(f"Error: Model directory not found: {model_folder}")
            raise SystemExit(1)
        
        print(f"Processing: {model_name}\n")
        result = merge_padding_results(model_folder)
        
        if result is not None:
            print(f"✓ Done! Merged {result} rows")
        else:
            print("✗ No results to merge")
    else:
        # Process all models
        print(f"Scanning: {base_dir}\n")
        
        model_folders = sorted([
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ])
        
        processed = 0
        for model_name in model_folders:
            model_folder = os.path.join(base_dir, model_name)
            print(f"Processing: {model_name}")
            result = merge_padding_results(model_folder)
            if result is not None:
                processed += 1
        
        print(f"\n✓ Done! Processed {processed} model(s)")


if __name__ == '__main__':
    main()