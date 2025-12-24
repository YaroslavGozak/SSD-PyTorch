#!/usr/bin/env python3
"""
YouTube-BB Dataset Class Distribution Analyzer

Analyzes XML annotation files to calculate class distribution statistics.
Shows absolute counts and percentages for each class in the dataset.
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path

import yaml

from tools.yt.utils import YTConfig

config = YTConfig()


def parse_xml_annotation(xml_path):
    """
    Parse a single XML annotation file and extract class names.
    
    Args:
        xml_path: Path to the XML annotation file
        
    Returns:
        list: List of class names found in the annotation
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        classes = []
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is not None and name_elem.text:
                classes.append(name_elem.text.strip())
        
        return classes
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return []

def analyze_class_distribution():
    """Analyze class distribution across all annotation files."""
    
    print("YouTube-BB Dataset Class Distribution Analyzer")
    print("=" * 50)
    
    if not os.path.exists(config.annotations_dir):
        print(f"Annotations directory not found: {config.annotations_dir}")
        return
    
    # Dictionary to store class counts
    class_counts = Counter()
    total_annotations = 0
    total_files = 0
    processed_videos = 0
    
    print(f"Scanning annotations directory: {config.annotations_dir}")
    
    # Walk through all video directories
    for annotation_dir in os.listdir(config.annotations_dir):
        annotation_path = os.path.join(config.annotations_dir, annotation_dir)
        
        if not os.path.isdir(annotation_path):
            continue
            
        processed_videos += 1
        video_files = 0
        
        # Process all XML files in this video directory
        for filename in os.listdir(annotation_path):
            if not filename.lower().endswith('.xml'):
                continue
                
            xml_path = os.path.join(annotation_path, filename)
            classes = parse_xml_annotation(xml_path)
            
            if classes:
                total_files += 1
                video_files += 1
                
                # Count classes (should be 1 per file according to YT-BB format)
                for class_name in classes:
                    class_counts[class_name] += 1
                    total_annotations += 1
        
        if processed_videos % 100 == 0:
            print(f"Processed {processed_videos} videos, found {total_annotations} annotations so far...")
    
    print(f"\nScan completed!")
    print(f"Processed videos: {processed_videos}")
    print(f"Total annotation files: {total_files}")
    print(f"Total annotations: {total_annotations}")
    print(f"Unique classes found: {len(class_counts)}")
    
    if total_annotations == 0:
        print("No annotations found!")
        return
    
    # Sort classes by count (descending)
    sorted_classes = class_counts.most_common()
    
    print(f"\nClass Distribution:")
    print("=" * 60)
    print(f"{'Rank':<4} {'Class Name':<25} {'Count':<10} {'Percentage':<10}")
    print("-" * 60)
    
    for rank, (class_name, count) in enumerate(sorted_classes, 1):
        percentage = (count / total_annotations) * 100
        print(f"{rank:<4} {class_name:<25} {count:<10} {percentage:<10.2f}%")
    
    # Statistics summary
    print("\n" + "=" * 60)
    print("Statistics Summary:")
    print(f"Total annotations: {total_annotations:,}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"Most common class: {sorted_classes[0][0]} ({sorted_classes[0][1]:,} instances, {(sorted_classes[0][1]/total_annotations)*100:.2f}%)")
    print(f"Least common class: {sorted_classes[-1][0]} ({sorted_classes[-1][1]:,} instances, {(sorted_classes[-1][1]/total_annotations)*100:.2f}%)")
    
    # Calculate imbalance metrics
    max_count = sorted_classes[0][1]
    min_count = sorted_classes[-1][1]
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"Class imbalance ratio (max/min): {imbalance_ratio:.1f}:1")
    
    # Show top 10 and bottom 10 classes
    print(f"\nTop 10 Most Common Classes:")
    print("-" * 40)
    for rank, (class_name, count) in enumerate(sorted_classes[:10], 1):
        percentage = (count / total_annotations) * 100
        print(f"{rank:2}. {class_name:<20} {count:>8,} ({percentage:5.2f}%)")
    
    if len(sorted_classes) > 10:
        print(f"\nTop 10 Least Common Classes:")
        print("-" * 40)
        for rank, (class_name, count) in enumerate(sorted_classes[-10:], len(sorted_classes)-9):
            percentage = (count / total_annotations) * 100
            print(f"{rank:2}. {class_name:<20} {count:>8,} ({percentage:5.2f}%)")
    
    return class_counts, total_annotations

def save_class_distribution_csv(class_counts, total_annotations):
    """Save class distribution to a CSV file."""
    if not class_counts or total_annotations == 0:
        print("No data to save to CSV.")
        return
    
    csv_path = os.path.join(config.root_dir, 'class_distribution.csv')
    
    try:
        import csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['rank', 'class_name', 'count', 'percentage'])
            
            # Write data sorted by count (descending)
            sorted_classes = class_counts.most_common()
            for rank, (class_name, count) in enumerate(sorted_classes, 1):
                percentage = (count / total_annotations) * 100
                writer.writerow([rank, class_name, count, f"{percentage:.2f}"])
        
        print(f"\nClass distribution saved to: {csv_path}")
        
    except Exception as e:
        print(f"Error saving CSV file: {e}")

def analyze_by_video():
    """Analyze class distribution on a per-video basis."""
    
    print("\nPer-Video Class Analysis:")
    print("=" * 30)
    
    if not os.path.exists(config.annotations_dir):
        print(f"Annotations directory not found: {config.annotations_dir}")
        return
    
    video_class_stats = {}
    
    # Walk through all video directories
    for video_dir in os.listdir(config.annotations_dir):
        video_path = os.path.join(config.annotations_dir, video_dir)
        
        if not os.path.isdir(video_path):
            continue
        
        video_classes = Counter()
        
        # Process all XML files in this video directory
        for filename in os.listdir(video_path):
            if not filename.lower().endswith('.xml'):
                continue
                
            xml_path = os.path.join(video_path, filename)
            classes = parse_xml_annotation(xml_path)
            
            for class_name in classes:
                video_classes[class_name] += 1
        
        if video_classes:
            video_class_stats[video_dir] = video_classes
    
    # Analyze how many videos each class appears in
    class_video_counts = Counter()
    for video_id, video_classes in video_class_stats.items():
        for class_name in video_classes.keys():
            class_video_counts[class_name] += 1
    
    total_videos = len(video_class_stats)
    
    print(f"Class presence across {total_videos} videos:")
    print(f"{'Class Name':<25} {'Videos':<10} {'Video %':<10}")
    print("-" * 45)
    
    for class_name, video_count in class_video_counts.most_common():
        video_percentage = (video_count / total_videos) * 100
        print(f"{class_name:<25} {video_count:<10} {video_percentage:<10.1f}%")

def main():
    """Main function to run the class distribution analysis."""
   
    # Perform main analysis
    class_counts, total_annotations = analyze_class_distribution()
    
    if class_counts:
        # Save results to CSV
        save_class_distribution_csv(class_counts, total_annotations)
        
        # Perform per-video analysis
        analyze_by_video()
        
        print(f"\nAnalysis complete! Results saved to {config.root_dir}")
    
if __name__ == "__main__":
    main()