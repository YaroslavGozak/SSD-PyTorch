#!/usr/bin/env python3
"""
Intelligent video selection for YouTube-BB dataset to reduce size and improve class balance.
Drops videos containing over-represented classes while preserving rare classes.
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import random
import shutil

from tools.yt.utils import YTConfig

# Configuration
TARGET_DATASET_SIZE = 0.5  # Keep 50% of videos
MIN_FRAMES_PER_CLASS = 1000  # Minimum frames to keep per class
MAX_FRAMES_PER_CLASS = 10000  # Maximum frames to keep per class

def analyze_video_classes(annotations_dir):
    """Analyze which classes each video contains and frame counts."""
    video_class_counts = defaultdict(lambda: defaultdict(int))
    
    for video_id in os.listdir(annotations_dir):
        video_ann_dir = os.path.join(annotations_dir, video_id)
        if not os.path.isdir(video_ann_dir):
            continue
            
        for xml_file in os.listdir(video_ann_dir):
            if not xml_file.endswith('.xml'):
                continue
                
            xml_path = os.path.join(video_ann_dir, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    video_class_counts[video_id][class_name] += 1
                    
            except Exception as e:
                print(f"Error processing {xml_path}: {e}")
    
    return dict(video_class_counts)

def select_videos_for_balanced_dataset(video_class_counts):
    """Select videos to keep based on class balance requirements."""
    
    # Calculate current class totals
    class_totals = defaultdict(int)
    for video_id, classes in video_class_counts.items():
        for class_name, count in classes.items():
            class_totals[class_name] += count
    
    # Sort classes by frequency (rarest first)
    sorted_classes = sorted(class_totals.items(), key=lambda x: x[1])
    
    print("Current class distribution:")
    for class_name, count in sorted_classes:
        print(f"  {class_name}: {count} frames")
    
    # Select videos to keep
    selected_videos = set()
    class_frame_counts = defaultdict(int)
    
    # Priority 1: Keep all videos with rare classes
    rare_threshold = sorted_classes[len(sorted_classes)//3][1]  # Bottom 1/3
    for video_id, classes in video_class_counts.items():
        for class_name, count in classes.items():
            if class_totals[class_name] <= rare_threshold:
                selected_videos.add(video_id)
                for cls, cnt in classes.items():
                    class_frame_counts[cls] += cnt
                break
    
    # Priority 2: Randomly select from remaining videos to balance classes
    remaining_videos = [(vid, classes) for vid, classes in video_class_counts.items() 
                       if vid not in selected_videos]
    random.shuffle(remaining_videos)
    
    for video_id, classes in remaining_videos:
        # Check if adding this video would help balance
        should_add = False
        
        for class_name, count in classes.items():
            if class_frame_counts[class_name] < MIN_FRAMES_PER_CLASS:
                should_add = True
                break
            elif (class_frame_counts[class_name] < MAX_FRAMES_PER_CLASS and 
                  class_totals[class_name] > rare_threshold):
                # Add some videos from popular classes but not too many
                if random.random() < 0.3:  # 30% chance
                    should_add = True
                    break
        
        if should_add:
            selected_videos.add(video_id)
            for cls, cnt in classes.items():
                class_frame_counts[cls] += cnt
        
        # Stop if we've reached target size
        if len(selected_videos) >= len(video_class_counts) * TARGET_DATASET_SIZE:
            break
    
    return selected_videos, class_frame_counts

def create_balanced_dataset(output_dir):
    """Create a balanced subset of the dataset."""
    
    annotations_dir = os.path.join(output_dir, "SequenceAnnotations")
    ims_dir = os.path.join(output_dir, "ResizedSequences")

    print("Analyzing video classes...")
    video_class_counts = analyze_video_classes(annotations_dir)
    
    print(f"Found {len(video_class_counts)} videos")
    
    print("Selecting videos for balanced dataset...")
    selected_videos, final_class_counts = select_videos_for_balanced_dataset(video_class_counts)
    
    print(f"\nSelected {len(selected_videos)} videos ({len(selected_videos)/len(video_class_counts)*100:.1f}%)")
    
    # Show final class distribution
    print("\nFinal class distribution:")
    sorted_final = sorted(final_class_counts.items(), key=lambda x: x[1])
    total_frames = sum(final_class_counts.values())
    for class_name, count in sorted_final:
        percentage = count / total_frames * 100
        print(f"  {class_name}: {count} frames ({percentage:.1f}%)")
    
    print(f"\nTotal frames: {total_frames}")
    
    # Create backup directories
    backup_annotations = os.path.join(output_dir, "SequenceAnnotations_backup")
    backup_images = os.path.join(output_dir, "ResizedSequences_backup")
    
    if not os.path.exists(backup_annotations):
        print("Creating backup...")
        shutil.copytree(annotations_dir, backup_annotations)
        shutil.copytree(ims_dir, backup_images)
        print("✓ Backup created")
    
    # Remove non-selected videos
    removed_videos = 0
    for video_id in os.listdir(annotations_dir):
        if video_id not in selected_videos:
            # Remove from annotations
            video_ann_path = os.path.join(annotations_dir, video_id)
            if os.path.exists(video_ann_path):
                shutil.rmtree(video_ann_path)
            
            # Remove from images
            video_img_path = os.path.join(ims_dir, video_id)
            if os.path.exists(video_img_path):
                shutil.rmtree(video_img_path)
            
            removed_videos += 1
    
    print(f"✓ Removed {removed_videos} videos")
    
    # Update train.txt and val.txt files
    update_split_files(selected_videos, output_dir)

def update_split_files(selected_videos, output_dir):
    """Update train.txt and val.txt to only include selected videos."""

    ims_dir = os.path.join(output_dir, "ResizedSequences")
    
    for split_file in ['train.txt', 'val.txt']:
        split_path = os.path.join(output_dir, split_file)
        if os.path.exists(split_path):
            # Read existing entries
            with open(split_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # Filter to only selected videos
            filtered_lines = []
            for line in lines:
                video_id = line.split('/')[0]
                if video_id in selected_videos:
                    # Verify the files still exist
                    img_path = os.path.join(ims_dir, f"{line}.jpg")
                    if os.path.exists(img_path):
                        filtered_lines.append(line)
            
            # Write back filtered list
            with open(split_path, 'w') as f:
                for line in filtered_lines:
                    f.write(line + '\n')
            
            print(f"✓ Updated {split_file}: {len(filtered_lines)} entries")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results

    config = YTConfig()

    create_balanced_dataset(config.root_dir)