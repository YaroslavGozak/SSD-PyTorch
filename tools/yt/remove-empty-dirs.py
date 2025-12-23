#!/usr/bin/env python3
"""
Clean up empty directories created during YouTube-BB dataset download.

This script removes:
1. Empty video directories in the Images folder
2. Empty video directories in the Annotations folder
3. Video directories that have no corresponding images or annotations
"""

import os
import shutil
from pathlib import Path

# Configuration - update these paths to match your download script
OUTPUT_DIR = r"D:\\YouTube\\ytbb_dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "ResizedSequences")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "SequenceAnnotations")

def is_directory_empty(directory_path):
    """Check if a directory is empty or contains only empty subdirectories."""
    try:
        # Check if directory exists
        if not os.path.exists(directory_path):
            return True
        
        # Get all items in directory
        items = os.listdir(directory_path)
        
        # If no items, it's empty
        if not items:
            return True
        
        # Check if all items are empty directories
        for item in items:
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                return False  # Found a file, not empty
            elif os.path.isdir(item_path):
                if not is_directory_empty(item_path):
                    return False  # Found non-empty subdirectory
        
        return True  # All subdirectories are empty
    except Exception as e:
        print(f"Error checking directory {directory_path}: {e}")
        return False

def count_files_in_directory(directory_path, extensions=None):
    """Count files with specific extensions in a directory."""
    if not os.path.exists(directory_path):
        return 0
    
    count = 0
    try:
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                if extensions is None:
                    count += 1
                else:
                    if any(item.lower().endswith(ext.lower()) for ext in extensions):
                        count += 1
    except Exception as e:
        print(f"Error counting files in {directory_path}: {e}")
    
    return count

def clean_empty_directories():
    """Remove empty video directories from both images and annotations folders."""
    
    print("YouTube-BB Dataset Directory Cleaner")
    print("=" * 40)
    
    # Check if directories exist
    if not os.path.exists(IMAGES_DIR):
        print(f"Images directory not found: {IMAGES_DIR}")
        return
    
    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"Annotations directory not found: {ANNOTATIONS_DIR}")
        return
    
    # Get all video directories
    image_video_dirs = []
    annotation_video_dirs = []
    
    # Scan images directory
    try:
        for item in os.listdir(IMAGES_DIR):
            item_path = os.path.join(IMAGES_DIR, item)
            if os.path.isdir(item_path):
                image_video_dirs.append(item)
    except Exception as e:
        print(f"Error scanning images directory: {e}")
        return
    
    # Scan annotations directory
    try:
        for item in os.listdir(ANNOTATIONS_DIR):
            item_path = os.path.join(ANNOTATIONS_DIR, item)
            if os.path.isdir(item_path):
                annotation_video_dirs.append(item)
    except Exception as e:
        print(f"Error scanning annotations directory: {e}")
        return
    
    print(f"Found {len(image_video_dirs)} video directories in images folder")
    print(f"Found {len(annotation_video_dirs)} video directories in annotations folder")
    
    # Track statistics
    removed_image_dirs = 0
    removed_annotation_dirs = 0
    orphaned_dirs = 0
    
    # Clean empty image directories
    print("\nChecking image directories...")
    for video_id in image_video_dirs:
        video_dir_path = os.path.join(IMAGES_DIR, video_id)
        image_count = count_files_in_directory(video_dir_path, ['.jpg', '.jpeg', '.png'])
        
        if image_count == 0:
            try:
                shutil.rmtree(video_dir_path)
                print(f"Removed empty image directory: {video_id}")
                removed_image_dirs += 1
            except Exception as e:
                print(f"Failed to remove {video_dir_path}: {e}")
        else:
            print(f"Keeping {video_id}: {image_count} images")
    
    # Clean empty annotation directories
    print("\nChecking annotation directories...")
    for video_id in annotation_video_dirs:
        video_dir_path = os.path.join(ANNOTATIONS_DIR, video_id)
        xml_count = count_files_in_directory(video_dir_path, ['.xml'])
        
        if xml_count == 0:
            try:
                shutil.rmtree(video_dir_path)
                print(f"Removed empty annotation directory: {video_id}")
                removed_annotation_dirs += 1
            except Exception as e:
                print(f"Failed to remove {video_dir_path}: {e}")
        else:
            print(f"Keeping {video_id}: {xml_count} annotations")
    
    # Find and clean orphaned directories (annotations without images or vice versa)
    print("\nChecking for orphaned directories...")
    
    # Re-scan after cleanup
    remaining_image_dirs = set()
    remaining_annotation_dirs = set()
    
    if os.path.exists(IMAGES_DIR):
        try:
            remaining_image_dirs = {item for item in os.listdir(IMAGES_DIR) 
                                  if os.path.isdir(os.path.join(IMAGES_DIR, item))}
        except Exception as e:
            print(f"Error re-scanning images directory: {e}")
    
    if os.path.exists(ANNOTATIONS_DIR):
        try:
            remaining_annotation_dirs = {item for item in os.listdir(ANNOTATIONS_DIR) 
                                       if os.path.isdir(os.path.join(ANNOTATIONS_DIR, item))}
        except Exception as e:
            print(f"Error re-scanning annotations directory: {e}")
    
    # Remove annotation directories without corresponding image directories
    for video_id in remaining_annotation_dirs:
        if video_id not in remaining_image_dirs:
            annotation_dir_path = os.path.join(ANNOTATIONS_DIR, video_id)
            try:
                shutil.rmtree(annotation_dir_path)
                print(f"Removed orphaned annotation directory: {video_id}")
                orphaned_dirs += 1
            except Exception as e:
                print(f"Failed to remove orphaned annotation directory {annotation_dir_path}: {e}")
    
    # Remove image directories without corresponding annotation directories
    for video_id in remaining_image_dirs:
        if video_id not in remaining_annotation_dirs:
            image_dir_path = os.path.join(IMAGES_DIR, video_id)
            try:
                shutil.rmtree(image_dir_path)
                print(f"Removed orphaned image directory: {video_id}")
                orphaned_dirs += 1
            except Exception as e:
                print(f"Failed to remove orphaned image directory {image_dir_path}: {e}")
    
    # Final statistics
    print("\n" + "=" * 40)
    print("Cleanup Summary:")
    print(f"  Removed empty image directories: {removed_image_dirs}")
    print(f"  Removed empty annotation directories: {removed_annotation_dirs}")
    print(f"  Removed orphaned directories: {orphaned_dirs}")
    print(f"  Total directories removed: {removed_image_dirs + removed_annotation_dirs + orphaned_dirs}")
    
    # Show final counts
    final_image_dirs = 0
    final_annotation_dirs = 0
    
    if os.path.exists(IMAGES_DIR):
        try:
            final_image_dirs = len([item for item in os.listdir(IMAGES_DIR) 
                                  if os.path.isdir(os.path.join(IMAGES_DIR, item))])
        except Exception as e:
            print(f"Error getting final image directory count: {e}")
    
    if os.path.exists(ANNOTATIONS_DIR):
        try:
            final_annotation_dirs = len([item for item in os.listdir(ANNOTATIONS_DIR) 
                                       if os.path.isdir(os.path.join(ANNOTATIONS_DIR, item))])
        except Exception as e:
            print(f"Error getting final annotation directory count: {e}")
    
    print(f"\nRemaining directories:")
    print(f"  Image directories: {final_image_dirs}")
    print(f"  Annotation directories: {final_annotation_dirs}")

def dry_run():
    """Perform a dry run to show what would be removed without actually removing anything."""
    print("YouTube-BB Dataset Directory Cleaner - DRY RUN")
    print("=" * 50)
    print("This will show what would be removed without actually deleting anything.")
    print()
    
    # Check if directories exist
    if not os.path.exists(IMAGES_DIR):
        print(f"Images directory not found: {IMAGES_DIR}")
        return
    
    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"Annotations directory not found: {ANNOTATIONS_DIR}")
        return
    
    # Get all video directories
    image_video_dirs = []
    annotation_video_dirs = []
    
    # Scan directories
    try:
        for item in os.listdir(IMAGES_DIR):
            item_path = os.path.join(IMAGES_DIR, item)
            if os.path.isdir(item_path):
                image_video_dirs.append(item)
        
        for item in os.listdir(ANNOTATIONS_DIR):
            item_path = os.path.join(ANNOTATIONS_DIR, item)
            if os.path.isdir(item_path):
                annotation_video_dirs.append(item)
    except Exception as e:
        print(f"Error scanning directories: {e}")
        return
    
    print(f"Found {len(image_video_dirs)} video directories in images folder")
    print(f"Found {len(annotation_video_dirs)} video directories in annotations folder")
    
    # Check what would be removed
    would_remove_images = []
    would_remove_annotations = []
    
    print("\nWould remove these empty image directories:")
    for video_id in image_video_dirs:
        video_dir_path = os.path.join(IMAGES_DIR, video_id)
        image_count = count_files_in_directory(video_dir_path, ['.jpg', '.jpeg', '.png'])
        
        if image_count == 0:
            would_remove_images.append(video_id)
            print(f"  - {video_id}")
    
    print(f"\nWould remove these empty annotation directories:")
    for video_id in annotation_video_dirs:
        video_dir_path = os.path.join(ANNOTATIONS_DIR, video_id)
        xml_count = count_files_in_directory(video_dir_path, ['.xml'])
        
        if xml_count == 0:
            would_remove_annotations.append(video_id)
            print(f"  - {video_id}")
    
    print(f"\nSummary:")
    print(f"  Would remove {len(would_remove_images)} empty image directories")
    print(f"  Would remove {len(would_remove_annotations)} empty annotation directories")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        dry_run()
    else:
        print("Use --dry-run to see what would be removed without actually deleting anything.")
        response = input("Do you want to proceed with cleaning empty directories? (y/N): ")
        
        if response.lower() in ['y', 'yes']:
            clean_empty_directories()
        else:
            print("Operation cancelled.")