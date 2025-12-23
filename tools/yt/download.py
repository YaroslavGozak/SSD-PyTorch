#!/usr/bin/env python3
"""
YouTube-BB Dataset Downloader

Downloads video frames from YouTube based on the YouTube-BB detection annotations.
Based on: https://research.google.com/youtube-bb/download.html

CSV format: youtube_id,timestamp_ms,class_id,class_name,object_id,object_presence,xmin,ymin,xmax,ymax
Sample: AAB6lO-XiKE,238000,0,person,0,present,0.482,0.54,0.37166667,0.6166667

Requirements:
- pip install yt-dlp opencv-python pandas
"""

import os
import sys
import csv
import cv2
import pandas as pd
import subprocess
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

# Configuration
CSV_TRAIN_FILE = r"D:\\YouTube\\yt_bb_detection_train.csv\\youtube_boundingboxes_detection_train.csv"
CSV_VAL_FILE = r"D:\\YouTube\\yt_bb_detection_validation.csv\\youtube_boundingboxes_detection_validation.csv"
OUTPUT_DIR = r"D:\\YouTube\\ytbb_dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "ResizedSequences")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "SequenceAnnotations")
MAX_VIDEOS = None  # Limit for testing, set to None for all videos
FRAME_WIDTH = 640  # Target frame width
FRAME_HEIGHT = 480  # Target frame height

def setup_directories():
    """Create output directories if they don't exist."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    print(f"Output directories created:")
    print(f"  Images: {IMAGES_DIR}")
    print(f"  Annotations: {ANNOTATIONS_DIR}")

def check_dependencies():
    """Check if required tools are installed."""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        print("✓ yt-dlp is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ yt-dlp not found. Install with: pip install yt-dlp")
        return False
    
    try:
        import cv2
        print("✓ OpenCV is available")
    except ImportError:
        print("✗ OpenCV not found. Install with: pip install opencv-python")
        return False
        
    return True

def download_video(video_id, temp_dir):
    """
    Download a single video from YouTube.
    
    Args:
        video_id: YouTube video ID
        temp_dir: Temporary directory to save the video
    
    Returns:
        str: Path to downloaded video file, or None if failed
    """
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        temp_video_path = os.path.join(temp_dir, "video.%(ext)s")
        
        # Download video with yt-dlp (best quality up to 720p)
        download_cmd = [
            'yt-dlp',
            '--format', 'best[height<=720]',
            '--output', temp_video_path,
            '--quiet',
            video_url
        ]
        
        result = subprocess.run(download_cmd, capture_output=True)
        if result.returncode != 0:
            print(f"Failed to download video {video_id}")
            return None
        
        # Find the downloaded video file
        downloaded_files = list(Path(temp_dir).glob("video.*"))
        if not downloaded_files:
            print(f"No video file found for {video_id}")
            return None
        
        return str(downloaded_files[0])
        
    except Exception as e:
        print(f"Error downloading {video_id}: {str(e)}")
        return None

def extract_frames_from_video(video_path, video_id, frame_requests):
    """
    Extract multiple frames from a single video file.
    
    Args:
        video_path: Path to the video file
        video_id: YouTube video ID (for logging)
        frame_requests: List of tuples (timestamp_ms, output_path)
    
    Returns:
        int: Number of successfully extracted frames
    """
    successful_extractions = 0
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video {video_id}")
            return 0
        
        # Sort frame requests by timestamp for efficient seeking
        sorted_requests = sorted(frame_requests, key=lambda x: x[0])
        
        for timestamp_ms, output_path in sorted_requests:
            # Set video position to the desired timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            
            # Read the frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"Failed to extract frame from {video_id} at {timestamp_ms}ms")
                continue
            
            # Resize frame to target dimensions
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # Save the frame
            success = cv2.imwrite(output_path, frame)
            if success:
                print(f"✓ Saved frame: {os.path.basename(output_path)}")
                successful_extractions += 1
            else:
                print(f"Failed to save frame: {output_path}")
        
        cap.release()
        return successful_extractions
        
    except Exception as e:
        print(f"Error extracting frames from {video_id}: {str(e)}")
        return 0

def create_xml_annotation(image_path, annotations, video_id, image_width=FRAME_WIDTH, image_height=FRAME_HEIGHT):
    """
    Create Pascal VOC format XML annotation for the image.
    
    Args:
        image_path: Path to the image file
        annotations: List of annotation dictionaries
        video_id: YouTube video ID for organizing folders
        image_width: Width of the image
        image_height: Height of the image
    """
    root = ET.Element("annotation")
    
    # Folder and filename
    ET.SubElement(root, "folder").text = video_id
    ET.SubElement(root, "filename").text = os.path.basename(image_path)
    
    # Source
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "YouTube-BB"
    
    # Size
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image_width)
    ET.SubElement(size, "height").text = str(image_height)
    ET.SubElement(size, "depth").text = "3"
    
    # Segmented
    ET.SubElement(root, "segmented").text = "0"
    
    # Objects
    for ann in annotations:
        if ann['object_presence'] != 'present':
            continue
            
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = ann['class_name']
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        # Convert normalized coordinates to pixel coordinates
        xmin = int(ann['xmin'] * image_width)
        ymin = int(ann['ymin'] * image_height)
        xmax = int(ann['xmax'] * image_width)
        ymax = int(ann['ymax'] * image_height)
        
        # Ensure coordinates are within image bounds
        xmin = max(0, min(xmin, image_width - 1))
        ymin = max(0, min(ymin, image_height - 1))
        xmax = max(0, min(xmax, image_width - 1))
        ymax = max(0, min(ymax, image_height - 1))
        
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)
    
    # Create video-specific annotation directory
    video_ann_dir = os.path.join(ANNOTATIONS_DIR, video_id)
    os.makedirs(video_ann_dir, exist_ok=True)
    
    # Save XML file in video-specific subdirectory
    xml_path = os.path.join(video_ann_dir, os.path.splitext(os.path.basename(image_path))[0] + '.xml')
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    
    return xml_path

def read_existing_frames():
    """Read already downloaded frames from train.txt and val.txt files."""
    existing_frames = set()
    
    for split_file in ['train.txt', 'val.txt']:
        split_path = os.path.join(OUTPUT_DIR, split_file)
        if os.path.exists(split_path):
            try:
                with open(split_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            existing_frames.add(line)
                print(f"Found {len([l for l in open(split_path, 'r') if l.strip()])} existing frames in {split_file}")
            except Exception as e:
                print(f"Warning: Could not read {split_file}: {e}")
        else:
            print(f"Split file {split_file} does not exist yet")
    
    return existing_frames

def process_csv_file(split):
    """Process the YouTube-BB CSV file and download frames."""
    assert split in ['train', 'val'], "Split must be 'train' or 'val'"
    file_path = CSV_TRAIN_FILE if split == 'train' else CSV_VAL_FILE

    if not os.path.exists(file_path):
        print(f"CSV file not found: {file_path}")
        return
    
    print(f"Reading CSV file: {file_path}")
        
    # Read CSV file
    df = pd.read_csv(file_path, names=[
        'youtube_id', 'timestamp_ms', 'class_id', 'class_name', 
        'object_id', 'object_presence', 'xmin', 'xmax', 'ymin', 'ymax'
    ])
    
    print(f"Found {len(df)} annotations")

    # Read existing frames to avoid reprocessing
    existing_frames = read_existing_frames()
    print(f"Total existing frames to skip: {len(existing_frames)}")
    
    # Group annotations by video_id and timestamp, filtering out existing frames
    frame_annotations = defaultdict(list)
    skipped_count = 0
    
    for _, row in df.iterrows():
        video_id = row['youtube_id']
        timestamp_ms = row['timestamp_ms']
        
        # Create the frame identifier as it would appear in train.txt/val.txt
        frame_id = f"{video_id}/{int(timestamp_ms):06d}"
        
        # Skip if frame already exists in train.txt or val.txt
        if frame_id in existing_frames:
            skipped_count += 1
            continue
            
        key = (video_id, timestamp_ms)
        frame_annotations[key].append(row.to_dict())
    
    print(f"Skipped {skipped_count} already downloaded annotations")
    print(f"Processing {len(frame_annotations)} unique frames")
    
    # Group frames by video_id for efficient processing
    video_frames = defaultdict(list)
    for (video_id, timestamp_ms), annotations in frame_annotations.items():
        frame_filename = f"{int(timestamp_ms):06d}.jpg"
        
        # Store frame info without creating directories yet
        video_frames[video_id].append({
            'timestamp_ms': timestamp_ms,
            'frame_filename': frame_filename,
            'annotations': annotations
        })
    
    print(f"Need to process {len(video_frames)} videos")
    
    successful_downloads = 0
    failed_downloads = 0
    processed_videos = 0
    
    # Process each video
    for video_id, frames in video_frames.items():
        if MAX_VIDEOS and processed_videos >= MAX_VIDEOS:
            print(f"Reached maximum video limit ({MAX_VIDEOS})")
            break
            
        processed_videos += 1
        print(f"\nProcessing video {processed_videos}/{min(len(video_frames), MAX_VIDEOS or len(video_frames))}: {video_id} ({len(frames)} frames)")
        
        # Create temporary directory for this video
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the video once
            video_path = download_video(video_id, temp_dir)
            if not video_path:
                print(f"Failed to download video {video_id}")
                failed_downloads += len(frames)
                continue
            
            print(f"✓ Downloaded video {video_id}")
            
            # Create video-specific image directory only after successful download
            video_img_dir = os.path.join(IMAGES_DIR, video_id)
            os.makedirs(video_img_dir, exist_ok=True)
            
            # Prepare frame extraction requests with full paths
            frame_requests = []
            for frame in frames:
                frame_path = os.path.join(video_img_dir, frame['frame_filename'])
                frame['frame_path'] = frame_path  # Update frame dict with full path
                frame_requests.append((frame['timestamp_ms'], frame_path))
            
            # Extract all frames from this video
            extracted_count = extract_frames_from_video(video_path, video_id, frame_requests)
            
            # Create XML annotations for successfully extracted frames
            # Also add frmaes to test/val split file
            txt_frames = []
            for frame in frames:
                if os.path.exists(frame['frame_path']):
                    xml_path = create_xml_annotation(frame['frame_path'], frame['annotations'], video_id)
                    print(f"✓ Created annotation: {video_id}/{os.path.basename(xml_path)}")
                    successful_downloads += 1
                    video_frame = f"{video_id}/{os.path.splitext(frame['frame_filename'])[0]}"
                    txt_frames.append(video_frame)
                else:
                    failed_downloads += 1
            # Append to file
            with open(os.path.join(OUTPUT_DIR, f'{split}.txt'), 'a') as f:
                for txt_frame in txt_frames:
                    f.write(txt_frame + '\n')
    
    print(f"\nDownload complete!")
    print(f"Processed videos: {processed_videos}")
    print(f"Successful frames: {successful_downloads}")
    print(f"Failed frames: {failed_downloads}")

def create_split_files(split):
    """Create train/val split files listing the downloaded images, splitting by videos to preserve temporal sequences."""
    assert split in ['train', 'val'], "Split must be 'train' or 'val'"
    
    all_video_ids = []
    video_frames = {}
    
    # Collect all video IDs and their frames
    for video_id in os.listdir(IMAGES_DIR):
        video_dir = os.path.join(IMAGES_DIR, video_id)
        if os.path.isdir(video_dir):
            frames = []
            for frame_file in os.listdir(video_dir):
                if frame_file.endswith('.jpg'):
                    # Store as video_id/frame_name (without extension)
                    video_frame = f"{video_id}/{os.path.splitext(frame_file)[0]}"
                    frames.append(video_frame)
            
            if frames:  # Only add videos that have frames
                all_video_ids.append(video_id)
                video_frames[video_id] = sorted(frames)
    
    all_video_ids.sort()
    
    # Split videos (not individual frames) 80/20 for train/val
    split_idx = int(len(all_video_ids) * 0.8)
    train_videos = all_video_ids[:split_idx]
    val_videos = all_video_ids[split_idx:]
    
    # Collect all frames from train videos
    train_frames = []
    for video_id in train_videos:
        train_frames.extend(video_frames[video_id])
    
    # Collect all frames from val videos
    val_frames = []
    for video_id in val_videos:
        val_frames.extend(video_frames[video_id])
    
    # Write train.txt
    with open(os.path.join(OUTPUT_DIR, 'train.txt'), 'w') as f:
        for video_frame in train_frames:
            f.write(video_frame + '\n')
    
    # Write val.txt
    with open(os.path.join(OUTPUT_DIR, 'val.txt'), 'w') as f:
        for video_frame in val_frames:
            f.write(video_frame + '\n')
    
    print(f"Created split files (video-level split):")
    print(f"  train.txt: {len(train_frames)} frames from {len(train_videos)} videos")
    print(f"  val.txt: {len(val_frames)} frames from {len(val_videos)} videos")
    print(f"  Train videos: {train_videos[:5]}{'...' if len(train_videos) > 5 else ''}")
    print(f"  Val videos: {val_videos[:5]}{'...' if len(val_videos) > 5 else ''}")

def main():
    """Main function to download YouTube-BB dataset."""
    print("YouTube-BB Dataset Downloader")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Process the CSV file and download frames
    process_csv_file('train')
    process_csv_file('val')
    
    # Create train/val split files
    # create_split_files()
    
    print("\nDataset download completed!")
    print(f"Images saved to: {IMAGES_DIR}")
    print(f"Annotations saved to: {ANNOTATIONS_DIR}")

if __name__ == "__main__":
    main()