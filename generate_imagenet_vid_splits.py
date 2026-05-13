import os
import sys

# Find all actual frames in the dataset by scanning the directory tree
data_root = r"D:\ImageNet-VID\ImageNet\data\ImageNet2015\object_detection_from_video\ILSVRC2015\Data\VID"
split_name = 'train'
output_file = r"D:\ImageNet-VID\ImageNet\data\ImageNet2015\object_detection_from_video\ILSVRC2015\ImageSets\VID\train_actual.txt"

frames = []

# Scan Data/VID/train/
split_path = os.path.join(data_root, split_name)
if not os.path.exists(split_path):
    print(f"Error: {split_path} does not exist")
    sys.exit(1)

print(f"Scanning {split_path}...")

# List all video groups (e.g., ILSVRC2015_VID_train_0000, a, b, c, ...)
video_groups = os.listdir(split_path)
video_groups = [d for d in video_groups if os.path.isdir(os.path.join(split_path, d))]
video_groups.sort()

print(f"Found {len(video_groups)} video groups")

frame_count = 0
for video_group in video_groups:
    video_group_path = os.path.join(split_path, video_group)
    
    # List all videos in this group (e.g., ILSVRC2015_train_00000000)
    videos = os.listdir(video_group_path)
    videos = [v for v in videos if os.path.isdir(os.path.join(video_group_path, v))]
    videos.sort()
    
    for video_name in videos:
        video_path = os.path.join(video_group_path, video_name)
        
        # List all frame files (*.JPEG)
        frame_files = os.listdir(video_path)
        frame_files = sorted([f for f in frame_files if f.lower().endswith(('.jpeg', '.jpg'))])
        
        for frame_file in frame_files:
            frame_num = os.path.splitext(frame_file)[0]
            # Format: "video_group/frame_name label"
            frames.append(f"{video_group}/{video_name} 1")
            frame_count += 1

print(f"Found {frame_count} total frames")
print(f"Writing to {output_file}...")

# Write to file
with open(output_file, 'w', encoding='utf-8') as f:
    for frame_entry in frames:
        f.write(frame_entry + '\n')

print(f"Created {output_file} with {len(frames)} unique frame paths")
