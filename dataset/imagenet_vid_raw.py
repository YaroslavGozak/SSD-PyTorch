import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set

import torch
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image

from dataset.helpers.label_spaces import IMAGENET_VID_CLASSES, IMAGENET_VID_VOC_OVERLAP_CLASSES, build_label_maps


def load_images_and_anns_imagenet_vid(
    data_root: str,
    ann_root: str,
    label2idx: Dict[str, int],
    task: str = None,
    allowed_class_names: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Load ImageNet-VID frames by walking directory tree.
    
    Structure:
    Train data:
    - data_root/a/video_name/frame.JPEG
    - data_root/b/video_name/frame.JPEG
    - data_root/c,d,e/...
    
    Test/Val data:
    - data_root/video_name/frame.JPEG
    - data_root/video_name/frame.JPEG
    
    :param data_root: Root directory for split (train, val, or test)
    :param ann_root: Root directory for annotations
    :param label2idx: Class name to index mapping
    :param task: Optional task for early termination in demo mode
    :return: List of frame info dicts with video metadata
    """
    im_infos = []
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not os.path.exists(ann_root):
        raise FileNotFoundError(f"Annotation root not found: {ann_root}")
    
    # Check if train structure (has a,b,c,d,e folders) or test structure
    root_contents = os.listdir(data_root)
    has_class_folders = any(d in root_contents for d in ['a', 'b', 'c', 'd', 'e'])
    
    # Collect all (image_path, ann_path) pairs by walking directory tree
    frame_pairs = []
    
    if has_class_folders:
        # Train structure: walk a,b,c,d,e folders
        for class_folder in ['a', 'b', 'c', 'd', 'e']:
            class_path = os.path.join(data_root, class_folder)
            if not os.path.exists(class_path):
                continue
            
            # Walk video folders under this class
            for video_name in sorted(os.listdir(class_path)):
                video_path = os.path.join(class_path, video_name)
                if not os.path.isdir(video_path):
                    continue
                
                # Find all JPEG frames in this video
                video_frames = [
                    frame_file for frame_file in sorted(os.listdir(video_path))
                    if frame_file.lower().endswith(('.jpeg', '.jpg'))
                ]
                for frame_idx_in_video, frame_file in enumerate(video_frames):
                    frame_name = os.path.splitext(frame_file)[0]
                    img_path = os.path.join(video_path, frame_file)
                    ann_path = os.path.join(ann_root, class_folder, video_name, f"{frame_name}.xml")

                    if os.path.exists(ann_path):
                        frame_pairs.append(
                            (class_folder, video_name, frame_idx_in_video, frame_name, img_path, ann_path)
                        )
    else:
        # Test/Val structure: video folders directly under root
        for video_name in sorted(os.listdir(data_root)):
            video_path = os.path.join(data_root, video_name)
            if not os.path.isdir(video_path):
                continue
            
            # Find all JPEG frames in this video
            video_frames = [
                frame_file for frame_file in sorted(os.listdir(video_path))
                if frame_file.lower().endswith(('.jpeg', '.jpg'))
            ]
            for frame_idx_in_video, frame_file in enumerate(video_frames):
                frame_name = os.path.splitext(frame_file)[0]
                img_path = os.path.join(video_path, frame_file)
                ann_path = os.path.join(ann_root, video_name, f"{frame_name}.xml")

                if os.path.exists(ann_path):
                    frame_pairs.append(('', video_name, frame_idx_in_video, frame_name, img_path, ann_path))
    
    if not frame_pairs:
        raise ValueError(f"No frames found. Check data_root={data_root} and ann_root={ann_root}")
    
    print(f"Found {len(frame_pairs)} frame/annotation pairs")
    
    kept_frames = 0
    skipped_frames = 0
    dropped_objects = 0
    kept_objects = 0
    current_output_video_id = None
    seen_video_ids = set()

    for class_folder, video_name, original_frame_idx, frame_name, img_path, ann_path in frame_pairs:
        # Build video ID
        video_id = f"{class_folder}/{video_name}" if class_folder else video_name
        
        if video_id not in seen_video_ids:
            seen_video_ids.add(video_id)
        
        
        # Parse XML annotation
        try:
            ann_info = ET.parse(ann_path)
            root = ann_info.getroot()
            size = root.find('size')
            if size is None:
                print(f"Warning: no size element in {ann_path}")
                continue
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        except Exception as e:
            print(f"Error parsing {ann_path}: {e}")
            continue
        
        # Parse objects
        detections = []
        for obj in root.findall('object'):
            label_name = obj.find('class').text
            if allowed_class_names is not None and label_name not in allowed_class_names:
                dropped_objects += 1
                continue
            if label_name not in label2idx:
                print(f"Warning: unknown class '{label_name}' in {ann_path}; skipping object")
                continue
            
            difficult = 0 # Default to 0
            label = label2idx[label_name]
            bbox_info = obj.find('bndbox')
            bbox = [
                int(bbox_info.find('xmin').text) - 1,
                int(bbox_info.find('ymin').text) - 1,
                int(bbox_info.find('xmax').text) - 1,
                int(bbox_info.find('ymax').text) - 1,
            ]
            detections.append({'label': label, 'bbox': bbox, 'difficult': difficult})
            kept_objects += 1

        if not detections:
            skipped_frames += 1
            continue

        info = {
            "img_id": frame_name,
            "filename": img_path,
            "width": width,
            "height": height,
            "video_id": video_id,
            "frame_idx": original_frame_idx,
            "is_first_frame": video_id != current_output_video_id,
        }
        
        info['detections'] = detections
        im_infos.append(info)
        current_output_video_id = video_id
        kept_frames += 1
        
        if task == 'demo' and len(im_infos) >= 1000:
            break
    
    print(
        'Total {} frames loaded (kept_frames={}, skipped_frames={}, kept_objects={}, dropped_objects={})'.format(
            len(im_infos), kept_frames, skipped_frames, kept_objects, dropped_objects
        )
    )
    return im_infos


class ImageNetVidRawDataset(Dataset):
    """
    ImageNet-VID dataset with video sequence support.
    
    Train structure: data_root/{a,b,c,d,e}/video/frame.JPEG
    Test/Val structure: data_root/video/frame.JPEG
    """

    def _labels_getter(self, transform_input):
        """Helper function for SanitizeBoundingBoxes to extract labels and difficult flags."""
        return (transform_input[1]["labels"], transform_input[1]["difficult"])
    
    
    def __init__(
        self,
        split: str,
        train_data_root: str,
        train_ann_root: str,
        test_data_root: str,
        test_ann_root: str,
        im_size: int = 300,
        task: str = None,
        filter_voc_overlap: bool = False,
    ):
        """
        :param split: 'train' or 'test'
        :param train_data_root: Training data root directory
        :param train_ann_root: Training annotations root directory
        :param test_data_root: Test/val data root directory
        :param test_ann_root: Test/val annotations root directory
        :param im_size: Target image size for model input
        :param task: Optional task mode (e.g., 'demo' for early stopping)
        """
        self.split = split
        self.task = task
        self.im_size = im_size
        self.filter_voc_overlap = filter_voc_overlap
        
        # Select data root based on split
        if split == 'train':
            data_root = train_data_root
            ann_root = train_ann_root
        else:  # test, val
            data_root = test_data_root
            ann_root = test_ann_root
        
        # ImageNet-VID classes (30 object classes + background)
        # These are the standard 30 classes used in ImageNet-VID evaluation
        self.classes = list(IMAGENET_VID_CLASSES)
        self.label2idx, self.idx2label = build_label_maps(self.classes)

        allowed_class_names = None
        if self.filter_voc_overlap:
            allowed_class_names = set(IMAGENET_VID_VOC_OVERLAP_CLASSES)
        
        # Load dataset
        self.images_info = load_images_and_anns_imagenet_vid(
            data_root,
            ann_root,
            self.label2idx,
            task=self.task,
            allowed_class_names=allowed_class_names,
        )
        
        if len(self.images_info) == 0:
            raise RuntimeError(
                f"No frames loaded for split '{split}'. Check data_root={data_root}"
            )
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, item):
        """
        Supports:
            item = index
        or:
            item = (index, mode, out_size)
        """
        if isinstance(item, tuple):
            index, mode, out_size = item
        else:
            index = item
            mode = "full"
            out_size = 300
        im_info = self.images_info[index]
        image = read_image(im_info['filename']).float() / 255.0   # CHW in [0,1]
        
        # Prepare targets
        boxes_abs = torch.tensor(
            [det['bbox'] for det in im_info['detections']],
            dtype=torch.float32
        )
        labels = torch.tensor(
            [det['label'] for det in im_info['detections']],
            dtype=torch.int64
        )
        difficult = torch.tensor(
            [det['difficult'] for det in im_info['detections']],
            dtype=torch.int64
        )

        h, w = image.shape[-2:]
        if boxes_abs.numel() == 0:
            boxes_norm = torch.zeros((0, 4), dtype=torch.float32)
        else:
            wh = torch.tensor([w, h, w, h], dtype=torch.float32).unsqueeze(0)
            boxes_norm = boxes_abs / wh

        target = {
            "boxes": boxes_norm,      # normalized xyxy in [0,1]
            "labels": labels,
            "difficult": difficult,
            "orig_size": (h, w),
            "image_id": im_info['img_id'],
            "filename": im_info['filename'],
        }

        return {
            "image": image,
            "target": target,
            "mode": mode,
            "out_size": out_size,
        }
