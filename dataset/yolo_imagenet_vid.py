import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
import yaml
from torch.utils.data.dataset import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image

from dataset.transforms.fixed_padding_roi_crop_test_transform import FixedPaddingRoiCropTestTransform
from dataset.transforms.fixed_padding_roi_crop_yolo_test_transform import FixedPaddingRoiCropYOLOTestTransform
from dataset.transforms.letterbox_transform import LetterboxTransform
from dataset.transforms.no_resize_transform import NoResizeTransform
from dataset.transforms.resize_longer_edge_test_transform import ResizeLongerEdgeTestTransform
from dataset.transforms.roi_crop_test_transform import RoiCropTestTransform
from dataset.transforms.ssd_transform import SsdTransform


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_FRAME_RE = re.compile(r".*_(\d+)_(\d+)$")


def _load_yolo_data_yaml(yaml_path: str) -> Dict[str, Any]:
    print(f"Loading YOLO dataset yaml: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"YOLO dataset yaml must be a mapping, got {type(cfg).__name__}")

    root_path = cfg.get("path", "")
    if not root_path:
        raise ValueError("YOLO dataset yaml is missing required key: 'path'")

    names_obj = cfg.get("names")
    if names_obj is None:
        raise ValueError("YOLO dataset yaml is missing required key: 'names'")

    if isinstance(names_obj, dict):
        name_items: List[Tuple[int, str]] = []
        for k, v in names_obj.items():
            idx = int(k)
            name_items.append((idx, str(v)))
        name_items.sort(key=lambda x: x[0])
        names = [name for _, name in name_items]
    elif isinstance(names_obj, list):
        names = [str(x) for x in names_obj]
    else:
        raise TypeError(f"YOLO 'names' must be dict or list, got {type(names_obj).__name__}")

    if not names:
        raise ValueError("YOLO dataset yaml 'names' is empty")

    return {
        "path": str(root_path),
        "train": str(cfg.get("train", "images/train")),
        "val": str(cfg.get("val", "images/val")),
        "names": names,
    }


def _resolve_split_roots(yolo_cfg: Dict[str, Any], split: str) -> Tuple[str, str]:
    split = str(split)
    if split == "train":
        image_split_rel = yolo_cfg["train"]
    elif split in ("test", "val"):
        image_split_rel = yolo_cfg["val"]
    else:
        raise ValueError(f"Unsupported split: {split!r}. Expected train/test/val")

    root = Path(yolo_cfg["path"])
    image_root = (root / image_split_rel).resolve()

    # Prefer replacing leading images/ with labels/ to preserve nested structure.
    image_split_rel_norm = image_split_rel.replace("\\", "/")
    if image_split_rel_norm.startswith("images/"):
        label_split_rel = "labels/" + image_split_rel_norm[len("images/"):]
    else:
        label_split_rel = image_split_rel_norm.replace("images", "labels", 1)

    label_root = (root / label_split_rel).resolve()
    return str(image_root), str(label_root)


def _discover_images(image_root: str) -> List[str]:
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image split root not found: {image_root}")

    image_paths: List[str] = []
    for dirpath, _, filenames in os.walk(image_root):
        for filename in sorted(filenames):
            ext = os.path.splitext(filename)[1].lower()
            if ext in _IMAGE_EXTS:
                image_paths.append(os.path.join(dirpath, filename))

    image_paths.sort()
    if not image_paths:
        raise ValueError(f"No images found under: {image_root}")
    return image_paths


def _extract_video_tokens(stem: str, parent_dir_name: str) -> Tuple[str, Optional[int]]:
    match = _VIDEO_FRAME_RE.match(stem)
    if match:
        video_id = match.group(1)
        frame_idx = int(match.group(2))
        return video_id, frame_idx
    return parent_dir_name, None


def _parse_yolo_label_file(
    label_path: str,
    img_w: int,
    img_h: int,
    class_names: List[str],
    label2idx: Dict[str, int],
) -> List[Dict[str, Any]]:
    if not os.path.exists(label_path):
        return []

    detections: List[Dict[str, Any]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                continue

            if class_id < 0 or class_id >= len(class_names):
                continue

            class_name = class_names[class_id]
            label_idx = label2idx[class_name]

            x1 = (xc - bw / 2.0) * img_w
            y1 = (yc - bh / 2.0) * img_h
            x2 = (xc + bw / 2.0) * img_w
            y2 = (yc + bh / 2.0) * img_h

            x1 = max(0.0, min(float(img_w - 1), x1))
            y1 = max(0.0, min(float(img_h - 1), y1))
            x2 = max(0.0, min(float(img_w - 1), x2))
            y2 = max(0.0, min(float(img_h - 1), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "label": int(label_idx),
                    "bbox": [x1, y1, x2, y2],
                    "difficult": 0,
                }
            )

    return detections


def _build_images_info(
    image_root: str,
    label_root: str,
    class_names: List[str],
    label2idx: Dict[str, int],
    task: Optional[str] = None,
) -> List[Dict[str, Any]]:
    image_paths = _discover_images(image_root)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    print(f"Processing {len(image_paths)} images and corresponding labels...")
    for _, image_path in enumerate(tqdm(image_paths, desc='Processing images')):
        rel_path = os.path.relpath(image_path, image_root)
        rel_no_ext = os.path.splitext(rel_path)[0]
        label_path = os.path.join(label_root, rel_no_ext + ".txt")

        im = read_image(image_path)
        img_h, img_w = int(im.shape[-2]), int(im.shape[-1])
        detections = _parse_yolo_label_file(label_path, img_w, img_h, class_names, label2idx)
        if not detections:
            continue

        stem = os.path.splitext(os.path.basename(image_path))[0]
        parent_dir = os.path.basename(os.path.dirname(image_path))
        video_id, parsed_frame_idx = _extract_video_tokens(stem, parent_dir)

        frame_info = {
            "img_id": stem,
            "filename": image_path,
            "width": img_w,
            "height": img_h,
            "video_id": video_id,
            "parsed_frame_idx": parsed_frame_idx,
            "detections": detections,
        }
        grouped[video_id].append(frame_info)

    if not grouped:
        raise ValueError(
            f"No labeled frames found for image_root={image_root} label_root={label_root}"
        )

    im_infos: List[Dict[str, Any]] = []
    for _, video_id in enumerate(tqdm(sorted(grouped.keys()), desc='Processing images')):
        frames = grouped[video_id]
        frames.sort(
            key=lambda x: (
                x["parsed_frame_idx"] is None,
                x["parsed_frame_idx"] if x["parsed_frame_idx"] is not None else 10**18,
                x["filename"],
            )
        )

        for idx_in_video, frame in enumerate(frames):
            frame_idx = (
                int(frame["parsed_frame_idx"])
                if frame["parsed_frame_idx"] is not None
                else idx_in_video
            )
            frame["frame_idx"] = frame_idx
            frame["is_first_frame"] = idx_in_video == 0
            frame.pop("parsed_frame_idx", None)
            im_infos.append(frame)

            if task == "demo" and len(im_infos) >= 1000:
                print(f"Total {len(im_infos)} labeled frames found (demo mode)")
                return im_infos

    print(f"Total {len(im_infos)} labeled frames found across {len(grouped)} videos")
    return im_infos


class YoloImageNetVidDataset(Dataset):
    def __init__(
        self,
        split: str,
        yolo_dataset_yaml: str,
        im_size: int = 300,
        transform_name: str = "ssd",
        task: Optional[str] = None,
    ):
        self.split = split
        self.transform_split = "train" if str(split) == "train" else "test"
        self.task = task
        self.transform_name = transform_name
        self.im_size = im_size

        self.im_mean = [123.0, 117.0, 104.0]
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

        yolo_cfg = _load_yolo_data_yaml(yolo_dataset_yaml)
        class_names = list(yolo_cfg["names"])

        self.classes = ["background"] + class_names
        self.label2idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx2label = {idx: class_name for idx, class_name in enumerate(self.classes)}

        if self.transform_name == "ssd":
            self.transforms = SsdTransform(im_size, self.im_mean, self.imagenet_mean, self.imagenet_std).transforms
        elif self.transform_name == "letterbox":
            self.transforms = LetterboxTransform(im_size, self.im_mean, self.imagenet_mean, self.imagenet_std).transforms
        elif self.transform_name == "resize_longer_edge":
            self.transforms = ResizeLongerEdgeTestTransform(im_size, self.im_mean, self.imagenet_mean, self.imagenet_std).transforms
        elif self.transform_name == "roi_crop_test_transform":
            self.transforms = RoiCropTestTransform(im_size, self.im_mean, self.imagenet_mean, self.imagenet_std).transforms
        elif self.transform_name == "no_resize_transform":
            self.transforms = NoResizeTransform(self.im_mean, self.imagenet_mean, self.imagenet_std).transforms
        elif self.transform_name.startswith("fixed_padding_roi_crop_yolo_"):
            pad_value = int(self.transform_name.split("_")[-1])
            self.transforms = FixedPaddingRoiCropYOLOTestTransform(
                300,
                self.imagenet_mean,
                self.imagenet_std,
                pad_x=pad_value,
                pad_y=pad_value,
            ).transforms
        elif self.transform_name.startswith("fixed_padding_roi_crop_"):
            pad_value = int(self.transform_name.split("_")[-1])
            self.transforms = FixedPaddingRoiCropTestTransform(
                im_size,
                self.imagenet_mean,
                self.imagenet_std,
                pad_x=pad_value,
                pad_y=pad_value,
            ).transforms
        else:
            raise ValueError(f'Unknown transform name "{self.transform_name}"')

        image_root, label_root = _resolve_split_roots(yolo_cfg, split)
        self.images_info = _build_images_info(
            image_root=image_root,
            label_root=label_root,
            class_names=class_names,
            label2idx=self.label2idx,
            task=task,
        )

    def __len__(self) -> int:
        return len(self.images_info)

    def __getitem__(self, index: int):
        im_info = self.images_info[index]
        im = read_image(im_info["filename"])

        targets: Dict[str, Any] = {}
        targets["bboxes"] = tv_tensors.BoundingBoxes(
            [detection["bbox"] for detection in im_info["detections"]],
            format="XYXY",
            canvas_size=im.shape[-2:],
        )
        targets["labels"] = torch.as_tensor(
            [detection["label"] for detection in im_info["detections"]]
        )
        targets["difficult"] = torch.as_tensor(
            [detection["difficult"] for detection in im_info["detections"]]
        )
        orig_h, orig_w = im.shape[-2:]
        targets["orig_size"] = (orig_h, orig_w)

        targets["video_id"] = im_info["video_id"]
        targets["frame_idx"] = im_info["frame_idx"]
        targets["is_first_frame"] = im_info["is_first_frame"]

        im_tensor, targets = self.transforms[self.transform_split](im, targets)

        h, w = im_tensor.shape[-2:]
        wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(targets["bboxes"])
        targets["bboxes"] = targets["bboxes"] / wh_tensor

        return im_tensor, targets, im_info["filename"]

class YoloImageNetVidRawDataset(Dataset):
    """
    YOLO-format ImageNet-VID dataset with ImageNetVidRawDataset-compatible output.

    Supports:
        item = index
    or:
        item = (index, mode, out_size)
    """

    def __init__(
        self,
        split: str,
        yolo_dataset_yaml: str,
        im_size: int = 300,
        task: Optional[str] = None,
    ):
        self.split = split
        self.task = task
        self.im_size = im_size

        yolo_cfg = _load_yolo_data_yaml(yolo_dataset_yaml)
        class_names = list(yolo_cfg["names"])

        self.classes = ["background"] + class_names
        self.label2idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx2label = {idx: class_name for idx, class_name in enumerate(self.classes)}

        image_root, label_root = _resolve_split_roots(yolo_cfg, split)
        self.images_info = _build_images_info(
            image_root=image_root,
            label_root=label_root,
            class_names=class_names,
            label2idx=self.label2idx,
            task=task,
        )

        if len(self.images_info) == 0:
            raise RuntimeError(
                f"No frames loaded for split '{split}'. Check image_root={image_root} label_root={label_root}"
            )

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            index, mode, out_size = item
        else:
            index = item
            mode = "full"
            out_size = 300

        im_info = self.images_info[index]
        image = read_image(im_info["filename"]).float() / 255.0  # CHW in [0,1]

        boxes_abs = torch.tensor(
            [det["bbox"] for det in im_info["detections"]],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [det["label"] for det in im_info["detections"]],
            dtype=torch.int64,
        )
        difficult = torch.tensor(
            [det["difficult"] for det in im_info["detections"]],
            dtype=torch.int64,
        )

        h, w = image.shape[-2:]
        if boxes_abs.numel() == 0:
            boxes_norm = torch.zeros((0, 4), dtype=torch.float32)
        else:
            wh = torch.tensor([w, h, w, h], dtype=torch.float32).unsqueeze(0)
            boxes_norm = boxes_abs / wh

        target = {
            "boxes": boxes_norm,  # normalized xyxy in [0,1]
            "labels": labels,
            "difficult": difficult,
            "orig_size": (h, w),
            "image_id": im_info["img_id"],
            "filename": im_info["filename"],
            # optional extra metadata (safe to keep):
            "video_id": im_info["video_id"],
            "frame_idx": im_info["frame_idx"],
            "is_first_frame": im_info["is_first_frame"],
        }

        return {
            "image": image,
            "target": target,
            "mode": mode,
            "out_size": out_size,
        }
