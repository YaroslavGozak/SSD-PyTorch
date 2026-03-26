import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import torch
from torch.utils.data.dataset import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image

from dataset.transforms.fixed_padding_roi_crop_test_transform import FixedPaddingRoiCropTestTransform
from dataset.transforms.letterbox_transform import LetterboxTransform
from dataset.transforms.no_resize_transform import NoResizeTransform
from dataset.transforms.resize_longer_edge_test_transform import ResizeLongerEdgeTestTransform
from dataset.transforms.roi_crop_test_transform import RoiCropTestTransform
from dataset.transforms.ssd_transform import SsdTransform


def _is_voc_video_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isdir(os.path.join(path, "Annotations"))
        and os.path.isdir(os.path.join(path, "ImageSets"))
        and os.path.isdir(os.path.join(path, "JPEGImages"))
    )


def _discover_video_dirs(im_sets: List[str]) -> List[str]:
    """
    Supports two layouts:
    1) im_sets contains direct video dirs (.../video_01)
    2) im_sets contains a root dir that has many video dirs as children
    """
    out = []
    for root in im_sets:
        if _is_voc_video_dir(root):
            out.append(root)
            continue

        if not os.path.isdir(root):
            continue

        children = [os.path.join(root, d) for d in sorted(os.listdir(root))]
        for child in children:
            if _is_voc_video_dir(child):
                out.append(child)

    return out


def load_images_and_anns_video(
    im_sets: List[str],
    label2idx: Dict[str, int],
    ann_fname: str,
    task: str = None,
) -> List[Dict]:
    """
    Reads many VOC-like video folders and returns a single ordered frame list.
    Each item includes boundary metadata to support tracker reset.
    """
    im_infos = []
    video_dirs = _discover_video_dirs(im_sets)

    for video_dir in video_dirs:
        video_id = os.path.basename(video_dir)
        split_file = os.path.join(video_dir, "ImageSets", "Main", f"{ann_fname}.txt")
        if not os.path.exists(split_file):
            continue

        with open(split_file, "r") as f:
            im_names = [line.strip() for line in f if line.strip()]

        ann_dir = os.path.join(video_dir, "Annotations")
        im_dir = os.path.join(video_dir, "JPEGImages")

        for frame_idx, im_name in enumerate(im_names):
            ann_file = os.path.join(ann_dir, f"{im_name}.xml")
            if not os.path.exists(ann_file):
                continue

            ann_info = ET.parse(ann_file)
            root = ann_info.getroot()
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)

            info = {
                "img_id": os.path.basename(ann_file).split(".xml")[0],
                "filename": os.path.join(im_dir, f"{im_name}.jpg"),
                "width": width,
                "height": height,
                "video_id": video_id,
                "frame_idx": frame_idx,
                "is_first_frame": frame_idx == 0,
            }

            detections = []
            for obj in ann_info.findall("object"):
                label = label2idx[obj.find("name").text]
                difficult = int(obj.find("difficult").text)
                bbox_info = obj.find("bndbox")
                bbox = [
                    int(bbox_info.find("xmin").text) - 1,
                    int(bbox_info.find("ymin").text) - 1,
                    int(bbox_info.find("xmax").text) - 1,
                    int(bbox_info.find("ymax").text) - 1,
                ]
                detections.append({"label": label, "bbox": bbox, "difficult": difficult})

            info["detections"] = detections
            im_infos.append(info)

            if task == "demo" and len(im_infos) >= 10:
                break

    print("Total {} frames found across {} videos".format(len(im_infos), len(video_dirs)))
    return im_infos


class VOCVideoDataset(Dataset):
    def __init__(self, split, im_sets, im_size=300, transform_name="ssd", task=None):
        self.split = split
        self.task = task
        self.transform_name = transform_name
        self.im_sets = im_sets
        self.fname = "trainval" if self.split == "train" else "test"
        self.im_size = im_size
        self.im_mean = [123.0, 117.0, 104.0]
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

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
        elif self.transform_name.startswith("fixed_padding_roi_crop_"):
            pad_value = int(self.transform_name.split("_")[-1])
            self.transforms = FixedPaddingRoiCropTestTransform(
                im_size, self.imagenet_mean, self.imagenet_std, pad_x=pad_value, pad_y=pad_value
            ).transforms
        else:
            raise Exception('Unknown transform name "{}"'.format(self.transform_name))

        classes = [
            "person", "bird", "cat", "cow", "dog", "horse", "sheep",
            "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
            "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor",
        ]
        classes = ["background"] + sorted(classes)
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}

        self.images_info = load_images_and_anns_video(
            self.im_sets,
            self.label2idx,
            self.fname,
            self.task,
        )

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = read_image(im_info["filename"])

        targets = {}
        targets["bboxes"] = tv_tensors.BoundingBoxes(
            [detection["bbox"] for detection in im_info["detections"]],
            format="XYXY",
            canvas_size=im.shape[-2:],
        )
        targets["labels"] = torch.as_tensor([detection["label"] for detection in im_info["detections"]])
        targets["difficult"] = torch.as_tensor([detection["difficult"] for detection in im_info["detections"]])
        orig_h, orig_w = im.shape[-2:]
        targets["orig_size"] = (orig_h, orig_w)

        # Video-boundary metadata for sequence pipelines.
        targets["video_id"] = im_info["video_id"]
        targets["frame_idx"] = im_info["frame_idx"]
        targets["is_first_frame"] = im_info["is_first_frame"]

        im_tensor, targets = self.transforms[self.split](im, targets)

        h, w = im_tensor.shape[-2:]
        wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(targets["bboxes"])
        targets["bboxes"] = targets["bboxes"] / wh_tensor
        return im_tensor, targets, im_info["filename"]