from glob import glob
import os
from pathlib import Path
import torch
import torchvision.transforms.v2
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
from torchvision import tv_tensors
from torchvision.io import read_image
import random


def load_images_and_anns(im_sets, ann_fname):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_sets: Sets of images to consider
    :param label2idx: Class Name to index mapping for dataset
    :param ann_fname: txt file containing image names{trainval.txt/test.txt}
    :param split: train/test
    :return:
    """

    categories = [
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "bicycle"},
        {"id": 4, "name": "car"},
        {"id": 5, "name": "van"},
        {"id": 6, "name": "truck"},
        {"id": 7, "name": "tricycle"},
        {"id": 8, "name": "awning-tricycle"},
        {"id": 9, "name": "bus"},
        {"id": 10, "name": "motor"},
    ]

    # im_infos = []

    for im_set in im_sets:
        # im_names = []
        # Fetch all image names in txt file for this imageset

        videos = sorted(os.listdir(os.path.join(im_set, "annotations")))
        data = {"videos": [], "frames": [], "categories": categories}

        for vid_idx, vid in enumerate(videos):
            ann_files = sorted(glob(os.path.join(im_set, "annotations", vid)))
            data["videos"].append({"id": vid_idx + 1, "name": vid})
            print(ann_files)
            for _, ann_file in enumerate(ann_files):
                with open(ann_file) as f:
                    for line in f:
                        vals = line.strip().split(',')
                        if len(vals) < 10:
                            continue
                        frame_idx, tid, x, y, w, h, score, cat, trunc, occ = map(float, vals)
                        frame_name = f"{int(frame_idx):07d}.jpg"
                        frame_name = os.path.join(im_set, "sequences", vid[:-4], frame_name)
                        for i, frame_info in enumerate(data["frames"]):
                            if frame_info['filename'] == frame_name:
                                data["frames"][i]["detections"].append({
                                    "target_id": int(tid),
                                    "label": int(cat),
                                    "bbox": [x, y, x + w, y + h],
                                    "area": w*h,
                                    "difficult": bool(int(trunc) + int(occ))
                                })
                                break
                        else:
                            data["frames"].append({
                                "filename": frame_name,
                                "frame_id": frame_idx,
                                "video_id": vid_idx + 1,
                                "detections": [{
                                    "target_id": int(tid),
                                    "label": int(cat),
                                    "bbox": [x, y, x + w, y + h],
                                    "area": w*h,
                                    "difficult": bool(int(trunc) + int(occ))
                                }]
                            })
            break
    print('Total {} images found'.format(len(data["frames"])))
    return data


class VisDroneDataset(Dataset):
    def __init__(self, split, im_sets, im_size=300):
        self.split = split

        # Imagesets for this dataset instance (VOC2007/VOC2007+VOC2012/VOC2007-test)
        self.im_sets = im_sets
        self.fname = '_v' #'trainval' if self.split == 'train' else 'test'
        self.im_size = im_size
        self.im_mean = [123.0, 117.0, 104.0]
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

        # Train and test transformations
        self.transforms = {
            'train': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.RandomPhotometricDistort(),
                torchvision.transforms.v2.RandomZoomOut(fill=self.im_mean),
                torchvision.transforms.v2.RandomIoUCrop(),
                torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.v2.Resize(size=(self.im_size, self.im_size)),
                torchvision.transforms.v2.SanitizeBoundingBoxes(
                    labels_getter=lambda transform_input:
                    (transform_input[1]["labels"], transform_input[1]["difficult"])),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                                                    std=self.imagenet_std)

            ]),
            'test': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.Resize(size=(self.im_size, self.im_size)),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                                                    std=self.imagenet_std)
            ]),
        }
        
        self.images_info = load_images_and_anns(self.im_sets,
                                                self.fname)
        
        # Extract class names from categories list
        classes = [category['name'] for category in self.images_info['categories']]
        classes = sorted(classes)
        # We need to add background class as well with 0 index
        classes = ['background'] + classes

        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)

    def __len__(self):
        return len(self.images_info["frames"])

    def __getitem__(self, index):
        im_info = self.images_info["frames"][index]
        im = read_image(im_info['filename'])

        # Get annotations for this image
        targets = {}
        targets['bboxes'] = tv_tensors.BoundingBoxes(
            [detection['bbox'] for detection in im_info['detections']],
            format='XYXY', canvas_size=im.shape[-2:])
        targets['labels'] = torch.as_tensor(
            [detection['label'] for detection in im_info['detections']])
        targets['difficult'] = torch.as_tensor(
            [detection['difficult']for detection in im_info['detections']])

        # Transform the image and targets
        transformed_info = self.transforms[self.split](im, targets)
        im_tensor, targets = transformed_info

        h, w = im_tensor.shape[-2:]
        wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(targets['bboxes'])
        targets['bboxes'] = targets['bboxes'] / wh_tensor
        return im_tensor, targets, str(im_info['filename'])
