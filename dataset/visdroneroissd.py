from glob import glob
import os
from pathlib import Path
from random import shuffle
from tools.utils import read_annotation_file
import torch
import torchvision.transforms.v2
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
from torchvision import tv_tensors
from torchvision.io import read_image

from transformers.random_roi_crop import RandomROICrop

def labels_getter(transform_input):
    return (transform_input[1]["labels"], transform_input[1]["difficult"])

def load_images_and_anns(im_sets, label2idx, ann_fname):
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

    videos = []

    for im_set in im_sets:
        # im_names = []
        # Fetch all image names in txt file for this imageset

        video_dirs = sorted(os.listdir(os.path.join(im_set, "SequenceAnnotations")))
        num_videos = len(video_dirs)

        for vid_idx, vid in enumerate(video_dirs):
            frames = []
            print(f'Loading video {vid_idx + 1}/{num_videos}')
            im_dir = os.path.join(im_set, 'ResizedSequences', vid)
            for (_, _, filenames) in os.walk(os.path.join(im_set, "SequenceAnnotations", vid)):
                for idx, ann_file in enumerate(filenames):
                    ann_dir = os.path.join(im_set, "SequenceAnnotations", vid)
                    im_info, success = read_annotation_file(ann_dir, im_dir, ann_file, label2idx)
                    if not success:
                        continue

                    # Skip images with no detections
                    if len(im_info.get('detections', [])) == 0:
                        continue
                    
                    frames.append(im_info)
                # Iterate for every annotation file
            # Iterating over sequence annotations
            if len(frames) > 0:
                videos.append({'video_id': vid_idx, 'frames': frames})
        # Iterating over video sequence folders
    print('Total {} images found'.format(sum([len(video['frames']) for video in videos])))
    if len(videos) == 0:
        raise ValueError('No videos found for the specified im_sets')
    return videos


class VisDroneRoiSsdDataset(Dataset):
    def __init__(self, split, im_sets, im_size=512):
        self.split = split

        # Imagesets for this dataset instance (VOC2007/VOC2007+VOC2012/VOC2007-test)
        self.im_sets = im_sets
        self.fname = 'trainval' if self.split == 'train' else 'test'
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
                    labels_getter=labels_getter),
                RandomROICrop(
                    p=0.5,
                    alpha_w=0.3,
                    alpha_h=0.3,
                    delta_x=8.0,
                    delta_y=8.0,
                    area_ratio_max=1.4,
                    min_box_area=4.0
                ),
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

        # Extract class names from categories list
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
        classes = [category['name'] for category in categories]
        # classes = sorted(classes)
        # We need to add background class as well with 0 index
        classes = ['background'] + classes

        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        
        self.videos = load_images_and_anns(self.im_sets,
                                                self.label2idx,
                                                self.fname)
        self.pair_indices = []
        for idx, video in enumerate(self.videos):
            num_frames = len(video['frames'])
            for i in range(num_frames - 1):
                self.pair_indices.append({'video_id': idx, 'frame_pair': (video['frames'][i], video['frames'][i + 1])})
        shuffle(self.pair_indices)


    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, index):
        
        pair_info = self.pair_indices[index]
        frame_pair = pair_info['frame_pair']

        frame_info_1 = frame_pair[0]
        frame_info_2 = frame_pair[1]

        im_tensor1, targets1, filename1 = self.get_im_and_targets(frame_info_1)
        _, targets2, _ = self.get_im_and_targets(frame_info_2)
        
        return im_tensor1, targets1, filename1, targets2
    
    def get_im_and_targets(self, im_info):
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
