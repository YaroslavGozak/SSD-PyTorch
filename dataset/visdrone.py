from glob import glob
import os
from pathlib import Path
import torch
import torchvision.transforms.v2
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
from torchvision import tv_tensors
from torchvision.io import read_image

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

    im_infos = []

    for im_set in im_sets:
        # im_names = []
        # Fetch all image names in txt file for this imageset

        videos = sorted(os.listdir(os.path.join(im_set, "SequenceAnnotations")))
        num_videos = len(videos)

        for vid_idx, vid in enumerate(videos):
            print(f'Loading video {vid_idx + 1}/{num_videos}')
            im_dir = os.path.join(im_set, 'sequences', vid)
            for (_, _, filenames) in os.walk(os.path.join(im_set, "SequenceAnnotations", vid)):
                for _, ann_file in enumerate(filenames):
                    im_info = {}
                    ann_info = ET.parse(os.path.join(im_set, "SequenceAnnotations", vid, ann_file))
                    root = ann_info.getroot()
                    size = root.find('size')
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
                    im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
                    im_info['filename'] = os.path.join(
                        im_dir, '{}.jpg'.format(im_info['img_id'])
                    )
                    im_info['width'] = width
                    im_info['height'] = height
                    detections = []

                    for obj in ann_info.findall('object'):
                        det = {}
                        difficult = int(obj.find('truncated').text)
                        bbox_info = obj.find('bndbox')
                        bbox = [
                            int(bbox_info.find('xmin').text) - 1,
                            int(bbox_info.find('ymin').text) - 1,
                            int(bbox_info.find('xmax').text) - 1,
                            int(bbox_info.find('ymax').text) - 1
                        ]
                        det['bbox'] = bbox
                        det['difficult'] = difficult
                        try:
                            label = label2idx[obj.find('name').text]
                            det['label'] = label
                            if label == 0:
                                print('Found background label for object {} in image {}. Skipping...'.format(ET.tostring(obj, encoding='unicode'), im_info['filename']))
                                continue
                        except KeyError:
                            continue
                        
                        # At test time eval does the job of ignoring difficult
                        detections.append(det)

                    # Skip images with no detections
                    if len(detections) == 0:
                        continue
                    
                    im_info['detections'] = detections
                    im_infos.append(im_info)
                # Iterate for every annotation file
            # Iterating over sequence annotations
        # Iterating over video sequence folders
    print('Total {} images found'.format(len(im_infos)))
    if len(im_infos) == 0:
        raise ValueError('No images found for the specified im_sets')
    return im_infos


class VisDroneDataset(Dataset):
    def __init__(self, split, im_sets, im_size=300):
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
        
        self.images_info = load_images_and_anns(self.im_sets,
                                                self.label2idx,
                                                self.fname)


    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        im_info = self.images_info[index]
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
