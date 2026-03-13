import os
from dataset.transforms.fixed_padding_roi_crop_test_transform import FixedPaddingRoiCropTestTransform
from dataset.transforms.letterbox_transform import LetterboxTransform
from dataset.transforms.no_resize_transform import NoResizeTransform
from dataset.transforms.resize_longer_edge_test_transform import ResizeLongerEdgeTestTransform
from dataset.transforms.roi_crop_test_transform import RoiCropTestTransform
from dataset.transforms.ssd_transform import SsdTransform
import torch
import torchvision.transforms.v2
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
from torchvision import tv_tensors
from torchvision.io import read_image


def load_images_and_anns(im_sets, label2idx, ann_fname, split, task=None):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_sets: Sets of images to consider
    :param label2idx: Class Name to index mapping for dataset
    :param ann_fname: txt file containing image names{trainval.txt/test.txt}
    :param split: train/test
    :param task: Optional task parameter
    :return:
    """
    im_infos = []

    for im_set in im_sets:
        # Fetch all image names in txt file for this imageset,
        # using dict.fromkeys to deduplicate while preserving order
        with open(os.path.join(im_set, 'ImageSets', 'Main', 'small_objects_{split}.txt'.format(split=split))) as f:
            im_names = list(dict.fromkeys(line.strip() for line in f if line.strip()))

        # Set annotation and image path
        ann_dir = os.path.join(im_set, 'Annotations')
        im_dir = os.path.join(im_set, 'JPEGImages')
        for im_name in im_names:
            ann_file = os.path.join(ann_dir, '{}.xml'.format(im_name))
            im_info = {}
            ann_info = ET.parse(ann_file)
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
                label = label2idx[obj.find('name').text]
                difficult = int(obj.find('difficult').text)
                bbox_info = obj.find('bndbox')
                bbox = [
                    int(bbox_info.find('xmin').text) - 1,
                    int(bbox_info.find('ymin').text) - 1,
                    int(bbox_info.find('xmax').text) - 1,
                    int(bbox_info.find('ymax').text) - 1
                ]
                det['label'] = label
                det['bbox'] = bbox
                det['difficult'] = difficult
                # At test time eval does the job of ignoring difficult
                detections.append(det)

            im_info['detections'] = detections
            im_infos.append(im_info)
            if task == 'demo' and len(im_infos) >= 2:
                break
    print('Total {} images found'.format(len(im_infos)))
    return im_infos


class VOCSmallObjectsDataset(Dataset):

    def _labels_getter(self, transform_input):
        """Helper function for SanitizeBoundingBoxes to extract labels and difficult flags."""
        return (transform_input[1]["labels"], transform_input[1]["difficult"])
    
    def __init__(self, split, im_sets, im_size=300, transform_name='ssd', task=None):
        self.split = split
        self.task = task
        self.transform_name = transform_name

        # Imagesets for this dataset instance (VOC2007/VOC2007+VOC2012/VOC2007-test)
        self.im_sets = im_sets
        self.fname = 'trainval' if self.split == 'train' else 'test'
        self.im_size = im_size
        self.im_mean = [123.0, 117.0, 104.0]
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        self.im_mean = tuple(a + b for a, b in zip(self.im_mean, (20, 20, 15))) # correct color

        # Train and test transformations
        if self.transform_name == 'ssd':
            self.transforms = SsdTransform(im_size, self.im_mean, self.imagenet_mean, self.imagenet_std).transforms
        elif self.transform_name == 'letterbox':
            self.transforms = LetterboxTransform(im_size, self.im_mean, self.imagenet_mean, self.imagenet_std).transforms
        elif self.transform_name == 'resize_longer_edge':
            self.transforms = ResizeLongerEdgeTestTransform(im_size, self.im_mean, self.imagenet_mean, self.imagenet_std).transforms
        elif self.transform_name == 'roi_crop_test_transform':
            self.transforms = RoiCropTestTransform(im_size, self.im_mean, self.imagenet_mean, self.imagenet_std).transforms
        elif self.transform_name == 'no_resize_transform':
            self.transforms = NoResizeTransform(self.im_mean, self.imagenet_mean, self.imagenet_std).transforms
        elif self.transform_name == 'fixed_padding_20_roi_crop':
            self.transforms = FixedPaddingRoiCropTestTransform(im_size, self.imagenet_mean, self.imagenet_std, pad_x=20, pad_y=20).transforms
        elif self.transform_name == 'fixed_padding_50_roi_crop':
            self.transforms = FixedPaddingRoiCropTestTransform(im_size, self.imagenet_mean, self.imagenet_std, pad_x=50, pad_y=50).transforms
        else:
            raise Exception('Unknown transform name "{}"'.format(self.transform_name))

        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        # We need to add background class as well with 0 index
        classes = ['background'] + classes

        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(self.im_sets,
                                                self.label2idx,
                                                self.fname,
                                                self.split,
                                                self.task)

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
        orig_h, orig_w = im.shape[-2:]
        targets['orig_size'] = (orig_h, orig_w)

        # Transform the image and targets
        transformed_info = self.transforms[self.split](im, targets)
        im_tensor, targets = transformed_info

        h, w = im_tensor.shape[-2:]
        wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(targets['bboxes'])
        targets['bboxes'] = targets['bboxes'] / wh_tensor
        return im_tensor, targets, im_info['filename']
