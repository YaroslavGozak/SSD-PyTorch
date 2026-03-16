import os
import torch
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
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

    print('Loading images and annotations for {} split...'.format(split))

    for im_set in im_sets:
        im_names = []
        # Fetch all image names in txt file for this imageset
        for line in open(os.path.join(
                im_set, 'ImageSets', 'Main', '{}.txt'.format(ann_fname))):
            im_names.append(line.strip())

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
            if task == 'demo' and len(im_infos) >= 10:
                break
    print('Total {} images found'.format(len(im_infos)))
    return im_infos


class VOCRawDataset(Dataset):

    def _labels_getter(self, transform_input):
        """Helper function for SanitizeBoundingBoxes to extract labels and difficult flags."""
        return (transform_input[1]["labels"], transform_input[1]["difficult"])
    
    def __init__(self, split, im_sets, im_size=300, task=None):
        self.split = split
        self.task = task

        # Imagesets for this dataset instance (VOC2007/VOC2007+VOC2012/VOC2007-test)
        self.im_sets = im_sets
        self.fname = 'trainval' if self.split == 'train' else 'test'

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
        self.images_info = load_images_and_anns(self.im_sets,
                                                self.label2idx,
                                                self.fname,
                                                self.split,
                                                self.task)

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
