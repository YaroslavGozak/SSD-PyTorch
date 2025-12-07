from tools.roi_merger import greedy_roi_merge, simple_roi_merge, simple_roi_merge_v2
from tools.utils import read_annotation_file
import torch
import torchvision.transforms.v2
from torchvision import tv_tensors
from torchvision.io import read_image

from transformers.random_roi_crop import RandomROICrop

def labels_getter(transform_input):
    return (transform_input[1]["labels"], transform_input[1]["difficult"])

def load_image_and_ann(im_path, label2idx):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_path: Ex: "H:\Projects\University\NeuralNetworks_ModelsAndDatasets\Datasets\VisDrone2019-VID-train\VisDrone2019-VID-train\ResizedSequences\uav0000263_03289_v\0000008.jpg"
    :param label2idx: Class Name to index mapping for dataset
    :return: image info
    """

    ann_path = im_path.replace('ResizedSequences', 'SequenceAnnotations').replace('.jpg', '.xml')
    im_info, success = read_annotation_file(ann_path, im_path, label2idx)
    if not success:
        return None, False

    # Skip images with no detections
    if len(im_info.get('detections', [])) == 0:
        return None, False

    return im_info


class TestTransformDataset():
    def __init__(self, 
        im_size=512, 
        alpha_w: float = 0.3,
        alpha_h: float = 0.3,
        delta_x: float = 8.0,
        delta_y: float = 8.0,):

        # Imagesets for this dataset instance (VOC2007/VOC2007+VOC2012/VOC2007-test)
        self.fname = 'trainval'
        self.im_size = im_size
        self.im_mean = [123.0, 117.0, 104.0]
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        self.alpha_w = alpha_w
        self.alpha_h = alpha_h
        self.delta_x = delta_x
        self.delta_y = delta_y

        # Train and test transformations
        self.transforms = {
            'train': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.RandomPhotometricDistort(),
                torchvision.transforms.v2.RandomZoomOut(fill=self.im_mean),
                # torchvision.transforms.v2.RandomIoUCrop(),
                torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.v2.Resize(size=(self.im_size, self.im_size)),
                # torchvision.transforms.v2.SanitizeBoundingBoxes(
                #     labels_getter=labels_getter),
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
            ])
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


    def get_image(self, im_path):
        im_info = load_image_and_ann(im_path, self.label2idx)
        im, targets, transformed_im, simple_targets, _, simple_v2_targets, _, greedy_targets = self.__get_im_and_targets(im_info)
        
        return im, targets, transformed_im, simple_targets, simple_v2_targets, greedy_targets
    
    def __get_im_and_targets(self, im_info):
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
        
        bboxes = [self.add_padding_to_bbox(bbox, im.shape[-1], im.shape[-2], self.alpha_w, self.alpha_h, self.delta_x, self.delta_y) for bbox in targets['bboxes'].tolist()]
        rois = simple_roi_merge(bboxes)
        simple_merge_im = im.clone()
        simple_merge_targets = {}
        simple_merge_targets['bboxes'] = tv_tensors.BoundingBoxes(
            [detection for detection in rois],
            format='XYXY', canvas_size=im.shape[-2:])
        simple_merge_targets['labels'] = torch.as_tensor(
            [detection['label'] for detection in im_info['detections']])
        simple_merge_targets['difficult'] = torch.as_tensor(
            [detection['difficult']for detection in im_info['detections']])
        
        bboxes = [self.add_padding_to_bbox(bbox, im.shape[-1], im.shape[-2], self.alpha_w, self.alpha_h, self.delta_x, self.delta_y) for bbox in targets['bboxes'].tolist()]
        rois = simple_roi_merge_v2(bboxes)
        simple_merge_v2_im = im.clone()
        simple_merge_v2_targets = {}
        simple_merge_v2_targets['bboxes'] = tv_tensors.BoundingBoxes(
            [detection for detection in rois],
            format='XYXY', canvas_size=im.shape[-2:])
        simple_merge_v2_targets['labels'] = torch.as_tensor(
            [detection['label'] for detection in im_info['detections']])
        simple_merge_v2_targets['difficult'] = torch.as_tensor(
            [detection['difficult']for detection in im_info['detections']])
        
        bboxes = [self.add_padding_to_bbox(bbox, im.shape[-1], im.shape[-2], self.alpha_w, self.alpha_h, self.delta_x, self.delta_y) for bbox in targets['bboxes'].tolist()]
        rois = greedy_roi_merge(bboxes)
        greedy_merge_im = im.clone()
        greedy_merge_targets = {}
        greedy_merge_targets['bboxes'] = tv_tensors.BoundingBoxes(
            [detection for detection in rois],
            format='XYXY', canvas_size=im.shape[-2:])
        greedy_merge_targets['labels'] = torch.as_tensor(
            [detection['label'] for detection in im_info['detections']])
        greedy_merge_targets['difficult'] = torch.as_tensor(
            [detection['difficult']for detection in im_info['detections']])

        # Transform the image and targets
        # transformed_info = self.transforms['train'](im, targets)
        # transformed_im, transformed_targets = transformed_info

        # h, w = transformed_im.shape[-2:]
        # wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(transformed_targets['bboxes'])
        # transformed_targets['bboxes'] = transformed_targets['bboxes'] / wh_tensor
        return im, targets, simple_merge_im, simple_merge_targets, simple_merge_v2_im, simple_merge_v2_targets, greedy_merge_im, greedy_merge_targets
    
    def add_padding_to_bbox(self, bbox, im_width, im_height, alpha_w, alpha_h, delta_x, delta_y):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        px = alpha_w * w + delta_x
        py = alpha_h * h + delta_y

        new_x1 = max(0.0, x1 - px)
        new_y1 = max(0.0, y1 - py)
        new_x2 = min(im_width, x2 + px)
        new_y2 = min(im_height, y2 + py)

        return [new_x1, new_y1, new_x2, new_y2]
    
