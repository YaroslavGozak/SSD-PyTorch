from typing import Dict, List, Tuple


VOC_FOREGROUND_CLASSES = [
    'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
    'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
    'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor',
]

IMAGENET_VID_FOREGROUND_CLASSES = [
    'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
    'cattle', 'dog', 'domestic cat', 'elephant', 'fox', 'giant panda',
    'giraffe', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle',
    'otter', 'panda', 'person', 'potted plant', 'rabbit', 'red panda',
    'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle', 'whale', 'zebra',
]

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

VOC_CLASSES = ['background'] + sorted(VOC_FOREGROUND_CLASSES)
IMAGENET_VID_CLASSES = ['background'] + sorted(IMAGENET_VID_FOREGROUND_CLASSES)

VOC_TO_IMAGENET_VID_NAME = {
    'aeroplane': 'airplane',
    'bicycle': 'bicycle',
    'bird': 'bird',
    'bus': 'bus',
    'car': 'car',
    'cat': 'domestic cat',
    'cow': 'cattle',
    'dog': 'dog',
    'horse': 'horse',
    'motorbike': 'motorcycle',
    'person': 'person',
    'pottedplant': 'potted plant',
    'sheep': 'sheep',
    'train': 'train',
}

COCO_TO_IMAGENET_VID_NAME = {
    'person': 'person',
    'bicycle': 'bicycle',
    'car': 'car',
    'motorbike': 'motorcycle',
    'aeroplane': 'airplane',
    'bus': 'bus',
    'train': 'train',
    'cat': 'cat',
    'dog': 'dog',
    'horse': 'horse',
    'sheep': 'sheep',
    'cow': 'cattle',
    'elephant': 'elephant',
    'bear': 'bear',
    'zebra': 'zebra',
    'giraffe': 'giraffe',
}

IMAGENET_VID_VOC_OVERLAP_CLASSES = set(VOC_TO_IMAGENET_VID_NAME.values())

COCO_TO_VID = {
    4: 0,   # airplane -> airplane
    1: 1,   # bicycle -> bicycle
    14: 2,  # bird -> bird
    8: 3,   # boat -> watercraft
    5: 4,   # bus -> bus
    2: 5,   # car -> car
    15: 6,  # cat -> domestic_cat
    19: 7,  # cow -> cattle
    16: 8,  # dog -> dog
    17: 9,  # horse -> horse
    3: 10,  # motorcycle -> motorcycle
    18: 11, # sheep -> sheep
    6: 12,  # train -> train
}


def build_label_maps(classes: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2idx = {class_name: idx for idx, class_name in enumerate(classes)}
    idx2label = {idx: class_name for idx, class_name in enumerate(classes)}
    return label2idx, idx2label


def infer_dataset_label_space(dataset_name: str) -> str:
    dataset_name = str(dataset_name)
    if dataset_name in {'voc', 'voc-small-objects', 'voc-vid', 'voc-video'}:
        return 'voc'
    if dataset_name == 'imagenet-vid':
        return 'imagenet-vid'
    if dataset_name == 'coco':
        return 'coco'
    return dataset_name


def get_label_space_classes(label_space: str) -> List[str]:
    label_space = str(label_space)
    if label_space == 'voc':
        return list(VOC_CLASSES)
    if label_space == 'imagenet-vid':
        return list(IMAGENET_VID_CLASSES)
    if label_space == 'coco':
        return list(COCO_CLASSES)
    raise ValueError(f'Unknown label space {label_space!r}')


def get_label_space_num_classes(label_space: str) -> int:
    return len(get_label_space_classes(label_space))
