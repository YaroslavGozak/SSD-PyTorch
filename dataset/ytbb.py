import os
from random import shuffle
import torch
import torchvision.transforms.v2
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
from torchvision import tv_tensors
from torchvision.io import read_image
from tqdm import tqdm
from itertools import islice

def labels_getter(transform_input):
    return (transform_input[1]["labels"], transform_input[1]["difficult"])

def load_images_and_anns(root_dir, label2idx, ann_fname):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param root_dir: Root directory containing ResizedSequences and SequenceAnnotations
    :param label2idx: Class Name to index mapping for dataset
    :param ann_fname: txt file containing frame paths in format {video}/{frame}
    :return: List of image information with annotations
    """
    
    im_infos = []
    no_detection_frames = 0
    error_paeses = 0
    
    images_dir = os.path.join(root_dir, 'ResizedSequences')
    annotations_dir = os.path.join(root_dir, 'SequenceAnnotations')
    
    # Read the annotation file to get list of video/frame pairs
    with open(ann_fname, 'r') as f:
        frame_paths = [line.strip() for line in f.readlines() if line.strip()]
        # frame_paths = [line.strip() for line in islice(f, 10000) if line.strip()]

    print(f'Found {len(frame_paths)} frame paths to process')
    for frame_path in tqdm(frame_paths, desc='Loading frames and annotations'):
        # Parse video/frame format
        video_name, frame_name = frame_path.split('/')
        frame_id = os.path.splitext(frame_name)[0]  # Remove .jpg extension if present
        
        # Construct paths
        img_path = os.path.join(images_dir, video_name, f"{frame_id}.jpg")
        xml_path = os.path.join(annotations_dir, video_name, f"{frame_id}.xml")
        
        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            print(f"Warning: Missing image or annotation for {frame_path}, skipping.")
            continue
            
        # Parse XML annotation
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image dimensions from XML
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                # If size not in XML, we'll get it from the image later
                width = None
                height = None
            
            im_info = {
                'img_id': frame_id,
                'filename': img_path,
                'width': width,
                'height': height
            }
            
            detections = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in label2idx:
                    print(f"Warning: Class '{class_name}' not in label2idx mapping, skipping object.")
                    continue
                    
                bbox_info = obj.find('bndbox')
                bbox = [
                    int(float(bbox_info.find('xmin').text)) - 1,  # Convert to 0-based
                    int(float(bbox_info.find('ymin').text)) - 1,
                    int(float(bbox_info.find('xmax').text)) - 1,
                    int(float(bbox_info.find('ymax').text)) - 1
                ]
                
                difficult_elem = obj.find('difficult')
                difficult = int(difficult_elem.text) if difficult_elem is not None else 0
                
                det = {
                    'label': label2idx[class_name],
                    'bbox': bbox,
                    'difficult': difficult
                }
                detections.append(det)
            
            if detections == []:
                no_detection_frames += 1
                continue
            im_info['detections'] = detections
            im_infos.append(im_info)

            # if len(im_infos) > 99:
            #     break
            
        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_path}: {e}")
            error_paeses += 1
            continue
    
    print('Total {} images found'.format(len(im_infos)))
    print(f"Frames with no detections: {no_detection_frames}")
    print(f"Frames with XML parse errors: {error_paeses}") 
    return im_infos


class YTBBDataset(Dataset):
    def __init__(self, split, root_dir, im_size=300):
        assert split in ['train', 'test'], "Split must be 'train' or 'test'"

        self.split = split
        self.im_size = im_size
        
        # Dataset configuration - using same values as VOC
        self.im_mean = [123.0, 117.0, 104.0]
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        # Train and test transformations - same as VOC
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
                # torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                #                                     std=self.imagenet_std)

            ]),
            'test': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.Resize(size=(self.im_size, self.im_size)),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                                                    std=self.imagenet_std)
            ]),
        }
        
        # Class mapping for YTBB (customize based on your dataset)
        # Update these classes based on your actual dataset classes
        classes = [
            'toilet', 'bear', 'umbrella', 'knife', 'skateboard', 'zebra',
            'person', 'airplane', 'bus', 'motorcycle', 'bicycle', 'car',
            'boat', 'train', 'truck', 'bird', 'cat', 'dog', 'horse', 'cow', 'elephant', 'potted plant',
            'giraffe'
        ]
        
        classes = sorted(classes)
        # Add background class as first class with 0 index
        classes = ['background'] + classes
        
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        
        # Load dataset
        ann_fname = f"{split}.txt"  # Assumes train.txt or test.txt files exist in root_dir
        ann_path = os.path.join(root_dir, ann_fname)
        
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
            
        self.images_info = load_images_and_anns(root_dir, self.label2idx, ann_path)
        shuffle(self.images_info)

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
        return im_tensor, targets, im_info['filename']
