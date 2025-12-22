import os
import xml.etree.ElementTree as ET


def read_annotation_file(ann_dir, im_dir, ann_file, label2idx):
    """Reads an annotation file and returns its content as a list of lines.

    Args:
        path (str): The path to the annotation file.
    """
    im_info = {}
    ann_info = ET.parse(os.path.join(ann_dir, ann_file))
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

    success = True
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
                success = False
                return im_info, success
        except KeyError:
            success = False
            return im_info, success
                        
        # At test time eval does the job of ignoring difficult
        detections.append(det)
                    
    im_info['detections'] = detections
    return im_info, success

def read_annotation_file(ann_path, im_path, label2idx):
    """Reads an annotation file and returns its content as a list of lines.

    Args:
        ann_path (str): The path to the annotation file.
        im_path (str): The path to the image file.
        label2idx (dict): A dictionary mapping label names to indices.
    """
    im_info = {}
    ann_info = ET.parse(ann_path)
    root = ann_info.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    im_info['img_id'] = os.path.basename(ann_path).split('.xml')[0]
    im_info['filename'] = im_path
    im_info['width'] = width
    im_info['height'] = height
    detections = []

    success = True
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
                success = False
                return im_info, success
        except KeyError:
            print('Label not found for object {} in image {}. Skipping...'.format(ET.tostring(obj, encoding='unicode'), im_info['filename']))
            success = False
            return im_info, success
                        
        # At test time eval does the job of ignoring difficult
        detections.append(det)
                    
    im_info['detections'] = detections
    return im_info, success
