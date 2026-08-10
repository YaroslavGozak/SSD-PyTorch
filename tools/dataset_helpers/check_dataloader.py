import torch
import argparse
import os
import yaml
import random
from tqdm import tqdm
from model.ssd import SSD
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataset.visdrone import VisDroneDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area', difficult=None):
    # det_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, score], ...],
    #       'car' : [[x1, y1, x2, y2, score], ...]
    #   }
    #   {det_boxes_img_2},
    #   ...
    #   {det_boxes_img_N},
    # ]
    #
    # gt_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2], ...],
    #       'car' : [[x1, y1, x2, y2], ...]
    #   },
    #   {gt_boxes_img_2},
    #   ...
    #   {gt_boxes_img_N},
    # ]

    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)

    all_aps = {}
    # average precisions for ALL classes
    aps = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
        #   ...
        # ]

        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        num_difficults = sum([sum(difficults_label[label]) for difficults_label in difficult])

        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            im_gt_difficults = difficult[im_idx][label]

            max_iou_found = -1
            max_iou_gt_idx = -1

            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            if max_iou_found >= iou_threshold:
                if not im_gt_difficults[max_iou_gt_idx]:
                    if not gt_matched[im_idx][max_iou_gt_idx]:
                        # If tp then we set this gt box as matched
                        gt_matched[im_idx][max_iou_gt_idx] = True
                        tp[det_idx] = 1
                    else:
                        fp[det_idx] = 1
            else:
                fp[det_idx] = 1

        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        # recalls = tp / np.maximum(num_gts, eps)
        recalls = tp / np.maximum(num_gts - num_difficults, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Get max of those precision values
                prec_interp_pt= prec_interp_pt.max() if prec_interp_pt.size>0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps


def load_model_and_dataset(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    dataset = VisDroneDataset('test',
                     im_sets=dataset_config['test_im_sets'])
    test_dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SSD(config=model_config,
                num_classes=dataset_config['num_classes'])
    model.to(device=torch.device(device))
    model.eval()

    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['ckpt_name'])), \
        "No checkpoint exists at {}".format(os.path.join(train_config['task_name'],
                                                         train_config['ckpt_name']))
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                       train_config['ckpt_name']),
                                     map_location=device))
    return model, dataset, test_dataset_loader, config


def display_dataloader_samples(config_path, n_images=5, save_images=True, show_images=False):
    """
    Display first N images with their bounding boxes from the dataloader
    
    Args:
        config_path (str): Path to config file
        n_images (int): Number of images to display
        save_images (bool): Whether to save images to disk
        show_images (bool): Whether to display images using matplotlib
    """
    # Read the config file
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    dataset_config = config['dataset_params']
    
    # Create dataset and dataloader
    dataset = VisDroneDataset('test', im_sets=dataset_config['test_im_sets'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    if save_images and not os.path.exists('dataloader_samples'):
        os.mkdir('dataloader_samples')
    
    print(f"Displaying first {n_images} images from dataloader...")
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset_config['num_classes']}")
    
    # Get first n_images from dataloader
    for i, (image_tensors, target, filenames) in enumerate(dataloader):
        if i >= n_images:
            break
            
        # Get the first (and only) item from batch (batch_size=1)
        image_tensor = image_tensors[0]
        # print(targets)
        # target = targets[0]
        filename = filenames[0]
        
        # Load original image for visualization
        assert isinstance(filename, str), "Filename should be a string, got {}".format(type(filename))
        original_img = cv2.imread(filename)
        if original_img is None:
            print(f"Could not load image: {filename}")
            continue
            
        h, w = original_img.shape[:2]
        display_img = original_img.copy()

        # # Convert tensor -> NumPy (H, W, C)
        # img = image_tensor.permute(1, 2, 0).cpu().numpy()

        # # Convert RGB -> BGR and scale
        # display_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Get bounding boxes and labels
        bboxes = target['bboxes']
        labels = target['labels']
        
        print(f"\nImage {i+1}: {os.path.basename(filename)}")
        print(f"  Image size: {w}x{h}")
        print(f"  Number of objects: {len(bboxes)}")
        print(target)
        
        # Draw bounding boxes
        for j, (bbox, label) in enumerate(zip(bboxes[0], labels[0])):
            # Convert normalized coordinates to pixel coordinates
            x1, y1, x2, y2 = bbox.numpy()
            x1, y1, x2, y2 = int(w * x1), int(h * y1), int(w * x2), int(h * y2)
            
            # Get class name
            class_name = dataset.idx2label[label.item()]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green color for GT boxes
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            
            # Add label text
            label_text = f"{class_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            
            # Get text size for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(display_img, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(display_img, label_text, (x1, y1 - 5), 
                       font, font_scale, (0, 0, 0), thickness)
            
            print(f"    Object {j+1}: {class_name} at [{x1}, {y1}, {x2}, {y2}]")
        
        if save_images:
            output_path = f'dataloader_samples/sample_{i+1:03d}.jpg'
            cv2.imwrite(output_path, display_img)
            print(f"  Saved to: {output_path}")
        
        if show_images:
            # Convert BGR to RGB for matplotlib
            display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(display_img_rgb)
            plt.title(f"Sample {i+1}: {os.path.basename(filename)} ({len(bboxes)} objects)")
            plt.axis('off')
            plt.show()
    
    print(f"\nDisplayed {min(n_images, len(dataloader))} images from dataloader")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd inference and dataloader visualization')
    parser.add_argument('--config', dest='config_path',
                        default='config/vis-drone.yaml', type=str)
    parser.add_argument('--display_dataloader', dest='display_dataloader',
                        default=True, type=bool,
                        help='Display first N images from dataloader with bounding boxes')
    parser.add_argument('--n_images', dest='n_images',
                        default=10, type=int,
                        help='Number of images to display from dataloader')
    parser.add_argument('--show_images', dest='show_images',
                        default=True, type=bool,
                        help='Show images using matplotlib (requires display)')
    args = parser.parse_args()

    if args.display_dataloader:
        display_dataloader_samples(
            config_path=args.config_path,
            n_images=args.n_images,
            save_images=False,
            show_images=args.show_images
        )
