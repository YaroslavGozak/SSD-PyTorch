import os
import random
import yaml
from tools.yt.utils import YTConfig
import torch
import cv2
import numpy as np
from torchvision.io import read_image
from model.ssd import SSD, generate_default_boxes, generate_ignore_regions
from dataset.ytbb import YTBBDataset
import torchvision

# Load config
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def denormalize_img(img_tensor):
    # img_tensor: (3, H, W), float32, [0,1] or normalized
    img = img_tensor.clone().detach().cpu().numpy()
    if img.max() <= 1.0:
        img = img * 255.0
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # (H, W, C)
    return img

def draw_boxes(img, boxes, color, label=None, thickness=1):
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(img, label, (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def main():
    config_path = 'config/ytbb.yaml'
    config = YTConfig(config_path)
    dataset_params = config.dataset_params
    model_params = config.model_params

    # Load dataset (test split for random sample)
    dataset = YTBBDataset('train', config.root_dir, im_size=512)
    idx = random.randint(0, len(dataset)-1)
    img_tensor, targets, filename = dataset[idx]
    print(f"Loaded image: {filename} with {len(targets['bboxes'])} GT boxes")
    img = denormalize_img(img_tensor)
    h, w = img.shape[:2]

    # Prepare GT boxes (unnormalize)
    gt_boxes = targets['bboxes'] * torch.tensor([w, h, w, h])
    gt_boxes = gt_boxes.numpy().astype(np.int32)

    # Load SSD model (no weights needed for default boxes)
    model = SSD(model_params, num_classes=dataset_params['num_classes'])
    model.eval()
    with torch.no_grad():
        # Forward pass to get feature maps
        feats = model.features(img_tensor.unsqueeze(0))
        conv_4_3_out_scaled = (model.scale_weight.view(1, -1, 1, 1) * torch.nn.functional.normalize(feats))
        conv_5_3_fc_out = model.conv5_3_fc(feats)
        conv8_2_out = model.conv8_2(conv_5_3_fc_out)
        conv9_2_out = model.conv9_2(conv8_2_out)
        conv10_2_out = model.conv10_2(conv9_2_out)
        conv11_2_out = model.conv11_2(conv10_2_out)
        outputs = [conv_4_3_out_scaled, conv_5_3_fc_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out]
        default_boxes = generate_default_boxes(outputs, model.aspect_ratios, model.scales)
        default_boxes_img = default_boxes[0] * torch.tensor([w, h, w, h], device=default_boxes[0].device)
        default_boxes_img = default_boxes_img.cpu().numpy().astype(np.int32)

    img_db = img.copy()
    # Draw all default boxes (in blue, very dense!)
    # img_db = draw_boxes(img_db, default_boxes_img, color=(255,0,0), thickness=1)

    # Draw GT boxes (in green)
    img_db = draw_boxes(img_db, gt_boxes, color=(0,255,0), thickness=2)

    # Prepare detector_outputs and gt_targets for ignore region calculation
    pretrained_detector = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
    pretrained_detector.eval()
    with torch.no_grad():
        detector_outputs_raw = pretrained_detector(img_tensor.unsqueeze(0))
    # detector_outputs_raw is a list of dicts, each with 'boxes', 'scores', 'labels'
    score_thresh = 0.5
    detector_outputs = []
    for det in detector_outputs_raw:
        boxes = det['boxes']
        scores = det['scores']
        keep = scores > score_thresh
        boxes = boxes[keep]
        boxes = boxes / torch.tensor([w, h, w, h], device=boxes.device)  # normalize
        detector_outputs.append({'boxes': boxes})
        print(f"Detected detector_outputs boxes: {boxes}")

        boxes_draw = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        boxes_draw = boxes_draw.cpu().numpy().astype(np.int32)
        print(f"Detected boxes_draw boxes: {boxes_draw}")
        # Draw ignore regions (in white)
        img_db = draw_boxes(img_db, boxes_draw, color=(255,255,255), thickness=2)

    gt_targets = [{'boxes': targets['bboxes']}]  # already normalized
    print(f"gt_targets boxes: {gt_targets[0]['boxes']}")
    ignore_regions = generate_ignore_regions(detector_outputs, gt_targets)
    if ignore_regions[0] is not None:
        with torch.no_grad():
            ignore_boxes = ignore_regions[0] * torch.tensor([w, h, w, h])
            ignore_boxes = ignore_boxes.cpu().numpy().astype(np.int32)
            print(f"Ignore regions: {ignore_boxes} boxes")
            # Draw ignore regions (in red)
            img_db = draw_boxes(img_db, ignore_boxes, color=(0,0,255), thickness=2)

    cv2.imshow('Default Boxes (blue), GT (green), Ignore (red)', img_db)
    print(f"Image: {filename}\nGreen: GT, Blue: Default, Red: Ignore regions")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
