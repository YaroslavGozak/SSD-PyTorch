import torch
import torch.nn as nn


class CocoToVocAdapter(nn.Module):
    def __init__(self, base_model, voc_label2idx, conf_threshold=0.05, normalize_boxes=True):
        super().__init__()
        self.base_model = base_model
        self.conf_threshold = conf_threshold
        self.normalize_boxes = normalize_boxes

        # COCO class id -> VOC class name
        coco_to_voc_name = {
            1: 'person',
            2: 'bicycle',
            3: 'car',
            4: 'motorbike',     # COCO "motorcycle"
            5: 'aeroplane',     # COCO "airplane"
            6: 'bus',
            7: 'train',
            9: 'boat',
            16: 'bird',
            17: 'cat',
            18: 'dog',
            19: 'horse',
            20: 'sheep',
            21: 'cow',
            44: 'bottle',
            62: 'chair',
            63: 'sofa',         # COCO "couch"
            64: 'pottedplant',  # COCO "potted plant"
            67: 'diningtable',  # COCO "dining table"
            72: 'tvmonitor',    # COCO "tv"
        }

        self.coco_to_voc_idx = {
            coco_id: voc_label2idx[voc_name]
            for coco_id, voc_name in coco_to_voc_name.items()
            if voc_name in voc_label2idx
        }

    def forward(self, images, *args, **kwargs):
        detections = self.base_model(images, *args, **kwargs)

        # In training mode torchvision detectors return a loss dict
        if isinstance(detections, dict):
            return detections

        out = []
        for img, det in zip(images, detections):
            boxes = det['boxes']
            labels = det['labels']
            scores = det['scores']

            keep = torch.zeros_like(labels, dtype=torch.bool)
            mapped_labels = torch.zeros_like(labels)

            for coco_id, voc_id in self.coco_to_voc_idx.items():
                m = (labels == coco_id)
                keep |= m
                mapped_labels[m] = voc_id

            if self.conf_threshold is not None:
                keep &= (scores >= float(self.conf_threshold))

            boxes = boxes[keep]
            scores = scores[keep]
            labels = mapped_labels[keep]

            if self.normalize_boxes and boxes.numel() > 0:
                h, w = img.shape[-2:]
                scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
                boxes = (boxes / scale).clamp(0.0, 1.0)

            out.append({
                'boxes': boxes,
                'labels': labels.long(),
                'scores': scores.float(),
            })
        return out