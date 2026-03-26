"""
Shared inference pipeline utilities.

Consumed by:
    - tools/infer_sequentially_kalman_roi.py  (streaming visualisation)
    - tools/benchmarks/benchmark_framework_vid.py  (headless benchmarking)
"""
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml

from dataset.visdrone import VisDroneDataset
from dataset.voc import VOCDataset
from dataset.voc_vid import VOCVideoDataset
from dataset.voc_small_objects import VOCSmallObjectsDataset
from dataset.ytbb import YTBBDataset
from model.roissd import RoiSSD
from model.ssd import SSD
from tools.helpers.roi_merger import greedy_roi_merge, simple_roi_merge, simple_roi_merge_v2
from tools.trackers.kalman_roi_tracker import KalmanRoiTracker
from tools.trackers.static_padding_tracker import StaticPaddingTracker
from tools.trackers.relative_object_size_padding_tracker import RelativeObjectSizePaddingTracker
from torch.utils.data.dataloader import DataLoader


# ---------------------------------------------------------------------------
# Device selection (module-level, shared across all callers)
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Merge strategy registry
# ---------------------------------------------------------------------------
def _merge_none(rois: List, **_) -> List:
    return list(rois)

MERGE_STRATEGIES: Dict[str, Callable] = {
    "greedy":    lambda rois, tau=150000.0: greedy_roi_merge(rois, tau=tau),
    "simple":    lambda rois, **_: simple_roi_merge(rois),
    "simple_v2": lambda rois, **_: simple_roi_merge_v2(rois),
    "none":      _merge_none,
}


# ---------------------------------------------------------------------------
# Tracker factory
# ---------------------------------------------------------------------------
def build_tracker(tracker_cfg: Dict[str, Any]):
    """Instantiate a tracker from a config dict with a 'type' key."""
    kind = tracker_cfg["type"]
    if kind == "kalman":
        p = tracker_cfg.get("kalman", {})
        return KalmanRoiTracker(**p)
    elif kind == "static_padding":
        p = tracker_cfg.get("static_padding", {})
        return StaticPaddingTracker(**p)
    elif kind == "relative_padding":
        p = tracker_cfg.get("relative_padding", {})
        return RelativeObjectSizePaddingTracker(**p)
    else:
        raise ValueError(f"Unknown tracker type: {kind!r}")


# ---------------------------------------------------------------------------
# Model + dataset loading
# ---------------------------------------------------------------------------
def load_model_and_dataset(args):
    """Load model and dataset from a training config path (args.config_path)."""
    with open(args.config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise

    dataset_config = config['dataset_params']
    train_config   = config['train_params']
    dataset_name   = str(train_config['dataset'])

    if dataset_name == 'vis-drone':
        dataset = VisDroneDataset(
            'test',
            im_sets=dataset_config['test_im_sets'],
            im_size=dataset_config['im_size'],
        )
    elif dataset_name == 'ytbb':
        dataset = YTBBDataset(
            'test',
            root_dir=dataset_config['root_dir'],
            im_size=dataset_config['im_size'],
        )
    elif dataset_name == 'voc':
        dataset = VOCDataset(
            'test',
            im_sets=dataset_config['test_im_sets'],
            im_size=dataset_config['im_size'],
            transform_name=dataset_config['transform_name'],
        )
    elif dataset_name in ('voc-vid', 'voc-video'):
        dataset = VOCVideoDataset(
            'test',
            im_sets=dataset_config['test_im_sets'],
            im_size=dataset_config['im_size'],
            transform_name=dataset_config['transform_name'],
        )
    elif dataset_name == 'voc-small-objects':
        dataset = VOCSmallObjectsDataset(
            'test',
            im_sets=dataset_config['test_im_sets'],
            im_size=dataset_config['im_size'],
            transform_name=dataset_config['transform_name'],
        )
    else:
        raise Exception(f'Unknown dataset name {dataset_name!r}')

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model_name = str(train_config['model'])
    if model_name == 'ssd':
        model = SSD(config=config['model_params'], num_classes=dataset_config['num_classes'])
    elif model_name == 'roissd':
        model = RoiSSD(config=config['model_params'], num_classes=dataset_config['num_classes'])
    else:
        raise Exception(f'Unknown model name {model_name!r}')

    model.to(device=device)
    model.eval()

    model_task_path = os.path.join('trained_models', train_config['task_name'])
    ckpt_path = os.path.join(model_task_path, train_config['ckpt_name'])
    assert os.path.exists(ckpt_path), f'No checkpoint exists at {ckpt_path}'

    print('Loading checkpoint...')
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print('Loaded model from full checkpoint format')
    else:
        model.load_state_dict(checkpoint)
        print('Loaded model only (old checkpoint format)')

    return model, dataset, data_loader, config


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def ensure_im_size_tuple(im_size: Any) -> Tuple[int, int]:
    if isinstance(im_size, int):
        return im_size, im_size
    if isinstance(im_size, (list, tuple)) and len(im_size) == 2:
        return int(im_size[0]), int(im_size[1])
    raise ValueError(f'Unsupported im_size format: {im_size}')


def preprocess_bgr_for_model(
    image_bgr: np.ndarray,
    im_size_hw: Tuple[int, int],
    target_device: torch.device = None,
) -> torch.Tensor:
    """Crop BGR → normalised float tensor on target_device (defaults to module device)."""
    if target_device is None:
        target_device = device
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=im_size_hw,
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def tensor_to_detection_list(
    detections: Dict[str, torch.Tensor],
    idx2label: Dict[int, str],
    image_w: int,
    image_h: int,
    offset_xy: Tuple[int, int] = (0, 0),
) -> List[Dict[str, Any]]:
    """Convert raw model output dict to a list of pixel-coordinate detection dicts."""
    ox, oy = offset_xy
    out: List[Dict[str, Any]] = []
    for idx, box in enumerate(detections['boxes']):
        x1, y1, x2, y2 = box.detach().cpu().numpy().tolist()
        out.append({
            'bbox': [
                int(round(ox + image_w * x1)),
                int(round(oy + image_h * y1)),
                int(round(ox + image_w * x2)),
                int(round(oy + image_h * y2)),
            ],
            'class':      idx2label[int(detections['labels'][idx].detach().cpu().item())],
            'confidence': float(detections['scores'][idx].detach().cpu().item()),
        })
    return out


def clip_bbox(box: List[int], frame_w: int, frame_h: int) -> Optional[List[int]]:
    """Clamp a bbox to frame bounds; return None if the result is degenerate."""
    x1 = max(0, min(frame_w - 1, box[0]))
    y1 = max(0, min(frame_h - 1, box[1]))
    x2 = max(0, min(frame_w - 1, box[2]))
    y2 = max(0, min(frame_h - 1, box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def merge_detections_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Class-wise NMS over a merged list of pixel-coordinate detections."""
    if not detections:
        return []
    class_to_indices: Dict[str, List[int]] = {}
    for i, det in enumerate(detections):
        class_to_indices.setdefault(str(det['class']), []).append(i)
    kept: List[Dict[str, Any]] = []
    for _, indices in class_to_indices.items():
        boxes  = torch.tensor([detections[i]['bbox']       for i in indices], dtype=torch.float32)
        scores = torch.tensor([detections[i]['confidence'] for i in indices], dtype=torch.float32)
        for li in torchvision.ops.nms(boxes, scores, iou_threshold).tolist():
            kept.append(detections[indices[li]])
    return kept


def extract_gt_boxes(
    target: Dict[str, Any],
    frame_w: int,
    frame_h: int,
) -> List[List[int]]:
    """Return GT boxes as pixel-coordinate integer lists (no label info)."""
    if 'bboxes' in target:
        raw = target['bboxes'][0]
    elif 'boxes' in target:
        raw = target['boxes'][0]
    else:
        return []
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy().tolist()
    out = []
    for box in raw:
        clipped = clip_bbox(
            [int(round(float(box[0]) * frame_w)),
             int(round(float(box[1]) * frame_h)),
             int(round(float(box[2]) * frame_w)),
             int(round(float(box[3]) * frame_h))],
            frame_w, frame_h,
        )
        if clipped is not None:
            out.append(clipped)
    return out


# ---------------------------------------------------------------------------
# Per-frame result container
# ---------------------------------------------------------------------------
@dataclass
class FrameResult:
    final_detections:  List[Dict[str, Any]]  = field(default_factory=list)
    rois_used:         List[List[int]]        = field(default_factory=list)
    next_frame_rois:   List[List[int]]        = field(default_factory=list)
    use_full_frame:    bool                   = True
    latency_s:         float                  = 0.0  # total inference + NMS + tracker update
    merge_latency_s:   float                  = 0.0  # ROI merge step only (0.0 for full-frame)


# ---------------------------------------------------------------------------
# Core per-frame processing (the shared inner loop body)
# ---------------------------------------------------------------------------
def process_frame(
    *,
    model,
    idx2label: Dict[int, str],
    frame_bgr: np.ndarray,
    im_tensor: torch.Tensor,          # pre-loaded by dataloader (full-frame transform)
    tracker,
    next_frame_rois: List[List[int]],
    frame_idx: int,
    key_frame_interval: int,
    im_size_hw: Tuple[int, int],
    conf_threshold: float,
    nms_iou: float,
    merge_fn: Callable = None,        # defaults to greedy
    merge_tau: float = 150000.0,
    model_device: torch.device = None,
) -> FrameResult:
    """
    Run one frame through the full pipeline:
      1. Decide full-frame vs ROI mode.
      2. Run model inference (optionally per-ROI crop).
      3. NMS + confidence filter.
      4. Update tracker.
    Returns a FrameResult with detections, ROI metadata, and timing.
    """
    if model_device is None:
        model_device = device
    if merge_fn is None:
        merge_fn = MERGE_STRATEGIES["greedy"]

    frame_h, frame_w = frame_bgr.shape[:2]
    is_key_frame  = (frame_idx % key_frame_interval == 0)
    use_full_frame = (not next_frame_rois) or is_key_frame

    all_detections: List[Dict[str, Any]] = []
    rois_used: List[List[int]] = []
    merge_latency_s = 0.0

    t0 = time.perf_counter()

    if use_full_frame:
        _, raw = model(im_tensor.float().to(model_device), None)
        all_detections = tensor_to_detection_list(raw[0], idx2label, frame_w, frame_h)
    else:
        # Merge ROIs, then run one inference pass per cluster
        tm = time.perf_counter()
        clusters = merge_fn(next_frame_rois, tau=merge_tau)
        merge_latency_s = time.perf_counter() - tm

        for roi in clusters:
            roi_c = clip_bbox(roi, frame_w, frame_h)
            if roi_c is None:
                continue
            rx1, ry1, rx2, ry2 = roi_c
            crop = frame_bgr[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue
            rois_used.append(roi_c)
            crop_tensor = preprocess_bgr_for_model(crop, im_size_hw, target_device=model_device)
            _, raw = model(crop_tensor.unsqueeze(0).to(model_device), None)
            all_detections.extend(
                tensor_to_detection_list(
                    raw[0], idx2label, crop.shape[1], crop.shape[0],
                    offset_xy=(rx1, ry1),
                )
            )

    # NMS + confidence filter
    merged = merge_detections_nms(all_detections, iou_threshold=nms_iou)
    final_detections: List[Dict[str, Any]] = []
    tracker_input: List[Dict[str, Any]] = []
    for det in merged:
        if float(det['confidence']) < conf_threshold:
            continue
        c = clip_bbox(det['bbox'], frame_w, frame_h)
        if c is None:
            continue
        d = {'bbox': c, 'class': det['class'], 'confidence': float(det['confidence'])}
        tracker_input.append(d)
        final_detections.append(d)

    tracker_result = tracker.update(tracker_input, frame_shape=(frame_h, frame_w))
    latency_s = time.perf_counter() - t0

    return FrameResult(
        final_detections=final_detections,
        rois_used=rois_used,
        next_frame_rois=[r['roi'] for r in tracker_result['rois']],
        use_full_frame=use_full_frame,
        latency_s=latency_s,
        merge_latency_s=merge_latency_s,
    )
