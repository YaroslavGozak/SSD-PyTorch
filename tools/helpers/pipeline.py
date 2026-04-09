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
from tools.trackers.oracle_gt_tracker import OracleGtTracker
from torch.utils.data.dataloader import DataLoader



IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class YoloV8Adapter:
    """YOLO wrapper that returns normalized detections in project format."""

    def __init__(self, weights_path: str, device: torch.device):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLO inference requires ultralytics. Install with: pip install ultralytics"
            ) from e
        self._yolo = YOLO(weights_path)
        self.device = device

    def to(self, device: torch.device = None, **_kwargs):
        if device is not None:
            self.device = device
        return self

    def eval(self):
        return self
    
    def parameters(self):
        """Delegate to the underlying PyTorch module so callers can do next(model.parameters()).device."""
        return self._yolo.model.parameters()

    def __call__(self, images: torch.Tensor, _targets=None):
        # Reverse ImageNet normalization so YOLO receives images in [0, 1] float range.
        # The SSD dataset transform applies mean/std normalization; YOLO does its own
        # preprocessing and expects un-normalized [0, 1] float tensors.
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=images.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=images.device).view(1, 3, 1, 1)
        images_01 = (images.float() * std + mean).clamp(0.0, 1.0)
        results = self._yolo.predict(source=images_01, verbose=False)
        _, _, h, w = images.shape
        out = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                out.append({
                    'boxes': torch.empty((0, 4), dtype=torch.float32, device=images.device),
                    'labels': torch.empty((0,), dtype=torch.int64, device=images.device),
                    'scores': torch.empty((0,), dtype=torch.float32, device=images.device),
                })
                continue

            xyxy = res.boxes.xyxy.to(images.device).float()
            boxes = xyxy.clone()
            boxes[:, [0, 2]] /= float(w)
            boxes[:, [1, 3]] /= float(h)

            out.append({
                'boxes': boxes,
                'labels': (res.boxes.cls.to(images.device).long() + 1),
                'scores': res.boxes.conf.to(images.device).float(),
            })
        return None, out


def run_model_inference(model, images: torch.Tensor):
    """Run model forward and normalize output shape to (raw, detections)."""
    try:
        raw, detections = model(images, None)
    except TypeError:
        out = model(images)
        if isinstance(out, tuple) and len(out) == 2:
            raw, detections = out
        else:
            raw, detections = None, out

    if isinstance(detections, dict):
        detections = [detections]
    return raw, detections


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
# Model ROI grid alignment
# ---------------------------------------------------------------------------
def _model_roi_grid(model) -> int:
    """Return the dimension alignment (in tensor pixels) required for ROI crops.

    YOLO models have a stride of 32, so any crop fed to them must have H and W
    divisible by 32.  SSD / RoiSSD accept arbitrary sizes.
    """
    if isinstance(model, YoloV8Adapter):
        return 32
    try:
        from ultralytics import YOLO as _YOLO
        if isinstance(model, _YOLO):
            return 32
    except ImportError:
        pass
    return 1


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
    elif kind == "oracle_gt":
        p = tracker_cfg.get("oracle_gt", {})
        return OracleGtTracker(**p)
    else:
        raise ValueError(f"Unknown tracker type: {kind!r}")


# ---------------------------------------------------------------------------
# Model + dataset loading
# ---------------------------------------------------------------------------
def load_model_and_dataset(device, args):
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
    elif model_name == 'ssd-original':
        model = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
    elif model_name == 'roissd':
        model = RoiSSD(config=config['model_params'], num_classes=dataset_config['num_classes'])
    elif model_name == 'yolo':
        # yolo_weights = train_config.get('yolo_weights', train_config.get('ckpt_name', 'yolov8n.pt'))
        # model = YoloV8Adapter(weights_path=yolo_weights, device=device)
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                'YOLO inference requires ultralytics. Install with: pip install ultralytics'
            ) from e
        weights_path = train_config.get('yolo_weights', train_config.get('ckpt_name', 'yolov8n.pt'))
        print('yolo weights path from config: {}'.format(weights_path))
        if not os.path.exists(weights_path):
            task_name = train_config.get('task_name', '')
            candidate = os.path.join('trained_models', task_name, weights_path)
            if os.path.exists(candidate):
                weights_path = candidate
        model = YoloV8Adapter(weights_path=weights_path, device=device)
    else:
        raise Exception(f'Unknown model name {model_name!r}')

    model.to(device=device)
    model.eval()

    if model_name == 'ssd-original':
        print('Loaded SSD300_VGG16 pretrained weights from torchvision...')
        return model, dataset, data_loader, config
    if model_name == 'yolo':
        print(f"Loaded YOLO weights: {train_config.get('yolo_weights', train_config.get('ckpt_name', 'yolov8n.pt'))}")
        return model, dataset, data_loader, config

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


def convert_crop_to_input_tensor(
    im_tensor: Optional[torch.Tensor] = None,
    crop: Optional[List[int]] = None,
    original_image_size: Optional[Tuple[int, int]] = None,
    roi_grid: int = 1,
) -> Tuple[torch.Tensor, List[int]]:
    """Crop a pre-loaded full-frame tensor to a ROI.

    Returns ``(crop_tensor, snapped_frame_roi)`` where ``snapped_frame_roi`` is
    ``[x1, y1, x2, y2]`` in frame-pixel coordinates that correspond *exactly* to
    the returned tensor slice.  With ``roi_grid=1`` (default) the snapped ROI
    equals the input ``crop``.

    When ``roi_grid > 1`` (e.g. 32 for YOLO) the tensor-space crop dimensions are
    rounded up to the nearest multiple of ``roi_grid`` around the crop center.
    If any edge exceeds bounds, the entire crop window is shifted to remain
    inside image boundaries while preserving the snapped size.
    """

    tensor_chw = im_tensor[0] if im_tensor.dim() == 4 else im_tensor
    if tensor_chw.dim() != 3:
        raise ValueError(f"Expected im_tensor as CHW or BCHW, got shape {tuple(im_tensor.shape)}")

    orig_h, orig_w = int(original_image_size[0]), int(original_image_size[1])
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid original_image_size: {original_image_size}")

    _, tensor_h, tensor_w = tensor_chw.shape
    x1, y1, x2, y2 = [int(v) for v in crop]

    sx = tensor_w / float(orig_w)
    sy = tensor_h / float(orig_h)

    tx1 = max(0, min(tensor_w - 1, int(np.floor(x1 * sx))))
    ty1 = max(0, min(tensor_h - 1, int(np.floor(y1 * sy))))
    tx2 = max(tx1 + 1, min(tensor_w, int(np.ceil(x2 * sx))))
    ty2 = max(ty1 + 1, min(tensor_h, int(np.ceil(y2 * sy))))

    def _centered_snap(start: int, end: int, bound: int, grid: int) -> Tuple[int, int]:
        size = max(1, end - start)
        target = int(np.ceil(size / grid)) * grid if grid > 1 else size
        if target >= bound:
            return 0, bound

        center = 0.5 * (start + end)
        new_start = int(np.floor(center - target / 2.0))
        new_end = new_start + target

        # Preserve size and move the whole window back in-bounds.
        if new_start < 0:
            shift = -new_start
            new_start += shift
            new_end += shift
        if new_end > bound:
            shift = new_end - bound
            new_start -= shift
            new_end -= shift

        new_start = max(0, min(new_start, bound - target))
        new_end = new_start + target
        return new_start, new_end

    tx1, tx2 = _centered_snap(tx1, tx2, tensor_w, roi_grid)
    ty1, ty2 = _centered_snap(ty1, ty2, tensor_h, roi_grid)

    tensor_crop = tensor_chw[:, ty1:ty2, tx1:tx2]

    # Back-project to frame space and keep the full snapped window in-bounds.
    fx1 = int(np.floor(tx1 / sx))
    fy1 = int(np.floor(ty1 / sy))
    fx2 = int(np.ceil(tx2 / sx))
    fy2 = int(np.ceil(ty2 / sy))

    frame_w = max(1, fx2 - fx1)
    frame_h = max(1, fy2 - fy1)

    if fx1 < 0:
        shift = -fx1
        fx1 += shift
        fx2 += shift
    if fy1 < 0:
        shift = -fy1
        fy1 += shift
        fy2 += shift
    if fx2 > orig_w:
        shift = fx2 - orig_w
        fx1 -= shift
        fx2 -= shift
    if fy2 > orig_h:
        shift = fy2 - orig_h
        fy1 -= shift
        fy2 -= shift

    fx1 = max(0, min(fx1, max(0, orig_w - frame_w)))
    fy1 = max(0, min(fy1, max(0, orig_h - frame_h)))
    fx2 = min(orig_w, fx1 + frame_w)
    fy2 = min(orig_h, fy1 + frame_h)

    snapped_roi = [fx1, fy1, fx2, fy2]
    return tensor_crop, snapped_roi


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

# --------------------------------------------------------------------------- #
#  Ground-truth extraction for mAP (pixel coordinates + labels for mAP)      #
# --------------------------------------------------------------------------- #
def extract_gt_for_map(
    target: Dict[str, Any],
    idx2label: Dict[int, str],
    frame_w: int,
    frame_h: int,
) -> Tuple[Dict[str, List], Dict[str, List]]:
    """Return (gt_dict, difficult_dict) in compute_map format, pixel coords."""
    gt_dict: Dict[str, List] = {}
    diff_dict: Dict[str, List] = {}

    boxes_raw = labels_raw = None
    if "bboxes" in target:
        boxes_raw, labels_raw = target["bboxes"][0], target["labels"][0]
    elif "boxes" in target:
        boxes_raw, labels_raw = target["boxes"][0], target["labels"][0]
    else:
        return gt_dict, diff_dict

    if isinstance(boxes_raw, torch.Tensor):
        boxes_raw = boxes_raw.cpu().tolist()
    if isinstance(labels_raw, torch.Tensor):
        labels_raw = labels_raw.cpu().tolist()

    diff_raw = None
    if "difficult" in target:
        dr = target["difficult"][0]
        diff_raw = dr.cpu().tolist() if isinstance(dr, torch.Tensor) else list(dr)

    for i, (box, lbl) in enumerate(zip(boxes_raw, labels_raw)):
        x1 = float(box[0]) * frame_w
        y1 = float(box[1]) * frame_h
        x2 = float(box[2]) * frame_w
        y2 = float(box[3]) * frame_h
        cls = idx2label[int(lbl)]
        gt_dict.setdefault(cls, []).append([x1, y1, x2, y2])
        diff_dict.setdefault(cls, []).append(int(diff_raw[i]) if diff_raw else 0)

    return gt_dict, diff_dict

def extract_gt_for_tracker(
    target: Dict[str, Any],
    idx2label: Dict[int, str],
    frame_w: int,
    frame_h: int,
) -> List[Dict[str, Any]]:
    """Return GT as tracker-style detections in pixel coordinates."""
    gt_d, _ = extract_gt_for_map(target, idx2label, frame_w, frame_h)
    out: List[Dict[str, Any]] = []
    for cls, boxes in gt_d.items():
        for x1, y1, x2, y2 in boxes:
            clipped = clip_bbox([int(x1), int(y1), int(x2), int(y2)], frame_w, frame_h)
            if clipped is None:
                continue
            out.append(
                {
                    "bbox": [float(clipped[0]), float(clipped[1]), float(clipped[2]), float(clipped[3])],
                    "class": str(cls),
                    "confidence": 1.0,
                }
            )
    return out

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


def apply_tracker_input_dropout(
    tracker_input: List[Dict[str, Any]],
    frame_idx: int,
    tracker_input_dropout_cfg: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Apply configurable detection dropout before passing detections to tracker.

    Supported modes:
      - frame: drop all detections for a frame with probability `prob`
      - detection: drop each detection independently with probability `prob`
    """
    if not tracker_input_dropout_cfg:
        return tracker_input

    enabled = bool(tracker_input_dropout_cfg.get("enabled", False))
    if not enabled:
        return tracker_input

    prob = float(tracker_input_dropout_cfg.get("prob", 0.0))
    if prob <= 0.0:
        return tracker_input
    prob = min(max(prob, 0.0), 1.0)

    warmup_frames = max(0, int(tracker_input_dropout_cfg.get("warmup_frames", 0)))
    if frame_idx <= warmup_frames:
        return tracker_input

    mode = str(tracker_input_dropout_cfg.get("mode", "frame")).strip().lower()
    seed = tracker_input_dropout_cfg.get("seed", None)

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(int(seed) + int(frame_idx) * 1000003)

    if mode == "frame":
        return [] if float(rng.random()) < prob else tracker_input

    if mode == "detection":
        kept = [det for det in tracker_input if float(rng.random()) >= prob]
        return kept

    raise ValueError(f"Unknown tracker_input_dropout mode: {mode!r}")


# ---------------------------------------------------------------------------
# Per-frame result container
# ---------------------------------------------------------------------------
@dataclass
class FrameResult:
    final_detections:  List[Dict[str, Any]]  = field(default_factory=list)
    dropped_tracker_detections: List[Dict[str, Any]] = field(default_factory=list)
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
    tracker_input_dropout_cfg: Optional[Dict[str, Any]] = None,
    roi_grid: Optional[int] = None,   # None = auto-detect from model type
) -> FrameResult:
    """
    Run one frame through the full pipeline:
      1. Decide full-frame vs ROI mode.
      2. Run model inference (optionally per-ROI crop).
      3. NMS + confidence filter.
      4. Update tracker.
    Returns a FrameResult with detections, ROI metadata, and timing.
    """
    if merge_fn is None:
        merge_fn = MERGE_STRATEGIES["greedy"]
    if roi_grid is None:
        roi_grid = _model_roi_grid(model)

    frame_h, frame_w = frame_bgr.shape[:2]
    is_key_frame  = (frame_idx % key_frame_interval == 0)
    use_full_frame = (not next_frame_rois) or is_key_frame

    all_detections: List[Dict[str, Any]] = []
    rois_used: List[List[int]] = []
    merge_latency_s = 0.0

    t0 = time.perf_counter()

    if use_full_frame:
        _, model_batch_detections = run_model_inference(model, im_tensor.float().to(model_device))
        all_detections = tensor_to_detection_list(model_batch_detections[0], idx2label, frame_w, frame_h)
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
            crop_w = rx2 - rx1
            crop_h = ry2 - ry1
            if crop_w <= 0 or crop_h <= 0:
                continue
            crop_tensor, snapped_roi = convert_crop_to_input_tensor(
                im_tensor=im_tensor,
                crop=roi_c,
                original_image_size=(frame_h, frame_w),
                roi_grid=roi_grid,
            )
            sx1, sy1, sx2, sy2 = snapped_roi
            rois_used.append(snapped_roi)

            _, model_batch_detections = run_model_inference(model, crop_tensor.unsqueeze(0).to(model_device))
            all_detections.extend(
                tensor_to_detection_list(
                    model_batch_detections[0], idx2label, sx2 - sx1, sy2 - sy1,
                    offset_xy=(sx1, sy1),
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

    tracker_input_before_dropout = list(tracker_input)
    tracker_input = apply_tracker_input_dropout(
        tracker_input=tracker_input,
        frame_idx=frame_idx,
        tracker_input_dropout_cfg=tracker_input_dropout_cfg,
    )
    kept_ids = {id(det) for det in tracker_input}
    dropped_tracker_detections = [
        det for det in tracker_input_before_dropout if id(det) not in kept_ids
    ]

    tracker_result = tracker.update(tracker_input, frame_shape=(frame_h, frame_w))
    latency_s = time.perf_counter() - t0

    return FrameResult(
        final_detections=final_detections,
        dropped_tracker_detections=dropped_tracker_detections,
        rois_used=rois_used,
        next_frame_rois=[r['roi'] for r in tracker_result['rois']],
        use_full_frame=use_full_frame,
        latency_s=latency_s,
        merge_latency_s=merge_latency_s,
    )
