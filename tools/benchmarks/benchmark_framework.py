"""
Benchmark Framework for SSD/RoiSSD inference evaluation.

Loads a model and dataset, runs inference, and measures:
- mAP (mean Average Precision)
- Recall
- FPS (frames per second)
- Latency (mean, p50, p95, p99)
- Per-class metrics

Results are saved to CSV and printed to console.
"""

import argparse
import csv
import os
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset.visdrone import VisDroneDataset
from dataset.voc import VOCDataset
from dataset.voc_small_objects import VOCSmallObjectsDataset
from dataset.ytbb import YTBBDataset
from model.roissd import RoiSSD
from model.ssd import SSD
from model.ssd_mobilenet import SSDMobileNet
from tools.infer import compute_map
from config.default import default_model_config_params

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class YoloV8Adapter:
    """Adapter that returns detections in the same list-of-dicts shape as SSD models."""

    def __init__(self, weights_path: str, device: torch.device):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLO benchmark requires ultralytics. Install with: pip install ultralytics"
            ) from e

        self.device = device
        self._yolo = YOLO(weights_path)

    def to(self, device: torch.device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, images: torch.Tensor, _targets=None):
        # Ultralytics returns pixel-space xyxy boxes. Convert to normalized [0,1]
        # so downstream metric code can stay model-agnostic.
        results = self._yolo.predict(source=images, verbose=False)
        b, _, h, w = images.shape
        detections = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                detections.append({
                    'boxes': torch.empty((0, 4), dtype=torch.float32, device=images.device),
                    'labels': torch.empty((0,), dtype=torch.int64, device=images.device),
                    'scores': torch.empty((0,), dtype=torch.float32, device=images.device),
                })
                continue

            boxes_xyxy = res.boxes.xyxy.to(images.device).float()
            boxes_norm = boxes_xyxy.clone()
            boxes_norm[:, [0, 2]] /= float(w)
            boxes_norm[:, [1, 3]] /= float(h)

            detections.append({
                'boxes': boxes_norm,
                # Project dataset labels are 1-based while YOLO classes are 0-based.
                'labels': (res.boxes.cls.to(images.device).long() + 1),
                'scores': res.boxes.conf.to(images.device).float(),
            })

        return None, detections


class BenchmarkFramework:
    def __init__(self, config_path: str, output_dir: Optional[str] = None):
        """
        Initialize the benchmark framework.
        
        Args:
            config_path: Path to benchmark config YAML
            output_dir: Optional override for output directory
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.idx2label = None
        
        # Metrics storage
        self.latencies = []
        self.predictions = []
        self.ground_truths = []
        self.difficulties = []
        self.per_class_metrics = {}
        
        # Setup output directory
        self.output_dir = output_dir or self.config['benchmark_params']['output']['results_dir']
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"Error loading config: {e}")
                raise
    
    def _setup_device(self) -> torch.device:
        """Setup compute device (CUDA, MPS, CPU)."""
        device_config = self.config.get('device', {})
        device_type = device_config.get('type', 'cuda')
        
        if device_type == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif device_type == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        
        return device
    
    def load_model_and_dataset(self) -> None:
        """Load model and dataset from config."""
        bench_params = self.config['benchmark_params']
        model_config = bench_params['model']
        dataset_config = bench_params['dataset']
        
        # Parse im_size (can be int or [h, w])
        im_size_val = dataset_config['im_size']
        if isinstance(im_size_val, int):
            im_size = im_size_val
        elif isinstance(im_size_val, (list, tuple)):
            im_size = im_size_val[0]
        else:
            raise ValueError(f"Invalid im_size format: {im_size_val}")
        
        # Load dataset
        dataset_name = dataset_config['name']
        
        if dataset_name == 'vis-drone':
            self.dataset = VisDroneDataset('test', im_size=im_size)
        elif dataset_name == 'ytbb':
            self.dataset = YTBBDataset('test', im_size=im_size)
        elif dataset_name == 'voc':
            test_im_sets = dataset_config.get('test_im_sets', ['test'])
            transform_name = dataset_config.get('transform_name', 'ssd')  # Default to 'ssd' transform
            self.dataset = VOCDataset('test', im_sets=test_im_sets, im_size=im_size, transform_name=transform_name)
        elif dataset_name == 'voc-small-objects':
            test_im_sets = dataset_config.get('test_im_sets', ['test'])
            transform_name = dataset_config.get('transform_name', 'ssd')  # Default to 'ssd' transform
            self.dataset = VOCSmallObjectsDataset('test', im_sets=test_im_sets, im_size=im_size, transform_name=transform_name)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.idx2label = self.dataset.idx2label
        
        from torch.utils.data import DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=bench_params['inference']['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        print(f"Loaded {dataset_name} dataset: {len(self.dataset)} images")
        
        # Load model
        model_name = model_config['name']
        checkpoint_path = model_config['checkpoint_path']
        num_classes = len(self.idx2label)
        
        # Try to load training config for model parameters
        try:
            # Look for model config in any existing training config
            training_config = self._find_training_config()
            model_config_params = training_config.get('model_params', {})
        except:
            model_config_params = {}
        
        if not model_config_params:
            # Provide defaults if not found
            model_config_params = default_model_config_params
        
        if model_name == 'ssd':
            self.model = SSD(config=model_config_params, num_classes=num_classes)
        elif model_name == 'roissd':
            self.model = RoiSSD(config=model_config_params, num_classes=num_classes)
        elif model_name == 'ssd_mobilenet':
            self.model = SSDMobileNet(config=model_config_params, num_classes=num_classes)
        elif model_name == 'yolo':
            self.model = YoloV8Adapter(weights_path=checkpoint_path, device=self.device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint
        if model_name == 'yolo':
            print(f"Loaded YOLO weights: {checkpoint_path}")
        elif os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                print(f"Loaded model from full checkpoint format: {checkpoint_path}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Loaded model weights: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

    
    def _find_training_config(self) -> Dict[str, Any]:
        """Search for a training config with model parameters."""
        config_dir = Path('config')
        for config_file in config_dir.glob('*.yaml'):
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
                if cfg and 'model_params' in cfg:
                    return cfg
        return {}
    
    def run_inference(self) -> None:
        """Run inference on the entire dataset and collect metrics."""
        bench_params = self.config['benchmark_params']
        inference_config = bench_params['inference']
        
        conf_threshold = inference_config['confidence_threshold']
        if hasattr(self.model, 'low_score_threshold'):
            self.model.low_score_threshold = conf_threshold
        
        total_images = len(self.dataset)
        verbose = bench_params['output']['verbose']
        
        # Get im_size for denormalization
        im_size_val = bench_params['dataset']['im_size']
        if isinstance(im_size_val, int):
            im_size = (im_size_val, im_size_val)
        else:
            im_size = tuple(im_size_val)
        
        print(f"\nRunning inference on {total_images} images...")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"NMS IoU threshold: {inference_config['nms_iou']}")
        
        with torch.no_grad():
            for batch_idx, (images, targets, fnames) in enumerate(self.dataloader):
                if verbose and batch_idx % max(1, total_images // 20) == 0:
                    progress = (batch_idx * len(images)) / total_images * 100
                    print(f"  Progress: {progress:.1f}% ({batch_idx * len(images)}/{total_images})")
                
                images = images.float().to(self.device)
                
                # Measure latency for this batch
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_start = time.perf_counter()
                
                _, detections = self.model(images, None)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_end = time.perf_counter()
                
                batch_latency = (t_end - t_start) / len(images)  # Per-image latency
                self.latencies.append(batch_latency)

                if not isinstance(targets, (list)):
                    targets = [targets]
                
                # Process detections and ground truth for each image in batch
                for img_idx, (im_detections, target) in enumerate(zip(detections, targets)):
                    # Convert detections to VOC format
                    pred_dict = self._tensor_detections_to_dict(
                        im_detections, 
                        self.dataset.idx2label,
                        im_size
                    )
                    self.predictions.append(pred_dict)
                    
                    # Extract ground truth
                    gt_dict, difficult = self._extract_ground_truth(target)
                    self.ground_truths.append(gt_dict)
                    self.difficulties.append(difficult)
        
        print(f"Completed inference on {total_images} images")
    
    def _tensor_detections_to_dict(
        self,
        detections: Dict[str, torch.Tensor],
        idx2label: Dict[int, str],
        im_size: Tuple[int, int]
    ) -> Dict[str, List[List[float]]]:
        """Convert tensor detections to VOC format dict."""
        result = {}
        
        if detections is None or not detections:
            return result
        
        boxes = detections.get('boxes', [])
        labels = detections.get('labels', [])
        scores = detections.get('scores', [])
        
        if len(boxes) == 0:
            return result
        
        # Get image dimensions (normalized coordinates are [0,1])
        boxes_np = boxes.cpu().numpy()
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        for box, label, score in zip(boxes_np, labels_np, scores_np):
            x1, y1, x2, y2 = box
            label_name = idx2label[int(label)]
            
            if label_name not in result:
                result[label_name] = []
            
            result[label_name].append([x1, y1, x2, y2, float(score)])
        
        return result
    
    def _extract_ground_truth(self, target: Dict[str, Any]) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[int]]]:
        """Extract ground truth boxes from target dict."""
        gt_dict = {}
        difficult_dict = {}
        
        if 'bboxes' in target:
            boxes = target['bboxes'][0].cpu().numpy() if isinstance(target['bboxes'][0], torch.Tensor) else target['bboxes'][0]
            labels = target['labels'][0].cpu().numpy() if isinstance(target['labels'][0], torch.Tensor) else target['labels'][0]
        else:
            return gt_dict, difficult_dict
        
        for box, label in zip(boxes, labels):
            label_name = self.idx2label[int(label)]
            if label_name not in gt_dict:
                gt_dict[label_name] = []
                difficult_dict[label_name] = []
            
            x1, y1, x2, y2 = box
            gt_dict[label_name].append([x1, y1, x2, y2])
            
            # Assume no difficult boxes by default
            difficult_dict[label_name].append(0)
        
        return gt_dict, difficult_dict

    @staticmethod
    def _iou_xyxy(box_a: List[float], box_b: List[float]) -> float:
        """Compute IoU for two boxes in [x1, y1, x2, y2] format."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    def _compute_global_recall(self, iou_threshold: float) -> Tuple[float, int, int]:
        """
        Compute class-aware, one-to-one matched recall over the full dataset.
        Recall = TP / (num_non_difficult_gt).
        """
        tp = 0
        total_gt = 0

        for preds_img, gts_img, diffs_img in zip(self.predictions, self.ground_truths, self.difficulties):
            labels = set(gts_img.keys()) | set(preds_img.keys())

            for label in labels:
                gt_boxes = gts_img.get(label, [])
                pred_boxes = preds_img.get(label, [])
                difficult_flags = diffs_img.get(label, [0] * len(gt_boxes))

                if len(difficult_flags) < len(gt_boxes):
                    difficult_flags = difficult_flags + [0] * (len(gt_boxes) - len(difficult_flags))

                valid_gt_idx = [i for i, d in enumerate(difficult_flags[:len(gt_boxes)]) if not d]
                total_gt += len(valid_gt_idx)

                if not valid_gt_idx or not pred_boxes:
                    continue

                matched = [False] * len(gt_boxes)
                pred_boxes_sorted = sorted(pred_boxes, key=lambda b: -float(b[4]))

                for pred in pred_boxes_sorted:
                    best_iou = -1.0
                    best_gt = -1
                    for gi in valid_gt_idx:
                        if matched[gi]:
                            continue
                        iou = self._iou_xyxy(pred[:4], gt_boxes[gi])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = gi

                    if best_gt >= 0 and best_iou >= iou_threshold:
                        matched[best_gt] = True
                        tp += 1

        recall = float(tp / max(total_gt, 1))
        return recall, tp, total_gt
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all metrics (mAP, Recall, FPS, latency)."""
        metrics = {}
        iou_threshold = float(self.config['benchmark_params']['inference']['iou_threshold'])
        
        # 1. mAP and per-class AP
        print("\nComputing mAP and per-class metrics...")
        mean_ap, all_aps, detector_recall, class_recalls = compute_map(
            self.predictions,
            self.ground_truths,
            iou_threshold=iou_threshold,
            difficult=self.difficulties
        )

        mean_ap_95, all_aps_95, detector_recall_95, class_recalls_95 = compute_map(
            self.predictions,
            self.ground_truths,
            iou_threshold=0.95,
            difficult=self.difficulties
        )

        metrics['mAP'] = float(mean_ap)
        metrics['mAP95'] = float(mean_ap_95)
        metrics['map_iou_threshold'] = iou_threshold
        metrics['per_class_ap'] = {k: float(v) for k, v in all_aps.items()}
        metrics['per_class_ap95'] = {k: float(v) for k, v in all_aps_95.items()}
        metrics['detector_recall'] = float(detector_recall)
        metrics['detector_recall95'] = float(detector_recall_95)
        metrics['per_class_detector_recall'] = {k: float(v) for k, v in class_recalls.items()}
        metrics['per_class_detector_recall95'] = {k: float(v) for k, v in class_recalls_95.items()}
        
        # 2. Recall (class-aware matched recall, bounded to [0, 1])
        recall, matched_tp, total_gt = self._compute_global_recall(iou_threshold=iou_threshold)
        total_detections = sum(len(det_list) for preds in self.predictions for det_list in preds.values())
        metrics['recall'] = float(recall)
        metrics['recall_iou_threshold'] = iou_threshold
        metrics['matched_true_positives'] = int(matched_tp)
        metrics['total_gt_boxes'] = int(total_gt)
        metrics['total_detections'] = int(total_detections)
        
        # 3. Latency statistics
        latencies_ms = np.array(self.latencies) * 1000  # Convert to ms
        metrics['fps'] = float(1.0 / np.mean(self.latencies))
        metrics['latency_mean_ms'] = float(np.mean(latencies_ms))
        metrics['latency_p50_ms'] = float(np.percentile(latencies_ms, 50))
        metrics['latency_p95_ms'] = float(np.percentile(latencies_ms, 95))
        metrics['latency_p99_ms'] = float(np.percentile(latencies_ms, 99))
        metrics['latency_min_ms'] = float(np.min(latencies_ms))
        metrics['latency_max_ms'] = float(np.max(latencies_ms))
        
        # 4. Dataset and model info
        metrics['dataset'] = self.config['benchmark_params']['dataset']['name']
        metrics['model'] = self.config['benchmark_params']['model']['name']
        metrics['num_images'] = len(self.dataset)
        metrics['im_size'] = str(self.config['benchmark_params']['dataset']['im_size'])
        metrics['device'] = str(self.device)
        
        return metrics
    
    def save_results(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to CSV and print summary."""
        output_file = os.path.join(self.output_dir, self.config['benchmark_params']['output']['results_filename'])
        
        # Flatten metrics for CSV (handle nested dicts)
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    flat_metrics[f"{key}_{k}"] = v
            else:
                flat_metrics[key] = value
        
        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(flat_metrics.keys()))
            writer.writeheader()
            writer.writerow(flat_metrics)
        
        print(f"\nResults saved to: {output_file}")
        
        # Print summary to console
        self._print_summary(metrics)
    
    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print a formatted summary of metrics."""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        print(f"\nDataset: {metrics['dataset']}")
        print(f"Model: {metrics['model']}")
        print(f"Image Size: {metrics['im_size']}")
        print(f"Device: {metrics['device']}")
        print(f"Num Images: {metrics['num_images']}")
        
        print("\n" + "-"*70)
        print("DETECTION QUALITY METRICS")
        print("-"*70)
        print(f"mAP @ IoU={metrics['map_iou_threshold']:.2f}: {metrics['mAP']:.4f}")
        print(f"mAP @ IoU=0.95: {metrics['mAP95']:.4f}")
        print(
            f"Recall @ IoU={metrics['recall_iou_threshold']:.2f}: {metrics['recall']:.4f} "
            f"(TP={metrics['matched_true_positives']}, GT={metrics['total_gt_boxes']}, Dets={metrics['total_detections']})"
        )
        
        print("\nPer-class AP:")
        for cls_name, ap in sorted(metrics['per_class_ap'].items()):
            if not np.isnan(ap):
                print(f"  {cls_name:20s}: {ap:.4f}")
        
        print("\n" + "-"*70)
        print("PERFORMANCE METRICS")
        print("-"*70)
        print(f"FPS: {metrics['fps']:.2f}")
        print(f"Latency (mean): {metrics['latency_mean_ms']:.3f} ms")
        print(f"Latency (p50): {metrics['latency_p50_ms']:.3f} ms")
        print(f"Latency (p95): {metrics['latency_p95_ms']:.3f} ms")
        print(f"Latency (p99): {metrics['latency_p99_ms']:.3f} ms")
        print(f"Latency (min/max): {metrics['latency_min_ms']:.3f} / {metrics['latency_max_ms']:.3f} ms")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='SSD/RoiSSD Benchmark Framework')
    parser.add_argument('--benchmark-config', type=str, default='config/benchmark.yaml',
                        help='Path to benchmark configuration YAML')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory for results')
    args = parser.parse_args()
    print('Starting...')
    
    # Initialize and run benchmark
    benchmark = BenchmarkFramework(args.benchmark_config, args.output_dir)
    benchmark.load_model_and_dataset()
    benchmark.run_inference()
    metrics = benchmark.compute_metrics()
    benchmark.save_results(metrics)


if __name__ == '__main__':
    main()
