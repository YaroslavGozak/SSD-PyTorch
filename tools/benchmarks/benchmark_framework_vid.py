"""
Lane B — Video Sequence Benchmark.

Runs the same sequential inference pipeline as infer_sequentially_kalman_roi.py
but without visualization. Accumulates per-frame data and reports aggregate metrics:

Detection quality  : mAP@0.50, mAP@0.95
Speed              : fps, latency mean/p50/p95, split by full-frame vs ROI-only, merge time
ROI efficiency     : processed area ratio, ROI count pre/post merge
Coverage           : GT ROI coverage mean and p5

Usage:
    python -m tools.benchmarks.benchmark_framework_vid \
        --benchmark-config config/benchmark_vid.yaml
"""
import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from tools.helpers.pipeline import (
    MERGE_STRATEGIES,
    FrameResult,
    build_tracker,
    ensure_im_size_tuple,
    load_model_and_dataset,
    process_frame,
    extract_gt_for_map,
    extract_gt_for_tracker
)
from tools.infer import compute_map
from tools.helpers.config_reader import load_config


# --------------------------------------------------------------------------- #
#  Ground-truth extraction for mAP (pixel coordinates + labels for mAP)      #
# --------------------------------------------------------------------------- #

def _detections_to_map_pred(detections: List[Dict[str, Any]]) -> Dict[str, List]:
    """Convert detection list to compute_map prediction format (pixel coords)."""
    pred: Dict[str, List] = {}
    for det in detections:
        cls = str(det["class"])
        x1, y1, x2, y2 = det["bbox"]
        pred.setdefault(cls, []).append(
            [float(x1), float(y1), float(x2), float(y2), float(det["confidence"])]
        )
    return pred


def _gt_roi_coverage(
    rois: List[List[int]],
    gt_boxes: List[List[float]],
    threshold: float = 0.5,
) -> float:
    """Fraction of GT boxes where (intersection area / GT area) >= threshold."""
    if not gt_boxes:
        return float("nan")
    if not rois:
        return 0.0
    covered = 0
    for gx1, gy1, gx2, gy2 in gt_boxes:
        gt_area = max(0.0, (gx2 - gx1) * (gy2 - gy1))
        for rx1, ry1, rx2, ry2 in rois:
            ix1, iy1 = max(gx1, rx1), max(gy1, ry1)
            ix2, iy2 = min(gx2, rx2), min(gy2, ry2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if gt_area > 0 and inter / gt_area >= threshold:
                covered += 1
                break
    return covered / len(gt_boxes)


def _extract_sequence_meta(target: Dict[str, Any], fname: Any) -> Tuple[str, bool, Optional[int]]:
    """
    Returns (video_id, is_first_frame, frame_idx_in_video). Falls back to filename parent dir if
    metadata is unavailable.
    """
    default_path = fname[0] if isinstance(fname, (list, tuple)) else fname
    default_video_id = os.path.basename(os.path.dirname(str(default_path)))

    # Targets from default collate may wrap python scalars/strings in lists.
    video_id = target.get("video_id", default_video_id)
    if isinstance(video_id, list):
        video_id = video_id[0] if video_id else default_video_id

    is_first = target.get("is_first_frame", False)
    if isinstance(is_first, list):
        is_first = is_first[0] if is_first else False
    if isinstance(is_first, torch.Tensor):
        is_first = bool(is_first.item())

    frame_idx = target.get("frame_idx", None)
    if isinstance(frame_idx, list):
        frame_idx = frame_idx[0] if frame_idx else None
    if isinstance(frame_idx, torch.Tensor):
        frame_idx = int(frame_idx.item())
    if frame_idx is not None:
        frame_idx = int(frame_idx)

    return str(video_id), bool(is_first), frame_idx


# --------------------------------------------------------------------------- #
#  Main benchmark class                                                         #
# --------------------------------------------------------------------------- #
class VideoSequenceBenchmark:
    """
    Lane B benchmark: sequential inference without visualization.
    Accepts any tracker with update(detections, frame_shape) -> Dict interface.
    Config is the benchmark_vid YAML.
    """

    def __init__(self, benchmark_config_path: str, extra_run_metadata: Optional[Dict[str, Any]] = None):
        self.cfg = load_config(benchmark_config_path)
        self._extra_run_metadata = dict(extra_run_metadata or {})
        self._init_from_cfg()

    @classmethod
    def from_config_dict(
        cls,
        cfg: Dict[str, Any],
        extra_run_metadata: Optional[Dict[str, Any]] = None,
    ) -> "VideoSequenceBenchmark":
        """Construct benchmark from an already-loaded config dict."""
        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj._extra_run_metadata = dict(extra_run_metadata or {})
        obj._init_from_cfg()
        return obj

    def _init_from_cfg(self) -> None:
        """Initialize benchmark state from self.cfg."""

        p = self.cfg["benchmark_vid_params"]
        self.tracker = build_tracker(p["tracker"])
        inf = p["inference"]
        self.key_frame_interval = max(1, int(inf["key_frame_interval"]))
        self.nms_iou = float(inf["nms_iou"])
        roi_m = p["roi_merge"]
        self.merge_fn = MERGE_STRATEGIES[roi_m["strategy"]]
        self.merge_tau = float(roi_m.get("tau", 150000.0))
        self.tracker_input_dropout_cfg = p.get("tracker_input_dropout", None)
        self.coverage_threshold = float(p["metrics"]["roi_coverage_threshold"])
        out = p["output"]
        self.output_dir = out["results_dir"]
        self.results_filename = out["results_filename"]
        self.verbose = bool(out.get("verbose", True))

        tracker_cfg = p.get("tracker", {})
        kalman_cfg = tracker_cfg.get("kalman", {})
        static_padding_cfg = tracker_cfg.get("static_padding", {})
        relative_padding_cfg = tracker_cfg.get("relative_padding", {})
        oracle_gt_cfg = tracker_cfg.get("oracle_gt", {})
        dropout_cfg = p.get("tracker_input_dropout", {})

        self.run_metadata = {
            "benchmark_device": str(p.get("device", "cpu")),
            "key_frame_interval": self.key_frame_interval,
            "benchmark_tracker_type": "none" if self.key_frame_interval == 1 else str(tracker_cfg.get("type", "unknown")),
            "merge_fn": str(roi_m.get("strategy", "unknown")),
            "merge_tau": self.merge_tau,
            "kalman_pmin": int(kalman_cfg.get("pmin", 0)),
            "static_pad_x": int(static_padding_cfg.get("pad_x", 0)),
            "static_pad_y": int(static_padding_cfg.get("pad_y", 0)),
            "relative_pad_ratio_x": float(relative_padding_cfg.get("pad_ratio_x", 0.0)),
            "relative_pad_ratio_y": float(relative_padding_cfg.get("pad_ratio_y", 0.0)),
            "relative_min_pad_x": int(relative_padding_cfg.get("min_pad_x", 0)),
            "relative_min_pad_y": int(relative_padding_cfg.get("min_pad_y", 0)),
            "oracle_gt_pad_x": int(oracle_gt_cfg.get("pad_x", 0)),
            "oracle_gt_pad_y": int(oracle_gt_cfg.get("pad_y", 0)),
            "static_hold_last_for_frames": int(static_padding_cfg.get("hold_last_for_frames", 0)),
            "tracker_dropout_enabled": bool(dropout_cfg.get("enabled", False)),
            "tracker_dropout_mode": str(dropout_cfg.get("mode", "none")),
            "tracker_dropout_prob": float(dropout_cfg.get("prob", 0.0)),
            "tracker_dropout_warmup_frames": int(dropout_cfg.get("warmup_frames", 0)),
            "tracker_dropout_seed": dropout_cfg.get("seed", ""),
            **self._extra_run_metadata,
        }

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Build an args namespace that load_model_and_dataset expects
        self._train_config_path = self.cfg["train_config_path"]

    def run(self) -> Dict[str, Any]:
        run_start_time = time.perf_counter()
        args = argparse.Namespace(config_path=self._train_config_path)
        model, dataset, data_loader, train_cfg = load_model_and_dataset(self.cfg["benchmark_vid_params"]["device"], args)

        conf_threshold = train_cfg["train_params"]["infer_conf_threshold"]
        model.low_score_threshold = conf_threshold

        im_size_hw = ensure_im_size_tuple(train_cfg["dataset_params"]["im_size"])
        model_device = next(model.parameters()).device
        total_frames = len(data_loader)

        self.tracker.reset()
        next_frame_rois: List[List[int]] = []
        current_video_id: Optional[str] = None

        predictions, ground_truths, difficulties = [], [], []
        lat_full, lat_roi, lat_merge = [], [], []
        area_ratios, roi_counts_pre, roi_counts_post, gt_coverages = [], [], [], []

        if self.verbose:
            print(f"Total frames: {total_frames}  |  key_frame_interval: {self.key_frame_interval}")

        frame_idx = 0
        with torch.no_grad():
            for im_tensor, target, fname in data_loader:
                frame_idx += 1
                if self.verbose and frame_idx % max(1, total_frames // 10) == 0:
                    pct = frame_idx / total_frames * 100
                    print(f"  {frame_idx}/{total_frames} ({pct:.0f}%)")

                fpath = os.path.abspath(fname[0] if isinstance(fname, (list, tuple)) else fname)
                frame_bgr = cv2.imread(fpath)
                if frame_bgr is None:
                    continue
                frame_h, frame_w = frame_bgr.shape[:2]
                frame_area = frame_w * frame_h

                tgt = target[0] if isinstance(target, list) else target
                video_id, is_first_frame, frame_idx_in_video = _extract_sequence_meta(tgt, fname)
                if current_video_id is None:
                    current_video_id = video_id

                # Hard reset on video boundary to prevent state leakage.
                if is_first_frame or video_id != current_video_id:
                    self.tracker.reset()
                    next_frame_rois = []
                    current_video_id = video_id

                # Oracle detector mode: use GT detections for ROI generation instead of tracker output.
                tracker_type = str(self.cfg["benchmark_vid_params"]["tracker"]["type"])
                if tracker_type == "oracle_gt":
                    oracle_dets = extract_gt_for_tracker(tgt, dataset.idx2label, frame_w, frame_h)
                    if hasattr(self.tracker, "set_oracle_detections"):
                        self.tracker.set_oracle_detections(oracle_dets)
                    if hasattr(self.tracker, "preview_rois"):
                        next_frame_rois = self.tracker.preview_rois((frame_h, frame_w))

                effective_frame_idx = frame_idx_in_video if frame_idx_in_video is not None else frame_idx

                result: FrameResult = process_frame(
                    model=model,
                    idx2label=dataset.idx2label,
                    frame_bgr=frame_bgr,
                    im_tensor=im_tensor,
                    tracker=self.tracker,
                    next_frame_rois=next_frame_rois,
                    frame_idx=effective_frame_idx,
                    key_frame_interval=self.key_frame_interval,
                    im_size_hw=im_size_hw,
                    conf_threshold=conf_threshold,
                    nms_iou=self.nms_iou,
                    merge_fn=self.merge_fn,
                    merge_tau=self.merge_tau,
                    model_device=model_device,
                    tracker_input_dropout_cfg=self.tracker_input_dropout_cfg,
                )
                next_frame_rois = result.next_frame_rois
                final_dets = result.final_detections
                rois_used = result.rois_used

                # Accumulate timing into separate buckets for reporting
                if result.use_full_frame:
                    lat_full.append(result.latency_s)
                else:
                    lat_roi.append(result.latency_s)
                    lat_merge.append(result.merge_latency_s)
                    roi_counts_pre.append(len(next_frame_rois) + len(rois_used))  # pre-merge estimate
                    roi_counts_post.append(len(rois_used))

                # Accumulate predictions / GT for mAP
                predictions.append(_detections_to_map_pred(final_dets))
                gt_d, diff_d = extract_gt_for_map(tgt, dataset.idx2label, frame_w, frame_h)
                ground_truths.append(gt_d)
                difficulties.append(diff_d)

                # Processed area ratio
                if result.use_full_frame:
                    area_ratios.append(1.0)
                elif rois_used:
                    roi_area = sum(max(0, r[2]-r[0]) * max(0, r[3]-r[1]) for r in rois_used)
                    area_ratios.append(roi_area / frame_area)

                # GT ROI coverage
                all_gt = [b for boxes in gt_d.values() for b in boxes]
                search_rois = rois_used if not result.use_full_frame else [[0, 0, frame_w - 1, frame_h - 1]]
                gt_coverages.append(_gt_roi_coverage(search_rois, all_gt, self.coverage_threshold))

        metrics = self._compute(
            predictions, ground_truths, difficulties,
            lat_full, lat_roi, lat_merge,
            area_ratios, roi_counts_pre, roi_counts_post, gt_coverages,
            frame_idx, train_cfg
        )
        self._print(metrics, elapsed_s=time.perf_counter() - run_start_time)
        self._save(metrics)
        return metrics

    # ------------------------------------------------------------------ #
    def _compute(
        self, predictions, ground_truths, difficulties,
        lat_full, lat_roi, lat_merge,
        area_ratios, roi_counts_pre, roi_counts_post, gt_coverages,
        n_frames, cfg,
        detector_recall50=float('nan'), class_recalls50=None,
        detector_recall95=float('nan'), class_recalls95=None,
    ) -> Dict[str, Any]:
        if self.verbose:
            print("\nComputing metrics...")

        train_params = cfg["train_params"]
        model_family = str(train_params["model"])
        task_name = str(train_params.get("task_name", ""))
        checkpoint_name = str(train_params.get("ckpt_name", ""))
        yolo_weights = str(train_params.get("yolo_weights", ""))
        model_label = model_family
        if model_family == "yolo" and task_name:
            model_label = task_name

        mAP50, aps50, detector_recall50, class_recalls50 = compute_map(predictions, ground_truths, iou_threshold=0.5,  difficult=difficulties)
        mAP95, aps95, detector_recall95, class_recalls95 = compute_map(predictions, ground_truths, iou_threshold=0.95, difficult=difficulties)

        def ms(arr): return np.array(arr) * 1000 if arr else np.array([float("nan")])
        def smean(a): return float(np.nanmean(a)) if len(a) else float("nan")
        def sperc(a, p): return float(np.nanpercentile(a, p)) if len(a) else float("nan")

        all_lat_ms = ms(lat_full + lat_roi)
        return {
            "dataset":   cfg["train_params"]["dataset"],
            "model":     model_label,
            "model_family": model_family,
            "model_task_name": task_name,
            "model_checkpoint": checkpoint_name,
            "model_yolo_weights": yolo_weights,
            "num_frames": n_frames,
            "key_frame_interval": self.key_frame_interval,
            "tracker_type": self.cfg["benchmark_vid_params"]["tracker"]["type"],
            # Detection quality
            "mAP50": float(mAP50),
            "mAP95": float(mAP95),
            "per_class_ap50": {k: float(v) for k, v in aps50.items()},
            "per_class_ap95": {k: float(v) for k, v in aps95.items()},
            "detector_recall50": float(detector_recall50),
            "detector_recall95": float(detector_recall95),
            "per_class_detector_recall50": {k: float(v) for k, v in class_recalls50.items()},
            "per_class_detector_recall95": {k: float(v) for k, v in class_recalls95.items()},
            # Speed
            "fps_total":               float(len(lat_full + lat_roi) / max(sum(lat_full + lat_roi), 1e-9)),
            "latency_mean_ms":         smean(all_lat_ms),
            "latency_p50_ms":          sperc(all_lat_ms, 50),
            "latency_p95_ms":          sperc(all_lat_ms, 95),
            "latency_full_frame_mean_ms": smean(ms(lat_full)),
            "latency_roi_mean_ms":     smean(ms(lat_roi)),
            "merge_latency_mean_ms":   smean(ms(lat_merge)),
            # ROI efficiency
            "full_frame_fraction":      len(lat_full) / max(n_frames, 1),
            "processed_area_ratio_mean": smean(np.array(area_ratios)),
            "processed_area_ratio_p95":  sperc(np.array(area_ratios), 95),
            "roi_count_pre_merge_mean":  smean(np.array(roi_counts_pre))  if roi_counts_pre  else float("nan"),
            "roi_count_post_merge_mean": smean(np.array(roi_counts_post)) if roi_counts_post else float("nan"),
            # Coverage
            "gt_roi_coverage_mean": smean(np.array(gt_coverages)),
            "gt_roi_coverage_p5":   sperc(np.array(gt_coverages), 5),
        }

    def _print(self, m: Dict[str, Any], elapsed_s: float) -> None:
        print("\n" + "=" * 70)
        print("VIDEO BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Dataset: {m['dataset']}  Model: {m['model']}  Tracker: {m['tracker_type']}")
        if m.get("model_checkpoint"):
            print(f"Checkpoint: {m['model_checkpoint']}")
        if m.get("model_yolo_weights"):
            print(f"YOLO weights: {m['model_yolo_weights']}")
        print(f"Frames: {m['num_frames']}  Key-frame interval: {m['key_frame_interval']}")
        print(f"Elapsed total time: {elapsed_s:.1f} s")
        print(f"\n{'Detection Quality':─<35}")
        print(f"  mAP@0.50          : {m['mAP50']:.4f}")
        print(f"  mAP@0.95          : {m['mAP95']:.4f}")
        print(f"  detector_recall@50: {m['detector_recall50']:.4f}")
        print(f"  detector_recall@95: {m['detector_recall95']:.4f}")
        print(f"\n{'Speed':─<35}")
        print(f"  FPS (total)               : {m['fps_total']:.2f}")
        print(f"  Latency mean / p50 / p95  : {m['latency_mean_ms']:.1f} / {m['latency_p50_ms']:.1f} / {m['latency_p95_ms']:.1f} ms")
        print(f"  Full-frame latency (mean) : {m['latency_full_frame_mean_ms']:.1f} ms")
        print(f"  ROI-only   latency (mean) : {m['latency_roi_mean_ms']:.1f} ms")
        print(f"  ROI merge  latency (mean) : {m['merge_latency_mean_ms']:.2f} ms")
        print(f"\n{'ROI Efficiency':─<35}")
        print(f"  Full-frame fraction       : {m['full_frame_fraction']:.3f}")
        print(f"  Processed area ratio mean : {m['processed_area_ratio_mean']:.3f}  (p95={m['processed_area_ratio_p95']:.3f})")
        print(f"  ROI count pre-merge (mean): {m['roi_count_pre_merge_mean']:.1f}")
        print(f"  ROI count post-merge(mean): {m['roi_count_post_merge_mean']:.1f}")
        print(f"  GT ROI coverage mean      : {m['gt_roi_coverage_mean']:.3f}  (p5={m['gt_roi_coverage_p5']:.3f})")
        print("=" * 70)

    def _save(self, m: Dict[str, Any]) -> None:
        path = os.path.join(self.output_dir, self.results_filename)

        flat: Dict[str, Any] = {}
        for k, v in m.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}_{kk}"] = vv
            else:
                flat[k] = v

        row = {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **self.run_metadata,
            **flat,
        }
        fieldnames = self._order_csv_fieldnames(row)

        output_path, output_fieldnames, migrated = self._resolve_output_path(path, fieldnames)
        need_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0

        if not need_header and self._file_missing_trailing_newline(output_path):
            with open(output_path, "a", encoding="utf-8") as f:
                f.write("\n")

        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            if need_header:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in output_fieldnames})

        if migrated:
            print(f"\nSchema changed. Appending to migrated CSV: {output_path}")
        else:
            print(f"\nResults appended to: {output_path}")

    @staticmethod
    def _order_csv_fieldnames(row: Dict[str, Any]) -> List[str]:
        """Keep class-wise metric columns grouped at the end of the CSV."""
        classwise_prefixes = (
            "per_class_ap50_",
            "per_class_ap95_",
            "per_class_detector_recall50_",
            "per_class_detector_recall95_",
        )

        non_class_fields = sorted(
            key for key in row.keys()
            if not any(key.startswith(prefix) for prefix in classwise_prefixes)
        )
        class_fields = sorted(
            key for key in row.keys()
            if any(key.startswith(prefix) for prefix in classwise_prefixes)
        )
        return non_class_fields + class_fields

    def _resolve_output_path(self, preferred_path: str, new_fieldnames: List[str]) -> Tuple[str, List[str], bool]:
        """Find a CSV target path that matches schema, auto-migrating to _vN on mismatch."""
        if not os.path.exists(preferred_path) or os.path.getsize(preferred_path) == 0:
            return preferred_path, new_fieldnames, False

        existing_header = self._read_header(preferred_path)
        if existing_header == new_fieldnames:
            return preferred_path, existing_header, False

        base, ext = os.path.splitext(preferred_path)
        version = 2
        while True:
            candidate = f"{base}_v{version}{ext}"
            if not os.path.exists(candidate) or os.path.getsize(candidate) == 0:
                return candidate, new_fieldnames, True

            candidate_header = self._read_header(candidate)
            if candidate_header == new_fieldnames:
                return candidate, candidate_header, True

            version += 1

    @staticmethod
    def _read_header(path: str) -> List[str]:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        return header or []

    @staticmethod
    def _file_missing_trailing_newline(path: str) -> bool:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False

        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last_byte = f.read(1)
        return last_byte not in (b"\n", b"\r")


# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Lane B Video Sequence Benchmark")
    parser.add_argument("--benchmark-config", default="config/benchmark-vid.yaml",
                        help="Path to benchmark_vid YAML configuration")
    args = parser.parse_args()
    bench = VideoSequenceBenchmark(args.benchmark_config)
    bench.run()


if __name__ == "__main__":
    main()