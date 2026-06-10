"""
Async 30 FPS video runner for Lane B.

This module introduces a real-time style runtime where frames are ingested on a
fixed clock (default 30 FPS) while model processing runs in a worker thread.
When the worker cannot keep up, the producer overwrites pending work so the
worker processes the latest frame.

Modes:
  - interactive: visualize current frame with latest available predictions.
  - benchmark: compute metrics on every frame. If a frame has no fresh
    prediction, hold-last predictions are used.

Usage:
  python -m tools.benchmarks.benchmark_async_vid --benchmark-config config/benchmark-vid-base.yaml --mode benchmark
  python -m tools.benchmarks.benchmark_async_vid --benchmark-config config/benchmark-vid-base.yaml --mode interactive
  python -m tools.benchmarks.benchmark_async_vid --benchmark-config config/benchmark-vid-base.yaml --sweep-config config/benchmark-vid-sweep-example.yaml
"""

import argparse
import copy
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from tools.benchmarks.benchmark_framework_vid import (
    VideoSequenceBenchmark,
    _detections_to_map_pred,
    _extract_sequence_meta,
    _gt_roi_coverage,
)
from tools.helpers.config_reader import load_config
from tools.helpers.pipeline import (
    FrameResult,
    ensure_im_size_tuple,
    extract_gt_for_map,
    extract_gt_for_tracker,
    load_model_and_dataset,
    process_frame,
)

SUPPORTED_TRACKERS = {"static_padding", "relative_padding", "kalman", "oracle_gt", "sort"}
SUPPORTED_ROI_MERGE_STRATEGIES = {"greedy", "simple", "simple_v2", "none"}


@dataclass
class FrameTask:
    frame_idx: int
    effective_frame_idx: int
    frame_bgr: np.ndarray
    im_tensor: torch.Tensor
    target: Dict[str, Any]
    video_id: str
    is_first_frame: bool
    frame_w: int
    frame_h: int


@dataclass
class ProcessedFrame:
    frame_idx: int
    effective_frame_idx: int
    frame_result: FrameResult
    frame_w: int
    frame_h: int
    started_at: float
    completed_at: float


class LatestFrameSlot:
    """Single-slot handoff with overwrite semantics for latest-frame processing."""

    def __init__(self):
        self._cond = threading.Condition()
        self._task: Optional[FrameTask] = None
        self._closed = False

    def submit(self, task: FrameTask) -> bool:
        """Submit a task. Returns True if a previous pending task was overwritten."""
        with self._cond:
            overwritten = self._task is not None
            self._task = task
            self._cond.notify()
            return overwritten

    def pop(self) -> Optional[FrameTask]:
        with self._cond:
            while self._task is None and not self._closed:
                self._cond.wait()
            if self._task is None and self._closed:
                return None
            out = self._task
            self._task = None
            return out

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class AsyncVideoSequenceBenchmark(VideoSequenceBenchmark):
    """Async variant of Lane B benchmark with fixed-rate ingestion."""

    def _init_from_cfg(self) -> None:
        super()._init_from_cfg()
        p = self.cfg["benchmark_vid_params"]
        async_cfg = p.get("async_runtime", {})
        self.target_fps = float(async_cfg.get("target_fps", 30.0))
        self.max_target_fps = float(async_cfg.get("max_target_fps", 240.0))
        self.display_window_name = str(async_cfg.get("window_name", "Async Lane B"))
        self.show_gt = bool(async_cfg.get("show_gt", True))
        self.show_rois = bool(async_cfg.get("show_rois", True))
        self.hold_last_policy = str(async_cfg.get("benchmark_reuse_policy", "hold_last")).strip().lower()
        if self.hold_last_policy != "hold_last":
            raise ValueError("Only benchmark_reuse_policy='hold_last' is currently supported.")

        # Adaptive tau assumes ordered per-frame cost samples. Disable here to
        # prevent mixing skipped/reused timeline frames with worker samples.
        if self.adaptive_tau_enabled:
            print("[async_runtime] Disabling adaptive tau for async mode to keep cost model semantics valid.")
            self.adaptive_tau_enabled = False
            self.run_metadata["adaptive_tau_enabled"] = False

        self.run_metadata.update(
            {
                "runtime_mode": "async",
                "async_target_fps": self.target_fps,
                "async_backend": "thread_latest_slot",
                "benchmark_reuse_policy": self.hold_last_policy,
            }
        )

    def _draw_text_with_bg(
        self,
        image: Any,
        text: str,
        org: Tuple[int, int],
        *,
        font: int = cv2.FONT_HERSHEY_PLAIN,
        font_scale: float = 1.0,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        thickness: int = 1,
        padding: int = 2,
    ) -> None:
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        x, y = org

        x = max(0, x)
        y = max(text_h + padding, y)
        x2 = x + text_w + 2 * padding
        y2 = y + baseline + padding
        y1 = y - text_h - padding

        cv2.rectangle(image, (x, y1), (x2, y2), bg_color, thickness=-1)
        cv2.putText(image, text, (x + padding, y), font, font_scale, text_color, thickness)

    def _worker_loop(
        self,
        *,
        slot: LatestFrameSlot,
        result_queue: "queue.Queue[ProcessedFrame]",
        model: Any,
        dataset: Any,
        model_device: torch.device,
        conf_threshold: float,
        im_size_hw: Tuple[int, int],
    ) -> None:
        tracker = self.tracker
        tracker.reset()
        current_video_id: Optional[str] = None
        next_frame_rois: List[List[int]] = []
        tracker_type = str(self.cfg["benchmark_vid_params"]["tracker"]["type"])

        with torch.inference_mode():
            while True:
                task = slot.pop()
                if task is None:
                    break

                if current_video_id is None:
                    current_video_id = task.video_id
                if task.is_first_frame or task.video_id != current_video_id:
                    tracker.reset()
                    next_frame_rois = []
                    current_video_id = task.video_id

                if tracker_type == "oracle_gt":
                    oracle_dets = extract_gt_for_tracker(task.target, dataset.idx2label, task.frame_w, task.frame_h)
                    if hasattr(tracker, "set_oracle_detections"):
                        tracker.set_oracle_detections(oracle_dets)
                    if hasattr(tracker, "preview_rois"):
                        next_frame_rois = tracker.preview_rois((task.frame_h, task.frame_w))

                started_at = time.perf_counter()
                result = process_frame(
                    model=model,
                    idx2label=dataset.idx2label,
                    frame_bgr=task.frame_bgr,
                    im_tensor=task.im_tensor,
                    tracker=tracker,
                    next_frame_rois=next_frame_rois,
                    frame_idx=task.effective_frame_idx,
                    key_frame_interval=self.key_frame_interval,
                    im_size_hw=im_size_hw,
                    conf_threshold=conf_threshold,
                    nms_iou=self.nms_iou,
                    merge_fn=self.merge_fn,
                    merge_tau=self.merge_tau,
                    model_device=model_device,
                    tracker_input_dropout_cfg=self.tracker_input_dropout_cfg,
                )
                completed_at = time.perf_counter()

                next_frame_rois = result.next_frame_rois
                result_queue.put(
                    ProcessedFrame(
                        frame_idx=task.frame_idx,
                        effective_frame_idx=task.effective_frame_idx,
                        frame_result=result,
                        frame_w=task.frame_w,
                        frame_h=task.frame_h,
                        started_at=started_at,
                        completed_at=completed_at,
                    )
                )

    def _drain_results(
        self,
        *,
        result_queue: "queue.Queue[ProcessedFrame]",
        latest_processed: Optional[ProcessedFrame],
        lat_full: List[float],
        lat_roi: List[float],
        lat_merge: List[float],
        roi_counts_pre: List[float],
        roi_counts_post: List[float],
        processed_frames_counter: List[int],
    ) -> Optional[ProcessedFrame]:
        while True:
            try:
                processed = result_queue.get_nowait()
            except queue.Empty:
                break

            processed_frames_counter[0] += 1
            if processed.frame_result.use_full_frame:
                lat_full.append(processed.frame_result.latency_s)
            else:
                lat_roi.append(processed.frame_result.latency_s)
                lat_merge.append(processed.frame_result.merge_latency_s)
                roi_counts_post.append(len(processed.frame_result.rois_used))
                roi_counts_pre.append(
                    float(
                        len(processed.frame_result.rois_used)
                        + len(processed.frame_result.next_frame_rois)
                    )
                )
            latest_processed = processed
        return latest_processed

    def run_mode(self, *, mode: str = "benchmark") -> Dict[str, Any]:
        run_start_time = time.perf_counter()
        args = argparse.Namespace(config_path=self._train_config_path)
        model, dataset, data_loader, train_cfg = load_model_and_dataset(
            self.cfg["benchmark_vid_params"]["device"],
            args,
        )

        conf_threshold = train_cfg["train_params"]["infer_conf_threshold"]
        model.low_score_threshold = conf_threshold
        im_size_hw = ensure_im_size_tuple(train_cfg["dataset_params"]["im_size"])
        model_device = next(model.parameters()).device

        if self.target_fps <= 0.0 or self.target_fps > self.max_target_fps:
            raise ValueError(f"async target_fps must be in (0, {self.max_target_fps}] but got {self.target_fps}")

        total_frames = len(data_loader)
        predictions: List[Dict[str, List]] = []
        ground_truths: List[Dict[str, List]] = []
        difficulties: List[Dict[str, List]] = []

        lat_full: List[float] = []
        lat_roi: List[float] = []
        lat_merge: List[float] = []
        area_ratios: List[float] = []
        roi_counts_pre: List[float] = []
        roi_counts_post: List[float] = []
        gt_coverages: List[float] = []

        produced_frames = 0
        submitted_frames = 0
        overwritten_frames = 0
        reused_prediction_frames = 0
        no_prediction_frames = 0
        prediction_ages: List[float] = []
        processed_frames_counter = [0]

        slot = LatestFrameSlot()
        result_queue: "queue.Queue[ProcessedFrame]" = queue.Queue()
        worker = threading.Thread(
            target=self._worker_loop,
            kwargs={
                "slot": slot,
                "result_queue": result_queue,
                "model": model,
                "dataset": dataset,
                "model_device": model_device,
                "conf_threshold": conf_threshold,
                "im_size_hw": im_size_hw,
            },
            daemon=True,
        )
        worker.start()

        latest_processed: Optional[ProcessedFrame] = None

        if mode == "interactive":
            cv2.namedWindow(self.display_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.display_window_name, 1280, 720)
            print("Controls: q or ESC to quit.")

        frame_period = 1.0 / self.target_fps
        start_wall = time.perf_counter()

        try:
            with torch.no_grad():
                for frame_idx, (im_tensor, target, fname) in enumerate(data_loader, start=1):
                    deadline = start_wall + (frame_idx - 1) * frame_period
                    now = time.perf_counter()
                    if now < deadline:
                        time.sleep(deadline - now)

                    produced_frames += 1

                    latest_processed = self._drain_results(
                        result_queue=result_queue,
                        latest_processed=latest_processed,
                        lat_full=lat_full,
                        lat_roi=lat_roi,
                        lat_merge=lat_merge,
                        roi_counts_pre=roi_counts_pre,
                        roi_counts_post=roi_counts_post,
                        processed_frames_counter=processed_frames_counter,
                    )

                    fpath = os.path.abspath(fname[0] if isinstance(fname, (list, tuple)) else fname)
                    frame_bgr = cv2.imread(fpath)
                    if frame_bgr is None:
                        continue
                    frame_h, frame_w = frame_bgr.shape[:2]

                    tgt = target[0] if isinstance(target, list) else target
                    video_id, is_first_frame, frame_idx_in_video = _extract_sequence_meta(tgt, fname)
                    effective_frame_idx = frame_idx_in_video if frame_idx_in_video is not None else frame_idx

                    task = FrameTask(
                        frame_idx=frame_idx,
                        effective_frame_idx=effective_frame_idx,
                        frame_bgr=frame_bgr,
                        im_tensor=im_tensor,
                        target=tgt,
                        video_id=video_id,
                        is_first_frame=is_first_frame,
                        frame_w=frame_w,
                        frame_h=frame_h,
                    )

                    if slot.submit(task):
                        overwritten_frames += 1
                    submitted_frames += 1

                    gt_d, diff_d = extract_gt_for_map(tgt, dataset.idx2label, frame_w, frame_h)
                    ground_truths.append(gt_d)
                    difficulties.append(diff_d)

                    if latest_processed is None:
                        predictions.append({})
                        no_prediction_frames += 1
                        search_rois: List[List[int]] = []
                    else:
                        predictions.append(_detections_to_map_pred(latest_processed.frame_result.final_detections))
                        age = frame_idx - latest_processed.frame_idx
                        prediction_ages.append(float(age))
                        if age > 0:
                            reused_prediction_frames += 1

                        if latest_processed.frame_result.use_full_frame:
                            search_rois = [[0, 0, frame_w - 1, frame_h - 1]]
                        else:
                            search_rois = latest_processed.frame_result.rois_used

                    all_gt = [b for boxes in gt_d.values() for b in boxes]
                    gt_coverages.append(_gt_roi_coverage(search_rois, all_gt, self.coverage_threshold))

                    if latest_processed is None:
                        area_ratios.append(float("nan"))
                    elif latest_processed.frame_result.use_full_frame:
                        area_ratios.append(1.0)
                    elif latest_processed.frame_result.rois_used:
                        roi_area = sum(
                            max(0, r[2] - r[0]) * max(0, r[3] - r[1])
                            for r in latest_processed.frame_result.rois_used
                        )
                        area_ratios.append(float(roi_area) / max(float(frame_w * frame_h), 1.0))
                    else:
                        area_ratios.append(float("nan"))

                    if mode == "interactive":
                        display = frame_bgr.copy()
                        dets = latest_processed.frame_result.final_detections if latest_processed is not None else []
                        rois_used = latest_processed.frame_result.rois_used if latest_processed is not None else []
                        for det in dets:
                            x1, y1, x2, y2 = det["bbox"]
                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = "{}:{:.2f}".format(det["class"], det["confidence"])
                            self._draw_text_with_bg(display, label, (x1 + 2, max(12, y1 - 4)))

                        if self.show_gt:
                            for cls, gt_boxes in gt_d.items():
                                _ = cls
                                for gx1, gy1, gx2, gy2 in gt_boxes:
                                    cv2.rectangle(
                                        display,
                                        (int(gx1), int(gy1)),
                                        (int(gx2), int(gy2)),
                                        (0, 255, 0),
                                        2,
                                    )

                        if self.show_rois:
                            for roi in rois_used:
                                x1, y1, x2, y2 = roi
                                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        age = int(prediction_ages[-1]) if prediction_ages else -1
                        mode_text = "N/A"
                        if latest_processed is not None:
                            mode_text = "FULL" if latest_processed.frame_result.use_full_frame else "ROI"
                        overlay = (
                            f"Frame {frame_idx}/{total_frames} | target_fps={self.target_fps:.2f} "
                            f"| pred_age={age} | mode={mode_text} | overwritten={overwritten_frames}"
                        )
                        cv2.putText(display, overlay, (10, 24), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
                        self._draw_text_with_bg(display, f"Video: {video_id}", (10, 46))
                        cv2.imshow(self.display_window_name, display)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord("q"):
                            print("Exiting interactive run...")
                            break

                    if self.verbose and frame_idx % max(1, total_frames // 10) == 0:
                        print(f"  {frame_idx}/{total_frames} ({(frame_idx / max(total_frames, 1)) * 100:.0f}%)")
        finally:
            slot.close()
            worker.join(timeout=30.0)
            latest_processed = self._drain_results(
                result_queue=result_queue,
                latest_processed=latest_processed,
                lat_full=lat_full,
                lat_roi=lat_roi,
                lat_merge=lat_merge,
                roi_counts_pre=roi_counts_pre,
                roi_counts_post=roi_counts_post,
                processed_frames_counter=processed_frames_counter,
            )
            if mode == "interactive":
                cv2.destroyAllWindows()

        metrics = self._compute(
            predictions,
            ground_truths,
            difficulties,
            lat_full,
            lat_roi,
            lat_merge,
            area_ratios,
            roi_counts_pre,
            roi_counts_post,
            gt_coverages,
            len(predictions),
            train_cfg,
        )

        elapsed_s = time.perf_counter() - run_start_time
        achieved_stream_fps = float(len(predictions) / max(elapsed_s, 1e-9))
        metrics.update(
            {
                "runtime_mode": mode,
                "async_target_fps": float(self.target_fps),
                "stream_fps_achieved": achieved_stream_fps,
                "frames_ingested": int(produced_frames),
                "frames_submitted": int(submitted_frames),
                "frames_overwritten": int(overwritten_frames),
                "frames_processed": int(processed_frames_counter[0]),
                "prediction_reused_frames": int(reused_prediction_frames),
                "prediction_missing_frames": int(no_prediction_frames),
                "prediction_age_mean_frames": float(np.nanmean(np.array(prediction_ages))) if prediction_ages else float("nan"),
                "prediction_age_p95_frames": float(np.nanpercentile(np.array(prediction_ages), 95)) if prediction_ages else float("nan"),
                "worker_processing_fps": float(processed_frames_counter[0] / max(elapsed_s, 1e-9)),
            }
        )

        if mode == "benchmark":
            self._print(metrics, elapsed_s=elapsed_s)
            self._save(metrics)
        else:
            print("Interactive run summary")
            print(f"  frames_ingested: {metrics['frames_ingested']}")
            print(f"  frames_processed: {metrics['frames_processed']}")
            print(f"  stream_fps_achieved: {metrics['stream_fps_achieved']:.2f}")
            print(f"  worker_processing_fps: {metrics['worker_processing_fps']:.2f}")

        return metrics


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_sweep_config(sweep_cfg: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    if not isinstance(sweep_cfg, dict):
        raise TypeError("Sweep config must be a YAML mapping.")

    sweep_name = str(sweep_cfg.get("sweep_name", "tracker_sweep"))
    experiments = sweep_cfg.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Sweep config must contain a non-empty 'experiments' list.")

    validated: List[Dict[str, Any]] = []
    for i, exp in enumerate(experiments, start=1):
        if not isinstance(exp, dict):
            raise TypeError(f"Experiment #{i} must be a mapping.")

        name = exp.get("name")
        tracker_type = exp.get("tracker_type")
        tracker_params = exp.get("tracker_params", {})
        benchmark_overrides = exp.get("benchmark_overrides", {})
        inference = exp.get("inference", {})
        roi_merge = exp.get("roi_merge", {})

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Experiment #{i} has invalid 'name'.")
        if tracker_type not in SUPPORTED_TRACKERS:
            raise ValueError(
                f"Experiment '{name}' has unsupported tracker_type '{tracker_type}'. "
                f"Expected one of {sorted(SUPPORTED_TRACKERS)}."
            )
        if not isinstance(tracker_params, dict):
            raise TypeError(f"Experiment '{name}' has non-dict 'tracker_params'.")
        if not isinstance(benchmark_overrides, dict):
            raise TypeError(f"Experiment '{name}' has non-dict 'benchmark_overrides'.")
        if not isinstance(inference, dict):
            raise TypeError(f"Experiment '{name}' has non-dict 'inference'.")
        if not isinstance(roi_merge, dict):
            raise TypeError(f"Experiment '{name}' has non-dict 'roi_merge'.")

        strategy = roi_merge.get("strategy")
        if strategy is not None and strategy not in SUPPORTED_ROI_MERGE_STRATEGIES:
            raise ValueError(
                f"Experiment '{name}' has unsupported roi_merge.strategy '{strategy}'. "
                f"Expected one of {sorted(SUPPORTED_ROI_MERGE_STRATEGIES)}."
            )

        validated.append(
            {
                "name": name,
                "tracker_type": tracker_type,
                "tracker_params": tracker_params,
                "benchmark_overrides": benchmark_overrides,
                "inference": inference,
                "roi_merge": roi_merge,
            }
        )

    return sweep_name, validated


def _build_experiment_cfg(base_cfg: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg = _deep_merge(cfg, exp["benchmark_overrides"])

    benchmark_vid_params = cfg.setdefault("benchmark_vid_params", {})
    tracker_cfg = benchmark_vid_params.setdefault("tracker", {})

    tracker_type = str(exp["tracker_type"])
    tracker_cfg["type"] = tracker_type
    tracker_cfg.setdefault(tracker_type, {})
    tracker_cfg[tracker_type] = _deep_merge(tracker_cfg[tracker_type], exp["tracker_params"])

    inference_override = exp.get("inference", {})
    if inference_override:
        inference_cfg = benchmark_vid_params.setdefault("inference", {})
        benchmark_vid_params["inference"] = _deep_merge(inference_cfg, inference_override)

    roi_merge_override = exp.get("roi_merge", {})
    if roi_merge_override:
        roi_merge_cfg = benchmark_vid_params.setdefault("roi_merge", {})
        benchmark_vid_params["roi_merge"] = _deep_merge(roi_merge_cfg, roi_merge_override)

    return cfg


def _run_single_config(
    cfg: Dict[str, Any],
    *,
    mode: str,
    extra_run_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bench = AsyncVideoSequenceBenchmark.from_config_dict(cfg, extra_run_metadata=extra_run_metadata)
    return bench.run_mode(mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Async 30 FPS Lane B runner")
    parser.add_argument(
        "--benchmark-config",
        required=True,
        help="Base benchmark config (same format as benchmark_framework_vid.py)",
    )
    parser.add_argument(
        "--mode",
        default="benchmark",
        choices=["benchmark", "interactive"],
        help="benchmark: save CSV metrics, interactive: visualize latest predictions",
    )
    parser.add_argument(
        "--sweep-config",
        default=None,
        help="Optional sweep config with same schema as batch_benchmark_vid.py",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when a sweep experiment fails",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.benchmark_config)

    if not args.sweep_config:
        _run_single_config(base_cfg, mode=args.mode)
        return

    if args.mode != "benchmark":
        raise ValueError("Sweep mode is only supported with --mode benchmark.")

    sweep_cfg = load_config(args.sweep_config)
    sweep_name, experiments = _validate_sweep_config(sweep_cfg)

    output_cfg = base_cfg.get("benchmark_vid_params", {}).get("output", {})
    results_dir = output_cfg.get("results_dir", "")
    results_filename = output_cfg.get("results_filename", "")

    print("=" * 80)
    print(f"Async batch benchmark sweep: {sweep_name}")
    print(f"Experiments: {len(experiments)}")
    print(f"Output CSV (from benchmark config): {Path(results_dir) / results_filename}")
    print("=" * 80)

    failures: List[Tuple[str, str]] = []
    for idx, exp in enumerate(experiments, start=1):
        exp_name = exp["name"]
        print("\n" + "-" * 80)
        print(f"[{idx}/{len(experiments)}] Running experiment: {exp_name}")
        print(f"  tracker_type: {exp['tracker_type']}")
        print(f"  tracker_params: {exp['tracker_params']}")

        try:
            exp_cfg = _build_experiment_cfg(base_cfg, exp)
            _run_single_config(
                exp_cfg,
                mode="benchmark",
                extra_run_metadata={
                    "sweep_name": sweep_name,
                    "experiment_name": exp_name,
                    "experiment_index": idx,
                },
            )
        except Exception as exc:
            failures.append((exp_name, str(exc)))
            print(f"  FAILED: {exc}")
            if args.fail_fast:
                break

    succeeded = len(experiments) - len(failures)
    print("\n" + "=" * 80)
    print("Async batch sweep finished")
    print(f"Succeeded: {succeeded}")
    print(f"Failed   : {len(failures)}")

    if failures:
        print("Failures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
