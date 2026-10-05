"""Auditable outputs for the existing sequential video benchmark."""
import hashlib
import json
import math
import subprocess
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.cost_model_validation.common import system_metadata, read_cpu_temp, read_cpu_freq
from tools.infer import compute_map


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def hardware_snapshot():
    throttling = None
    if shutil.which("vcgencmd"):
        try:
            throttling = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return dict(cpu_temp_c=read_cpu_temp(), cpu_freq_mhz=read_cpu_freq(), throttling=throttling)


def timed_batches(loader):
    iterator = iter(loader)
    while True:
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        yield batch, time.perf_counter() - start


def detection_metrics(predictions, targets, difficult):
    values = [compute_map(predictions, targets, iou_threshold=float(t), difficult=difficult)
              for t in np.linspace(.5, .95, 10)]
    return dict(mAP50=float(values[0][0]), mAP50_95=float(np.mean([v[0] for v in values])),
                recall50=float(values[0][2]))


def aggregate(rows, predictions, targets, difficult):
    durations = [r["total_latency_s"] for r in rows]
    total = sum(durations)
    pixels = sum(r["full_tensor_pixels"] for r in rows)
    gt_count = sum(r["gt_count"] for r in rows)
    return dict(
        **detection_metrics(predictions, targets, difficult),
        num_frames=len(rows), total_time_s=total, FPS_total=len(rows)/total if total else None,
        latency_mean_ms=float(np.mean(durations)*1000),
        latency_median_ms=float(np.median(durations)*1000),
        latency_p95_ms=float(np.percentile(durations, 95)*1000),
        inference_time_s=sum(r["inference_time_s"] for r in rows),
        inference_calls=sum(r["inference_calls"] for r in rows),
        inference_calls_per_frame=sum(r["inference_calls"] for r in rows)/len(rows),
        roi_count_per_frame=sum(r["roi_count"] for r in rows)/len(rows),
        merges=sum(r["merges"] for r in rows),
        full_frame_fraction=sum(r["full_frame"] for r in rows)/len(rows),
        effective_processed_area_ratio=sum(r["tensor_pixels"] for r in rows)/pixels if pixels else None,
        gt_roi_coverage=sum(r["gt_covered"] for r in rows)/gt_count if gt_count else None,
        gt_outside_roi=sum(r["gt_count"]-r["gt_covered"] for r in rows),
        fallback_count=sum(r["fallback_count"] for r in rows),
        candidate_evaluations=sum(len(r["merge_decisions"]) for r in rows),
        fallback_reasons=dict(Counter(d["reason"] for r in rows for d in r["merge_decisions"] if d["reason"])),
        tracker_time_s=sum(r["tracker_time_s"] for r in rows),
        roi_formation_merge_time_s=sum(r["merge_time_s"] for r in rows),
        crop_time_s=sum(r["crop_time_s"] for r in rows),
        postprocess_time_s=sum(r["postprocess_time_s"] for r in rows))


def save_run(directory, rows, predictions, targets, difficult, cfg, train_cfg, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = [{k:r[k] for k in ("path", "video_id", "frame_index", "width", "height")}
                | {"ground_truth": gt, "difficult": diff}
                for r, gt, diff in zip(rows, targets, difficult)]
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(directory / "manifest.json", dict(sha256=manifest_hash, frames=manifest))
    with (directory / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(clean(row), allow_nan=False)+"\n")
    write_json(directory / "detections.json", dict(predictions=predictions, ground_truths=targets, difficulties=difficult))
    per_video = {}
    for video in dict.fromkeys(r["video_id"] for r in rows):
        indices = [i for i,r in enumerate(rows) if r["video_id"] == video]
        per_video[video] = aggregate(*[[v[i] for i in indices] for v in (rows, predictions, targets, difficult)])
    write_json(directory / "videos.json", per_video)
    summary = aggregate(rows, predictions, targets, difficult)
    fallback_fraction = summary["fallback_count"] / max(1, summary["candidate_evaluations"])
    warnings = list(metadata.get("warnings", []))
    if summary["candidate_evaluations"] == 0 and metadata.get("policy") in {"linear_direct_cost", "shape_lookup_conservative"}:
        warnings.append("No candidate pairs in this fragment; zero fallback/merge counts do not establish lookup coverage or policy differences.")
    if fallback_fraction > .1:
        warnings.append(f"Fallback in {fallback_fraction:.1%} of candidate evaluations: expand independent calibration before a full run.")
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    summary.update(config=cfg, train_config=train_cfg, command=list(getattr(sys, "orig_argv", [sys.executable, *sys.argv])),
                   timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
                   dataset_manifest_sha256=manifest_hash, system=system_metadata(),
                   metadata=metadata, hardware_after=hardware_snapshot(), warnings=warnings,
                   definitions={
                       "total_time": "Sum of loader/decode/preprocess, second image read and synchronized process_frame; excludes initialization, warmup, evaluation, artifact I/O and metric bookkeeping.",
                       "inference_time": "Synchronized transfer plus model forward, including detector internal postprocessing; excludes external NMS and tracker.",
                       "area_ratio": "Sum of actual input tensor H*W per inference / sum of full-frame transformed tensor H*W on every retained frame.",
                       "coverage": "GT object covered when at least one actual snapped ROI intersects >= configured fraction of GT area; all GT including difficult. Full-frame covers all. No-ROI frame contributes zero pixels/calls if no inference; existing empty-proposal fallback uses full frame.",
                       "mAP50_95": "Arithmetic mean of existing evaluator AP at 10 IoUs .50:.05:.95; not the COCO evaluator.",
                       "uncertainty": "No frame-iid confidence intervals; videos are the sampling units.",
                       "roi_formation": "Tracker update includes next-frame proposal formation; separate merge timer includes coordinate conversion and effective-shape calculation."})
    write_json(directory / "summary.json", summary)
    return summary
