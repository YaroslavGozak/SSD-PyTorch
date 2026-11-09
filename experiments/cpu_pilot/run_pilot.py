"""Run the locked Stage 3 CPU pilot on its preparation manifests."""

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
import yaml
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

CPU_THREADS = 8
os.environ['SSD_INFERENCE_DEVICE'] = 'cpu'
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from experiments.cost_model_validation.common import file_sha256
from experiments.cpu_pilot.prepare import CONFIDENCE_THRESHOLD, DEFAULT_MODELS, MODEL_CHECKPOINTS
from tools.benchmarks.video_artifacts import detection_metrics, write_json
from tools.helpers.pipeline import (
    _convert_rois_between_spaces,
    convert_crop_to_input_tensor,
    extract_gt_for_map,
    load_model_and_dataset,
    run_model_inference,
    tensor_to_detection_list,
    process_frame,
)
from tools.trackers.static_padding_tracker import StaticPaddingTracker


ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "final_experiments" / "pilot"
SPLIT_METADATA = PILOT_DIR / "split_metadata.json"
MODEL_NAMES = DEFAULT_MODELS
DEFAULT_EVALUATION_CONF = CONFIDENCE_THRESHOLD
PADDING_VALUES = (32, 64, 96, 112, 128, 144, 160, 192, 224)
KEY_FRAME_INTERVAL = 10
ROI_GRID = {"yolo26n": 32, "roi_ssd": 1, "roi_ssd_mobilenet": 1}
CALIBRATION_SHAPES = (
    (32, 32), (64, 64), (96, 96), (128, 128), (192, 192), (256, 256), (320, 320),
    (64, 128), (128, 64), (160, 320), (320, 160),
)


def _sha256(path):
    return file_sha256(str(path))


def _read_tsv_manifest(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value
    return value


def _configure_runtime():
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)


def _cpu_model():
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                model, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if model.strip():
                    return model.strip()
        except OSError:
            pass
    model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    if not model:
        raise RuntimeError("Unable to determine CPU model for Stage 3 runtime metadata")
    return model


def _runtime_environment():
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    if physical_cores is None or logical_cores is None:
        raise RuntimeError("Unable to determine physical and logical CPU core counts")
    return {
        "cpu_model": _cpu_model(),
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "pytorch_intra_op_threads": torch.get_num_threads(),
        "pytorch_inter_op_threads": torch.get_num_interop_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "os": platform.platform(),
        "python": sys.version,
        "pytorch": str(torch.__version__),
    }


def _load_model(model_name, device, evaluation_conf=DEFAULT_EVALUATION_CONF):
    if device.type != "cpu":
        raise ValueError("The CPU pilot requires device=cpu")
    snapshot = PILOT_DIR / "configs" / f"{model_name}.yaml"
    print(f"Loading {model_name} model from {snapshot}", flush=True)
    args = SimpleNamespace(config_path=str(snapshot))
    model, dataset, _loader, config = load_model_and_dataset(device, args)
    model.eval()
    if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
        raise RuntimeError('CPU pilot loaded non-CPU model parameters')
    evaluation_conf = _validate_confidence("evaluation_conf", evaluation_conf)
    model.low_score_threshold = evaluation_conf
    if model_name == 'yolo26n':
        model.conf_threshold = evaluation_conf
    if model_name == "yolo26n":
        if not hasattr(model, "max_detections"):
            raise TypeError("Selected YOLO config did not load the project YOLO adapter")
        model.max_detections = 200
        model.iou_threshold = 0.45
        model.input_normalization = "imagenet"
        class_names = model._yolo.names
        if isinstance(class_names, dict):
            class_names = [class_names[index] for index in sorted(class_names)]
        expected_names = [dataset.idx2label[index] for index in sorted(dataset.idx2label) if index > 0]
        if list(class_names) != expected_names:
            raise ValueError("YOLO checkpoint class order does not match the dataset class order")
    torch.set_num_threads(CPU_THREADS)
    if torch.get_num_threads() != CPU_THREADS:
        raise RuntimeError(f"CPU thread limit was not applied: {torch.get_num_threads()}")
    return model, dataset, config, evaluation_conf


def _attach_shape_observer(model):
    observed = []
    base_model = getattr(model, "_yolo", None)
    if base_model is not None:
        base_model = base_model.model
    else:
        base_model = model

    def record(_module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            observed.append(tuple(map(int, inputs[0].shape[-2:])))

    handle = base_model.register_forward_pre_hook(record)
    return observed, handle


def _dataset_indices(dataset, manifest_rows, dataset_root):
    indices_by_path = {
        Path(info["filename"]).resolve(): index
        for index, info in enumerate(dataset.images_info)
    }
    selected = []
    missing = []
    for row in manifest_rows:
        image_path = (dataset_root / Path(row["image"])).resolve()
        index = indices_by_path.get(image_path)
        if index is None:
            missing.append(str(image_path))
        else:
            selected.append(index)
    if missing:
        raise ValueError(f"{len(missing)} pilot images are absent from the configured dataset; first={missing[0]}")
    return selected


def _dataset_root(config):
    dataset_yaml = Path(config["dataset_params"]["yolo_dataset_yaml"])
    import yaml

    parsed = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    return Path(parsed["path"]).resolve()


def _dataloader(dataset, indices):
    return DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)


def _gt_objects(gt):
    return [box for boxes in gt.values() for box in boxes]


def _coverage(gt, rois, frame_w, frame_h, is_full_frame):
    objects = _gt_objects(gt)
    if not objects:
        return 0, 0
    if is_full_frame:
        return len(objects), len(objects)
    covered = 0
    for gx1, gy1, gx2, gy2 in objects:
        area = max(0.0, (gx2 - gx1) * (gy2 - gy1))
        for rx1, ry1, rx2, ry2 in rois:
            intersection = (
                max(0.0, min(gx2, rx2) - max(gx1, rx1))
                * max(0.0, min(gy2, ry2) - max(gy1, ry1))
            )
            if area > 0 and intersection / area >= 0.5:
                covered += 1
                break
    return covered, len(objects)


def _cost_ms(shape, coefficients):
    height, width = shape
    return coefficients["K_ms"] + coefficients["c_ms_per_pixel"] * height * width


def _validate_confidence(name, value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"--{name.replace('_', '-')} must be a finite value between 0 and 1")
    return value


class AffineCostMerger:
    """Greedily merge only when measured affine inference cost decreases."""

    requires_effective_shape = True

    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.decisions = []

    def __call__(self, rois, effective_shape=None, **_kwargs):
        clusters = [list(map(int, roi)) for roi in rois]
        self.decisions = []
        if effective_shape is None:
            raise ValueError("Cost-aware merging requires measured effective tensor shapes")
        while len(clusters) > 1:
            best = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    first, second = clusters[i], clusters[j]
                    union = [
                        min(first[0], second[0]), min(first[1], second[1]),
                        max(first[2], second[2]), max(first[3], second[3]),
                    ]
                    separate_shapes = (effective_shape(first), effective_shape(second))
                    merged_shape = effective_shape(union)
                    separate_ms = sum(_cost_ms(shape, self.coefficients) for shape in separate_shapes)
                    merged_ms = _cost_ms(merged_shape, self.coefficients)
                    gain_ms = separate_ms - merged_ms
                    self.decisions.append({
                        "first": first, "second": second, "union": union,
                        "first_shape_hw": separate_shapes[0],
                        "second_shape_hw": separate_shapes[1],
                        "union_shape_hw": merged_shape,
                        "separate_predicted_ms": separate_ms,
                        "merged_predicted_ms": merged_ms,
                        "predicted_gain_ms": gain_ms,
                        "merged": gain_ms > 0.0,
                    })
                    if gain_ms > 0.0 and (best is None or gain_ms > best[0]):
                        best = gain_ms, i, j, union
            if best is None:
                break
            _, i, j, union = best
            clusters = [roi for k, roi in enumerate(clusters) if k not in (i, j)]
            clusters.append(union)
        return clusters


def _effective_shape_factory(im_tensor, frame_hw, roi_grid):
    frame_h, frame_w = frame_hw
    tensor_h, tensor_w = map(int, im_tensor.shape[-2:])

    def effective_shape(model_rect):
        original_roi = _convert_rois_between_spaces(
            [model_rect], (tensor_w, tensor_h), (frame_w, frame_h)
        )[0]
        crop_tensor, _ = convert_crop_to_input_tensor(
            im_tensor=im_tensor,
            crop=original_roi,
            original_image_size=(frame_h, frame_w),
            roi_grid=roi_grid,
        )
        return tuple(map(int, crop_tensor.shape[-2:]))

    return effective_shape


def _merge_candidates(model_space_rois, effective_shape, merger):
    return merger(model_space_rois, effective_shape=effective_shape)


def _warmup(model, first_batch, device, observed):
    images = first_batch[0].float().to(device)
    before = len(observed)
    with torch.inference_mode():
        for _ in range(2):
            run_model_inference(model, images)
    if len(observed) - before < 2:
        raise RuntimeError("Warm-up did not reach the detector network")


def _map_prediction(detections):
    output = {}
    for det in detections:
        class_name = str(det["class"])
        x1, y1, x2, y2 = det["bbox"]
        output.setdefault(class_name, []).append(
            [float(x1), float(y1), float(x2), float(y2), float(det["confidence"])]
        )
    return output


def _save_debug(
    model, idx2label, im_tensor, frame_bgr, result, gt, target_path,
    frame_meta, model_name, run_name, debug_index, roi_grid,
):
    if debug_index >= 20:
        return False
    frame_h, frame_w = frame_bgr.shape[:2]
    source_rois = result.rois_used or [[0, 0, frame_w, frame_h]]
    crop_records = []
    with torch.inference_mode():
        for source_roi in source_rois:
            crop_tensor, snapped = convert_crop_to_input_tensor(
                im_tensor=im_tensor,
                crop=source_roi,
                original_image_size=(frame_h, frame_w),
                roi_grid=roi_grid,
            )
            _, batch_detections = run_model_inference(model, crop_tensor.unsqueeze(0))
            sx1, sy1, sx2, sy2 = snapped
            local = tensor_to_detection_list(
                batch_detections[0], idx2label, sx2 - sx1, sy2 - sy1
            )
            global_detections = tensor_to_detection_list(
                batch_detections[0], idx2label, sx2 - sx1, sy2 - sy1,
                offset_xy=(sx1, sy1),
            )
            for local_detection, global_detection in zip(local, global_detections):
                expected = [
                    local_detection["bbox"][0] + sx1,
                    local_detection["bbox"][1] + sy1,
                    local_detection["bbox"][2] + sx1,
                    local_detection["bbox"][3] + sy1,
                ]
                if global_detection["bbox"] != expected:
                    raise AssertionError("Local-to-global detection coordinate remapping failed")
            crop_records.append({
                "source_roi_xyxy": snapped,
                "local_detections": local,
                "global_detections_before_frame_nms": global_detections,
            })

    canvas = frame_bgr.copy()
    for x1, y1, x2, y2 in source_rois:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 200, 0), 2)
    for boxes in gt.values():
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    for detection in result.operational_detections:
        x1, y1, x2, y2 = detection["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f'{detection["class"]}:{detection["confidence"]:.2f}',
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    debug_dir = PILOT_DIR / "debug_frames" / model_name / run_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{debug_index:03d}_{frame_meta["video_id"]}_{frame_meta["frame_index"]}'
    image_path = debug_dir / f"{stem}.jpg"
    json_path = debug_dir / f"{stem}.json"
    if not cv2.imwrite(str(image_path), canvas):
        raise OSError(f"Could not write debug image: {image_path}")
    write_json(json_path, {
        "image": str(target_path),
        "video_id": frame_meta["video_id"],
        "frame_index": frame_meta["frame_index"],
        "coordinate_contract": (
            "local detections are measured in each submitted crop; global detections are "
            "the same crop outputs offset by the exact snapped source ROI"
        ),
        "candidate_source_rois": result.rois_used,
        "crop_detections": crop_records,
        "final_global_detections_after_frame_nms_and_threshold": result.operational_detections,
        "global_operational_detections": result.operational_detections,
        "ground_truth": gt,
        "overlay": str(image_path.relative_to(PILOT_DIR)),
    })
    return True


def _run_pipeline(model_name, model, dataset, config, evaluation_conf, operational_conf,
                  manifest_rows, indices, run_name, mode, padding, coefficients=None,
                  debug=False):
    device = torch.device("cpu")
    idx2label = dataset.idx2label
    dataset_root = _dataset_root(config)
    loader = _dataloader(dataset, indices)
    iterator = iter(loader)
    first_batch = next(iterator)
    shape_events, hook = _attach_shape_observer(model)
    _warmup(model, first_batch, device, shape_events)
    full_network_hw = shape_events[0]
    iterator = iter(_dataloader(dataset, indices))

    merger = AffineCostMerger(coefficients) if mode in {"CA", "FULL-TECH"} else None
    tracker = None
    tracker_video = None
    next_rois = []
    rows = []
    predictions = []
    targets = []
    difficult = []
    nn_call_latencies_s = []
    debug_count = 0
    image_count = len(indices)
    config_inference = config.get("benchmark_params", {}).get("inference", {})
    nms_iou = float(config_inference.get("nms_iou", 0.45))
    im_size_hw = (
        (int(config["dataset_params"]["im_size"]),) * 2
        if isinstance(config["dataset_params"]["im_size"], int)
        else tuple(map(int, config["dataset_params"]["im_size"]))
    )

    try:
        with torch.inference_mode():
            for frame_number, (batch, manifest_row) in enumerate(zip(iterator, manifest_rows)):
                im_tensor, target_batch, fname = batch
                image_path = fname[0] if isinstance(fname, (list, tuple)) else fname
                frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise OSError(f"Cannot read pilot image: {image_path}")
                frame_h, frame_w = frame_bgr.shape[:2]
                target = {
                    key: value
                    for key, value in target_batch.items()
                }
                gt, diff = extract_gt_for_map(target, idx2label, frame_w, frame_h)
                video_id = str(_first(target.get("video_id", manifest_row["video_id"])))
                frame_idx = int(_first(target.get("frame_idx", manifest_row["frame_index"])))
                is_first = bool(_first(target.get("is_first_frame", False)))
                if tracker is None or is_first or video_id != tracker_video:
                    tracker = StaticPaddingTracker(
                        pad_x=padding,
                        pad_y=padding,
                        conf_det_min=operational_conf,
                        model_input_size=(im_size_hw[0], im_size_hw[1]),
                        hold_last_for_frames=0,
                    )
                    next_rois = []
                    tracker_video = video_id

                candidate_count = 0 if mode == "FF" else len(next_rois)
                full_area = frame_h * frame_w
                model_h, model_w = map(int, im_tensor.shape[-2:])
                roi_grid = ROI_GRID[model_name]
                fallback_reason = ""
                candidate_merged_count = candidate_count
                segment_cost = None
                full_cost = _cost_ms(full_network_hw, coefficients) if coefficients else None
                selector_started = time.perf_counter()
                force_full = mode == "FF"
                is_key_frame = frame_idx % KEY_FRAME_INTERVAL == 0
                if mode == "FF":
                    fallback_reason = "full_frame_mode"
                elif not next_rois:
                    force_full = True
                    fallback_reason = "no_segments"
                elif is_key_frame:
                    force_full = True
                    fallback_reason = "periodic_keyframe"
                elif mode == "FULL-TECH":
                    model_rois = _convert_rois_between_spaces(
                        next_rois, (frame_w, frame_h), (model_w, model_h)
                    )
                    effective_shape = _effective_shape_factory(
                        im_tensor, (frame_h, frame_w), roi_grid
                    )
                    merged_rois = _merge_candidates(model_rois, effective_shape, merger)
                    candidate_merged_count = len(merged_rois)
                    segment_cost = sum(
                        _cost_ms(effective_shape(roi), coefficients) for roi in merged_rois
                    )
                    full_cost = _cost_ms(full_network_hw, coefficients)
                    if segment_cost >= full_cost:
                        force_full = True
                        fallback_reason = "cost_prefers_full_frame"
                    else:
                        fallback_reason = ""

                selector_time = time.perf_counter() - selector_started
                next_for_frame = [] if force_full else next_rois
                before_events = len(shape_events)
                result = process_frame(
                    model=model,
                    idx2label=idx2label,
                    frame_bgr=frame_bgr,
                    im_tensor=im_tensor.float(),
                    tracker=tracker,
                    next_frame_rois=next_for_frame,
                    frame_idx=frame_idx,
                    key_frame_interval=KEY_FRAME_INTERVAL,
                    im_size_hw=im_size_hw,
                    conf_threshold=operational_conf,
                    evaluation_conf_threshold=evaluation_conf,
                    nms_iou=nms_iou,
                    merge_fn=merger if merger is not None else (lambda rois, **_: list(rois)),
                    model_device=device,
                    roi_grid=roi_grid,
                )
                frame_shapes = shape_events[before_events:]
                if not frame_shapes:
                    raise RuntimeError("Frame produced no observed detector-network call")
                requested_shape_counts = {
                    (height, width): len(latencies)
                    for (width, height), latencies in result.roi_latencies_s.items()
                }
                if sum(requested_shape_counts.values()) != len(frame_shapes):
                    raise RuntimeError(
                        "Detector wrapper input count does not match observed network forwards"
                    )
                next_rois = result.next_frame_rois
                if mode == "CA":
                    segment_cost = sum(_cost_ms(shape, coefficients) for shape in frame_shapes)

                if result.use_full_frame:
                    candidate_merged_count = 0
                    geom_area = full_area
                else:
                    if mode != "FULL-TECH":
                        candidate_merged_count = candidate_count - result.merge_count
                    geom_area = sum(
                        max(0, x2 - x1) * max(0, y2 - y1)
                        for x1, y1, x2, y2 in result.rois_used
                    )
                covered, gt_count = _coverage(
                    gt, result.rois_used, frame_w, frame_h, result.use_full_frame
                )
                predictions.append(_map_prediction(result.evaluation_detections))
                targets.append(gt)
                difficult.append(diff)
                total_tensor_pixels = sum(h * w for h, w in frame_shapes)
                frame_nn_latencies = [
                    latency
                    for shape_latencies in result.roi_latencies_s.values()
                    for latency in shape_latencies
                ]
                if len(frame_nn_latencies) != len(frame_shapes):
                    raise RuntimeError("Timing records do not match observed network forwards")
                nn_call_latencies_s.extend(frame_nn_latencies)
                rows.append({
                    "run": run_name,
                    "model": model_name,
                    "mode": mode,
                    "padding": padding,
                    "evaluation_conf": evaluation_conf,
                    "operational_conf": operational_conf,
                    "evaluation_detection_count": len(result.evaluation_detections),
                    "operational_detection_count": len(result.operational_detections),
                    "frame_id": f"{video_id}:{frame_idx}",
                    "predicted_segment_cost": segment_cost,
                    "predicted_full_frame_cost": full_cost,
                    "selector_decision": ("FF" if force_full else "SEG") if mode == "FULL-TECH" else "not_enabled",
                    "selector_time_s": selector_time if mode == "FULL-TECH" else 0.0,
                    "video_id": video_id,
                    "frame_index": frame_idx,
                    "image": str(image_path),
                    "input_mode": "full" if result.use_full_frame else "segments",
                    "candidate_segments": candidate_count,
                    "segments_after_merge": candidate_merged_count,
                    "nn_calls": len(frame_shapes),
                    "requested_input_shapes_hw": json.dumps({
                        f"{h}x{w}": count
                        for (h, w), count in sorted(requested_shape_counts.items())
                    }),
                    "actual_network_shapes_hw": json.dumps(frame_shapes),
                    "geometric_processed_area": geom_area,
                    "geometric_processed_area_ratio": geom_area / max(1, full_area),
                    "tensor_pixels": total_tensor_pixels,
                    "tensor_area_ratio": total_tensor_pixels / max(
                        1, full_network_hw[0] * full_network_hw[1]
                    ),
                    "gt_covered": covered,
                    "gt_count": gt_count,
                    "full_frame_fallback": bool(result.use_full_frame and mode != "FF"),
                    "fallback_reason": fallback_reason if result.use_full_frame else "",
                    "merge_decision_time_s": result.merge_latency_s,
                    "crop_preprocess_time_s": result.crop_latency_s,
                    "nn_time_s": sum(sum(v) for v in result.roi_latencies_s.values()),
                    "nn_call_latencies_ms": json.dumps(
                        [value * 1000.0 for value in frame_nn_latencies]
                    ),
                    "postprocess_time_s": result.postprocess_latency_s,
                    "tracker_formation_time_s": result.tracker_latency_s,
                    "total_process_frame_time_s": result.latency_s + (selector_time if mode == "FULL-TECH" else 0.0),
                    "detection_count": len(result.operational_detections),
                    "prediction_records": json.dumps(
                        result.evaluation_detections, separators=(",", ":")
                    ),
                    "operational_prediction_records": json.dumps(
                        result.operational_detections, separators=(",", ":")
                    ),
                    "ground_truth_records": json.dumps(gt, separators=(",", ":")),
                    "merge_decisions": json.dumps(result.merge_decisions, separators=(",", ":")),
                })
                if debug and debug_count < 20:
                    debug_count += int(_save_debug(
                        model=model,
                        idx2label=idx2label,
                        im_tensor=im_tensor.float(),
                        frame_bgr=frame_bgr,
                        result=result,
                        gt=gt,
                        target_path=image_path,
                        frame_meta={"video_id": video_id, "frame_index": frame_idx},
                        model_name=model_name,
                        run_name=run_name,
                        debug_index=debug_count,
                        roi_grid=roi_grid,
                    ))
                if (frame_number + 1) % 50 == 0 or frame_number + 1 == image_count:
                    print(
                        f"[{model_name}/{run_name}] {frame_number + 1}/{image_count} frames",
                        flush=True,
                    )
    finally:
        hook.remove()

    if debug and debug_count < 20:
        raise RuntimeError(f"Expected at least 20 debug frames; exported {debug_count}")
    metrics = detection_metrics(predictions, targets, difficult)
    durations = np.asarray([row["total_process_frame_time_s"] for row in rows], dtype=np.float64)
    n = max(1, len(rows))
    total_gt = sum(row["gt_count"] for row in rows)
    total_pixels = sum(row["tensor_pixels"] for row in rows)
    full_tensor_pixels = len(rows) * full_network_hw[0] * full_network_hw[1]
    summary = {
        "run": run_name,
        "model": model_name,
        "mode": mode,
        "padding": padding,
        "num_frames": len(rows),
        **metrics,
        "latency_total_s": float(durations.sum()),
        "latency_mean_ms": float(durations.mean() * 1000),
        "latency_p50_ms": float(np.percentile(durations, 50) * 1000),
        "latency_p95_ms": float(np.percentile(durations, 95) * 1000),
        "fps": float(len(rows) / durations.sum()) if durations.sum() else None,
        "inference_time_s": float(sum(nn_call_latencies_s)),
        "inference_latency_mean_ms": (
            float(np.mean(nn_call_latencies_s) * 1000) if nn_call_latencies_s else None
        ),
        "inference_latency_p50_ms": (
            float(np.percentile(nn_call_latencies_s, 50) * 1000) if nn_call_latencies_s else None
        ),
        "inference_latency_p95_ms": (
            float(np.percentile(nn_call_latencies_s, 95) * 1000) if nn_call_latencies_s else None
        ),
        "gt_roi_coverage_mean": sum(row["gt_covered"] for row in rows) / max(1, total_gt),
        "gt_roi_coverage_p5": float(np.percentile(
            [row["gt_covered"] / row["gt_count"] if row["gt_count"] else 0.0 for row in rows],
            5,
        )),
        "processed_area_ratio_mean": float(np.mean(
            [row["geometric_processed_area_ratio"] for row in rows]
        )),
        "tensor_area_ratio_mean": float(np.mean(
            [row["tensor_area_ratio"] for row in rows]
        )),
        "full_frame_fraction": sum(row["input_mode"] == "full" for row in rows) / n,
        "candidate_segments_mean": float(np.mean([row["candidate_segments"] for row in rows])),
        "segments_after_merge_mean": float(np.mean([row["segments_after_merge"] for row in rows])),
        "nn_calls_per_frame": float(np.mean([row["nn_calls"] for row in rows])),
        "nn_calls_total": int(sum(row["nn_calls"] for row in rows)),
        "tensor_pixels_total": int(total_pixels),
        "full_tensor_pixels_total": int(full_tensor_pixels),
        "fallback_counts": dict(Counter(
            row["fallback_reason"] for row in rows if row["fallback_reason"]
        )),
        "merge_time_s": float(sum(row["merge_decision_time_s"] for row in rows)),
        "crop_preprocess_time_s": float(sum(row["crop_preprocess_time_s"] for row in rows)),
        "postprocess_time_s": float(sum(row["postprocess_time_s"] for row in rows)),
        "tracker_formation_time_s": float(sum(row["tracker_formation_time_s"] for row in rows)),
        "evaluation_conf": evaluation_conf,
        "operational_conf": operational_conf,
        "operational_confidence_sha256": (_sha256(PILOT_DIR / 'operational_confidence.json')
                                           if (PILOT_DIR / 'operational_confidence.json').is_file() else None),
    }
    return summary, rows


def _profile_cost(model_name, model, dataset, indices, repetitions):
    sample = dataset[indices[0]][0].unsqueeze(0).float()
    events, hook = _attach_shape_observer(model)
    device = torch.device("cpu")
    observations = {}
    raw = []
    try:
        with torch.inference_mode():
            for height, width in CALIBRATION_SHAPES:
                if model_name == "yolo26n" and (height % 32 or width % 32):
                    raise ValueError(f"YOLO calibration shape must be stride-32 aligned: {(height,width)}")
                batch = F.interpolate(sample, size=(height, width), mode="bilinear", align_corners=False)
                for _ in range(2):
                    run_model_inference(model, batch.to(device))
                for repetition in range(repetitions):
                    before = len(events)
                    start = time.perf_counter()
                    run_model_inference(model, batch.to(device))
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    if len(events) != before + 1:
                        raise RuntimeError("Cost profile call did not produce exactly one network forward")
                    actual_shape = events[-1]
                    observations.setdefault(actual_shape, []).append(elapsed_ms)
                    raw.append({
                        "model": model_name,
                        "requested_h": height,
                        "requested_w": width,
                        "actual_network_h": actual_shape[0],
                        "actual_network_w": actual_shape[1],
                        "repetition": repetition,
                        "latency_ms": elapsed_ms,
                    })
    finally:
        hook.remove()

    shape_rows = [
        (shape, float(np.median(values)), float(np.percentile(values, 95)))
        for shape, values in sorted(observations.items())
    ]
    if len(shape_rows) < 3:
        raise ValueError(
            f"{model_name} produced only {len(shape_rows)} actual shapes; cannot fit T(A)"
        )
    areas = np.asarray([shape[0] * shape[1] for shape, _, _ in shape_rows], dtype=np.float64)
    times_ms = np.asarray([median for _, median, _ in shape_rows], dtype=np.float64)
    matrix = np.column_stack((np.ones_like(areas), areas))
    intercept, slope = np.linalg.lstsq(matrix, times_ms, rcond=None)[0]
    predicted = matrix @ np.asarray([intercept, slope])
    residual = times_ms - predicted
    ss_total = float(np.sum((times_ms - times_ms.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual ** 2)) / ss_total if ss_total > 0 else 1.0
    fit = {
        "K_ms": float(intercept),
        "c_ms_per_pixel": float(slope),
        "tau_pixels": float(intercept / slope) if slope > 0 else None,
        "r_squared": r_squared,
        "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(np.sqrt(np.mean(residual ** 2))),
        "warmup_per_shape": 2,
        "repetitions_per_shape": repetitions,
        "actual_shape_count": len(shape_rows),
    }
    if not math.isfinite(fit["K_ms"]) or fit["K_ms"] <= 0 or not math.isfinite(fit["c_ms_per_pixel"]) or fit["c_ms_per_pixel"] <= 0:
        raise ValueError(f"Nonphysical CPU affine fit for {model_name}: {fit}")
    return fit, shape_rows, raw


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_session():
    if not SPLIT_METADATA.is_file():
        raise FileNotFoundError("Run experiments/cpu_pilot/prepare.py before the pilot")
    split = json.loads(SPLIT_METADATA.read_text(encoding="utf-8"))
    selected_checkpoints = split["provenance"]["selected_checkpoints"]
    for name in MODEL_NAMES:
        if name not in selected_checkpoints:
            raise ValueError(
                f"Prepared pilot is missing {name}; prepare a new output directory "
                "with the current Stage 3 model set"
            )
        prepared = selected_checkpoints[name]
        checkpoint, expected_hash = MODEL_CHECKPOINTS[name]
        if (
            Path(prepared["path"]).as_posix() != checkpoint.relative_to(ROOT).as_posix()
            or prepared["sha256"] != expected_hash
        ):
            raise ValueError(f"Prepared checkpoint selection does not match the locked Stage 3 selection: {name}")
        if _sha256(ROOT / prepared["path"]) != prepared["sha256"]:
            raise ValueError(f"Locked checkpoint changed: {name}")
        snapshot = PILOT_DIR / "configs" / f"{name}.yaml"
        expected = split["provenance"].get("model_configs", {}).get(name, {}).get("sha256")
        if expected and _sha256(snapshot) != expected:
            raise ValueError(f"Locked config changed: {name}")
        snapshot_config = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        configured_threshold = float(snapshot_config["train_params"]["infer_conf_threshold"])
        if configured_threshold != CONFIDENCE_THRESHOLD:
            raise ValueError(
                f"Prepared {name} config uses confidence threshold {configured_threshold}; "
                f"this Stage 3 run requires {CONFIDENCE_THRESHOLD}. Prepare a fresh output directory."
            )
    if split["status"] != "PREPARED_NOT_BENCHMARKED":
        raise ValueError("Unexpected preparation metadata status")
    manifests = {
        "pilot": _read_tsv_manifest(PILOT_DIR / "pilot_manifest.txt"),
        "final": _read_tsv_manifest(PILOT_DIR / "final_eval_manifest_candidate.txt"),
    }
    if _sha256(PILOT_DIR / "pilot_manifest.txt") != split["provenance"]["pilot_manifest_sha256"]:
        raise ValueError("Pilot manifest changed after preparation")
    if _sha256(PILOT_DIR / "final_eval_manifest_candidate.txt") != split["provenance"]["final_eval_manifest_candidate_sha256"]:
        raise ValueError("Final evaluation candidate changed after preparation")
    return split, manifests


def _metadata(model_name, model, dataset, config, evaluation_conf, operational_conf, split):
    config_path = PILOT_DIR / "configs" / f"{model_name}.yaml"
    train = config["train_params"]
    if model_name == "yolo26n":
        weights = Path(train["yolo_weights"])
    else:
        weights = ROOT / "trained_models" / train["task_name"] / train["ckpt_name"]
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.is_file():
        raise FileNotFoundError(f"Selected checkpoint missing: {weights}")
    info = {
        "model": model_name,
        "checkpoint": str(weights.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(weights),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "evaluation_conf": evaluation_conf,
        "operational_conf": operational_conf,
        "operational_confidence_sha256": (_sha256(PILOT_DIR / 'operational_confidence.json')
                                           if (PILOT_DIR / 'operational_confidence.json').is_file() else None),
        "input_size": config["dataset_params"]["im_size"],
        "transform_name": config["dataset_params"]["transform_name"],
        "dataset_class_count": len(dataset.idx2label) - 1,
        "dataset_label_names": [dataset.idx2label[i] for i in sorted(dataset.idx2label) if i > 0],
        "roi_grid": ROI_GRID[model_name],
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "opencv_threads": cv2.getNumThreads(),
    }
    info.update(_runtime_environment())
    if model_name == "yolo26n":
        info["max_detections"] = model.max_detections
        info["model_nms_iou"] = model.iou_threshold
        info["input_normalization"] = model.input_normalization
    return info


def _resolve_operational_conf(model_name, config, evaluation_conf, override=None):
    if override is None:
        from .confidence import load_frozen_threshold
        operational_conf = load_frozen_threshold(PILOT_DIR, model_name, config)
    else:
        operational_conf = _validate_confidence("operational_conf", override)
    if operational_conf < evaluation_conf:
        raise ValueError(
            f"{model_name} operational_conf ({operational_conf}) must be at least "
            f"evaluation_conf ({evaluation_conf})"
        )
    return operational_conf


def _require_matching_stage_thresholds(decisions, profiles, model_names):
    for model_name in model_names:
        padding = decisions[model_name]
        profile = profiles[model_name]
        for field in ("evaluation_conf", "operational_conf"):
            if field not in padding or field not in profile:
                raise ValueError(
                    f"Saved padding/cost artifacts for {model_name} lack {field}; "
                    "rerun --stage padding and --stage cost with frozen thresholds"
                )
            if float(padding[field]) != float(profile[field]):
                raise ValueError(
                    f"Saved padding and cost {field} differ for {model_name}; "
                    "rerun both stages before pipeline"
                )


def run_padding_sweep(split, manifest_rows, device, max_frames=None, partial=False,
                      evaluation_conf=DEFAULT_EVALUATION_CONF, operational_conf=None):
    out_rows = []
    summaries = []
    resolved_operational = {}
    for model_name in MODEL_NAMES:
        config = yaml.safe_load(
            (PILOT_DIR / "configs" / f"{model_name}.yaml").read_text(encoding="utf-8")
        )
        operational = _resolve_operational_conf(
            model_name, config, evaluation_conf, operational_conf
        )
        resolved_operational[model_name] = operational
    for model_name in MODEL_NAMES:
        model, dataset, config, _ = _load_model(model_name, device, evaluation_conf)
        operational = resolved_operational[model_name]
        all_indices = _dataset_indices(dataset, manifest_rows, _dataset_root(config))
        rows = manifest_rows[:max_frames] if max_frames else manifest_rows
        indices = all_indices[:len(rows)]
        model_info = _metadata(
            model_name, model, dataset, config, evaluation_conf, operational, split
        )
        for padding in ("full_frame",) + PADDING_VALUES:
            run_mode = "FF" if padding == "full_frame" else "SEG"
            numeric_padding = 0 if padding == "full_frame" else int(padding)
            summary, frame_rows = _run_pipeline(
                model_name, model, dataset, config, evaluation_conf, operational,
                rows, indices,
                f"padding_{padding}", run_mode, numeric_padding,
            )
            summary.update(model_info)
            summaries.append(summary)
            out_rows.extend(frame_rows)
        del model, dataset
    output_name = "padding_smoke.csv" if partial else "padding_sweep.csv"
    _write_csv(PILOT_DIR / output_name, summaries)
    _merge_frame_rows(out_rows)

    decisions = {}
    for model_name in MODEL_NAMES:
        model_summaries = [row for row in summaries if row["model"] == model_name]
        full = next(row for row in model_summaries if row["mode"] == "FF")
        eligible = [
            row for row in model_summaries
            if row["mode"] == "SEG"
            and full["mAP50"] > 0
            and row["mAP50"] / full["mAP50"] >= 0.95
            and row["gt_roi_coverage_mean"] >= 0.98
        ]
        chosen = min((int(row["padding"]) for row in eligible), default=max(PADDING_VALUES))
        selected = next(row for row in model_summaries if row["mode"] == "SEG" and int(row["padding"]) == chosen)
        map_ratio = selected["mAP50"] / full["mAP50"] if full["mAP50"] > 0 else None
        decisions[model_name] = {
            "status": "PASS" if eligible else "PADDING_GATE_FAILED",
            "evaluation_conf": evaluation_conf,
            "operational_conf": resolved_operational[model_name],
            "padding_final": chosen,
            "criterion_met": bool(eligible),
            "full_frame_map50": full["mAP50"],
            "selected_segment_map50": selected["mAP50"],
            "selected_segment_map50_ratio": map_ratio,
            "selected_segment_gt_roi_coverage_mean": selected["gt_roi_coverage_mean"],
            "accuracy_gate_passed": map_ratio is not None and map_ratio >= 0.95,
            "eligible_padding_values": sorted(int(row["padding"]) for row in eligible),
            "selection_rule": (
                "smallest padding with segment/full-frame mAP50 >= 0.95 and GT coverage >= 0.98; "
                "otherwise largest tested padding"
            ),
        }
    write_json(
        PILOT_DIR / ("padding_smoke_decision.json" if partial else "padding_decision.json"),
        decisions,
    )


def run_cost_profiles(split, manifest_rows, device, repetitions,
                      evaluation_conf=DEFAULT_EVALUATION_CONF, operational_conf=None):
    profiles = {}
    rows = []
    raw_rows = []
    resolved_operational = {}
    for model_name in MODEL_NAMES:
        config = yaml.safe_load(
            (PILOT_DIR / "configs" / f"{model_name}.yaml").read_text(encoding="utf-8")
        )
        resolved_operational[model_name] = _resolve_operational_conf(
            model_name, config, evaluation_conf, operational_conf
        )
    for model_name in MODEL_NAMES:
        model, dataset, config, _ = _load_model(model_name, device, evaluation_conf)
        operational = resolved_operational[model_name]
        all_indices = _dataset_indices(dataset, manifest_rows, _dataset_root(config))
        fit, shape_rows, _raw = _profile_cost(model_name, model, dataset, all_indices, repetitions)
        for raw_row in _raw:
            raw_row["evaluation_conf"] = evaluation_conf
            raw_row["operational_conf"] = operational
            raw_row["timing_mode"] = "shared_detector_call_at_evaluation_floor"
        raw_rows.extend(_raw)
        profiles[model_name] = fit
        model_info = _metadata(
            model_name, model, dataset, config, evaluation_conf, operational, split
        )
        for shape, median, p95 in shape_rows:
            rows.append({
                **model_info,
                "timing_mode": "shared_detector_call_at_evaluation_floor",
                "actual_network_h": shape[0],
                "actual_network_w": shape[1],
                "actual_tensor_pixels": shape[0] * shape[1],
                "median_latency_ms": median,
                "p95_latency_ms": p95,
                **fit,
            })
        profiles[model_name]["evaluation_conf"] = evaluation_conf
        profiles[model_name]["operational_conf"] = operational
        profiles[model_name]["timing_mode"] = "shared_detector_call_at_evaluation_floor"
        del model, dataset
    write_json(PILOT_DIR / "cost_profiles_cpu.json", profiles)
    _write_csv(PILOT_DIR / "cost_profiles_cpu.csv", rows)
    _write_csv(PILOT_DIR / "cost_measurements_cpu.csv", raw_rows)


def run_pipeline_modes(split, manifest_rows, device, max_frames=None, debug=True, partial=False,
                       evaluation_conf=DEFAULT_EVALUATION_CONF, operational_conf=None):
    decision_path = PILOT_DIR / "padding_decision.json"
    profile_path = PILOT_DIR / "cost_profiles_cpu.json"
    if not decision_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError("Run the padding sweep and CPU calibration before pipeline modes")
    decisions = json.loads(decision_path.read_text(encoding="utf-8"))
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    for model_name in MODEL_NAMES:
        if not decisions[model_name].get("criterion_met", False) and not split.get("diagnostic_on_padding_gate_failure", False):
            raise RuntimeError(
                f"STOP: {model_name} has no tested padding satisfying both segment/full-frame "
                "mAP50 >= 0.95 and GT coverage >= 0.98. Investigate before running CA/FULL-TECH."
            )
    rows = manifest_rows[:max_frames] if max_frames else manifest_rows
    summaries = []
    frame_rows = []
    configs = {
        model_name: yaml.safe_load(
            (PILOT_DIR / "configs" / f"{model_name}.yaml").read_text(encoding="utf-8")
        )
        for model_name in MODEL_NAMES
    }
    operational_by_model = {
        model_name: _resolve_operational_conf(
            model_name, configs[model_name], evaluation_conf, operational_conf
        )
        for model_name in MODEL_NAMES
    }
    _require_matching_stage_thresholds(decisions, profiles, MODEL_NAMES)
    for model_name in MODEL_NAMES:
        if (
            float(decisions[model_name]["evaluation_conf"]) != evaluation_conf
            or float(decisions[model_name]["operational_conf"])
            != operational_by_model[model_name]
        ):
            raise ValueError(
                f"Pipeline thresholds differ from the padding sweep for {model_name}; "
                "rerun padding and cost with the same evaluation_conf/operational_conf"
            )
    for model_name in MODEL_NAMES:
        model, dataset, config, _ = _load_model(model_name, device, evaluation_conf)
        operational = operational_by_model[model_name]
        indices = _dataset_indices(dataset, rows, _dataset_root(config))
        padding = int(decisions[model_name]["padding_final"])
        for mode in ("FF", "SEG", "CA", "FULL-TECH"):
            summary, mode_rows = _run_pipeline(
                model_name, model, dataset, config, evaluation_conf, operational,
                rows, indices,
                f"final_{mode.lower()}", mode, padding,
                coefficients=profiles[model_name] if mode in {"CA", "FULL-TECH"} else None,
                debug=debug and mode == "SEG",
            )
            summary.update(_metadata(
                model_name, model, dataset, config, evaluation_conf, operational, split
            ))
            summary["padding_final"] = padding
            summary["padding_gate_status"] = "PASS" if decisions[model_name].get("criterion_met") else "PADDING_GATE_FAILED"
            summary["cost_profile"] = profiles[model_name]
            summaries.append(summary)
            frame_rows.extend(mode_rows)
        del model, dataset
    if partial:
        _write_csv(PILOT_DIR / "pipeline_smoke_summary.csv", summaries)
        _write_csv(PILOT_DIR / "pipeline_smoke_frame_metrics.csv", frame_rows)
    else:
        _merge_frame_rows(frame_rows)
        _write_csv(PILOT_DIR / "pipeline_summary.csv", summaries)


def _merge_frame_rows(new_rows):
    path = PILOT_DIR / "frame_level_metrics.csv"
    existing = []
    if path.is_file():
        field_limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(field_limit)
                break
            except OverflowError:
                field_limit //= 10
        with path.open("r", encoding="utf-8", newline="") as stream:
            existing = list(csv.DictReader(stream))
    keyed = {
        (row["run"], row["model"], row["video_id"], str(row["frame_index"])): row
        for row in existing
    }
    for row in new_rows:
        key = (row["run"], row["model"], row["video_id"], str(row["frame_index"]))
        keyed[key] = row
    _write_csv(path, list(keyed.values()))


def run_smoke(split, manifest_rows, device, max_frames=20,
              evaluation_conf=DEFAULT_EVALUATION_CONF, operational_conf=None):
    if max_frames < 20:
        raise ValueError("Coordinate smoke must process at least 20 frames")
    rows = manifest_rows[:max_frames]
    summaries = []
    frame_rows = []
    configs = {
        model_name: yaml.safe_load(
            (PILOT_DIR / "configs" / f"{model_name}.yaml").read_text(encoding="utf-8")
        )
        for model_name in MODEL_NAMES
    }
    operational_by_model = {
        model_name: _resolve_operational_conf(
            model_name, configs[model_name], evaluation_conf, operational_conf
        )
        for model_name in MODEL_NAMES
    }
    for model_name in MODEL_NAMES:
        model, dataset, config, _ = _load_model(model_name, device, evaluation_conf)
        operational = operational_by_model[model_name]
        indices = _dataset_indices(dataset, rows, _dataset_root(config))
        for mode in ("FF", "SEG"):
            summary, model_rows = _run_pipeline(
                model_name, model, dataset, config, evaluation_conf, operational,
                rows, indices,
                f"coordinate_smoke_{mode.lower()}", mode, 64,
                debug=mode == "SEG",
            )
            if any(
                row["evaluation_detection_count"] < row["operational_detection_count"]
                for row in model_rows
            ):
                raise AssertionError(
                    f"{model_name} evaluation detections fewer than operational detections"
                )
            summary.update(_metadata(
                model_name, model, dataset, config, evaluation_conf, operational, split
            ))
            summaries.append(summary)
            frame_rows.extend(model_rows)
        del model, dataset
    _write_csv(PILOT_DIR / "smoke_summary.csv", summaries)
    _write_csv(PILOT_DIR / "smoke_frame_metrics.csv", frame_rows)
    accuracy_smoke = {}
    for model_name in MODEL_NAMES:
        accuracy_smoke[model_name] = {
            row["mode"]: {
                "mAP50": row["mAP50"],
                "recall50": row["recall50"],
            }
            for row in summaries
            if row["model"] == model_name
        }
    smoke = {
        "status": "PASS",
        "frames_per_model": len(rows),
        "models": list(MODEL_NAMES),
        "evaluation_conf": evaluation_conf,
        "operational_conf_by_model": operational_by_model,
        "detection_counts": {
            model_name: {
                "evaluation_detections": sum(
                    int(row["evaluation_detection_count"])
                    for row in frame_rows if row["model"] == model_name
                ),
                "operational_detections": sum(
                    int(row["operational_detection_count"])
                    for row in frame_rows if row["model"] == model_name
                ),
                "candidate_segments": sum(
                    int(row["candidate_segments"])
                    for row in frame_rows if row["model"] == model_name
                ),
            }
            for model_name in MODEL_NAMES
        },
        "debug_frame_count": sum(
            len(list((PILOT_DIR / "debug_frames" / model / "coordinate_smoke_seg").glob("*.jpg")))
            for model in MODEL_NAMES
        ),
        "checks": [
            "all selected models loaded from locked configs/checkpoints",
            "model class IDs match the dataset class order",
            "FF and SEG detector forwards ran on the CPU for at least 20 pilot frames per model",
            "crop-local detections remapped to source-frame coordinates exactly",
            "GT, segments, local outputs, and global outputs exported for visual inspection",
            "evaluation detections are a superset of operational detections",
        ],
        "accuracy_smoke_not_decision_grade": accuracy_smoke,
    }
    write_json(PILOT_DIR / "smoke_metadata.json", smoke)
    return smoke


def _write_metadata(split, stage, arguments):
    metadata_path = PILOT_DIR / "metadata.json"
    previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    selected_models = {}
    for name in MODEL_NAMES:
        config_path = PILOT_DIR / "configs" / f"{name}.yaml"
        selected_models[name] = {
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256(config_path),
        }
    runtime = _runtime_environment()
    metadata = {
        "experiment": "Stage 3 CPU full-pipeline pilot",
        "status": previous.get("status", "PREPARED_NOT_BENCHMARKED"),
        "stage_requested": stage,
        "arguments": arguments,
        "confidence_protocol": {
            "evaluation_conf": arguments["evaluation_conf"],
            "operational_conf": arguments["operational_conf"],
            "operational_conf_by_model": arguments.get("operational_conf_by_model", {}),
            "operational_conf_source": (
                "selected_by_confidence_stage" if stage == "confidence" else
                "explicit_cli_override" if arguments["operational_conf"] is not None else
                "frozen_confidence_result"
            ),
        },
        "completed_stages": previous.get("completed_stages", []),
        "pilot_manifest_sha256": split["provenance"]["pilot_manifest_sha256"],
        "final_eval_manifest_candidate_sha256": split["provenance"]["final_eval_manifest_candidate_sha256"],
        "selected_models": selected_models,
        "selected_checkpoints": split["provenance"]["selected_checkpoints"],
        "frozen_protocol": {
            "dataset": "sampled ImageNet-VID validation, video-disjoint pilot",
            "default_evaluation_conf": DEFAULT_EVALUATION_CONF,
            "operational_conf_source": "per-model frozen result from --stage confidence",
            "preprocessing": "dataset ssd transform, 320x320 base input",
            "segment_formation": "Static/previous localization",
            "hold_last_for_frames": 0,
            "key_frame_interval": KEY_FRAME_INTERVAL,
            "empty_segment_fallback": "full frame",
            "padding_sweep_pixels": list(PADDING_VALUES),
            "padding_selection": (
                "smallest p with segment/full-frame mAP50 >= 0.95 and GT coverage >= 0.98; "
                f"otherwise p={max(PADDING_VALUES)}"
            ),
            "pipeline_modes": ["FF", "SEG", "CA", "FULL-TECH"],
            "merge_cost_model": "T(A)=K+cA fitted on actual network input shape/area",
            "roi_grid_pixels": ROI_GRID,
            "timing_warmups": 2,
            "cpu_cost_repetitions_per_shape_default": 8,
            "yolo_max_detections": 200,
            "nms_iou": 0.45,
        },
        "metric_definitions": {
            "mAP50_95": "mean of the existing evaluator AP at IoUs 0.50 through 0.95 in 0.05 steps",
            "gt_roi_coverage_mean": "fraction of GT boxes with at least 50 percent of GT area covered by a submitted segment",
            "gt_roi_coverage_p5": "5th percentile of per-frame covered-GT fraction",
            "processed_area_ratio_mean": "mean per-frame sum of snapped source ROI areas / source frame area; overlap is counted per inference",
            "tensor_area_ratio_mean": "mean per-frame actual network tensor pixels / full-frame actual network tensor pixels",
            "latency": "process_frame time excludes dataset loading, evaluation-only filtering, and metric calculation",
            "inference_latency": "shared detector call at evaluation_conf, including network forward and detector-internal NMS; evaluation metrics are outside the timing",
            "cost_timing_mode": "shared_detector_call_at_evaluation_floor",
        },
        "runtime_controls": {
            "device": "cpu",
            **runtime,
            "torch_threads": runtime["pytorch_intra_op_threads"],
            "torch_interop_threads": runtime["pytorch_inter_op_threads"],
            "platform": runtime["os"],
            "processor": runtime["cpu_model"],
            "cpu_count": runtime["logical_cores"],
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
    }
    write_json(metadata_path, metadata)


def _mark_stage(stage, status):
    path = PILOT_DIR / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    completed = set(metadata.get("completed_stages", []))
    completed.add(stage)
    metadata["completed_stages"] = sorted(completed)
    if "pipeline_modes" in completed and status in {
        "PILOT_RUNS_COMPLETE_PENDING_REVIEW",
        "PARTIAL_PIPELINE_SMOKE",
    }:
        metadata["status"] = status
    elif {"padding_sweep", "cpu_cost_calibration"} <= completed:
        metadata["status"] = "PADDING_AND_CPU_COST_READY_PIPELINE_PENDING"
    elif {"coordinate_smoke", "cpu_cost_calibration"} <= completed:
        metadata["status"] = "SMOKE_AND_CPU_COST_COMPLETE_PADDING_PENDING"
    elif "coordinate_smoke" in completed:
        metadata["status"] = "COORDINATE_SMOKE_PASS_FULL_PILOT_PENDING"
    else:
        metadata["status"] = status
    write_json(path, metadata)


def _refresh_report():
    report_path = PILOT_DIR / "pilot_report.md"
    report = report_path.read_text(encoding="utf-8")
    metadata_path = PILOT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    decision_path = PILOT_DIR / "padding_decision.json"
    profile_path = PILOT_DIR / "cost_profiles_cpu.json"
    summary_path = PILOT_DIR / "pipeline_summary.csv"
    sections = []
    runtime = metadata.get("runtime_controls", {})
    if runtime:
        sections.extend([
            "## Runtime environment",
            "",
            "| Property | Value |",
            "|---|---|",
            f'| CPU model | {runtime["cpu_model"]} |',
            f'| Physical cores | {runtime["physical_cores"]} |',
            f'| Logical cores | {runtime["logical_cores"]} |',
            f'| PyTorch intra-op threads | {runtime["pytorch_intra_op_threads"]} |',
            f'| PyTorch inter-op threads | {runtime["pytorch_inter_op_threads"]} |',
            f'| OpenCV threads | {runtime["opencv_threads"]} |',
            f'| OS | {runtime["os"]} |',
            f'| Python | {runtime["python"]} |',
            f'| PyTorch | {runtime["pytorch"]} |',
            "",
        ])
    operational_path = PILOT_DIR / 'operational_confidence.json'
    operational = None
    if operational_path.is_file():
        operational = json.loads(operational_path.read_text(encoding='utf-8'))
        if "models" not in operational:
            sections.extend([
                "## Frozen operational confidence",
                "",
                "A legacy confidence result is present and is not valid for the dual-threshold "
                "protocol. Use a fresh output directory and rerun `--stage confidence`.",
                "",
            ])
            operational = None
    if operational is not None:
        candidate_grid = operational.get("candidate_grid")
        if candidate_grid is None:
            candidate_grid = next(iter(operational["models"].values()))["candidate_grid"]
        sections.extend([
            '## Frozen operational confidence', '',
            f"Evaluation floor: {operational['evaluation_conf']}.",
            operational["selection_rule"],
            f"Candidate grid: {candidate_grid}.",
            "",
            "| Model | Operational confidence | Micro F1@0.5 | Recall |",
            "|---|---:|---:|---:|",
        ])
        for model_name, decision in operational["models"].items():
            sections.append(
                f"| {model_name} | {decision['operational_conf']:.4f} | "
                f"{decision['f1']:.6f} | {decision['recall']:.6f} |"
            )
        sections.extend([
            "",
            "Candidates are scored using detections retained at evaluation_conf=0.01; "
            "ties resolve by higher recall, then lower threshold.",
            'Decision and evidence hashes: [operational_confidence.json](operational_confidence.json).', '',
        ])
    smoke_path = PILOT_DIR / "smoke_metadata.json"
    if smoke_path.is_file():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        sections.extend([
            "## CPU smoke (not decision-grade)",
            "",
            "Coordinate/class checks passed on 20 frames per model; 20 overlays per model "
            "were exported. These metrics are diagnostic only and must not select padding.",
            "",
            "| Model | Mode | mAP50 | Recall50 |",
            "|---|---|---:|---:|",
        ])
        for model, modes in smoke.get("accuracy_smoke_not_decision_grade", {}).items():
            for mode, values in modes.items():
                sections.append(
                    f'| {model} | {mode} | {values["mAP50"]:.4f} | {values["recall50"]:.4f} |'
                )
        sections.extend([
            "",
            f"Thresholds: evaluation_conf={smoke['evaluation_conf']}; "
            "operational_conf by model is recorded in smoke_metadata.json.",
            "",
            "| Model | Evaluation detections | Operational detections | Candidate segments |",
            "|---|---:|---:|---:|",
        ])
        for model_name, counts in smoke["detection_counts"].items():
            sections.append(
                f"| {model_name} | {counts['evaluation_detections']} | "
                f"{counts['operational_detections']} | {counts['candidate_segments']} |"
            )
        for model_name, model_results in smoke.get("accuracy_smoke_not_decision_grade", {}).items():
            if not model_name.startswith("roi_ssd"):
                continue
            if "FF" in model_results and "SEG" in model_results:
                ratio = (
                    model_results["SEG"]["mAP50"] / model_results["FF"]["mAP50"]
                    if model_results["FF"]["mAP50"] > 0 else None
                )
                if ratio is not None and ratio < 0.95:
                    sections.extend([
                        "",
                        f"**STOP warning:** {model_name} smoke segment/full-frame mAP50 ratio is "
                        f"{ratio:.3f}. The sample is too small for a conclusion, but the full "
                        "padding sweep must resolve this before CA/FULL-TECH is run.",
                    ])
        sections.append("")
    if decision_path.is_file():
        decisions = json.loads(decision_path.read_text(encoding="utf-8"))
        sweep_path = PILOT_DIR / "padding_sweep.csv"
        if sweep_path.is_file():
            with sweep_path.open(encoding="utf-8", newline="") as stream:
                sweep_rows = list(csv.DictReader(stream))
            sections.extend([
                "## Measured full-frame and padding results", "",
                "All entries use the locked pilot; image loading and metric computation are excluded from process-frame timing.", "",
                "| Model | Mode | Padding | evaluation_conf | operational_conf | mAP50 | mAP50:95 | Recall50 | GT coverage | Mean latency (ms) | FPS |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for row in sweep_rows:
                sections.append(
                    f'| {row["model"]} | {row["mode"]} | {row["padding"]} | '
                    f'{float(row["evaluation_conf"]):.4f} | {float(row["operational_conf"]):.4f} | '
                    f'{float(row["mAP50"]):.4f} | {float(row["mAP50_95"]):.4f} | '
                    f'{float(row["recall50"]):.4f} | {float(row["gt_roi_coverage_mean"]):.4f} | '
                    f'{float(row["latency_mean_ms"]):.2f} | {float(row["fps"]):.2f} |'
                )
            sections.extend(["", "Sources: [padding sweep](padding_sweep.csv), [frame-level metrics](frame_level_metrics.csv), [preflight](stage2_preflight.json).", ""])
        sections.extend([
            "## Locked padding decisions",
            "",
            "| Model | evaluation_conf | operational_conf | Final padding | Criterion met | Eligible values |",
            "|---|---:|---:|---:|:---:|---|",
        ])
        for name, value in decisions.items():
            sections.append(
                f'| {name} | {value["evaluation_conf"]:.4f} | '
                f'{value["operational_conf"]:.4f} | {value["padding_final"]} | '
                f'{value["criterion_met"]} | '
                f'{value["eligible_padding_values"]} |'
            )
        sections.append("")
    if profile_path.is_file():
        profiles = json.loads(profile_path.read_text(encoding="utf-8"))
        sections.extend([
            "## Fresh CPU cost profiles",
            "",
            "Timing mode: shared detector call at evaluation_conf; quality-only filtering/metrics are not timed.",
            "",
            "| Model | evaluation_conf | operational_conf | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name, value in profiles.items():
            sections.append(
                f'| {name} | {value["evaluation_conf"]:.4f} | '
                f'{value["operational_conf"]:.4f} | {value["K_ms"]:.6g} | '
                f'{value["c_ms_per_pixel"]:.6g} | '
                f'{value["tau_pixels"]:.3f} | {value["r_squared"]:.4f} | '
                f'{value["mae_ms"]:.4f} | {value["rmse_ms"]:.4f} |'
            )
        sections.append("")
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8", newline="") as stream:
            summaries = list(csv.DictReader(stream))
        sections.extend([
            "## FF / SEG / CA / FULL-TECH results",
            "",
            "| Model | Mode | evaluation_conf | operational_conf | mAP50 | mAP50:95 | Recall50 | total p50 ms | total p95 ms | inference p50 ms | inference p95 ms | ROI coverage | Tensor area ratio | Full-frame fraction | NN calls/frame |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in summaries:
            sections.append(
                f'| {row["model"]} | {row["mode"]} | '
                f'{float(row["evaluation_conf"]):.4f} | {float(row["operational_conf"]):.4f} | '
                f'{float(row["mAP50"]):.4f} | '
                f'{float(row["mAP50_95"]):.4f} | {float(row["recall50"]):.4f} | '
                f'{float(row["latency_p50_ms"]):.2f} | {float(row["latency_p95_ms"]):.2f} | '
                f'{float(row["inference_latency_p50_ms"]):.2f} | '
                f'{float(row["inference_latency_p95_ms"]):.2f} | '
                f'{float(row["gt_roi_coverage_mean"]):.4f} | '
                f'{float(row["tensor_area_ratio_mean"]):.4f} | '
                f'{float(row["full_frame_fraction"]):.4f} | '
                f'{float(row["nn_calls_per_frame"]):.3f} |'
            )
        sections.append("")
    decisions = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.is_file() else {}
    profiles = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else {}
    operational = operational or {"models": {}}
    op_values = {
        name: decision.get("operational_conf", "NOT SELECTED")
        for name, decision in operational["models"].items()
    }
    sections.extend([
        "## Frozen pilot parameters",
        "",
        f'YOLO_OPERATIONAL_CONF = {op_values.get("yolo26n", "NOT SELECTED")}',
        f'ROISSD_OPERATIONAL_CONF = {op_values.get("roi_ssd", "NOT SELECTED")}',
        f'ROISSD_MOBILENET_OPERATIONAL_CONF = {op_values.get("roi_ssd_mobilenet", "NOT SELECTED")}',
        "PADDING_EVALUATION_CONF = evaluation_conf (default 0.01)",
        "PIPELINE_OPERATIONAL_CONF = per-model frozen value above",
        f'YOLO_PADDING_FINAL = {decisions.get("yolo26n", {}).get("padding_final", "NOT RUN")}',
        f'ROISSD_PADDING_FINAL = {decisions.get("roi_ssd", {}).get("padding_final", "NOT RUN")}',
        f'ROISSD_MOBILENET_PADDING_FINAL = {decisions.get("roi_ssd_mobilenet", {}).get("padding_final", "NOT RUN")}',
        f'YOLO_CPU_K = {profiles.get("yolo26n", {}).get("K_ms", "NOT RUN")}',
        f'YOLO_CPU_c = {profiles.get("yolo26n", {}).get("c_ms_per_pixel", "NOT RUN")}',
        f'YOLO_CPU_tau = {profiles.get("yolo26n", {}).get("tau_pixels", "NOT RUN")}',
        f'ROISSD_CPU_K = {profiles.get("roi_ssd", {}).get("K_ms", "NOT RUN")}',
        f'ROISSD_CPU_c = {profiles.get("roi_ssd", {}).get("c_ms_per_pixel", "NOT RUN")}',
        f'ROISSD_CPU_tau = {profiles.get("roi_ssd", {}).get("tau_pixels", "NOT RUN")}',
        f'ROISSD_MOBILENET_CPU_K = {profiles.get("roi_ssd_mobilenet", {}).get("K_ms", "NOT RUN")}',
        f'ROISSD_MOBILENET_CPU_c = {profiles.get("roi_ssd_mobilenet", {}).get("c_ms_per_pixel", "NOT RUN")}',
        f'ROISSD_MOBILENET_CPU_tau = {profiles.get("roi_ssd_mobilenet", {}).get("tau_pixels", "NOT RUN")}',
        "",
    ])
    base = report.split("\n## Locked padding decisions", 1)[0]
    base = base.split("\n## CPU smoke (not decision-grade)", 1)[0]
    base = base.split("\n## Fresh CPU cost profiles", 1)[0]
    base = base.split("\n## FF / SEG / CA / FULL-TECH results", 1)[0]
    base = base.split("\n## Frozen pilot parameters", 1)[0]
    base = base.split("\n## Frozen operational confidence", 1)[0]
    base = base.split("\n## Measured full-frame and padding results", 1)[0]
    base = base.split("\n## Runtime environment", 1)[0]
    status = metadata.get("status", "PREPARED_NOT_BENCHMARKED")
    completed = metadata.get("completed_stages", [])
    prefix, after_status = base.split("## Preparation status", 1)
    _, after_locked_status = after_status.split("## Locked split", 1)
    base = (
        prefix
        + "## Preparation status\n\n"
        + f"`{status}`. Completed stages: {completed or 'none'}. "
        + ("The full padding sweep is complete. " if "padding_sweep" in completed else "The full padding sweep remains pending. ")
        + (
            "FF/SEG/CA/FULL-TECH pipeline runs are recorded.\n\n"
            if "pipeline_modes" in completed else
            "Pipeline summary exists, but the pipeline stage did not finish cleanly.\n\n"
            if summary_path.is_file() else
            "CA/FULL-TECH pipeline runs have not been completed.\n\n"
        )
        + ("**Execution stop:** " + metadata["stop_reason"] + "\n\n" if metadata.get("stop_reason") else "")
        + "## Locked split"
        + after_locked_status
    )
    base = base.split("\n## Remaining protocol", 1)[0]
    pending_steps = ["## Remaining protocol", ""]
    if not decision_path.is_file():
        pending_steps.append("1. Run `python -m experiments.cpu_pilot.run_pilot --stage padding`.")
    if not profile_path.is_file():
        pending_steps.append("2. Run `python -m experiments.cpu_pilot.run_pilot --stage cost`.")
    if decision_path.is_file() and profile_path.is_file() and not summary_path.is_file():
        if metadata.get("stop_reason"):
            pending_steps.append("Investigate failed segment-accuracy/coverage criteria for every model before rerunning CA/FULL-TECH. Do not relax thresholds or substitute checkpoints post hoc. Stage 4 / Raspberry Pi runs remain on hold.")
        else:
            pending_steps.append("Review padding eligibility/accuracy gate, then run `--stage pipeline`.")
    if summary_path.is_file():
        pending_steps.append("Review acceptance checks and final settings before Stage 4.")
    pending_steps.append("")
    base = base.rstrip() + "\n\n" + "\n".join(pending_steps)
    if sections:
        base = base.rstrip() + "\n\n" + "\n".join(sections).rstrip() + "\n"
    report_path.write_text(base, encoding="utf-8")


def main():
    global PILOT_DIR, SPLIT_METADATA, MODEL_NAMES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--stage", choices=("confidence", "smoke", "padding", "cost", "pipeline", "all"), required=True)
    parser.add_argument("--max-frames", type=int, default=None, help="Use only the first N pilot frames for smoke testing")
    parser.add_argument("--cost-repetitions", type=int, default=8)
    parser.add_argument(
        "--evaluation-conf",
        type=float,
        default=DEFAULT_EVALUATION_CONF,
        help="Evaluation detection floor for smoke, padding, cost, and pipeline (default: 0.01; confidence stage is fixed)",
    )
    parser.add_argument(
        "--operational-conf",
        type=float,
        default=None,
        help="Explicit operational threshold override; otherwise use the per-model frozen --stage confidence result",
    )
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--models", nargs="+", choices=tuple(ROI_GRID), default=DEFAULT_MODELS)
    args = parser.parse_args()
    args.evaluation_conf = _validate_confidence("evaluation_conf", args.evaluation_conf)
    if args.operational_conf is not None:
        args.operational_conf = _validate_confidence("operational_conf", args.operational_conf)
    MODEL_NAMES = tuple(dict.fromkeys(args.models))
    PILOT_DIR = args.output_dir.resolve()
    SPLIT_METADATA = PILOT_DIR / "split_metadata.json"
    if args.stage == "confidence" and args.max_frames is not None:
        raise ValueError("Confidence selection requires the full pilot")
    if args.stage == "confidence" and args.operational_conf is not None:
        raise ValueError("--operational-conf is selected by --stage confidence and cannot be overridden there")
    if args.stage == "confidence" and args.evaluation_conf != DEFAULT_EVALUATION_CONF:
        raise ValueError("--stage confidence requires evaluation_conf=0.01")
    if args.stage == "all" and args.max_frames is not None:
        raise ValueError("--stage all cannot be combined with --max-frames; partial runs are not pilot results")
    if args.max_frames is not None and args.max_frames < 20 and args.stage in {"pipeline", "all"} and not args.no_debug:
        raise ValueError("At least 20 frames are required when debug export is enabled")
    _configure_runtime()
    split, manifests = _load_session()
    args.operational_conf_by_model = {}
    if args.stage != "confidence":
        for model_name in MODEL_NAMES:
            config = yaml.safe_load(
                (PILOT_DIR / "configs" / f"{model_name}.yaml").read_text(encoding="utf-8")
            )
            args.operational_conf_by_model[model_name] = _resolve_operational_conf(
                model_name, config, args.evaluation_conf, args.operational_conf
            )
    _write_metadata(split, args.stage, {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    if args.stage == 'confidence':
        from .confidence import run_selection
        run_selection(
            split, manifests['pilot'], torch.device('cpu'),
            pilot_module=sys.modules[__name__],
        )
        metadata_path = PILOT_DIR / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frozen = json.loads(
            (PILOT_DIR / "operational_confidence.json").read_text(encoding="utf-8")
        )
        metadata["confidence_protocol"]["operational_conf_by_model"] = {
            name: decision["operational_conf"]
            for name, decision in frozen["models"].items()
        }
        write_json(metadata_path, metadata)
        _mark_stage('operational_confidence_selection', 'OPERATIONAL_CONFIDENCE_FROZEN')
        _refresh_report()
        return
    if args.stage == "smoke":
        smoke = run_smoke(
            split,
            manifests["pilot"],
            torch.device("cpu"),
            args.max_frames or 20,
            evaluation_conf=args.evaluation_conf,
            operational_conf=args.operational_conf,
        )
        _mark_stage("coordinate_smoke", "SMOKE_PASS_FULL_PILOT_PENDING")
        _refresh_report()
        print(json.dumps(smoke, indent=2))
        return
    if args.stage in {"padding", "all"}:
        partial = args.max_frames is not None
        run_padding_sweep(
            split, manifests["pilot"], torch.device("cpu"), args.max_frames, partial=partial,
            evaluation_conf=args.evaluation_conf,
            operational_conf=args.operational_conf,
        )
        _mark_stage(
            "padding_smoke" if partial else "padding_sweep",
            "PADDING_SMOKE_COMPLETE" if partial else "PADDING_SWEEP_COMPLETE",
        )
    if args.stage in {"cost", "all"}:
        run_cost_profiles(
            split, manifests["pilot"], torch.device("cpu"), args.cost_repetitions,
            evaluation_conf=args.evaluation_conf,
            operational_conf=args.operational_conf,
        )
        _mark_stage("cpu_cost_calibration", "CPU_COST_CALIBRATION_COMPLETE")
    if args.stage in {"pipeline", "all"}:
        try:
            run_pipeline_modes(
                split, manifests["pilot"], torch.device("cpu"), args.max_frames,
                debug=not args.no_debug,
                partial=args.max_frames is not None,
                evaluation_conf=args.evaluation_conf,
                operational_conf=args.operational_conf,
            )
        except Exception as error:
            path = PILOT_DIR / "metadata.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["status"] = "BLOCKED_PIPELINE_ACCEPTANCE" if str(error).startswith("STOP:") else "PIPELINE_ERROR"
            metadata["stop_reason"] = str(error)
            write_json(path, metadata)
            _refresh_report()
            raise
        status = "PILOT_RUNS_COMPLETE_PENDING_REVIEW" if args.max_frames is None else "PARTIAL_PIPELINE_SMOKE"
        _mark_stage("pipeline_modes", status)
    _refresh_report()


if __name__ == "__main__":
    main()
