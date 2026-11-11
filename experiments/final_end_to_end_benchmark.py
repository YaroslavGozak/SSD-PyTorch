"""Run the locked Stage 4 end-to-end video benchmark on one CPU platform.

From the repository root, run ``--stage all`` once on each target host with a
distinct device label, then run ``--stage aggregate`` after both host result
folders are available under the output root. A Raspberry Pi may use two timing
repeats; the same repeat count is applied to every mode and model on that host.
Use ``--dataset-yaml`` when the frozen dataset is mounted at a different path;
it changes only the dataset location, not the manifest or model configuration.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from experiments.cpu_pilot.portable_paths import (
    CHECKPOINT_RELATIVE_PATHS,
    CHECKPOINT_SHA256,
    cpu_model_fallback,
    dataset_root_path,
    dataset_yaml_path,
    env_path,
    portable_dataset_config,
    stage3_dir,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = env_path(
    "SSD_OUTPUT_ROOT", ROOT / "raspberry-outputs" / "stage4"
)
PILOT_DIR = stage3_dir(ROOT / "final_experiments" / "pilot_20261110")
PILOT_REPORT = PILOT_DIR / "pilot_report.md"
PILOT_MANIFEST = PILOT_DIR / "pilot_manifest.txt"
FINAL_MANIFEST = PILOT_DIR / "final_eval_manifest_candidate.txt"
AUDIT_MANIFEST = ROOT / "final_experiments" / "01_audit" / "val_manifest.jsonl"
DIAGNOSTIC_REPORT = ROOT / "final_experiments" / "02_diagnostics_128" / "diagnostic_report.md"

MODEL_NAMES = ("yolo26n", "roi_ssd_mobilenet")
MODEL_CONFIGS = {
    "yolo26n": {
        "pilot_name": "yolo26n",
        "checkpoint_sha256": CHECKPOINT_SHA256["yolo26n"],
        "config": PILOT_DIR / "configs" / "yolo26n.yaml",
    },
    "roi_ssd_mobilenet": {
        "pilot_name": "roi_ssd_mobilenet",
        "checkpoint_sha256": CHECKPOINT_SHA256["roi_ssd_mobilenet"],
        "config": PILOT_DIR / "configs" / "roi_ssd_mobilenet.yaml",
    },
}
MODES = ("FF", "SEG", "CA", "FULL-TECH")
EXPECTED_PILOT_SHA256 = "817f5b145b774dc91ee0fa8d271e02c11d71d70bb50ff13082a0ba9487aa63cb"
EXPECTED_FINAL_SHA256 = "4f36f3455ced1a344ef69329479cd09013934d71aafa8dc800d7f7e5ea6bf6d4"
EXPECTED_SOURCE_SHA256 = "a375ebf0a2fb4c4aaac8cb810f5fef0d3579ad3f283484c536bb8904e89b1ac0"
EXPECTED_FINAL_FRAMES = 4744
EXPECTED_FINAL_VIDEOS = 476
EXPECTED_PILOT_FRAMES = 320
EXPECTED_PILOT_VIDEOS = 32
QUALITY_CONFIDENCE = 0.01

FRAME_FIELDS = (
    "run_id", "model", "device", "video_id", "frame_id", "pipeline_mode",
    "used_full_frame", "fallback_reason", "num_segments_pre_merge",
    "num_segments_post_merge", "nn_calls", "processed_geometric_pixels",
    "processed_area_ratio", "tensor_pixels", "tensor_area_ratio",
    "formation_ms", "merge_ms", "preprocess_ms", "inference_ms",
    "postprocess_ms", "total_process_frame_ms",
)
TIMING_FIELDS = (
    "run_id", "model", "device", "mode", "repeat", "num_frames", "fps",
    "total_latency_ms", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms",
    "temperature_before_c", "temperature_after_c", "temperature_max_c",
    "cpu_frequency_before_mhz", "cpu_frequency_after_mhz", "throttling_status",
    "power_mode_before", "power_mode_after", "system_load_note",
)
SUMMARY_FIELDS = (
    "Model", "Device", "Mode", "FPS", "Speedup vs FF", "mAP50", "ΔmAP50",
    "mAP50:95", "Recall50", "Area ratio", "Tensor ratio",
    "NN calls/frame", "Full-frame fraction", "Mean latency ms",
    "Latency p50 ms", "Latency p95 ms", "FPS std", "Min run FPS", "Max run FPS",
)
FULLTECH_FIELDS = (
    "Model", "Device", "FF FPS", "FULL-TECH FPS", "Speedup",
    "FF mAP50", "FULL-TECH mAP50", "ΔmAP50", "FF Recall", "FULL-TECH Recall",
)
COST_FIELDS = (
    "model", "device", "K_ms", "c_ms_per_pixel", "tau_pixels", "r_squared",
    "mae_ms", "rmse_ms", "shapes", "repetitions_per_shape", "warmup_per_shape",
    "calibration_input", "software", "hardware", "runtime_signature",
)
TIMING_RUNS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _git_provenance() -> Dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip())
    return {"git_commit": commit, "git_dirty": dirty}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _required_inputs(dataset_yaml_override: Optional[Path] = None) -> Path:
    required = (
        PILOT_REPORT, PILOT_MANIFEST, FINAL_MANIFEST, AUDIT_MANIFEST,
        DIAGNOSTIC_REPORT,
        ROOT / "final_experiments" / "FINAL_EXPERIMENTS_MANIFEST.md",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Stage 4 lock input is missing: {missing[0]}")
    dataset_yaml_paths = set()
    for model_name, model_info in MODEL_CONFIGS.items():
        if not model_info["config"].is_file():
            raise FileNotFoundError(f"Frozen Stage 3 config is missing for {model_name}: {model_info['config']}")
        selection_path = _selection_path(model_name)
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"Frozen Stage 3 operational selection is missing for {model_name}: "
                f"{selection_path}"
            )
        config_text = model_info["config"].read_text(encoding="utf-8")
        dataset_match = re.search(
            r"^\s*yolo_dataset_yaml:\s*(.+?)\s*$", config_text, re.MULTILINE
        )
        if dataset_match is None:
            raise ValueError(f"Frozen config for {model_name} has no yolo_dataset_yaml path")
        dataset_yaml = dataset_yaml_path(
            Path(dataset_match.group(1).strip().strip("\"'")),
            dataset_yaml_override,
        )
        dataset_yaml_paths.add(dataset_yaml)
        if not dataset_yaml.is_file():
            raise FileNotFoundError(
                f"Stage 4 dataset YAML required by {model_name} does not exist: {dataset_yaml}. "
                "Make the frozen dataset available at this path before running the benchmark."
            )
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to validate the configured dataset YAML"
            ) from exc
        dataset_config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        selected_root = dataset_root_path(dataset_config.get("path"))
        if selected_root is None or not selected_root.is_dir():
            raise FileNotFoundError(f"Dataset root is missing: {selected_root}")
    if len(dataset_yaml_paths) != 1:
        raise ValueError(
            "The frozen model configs must reference the same validation dataset YAML"
        )
    return next(iter(dataset_yaml_paths))


def _validate_frozen_inputs(
    dataset_yaml_override: Optional[Path] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict]:
    dataset_yaml = _required_inputs(dataset_yaml_override)
    import yaml

    dataset_root = dataset_root_path(
        yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")).get("path")
    )
    if dataset_root is None or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root is missing: {dataset_root}")
    if _sha256(PILOT_MANIFEST) != EXPECTED_PILOT_SHA256:
        raise ValueError("Stage 3 pilot manifest checksum differs from the locked report")
    if _sha256(FINAL_MANIFEST) != EXPECTED_FINAL_SHA256:
        raise ValueError("Stage 3 final-evaluation candidate checksum differs from the locked report")

    pilot_rows = _read_tsv(PILOT_MANIFEST)
    final_rows = _read_tsv(FINAL_MANIFEST)
    source_sha256 = _sha256(AUDIT_MANIFEST)
    final_videos = {row["video_id"] for row in final_rows}
    pilot_videos = {row["video_id"] for row in pilot_rows}
    if len(pilot_rows) != EXPECTED_PILOT_FRAMES or len(pilot_videos) != EXPECTED_PILOT_VIDEOS:
        raise ValueError(
            f"Pilot manifest count mismatch: {len(pilot_rows)} frames, "
            f"{len(pilot_videos)} videos"
        )
    if len(final_rows) != EXPECTED_FINAL_FRAMES or len(final_videos) != EXPECTED_FINAL_VIDEOS:
        raise ValueError(
            f"Final manifest count mismatch: {len(final_rows)} frames, "
            f"{len(final_videos)} videos"
        )
    if pilot_videos & final_videos:
        raise ValueError("Pilot profiling videos overlap the frozen final-evaluation videos")

    source_rows = _read_jsonl(AUDIT_MANIFEST)
    source_by_key = {
        (str(row["video_id"]), int(row["frame_index"])): row for row in source_rows
    }
    source_order = {
        (str(row["video_id"]), int(row["frame_index"])): index
        for index, row in enumerate(source_rows)
    }
    final_positions = []
    seen = set()
    for row in final_rows:
        key = (str(row["video_id"]), int(row["frame_index"]))
        if key in seen:
            raise ValueError(f"Duplicate frame in final evaluation manifest: {key}")
        seen.add(key)
        source = source_by_key.get(key)
        if source is None or source["image"] != row["image"] or source["label"] != row["label"]:
            raise ValueError(f"Final manifest row is not in the Stage 3 source evaluation manifest: {key}")
        if json.loads(row["gt_json"]) != source["gt"]:
            raise ValueError(f"Final manifest GT differs from source manifest: {key}")
        final_positions.append(source_order[key])
    if final_positions != sorted(final_positions):
        raise ValueError("Final manifest ordering differs from the frozen source manifest")
    stage3_selections = {
        model_name: {
            **_read_stage3_selection(model_name),
            "selection_path": str(_selection_path(model_name).relative_to(ROOT)),
            "selection_sha256": _sha256(_selection_path(model_name)),
        }
        for model_name in MODEL_NAMES
    }
    return pilot_rows, final_rows, {
        "pilot_sha256": EXPECTED_PILOT_SHA256,
        "final_sha256": EXPECTED_FINAL_SHA256,
        "pilot_frames": len(pilot_rows),
        "final_frames": len(final_rows),
        "final_videos": len(final_videos),
        "pilot_final_video_overlap": 0,
        "source_audit_manifest_sha256": source_sha256,
        "source_audit_manifest_sha256_matches_documented": source_sha256 == EXPECTED_SOURCE_SHA256,
        "dataset_yaml": str(dataset_yaml),
        "dataset_yaml_sha256": _sha256(dataset_yaml),
        "dataset_root": str(dataset_root),
        **_git_provenance(),
        "stage3_selections": stage3_selections,
        "pilot_report_sha256": _sha256(PILOT_REPORT),
        "diagnostic_report_sha256": _sha256(DIAGNOSTIC_REPORT),
    }


def _read_locked_padding(model_name: str) -> int:
    text = PILOT_REPORT.read_text(encoding="utf-8")
    key = "YOLO_PADDING_FINAL" if model_name == "yolo26n" else "ROISSD_MOBILENET_PADDING_FINAL"
    match = re.search(rf"^{key}\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Pilot report does not contain locked {key}")
    padding = int(match.group(1))
    return padding


def _selection_path(model_name: str) -> Path:
    return PILOT_DIR / "calibration" / model_name / "selected_operational_config.json"


def _read_stage3_selection(model_name: str) -> Dict:
    selection_path = _selection_path(model_name)
    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing frozen Stage 3 selection: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected = {
        "status": "CALIBRATION_PASS",
        "model": model_name,
        "checkpoint_sha256": MODEL_CONFIGS[model_name]["checkpoint_sha256"],
        "pilot_manifest_sha256": EXPECTED_PILOT_SHA256,
        "config_sha256": _sha256(MODEL_CONFIGS[model_name]["config"]),
        "evaluation_conf": QUALITY_CONFIDENCE,
    }
    mismatches = {
        key: (selection.get(key), value)
        for key, value in expected.items()
        if selection.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Stage 3 operational selection for {model_name} does not match the "
            f"locked Stage 4 inputs: {mismatches}"
        )
    operational_conf = float(selection.get("operational_conf", float("nan")))
    if not math.isfinite(operational_conf) or not 0.0 <= operational_conf <= 1.0:
        raise ValueError(f"Invalid Stage 3 operational confidence for {model_name}")
    selected_padding = int(selection.get("padding", -1))
    if selected_padding != _read_locked_padding(model_name):
        raise ValueError(
            f"Stage 3 selected padding for {model_name} differs between the "
            "selection artifact and pilot report"
        )
    return selection


def _operational_confidence(model_name: str) -> float:
    return float(_read_stage3_selection(model_name)["operational_conf"])


def _device_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not value:
        raise ValueError("Device label must contain letters or digits")
    return value


def _runtime_signature(runtime: Dict) -> str:
    canonical = json.dumps(runtime, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_device_label(device_label: str, runtime: Dict) -> None:
    normalized = device_label.lower().replace("-", "_")
    cpu = str(runtime.get("cpu_model", "")).lower()
    if "11700k" in normalized:
        raise ValueError(
            f"Device label {device_label!r} names an outdated Stage 4 target; "
            "the required desktop target is Intel Core i9-9900KF"
        )
    if ("i9" in normalized or "9900kf" in normalized) and "9900kf" not in cpu:
        raise ValueError(
            f"Device label {device_label!r} requires an Intel Core i9-9900KF; "
            f"detected CPU is {runtime.get('cpu_model')!r}"
        )
    if "raspberry" in normalized or re.search(r"(^|_)pi(_|5|$)", normalized):
        model_path = Path("/proc/device-tree/model")
        board = model_path.read_text(encoding="utf-8", errors="replace").strip("\x00 \r\n") \
            if model_path.is_file() else ""
        if "raspberry pi 5" not in board.lower():
            raise ValueError(
                f"Device label {device_label!r} requires Raspberry Pi 5 hardware; "
                f"detected board model is {board or 'unavailable'}"
            )


def _load_pilot_module():
    from experiments.cpu_pilot import run_pilot as pilot

    pilot.PILOT_DIR = PILOT_DIR
    pilot.SPLIT_METADATA = PILOT_DIR / "split_metadata.json"
    pilot._cpu_model = cpu_model_fallback
    pilot._configure_runtime()
    return pilot


class HardwareMonitor:
    def __init__(self):
        self._stop = threading.Event()
        self._samples = []
        self._thread = None

    def _sample(self):
        from experiments.cost_model_validation.common import read_cpu_freq, read_cpu_temp
        from tools.benchmarks.video_artifacts import hardware_snapshot

        while not self._stop.wait(2.0):
            snap = hardware_snapshot()
            snap["cpu_freq_mhz"] = read_cpu_freq()
            snap["cpu_temp_c"] = read_cpu_temp()
            self._samples.append(snap)

    def start(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def samples(self):
        return list(self._samples)


def _hardware_snapshot() -> Dict:
    from experiments.cost_model_validation.common import read_cpu_freq, read_cpu_temp, read_power_mode
    from tools.benchmarks.video_artifacts import hardware_snapshot

    result = hardware_snapshot()
    result["cpu_freq_mhz"] = read_cpu_freq()
    result["cpu_temp_c"] = read_cpu_temp()
    result["power_mode"] = read_power_mode()
    return result


def _monitor_summary(before: Dict, after: Dict, samples: List[Dict], note: str) -> Dict:
    temperatures = [
        item["cpu_temp_c"] for item in [before, *samples, after]
        if item.get("cpu_temp_c") is not None
    ]
    throttling = sorted({
        str(item["throttling"]) for item in [before, *samples, after]
        if item.get("throttling") is not None
    })
    return {
        "temperature_before_c": before.get("cpu_temp_c"),
        "temperature_after_c": after.get("cpu_temp_c"),
        "temperature_max_c": max(temperatures) if temperatures else None,
        "cpu_frequency_before_mhz": before.get("cpu_freq_mhz"),
        "cpu_frequency_after_mhz": after.get("cpu_freq_mhz"),
        "throttling_status": "|".join(throttling),
        "power_mode_before": json.dumps(before.get("power_mode"), sort_keys=True),
        "power_mode_after": json.dumps(after.get("power_mode"), sort_keys=True),
        "system_load_note": note,
        "monitor_samples": samples,
    }


def _load_model_and_dataset(
    pilot, model_name: str, dataset_yaml_override: Optional[Path] = None
):
    import torch
    import yaml

    model_info = MODEL_CONFIGS[model_name]
    padding = _read_locked_padding(model_name)
    checkpoint_path = ROOT / CHECKPOINT_RELATIVE_PATHS[model_name]
    checkpoint_hash = CHECKPOINT_SHA256[model_name]
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint_hash:
        raise ValueError(f"Selected checkpoint is missing or has a checksum mismatch for {model_name}")
    manifest_text = (ROOT / "final_experiments" / "FINAL_EXPERIMENTS_MANIFEST.md").read_text(
        encoding="utf-8"
    )
    checkpoint_relative_path = checkpoint_path.relative_to(ROOT).as_posix()
    if checkpoint_hash not in manifest_text or checkpoint_relative_path not in manifest_text:
        raise ValueError(f"Checkpoint selection for {model_name} differs from the canonical manifest")
    source_config = yaml.safe_load(
        model_info["config"].read_text(encoding="utf-8")
    )
    selected_dataset_yaml = dataset_yaml_path(
        source_config["dataset_params"]["yolo_dataset_yaml"],
        dataset_yaml_override,
    )
    if not selected_dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML is missing: {selected_dataset_yaml}")
    dataset_config = yaml.safe_load(selected_dataset_yaml.read_text(encoding="utf-8"))
    selected_dataset_root = dataset_root_path(dataset_config.get("path"))
    if selected_dataset_root is None or not selected_dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root is missing: {selected_dataset_root}")
    dataset_config = portable_dataset_config(dataset_config, selected_dataset_root)

    train_config = source_config["train_params"]
    if model_name == "yolo26n":
        train_config["yolo_weights"] = str(checkpoint_path)
    else:
        train_config["task_name"] = str(checkpoint_path.parent)
        train_config["ckpt_name"] = checkpoint_path.name

    with tempfile.TemporaryDirectory(prefix="stage4-portable-") as temp_dir:
        temp_dir = Path(temp_dir)
        runtime_yaml = temp_dir / selected_dataset_yaml.name
        runtime_yaml.write_text(
            yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8"
        )
        source_config["dataset_params"]["yolo_dataset_yaml"] = str(runtime_yaml)
        runtime_config_dir = temp_dir / "configs"
        runtime_config_dir.mkdir()
        runtime_config = runtime_config_dir / model_info["config"].name
        runtime_config.write_text(
            yaml.safe_dump(source_config, sort_keys=False), encoding="utf-8"
        )
        original_pilot_dir = pilot.PILOT_DIR
        pilot.PILOT_DIR = temp_dir
        try:
            model, dataset, config, evaluation_conf = pilot._load_model(
                model_info["pilot_name"], torch.device("cpu"),
                evaluation_conf=QUALITY_CONFIDENCE,
            )
        finally:
            pilot.PILOT_DIR = original_pilot_dir
    config["dataset_params"]["yolo_dataset_yaml"] = str(selected_dataset_yaml)
    config["train_params"] = yaml.safe_load(
        model_info["config"].read_text(encoding="utf-8")
    )["train_params"]
    if float(config["train_params"].get("infer_conf_threshold", -1)) != QUALITY_CONFIDENCE:
        raise ValueError(f"Frozen evaluation confidence changed for {model_name}")
    _read_stage3_selection(model_name)
    expected_family = "yolo" if model_name == "yolo26n" else "roissd-mobilenet"
    if config["train_params"].get("model") != expected_family:
        raise ValueError(f"Frozen model family changed for {model_name}")
    if set(dataset.idx2label) != set(range(31)):
        raise ValueError(f"Expected 30 foreground classes plus background for {model_name}")
    return model, dataset, config, evaluation_conf, padding


def _indices_for_manifest(pilot, dataset, config, rows):
    root = dataset_root_path()
    if root is None:
        root = pilot._dataset_root(config)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is missing: {root}")
    indices = pilot._dataset_indices(dataset, rows, root)
    if len(indices) != len(rows):
        raise ValueError("Manifest-to-dataset mapping did not preserve every frame")
    return indices


def _check_manifest_files(pilot, config, rows) -> None:
    import yaml

    dataset_root = dataset_root_path()
    if dataset_root is None:
        dataset_root = pilot._dataset_root(config)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root is missing: {dataset_root}")
    source_by_key = {
        (str(row["video_id"]), int(row["frame_index"])): row
        for row in _read_jsonl(AUDIT_MANIFEST)
    }
    for row in rows:
        image_path = dataset_root / row["image"]
        label_path = dataset_root / row["label"]
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Final manifest image/label is missing: {image_path} / {label_path}")
        source = source_by_key[(str(row["video_id"]), int(row["frame_index"]))]
        expected_hash = source["label_sha256"]
        if _sha256(label_path) != expected_hash:
            raise ValueError(f"Final manifest label checksum mismatch: {label_path}")
    dataset_yaml = Path(config["dataset_params"]["yolo_dataset_yaml"])
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML referenced by frozen config is missing: {dataset_yaml}")
    parsed = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not parsed.get("val"):
        raise ValueError("Dataset YAML has no validation split")


def _prepare_device_directory(output_root: Path, device_label: str,
                              final_rows: List[Dict], manifest_info: Dict) -> Path:
    device_dir = output_root / _device_slug(device_label)
    device_dir.mkdir(parents=True, exist_ok=True)
    manifest_copy = device_dir / "final_eval_manifest_candidate.txt"
    if manifest_copy.exists() and _sha256(manifest_copy) != EXPECTED_FINAL_SHA256:
        raise FileExistsError(f"Existing output manifest does not match frozen input: {manifest_copy}")
    if not manifest_copy.exists():
        shutil.copy2(FINAL_MANIFEST, manifest_copy)
    _write_json(device_dir / "manifest_provenance.json", manifest_info)
    configs_dir = device_dir / "config_snapshots"
    configs_dir.mkdir(exist_ok=True)
    for model_name, info in MODEL_CONFIGS.items():
        source = info["config"]
        destination = configs_dir / f"{model_name}.yaml"
        if destination.exists() and _sha256(destination) != _sha256(source):
            raise FileExistsError(f"Existing config snapshot differs from frozen config: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
    return device_dir


def _cost_profile_path(device_dir: Path) -> Path:
    return device_dir / "cost_profiles_final.json"


def _profile_models(pilot, device_dir: Path, device_label: str,
                    pilot_rows: List[Dict[str, str]], model_names: Sequence[str],
                    repetitions: int, dataset_yaml_override: Optional[Path] = None) -> List[Dict]:
    import torch

    profile_path = _cost_profile_path(device_dir)
    if profile_path.exists():
        existing_profiles = json.loads(profile_path.read_text(encoding="utf-8")).get("profiles")
        if existing_profiles:
            raise FileExistsError(
                f"Refusing to overwrite frozen per-device cost profiles: {profile_path}; "
                "use a fresh output directory to recalibrate"
            )
    profile_records = []
    raw_records = []
    for model_name in model_names:
        try:
            model, dataset, config, evaluation_conf, padding = _load_model_and_dataset(
                pilot, model_name, dataset_yaml_override
            )
            indices = _indices_for_manifest(pilot, dataset, config, pilot_rows)
            runtime = pilot._runtime_environment()
            _validate_device_label(device_label, runtime)
            before = _hardware_snapshot()
            class_list = [dataset.idx2label[index] for index in sorted(dataset.idx2label)]
            classes_path = device_dir / "class_list.json"
            if classes_path.exists():
                if json.loads(classes_path.read_text(encoding="utf-8")) != class_list:
                    raise ValueError("Selected models do not use the same frozen class ordering")
            else:
                _write_json(classes_path, class_list)
            fit, shape_rows, raw = pilot._profile_cost(
                model_name, model, dataset, indices, repetitions=repetitions
            )
            after = _hardware_snapshot()
            checkpoint = pilot.MODEL_CHECKPOINTS[model_name][1]
            record = {
                **fit,
                "model": model_name,
                "device": device_label,
                "torch_device": "cpu",
                "padding": padding,
                "operational_conf": _operational_confidence(model_name),
                "stage3_selection_sha256": _sha256(_selection_path(model_name)),
                "evaluation_conf": evaluation_conf,
                "checkpoint_sha256": checkpoint,
                "config_sha256": _sha256(MODEL_CONFIGS[model_name]["config"]),
                "profiling_manifest_sha256": EXPECTED_PILOT_SHA256,
                "profiling_manifest": _display_path(PILOT_MANIFEST),
                "shapes": [{"height": hw[0], "width": hw[1], "pixels": hw[0] * hw[1],
                            "median_ms": median, "p95_ms": p95}
                           for hw, median, p95 in shape_rows],
                "class_list": class_list,
                "runtime_environment": runtime,
                "runtime_signature": _runtime_signature(runtime),
                "hardware_before": before,
                "hardware_after": after,
                "software": {
                    "python": sys.version,
                    "torch": str(torch.__version__),
                    "numpy": str(np_version()),
                },
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            if record["K_ms"] <= 0 or record["c_ms_per_pixel"] <= 0:
                raise ValueError(f"Non-positive affine calibration coefficients for {model_name}")
            profile_records.append(record)
            for item in raw:
                raw_records.append({**item, "device": device_label,
                                    "profiling_manifest_sha256": EXPECTED_PILOT_SHA256})
            del model, dataset
        except Exception as exc:
            _error_record(device_dir, model_name, "cost_calibration", exc)
            print(f"FAILED cost calibration for {model_name}: {exc}", flush=True)

    _write_json(profile_path, {"profiles": profile_records})
    _write_csv(device_dir / "cost_calibration_measurements.csv", [
        "model", "device", "requested_h", "requested_w", "actual_network_h",
        "actual_network_w", "repetition", "latency_ms", "profiling_manifest_sha256",
    ], raw_records)
    _write_cost_csv(device_dir / "cost_profiles_final.csv", profile_records)
    return profile_records


def np_version():
    import numpy

    return numpy.__version__


def _write_cost_csv(path: Path, profiles: Sequence[Dict]) -> None:
    rows = []
    for record in profiles:
        rows.append({
            "model": record["model"], "device": record["device"],
            "K_ms": record["K_ms"], "c_ms_per_pixel": record["c_ms_per_pixel"],
            "tau_pixels": record["tau_pixels"], "r_squared": record["r_squared"],
            "mae_ms": record["mae_ms"], "rmse_ms": record["rmse_ms"],
            "shapes": json.dumps(record["shapes"], separators=(",", ":")),
            "repetitions_per_shape": record["repetitions_per_shape"],
            "warmup_per_shape": record["warmup_per_shape"],
            "calibration_input": record["profiling_manifest"],
            "software": json.dumps(record["software"], sort_keys=True),
            "hardware": json.dumps({
                "runtime": record["runtime_environment"],
                "before": record["hardware_before"], "after": record["hardware_after"],
            }, sort_keys=True),
            "runtime_signature": record["runtime_signature"],
        })
    _write_csv(path, COST_FIELDS, rows)


def _load_profiles(device_dir: Path, device_label: str, pilot) -> Dict[str, Dict]:
    path = _cost_profile_path(device_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Run --stage calibrate first; missing {path}")
    records = json.loads(path.read_text(encoding="utf-8"))["profiles"]
    by_model = {}
    for profile in records:
        model_name = profile.get("model")
        if model_name not in MODEL_NAMES or model_name in by_model:
            raise ValueError(f"Cost profile file contains an unknown or duplicate model: {model_name}")
        config_path = MODEL_CONFIGS[model_name]["config"]
        expected_checkpoint = pilot.MODEL_CHECKPOINTS[model_name][1]
        expected = {
            "device": device_label,
            "torch_device": "cpu",
            "checkpoint_sha256": expected_checkpoint,
            "config_sha256": _sha256(config_path),
            "profiling_manifest_sha256": EXPECTED_PILOT_SHA256,
            "runtime_signature": _runtime_signature(pilot._runtime_environment()),
            "operational_conf": _operational_confidence(model_name),
            "stage3_selection_sha256": _sha256(_selection_path(model_name)),
        }
        mismatches = [key for key, value in expected.items() if profile.get(key) != value]
        if profile.get("hardware_before", {}).get("power_mode") != _hardware_snapshot().get("power_mode"):
            mismatches.append("power_mode")
        if mismatches:
            error = ValueError(
                f"Frozen {model_name} cost profile mismatches current run: {mismatches}"
            )
            _error_record(device_dir, model_name, "profile_validation", error)
            print(f"REJECTED {model_name} cost profile: {error}", flush=True)
            continue
        by_model[model_name] = profile
    return by_model


def _per_class_metrics(rows: Sequence[Dict]) -> List[Dict]:
    import json
    from tools.infer import compute_map

    predictions = []
    ground_truths = []
    for row in rows:
        detections = json.loads(row["prediction_records"])
        pred = {}
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            pred.setdefault(str(detection["class"]), []).append([
                float(x1), float(y1), float(x2), float(y2),
                float(detection["confidence"]),
            ])
        predictions.append(pred)
        ground_truths.append(json.loads(row["ground_truth_records"]))
    difficult = [
        {label: [False] * len(boxes) for label, boxes in image_ground_truth.items()}
        for image_ground_truth in ground_truths
    ]
    _, ap50, _, _ = compute_map(predictions, ground_truths, iou_threshold=0.5, difficult=difficult)
    _, ap95, _, _ = compute_map(predictions, ground_truths, iou_threshold=0.95, difficult=difficult)
    classes = sorted(set(ap50) | set(ap95))
    return [
        {"class": str(name), "AP50": ap50.get(name), "AP95": ap95.get(name)}
        for name in classes
    ]


def _frame_log_row(raw: Dict, model_name: str, device_label: str, mode: str,
                   repeat: int, run_id: str) -> Dict:
    return {
        "run_id": run_id,
        "model": model_name,
        "device": device_label,
        "video_id": raw["video_id"],
        "frame_id": raw["frame_index"],
        "pipeline_mode": mode,
        "used_full_frame": raw["input_mode"] == "full",
        "fallback_reason": raw.get("fallback_reason", ""),
        "num_segments_pre_merge": raw["candidate_segments"],
        "num_segments_post_merge": raw["segments_after_merge"],
        "nn_calls": raw["nn_calls"],
        "processed_geometric_pixels": raw["geometric_processed_area"],
        "processed_area_ratio": raw["geometric_processed_area_ratio"],
        "tensor_pixels": raw["tensor_pixels"],
        "tensor_area_ratio": raw["tensor_area_ratio"],
        "formation_ms": raw["tracker_formation_time_s"] * 1000.0,
        "merge_ms": raw["merge_decision_time_s"] * 1000.0,
        "preprocess_ms": raw["crop_preprocess_time_s"] * 1000.0,
        "inference_ms": raw["nn_time_s"] * 1000.0,
        "postprocess_ms": raw["postprocess_time_s"] * 1000.0,
        "total_process_frame_ms": raw["total_process_frame_time_s"] * 1000.0,
    }


def _error_record(device_dir: Path, model: str, mode: str, exc: BaseException) -> None:
    path = device_dir / "errors.jsonl"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "mode": mode,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _run_benchmarks(pilot, device_dir: Path, device_label: str,
                    final_rows: List[Dict[str, str]], model_names: Sequence[str],
                    repeats: int, system_load_note: str,
                    dataset_yaml_override: Optional[Path] = None) -> bool:
    from experiments.cpu_pilot import run_pilot as pilot_module

    output_files = (
        "final_summary.csv", "final_fulltech_vs_fullframe.csv",
        "frame_level_metrics.csv", "timing_runs.csv",
    )
    existing = [device_dir / name for name in output_files if (device_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Benchmark outputs already exist: {existing[0]}; use a fresh output directory")

    profiles = _load_profiles(device_dir, device_label, pilot)
    frame_records = []
    run_records = []
    summaries = []
    class_records = []
    for model_name in model_names:
        if model_name not in profiles:
            error = RuntimeError(f"No valid frozen cost profile exists for {model_name}")
            _error_record(device_dir, model_name, "model_setup", error)
            print(f"SKIPPED {model_name}: {error}", flush=True)
            continue
        try:
            model, dataset, config, evaluation_conf, padding = _load_model_and_dataset(
                pilot, model_name, dataset_yaml_override
            )
            _check_manifest_files(pilot, config, final_rows)
            indices = _indices_for_manifest(pilot, dataset, config, final_rows)
            coefficients = {
                "K_ms": profiles[model_name]["K_ms"],
                "c_ms_per_pixel": profiles[model_name]["c_ms_per_pixel"],
            }
            class_path = device_dir / "class_list.json"
            class_list = [dataset.idx2label[index] for index in sorted(dataset.idx2label)]
            if class_path.exists():
                recorded_classes = json.loads(class_path.read_text(encoding="utf-8"))
                if recorded_classes != class_list:
                    raise ValueError("Selected models do not use the same frozen class ordering")
            else:
                _write_json(class_path, class_list)
            model_summaries = []
            for repeat in range(1, repeats + 1):
                for mode in MODES:
                    run_id = f"{device_label}_{model_name}_{mode}_r{repeat}_{int(time.time())}"
                    before = _hardware_snapshot()
                    monitor = HardwareMonitor()
                    monitor.start()
                    try:
                        summary, rows = pilot_module._run_pipeline(
                            model_name=model_name,
                            model=model,
                            dataset=dataset,
                            config=config,
                            evaluation_conf=evaluation_conf,
                            operational_conf=_operational_confidence(model_name),
                            manifest_rows=final_rows,
                            indices=indices,
                            run_name=run_id,
                            mode=mode,
                            padding=padding,
                            coefficients=coefficients,
                            debug=False,
                        )
                    except Exception as exc:
                        monitor.stop()
                        after = _hardware_snapshot()
                        _error_record(device_dir, model_name, mode, exc)
                        _write_json(device_dir / f"crash_{model_name}_{mode}_r{repeat}.json", {
                            "device": device_label, "model": model_name, "mode": mode,
                            "repeat": repeat, "exception": repr(exc),
                            "traceback": traceback.format_exc(),
                            "hardware_before": before, "hardware_after": after,
                            "monitor_samples": monitor.samples,
                        })
                        print(f"FAILED {model_name}/{mode}/repeat {repeat}: {exc}", flush=True)
                        continue
                    finally:
                        if monitor._thread is not None and monitor._thread.is_alive():
                            monitor.stop()
                    after = _hardware_snapshot()
                    hardware = _monitor_summary(before, after, monitor.samples, system_load_note)
                    for row in rows:
                        frame_records.append(_frame_log_row(
                            row, model_name, device_label, mode, repeat, run_id
                        ))
                    timing = {
                        "run_id": run_id,
                        "model": model_name,
                        "device": device_label,
                        "mode": mode,
                        "repeat": repeat,
                        "num_frames": summary["num_frames"],
                        "fps": summary["fps"],
                        "total_latency_ms": summary["latency_total_s"] * 1000.0,
                        "mean_latency_ms": summary["latency_mean_ms"],
                        "p50_latency_ms": summary["latency_p50_ms"],
                        "p95_latency_ms": summary["latency_p95_ms"],
                        "temperature_before_c": hardware["temperature_before_c"],
                        "temperature_after_c": hardware["temperature_after_c"],
                        "temperature_max_c": hardware["temperature_max_c"],
                        "cpu_frequency_before_mhz": hardware["cpu_frequency_before_mhz"],
                        "cpu_frequency_after_mhz": hardware["cpu_frequency_after_mhz"],
                        "throttling_status": hardware["throttling_status"],
                        "power_mode_before": hardware["power_mode_before"],
                        "power_mode_after": hardware["power_mode_after"],
                        "system_load_note": system_load_note,
                    }
                    run_records.append(timing)
                    model_summaries.append({**summary, "device": device_label, "repeat": repeat})
                    if repeat == 1:
                        try:
                            for item in _per_class_metrics(rows):
                                class_records.append({
                                    "model": model_name, "device": device_label,
                                    "mode": mode, **item,
                                })
                        except Exception as exc:
                            _error_record(device_dir, model_name, f"{mode}_per_class_metrics", exc)
                            print(
                                f"FAILED per-class metrics for {model_name}/{mode}: {exc}",
                                flush=True,
                            )
                    print(
                        f"Completed {model_name}/{mode}/repeat {repeat}: "
                        f"FPS={summary['fps']:.4f}, mAP50={summary['mAP50']:.4f}",
                        flush=True,
                    )
            summaries.extend(model_summaries)
            del model, dataset
        except Exception as exc:
            _error_record(device_dir, model_name, "model_setup", exc)
            print(f"FAILED to initialize {model_name}: {exc}", flush=True)

    if not run_records:
        print(f"No benchmark runs completed; inspect crash evidence in {device_dir}", flush=True)
        return False
    _write_csv(device_dir / "frame_level_metrics.csv", FRAME_FIELDS, frame_records)
    _write_csv(device_dir / "timing_runs.csv", TIMING_FIELDS, run_records)
    _write_csv(device_dir / "per_class_metrics.csv",
               ("model", "device", "mode", "class", "AP50", "AP95"), class_records)
    final_summary = _aggregate_summary(summaries, repeats)
    _write_csv(device_dir / "final_summary.csv", SUMMARY_FIELDS, final_summary)
    fulltech = _fulltech_summary(final_summary)
    _write_csv(device_dir / "final_fulltech_vs_fullframe.csv", FULLTECH_FIELDS, fulltech)
    _write_report(device_dir, final_summary, fulltech, list(profiles.values()), repeats)
    _write_environment(device_dir, pilot, device_label, system_load_note, run_records)
    _make_plots(device_dir, final_summary, frame_records)
    counts = {
        model_name: {
            mode: sum(1 for row in run_records if row["model"] == model_name and row["mode"] == mode)
            for mode in MODES
        }
        for model_name in model_names
    }
    incomplete = {
        model_name: modes for model_name, modes in counts.items()
        if any(count != repeats for count in modes.values())
    }
    if incomplete:
        print(f"Incomplete benchmark combinations (required {repeats} repeats/mode): {incomplete}",
              flush=True)
    return not incomplete


def _aggregate_summary(raw_summaries: Sequence[Dict], required_repeats: int) -> List[Dict]:
    grouped = {}
    for item in raw_summaries:
        grouped.setdefault((item["model"], item["device"], item["mode"]), []).append(item)
    output = []
    for model in MODEL_NAMES:
        for device in sorted({key[1] for key in grouped}):
            full = grouped.get((model, device, "FF"), [])
            full_fps = (
                statistics.mean(float(row["fps"]) for row in full)
                if len(full) == required_repeats else None
            )
            for mode in MODES:
                runs = grouped.get((model, device, mode), [])
                if len(runs) != required_repeats:
                    continue
                fps_values = [float(row["fps"]) for row in runs]
                mean_fps = statistics.mean(fps_values)
                quality = next((row for row in runs if int(row["repeat"]) == 1), runs[0])
                output.append({
                    "Model": model,
                    "Device": device,
                    "Mode": mode,
                    "FPS": mean_fps,
                    "Speedup vs FF": (
                        None if full_fps is None or full_fps <= 0
                        else 1.0 if mode == "FF"
                        else mean_fps / full_fps
                    ),
                    "mAP50": quality["mAP50"],
                    "ΔmAP50": (
                        quality["mAP50"] - full[0]["mAP50"]
                        if full else None
                    ),
                    "mAP50:95": quality["mAP50_95"],
                    "Recall50": quality["recall50"],
                    "Area ratio": quality["processed_area_ratio_mean"],
                    "Tensor ratio": quality["tensor_area_ratio_mean"],
                    "NN calls/frame": quality["nn_calls_per_frame"],
                    "Full-frame fraction": quality["full_frame_fraction"],
                    "Mean latency ms": statistics.mean(float(row["latency_mean_ms"]) for row in runs),
                    "Latency p50 ms": statistics.mean(float(row["latency_p50_ms"]) for row in runs),
                    "Latency p95 ms": statistics.mean(float(row["latency_p95_ms"]) for row in runs),
                    "FPS std": statistics.stdev(fps_values) if len(fps_values) > 1 else None,
                    "Min run FPS": min(fps_values),
                    "Max run FPS": max(fps_values),
                    "_completed_repeats": len(runs),
                    "_required_repeats": required_repeats,
                })
    return output


def _recompute_summary_speedups(summary: List[Dict]) -> None:
    full_fps = {
        (row["Model"], row["Device"]): float(row["FPS"])
        for row in summary
        if row["Mode"] == "FF" and row.get("FPS") not in (None, "")
    }
    for row in summary:
        baseline = full_fps.get((row["Model"], row["Device"]))
        if baseline is None or baseline <= 0:
            row["Speedup vs FF"] = None
        elif row["Mode"] == "FF":
            row["Speedup vs FF"] = 1.0
        else:
            row["Speedup vs FF"] = float(row["FPS"]) / baseline


def _fulltech_summary(summary: Sequence[Dict]) -> List[Dict]:
    by_key = {(row["Model"], row["Device"], row["Mode"]): row for row in summary}
    output = []
    for model, device, _mode in sorted(by_key):
        if _mode != "FF":
            continue
        full = by_key[(model, device, "FF")]
        tech = by_key.get((model, device, "FULL-TECH"))
        if tech is None:
            continue
        output.append({
            "Model": model,
            "Device": device,
            "FF FPS": full["FPS"],
            "FULL-TECH FPS": tech["FPS"],
            "Speedup": tech["FPS"] / full["FPS"] if full["FPS"] else None,
            "FF mAP50": full["mAP50"],
            "FULL-TECH mAP50": tech["mAP50"],
            "ΔmAP50": tech["mAP50"] - full["mAP50"],
            "FF Recall": full["Recall50"],
            "FULL-TECH Recall": tech["Recall50"],
        })
    return output


def _write_report(directory: Path, summary: Sequence[Dict], fulltech: Sequence[Dict],
                  profiles: Sequence[Dict], repeats: int) -> None:
    provenance_path = directory / "manifest_provenance.json"
    if not provenance_path.is_file():
        provenance_path = directory / "final_eval_manifest_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file() else {}
    )
    if provenance.get("dataset_yaml"):
        dataset_line = (
            f"- Dataset YAML: `{provenance['dataset_yaml']}` "
            f"(SHA-256 `{provenance['dataset_yaml_sha256']}`)."
        )
    elif provenance.get("device_dataset_yamls"):
        dataset_line = (
            "- Dataset YAMLs by device: `"
            + json.dumps(provenance["device_dataset_yamls"], sort_keys=True)
            + "`."
        )
    else:
        dataset_line = "- Dataset YAML location and checksum are recorded in manifest provenance."
    lines = [
        "# Final end-to-end benchmark",
        "",
        "## 1. Dataset and compatibility",
        f"- Final-evaluation manifest: `{_display_path(FINAL_MANIFEST)}`",
        f"- SHA-256: `{EXPECTED_FINAL_SHA256}`; 4,744 ordered frames from 476 videos.",
        dataset_line,
        "- Stage 3 pilot frames are checked for video-level disjointness from final frames.",
        "- The source validation split was previously used for model selection/diagnostics; it is not an untouched external test set.",
        "- Stage 3 reported `PIPELINE_ERROR` before CA/FULL-TECH completion; this runner executes the frozen modes and records any failures without replacing them with placeholder results.",
        "",
        "## 2. Selected models",
        "- YOLO26n and ROI-SSD-MobileNet use the Stage 3 selected checkpoints and config snapshots.",
        "",
        "## 3. Fixed pipeline configuration",
        "- Static/previous localization; hold=0; keyframe interval=10; empty segments fall back to full frame.",
        "- Frozen Stage 3 padding: YOLO26n=64, ROI-SSD-MobileNet=128.",
        "- Evaluation confidence=0.01; operational confidence is read from the hash-validated Stage 3 calibration selections.",
        *[
            f"- Stage 3 operational confidence ({model_name}): "
            f"{selection['operational_conf']} (selection SHA-256 "
            f"`{selection['selection_sha256']}`)."
            for model_name, selection in provenance.get("stage3_selections", {}).items()
        ],
        "- No final-set parameter tuning. Online/adaptive tau is disabled.",
        "",
        "## 4. Device-specific cost calibration",
        "- Each profile is freshly measured on this device using the separate Stage 3 pilot manifest and actual observed network tensor shapes.",
        "- The affine rule is T(A)=K+cA, with tau=K/c; coefficients remain frozen for all final runs.",
        "",
        "| Model | Device | Operational conf | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) | Shapes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if provenance.get("source_audit_manifest_sha256_matches_documented") is False:
        lines[8:8] = [
            "- WARNING: the current source audit-manifest checksum differs from the checksum recorded in the Stage 1 manifest; the frozen Stage 3 candidate checksum is verified, and every candidate row was cross-checked against current source content.",
            "",
        ]
    for row in profiles:
        lines.append(
            f"| {row['model']} | {row['device']} | {row['operational_conf']:.4f} | "
            f"{row['K_ms']:.6g} | "
            f"{row['c_ms_per_pixel']:.9g} | {row['tau_pixels']:.3f} | "
            f"{row['r_squared']:.4f} | {row['mae_ms']:.4f} | {row['rmse_ms']:.4f} | "
            f"{len(row['shapes'])} |"
        )
    lines.extend(["", "## 5. Full-frame baselines", ""])
    lines.extend(_markdown_table(summary, ("Model", "Device", "Mode", "FPS", "mAP50", "Recall50")))
    lines.extend(["", "## 6. Segment formation only", ""])
    lines.extend(_markdown_table(summary, ("Model", "Device", "Mode", "FPS", "Speedup vs FF", "ΔmAP50", "Tensor ratio")))
    lines.extend(["", "## 7. Cost-aware merge", ""])
    lines.extend(_markdown_table(summary, ("Model", "Device", "Mode", "FPS", "Speedup vs FF", "NN calls/frame", "Mean latency ms")))
    lines.extend(["", "## 8. Full technology", ""])
    lines.extend(_markdown_table(fulltech, FULLTECH_FIELDS))
    lines.extend(["", "## 9. Ablation: FF -> SEG -> CA -> FULL-TECH", ""])
    for model, device in sorted({(row["Model"], row["Device"]) for row in summary}):
        entries = {row["Mode"]: row for row in summary if row["Model"] == model and row["Device"] == device}
        lines.append(f"### {model} / {device}")
        lines.append("")
        for before, after in (("FF", "SEG"), ("SEG", "CA"), ("CA", "FULL-TECH")):
            if before not in entries or after not in entries:
                lines.append(f"- {before} -> {after}: insufficient completed runs.")
                continue
            left, right = entries[before], entries[after]
            lines.append(
                f"- {before} -> {after}: FPS {left['FPS']:.4f} -> {right['FPS']:.4f}; "
                f"mAP50 {left['mAP50']:.4f} -> {right['mAP50']:.4f}; "
                f"tensor ratio {left['Tensor ratio']:.4f} -> {right['Tensor ratio']:.4f}; "
                f"NN calls/frame {left['NN calls/frame']:.3f} -> {right['NN calls/frame']:.3f}."
            )
        lines.append("")
    lines.extend([
        "## 10. YOLO26n: Core i9-9900KF vs Raspberry Pi 5",
        *_device_comparison_lines(summary, "yolo26n"),
        "",
        "## 11. ROI-SSD-MobileNet: Core i9-9900KF vs Raspberry Pi 5",
        *_device_comparison_lines(summary, "roi_ssd_mobilenet"),
        "",
        "## 12. Accuracy-speed trade-off",
        "- See the numeric mode-by-mode table above; negative or neutral speed/accuracy changes are retained.",
        "",
        "## 13. Negative or unexpected findings",
        "- Inspect `timing_runs.csv`, `frame_level_metrics.csv`, and crash/error records. Thermal-throttling status and temperatures are recorded for every run.",
        "",
        "## 14. Dissertation-ready conclusions",
        "- The FULL-TECH vs FF numbers above are direct measurements for this model/device/data protocol only.",
        f"- Timing repeats: {repeats} per mode. Bootstrap confidence intervals were not computed; adjacent frames are not treated as independent samples.",
        "",
        "## Timing and metric definitions",
        "- mAP50: project area-integral AP at IoU 0.50; mAP50:95 is mean AP at ten IoU thresholds 0.50..0.95.",
        "- FPS and frame latency use the Stage 3 process-frame timing boundary; model/data initialization and evaluation are excluded.",
        "- Tensor ratio is actual network-input pixels relative to full-frame tensor pixels. Geometric area ratio is recorded per frame in the frame CSV.",
        "- Segmentation formation time is represented by tracker-update/formation time; preprocessing, inference, merge, postprocess and total process time are separate.",
        "- Timing summary reports mean ± sample standard deviation of run FPS and mean of run-level latency percentiles.",
    ])
    (directory / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(rows: Sequence[Dict], columns: Sequence[str]) -> List[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append("" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _device_comparison_lines(summary: Sequence[Dict], model: str) -> List[str]:
    selected = [row for row in summary if row["Model"] == model and row["Mode"] == "FULL-TECH"]
    if len(selected) < 2:
        return ["- Cross-device comparison is pending runs on both target platforms."]
    return _markdown_table(selected, ("Model", "Device", "FPS", "mAP50", "Recall50", "Tensor ratio"))


def _write_environment(directory: Path, pilot, device_label: str,
                       note: str, timing_rows: Sequence[Dict]) -> None:
    runtime = pilot._runtime_environment()
    lines = [
        "# Hardware and software environment", "",
        f"- Device label: {device_label}",
        f"- Platform: {platform.platform()}",
        f"- Processor: {runtime['cpu_model']}",
        f"- Physical/logical cores: {runtime['physical_cores']}/{runtime['logical_cores']}",
        f"- Python: {sys.version}",
        f"- PyTorch: {runtime['pytorch']}",
        f"- PyTorch intra/inter-op threads: {runtime['pytorch_intra_op_threads']}/{runtime['pytorch_inter_op_threads']}",
        f"- OpenCV threads: {runtime['opencv_threads']}",
        f"- System-load note: {note or 'not supplied'}",
        "- Temperature, CPU frequency, maximum sampled temperature and throttling status: timing_runs.csv.",
        "",
        "## Repeated run hardware observations",
        "",
        "| Model | Mode | Repeat | Temp before | Temp after | Temp max | CPU MHz before/after | Power mode before/after | Throttling |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in timing_rows:
        lines.append(
            f"| {row['model']} | {row['mode']} | {row['repeat']} | "
            f"{row['temperature_before_c']} | {row['temperature_after_c']} | "
            f"{row['temperature_max_c']} | {row['cpu_frequency_before_mhz']}/"
            f"{row['cpu_frequency_after_mhz']} | {row['power_mode_before']}/"
            f"{row['power_mode_after']} | {row['throttling_status']} |"
        )
    (directory / "hardware_software_environment.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _make_plots(directory: Path, summary: Sequence[Dict], frames: Sequence[Dict]) -> None:
    if not summary:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (directory / "plots_not_generated.txt").write_text(
            "Plots were not generated because matplotlib is not installed.\n",
            encoding="utf-8",
        )
        return
    plots = directory / "plots"
    plots.mkdir(exist_ok=True)
    colors = {"FF": "#333333", "SEG": "#2878b5", "CA": "#e1812c", "FULL-TECH": "#3a923a"}
    fig, axis = plt.subplots()
    for row in summary:
        axis.scatter(row["FPS"], row["mAP50"], color=colors.get(row["Mode"]), label=row["Mode"])
        axis.annotate(f"{row['Model']}/{row['Device']}/{row['Mode']}", (row["FPS"], row["mAP50"]), fontsize=7)
    axis.set(xlabel="FPS", ylabel="mAP@0.5", title="FPS vs mAP@0.5")
    fig.savefig(plots / "fps_vs_map50.png", bbox_inches="tight")
    plt.close(fig)

    tech = _fulltech_summary(summary)
    if tech:
        fig, axis = plt.subplots()
        labels = [f"{row['Model']}\n{row['Device']}" for row in tech]
        axis.bar(labels, [row["Speedup"] for row in tech])
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set(ylabel="FULL-TECH speedup vs FF", title="Speedup by model and device")
        fig.savefig(plots / "speedup_by_model_device.png", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, key, title in zip(
        axes, ("Area ratio", "Tensor ratio", "NN calls/frame"),
        ("Geometric area ratio", "Tensor area ratio", "NN calls per frame"),
    ):
        labels = [f"{row['Model']}/{row['Device']}/{row['Mode']}" for row in summary]
        axis.bar(range(len(summary)), [row[key] for row in summary])
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(plots / "area_tensor_calls_comparison.png", bbox_inches="tight")
    plt.close(fig)

    if frames:
        grouped = {}
        for row in frames:
            grouped.setdefault((row["model"], row["device"], row["pipeline_mode"]), []).append(row)
        keys = ("formation_ms", "merge_ms", "preprocess_ms", "inference_ms", "postprocess_ms")
        labels = list(grouped)
        fig, axis = plt.subplots(figsize=(12, 5))
        bottoms = [0.0] * len(labels)
        for key in keys:
            values = [statistics.mean(float(item[key]) for item in grouped[label]) for label in labels]
            axis.bar(range(len(labels)), values, bottom=bottoms, label=key)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        axis.set_xticks(range(len(labels)), ["/".join(item) for item in labels], rotation=90, fontsize=7)
        axis.set_ylabel("Mean time (ms/frame)")
        axis.legend()
        fig.tight_layout()
        fig.savefig(plots / "latency_breakdown.png", bbox_inches="tight")
        plt.close(fig)


def _aggregate_device_runs(output_root: Path) -> None:
    device_dirs = sorted(path for path in output_root.iterdir() if path.is_dir() and
                         (path / "final_summary.csv").is_file())
    if not device_dirs:
        raise FileNotFoundError(f"No completed per-device results under {output_root}")
    summaries, fulltech, frames, timings, costs, per_class = [], [], [], [], [], []
    profiles = []
    device_provenance = {}
    for directory in device_dirs:
        provenance_path = directory / "manifest_provenance.json"
        if provenance_path.is_file():
            device_provenance[directory.name] = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
        for name, sink in (
            ("final_summary.csv", summaries),
            ("final_fulltech_vs_fullframe.csv", fulltech),
            ("frame_level_metrics.csv", frames),
            ("timing_runs.csv", timings),
            ("cost_profiles_final.csv", costs),
            ("per_class_metrics.csv", per_class),
        ):
            path = directory / name
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8", newline="") as stream:
                sink.extend(csv.DictReader(stream))
        profile_path = directory / "cost_profiles_final.json"
        if profile_path.is_file():
            profiles.extend(json.loads(profile_path.read_text(encoding="utf-8")).get("profiles", []))
    stage3_selection_fingerprints = {
        json.dumps(provenance.get("stage3_selections"), sort_keys=True)
        for provenance in device_provenance.values()
        if provenance.get("stage3_selections") is not None
    }
    if len(stage3_selection_fingerprints) > 1:
        raise ValueError("Device runs use different frozen Stage 3 operational selections")
    for row in summaries:
        for key in ("FPS", "Speedup vs FF", "mAP50", "ΔmAP50", "mAP50:95",
                    "Recall50", "Area ratio", "Tensor ratio", "NN calls/frame",
                    "Full-frame fraction", "Mean latency ms", "Latency p50 ms",
                    "Latency p95 ms", "FPS std", "Min run FPS", "Max run FPS"):
            row[key] = float(row[key]) if row.get(key) not in (None, "") else None
    for row in frames:
        for key in ("processed_geometric_pixels", "processed_area_ratio",
                    "tensor_pixels", "tensor_area_ratio", "formation_ms", "merge_ms",
                    "preprocess_ms", "inference_ms", "postprocess_ms",
                    "total_process_frame_ms"):
            row[key] = float(row[key]) if row.get(key) not in (None, "") else None
    _recompute_summary_speedups(summaries)
    _write_csv(output_root / "final_summary.csv", SUMMARY_FIELDS, summaries)
    _write_csv(output_root / "final_fulltech_vs_fullframe.csv", FULLTECH_FIELDS, fulltech)
    _write_csv(output_root / "frame_level_metrics.csv", FRAME_FIELDS, frames)
    _write_csv(output_root / "timing_runs.csv", TIMING_FIELDS, timings)
    _write_csv(output_root / "cost_profiles_final.csv", COST_FIELDS, costs)
    _write_csv(output_root / "per_class_metrics.csv",
               ("model", "device", "mode", "class", "AP50", "AP95"), per_class)
    shutil.copy2(FINAL_MANIFEST, output_root / "final_eval_manifest_candidate.txt")
    _write_json(output_root / "final_eval_manifest_provenance.json", {
        "sha256": EXPECTED_FINAL_SHA256,
        "source": str(FINAL_MANIFEST.relative_to(ROOT)),
        "device_runs": [path.name for path in device_dirs],
        "stage3_selections": device_provenance.get(
            device_dirs[0].name, {}
        ).get("stage3_selections"),
        "device_dataset_yamls": {
            name: {
                "path": provenance.get("dataset_yaml"),
                "sha256": provenance.get("dataset_yaml_sha256"),
            }
            for name, provenance in device_provenance.items()
        },
        "source_audit_manifest_sha256": device_provenance.get(
            device_dirs[0].name, {}
        ).get("source_audit_manifest_sha256"),
        "source_audit_manifest_sha256_matches_documented": device_provenance.get(
            device_dirs[0].name, {}
        ).get("source_audit_manifest_sha256_matches_documented"),
    })
    repeat_counts = [int(row["repeat"]) for row in timings if row.get("repeat")]
    repeats = max(repeat_counts) if repeat_counts else TIMING_RUNS
    _write_report(output_root, summaries, fulltech, profiles, repeats)
    _make_plots(output_root, summaries, frames)
    environment = ["# Aggregated hardware and software environments", ""]
    for directory in device_dirs:
        path = directory / "hardware_software_environment.md"
        if path.is_file():
            environment.extend([f"## {directory.name}", "", path.read_text(encoding="utf-8")])
    (output_root / "hardware_software_environment.md").write_text(
        "\n".join(environment), encoding="utf-8"
    )
    configs_target = output_root / "config_snapshots"
    configs_target.mkdir(exist_ok=True)
    for directory in device_dirs:
        snapshots = directory / "config_snapshots"
        if snapshots.is_dir():
            for source in snapshots.iterdir():
                destination = configs_target / f"{directory.name}_{source.name}"
                shutil.copy2(source, destination)


def _dry_run(
    output_root: Path, device_label: str,
    dataset_yaml_override: Optional[Path] = None,
) -> None:
    pilot_rows, final_rows, manifest_info = _validate_frozen_inputs(
        dataset_yaml_override
    )
    for model_name in MODEL_NAMES:
        _read_locked_padding(model_name)
        config = MODEL_CONFIGS[model_name]["config"]
        if not config.is_file():
            raise FileNotFoundError(config)
    print(f"Protocol inputs validated for device label {device_label}")
    print(f"Pilot frames: {len(pilot_rows)}; final frames: {len(final_rows)}")
    print(f"Final manifest SHA-256: {manifest_info['final_sha256']}")
    print(f"Dataset YAML: {manifest_info['dataset_yaml']}")
    for model_name, selection in manifest_info["stage3_selections"].items():
        print(
            f"Stage 3 operational confidence for {model_name}: "
            f"{selection['operational_conf']}"
        )
    print(f"Output root: {output_root}")
    print("No files written and no benchmark results produced.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "calibrate", "benchmark", "aggregate"),
                        default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dataset-yaml", type=Path,
        help="Use this copy of the frozen dataset YAML without changing model configs",
    )
    parser.add_argument("--device-label", default=platform.node() or platform.machine())
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--timing-repeats", type=int, default=TIMING_RUNS)
    parser.add_argument("--calibration-repetitions", type=int, default=8)
    parser.add_argument("--system-load-note", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.stage != "aggregate":
        if args.timing_repeats < 2:
            parser.error("--timing-repeats must be at least 2")
        if args.timing_repeats == 2 and not re.search(r"(pi|raspberry)", args.device_label, re.I):
            parser.error("Two timing repeats are only allowed for a Raspberry Pi device label")
        if args.calibration_repetitions < 3:
            parser.error("--calibration-repetitions must be at least 3")
        if args.dry_run:
            _dry_run(args.output_root, args.device_label, args.dataset_yaml)
            return 0

    if args.stage == "aggregate":
        _aggregate_device_runs(args.output_root)
        print(f"Aggregated device runs under {args.output_root}")
        return 0

    pilot_rows, final_rows, manifest_info = _validate_frozen_inputs(args.dataset_yaml)
    device_label = _device_slug(args.device_label)
    device_dir = _prepare_device_directory(
        args.output_root, device_label, final_rows, manifest_info
    )
    pilot = _load_pilot_module()

    if args.stage in ("all", "calibrate"):
        profiles = _profile_models(
            pilot, device_dir, device_label, pilot_rows,
            args.models, args.calibration_repetitions, args.dataset_yaml,
        )
        print(f"Frozen cost profiles written to {_cost_profile_path(device_dir)}")
        calibration_complete = len(profiles) == len(args.models)
    else:
        calibration_complete = True
    if args.stage in ("all", "benchmark"):
        benchmark_complete = _run_benchmarks(
            pilot, device_dir, device_label, final_rows, args.models,
            args.timing_repeats, args.system_load_note, args.dataset_yaml,
        )
        has_device_results = any(
            path.is_dir() and (path / "final_summary.csv").is_file()
            for path in args.output_root.iterdir()
        )
        if has_device_results:
            _aggregate_device_runs(args.output_root)
        print(f"Benchmark outputs written under {device_dir}")
    else:
        benchmark_complete = True
    return 0 if calibration_complete and benchmark_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
