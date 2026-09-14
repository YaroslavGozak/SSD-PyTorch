"""Configuration, metadata, CSV, and system-sensor helpers."""

import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


RAW_A_FIELDS = ["session_id", "run_id", "order_index", "timing_mode", "seed", "requested_w", "requested_h", "requested_area", "crop_w", "crop_h", "crop_area", "tensor_w", "tensor_h", "effective_area", "aspect_ratio", "model_stride", "preprocess_ms", "inference_ms", "postprocess_ms", "detector_call_ms", "prediction_count_before_nms", "prediction_count_after_nms", "cpu_temp_c", "cpu_freq_mhz", "process_rss_mb", "elapsed_s", "timestamp_utc"]


def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def append_csv(path: Path, row: Dict[str, Any], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_list = list(fields)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in field_list})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def system_metadata() -> Dict[str, Any]:
    return {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(), "os": platform.platform(),
            "architecture": platform.machine(), "logical_cpu_count": os.cpu_count(),
            "pid": os.getpid(), "cpu_temp_c": read_cpu_temp(), "cpu_freq_mhz": read_cpu_freq(),
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain"))}


def read_cpu_temp() -> Any:
    for path in ("/sys/class/thermal/thermal_zone0/temp",):
        try:
            return float(Path(path).read_text().strip()) / 1000.0
        except (OSError, ValueError):
            pass
    return None


def read_cpu_freq() -> Any:
    try:
        value = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read_text().strip()
        return float(value) / 1000.0
    except (OSError, ValueError):
        return None


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def deterministic_image(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(int(height), int(width), 3), dtype=np.uint8)


def bootstrap_ci(values: Iterable[float], seed: int = 0, iterations: int = 2000) -> List[float]:
    """Return percentile CI for already-generated bootstrap estimates.

    The values are bootstrap replicates, not independent observations. Do not
    resample them and compute a CI for their mean here.
    """
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return [float("nan"), float("nan")]
    return [float(value) for value in np.quantile(data, [0.025, 0.975])]