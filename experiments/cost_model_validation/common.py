"""Configuration, metadata, CSV, and system-sensor helpers."""

import csv
import hashlib
import json
import os
import platform
import subprocess
import time
import re
from importlib.metadata import PackageNotFoundError, version
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


_THREADPOOL_LIMIT = None


RAW_A_FIELDS = ["shape_id", "block_index", "position_in_block", "global_position", "latency_s", "session_id", "run_id", "order_index", "timing_mode", "seed", "requested_w", "requested_h", "requested_area", "crop_w", "crop_h", "crop_area", "tensor_w", "tensor_h", "effective_area", "aspect_ratio", "model_stride", "preprocess_ms", "inference_ms", "postprocess_ms", "detector_call_ms", "prediction_count_before_nms", "prediction_count_after_nms", "cpu_temp_c", "cpu_freq_mhz", "process_rss_mb", "elapsed_s", "timestamp_utc"]


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
        json.dump(_finite_json(value), handle, indent=2, sort_keys=True, default=_json_default, allow_nan=False)


def _finite_json(value):
    if isinstance(value, dict):
        return {key:_finite_json(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):
        return None
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


@lru_cache(maxsize=1)
def _static_system_metadata() -> Dict[str, Any]:
    return {"python": platform.python_version(), "os": platform.platform(),
            "architecture": platform.machine(), "logical_cpu_count": os.cpu_count(),
            "pid": os.getpid(), "cpu_temp_c": read_cpu_temp(), "cpu_freq_mhz": read_cpu_freq(),
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain"))}


def system_metadata() -> Dict[str, Any]:
    metadata = dict(_static_system_metadata())
    metadata["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return metadata


def _process_affinity() -> List[int] | None:
    """Return the logical CPUs available to this process."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(os.sched_getaffinity(0))
        except OSError:
            return None
    if os.name == "nt":
        try:
            import ctypes
            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.GetProcessAffinityMask.argtypes = (
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
            kernel32.GetProcessAffinityMask.restype = ctypes.c_int
            if not kernel32.GetProcessAffinityMask(
                    kernel32.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
                return None
            return [index for index in range(process_mask.value.bit_length())
                    if process_mask.value & (1 << index)]
        except (AttributeError, OSError):
            return None
    return None


def _set_process_affinity(cpus: List[int]) -> None:
    if not cpus or any(cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus):
        raise ValueError("runtime.cpu_affinity must contain unique non-negative CPU indices")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpus))
        return
    if os.name == "nt":
        import ctypes
        mask = sum(1 << cpu for cpu in cpus)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.c_size_t(mask)):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
        return
    raise RuntimeError("CPU affinity is not supported on this platform")


def read_power_mode() -> Dict[str, Any]:
    """Read the active OS power policy without changing machine-wide settings."""
    if os.name == "nt":
        try:
            output = subprocess.check_output(
                ["powercfg", "/getactivescheme"], text=True, errors="replace",
                stderr=subprocess.DEVNULL, timeout=3).strip()
            match = re.search(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}", output)
            label = None
            label_match = re.search(r"\(([^()]*)\)\s*$", output)
            if label_match:
                label = label_match.group(1)
            return {"source": "powercfg", "scheme_guid": match.group(0).lower() if match else None,
                    "label": label, "raw": output}
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {"source": "powercfg", "available": False}
    governors = {}
    for path in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"):
        try:
            governors[path.parent.name] = path.read_text().strip()
        except OSError:
            pass
    return {"source": "sysfs", "governors": governors, "available": bool(governors)}


def apply_runtime_controls(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply per-process CPU controls and return their observed state."""
    global _THREADPOOL_LIMIT
    requested = config.get("runtime", {}) or {}
    threads = requested.get("cpu_threads")
    interop_threads = requested.get("torch_interop_threads")
    if threads is not None:
        threads = int(threads)
        if threads < 1:
            raise ValueError("runtime.cpu_threads must be positive")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(threads)
        try:
            from threadpoolctl import threadpool_limits
            _THREADPOOL_LIMIT = threadpool_limits(limits=threads)
        except ImportError:
            _THREADPOOL_LIMIT = None
    if requested.get("cpu_affinity") is not None:
        _set_process_affinity([int(cpu) for cpu in requested["cpu_affinity"]])

    torch_threads = torch_interop = None
    if threads is not None or interop_threads is not None:
        import torch
        if threads is not None:
            torch.set_num_threads(threads)
        if interop_threads is not None:
            interop_threads = int(interop_threads)
            if interop_threads < 1:
                raise ValueError("runtime.torch_interop_threads must be positive")
            if torch.get_num_interop_threads() != interop_threads:
                try:
                    torch.set_num_interop_threads(interop_threads)
                except RuntimeError as exc:
                    raise RuntimeError("Torch inter-op threads must be configured before parallel work starts") from exc
        torch_threads = torch.get_num_threads()
        torch_interop = torch.get_num_interop_threads()
    else:
        try:
            import torch
            torch_threads = torch.get_num_threads()
            torch_interop = torch.get_num_interop_threads()
        except ImportError:
            pass

    power = read_power_mode()
    expected_guid = requested.get("expected_power_scheme_guid")
    if expected_guid:
        observed_guid = power.get("scheme_guid")
        if observed_guid != str(expected_guid).lower():
            raise RuntimeError(f"Active power scheme is {observed_guid!r}, expected {str(expected_guid).lower()!r}")
    expected_governor = requested.get("expected_cpu_governor")
    if expected_governor:
        observed = set(power.get("governors", {}).values())
        if observed != {str(expected_governor)}:
            raise RuntimeError(f"CPU governors are {sorted(observed)!r}, expected {expected_governor!r}")

    observed_affinity = _process_affinity()
    expected_affinity = requested.get("cpu_affinity")
    if expected_affinity is not None and observed_affinity != sorted(map(int, expected_affinity)):
        raise RuntimeError(f"CPU affinity is {observed_affinity!r}, expected {sorted(map(int, expected_affinity))!r}")
    return {
        "requested": requested,
        "process_affinity": observed_affinity,
        "torch_num_threads": torch_threads,
        "torch_num_interop_threads": torch_interop,
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "power_mode": power,
    }


def collection_provenance(config, adapter, timing_mode, runtime_environment=None):
    """Capture model identity at collection time, never from an analysis environment."""
    model = config.get("model", {})
    backend = model.get("backend", "fake")
    versions = {}
    for package in ("numpy", "torch", "torchvision", "ultralytics"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    weights = model.get("weights")
    preprocessing = adapter.preprocessing_metadata()
    return {
        "backend": backend,
        "backend_version": versions.get("ultralytics" if backend == "ultralytics" else "numpy" if backend == "fake" else "torch"),
        "versions": versions,
        "adapter": f"{type(adapter).__module__}.{type(adapter).__name__}",
        "device": str(getattr(adapter, "device", "cpu")),
        "weights_path": weights,
        "weights_sha256": file_sha256(weights) if weights else None,
        "model_config": model.get("model_config"),
        "model_config_sha256": file_sha256(model["model_config"]) if model.get("model_config") else None,
        "model_stride": adapter.stride,
        "preprocessing": preprocessing,
        "dtype": preprocessing.get("dtype"),
        "shape_policy": preprocessing.get("shape_policy"),
        "timing_mode": timing_mode,
        "seed": int(config.get("seed", 0)),
        "runtime_environment": runtime_environment,
        **system_metadata(),
    }


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
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
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
