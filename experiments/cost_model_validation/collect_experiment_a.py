"""Collect synthetic latency observations for Experiment A."""

import argparse
import json
import random
import time
import uuid
from pathlib import Path

import numpy as np

from .adapters import FakeAdapter, PreparedInput
from .common import RAW_A_FIELDS, append_csv, collection_provenance, deterministic_image, load_config, system_metadata, write_json
from .timing import measure


def requested_shapes(config):
    experiment = config.get("experiment_a", config)
    max_h, max_w = map(int, experiment.get("max_requested_hw", [640, 640]))
    fractions = experiment.get("area_fractions", np.linspace(.05, 1.0, 20).tolist())
    shapes = []
    for fraction in fractions:
        for ratio in ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0)):
            area = max_h * max_w * float(fraction)
            height = max(1, min(max_h, round((area / (ratio[0] / ratio[1])) ** .5)))
            width = max(1, min(max_w, round(height * ratio[0] / ratio[1])))
            shapes.append((width, height))
    return shapes


def build_adapter(config):
    model_cfg = config.get("model", {})
    if model_cfg.get("backend", "fake") == "fake":
        return FakeAdapter(int(model_cfg.get("stride", 32)))
    if model_cfg.get("backend") == "roissd":
        from .roissd_adapter import RoiSSDAdapter
        return RoiSSDAdapter(
            model_config=model_cfg["model_config"],
            weights=model_cfg["weights"],
            device=model_cfg.get("device", "cpu"),
            stride=int(model_cfg.get("stride", 1)),
        )
    if model_cfg.get("backend") != "ultralytics":
        raise ValueError("model.backend must be 'fake', 'ultralytics', or 'roissd'")
    from .ultralytics_adapter import UltralyticsAdapter
    return UltralyticsAdapter(model_cfg["weights"], model_cfg.get("device", "cpu"), int(model_cfg.get("stride", 32)))


def collect(config, output: str, overwrite: bool = False) -> None:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "experiment_a_raw.csv"
    if raw_path.exists() and overwrite:
        raw_path.unlink()
    adapter = build_adapter(config)
    experiment = config.get("experiment_a", config)
    seed = int(config.get("seed", 0))
    mode = str(experiment.get("timing_mode", "inference_only"))
    repetitions = max(30, int(experiment.get("repetitions_per_effective_shape", 40)))
    progress_every = max(1, int(experiment.get("progress_every_repetitions", 1)))
    image = deterministic_image(*map(int, experiment.get("max_requested_hw", [640, 640])), seed)
    requested = requested_shapes(config)
    shape_sources = {}
    shapes = []
    effective_keys = set()
    for requested_w, requested_h in requested:
        prepared = adapter.prepare(image, (requested_h, requested_w))
        key = prepared.tensor_hw
        shape_sources.setdefault(str(key), []).append([requested_w, requested_h])
        if key not in effective_keys:
            effective_keys.add(key)
            shapes.append((requested_w, requested_h))
    randomizer = random.Random(seed)
    metadata_path = output_dir / "metadata.json"
    session_id = str(uuid.uuid4())
    if raw_path.exists() and not overwrite and metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            session_id = json.load(handle).get("session_id", session_id)
    existing = set()
    if raw_path.exists():
        import csv
        with raw_path.open(newline="", encoding="utf-8") as handle:
            existing = {(row.get("session_id"), row.get("run_id"), row.get("requested_w"), row.get("requested_h")) for row in csv.DictReader(handle)}
    session_start = time.perf_counter()
    provenance = collection_provenance(config, adapter, mode)
    if raw_path.exists() and not overwrite:
        # A resumed legacy run must not acquire invented historical environment data.
        old_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        provenance = old_metadata.get("provenance", {})
    print(f"[experiment_a] starting {repetitions} repetitions across {len(shapes)} effective shapes; mode={mode}", flush=True)
    for _ in range(int(experiment.get("global_warmup_iterations", 50))):
        measure(adapter, image, shapes[0][::-1], mode)
    for run_id in range(repetitions):
        order = list(range(len(shapes)))
        randomizer.shuffle(order)
        for order_index, shape_index in enumerate(order):
            requested_w, requested_h = shapes[shape_index]
            if (session_id, str(run_id), str(requested_w), str(requested_h)) in existing:
                continue
            prepared, result = measure(adapter, image, (requested_h, requested_w), mode)
            tensor_h, tensor_w = prepared.tensor_hw
            row = {"session_id": session_id, "run_id": run_id, "order_index": order_index, "timing_mode": mode, "seed": seed,
                   "requested_w": requested_w, "requested_h": requested_h, "requested_area": requested_w * requested_h,
                   "crop_w": requested_w, "crop_h": requested_h, "crop_area": requested_w * requested_h,
                   "tensor_w": tensor_w, "tensor_h": tensor_h, "effective_area": tensor_h * tensor_w,
                   "aspect_ratio": requested_w / requested_h, "model_stride": adapter.stride,
                   "preprocess_ms": result.preprocess_ms, "inference_ms": result.inference_ms,
                   "postprocess_ms": result.postprocess_ms, "detector_call_ms": result.detector_call_ms,
                   "elapsed_s": time.perf_counter() - session_start, "timestamp_utc": system_metadata()["timestamp_utc"]}
            append_csv(raw_path, row, RAW_A_FIELDS)
        if (run_id + 1) % progress_every == 0 or run_id + 1 == repetitions:
            completed = run_id + 1
            elapsed = time.perf_counter() - session_start
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta = (repetitions - completed) / rate if rate > 0 else float("nan")
            print(f"[experiment_a] repetition {completed}/{repetitions}; shapes {len(shapes)}; "
                  f"elapsed {elapsed:.1f}s; {rate:.2f} repetitions/s; ETA {eta / 60:.1f} min",
                  flush=True)
    import csv
    with raw_path.open(newline="", encoding="utf-8") as handle:
        unique_shapes = {(row["tensor_h"], row["tensor_w"]) for row in csv.DictReader(handle)}
    if len(unique_shapes) < 2:
        raise RuntimeError("All requested shapes produced one effective tensor shape; benchmark is invalid")
    metadata = system_metadata()
    metadata.update({"config": config, "session_id": session_id, "timing_mode": mode, "model_stride": adapter.stride,
                     "effective_shape_sources": shape_sources,
                     "model_weights_sha256": provenance.get("weights_sha256"),
                     "provenance": provenance,
                     "warnings": ["temperature/frequency unavailable"]})
    write_json(output_dir / "metadata.json", metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    collect(load_config(args.config), args.output, args.overwrite)


if __name__ == "__main__":
    main()
