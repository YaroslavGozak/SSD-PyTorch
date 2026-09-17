"""Collect synthetic latency observations for Experiment A."""

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np

from .adapters import FakeAdapter
from .common import RAW_A_FIELDS, append_csv, collection_provenance, deterministic_image, load_config, system_metadata, write_json
from .timing import measure
from .reproducibility import canonical_hash, schedule, gate
from .geometry import stride_rounded_shape
from .models import CalibrationEnvelope


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
    repetitions = int(experiment.get("repetitions_per_effective_shape", 40))
    if repetitions < 2:
        raise ValueError("At least two repetitions required")
    progress_every = max(1, int(experiment.get("progress_every_repetitions", 1)))
    image = deterministic_image(*map(int, experiment.get("max_requested_hw", [640, 640])), seed)
    requested = requested_shapes(config)
    shape_sources = {}
    shapes = []
    effective_keys = set()
    for requested_w, requested_h in requested:
        prepared = adapter.prepare(image, (requested_h, requested_w))
        key = prepared.tensor_hw
        if any(v % adapter.stride for v in key):
            raise ValueError("Non-stride-aligned actual tensor shape")
        shape_sources.setdefault(str(key), []).append([requested_w, requested_h])
        if key not in effective_keys:
            effective_keys.add(key)
            shapes.append((requested_w, requested_h))
    metadata_path = output_dir / "metadata.json"
    schedule_path = output_dir / "experiment_a_schedule.json"
    schedule_seed = int(experiment.get("schedule_seed", seed))
    tensor_shapes = [adapter.prepare(image, (h, w)).tensor_hw for w, h in shapes]
    sources = dict(zip(tensor_shapes, shapes))
    provenance = collection_provenance(config, adapter, mode)
    warnings = []
    gate(not config.get("publication_run", False) or not provenance["git_dirty"], "git_dirty", config, warnings)
    identity = {k: provenance.get(k) for k in ("weights_sha256", "backend", "device", "preprocessing", "model_stride", "timing_mode")}
    settings = dict(shapes=tensor_shapes, repetitions=repetitions, seed=schedule_seed, identity=identity,
                    experiment=experiment)
    if raw_path.exists() and not overwrite:
        if not schedule_path.exists() or not metadata_path.exists():
            raise ValueError("Legacy run has no persisted schedule; use a new output directory")
        saved = json.loads(schedule_path.read_text(encoding="utf-8"))
        if saved["settings_hash"] != canonical_hash(settings) or saved["hash"] != canonical_hash(saved["items"]):
            raise ValueError("Schedule/config/provenance mismatch on resume")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        session_id = metadata["session_id"]
        provenance = metadata["provenance"]
    else:
        saved = dict(settings_hash=canonical_hash(settings), items=schedule(tensor_shapes, repetitions, schedule_seed))
        saved["hash"] = canonical_hash(saved["items"])
        write_json(schedule_path, saved)
        session_id = str(uuid.uuid4())
        metadata = system_metadata()
    prewarm = metadata.get("warmup_schedule")
    if prewarm is None:
        prewarm = schedule(tensor_shapes, int(experiment.get("prewarm_passes", 2)), schedule_seed+1)
    metadata.update(config=config, session_id=session_id, timing_mode=mode, model_stride=adapter.stride,
                    effective_shape_sources=shape_sources, model_weights_sha256=provenance.get("weights_sha256"),
                    provenance=provenance, schedule_hash=saved["hash"], schedule_seed=schedule_seed,
                    warmup_schedule=prewarm, warmup_seed=schedule_seed+1, warnings=warnings)
    write_json(metadata_path, metadata)
    existing = set()
    elapsed_offset = 0.
    if raw_path.exists():
        import csv
        with raw_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                position = int(row["global_position"])
                item = saved["items"][position]
                if position in existing or row["shape_id"] != item["shape_id"] or row["session_id"] != session_id:
                    raise ValueError("Raw observations disagree with saved schedule")
                existing.add(position)
                elapsed_offset = max(elapsed_offset, float(row["elapsed_s"]))
    session_start = time.perf_counter() - elapsed_offset
    control_path = output_dir / "experiment_a_controls.json"
    controls = json.loads(control_path.read_text()) if control_path.exists() and not overwrite else []
    control_shapes = [stride_rounded_shape(*hw, adapter.stride) for hw in experiment.get("control_shapes", [])]
    hs, ws = zip(*tensor_shapes)
    envelope = CalibrationEnvelope(min(h*w for h,w in tensor_shapes), max(h*w for h,w in tensor_shapes),
                                  min(hs), max(hs), min(ws), max(ws), min(w/h for h,w in tensor_shapes), max(w/h for h,w in tensor_shapes))
    if any(not envelope.contains(*hw) for hw in control_shapes):
        raise ValueError("Control shape outside calibration envelope")
    print(f"[experiment_a] starting {repetitions} repetitions across {len(shapes)} effective shapes; mode={mode}", flush=True)
    for _ in range(int(experiment.get("global_warmup_iterations", 50))):
        measure(adapter, image, shapes[0][::-1], mode)
    for item in prewarm:
        measure(adapter, image, sources[(item["tensor_h"], item["tensor_w"])][::-1], mode)
    for run_id in range(repetitions):
        for item in saved["items"][run_id*len(shapes):(run_id+1)*len(shapes)]:
            order_index = item["position_in_block"]
            requested_w, requested_h = sources[(item["tensor_h"], item["tensor_w"])]
            if item["global_position"] in existing:
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
            if prepared.tensor_hw != (item["tensor_h"], item["tensor_w"]):
                raise ValueError("Actual tensor shape changed from persisted schedule")
            row.update(item)
            row["latency_s"] = (result.inference_ms if mode == "inference_only" else result.detector_call_ms)/1000
            append_csv(raw_path, row, RAW_A_FIELDS)
        every = int(experiment.get("control_every_blocks", 2))
        if every > 0 and (run_id+1) % every == 0:
            offset = (run_id//every) % max(1, len(control_shapes))
            for hw in control_shapes[offset:] + control_shapes[:offset]:
                if any(r["block_index"] == run_id and (r["tensor_h"],r["tensor_w"]) == hw for r in controls):
                    continue
                prepared, result = measure(adapter, image, hw, mode)
                controls.append(dict(block_index=run_id, tensor_h=hw[0], tensor_w=hw[1], effective_area=hw[0]*hw[1],
                                     measured_ms=result.inference_ms if mode == "inference_only" else result.detector_call_ms,
                                     elapsed_s=time.perf_counter()-session_start, timestamp_utc=system_metadata()["timestamp_utc"]))
                write_json(control_path, controls)
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
    write_json(metadata_path, metadata)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    collect(load_config(args.config), args.output, args.overwrite)


if __name__ == "__main__":
    main()
