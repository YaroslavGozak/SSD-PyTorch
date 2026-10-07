"""Run calibration and pairwise cost-aware ROI merging validation on CPU."""

import argparse
import csv
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image

from experiments.cost_model_validation.collect_experiment_a import build_adapter
from experiments.cost_model_validation.common import (append_csv, apply_runtime_controls,
                                                       file_sha256, system_metadata, write_json)
from experiments.cost_model_validation.models import fit_linear_model
from experiments.cost_model_validation.timing import measure
from experiments.cost_model_validation.geometry import Rectangle, union_rectangle
from tools.helpers.config_reader import load_config
from tools.helpers.pipeline import load_dataset

from .core import (DEFAULT_SHAPES, DEFAULT_SIDES, decide, generate_pairs,
                   nearest_shape_cost, pair_geometry, policy_decisions, summarize)
from .replay import canonical_hash, frame_key, resolve_frames, validate_pairs


LOG = logging.getLogger(__name__)


DEFAULTS = {"split": "test", "canvas_hw": [640, 640], "frame_count": 6,
            "calibration_frames": 2, "pair_count": 48, "seed": 20261005,
            "warmup": 30, "calibration_repetitions": 30, "repetitions": 30,
            "timing_mode": "inference_only", "shapes": DEFAULT_SHAPES,
            "calibration_shapes": DEFAULT_SHAPES + [(32,512),(512,32),(32,640),(640,32),
                (64,512),(512,64),(64,640),(640,64),(128,512),(512,128),
                (128,640),(640,128),(256,512),(512,256),(320,512),(512,320),
                (320,640),(640,320),(480,480),(640,640)],
            "sides": DEFAULT_SIDES, "simple_iou": .01,
            "progress_interval_s": 15,
            "simple_distance": 40, "geometric_gamma": 1.4}


def _progress(stage, completed, total, started, last_reported, interval, include_edges=True):
    """Report long loops by time, including a rough remaining-time estimate."""
    now = time.monotonic()
    if (include_edges and (completed == 1 or completed == total)) or now - last_reported >= interval:
        elapsed = now - started
        remaining = elapsed / completed * (total - completed)
        LOG.info("%s: %d/%d (%.1f%%), elapsed %.1fs, estimated remaining %.1fs",
                 stage, completed, total, 100 * completed / total, elapsed, remaining)
        return now
    return last_reported


def _frame(dataset, index, canvas_hw):
    info = dataset.images_info[index]
    original = read_image(info["filename"], mode=ImageReadMode.RGB)
    original_hw = tuple(map(int, original.shape[-2:]))
    resized = F.interpolate(original[None].float(), size=canvas_hw,
                            mode="bilinear", align_corners=False)[0].round().byte()
    return resized.permute(1, 2, 0).contiguous().numpy(), original_hw, str(info["filename"])


def _crop(frame, rect):
    return frame[rect.y1:rect.y2, rect.x1:rect.x2].copy()


def _actual_shape(adapter, crop):
    h, w = crop.shape[:2]
    prepared = adapter.prepare(crop, (h,w))
    if prepared.tensor_hw != tuple(map(int, prepared.tensor.shape[-2:])):
        raise RuntimeError("Adapter reported a shape different from the network tensor")
    return prepared.tensor_hw


def _latency(adapter, crop, mode):
    h, w = crop.shape[:2]
    prepared, result = measure(adapter, crop, (h,w), mode)
    value = result.inference_ms if mode == "inference_only" else result.detector_call_ms
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError(f"Invalid measured latency: {value}")
    trace = adapter.execution_metadata() if hasattr(adapter, "execution_metadata") else None
    execution = dict(execution_trace_status="observed" if trace is not None else "not_applicable",
                     active_depth=trace["active_depth"] if trace else "",
                     active_head_count=trace["active_head_count"] if trace else "",
                     active_head_indices=json.dumps(trace["active_head_indices"]) if trace else "",
                     feature_maps=json.dumps(trace["feature_maps"], separators=(",", ":")) if trace else "")
    return prepared.tensor_hw, value, execution


def _model_settings(config, config_path, device, stride):
    train = config["train_params"]
    name = str(train["model"]).lower()
    if name in {"roissd", "roissd-mobilenet", "roissd_mobilenet"}:
        weights = Path("trained_models") / train["task_name"] / train["ckpt_name"]
        backend = "roissd"
    elif name == "yolo":
        weights = Path(train.get("yolo_weights", train.get("ckpt_name", "best.pt")))
        if not weights.exists():
            weights = Path("trained_models") / train.get("task_name", "") / weights
        backend = "ultralytics"
    else:
        raise ValueError(f"Unsupported model for this experiment: {name}")
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    return {"backend": backend, "model_config": str(config_path),
            "weights": str(weights), "device": device,
            "stride": stride if stride is not None else (32 if backend == "ultralytics" else 1)}


def _profile(adapter, frames, options):
    rng = random.Random(options["seed"] + 1)
    shapes = [tuple(map(int, s)) for s in options["calibration_shapes"]]
    samples = {}
    raw = []
    jobs = [(shape, rep) for rep in range(options["calibration_repetitions"]) for shape in shapes]
    rng.shuffle(jobs)
    started = last_reported = time.monotonic()
    LOG.info("Calibration profiling: %d requested shapes × %d repetitions = %d model calls",
             len(shapes), options["calibration_repetitions"], len(jobs))
    for index, (shape, rep) in enumerate(jobs):
        frame = frames[index % len(frames)][0]
        w, h = shape
        if w > frame.shape[1] or h > frame.shape[0]:
            raise ValueError(f"Calibration shape {shape} exceeds canvas {frame.shape[:2]}")
        x = rng.randrange(frame.shape[1]-w+1)
        y = rng.randrange(frame.shape[0]-h+1)
        actual_hw, latency, execution = _latency(adapter, frame[y:y+h,x:x+w].copy(), options["timing_mode"])
        samples.setdefault(actual_hw, []).append(latency)
        raw.append(dict(requested_w=w, requested_h=h, tensor_h=actual_hw[0], tensor_w=actual_hw[1],
                        repetition=rep, latency_ms=latency, **execution))
        last_reported = _progress("Calibration", index + 1, len(jobs), started,
                                  last_reported, options["progress_interval_s"])
    if len(samples) < 3:
        raise ValueError("Calibration produced fewer than three distinct network input shapes")
    table = {shape: float(np.median(values)) for shape,values in samples.items()}
    areas = [h*w for h,w in table]
    fit = fit_linear_model(areas, [v/1000 for v in table.values()])
    k, c = fit.coefficients["b0"], fit.coefficients["b1"]
    if not np.isfinite(k) or not np.isfinite(c) or k <= 0 or c <= 0:
        raise ValueError(f"Nonphysical affine fit: K={k}, c={c}; examine calibration samples")
    LOG.info("Calibration fit complete: %d actual tensor shapes, K=%.4f ms, c=%.6g ms/pixel, "
             "tau=%.0f px², R²=%.3f", len(table), k*1000, c*1000, k/c, fit.r2)
    return table, fit, raw


def _pair_definitions(options, tau, pairs_path=None):
    if pairs_path:
        payload = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
        validate_pairs(payload, options["canvas_hw"])
        return payload["pairs"], payload.get("generation_skips", {})
    generated, skips = generate_pairs(options["canvas_hw"], options["pair_count"],
                                      options["seed"], options["shapes"], options["sides"], tau)
    records = []
    for index, ((first, second), geometry, region) in enumerate(generated):
        records.append(dict(pair_id=index, r1=list(first.__dict__.values()),
                            r2=list(second.__dict__.values()), geometry_type=geometry,
                            boundary_region=region))
    return records, skips


def _measure_pair(adapter, frame, rectangles, options, seed, pair_number=None, pair_total=None):
    rng = random.Random(seed)
    first, second, merged = rectangles
    crops = [_crop(frame, rect) for rect in rectangles]
    shapes = [_actual_shape(adapter, crop) for crop in crops]
    observed = []
    started = last_reported = time.monotonic()
    for repetition in range(options["repetitions"]):
        # Vary method and separate-call order, so a fixed thermal trend does not
        # systematically favor one decision.
        separate_first = (repetition % 2 == 0)
        pair_order = [0,1]
        rng.shuffle(pair_order)
        slots = ("separate", "merged") if separate_first else ("merged", "separate")
        values = {}
        execution_fields = {}
        for slot in slots:
            if slot == "separate":
                for index in pair_order:
                    actual, latency, execution = _latency(adapter, crops[index], options["timing_mode"])
                    if actual != shapes[index]:
                        raise RuntimeError("Network tensor shape changed within one pair")
                    values[index] = latency
                    execution_fields.update({f"r{index+1}_{key}": value for key, value in execution.items()})
            else:
                actual, latency, execution = _latency(adapter, crops[2], options["timing_mode"])
                if actual != shapes[2]:
                    raise RuntimeError("Merged network tensor shape changed")
                values[2] = latency
                execution_fields.update({f"merged_{key}": value for key, value in execution.items()})
        observed.append(dict(repetition=repetition, order="separate_first" if separate_first else "merged_first",
                             separate_call_order=json.dumps([f"r{i+1}" for i in pair_order]),
                             r1_ms=values[0], r2_ms=values[1], separate_ms=values[0]+values[1],
                             merged_ms=values[2], delta_ms=values[0]+values[1]-values[2],
                             **{key: execution_fields[key] for key in sorted(execution_fields)}))
        if pair_number is not None:
            last_reported = _progress(f"Pair {pair_number}/{pair_total}", repetition + 1,
                                      options["repetitions"], started, last_reported,
                                      options["progress_interval_s"], include_edges=False)
    medians = [float(np.median([item[key] for item in observed])) for key in ("r1_ms", "r2_ms", "merged_ms")]
    # Paired median is the oracle metric; separately report the component medians.
    separate = float(np.median([item["separate_ms"] for item in observed]))
    merged_ms = float(np.median([item["merged_ms"] for item in observed]))
    return shapes, medians, separate, merged_ms, observed


def _plots(rows, summary, table, tau, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["area_extra"] for r in rows]
    y = [r["actual_delta_ms"] for r in rows]
    fig, ax = plt.subplots()
    ax.scatter(x,y,s=14)
    ax.axvline(tau,color="tab:red",label="affine τ")
    ax.axhline(0,color="black",linewidth=.8)
    ax.set(xlabel="Extra bounding area (px²)",ylabel="Separate − merged latency (ms)")
    ax.legend(); fig.tight_layout(); fig.savefig(output/"actual_delta_vs_area.png"); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,("affine_delta_ms","shape_delta_ms"),("Affine","Shape lookup")):
        ax.scatter([r[key] for r in rows],y,s=14)
        lo = min(min(y),min(r[key] for r in rows))
        hi = max(max(y),max(r[key] for r in rows))
        ax.plot([lo,hi],[lo,hi],color="black",linewidth=.8)
        ax.set(xlabel=f"Predicted ΔT, {title} (ms)",ylabel="Actual ΔT (ms)")
    fig.tight_layout(); fig.savefig(output/"predicted_vs_actual.png"); plt.close(fig)
    names = [s["method"] for s in summary if s["method"] not in ("no_merge","oracle")]
    fig, ax = plt.subplots(); ax.bar(names,[next(s["decision_accuracy"] for s in summary if s["method"]==name) for name in names]); ax.set(ylabel="Decision accuracy",ylim=(0,1)); ax.tick_params(axis="x",rotation=30); fig.tight_layout(); fig.savefig(output/"decision_accuracy.png"); plt.close(fig)
    fig, ax = plt.subplots(); ax.bar([s["method"] for s in summary],[s["mean_effective_latency_ms"] for s in summary]); ax.set(ylabel="Mean effective latency (ms)"); ax.tick_params(axis="x",rotation=30); fig.tight_layout(); fig.savefig(output/"effective_latency.png"); plt.close(fig)
    hs = sorted(set(h for h,w in table)); ws = sorted(set(w for h,w in table))
    values = np.full((len(hs),len(ws)),np.nan)
    for (h,w), latency in table.items(): values[hs.index(h),ws.index(w)] = latency
    fig, ax = plt.subplots(); image = ax.imshow(values,origin="lower",aspect="auto"); ax.set(xticks=range(len(ws)),xticklabels=ws,yticks=range(len(hs)),yticklabels=hs,xlabel="Tensor width",ylabel="Tensor height"); fig.colorbar(image,ax=ax,label="Median latency (ms)"); fig.tight_layout(); fig.savefig(output/"shape_latency_map.png"); plt.close(fig)


def run(config_path, output_dir, experiment_config=None, replay_pairs=None):
    run_started = time.monotonic()
    config_path = Path(config_path)
    LOG.info("Stage 1/7: loading configuration from %s", config_path)
    config = load_config(str(config_path))
    options = {**DEFAULTS, **config.get("cost_aware_merging", {})}
    if experiment_config:
        import yaml
        options.update(yaml.safe_load(Path(experiment_config).read_text(encoding="utf-8")) or {})
    replay_payload = None
    if replay_pairs:
        replay_payload = json.loads(Path(replay_pairs).read_text(encoding="utf-8"))
        validate_pairs(replay_payload, options["canvas_hw"])
        options["pair_count"] = len(replay_payload["pairs"])
        if replay_payload.get("frame_manifest"):
            options["calibration_frames"] = len(replay_payload["frame_manifest"]["calibration"])
            options["frame_count"] = len(replay_payload["frame_manifest"]["evaluation"])
        else:
            LOG.warning("Legacy pairs file: replay fixes geometry only; frame selection uses current seed/settings")
    if options["timing_mode"] not in ("inference_only","detector_call"):
        raise ValueError("timing_mode must be inference_only or detector_call")
    if min(options["frame_count"],options["calibration_frames"],options["repetitions"],options["calibration_repetitions"]) < 1:
        raise ValueError("Frame and repetition counts must be positive")
    if options["repetitions"] < 2 or options["repetitions"] % 2:
        raise ValueError("Pair repetitions must be an even number of at least two for balanced order")
    if options["frame_count"] < 2:
        raise ValueError("At least two frames are required to separate calibration and evaluation")
    if options["progress_interval_s"] <= 0:
        raise ValueError("progress_interval_s must be positive")
    LOG.info("Run settings: model=%s, dataset=%s, split=%s, canvas=%s, calibration frames=%d, "
             "evaluation frames=%d, pairs=%d, timing=%s",
             config["train_params"]["model"], config["train_params"]["dataset"], options["split"],
             options["canvas_hw"], options["calibration_frames"], options["frame_count"],
             options["pair_count"], options["timing_mode"])
    torch.manual_seed(options["seed"])
    np.random.seed(options["seed"])
    LOG.info("Stage 2/7: applying CPU settings and loading model")
    stage_started = time.monotonic()
    runtime = apply_runtime_controls(config)
    device = options.get("device", "cpu")
    if device != "cpu":
        raise ValueError("This experiment currently measures CPU only")
    model_settings = _model_settings(config, config_path, device, options.get("stride"))
    LOG.info("Checkpoint: %s; computing SHA-256", model_settings["weights"])
    checkpoint_hash = file_sha256(Path(model_settings["weights"]))
    LOG.info("Loading %s adapter on %s (stride %d)", model_settings["backend"],
             device, model_settings["stride"])
    adapter = build_adapter({"model": model_settings})
    if hasattr(adapter, "enable_execution_logging"):
        adapter.enable_execution_logging()
    if str(adapter.device) != "cpu":
        raise RuntimeError("Adapter is not running on CPU")
    LOG.info("Model ready in %.1fs", time.monotonic() - stage_started)
    LOG.info("Stage 3/7: indexing dataset and loading real frames")
    stage_started = time.monotonic()
    dataset = load_dataset(config, split=options["split"], transform_name="no_resize_transform")
    needed = options["frame_count"] + options["calibration_frames"]
    if len(dataset) < needed:
        raise ValueError(f"Dataset has {len(dataset)} frames; {needed} distinct frames required")
    rng = random.Random(options["seed"])
    indices = (resolve_frames(dataset.images_info, replay_payload["frame_manifest"])
               if replay_payload and replay_payload.get("frame_manifest")
               else rng.sample(range(len(dataset)), needed))
    LOG.info("Dataset indexed: %d frames; selected %d distinct frames", len(dataset), needed)
    frames = []
    frame_records = []
    expected_frames = ([item for group in ("calibration", "evaluation")
                        for item in replay_payload["frame_manifest"][group]]
                       if replay_payload and replay_payload.get("frame_manifest") else None)
    last_reported = time.monotonic()
    for position, index in enumerate(indices, start=1):
        frames.append(_frame(dataset,index,options["canvas_hw"]))
        record = dict(frame_key=frame_key(frames[-1][2]),
                      source_sha256=file_sha256(Path(frames[-1][2])),
                      canvas_sha256=hashlib.sha256(frames[-1][0].tobytes()).hexdigest())
        if expected_frames:
            expected = expected_frames[position-1]
            differences = "; ".join(
                f"{key}: expected {expected.get(key)!r}, got {record.get(key)!r}"
                for key in sorted(record.keys() | expected.keys())
                if key != "canvas_sha256" and record.get(key) != expected.get(key))
            if differences:
                raise ValueError(f"Replay frame content differs: {record['frame_key']} "
                                 f"({differences}); loaded from {frames[-1][2]}")
        frame_records.append(record)
        last_reported = _progress("Frame loading", position, needed, stage_started,
                                  last_reported, options["progress_interval_s"])
    calibration_frames = frames[:options["calibration_frames"]]
    evaluation_frames = frames[options["calibration_frames"]:]
    frame_manifest = dict(calibration=frame_records[:options["calibration_frames"]],
                          evaluation=frame_records[options["calibration_frames"]:])
    resolve_frames(dataset.images_info, frame_manifest)
    evaluation_by_key = {record["frame_key"]: frame for record, frame in
                         zip(frame_manifest["evaluation"], evaluation_frames)}
    LOG.info("Frames ready in %.1fs: %d calibration, %d evaluation",
             time.monotonic() - stage_started, len(calibration_frames), len(evaluation_frames))
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {out}")
    out.mkdir(parents=True,exist_ok=True)
    LOG.info("Stage 4/7: warming model (%d calls)", options["warmup"])
    stage_started = last_reported = time.monotonic()
    warmup_crop = calibration_frames[0][0][:32,:32].copy()
    for position in range(1, options["warmup"] + 1):
        _latency(adapter,warmup_crop,options["timing_mode"])
        last_reported = _progress("Warmup", position, options["warmup"], stage_started,
                                  last_reported, options["progress_interval_s"])
    LOG.info("Warmup complete in %.1fs", time.monotonic() - stage_started)
    LOG.info("Stage 5/7: profiling input shapes and fitting cost estimators")
    stage_started = time.monotonic()
    table, fit, calibration_raw = _profile(adapter,calibration_frames,options)
    k, c = fit.coefficients["b0"],fit.coefficients["b1"]
    tau = k/c
    LOG.info("Profiling complete in %.1fs", time.monotonic() - stage_started)
    LOG.info("Stage 6/7: %s rectangle pairs", "replaying" if replay_pairs else "generating")
    pairs, skips = _pair_definitions(options,tau,replay_pairs)
    if not pairs: raise ValueError("No valid pairs were generated")
    for index, pair in enumerate(pairs):
        if not replay_payload or not replay_payload.get("frame_manifest"):
            pair["frame_key"] = frame_manifest["evaluation"][index % len(evaluation_frames)]["frame_key"]
    pair_payload = dict(schema_version=2, canvas_hw=options["canvas_hw"],
                        seed=replay_payload.get("seed", options["seed"]) if replay_payload else options["seed"],
                        sampling_tau_pixels=replay_payload.get("sampling_tau_pixels") if replay_payload else tau,
                        generation_skips=skips, frame_manifest=frame_manifest, pairs=pairs)
    geometry_hash = validate_pairs(pair_payload, options["canvas_hw"])
    pair_payload["pairs_geometry_sha256"] = geometry_hash
    workload_hash = canonical_hash(dict(geometry_sha256=geometry_hash, frame_manifest=frame_manifest,
                                       assignments=[(pair["pair_id"], pair["frame_key"]) for pair in pairs]))
    # Boundary labels in the portable pairs belong to the generating run's calibration.
    current_regions = []
    for pair in pairs:
        ratio = pair_geometry(Rectangle(*pair["r1"]), Rectangle(*pair["r2"]))["area_extra"] / tau
        current_regions.append("near" if .8 <= ratio <= 1.2 else "below" if ratio < .8 else "above")
    boundary_counts = {region: current_regions.count(region) for region in ("below", "near", "above")}
    if min(boundary_counts.values()) == 0:
        LOG.warning("Affine boundary coverage is incomplete: %s", boundary_counts)
    LOG.info("Pair set ready: %d pairs; boundary regions=%s; skipped placements=%s",
             len(pairs), boundary_counts, skips)
    write_json(out/"pairs.json",pair_payload)
    for row in calibration_raw: append_csv(out/"calibration_raw.csv",row,list(row))
    write_json(out/"calibration.json",dict(K_s=k,c_s_per_pixel=c,tau_pixels=tau,
                                            R2=fit.r2,MAE_s=fit.mae,RMSE_s=fit.rmse,
                                            shape_lookup={f"{h}x{w}":v for (h,w),v in table.items()}))
    metadata = dict(system_metadata(),config_path=str(config_path.resolve()),resolved_config=config,
                    experiment_options=options,model=model_settings,dataset=config["train_params"]["dataset"],
                    frame_indices=indices,frame_paths=[frame[2] for frame in frames],
                    calibration_frame_indices=indices[:options["calibration_frames"]],
                    evaluation_frame_indices=indices[options["calibration_frames"]:],
                    timing_boundary=options["timing_mode"],runtime=runtime,
                    preprocessing=adapter.preprocessing_metadata(),generation_skips=skips,
                    pair_count=len(pairs),boundary_counts=boundary_counts,
                    checkpoint_sha256=checkpoint_hash,
                    pairs_geometry_sha256=geometry_hash, workload_sha256=workload_hash,
                    frame_manifest=frame_manifest,
                    execution_logging=dict(source="successful forward feature tensor shapes",
                        enabled=hasattr(adapter, "execution_metadata"),
                        timing_overhead="Shape tuple capture inside forward; serialization and file writes outside timing"),
                    calibration=dict(K_s=k,c_s_per_pixel=c,tau_pixels=tau),
                    replay_source=str(replay_pairs) if replay_pairs else None)
    write_json(out/"metadata.json",metadata)
    rows = []
    observations_path = out/"pair_observations.csv"
    raw_path = out/"pairs_raw.csv"
    LOG.info("Measuring %d pairs × %d repetitions × 3 model calls = %d model calls",
             len(pairs), options["repetitions"], 3*len(pairs)*options["repetitions"])
    stage_started = time.monotonic()
    for index, pair in enumerate(pairs):
        first, second = Rectangle(*pair["r1"]), Rectangle(*pair["r2"])
        merged = union_rectangle(first,second)
        frame, original_hw, path = evaluation_by_key[pair["frame_key"]]
        if any(r.x1<0 or r.y1<0 or r.x2>frame.shape[1] or r.y2>frame.shape[0] for r in (first,second,merged)):
            raise ValueError(f"Replay pair {index} exceeds frame canvas")
        shapes, component, separate_ms, merged_ms, observations = _measure_pair(
            adapter,frame,(first,second,merged),options,options["seed"]+index,
            pair_number=index+1,pair_total=len(pairs))
        affine_costs = [1000*(k+c*h*w) for h,w in shapes]
        lookup_values = [nearest_shape_cost(table,shape) for shape in shapes]
        lookup_costs = [entry[0] for entry in lookup_values]
        decisions = policy_decisions(first,second,(merged_ms,component[0],component[1]),
                                     (affine_costs[2],affine_costs[0],affine_costs[1]),
                                     (lookup_costs[2],lookup_costs[0],lookup_costs[1]),
                                     gamma=options["geometric_gamma"],simple_iou=options["simple_iou"],
                                     simple_distance=options["simple_distance"])
        # Oracle uses the paired separate total rather than a sum of component medians.
        decisions["oracle"] = merged_ms < separate_ms
        row = dict(pair_id=pair["pair_id"],frame_id=path,frame_key=pair["frame_key"],
                   frame_canvas_sha256=next(r["canvas_sha256"] for r in frame_manifest["evaluation"]
                                            if r["frame_key"] == pair["frame_key"]),
                   original_frame_h=original_hw[0],
                   original_frame_w=original_hw[1],frame_h=frame.shape[0],frame_w=frame.shape[1],
                   geometry_type=pair["geometry_type"],boundary_region=current_regions[index],
                   source_boundary_region=pair["boundary_region"],
                   **pair_geometry(first,second),actual_r1_ms=component[0],actual_r2_ms=component[1],
                   actual_separate_ms=separate_ms,actual_merged_ms=merged_ms,
                   actual_delta_ms=separate_ms-merged_ms,
                   affine_r1_ms=affine_costs[0],affine_r2_ms=affine_costs[1],
                   affine_merged_ms=affine_costs[2],affine_delta_ms=affine_costs[0]+affine_costs[1]-affine_costs[2],
                   shape_r1_ms=lookup_costs[0],shape_r2_ms=lookup_costs[1],
                   shape_merged_ms=lookup_costs[2],shape_delta_ms=lookup_costs[0]+lookup_costs[1]-lookup_costs[2])
        for i,label in enumerate(("r1","r2","merged")):
            row[f"{label}_tensor_h"],row[f"{label}_tensor_w"] = shapes[i]
            row[f"{label}_lookup_h"],row[f"{label}_lookup_w"] = lookup_values[i][1]
            for key in ("execution_trace_status", "active_depth", "active_head_count", "active_head_indices", "feature_maps"):
                row[f"{label}_{key}"] = observations[0][f"{label}_{key}"]
        for name,value in decisions.items():
            row[f"{name}_decision"] = value
            row[f"{name}_correct"] = value == decisions["oracle"]
        rows.append(row)
        append_csv(raw_path,row,list(row))
        for item in observations:
            append_csv(observations_path,{"pair_id":pair["pair_id"],**item},["pair_id",*item])
        completed = index + 1
        elapsed = time.monotonic() - stage_started
        eta = elapsed / completed * (len(pairs) - completed)
        LOG.info("Pair %d/%d complete: %s, actual gain %.3f ms, elapsed %.1fs, "
                 "estimated remaining %.1fs", completed, len(pairs), pair["geometry_type"],
                 row["actual_delta_ms"], elapsed, eta)
    LOG.info("Pair measurement complete in %.1fs", time.monotonic() - stage_started)
    LOG.info("Stage 7/7: writing summaries and plots")
    summary = summarize(rows)
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    write_json(out/"summary.json",summary)
    shape_count = 3*len(rows)
    exact_shapes = sum((row[f"{label}_tensor_h"],row[f"{label}_tensor_w"]) in table
                       for row in rows for label in ("r1","r2","merged"))
    metadata["lookup_exact_shape_fraction"] = exact_shapes/shape_count
    if exact_shapes/shape_count < .5:
        LOG.warning("Only %d/%d evaluated tensor shapes have exact lookup entries",
                    exact_shapes, shape_count)
    write_json(out/"metadata.json",metadata)
    equal_area = []
    by_area = {}
    for shape,latency in table.items(): by_area.setdefault(shape[0]*shape[1],[]).append((shape,latency))
    for area,items in by_area.items():
        if len(items)>1:
            equal_area.append(dict(area=area,shapes=[{"height":h,"width":w,"latency_ms":t} for (h,w),t in items],
                                   latency_spread_ms=max(t for _,t in items)-min(t for _,t in items)))
    write_json(out/"equal_area_shapes.json",equal_area)
    _plots(rows,summary,table,tau,out)
    LOG.info("Experiment complete in %.1fs; results written to %s",
             time.monotonic() - run_started, out.resolve())
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,help="Existing repository model/dataset YAML")
    parser.add_argument("--output",required=True)
    parser.add_argument("--experiment-config",help="Optional YAML mapping overriding experiment defaults")
    parser.add_argument("--pairs",help="Replay geometry and frame manifest from another run's pairs.json")
    args = parser.parse_args()
    run(args.config,args.output,args.experiment_config,args.pairs)


if __name__ == "__main__":
    main()
