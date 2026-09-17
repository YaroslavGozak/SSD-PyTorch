"""Collect paired separate-vs-union ROI timings for Experiment B."""

import argparse
import csv
import random
import time
import uuid
from pathlib import Path

import numpy as np

from .common import append_csv, deterministic_image, file_sha256, load_config, read_cpu_freq, read_cpu_temp, system_metadata, write_json
from .geometry import Rectangle, union_rectangle
from .models import control_predictions, decide_merge, load_latency_models
from .collect_experiment_a import build_adapter
from .timing import measure
from .pair_specs import load as load_pair_specs
from .common import collection_provenance
from .geometry import stride_rounded_shape
from .reproducibility import gate


FIELDS = ["order_block_id", "position_in_order_block", "primary_stratum", "stratum_tags", "session_id", "pair_id", "repetition", "order", "timing_mode", "geometry_type", "boundary_bin", "r1_x1", "r1_y1", "r1_x2", "r1_y2", "r2_x1", "r2_y1", "r2_x2", "r2_y2", "union_x1", "union_y1", "union_x2", "union_y2", "r1_tensor_w", "r1_tensor_h", "r2_tensor_w", "r2_tensor_h", "union_tensor_w", "union_tensor_h", "a1_effective", "a2_effective", "au_effective", "delta_effective_area", "tau_used", "predicted_merge", "linear_tau_predicted_merge", "quadratic_direct_predicted_merge", "piecewise_direct_predicted_merge", "linear_predicted_gain_ms", "quadratic_predicted_gain_ms", "piecewise_predicted_gain_ms", "linear_predicted_merged_cost_ms", "linear_predicted_separate_cost_ms", "quadratic_predicted_merged_cost_ms", "quadratic_predicted_separate_cost_ms", "piecewise_predicted_merged_cost_ms", "piecewise_predicted_separate_cost_ms", "separate_ms", "merged_ms", "difference_ms", "cpu_temp_c", "cpu_freq_mhz", "elapsed_s", "timestamp_utc"]


def generate_pairs(count: int, canvas_hw, tau: float, seed: int, quotas=None):
    """Generate diverse geometric pairs with explicit computational strata."""
    rng = random.Random(seed)
    canvas_h, canvas_w = map(int, canvas_hw)
    quotas = quotas or {"merge": .25, "near_low": .20, "near_high": .20, "separate": .35}
    targets = {name: int(round(count * fraction)) for name, fraction in quotas.items()}
    targets[list(targets)[-1]] += count - sum(targets.values())
    accepted = {name: [] for name in targets}
    keys = set()
    max_dim = max(16, min(canvas_h, canvas_w))
    attempts = 0
    geometry_types = ("horizontal_separation", "vertical_separation", "diagonal_separation", "partial_overlap", "containment", "containment", "containment")
    while sum(len(items) for items in accepted.values()) < count and attempts < count * 3000:
        attempts += 1
        remaining_bins = [name for name in targets if len(accepted[name]) < targets[name]]
        desired_bin = remaining_bins[0]
        geometry = geometry_types[attempts % len(geometry_types)]
        w1, h1 = rng.randint(16, max_dim), rng.randint(16, max_dim)
        w2, h2 = rng.randint(16, max_dim), rng.randint(16, max_dim)
        x1, y1 = rng.randint(0, canvas_w - w1), rng.randint(0, canvas_h - h1)
        if desired_bin in {"near_low", "near_high", "separate"} and geometry in {"horizontal_separation", "vertical_separation"}:
            target_ratio = {"near_low": .9, "near_high": 1.1, "separate": 1.35}[desired_bin]
            shared_h = rng.randint(16, max(16, canvas_h // 2))
            w1 = rng.randint(16, max(16, min(canvas_w // 4, 96)))
            w2 = rng.randint(16, max(16, min(canvas_w // 4, 96)))
            gap = round(target_ratio * tau / max(shared_h, 1))
            if geometry == "horizontal_separation" and w1 + w2 + gap <= canvas_w:
                h1 = h2 = shared_h
                x1, y1 = 0, rng.randint(0, canvas_h - h1)
                x2, y2 = x1 + w1 + gap, y1
            elif geometry == "vertical_separation":
                gap = round(target_ratio * tau / max(w1, 1))
                if h1 + h2 + gap > canvas_h:
                    continue
                x1, y1 = rng.randint(0, canvas_w - w1), 0
                x2, y2 = x1, y1 + h1 + gap
            else:
                continue
        elif geometry == "horizontal_separation":
            x2, y2 = rng.randint(0, max(0, canvas_w - w2)), y1 if h2 <= canvas_h - y1 else rng.randint(0, canvas_h - h2)
            x2 = min(canvas_w - w2, max(0, x1 + w1 + rng.randint(0, max(0, canvas_w - x1 - w1 - w2))))
        elif geometry == "vertical_separation":
            x2, y2 = x1 if w2 <= canvas_w - x1 else rng.randint(0, canvas_w - w2), rng.randint(0, max(0, canvas_h - h2))
            y2 = min(canvas_h - h2, max(0, y1 + h1 + rng.randint(0, max(0, canvas_h - y1 - h1 - h2))))
        elif geometry == "diagonal_separation":
            x2, y2 = canvas_w - w2, canvas_h - h2
        elif geometry == "containment":
            w1, h1 = max(w1, w2), max(h1, h2)
            x1, y1 = rng.randint(0, canvas_w - w1), rng.randint(0, canvas_h - h1)
            x2, y2 = x1 + rng.randint(0, w1 - w2), y1 + rng.randint(0, h1 - h2)
        else:
            x2 = min(canvas_w - w2, max(0, x1 + rng.randint(-w2 // 2, w1 // 2)))
            y2 = min(canvas_h - h2, max(0, y1 + rng.randint(-h2 // 2, h1 // 2)))
        first = Rectangle(x1, y1, x1 + w1, y1 + h1)
        second = Rectangle(x2, y2, x2 + w2, y2 + h2)
        union = union_rectangle(first, second)
        ratio = (union.area - first.area - second.area) / tau
        if ratio < .8:
            boundary_bin = "merge"
        elif ratio < 1.0:
            boundary_bin = "near_low"
        elif ratio <= 1.2:
            boundary_bin = "near_high"
        elif ratio <= 1.5:
            boundary_bin = "separate"
        else:
            continue
        key = (first.height, first.width, second.height, second.width, union.height, union.width)
        if len(accepted[boundary_bin]) >= targets[boundary_bin] or key in keys:
            continue
        keys.add(key)
        accepted[boundary_bin].append({"first": first, "second": second, "union": union,
                                       "delta": union.area - first.area - second.area,
                                       "boundary_bin": boundary_bin, "geometry_type": geometry})
    pairs = [pair for items in accepted.values() for pair in items]
    if len(pairs) < count:
        counts = {name: len(items) for name, items in accepted.items()}
        raise RuntimeError(f"Could only generate {len(pairs)} of {count} pairs; strata={counts}, targets={targets}, attempts={attempts}")
    rng.shuffle(pairs)
    return pairs


def balanced_orders(repetitions: int, seed: int):
    if repetitions < 2 or repetitions % 2:
        raise ValueError("Experiment B repetitions must be a positive even number")
    blocks = []
    for block_id in range(repetitions // 4):
        blocks.append(["merged_first", "separate_first", "separate_first", "merged_first"] if block_id % 2 == 0 else ["separate_first", "merged_first", "merged_first", "separate_first"])
    random.Random(seed).shuffle(blocks)
    if repetitions % 4:
        blocks.append(["separate_first", "merged_first"])
    return [order for block in blocks for order in block][:repetitions]


def validate_complete_experiment_a(output_dir: Path, config, expected_shapes: int):
    raw_path = output_dir / "experiment_a_raw.csv"
    if not raw_path.exists():
        raise RuntimeError(f"Experiment A raw observations are missing: {raw_path}")
    expected_repetitions = int(config.get("experiment_a", {}).get("repetitions_per_effective_shape", 40))
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    run_counts = {}
    for row in rows:
        run_counts[row.get("run_id", "")] = run_counts.get(row.get("run_id", ""), 0) + 1
    if len(run_counts) != expected_repetitions or any(count != expected_shapes for count in run_counts.values()):
        raise RuntimeError(
            f"Experiment A is incomplete: expected {expected_repetitions} runs x {expected_shapes} shapes, "
            f"found {len(run_counts)} runs with counts {sorted(set(run_counts.values()))}"
        )
    return len(rows)


def collect(config, fit_path: str, output: str, tau_override: float | None = None, overwrite: bool = False, pairs_path: str | None = None):
    import json
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "experiment_b_raw.csv"
    fit = json.loads(Path(fit_path).read_text(encoding="utf-8"))
    if tau_override is not None:
        raise ValueError("Frozen validation uses calibration coefficients; tau override is not supported")
    tau = float(fit["tau_pixels"])
    if tau <= 0:
        raise ValueError("Experiment B requires positive tau; use --tau-override explicitly for a manual override")
    experiment = config.get("experiment_b", {})
    latency_models = load_latency_models(fit)
    progress_every = max(1, int(experiment.get("progress_every_pairs", 1)))
    progress_every_repetitions = max(1, int(experiment.get("progress_every_repetitions", 5)))
    seed = int(config.get("seed", 0))
    adapter = build_adapter(config)
    mode = str(experiment.get("timing_mode", "inference_only"))
    current = collection_provenance(config,adapter,mode)
    quality_warnings = []
    gate(not config.get("publication_run",False) or not current["git_dirty"], "git_dirty", config, quality_warnings)
    for key in ("weights_sha256", "backend", "device", "preprocessing", "model_stride", "timing_mode", "dtype", "shape_policy"):
        if key in fit.get("provenance", {}) and current.get(key) != fit["provenance"][key]:
            raise ValueError(f"Calibration provenance mismatch: {key}")
    requested_count = int(experiment.get("pair_count", 500))
    canvas_hw = experiment.get("canvas_hw", [640,640])
    image = deterministic_image(*map(int,canvas_hw),seed)
    collection_start = time.perf_counter()
    specs_path = Path(pairs_path) if pairs_path else output_dir / "pair_specs.json"
    if not specs_path.exists():
        raise FileNotFoundError(f"Generate frozen pairs first with python -m experiments.cost_model_validation.pair_specs: {specs_path}")
    pairs, specs = load_pair_specs(specs_path,config,fit_path,adapter,image)
    computational_keys = {tuple(p["specification"]["computational_key"]) for p in pairs}
    quota_fractions = specs["payload"]["quotas"]
    domain_policy = "reject_out_of_domain"
    rejected_out_of_domain_candidate_count = rejected_out_of_domain_shape_check_count = accepted_out_of_calibration_pair_count = 0
    if raw_path.exists():
        if not overwrite:
            raise ValueError("Experiment B output already exists; use --overwrite or a new output directory")
        raw_path.unlink()
    repetitions = int(experiment.get("repetitions", 20))
    randomizer = random.Random(seed)
    session_id = str(uuid.uuid4())
    session_start = __import__("time").perf_counter()
    orders = {pair_id: balanced_orders(repetitions, seed + pair_id) for pair_id in range(len(pairs))}
    blocks = [(pair_id, list(range(start,min(start+4,repetitions)))) for pair_id in range(len(pairs)) for start in range(0,repetitions,4)]
    randomizer.shuffle(blocks)
    trials = [(pair_id,repetition) for pair_id,block in blocks for repetition in block]
    completed_pairs = set()
    pair_trial_counts = {}
    control_every_pairs = max(0, int(experiment.get("control_every_pairs", 25)))
    control_shapes = [stride_rounded_shape(*map(int, shape), adapter.stride) for shape in experiment.get("control_shapes", [[160, 160], [160, 320], [320, 320]])]
    invalid_controls = [shape for shape in control_shapes if not latency_models["linear"].is_in_domain(*shape)]
    if invalid_controls:
        raise ValueError(f"Primary control shapes are outside the calibration envelope: {invalid_controls}")
    control_records = []
    for completed_trial, (pair_id, repetition) in enumerate(trials, 1):
        pair = pairs[pair_id]
        prepared_shapes = pair["prepared"]
        shapes = [item.tensor_hw for item in prepared_shapes]
        separate_ms = merged_ms = None
        order = orders[pair_id][repetition]
        requested_shapes = pair["specification"]["requested_shapes"]
        timings = (("separate", requested_shapes[0:2]), ("merged", requested_shapes[2:3])) if order == "separate_first" else (("merged", requested_shapes[2:3]), ("separate", requested_shapes[0:2]))
        for kind, requested in timings:
            values = []
            for shape in requested:
                _, result = measure(adapter, image, shape, str(experiment.get("timing_mode", "inference_only")))
                values.append(result.inference_ms if experiment.get("timing_mode", "inference_only") == "inference_only" else result.detector_call_ms)
            if kind == "separate": separate_ms = sum(values)
            else: merged_ms = values[0]
        decisions = {name: decide_merge(model, shapes[0], shapes[1], shapes[2]) for name, model in latency_models.items()}
        linear_decision = decisions["linear"]

        def _gain_ms(name):
            return decisions[name]["predicted_gain_s"] * 1000.0 if name in decisions else "unavailable"

        def _cost_ms(name, key):
            return decisions[name][key] * 1000.0 if name in decisions else "unavailable"

        row = {"session_id": session_id, "pair_id": pair["specification"]["pair_id"], "order_block_id": repetition//4, "position_in_order_block": repetition%4, "primary_stratum": pair["boundary_bin"], "stratum_tags": json.dumps(pair["specification"]["stratum_tags"]), "repetition": repetition, "order": order, "timing_mode": experiment.get("timing_mode", "inference_only"), "geometry_type": pair["geometry_type"], "boundary_bin": pair["boundary_bin"],
                   **{f"r1_{key}": getattr(pair["first"], key) for key in ("x1", "y1", "x2", "y2")}, **{f"r2_{key}": getattr(pair["second"], key) for key in ("x1", "y1", "x2", "y2")}, **{f"union_{key}": getattr(pair["union"], key) for key in ("x1", "y1", "x2", "y2")},
                   "r1_tensor_w": shapes[0][1], "r1_tensor_h": shapes[0][0], "r2_tensor_w": shapes[1][1], "r2_tensor_h": shapes[1][0], "union_tensor_w": shapes[2][1], "union_tensor_h": shapes[2][0],
                   "a1_effective": prepared_shapes[0].effective_area, "a2_effective": prepared_shapes[1].effective_area, "au_effective": prepared_shapes[2].effective_area, "delta_effective_area": pair["delta"], "tau_used": tau, "predicted_merge": linear_decision["predicted_merge"], "linear_tau_predicted_merge": linear_decision["predicted_merge"], "quadratic_direct_predicted_merge": decisions.get("quadratic", {}).get("predicted_merge", "unavailable"), "piecewise_direct_predicted_merge": decisions.get("piecewise", {}).get("predicted_merge", "unavailable"), "linear_predicted_gain_ms": _gain_ms("linear"), "quadratic_predicted_gain_ms": _gain_ms("quadratic"), "piecewise_predicted_gain_ms": _gain_ms("piecewise"), "linear_predicted_merged_cost_ms": _cost_ms("linear", "predicted_merged_cost_s"), "linear_predicted_separate_cost_ms": _cost_ms("linear", "predicted_separate_cost_s"), "quadratic_predicted_merged_cost_ms": _cost_ms("quadratic", "predicted_merged_cost_s"), "quadratic_predicted_separate_cost_ms": _cost_ms("quadratic", "predicted_separate_cost_s"), "piecewise_predicted_merged_cost_ms": _cost_ms("piecewise", "predicted_merged_cost_s"), "piecewise_predicted_separate_cost_ms": _cost_ms("piecewise", "predicted_separate_cost_s"), "separate_ms": separate_ms, "merged_ms": merged_ms, "difference_ms": separate_ms - merged_ms, "cpu_temp_c": read_cpu_temp(), "cpu_freq_mhz": read_cpu_freq(), "elapsed_s": __import__("time").perf_counter() - session_start, "timestamp_utc": system_metadata()["timestamp_utc"]}
        append_csv(raw_path, row, FIELDS)
        pair_trial_counts[pair_id] = pair_trial_counts.get(pair_id, 0) + 1
        if completed_trial % progress_every_repetitions == 0 or completed_trial == len(trials):
            elapsed = time.perf_counter() - collection_start
            rate = completed_trial / elapsed if elapsed > 0 else 0.0
            eta = (len(trials) - completed_trial) / rate if rate > 0 else float("nan")
            print(f"[experiment_b] trial {completed_trial}/{len(trials)} (pair {pair_id + 1}/{len(pairs)}, repetition {repetition + 1}/{repetitions}); "
                  f"{rate:.2f} trials/s; ETA {eta / 60:.1f} min", flush=True)
        if pair_trial_counts[pair_id] == repetitions and pair_id not in completed_pairs:
            completed_pairs.add(pair_id)
            if control_every_pairs and len(completed_pairs) % control_every_pairs == 0:
                for control_h, control_w in control_shapes:
                    prepared_control, control_result = measure(adapter, image, (control_h, control_w), str(experiment.get("timing_mode", "inference_only")))
                    measured_ms = control_result.inference_ms if experiment.get("timing_mode", "inference_only") == "inference_only" else control_result.detector_call_ms
                    predictions = control_predictions(latency_models, prepared_control.effective_area, measured_ms)
                    control_records.append({"pair_completed": len(completed_pairs), "requested_h": control_h, "requested_w": control_w,
                                            "tensor_h": prepared_control.tensor_h, "tensor_w": prepared_control.tensor_w,
                                            "effective_area": prepared_control.effective_area,
                                            "measured_ms": measured_ms, "predictions": predictions, "elapsed_s": time.perf_counter()-session_start, "timestamp_utc": system_metadata()["timestamp_utc"]})
                print(f"[experiment_b] control checkpoint after {len(completed_pairs)} pairs: "
                      f"max abs linear model deviation {max(abs(item['predictions']['linear']['relative_error']) for item in control_records[-len(control_shapes):]):.1%}", flush=True)
            if len(completed_pairs) % progress_every == 0 or len(completed_pairs) == len(pairs):
                print(f"[experiment_b] completed {len(completed_pairs)}/{len(pairs)} pairs; elapsed {time.perf_counter() - collection_start:.1f}s", flush=True)
    hardware_shift = any(abs(item["predictions"]["linear"]["relative_error"]) > float(experiment.get("control_drift_threshold", .10)) for item in control_records)
    drift_error = None
    try:
        gate(not hardware_shift, "control_drift", config, quality_warnings)
    except ValueError as exc:
        drift_error = exc
    write_json(output_dir / "experiment_b_metadata.json", {**system_metadata(), "config": config, "quality_warnings": quality_warnings, "tau_used": tau, "tau_override": tau_override is not None, "session_id": session_id,
                                                            "pair_count": len(pairs), "near_pair_count": sum(pair["boundary_bin"] in {"near", "near_low", "near_high"} for pair in pairs),
                                                            "unique_computational_pair_count": len(computational_keys),
                                                            "domain_policy": domain_policy,
                                                            "rejected_out_of_domain_candidate_count": rejected_out_of_domain_candidate_count,
                                                            "rejected_out_of_domain_shape_check_count": rejected_out_of_domain_shape_check_count,
                                                            "accepted_out_of_calibration_pair_count": accepted_out_of_calibration_pair_count,
                                                            "control_shapes": control_shapes, "control_records": control_records,
                                                            "schema_version": 3, "pair_specs_hash": specs["sha256"], "pair_specs": specs["payload"], "provenance": current, "bootstrap": fit.get("bootstrap", {}), "calibration_reference": {"path": str(Path(fit_path).resolve()), "content_hash": file_sha256(fit_path)}, "latency_models": {name: model.metadata() for name, model in latency_models.items()},
                                                            "calibration_provenance": fit.get("provenance", {}),
                                                            "sampling_basis": "frozen_multi_model_strata", "sampling_tau_pixels": tau, "boundary_quotas": quota_fractions,
                                                            "measurement_protocol": {"timing_mode": experiment.get("timing_mode", "inference_only"), "order_design": "balanced_abba_baab",
                                                                                      "repetitions_per_pair": repetitions, "control_every_pairs": control_every_pairs},
                                                            "cpu_temp_sensor_available": read_cpu_temp() is not None, "cpu_freq_sensor_available": read_cpu_freq() is not None,
                                                            "hardware_state_shift": hardware_shift})
    if drift_error is not None:
        raise drift_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--fit", required=True); parser.add_argument("--output", required=True); parser.add_argument("--tau-override", type=float); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pairs", help="Frozen pair_specs.json; never regenerated in measure mode")
    args = parser.parse_args(); collect(load_config(args.config), args.fit, args.output, args.tau_override, args.overwrite, args.pairs)


if __name__ == "__main__": main()
