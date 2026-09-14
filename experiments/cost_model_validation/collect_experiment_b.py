"""Collect paired separate-vs-union ROI timings for Experiment B."""

import argparse
import csv
import random
import time
import uuid
from pathlib import Path

import numpy as np

from .adapters import FakeAdapter
from .common import append_csv, deterministic_image, load_config, system_metadata, write_json
from .geometry import Rectangle, union_rectangle
from .models import merge_decision
from .collect_experiment_a import build_adapter
from .timing import measure


FIELDS = ["session_id", "pair_id", "repetition", "order", "timing_mode", "geometry_type", "boundary_bin", "r1_x1", "r1_y1", "r1_x2", "r1_y2", "r2_x1", "r2_y1", "r2_x2", "r2_y2", "union_x1", "union_y1", "union_x2", "union_y2", "r1_tensor_w", "r1_tensor_h", "r2_tensor_w", "r2_tensor_h", "union_tensor_w", "union_tensor_h", "a1_effective", "a2_effective", "au_effective", "delta_effective_area", "tau_used", "predicted_merge", "separate_ms", "merged_ms", "difference_ms", "cpu_temp_c", "cpu_freq_mhz", "elapsed_s", "timestamp_utc"]


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
    orders = ["separate_first"] * (repetitions // 2) + ["merged_first"] * (repetitions // 2)
    random.Random(seed).shuffle(orders)
    return orders


def validate_complete_experiment_a(output_dir: Path, config, expected_shapes: int):
    raw_path = output_dir / "experiment_a_raw.csv"
    if not raw_path.exists():
        raise RuntimeError(f"Experiment A raw observations are missing: {raw_path}")
    expected_repetitions = max(30, int(config.get("experiment_a", {}).get("repetitions_per_effective_shape", 40)))
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


def collect(config, fit_path: str, output: str, tau_override: float | None = None, overwrite: bool = False):
    import json
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "experiment_b_raw.csv"
    if raw_path.exists() and overwrite:
        raw_path.unlink()
    fit = json.loads(Path(fit_path).read_text(encoding="utf-8"))
    tau = float(tau_override if tau_override is not None else fit["tau_pixels"])
    if tau <= 0:
        raise ValueError("Experiment B requires positive tau; use --tau-override explicitly for a manual override")
    experiment = config.get("experiment_b", {})
    progress_every = max(1, int(experiment.get("progress_every_pairs", 1)))
    progress_every_repetitions = max(1, int(experiment.get("progress_every_repetitions", 5)))
    seed = int(config.get("seed", 0))
    adapter = build_adapter(config)
    with (output_dir / "experiment_a_raw.csv").open(newline="", encoding="utf-8") as handle:
        expected_shapes = len({(row["tensor_h"], row["tensor_w"]) for row in csv.DictReader(handle)})
    a_rows = validate_complete_experiment_a(output_dir, config, expected_shapes=expected_shapes)
    requested_count = int(experiment.get("pair_count", 500))
    canvas_hw = experiment.get("canvas_hw", [640, 640])
    collection_start = time.perf_counter()
    print(f"[experiment_b] preparing {requested_count} pairs from complete Experiment A ({a_rows} rows); canvas={canvas_hw}, tau={tau:.3f}", flush=True)
    image = deterministic_image(*map(int, canvas_hw), seed)
    pairs = []
    computational_keys = set()
    quota_fractions = experiment.get("boundary_quotas") or {"merge": .25, "near_low": .20, "near_high": .20, "separate": .35}
    quota_targets = {name: int(round(requested_count * fraction)) for name, fraction in quota_fractions.items()}
    quota_targets[list(quota_targets)[-1]] += requested_count - sum(quota_targets.values())
    selected_by_bin = {name: 0 for name in quota_targets}
    envelope = fit.get("calibration_envelope")
    domain_policy = str(experiment.get("domain_policy", "reject_out_of_domain"))
    out_of_domain_pairs = 0
    out_of_domain_invocations = 0
    rejection_counts = {"out_of_domain": 0, "duplicate": 0, "outside_boundary_bins": 0, "quota_full": 0}
    batch_size = max(requested_count * 10, 1000)
    max_batches = max(1, int(experiment.get("max_pair_generation_batches", 20)))
    for batch_id in range(max_batches):
        if len(pairs) >= requested_count:
            break
        candidates = generate_pairs(
            batch_size, canvas_hw, tau, seed + batch_id,
            quotas=experiment.get("boundary_quotas"),
        )
        for pair in candidates:
            prepared = [adapter.prepare(image, (pair[name].height, pair[name].width)) for name in ("first", "second", "union")]
            areas = [item.effective_area for item in prepared]
            if envelope:
                in_domain = all(envelope["min_effective_area"] <= item.effective_area <= envelope["max_effective_area"] and
                                envelope["min_tensor_h"] <= item.tensor_h <= envelope["max_tensor_h"] and
                                envelope["min_tensor_w"] <= item.tensor_w <= envelope["max_tensor_w"] and
                                envelope.get("min_aspect_ratio", 0.0) <= item.tensor_w / item.tensor_h <= envelope.get("max_aspect_ratio", float("inf"))
                                for item in prepared)
                if not in_domain:
                    out_of_domain_pairs += 1
                    out_of_domain_invocations += 3
                    rejection_counts["out_of_domain"] += 1
                    if domain_policy == "reject_out_of_domain":
                        continue
            key = tuple(value for item in prepared for value in item.tensor_hw)
            if key in computational_keys:
                rejection_counts["duplicate"] += 1
                continue
            pair["prepared"] = prepared
            pair["delta"] = areas[2] - areas[0] - areas[1]
            ratio = pair["delta"] / tau
            if ratio < .8:
                pair["boundary_bin"] = "merge"
            elif ratio < 1.0:
                pair["boundary_bin"] = "near_low"
            elif ratio <= 1.2:
                pair["boundary_bin"] = "near_high"
            elif ratio <= 1.5:
                pair["boundary_bin"] = "separate"
            else:
                rejection_counts["outside_boundary_bins"] += 1
                continue
            if pair["boundary_bin"] not in quota_targets:
                rejection_counts["outside_boundary_bins"] += 1
                continue
            if selected_by_bin[pair["boundary_bin"]] >= quota_targets[pair["boundary_bin"]]:
                rejection_counts["quota_full"] += 1
                continue
            computational_keys.add(key)
            pairs.append(pair)
            selected_by_bin[pair["boundary_bin"]] += 1
            if len(pairs) >= requested_count:
                break
        print(f"[experiment_b] candidate batch {batch_id + 1}/{max_batches}: "
              f"selected {len(pairs)}/{requested_count}; strata={selected_by_bin}", flush=True)
    if len(pairs) < requested_count:
        raise RuntimeError(f"Could only generate {len(pairs)} unique computational pairs of {requested_count} requested "
                           f"after {max_batches} batches; strata={selected_by_bin}, rejections={rejection_counts}. "
                           "Increase max_pair_generation_batches, widen the calibration envelope, or use expand_calibration.")
    minimum_near = int(experiment.get("minimum_near_pairs", round(requested_count * .4)))
    near_count = sum(pair["boundary_bin"] in {"near", "near_low", "near_high"} for pair in pairs)
    if near_count < minimum_near:
        raise RuntimeError(f"Only {near_count} unique near-boundary pairs generated; required {minimum_near}")
    print(f"[experiment_b] prepared {len(pairs)} unique pairs ({near_count} near-boundary) in {time.perf_counter() - collection_start:.1f}s", flush=True)
    repetitions = max(20, int(experiment.get("repetitions", 20)))
    randomizer = random.Random(seed)
    session_id = str(uuid.uuid4())
    session_start = __import__("time").perf_counter()
    orders = {pair_id: balanced_orders(repetitions, seed + pair_id) for pair_id in range(len(pairs))}
    trials = [(pair_id, repetition) for pair_id in range(len(pairs)) for repetition in range(repetitions)]
    randomizer.shuffle(trials)
    completed_pairs = set()
    pair_trial_counts = {}
    control_every_pairs = max(0, int(experiment.get("control_every_pairs", 25)))
    control_shapes = [tuple(map(int, shape)) for shape in experiment.get("control_shapes", [[64, 64], [160, 160], [160, 320], [320, 320]])]
    control_records = []
    fit_k = float(fit.get("K_t_s", 0.0))
    fit_c = float(fit.get("c_t_s_per_pixel", 0.0))
    for completed_trial, (pair_id, repetition) in enumerate(trials, 1):
        pair = pairs[pair_id]
        prepared_shapes = pair["prepared"]
        shapes = [item.tensor_hw for item in prepared_shapes]
        separate_ms = merged_ms = None
        order = orders[pair_id][repetition]
        timings = (("separate", shapes[0:2]), ("merged", shapes[2:3])) if order == "separate_first" else (("merged", shapes[2:3]), ("separate", shapes[0:2]))
        for kind, requested in timings:
            values = []
            for shape in requested:
                _, result = measure(adapter, image, shape, str(experiment.get("timing_mode", "inference_only")))
                values.append(result.inference_ms if experiment.get("timing_mode", "inference_only") == "inference_only" else result.detector_call_ms)
            if kind == "separate": separate_ms = sum(values)
            else: merged_ms = values[0]
        row = {"session_id": session_id, "pair_id": pair_id, "repetition": repetition, "order": order, "timing_mode": experiment.get("timing_mode", "inference_only"), "geometry_type": pair["geometry_type"], "boundary_bin": pair["boundary_bin"],
                   **{f"r1_{key}": getattr(pair["first"], key) for key in ("x1", "y1", "x2", "y2")}, **{f"r2_{key}": getattr(pair["second"], key) for key in ("x1", "y1", "x2", "y2")}, **{f"union_{key}": getattr(pair["union"], key) for key in ("x1", "y1", "x2", "y2")},
                   "r1_tensor_w": shapes[0][1], "r1_tensor_h": shapes[0][0], "r2_tensor_w": shapes[1][1], "r2_tensor_h": shapes[1][0], "union_tensor_w": shapes[2][1], "union_tensor_h": shapes[2][0],
                   "a1_effective": prepared_shapes[0].effective_area, "a2_effective": prepared_shapes[1].effective_area, "au_effective": prepared_shapes[2].effective_area, "delta_effective_area": pair["delta"], "tau_used": tau, "predicted_merge": merge_decision(pair["delta"], tau), "separate_ms": separate_ms, "merged_ms": merged_ms, "difference_ms": separate_ms - merged_ms, "elapsed_s": __import__("time").perf_counter() - session_start, "timestamp_utc": system_metadata()["timestamp_utc"]}
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
                    _, control_result = measure(adapter, image, (control_h, control_w), str(experiment.get("timing_mode", "inference_only")))
                    measured_ms = control_result.inference_ms if experiment.get("timing_mode", "inference_only") == "inference_only" else control_result.detector_call_ms
                    prepared_control = adapter.prepare(image, (control_h, control_w))
                    predicted_ms = (fit_k + fit_c * prepared_control.effective_area) * 1000.0
                    relative_error = (measured_ms - predicted_ms) / predicted_ms if predicted_ms > 0 else float("nan")
                    control_records.append({"pair_completed": len(completed_pairs), "requested_h": control_h, "requested_w": control_w, "tensor_h": prepared_control.tensor_h, "tensor_w": prepared_control.tensor_w, "measured_ms": measured_ms, "predicted_ms": predicted_ms, "relative_error": relative_error})
                print(f"[experiment_b] control checkpoint after {len(completed_pairs)} pairs: "
                      f"max abs model deviation {max(abs(item['relative_error']) for item in control_records[-len(control_shapes):]):.1%}", flush=True)
            if len(completed_pairs) % progress_every == 0 or len(completed_pairs) == len(pairs):
                print(f"[experiment_b] completed {len(completed_pairs)}/{len(pairs)} pairs; elapsed {time.perf_counter() - collection_start:.1f}s", flush=True)
    write_json(output_dir / "experiment_b_metadata.json", {**system_metadata(), "config": config, "tau_used": tau, "tau_override": tau_override is not None, "session_id": session_id,
                                                            "pair_count": len(pairs), "near_pair_count": sum(pair["boundary_bin"] in {"near", "near_low", "near_high"} for pair in pairs),
                                                            "unique_computational_pair_count": len(computational_keys),
                                                            "domain_policy": domain_policy, "out_of_calibration_pair_count": out_of_domain_pairs,
                                                            "out_of_calibration_invocation_count": out_of_domain_invocations,
                                                            "control_shapes": control_shapes, "control_records": control_records,
                                                            "hardware_state_shift": any(abs(item["relative_error"]) > .10 for item in control_records)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--fit", required=True); parser.add_argument("--output", required=True); parser.add_argument("--tau-override", type=float); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); collect(load_config(args.config), args.fit, args.output, args.tau_override, args.overwrite)


if __name__ == "__main__": main()