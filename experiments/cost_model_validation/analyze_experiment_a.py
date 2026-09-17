"""Analyze Experiment A with observation- and configuration-level uncertainty."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .common import bootstrap_ci, write_json, file_sha256
from .models import fit_all, load_latency_models, control_predictions
from .reproducibility import statistics, design, fit_models, bootstrap_models, gate, canonical_hash


def _fit_summary(area, times):
    return {
        name: {
            "coefficients": fit.coefficients,
            "r2": fit.r2,
            "adjusted_r2": fit.adjusted_r2,
            "rmse_s": fit.rmse,
            "mae_s": fit.mae,
            "aic": fit.aic,
            "bic": fit.bic,
            "breakpoint": fit.breakpoint,
        }
        for name, fit in fit_all(area, times).items()
    }


def _write_rows(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_provenance(csv_path: str, metadata_path: str | None) -> Dict[str, Any]:
    path = Path(metadata_path) if metadata_path else Path(csv_path).with_name("metadata.json")
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    model_cfg = metadata.get("config", {}).get("model", {})
    provenance = {
        "source": str(path) if path.exists() else None,
        "session_id": metadata.get("session_id"),
        "seed": metadata.get("config", {}).get("seed"),
        "backend": model_cfg.get("backend"),
        "backend_version": metadata.get("backend_version"),
        "versions": metadata.get("versions", {}),
        "device": model_cfg.get("device"),
        "weights_path": model_cfg.get("weights"),
        "weights_sha256": metadata.get("model_weights_sha256"),
        "model_stride": metadata.get("model_stride"),
        "timing_mode": metadata.get("timing_mode"),
        "preprocessing": metadata.get("preprocessing"),
        "python": metadata.get("python"),
        "os": metadata.get("os"),
        "architecture": metadata.get("architecture"),
        "git_commit": metadata.get("git_commit"),
        "git_dirty": metadata.get("git_dirty"),
    }
    provenance.update(metadata.get("provenance", {}))
    required = ("backend", "backend_version", "device", "preprocessing", "git_commit", "seed")
    missing = [key for key in required if provenance.get(key) is None or provenance.get(key) == ""]
    if provenance.get("backend") != "fake" and not provenance.get("weights_sha256"):
        missing.append("weights_sha256")
    provenance["missing_fields"] = missing
    provenance["complete"] = not missing
    provenance["warnings"] = ["Historical collection metadata is missing: " + ", ".join(missing)] if missing else []
    return provenance


def analyze(path: str, output_dir: str | None = None, bootstrap_count: int = 2000, metadata_path: str | None = None):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Experiment A CSV is empty")

    area = np.asarray([float(row["effective_area"]) for row in rows])
    times = np.asarray([
        float(row["inference_ms" if row["timing_mode"] == "inference_only" else "detector_call_ms"]) / 1000.0
        for row in rows
    ])
    shape_groups = defaultdict(list)
    area_groups = defaultdict(list)
    for row, value in zip(rows, times):
        shape_groups[(int(row["tensor_h"]), int(row["tensor_w"]))].append(float(value))
        area_groups[int(row["effective_area"])].append(float(value))

    shape_rows = [
        {
            "tensor_h": height,
            "tensor_w": width,
            "effective_area": height * width,
            "count": len(values),
            "mean_s": float(np.mean(values)),
            "median_s": float(np.median(values)),
            "std_s": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        for (height, width), values in sorted(shape_groups.items())
    ]
    area_rows = [
        {
            "effective_area": effective,
            "count": len(values),
            "mean_s": float(np.mean(values)),
            "median_s": float(np.median(values)),
            "std_s": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        for effective, values in sorted(area_groups.items())
    ]

    observation_models = _fit_summary(area, times)
    shape_area = np.asarray([row["effective_area"] for row in shape_rows], dtype=float)
    shape_mean = np.asarray([row["mean_s"] for row in shape_rows], dtype=float)
    area_mean = np.asarray([row["mean_s"] for row in area_rows], dtype=float)
    shape_models = _fit_summary(shape_area, shape_mean) if len(shape_rows) >= 2 else {}
    area_models = _fit_summary(np.asarray([row["effective_area"] for row in area_rows]), area_mean) if len(area_rows) >= 2 else {}

    metadata_file = Path(metadata_path) if metadata_path else Path(path).with_name("metadata.json")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
    config = metadata.get("config", {})
    options = config.get("experiment_a", {})
    statistic = options.get("primary_shape_statistic", "trimmed_mean")
    level = options.get("primary_fit_level", "shape_level")
    trim = float(options.get("trim_fraction_each_tail", .1))
    support = int(options.get("breakpoint_min_support", 2))
    for shape in shape_rows:
        key = (shape["tensor_h"], shape["tensor_w"])
        shape.update(statistics(shape_groups[key], trim))
        positions = [int(r.get("global_position") or i) for i,r in enumerate(rows) if (int(r["tensor_h"]),int(r["tensor_w"])) == key]
        shape.update(first_measurement_position=min(positions), last_measurement_position=max(positions))
    alternatives = {stat: {lev: fit_models(*design(shape_groups, stat, lev, trim), support)
                          for lev in ("shape_level", "area_level")} for stat in {"mean", statistic, "trimmed_mean"}}
    primary = alternatives[statistic][level]
    if not primary["linear"]["valid"]:
        raise ValueError("Primary linear fit requires positive intercept and slope")
    k_t, c_t = primary["linear"]["coefficients"]["b0"], primary["linear"]["coefficients"]["b1"]
    uncertainty = bootstrap_models(shape_groups, bootstrap_count, int(options.get("bootstrap_seed", 0)), statistic, level, trim, support)
    bootstrap_rows = [dict(bootstrap_id=i, K_t_s=r["linear"]["coefficients"].get("b0"),
                           c_t_s_per_pixel=r["linear"]["coefficients"].get("b1"),
                           tau_pixels=r["linear"]["coefficients"]["b0"]/r["linear"]["coefficients"]["b1"] if r["linear"]["valid"] else None,
                           valid=r["linear"]["valid"]) for i,r in enumerate(uncertainty["replicates"])]
    valid_tau = [r["tau_pixels"] for r in bootstrap_rows if r["valid"]]
    envelope = {
        "min_effective_area": int(np.min(area)),
        "max_effective_area": int(np.max(area)),
        "min_tensor_h": min(int(row["tensor_h"]) for row in rows),
        "max_tensor_h": max(int(row["tensor_h"]) for row in rows),
        "min_tensor_w": min(int(row["tensor_w"]) for row in rows),
        "max_tensor_w": max(int(row["tensor_w"]) for row in rows),
        "min_aspect_ratio": min(float(row["tensor_w"]) / float(row["tensor_h"]) for row in rows),
        "max_aspect_ratio": max(float(row["tensor_w"]) / float(row["tensor_h"]) for row in rows),
    }
    summary = {
        "n_observations": len(rows),
        "unique_effective_shapes": len(shape_rows),
        "linear_fit": {
            "K_t_s": k_t,
            "c_t_s_per_pixel": c_t,
            "tau_pixels": k_t / c_t,
            "tau_bootstrap_ci": bootstrap_ci(valid_tau),
        },
        "fit_observation_level": observation_models,
        "fit_shape_level": shape_models,
        "fit_area_level": area_models,
        "calibration_envelope": envelope,
        "bootstrap_invalid_ratio_fraction": 1.0 - len(valid_tau) / max(len(bootstrap_rows), 1),
    }
    summary["provenance"] = _load_provenance(path, metadata_path)
    warnings = list(summary["provenance"].get("warnings", []))
    if metadata.get("schedule_hash"):
        schedule_file = Path(path).with_name("experiment_a_schedule.json")
        saved = json.loads(schedule_file.read_text(encoding="utf-8"))
        if canonical_hash(saved["items"]) != metadata["schedule_hash"] or saved["hash"] != metadata["schedule_hash"]:
            raise ValueError("Calibration schedule hash mismatch")
        positions = set()
        for row in rows:
            position = int(row["global_position"])
            if position in positions or not 0 <= position < len(saved["items"]):
                raise ValueError("Duplicate/invalid schedule position")
            positions.add(position)
            item = saved["items"][position]
            if any(int(row[k]) != item[k] for k in ("tensor_h","tensor_w","block_index","position_in_block")) or row["shape_id"] != item["shape_id"]:
                raise ValueError("Raw shape disagrees with schedule")
        designed_shapes = {(item["tensor_h"],item["tensor_w"]) for item in saved["items"]}
        gate(set(shape_groups) == designed_shapes and len(positions) == len(saved["items"]), "designed_shape_set", config, warnings)
        expected = int(options.get("repetitions_per_effective_shape", 40))
        gate(all(len(v) == expected for v in shape_groups.values()), "repetition_count", config, warnings)
        blocks = defaultdict(list)
        for row in rows:
            blocks[row["block_index"]].append(row["shape_id"])
        gate(len(blocks) == expected and all(len(v) == len(shape_groups) and len(set(v)) == len(v) for v in blocks.values()), "block_shapes", config, warnings)
    for row in rows:
        stride = int(row.get("model_stride") or metadata.get("model_stride", 1))
        gate(int(row["tensor_h"]) % stride == 0 and int(row["tensor_w"]) % stride == 0, "stride_alignment", config, warnings)
        gate(int(row["effective_area"]) == int(row["tensor_h"])*int(row["tensor_w"]), "effective_area", config, warnings)
    gate(len({r["timing_mode"] for r in rows}) == 1, "timing_mode", config, warnings)
    for name, model in primary.items():
        gate(model["valid"], "invalid_"+name, config, warnings)
        gate(uncertainty["summaries"][name]["invalid_fraction"] <= float(options.get("max_invalid_bootstrap_fraction", .2)), "invalid_bootstrap", config, warnings)
    summary.update(schema_version=3, primary_fit_selector=dict(statistic=statistic, level=level, trim_fraction_each_tail=trim),
                   alternative_fits=alternatives, bootstrap=uncertainty, schedule_hash=metadata.get("schedule_hash"),
                   raw_observations_reference=str(Path(path).resolve()), raw_observations_sha256=file_sha256(path), shape_statistics=shape_rows,
                   unique_effective_areas=len(area_rows), area_multiplicity={str(a):sum(r["effective_area"] == a for r in shape_rows) for a in sorted(area_groups)},
                   quality_warnings=warnings, config=config)
    shape_models = alternatives[statistic]["shape_level"]
    area_models = alternatives[statistic]["area_level"]
    summary["fit_shape_level"] = shape_models
    summary["fit_area_level"] = area_models
    threshold = float(options.get("fit_level_relative_difference_threshold", .2))
    for name in primary:
        left, right = shape_models[name], area_models[name]
        if left["valid"] and right["valid"]:
            values = [(v,right["coefficients"][k]) for k,v in left["coefficients"].items()]
            if name == "piecewise":
                values.append((left["breakpoint_area"],right["breakpoint_area"]))
            if any(abs(a-b)/max(abs(a),abs(b),1e-15)>threshold for a,b in values):
                warnings.append(f"{name}: coefficients/breakpoint depend on fit level (threshold {threshold})")
    summary["latency_models"] = primary
    summary["linear_fit"] = dict(K_t_s=k_t, c_t_s_per_pixel=c_t, tau_pixels=k_t/c_t, tau_bootstrap_ci=bootstrap_ci(valid_tau))
    models = load_latency_models(summary)
    controls_path = Path(path).with_name("experiment_a_controls.json")
    controls = json.loads(controls_path.read_text()) if controls_path.exists() else []
    for record in controls:
        record["predictions"] = control_predictions(models, record["effective_area"], record["measured_ms"])
    diagnostics = {}
    for key in sorted({(r["tensor_h"],r["tensor_w"]) for r in controls}):
        records = sorted([r for r in controls if (r["tensor_h"],r["tensor_w"]) == key], key=lambda r:r["elapsed_s"])
        values = np.array([r["measured_ms"] for r in records])
        thirds = np.array_split(values, 3)
        early, middle, late = [float(v.mean()) if len(v) else None for v in thirds]
        change = (late-early)/early if late is not None and early else None
        diagnostics[str(key)] = dict(early_mean_ms=early, middle_mean_ms=middle, late_mean_ms=late,
                                     median_ms=float(np.median(values)), min_ms=float(values.min()), max_ms=float(values.max()),
                                     relative_change=change, trend_ms_per_s=float(np.polyfit([r["elapsed_s"] for r in records], values, 1)[0]) if len(records)>1 else None)
    shifted = any(d["relative_change"] is not None and abs(d["relative_change"]) > float(options.get("control_drift_threshold", .1)) for d in diagnostics.values())
    gate(not shifted, "control_drift", config, warnings)
    summary.update(control_records=controls, control_diagnostics=diagnostics, hardware_state_shift=shifted)
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_rows(output / "experiment_a_shape_summary.csv", shape_rows)
        _write_rows(output / "experiment_a_area_summary.csv", area_rows)
        _write_rows(output / "bootstrap_fits.csv", bootstrap_rows)
        write_json(output / "linear_fit.json", summary | summary["linear_fit"])
        write_json(output / "bootstrap_models.json", uncertainty)
        write_json(output / "fit_area_level.json", alternatives[statistic]["area_level"])
        write_json(output / "fit_observation_level.json", observation_models)
        write_json(output / "fit_shape_level.json", shape_models)
        write_json(output / "experiment_a_summary.json", summary)
        write_json(output / "bootstrap_summary.json", {
            "count": len(bootstrap_rows),
            "valid_count": len(valid_tau),
            "tau_ci": bootstrap_ci(valid_tau),
            "K_t_ci": bootstrap_ci([row["K_t_s"] for row in bootstrap_rows]),
            "c_t_ci": bootstrap_ci([row["c_t_s_per_pixel"] for row in bootstrap_rows]),
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-count", type=int, default=2000)
    parser.add_argument("--metadata", help="Path to Experiment A metadata.json; defaults to metadata.json next to --input")
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.bootstrap_count, args.metadata), indent=2))


if __name__ == "__main__":
    main()
