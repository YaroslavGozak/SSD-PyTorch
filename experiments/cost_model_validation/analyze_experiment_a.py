"""Analyze Experiment A with observation- and configuration-level uncertainty."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .common import bootstrap_ci, write_json
from .models import fit_all


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

    linear = fit_all(area, times)["linear"]
    k_t, c_t = linear.coefficients["b0"], linear.coefficients["b1"]
    if k_t <= 0 or c_t <= 0:
        raise ValueError(f"Invalid positive cost fit: K_t={k_t}, c_t={c_t}")

    run_groups = defaultdict(list)
    for row, value in zip(rows, times):
        run_groups[row.get("run_id", "0")].append((float(row["effective_area"]), float(value)))
    run_ids = list(run_groups)
    rng = np.random.default_rng(0)
    bootstrap_rows = []
    for bootstrap_id in range(int(bootstrap_count)):
        sampled = rng.choice(run_ids, size=len(run_ids), replace=True)
        sampled_area = np.concatenate([
            np.asarray([item[0] for item in run_groups[run_id]]) for run_id in sampled
        ])
        sampled_time = np.concatenate([
            np.asarray([item[1] for item in run_groups[run_id]]) for run_id in sampled
        ])
        fit = fit_all(sampled_area, sampled_time)["linear"]
        k_boot, c_boot = fit.coefficients["b0"], fit.coefficients["b1"]
        bootstrap_rows.append({
            "bootstrap_id": bootstrap_id,
            "K_t_s": k_boot,
            "c_t_s_per_pixel": c_boot,
            "tau_pixels": k_boot / c_boot if k_boot > 0 and c_boot > 0 else None,
            "valid": bool(k_boot > 0 and c_boot > 0),
        })

    valid_tau = [row["tau_pixels"] for row in bootstrap_rows if row["valid"]]
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
    observation_linear = observation_models["linear"]
    observation_quadratic = observation_models["quadratic"]
    observation_piecewise = observation_models["piecewise"]

    def _fit_metrics(name: str) -> Dict[str, Any]:
        # Shape-level fit is the primary basis for model comparison; observation-level
        # is kept for backward compatibility and noise diagnostics.
        return {"fit_level": "shape_level_primary", "shape_level": shape_models.get(name),
                "observation_level": observation_models.get(name)}

    summary["schema_version"] = 2
    summary["provenance"] = _load_provenance(path, metadata_path)
    summary["latency_models"] = {
        "linear": {"formula": "b0 + b1*A", "coefficients": {"b0": k_t, "b1": c_t},
                    "tau_pixels": k_t / c_t, "fit_metrics": _fit_metrics("linear")},
        "quadratic": {"formula": "b0 + b1*A + b2*A^2", "coefficients": {
            "b0": observation_quadratic["coefficients"]["b0"], "b1": observation_quadratic["coefficients"]["b1"], "b2": observation_quadratic["coefficients"]["b2"]}, "fit_metrics": _fit_metrics("quadratic")},
        "piecewise": {"formula": "b0 + b1*A + b2*max(0,A-B)", "coefficients": {
            "b0": observation_piecewise["coefficients"]["b0"], "b1": observation_piecewise["coefficients"]["b1"], "b2": observation_piecewise["coefficients"]["b2"]},
            "breakpoint_area": observation_piecewise["breakpoint"], "fit_metrics": _fit_metrics("piecewise")},
    }
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_rows(output / "experiment_a_shape_summary.csv", shape_rows)
        _write_rows(output / "experiment_a_area_summary.csv", area_rows)
        _write_rows(output / "bootstrap_fits.csv", bootstrap_rows)
        write_json(output / "linear_fit.json", summary["linear_fit"] | {"schema_version": 2, "calibration_envelope": envelope, "latency_models": summary["latency_models"], "provenance": summary["provenance"]})
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
