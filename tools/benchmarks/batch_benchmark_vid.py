"""
Batch runner for Lane B video benchmark.

Runs multiple tracker experiments sequentially using one base benchmark config and a
separate sweep definition YAML. Each run appends one row to CSV using the same output
path logic as benchmark_framework_vid.py (read from benchmark config output section).

Usage:
    python -m tools.benchmarks.batch_benchmark_vid \
        --benchmark-config config/benchmark-vid-yolo.yaml \
        --sweep-config config/benchmark-vid-sweep-example.yaml
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tools.benchmarks.benchmark_framework_vid import VideoSequenceBenchmark
from tools.helpers.config_reader import load_config


SUPPORTED_TRACKERS = {"static_padding", "relative_padding", "kalman", "oracle_gt", "sort"}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_sweep_config(sweep_cfg: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    if not isinstance(sweep_cfg, dict):
        raise TypeError("Sweep config must be a YAML mapping.")

    sweep_name = str(sweep_cfg.get("sweep_name", "tracker_sweep"))
    experiments = sweep_cfg.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Sweep config must contain a non-empty 'experiments' list.")

    validated: List[Dict[str, Any]] = []
    for i, exp in enumerate(experiments, start=1):
        if not isinstance(exp, dict):
            raise TypeError(f"Experiment #{i} must be a mapping.")

        name = exp.get("name")
        tracker_type = exp.get("tracker_type")
        tracker_params = exp.get("tracker_params", {})
        benchmark_overrides = exp.get("benchmark_overrides", {})

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Experiment #{i} has invalid 'name'.")
        if tracker_type not in SUPPORTED_TRACKERS:
            raise ValueError(
                f"Experiment '{name}' has unsupported tracker_type '{tracker_type}'. "
                f"Expected one of {sorted(SUPPORTED_TRACKERS)}."
            )
        if not isinstance(tracker_params, dict):
            raise TypeError(f"Experiment '{name}' has non-dict 'tracker_params'.")
        if not isinstance(benchmark_overrides, dict):
            raise TypeError(f"Experiment '{name}' has non-dict 'benchmark_overrides'.")

        validated.append({
            "name": name,
            "tracker_type": tracker_type,
            "tracker_params": tracker_params,
            "benchmark_overrides": benchmark_overrides,
        })

    return sweep_name, validated


def _build_experiment_cfg(base_cfg: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg = _deep_merge(cfg, exp["benchmark_overrides"])

    benchmark_vid_params = cfg.setdefault("benchmark_vid_params", {})
    tracker_cfg = benchmark_vid_params.setdefault("tracker", {})

    tracker_type = str(exp["tracker_type"])
    tracker_cfg["type"] = tracker_type
    tracker_cfg.setdefault(tracker_type, {})
    tracker_cfg[tracker_type] = _deep_merge(tracker_cfg[tracker_type], exp["tracker_params"])

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Lane B video benchmark runner")
    parser.add_argument(
        "--benchmark-config",
        required=True,
        help="Base benchmark config (e.g. config/benchmark-vid-yolo.yaml or benchmark-vid-roissd.yaml)",
    )
    parser.add_argument(
        "--sweep-config",
        required=True,
        help="Sweep YAML with experiments list",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when an experiment fails",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.benchmark_config)
    sweep_cfg = load_config(args.sweep_config)
    sweep_name, experiments = _validate_sweep_config(sweep_cfg)

    output_cfg = base_cfg.get("benchmark_vid_params", {}).get("output", {})
    results_dir = output_cfg.get("results_dir", "")
    results_filename = output_cfg.get("results_filename", "")

    print("=" * 80)
    print(f"Batch benchmark sweep: {sweep_name}")
    print(f"Experiments: {len(experiments)}")
    print(f"Output CSV (from benchmark config): {Path(results_dir) / results_filename}")
    print("=" * 80)

    failures: List[Tuple[str, str]] = []
    for idx, exp in enumerate(experiments, start=1):
        exp_name = exp["name"]
        print("\n" + "-" * 80)
        print(f"[{idx}/{len(experiments)}] Running experiment: {exp_name}")
        print(f"  tracker_type: {exp['tracker_type']}")
        print(f"  tracker_params: {exp['tracker_params']}")

        try:
            exp_cfg = _build_experiment_cfg(base_cfg, exp)
            bench = VideoSequenceBenchmark.from_config_dict(
                exp_cfg,
                extra_run_metadata={
                    "sweep_name": sweep_name,
                    "experiment_name": exp_name,
                    "experiment_index": idx,
                },
            )
            bench.run()
        except Exception as exc:
            failures.append((exp_name, str(exc)))
            print(f"  FAILED: {exc}")
            if args.fail_fast:
                break

    succeeded = len(experiments) - len(failures)
    print("\n" + "=" * 80)
    print("Batch sweep finished")
    print(f"Succeeded: {succeeded}")
    print(f"Failed   : {len(failures)}")

    if failures:
        print("Failures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
