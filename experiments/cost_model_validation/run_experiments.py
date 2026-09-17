"""Collect and analyze experiments A and B sequentially."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/cost_model_validation/config.voc-yolo26n.yaml",
    )
    parser.add_argument("--output", default="outputs/cost_model_voc_yolo26n_v4")
    parser.add_argument(
        "--overwrite", action="store_true", help="Start fresh collections, replacing existing raw data."
    )
    parser.add_argument("--pairs", help="Existing frozen pair_specs.json")
    parser.add_argument("--regenerate-pairs", action="store_true", help="Explicitly regenerate validation geometry after calibration")
    args = parser.parse_args()
    output = Path(args.output)
    collection_args = ["--config", args.config, "--output", str(output)]
    if args.overwrite:
        collection_args.append("--overwrite")

    pairs = Path(args.pairs) if args.pairs else output / "pair_specs.json"
    steps = [
        ("collect_experiment_a", collection_args),
        ("analyze_experiment_a", ["--input", str(output / "experiment_a_raw.csv"), "--output", str(output)]),
        ("collect_experiment_b", collection_args + ["--fit", str(output / "linear_fit.json"), "--pairs", str(pairs)]),
        ("analyze_experiment_b", ["--input", str(output / "experiment_b_raw.csv"), "--output", str(output)]),
    ]
    if not pairs.exists() or args.regenerate_pairs:
        generation_args = ["--config", args.config, "--calibration", str(output / "linear_fit.json"), "--output", str(pairs)]
        if args.regenerate_pairs:
            generation_args.append("--regenerate-pairs")
        steps.insert(2, ("pair_specs", generation_args))
    for index, (module, arguments) in enumerate(steps, start=1):
        print(f"\n[{index}/{len(steps)}] {module}", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-m", f"experiments.cost_model_validation.{module}", *arguments],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"{module} failed; stopping (exit code {exc.returncode}).", file=sys.stderr)
            return exc.returncode
    print(f"\nAll steps completed. Results: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
