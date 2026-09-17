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
    args = parser.parse_args()
    output = Path(args.output)
    collection_args = ["--config", args.config, "--output", str(output)]
    if args.overwrite:
        collection_args.append("--overwrite")

    steps = [
        ("collect_experiment_a", collection_args),
        ("analyze_experiment_a", ["--input", str(output / "experiment_a_raw.csv"), "--output", str(output)]),
        ("collect_experiment_b", collection_args + ["--fit", str(output / "linear_fit.json")]),
        ("analyze_experiment_b", ["--input", str(output / "experiment_b_raw.csv"), "--output", str(output)]),
    ]
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
    print(f"\nAll four steps completed. Results: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
