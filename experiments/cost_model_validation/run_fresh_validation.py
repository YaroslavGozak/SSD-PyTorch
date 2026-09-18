"""Run calibration and separate predeclared representative/challenge validations."""
import argparse
from pathlib import Path
import subprocess
import sys

import yaml

from .common import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="experiments/cost_model_validation/config.voc-yolo26n.yaml")
    parser.add_argument("--representative-config",default="experiments/cost_model_validation/config.voc-yolo26n.representative.yaml")
    parser.add_argument("--output",required=True)
    parser.add_argument("--calibration",help="Already completed new full-grid calibration; skips A")
    parser.add_argument("--smoke-pairs",type=int,help="Explicit small validation size per design; scientific policies unchanged")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if args.smoke_pairs is not None and (args.smoke_pairs<10 or args.smoke_pairs%5):
        parser.error("--smoke-pairs must be a multiple of five and at least ten")
    configs = {}
    for design,source in (("representative",args.representative_config),("challenge",args.config)):
        config = load_config(source)
        if config["experiment_b"]["evaluation_design"] != design:
            raise ValueError(f"Wrong evaluation_design in {source}")
        if args.smoke_pairs:
            config["experiment_b"]["pair_count"] = args.smoke_pairs
            config["experiment_b"]["run_purpose"] = "fresh_hardware_smoke_not_policy_selection"
            config["experiment_b"]["control_every_pairs"] = max(1,args.smoke_pairs//4)
            config["experiment_b"]["progress_every_repetitions"] = 100
        path = output / (design+"_config.yaml")
        serialized = yaml.safe_dump(config,sort_keys=False)
        if path.exists() and path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"Refusing to change an existing frozen run config: {path}")
        path.write_text(serialized,encoding="utf-8")
        configs[design] = path
    calibration = Path(args.calibration) if args.calibration else output / "calibration" / "linear_fit.json"
    steps = []
    if not args.calibration:
        steps.extend([
            ("collect_experiment_a",["--config",args.config,"--output",str(calibration.parent)]),
            ("analyze_experiment_a",["--input",str(calibration.parent/"experiment_a_raw.csv"),"--output",str(calibration.parent)]),
        ])
    for design in ("representative","challenge"):
        pairs = output / (design+"_pair_specs.json")
        run = output / design
        steps.extend([
            ("pair_specs",["--config",str(configs[design]),"--calibration",str(calibration),"--output",str(pairs)]),
            ("collect_experiment_b",["--config",str(configs[design]),"--fit",str(calibration),"--pairs",str(pairs),"--output",str(run)]),
            ("analyze_experiment_b",["--input",str(run/"experiment_b_raw.csv"),"--output",str(run)]),
        ])
    for index,(module,arguments) in enumerate(steps,1):
        print(f"[{index}/{len(steps)}] {module}",flush=True)
        result = subprocess.run([sys.executable,"-m","experiments.cost_model_validation."+module,*arguments])
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
