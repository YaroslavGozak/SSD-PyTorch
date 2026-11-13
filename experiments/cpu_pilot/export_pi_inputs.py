"""Export only the frozen Stage 3 inputs needed by a Raspberry Pi run."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from experiments.cpu_pilot.portable_paths import CHECKPOINT_SHA256, CHECKPOINT_RELATIVE_PATHS
from experiments.pi_preflight import (
    EXPECTED_FINAL_SHA256,
    EXPECTED_PILOT_SHA256,
    EXPECTED_THRESHOLDS,
    IMPLEMENTATION_PATHS,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "final_experiments" / "pilot_20261110"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_files():
    files = [
        "split_metadata.json",
        "pilot_report.md",
        "pilot_manifest.txt",
        "final_eval_manifest_candidate.txt",
        "configs/yolo26n.yaml",
        "configs/roi_ssd_mobilenet.yaml",
    ]
    for model_name in EXPECTED_THRESHOLDS:
        directory = Path("calibration") / model_name
        files.extend(
            str(directory / filename)
            for filename in (
                "selected_operational_config.json",
                "selected_operational_config.sha256",
                "calibration_sweep.csv",
                "calibration_gate_matrix.csv",
                "calibration_report.md",
                "full_frame_baseline.json",
            )
        )
    return files


def _validate_source(source):
    missing = [source / relative for relative in _required_files() if not (source / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen Stage 3 input is missing: {missing[0]}")

    if _sha256(source / "pilot_manifest.txt") != EXPECTED_PILOT_SHA256:
        raise ValueError("Frozen pilot manifest SHA-256 mismatch")
    if _sha256(source / "final_eval_manifest_candidate.txt") != EXPECTED_FINAL_SHA256:
        raise ValueError("Frozen final manifest SHA-256 mismatch")

    split = json.loads((source / "split_metadata.json").read_text(encoding="utf-8"))
    if split.get("status") != "PREPARED_NOT_BENCHMARKED":
        raise ValueError("Stage 3 split metadata status is not PREPARED_NOT_BENCHMARKED")
    provenance = split.get("provenance", {})
    if provenance.get("pilot_manifest_sha256") != EXPECTED_PILOT_SHA256:
        raise ValueError("Split metadata pilot manifest provenance mismatch")
    if provenance.get("final_eval_manifest_candidate_sha256") != EXPECTED_FINAL_SHA256:
        raise ValueError("Split metadata final manifest provenance mismatch")

    implementations = {
        relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_PATHS
    }
    for model_name, expected in EXPECTED_THRESHOLDS.items():
        selection_dir = source / "calibration" / model_name
        selection_path = selection_dir / "selected_operational_config.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (selection_dir / "selected_operational_config.sha256").read_text(
            encoding="utf-8"
        ).strip() != _sha256(selection_path):
            raise ValueError(f"Frozen {model_name} selection receipt mismatch")
        for key, value in {
            "status": "CALIBRATION_PASS",
            "model": model_name,
            "checkpoint_sha256": CHECKPOINT_SHA256[model_name],
            "pilot_manifest_sha256": EXPECTED_PILOT_SHA256,
            "evaluation_conf": expected["evaluation_conf"],
            "operational_conf": expected["operational_conf"],
            "padding": expected["padding"],
        }.items():
            if selection.get(key) != value:
                raise ValueError(f"Frozen {model_name} selection has unexpected {key}")
        config_path = source / "configs" / f"{model_name}.yaml"
        if selection.get("config_sha256") != _sha256(config_path):
            raise ValueError(f"Frozen {model_name} config hash mismatch")
        if selection.get("implementation_sha256") != implementations:
            raise ValueError(f"Frozen {model_name} implementation hashes do not match this source")
        for key, filename in (
            ("calibration_sweep_sha256", "calibration_sweep.csv"),
            ("calibration_gate_matrix_sha256", "calibration_gate_matrix.csv"),
            ("full_frame_baseline_sha256", "full_frame_baseline.json"),
        ):
            if selection.get(key) != _sha256(selection_dir / filename):
                raise ValueError(f"Frozen {model_name} calibration artifact changed: {filename}")

    expected_paths = {name: path.as_posix() for name, path in CHECKPOINT_RELATIVE_PATHS.items()}
    selected_checkpoints = provenance.get("selected_checkpoints", {})
    for model_name, expected_path in expected_paths.items():
        checkpoint = selected_checkpoints.get(model_name)
        if checkpoint is None:
            raise ValueError(f"Split metadata is missing the selected checkpoint for {model_name}")
        if str(checkpoint.get("path", "")).replace("\\", "/") != expected_path:
            raise ValueError(f"Split metadata checkpoint identity changed for {model_name}")
        if checkpoint.get("sha256") != CHECKPOINT_SHA256[model_name]:
            raise ValueError(f"Split metadata checkpoint hash changed for {model_name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    _validate_source(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty export directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for relative in _required_files():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    print(f"Frozen Stage 3 inputs exported to {output}")
    print(f"Pilot manifest SHA-256: {EXPECTED_PILOT_SHA256}")
    print(f"Final manifest SHA-256: {EXPECTED_FINAL_SHA256}")
    print("Desktop cost profiles and timing outputs were not exported.")
    print("PI_FROZEN_INPUTS_EXPORTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
