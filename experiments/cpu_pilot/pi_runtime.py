"""Portable launcher for the frozen Stage 3 CPU workflow."""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import yaml

from experiments.cpu_pilot import run_pilot as pilot
from experiments.cpu_pilot.portable_paths import (
    CHECKPOINT_RELATIVE_PATHS,
    CHECKPOINT_SHA256,
    EXTERNAL_CHECKPOINT_RELATIVE_PATHS,
    cpu_model_fallback,
    dataset_root_path,
    dataset_yaml_path,
    env_path,
    portable_dataset_config,
    stage3_dir,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_split_paths(path):
    split = json.loads(path.read_text(encoding="utf-8"))
    selected = split["provenance"]["selected_checkpoints"]
    for model_name, relative_path in CHECKPOINT_RELATIVE_PATHS.items():
        checkpoint = selected[model_name]
        normalized = str(checkpoint["path"]).replace("\\", "/")
        if normalized != relative_path.as_posix():
            raise ValueError(f"Frozen checkpoint identity changed for {model_name}")
        checkpoint["path"] = relative_path.as_posix()
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")


def _configure():
    dataset_root = dataset_root_path()
    dataset_yaml = env_path("SSD_DATASET_YAML")
    models_root = env_path("SSD_MODELS_ROOT")
    output_dir = stage3_dir(ROOT / "final_experiments" / "pilot_pi5")
    if dataset_root is None or not dataset_root.is_dir():
        raise FileNotFoundError("Set SSD_DATASET_ROOT to the copied ImageNet-VID dataset root")
    if dataset_yaml is None or not dataset_yaml.is_file():
        raise FileNotFoundError("Set SSD_DATASET_YAML to the local imagenet_vid.yaml")
    if models_root is None or not models_root.is_dir():
        raise FileNotFoundError("Set SSD_MODELS_ROOT to the copied selected-checkpoint directory")
    try:
        output_dir.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("SSD_STAGE3_DIR must be inside the repository for this frozen runner") from exc
    split_metadata = output_dir / "split_metadata.json"
    if not split_metadata.is_file():
        raise FileNotFoundError(f"Frozen Stage 3 split metadata is missing: {split_metadata}")
    split_metadata_source_sha256 = _sha256(split_metadata)
    if (output_dir / "metadata.json").is_file():
        previous = json.loads(
            (output_dir / "metadata.json").read_text(encoding="utf-8")
        )
        split_metadata_source_sha256 = previous.get("portable_runtime", {}).get(
            "split_metadata_source_sha256", split_metadata_source_sha256
        )
    _normalize_split_paths(split_metadata)

    canonical_checkpoints = {}
    for model_name, relative_path in CHECKPOINT_RELATIVE_PATHS.items():
        canonical = ROOT / relative_path
        external = models_root / EXTERNAL_CHECKPOINT_RELATIVE_PATHS[model_name]
        if not external.is_file():
            raise FileNotFoundError(f"Selected {model_name} checkpoint is missing: {external}")
        if _sha256(external) != CHECKPOINT_SHA256[model_name]:
            raise ValueError(f"Selected {model_name} checkpoint SHA-256 mismatch: {external}")
        if not canonical.is_file() or canonical.resolve() != external.resolve():
            raise FileNotFoundError(
                f"Create the canonical checkpoint symlink {canonical} -> {external}"
            )
        canonical_checkpoints[model_name] = (canonical, CHECKPOINT_SHA256[model_name])

    pilot.MODEL_CHECKPOINTS = canonical_checkpoints
    pilot.PILOT_DIR = output_dir
    pilot.SPLIT_METADATA = output_dir / "split_metadata.json"
    pilot._dataset_root = lambda _config: dataset_root
    pilot._cpu_model = cpu_model_fallback
    original_loader = pilot.load_model_and_dataset

    def portable_loader(device, args):
        source_path = Path(args.config_path)
        source_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        selected_yaml = dataset_yaml_path(
            source_config["dataset_params"]["yolo_dataset_yaml"], dataset_yaml
        )
        if not selected_yaml.is_file():
            raise FileNotFoundError(f"Dataset YAML is missing: {selected_yaml}")
        data_config = yaml.safe_load(selected_yaml.read_text(encoding="utf-8"))
        data_config = portable_dataset_config(data_config, dataset_root)
        model_name = (
            "yolo26n"
            if source_config["train_params"]["model"] == "yolo"
            else "roi_ssd_mobilenet"
        )
        checkpoint = canonical_checkpoints[model_name][0]
        if model_name == "yolo26n":
            source_config["train_params"]["yolo_weights"] = str(checkpoint)
        else:
            source_config["train_params"]["task_name"] = str(checkpoint.parent)
            source_config["train_params"]["ckpt_name"] = checkpoint.name

        with tempfile.TemporaryDirectory(prefix="stage3-portable-") as temp_dir:
            temp_dir = Path(temp_dir)
            runtime_yaml = temp_dir / selected_yaml.name
            runtime_yaml.write_text(
                yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8"
            )
            source_config["dataset_params"]["yolo_dataset_yaml"] = str(runtime_yaml)
            runtime_config = temp_dir / source_path.name
            runtime_config.write_text(
                yaml.safe_dump(source_config, sort_keys=False), encoding="utf-8"
            )
            model, dataset, loader, config = original_loader(
                device, SimpleNamespace(config_path=str(runtime_config))
            )

        config["dataset_params"]["yolo_dataset_yaml"] = str(selected_yaml)
        config["train_params"] = yaml.safe_load(source_path.read_text(encoding="utf-8"))[
            "train_params"
        ]
        return model, dataset, loader, config

    pilot.load_model_and_dataset = portable_loader
    original_write_metadata = pilot._write_metadata

    def write_metadata(split, stage, arguments):
        original_write_metadata(split, stage, arguments)
        metadata_path = pilot.PILOT_DIR / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["portable_runtime"] = {
            "launcher_sha256": _sha256(Path(__file__)),
            "dataset_root": str(dataset_root),
            "dataset_yaml": str(dataset_yaml),
            "models_root": str(models_root),
            "checkpoint_sha256_by_model": CHECKPOINT_SHA256,
            "split_metadata_source_sha256": split_metadata_source_sha256,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    pilot._write_metadata = write_metadata


def main():
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        pilot.main()
        return
    _configure()
    pilot.main()


if __name__ == "__main__":
    main()
