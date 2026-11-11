"""Short Raspberry Pi readiness checks; does not load models or run inference."""

import csv
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from experiments.cpu_pilot.portable_paths import (
    CHECKPOINT_RELATIVE_PATHS,
    CHECKPOINT_SHA256,
    EXTERNAL_CHECKPOINT_RELATIVE_PATHS,
    env_path,
    portable_dataset_config,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PILOT_SHA256 = "817f5b145b774dc91ee0fa8d271e02c11d71d70bb50ff13082a0ba9487aa63cb"
EXPECTED_FINAL_SHA256 = "4f36f3455ced1a344ef69329479cd09013934d71aafa8dc800d7f7e5ea6bf6d4"
EXPECTED_AUDIT_SHA256 = "6daa585af9a75d9fc390be20157e850d62c24e03f970bfc32b16e70e3cdd8041"
EXPECTED_COUNTS = {"pilot_frames": 320, "pilot_videos": 32, "final_frames": 4744, "final_videos": 476}
EXPECTED_THRESHOLDS = {
    "yolo26n": {"evaluation_conf": 0.01, "operational_conf": 0.10, "padding": 64},
    "roi_ssd_mobilenet": {"evaluation_conf": 0.01, "operational_conf": 0.50, "padding": 128},
}
IMPLEMENTATION_PATHS = (
    "experiments/cpu_pilot/run_pilot.py",
    "experiments/cpu_pilot/calibration.py",
    "tools/helpers/pipeline.py",
    "tools/trackers/static_padding_tracker.py",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _audit_rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _key(row):
    return str(row["video_id"]), int(row["frame_index"])


def _validate_manifest_paths(rows, dataset_root, audit_by_key, audit_order, failures):
    keys = []
    positions = []
    for row in rows:
        key = _key(row)
        keys.append(key)
        source = audit_by_key.get(key)
        if source is None:
            failures.append(f"Manifest frame ID is absent from audit manifest: {key}")
            continue
        if row["image"] != source["image"] or row["label"] != source["label"]:
            failures.append(f"Manifest paths changed for frame {key}")
        if json.loads(row["gt_json"]) != source["gt"]:
            failures.append(f"Manifest annotations changed for frame {key}")
        for field in ("image", "label"):
            path_text = row[field]
            posix_path = PurePosixPath(path_text)
            if (
                not path_text
                or "\\" in path_text
                or posix_path.is_absolute()
                or ".." in posix_path.parts
            ):
                failures.append(f"Manifest {field} path is not a portable dataset-relative path: {path_text}")
                continue
            resolved = dataset_root.joinpath(*posix_path.parts)
            if not resolved.is_file():
                failures.append(f"Manifest {field} is missing: {resolved}")
            elif field == "label" and source.get("label_sha256"):
                if _sha256(resolved) != source["label_sha256"]:
                    failures.append(f"Annotation checksum mismatch for frame {key}: {resolved}")
        positions.append(audit_order.get(key, -1))
    if len(keys) != len(set(keys)):
        failures.append("Manifest contains duplicate (video_id, frame_index) keys")
    if any(position < 0 for position in positions) or positions != sorted(positions):
        failures.append("Manifest frame IDs are not in the original audit-manifest order")
    return keys


def _validate_frozen_selections(stage3, failures):
    configs = {
        "yolo26n": stage3 / "configs" / "yolo26n.yaml",
        "roi_ssd_mobilenet": stage3 / "configs" / "roi_ssd_mobilenet.yaml",
    }
    pilot_manifest = stage3 / "pilot_manifest.txt"
    if pilot_manifest.is_file() and _sha256(pilot_manifest) != EXPECTED_PILOT_SHA256:
        failures.append("Frozen pilot manifest SHA-256 does not match the locked value")

    current_implementations = {}
    for relative in IMPLEMENTATION_PATHS:
        path = ROOT / relative
        if path.is_file():
            current_implementations[relative] = _sha256(path)
        else:
            failures.append(f"Required implementation file is missing: {path}")

    for model_name, expected in EXPECTED_THRESHOLDS.items():
        selection_dir = stage3 / "calibration" / model_name
        selection_path = selection_dir / "selected_operational_config.json"
        receipt_path = selection_dir / "selected_operational_config.sha256"
        if not selection_path.is_file() or not receipt_path.is_file():
            failures.append(f"Frozen Stage 3 selection or receipt is missing for {model_name}")
            continue
        selection_hash = _sha256(selection_path)
        try:
            if receipt_path.read_text(encoding="utf-8").strip() != selection_hash:
                failures.append(f"Frozen Stage 3 selection receipt mismatch for {model_name}")
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"Could not read frozen Stage 3 selection for {model_name}: {exc}")
            continue
        if selection.get("status") != "CALIBRATION_PASS":
            failures.append(f"Frozen Stage 3 selection is not CALIBRATION_PASS for {model_name}")
        for key, value in {
            "model": model_name,
            "checkpoint_sha256": CHECKPOINT_SHA256[model_name],
            "pilot_manifest_sha256": EXPECTED_PILOT_SHA256,
            "evaluation_conf": expected["evaluation_conf"],
            "operational_conf": expected["operational_conf"],
            "padding": expected["padding"],
        }.items():
            if selection.get(key) != value:
                failures.append(f"Frozen {model_name} selection has unexpected {key}: {selection.get(key)!r}")
        config_path = configs[model_name]
        if not config_path.is_file():
            failures.append(f"Frozen Stage 3 config is missing: {config_path}")
        elif selection.get("config_sha256") != _sha256(config_path):
            failures.append(f"Frozen Stage 3 config hash mismatch for {model_name}")
        elif selection.get("implementation_sha256") != current_implementations:
            failures.append(
                f"Stage 3 implementation hashes differ from frozen calibration for {model_name}; "
                "use the calibrated source revision, do not edit the selection"
            )
        for hash_key, filename in (
            ("calibration_sweep_sha256", "calibration_sweep.csv"),
            ("calibration_gate_matrix_sha256", "calibration_gate_matrix.csv"),
            ("full_frame_baseline_sha256", "full_frame_baseline.json"),
        ):
            artifact = selection_dir / filename
            if not artifact.is_file() or selection.get(hash_key) != _sha256(artifact):
                failures.append(f"Frozen {model_name} calibration artifact is missing or changed: {filename}")


def _validate_split_metadata(stage3, failures):
    split_path = stage3 / "split_metadata.json"
    if not split_path.is_file():
        failures.append(f"Frozen Stage 3 split metadata is missing: {split_path}")
        return
    try:
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"Could not read frozen Stage 3 split metadata: {exc}")
        return
    if split.get("status") != "PREPARED_NOT_BENCHMARKED":
        failures.append("Stage 3 split metadata is not the frozen PREPARED_NOT_BENCHMARKED split")
    provenance = split.get("provenance", {})
    expected_hashes = {
        "pilot_manifest_sha256": EXPECTED_PILOT_SHA256,
        "final_eval_manifest_candidate_sha256": EXPECTED_FINAL_SHA256,
        "source_manifest_sha256": EXPECTED_AUDIT_SHA256,
    }
    for key, value in expected_hashes.items():
        if provenance.get(key) != value:
            failures.append(f"Stage 3 split metadata has unexpected {key}")
    expected_checkpoint_paths = {
        model: path.as_posix() for model, path in CHECKPOINT_RELATIVE_PATHS.items()
    }
    selected_checkpoints = provenance.get("selected_checkpoints", {})
    for model_name, expected_hash in CHECKPOINT_SHA256.items():
        selected = selected_checkpoints.get(model_name, {})
        normalized_path = str(selected.get("path", "")).replace("\\", "/")
        if normalized_path != expected_checkpoint_paths[model_name]:
            failures.append(f"Stage 3 split metadata checkpoint identity changed for {model_name}")
        if selected.get("sha256") != expected_hash:
            failures.append(f"Stage 3 split metadata checkpoint hash changed for {model_name}")
    if split.get("selection", {}).get("pilot_frames") != EXPECTED_COUNTS["pilot_frames"]:
        failures.append("Stage 3 split metadata pilot frame count changed")
    if split.get("selection", {}).get("candidate_final_eval_frames") != EXPECTED_COUNTS["final_frames"]:
        failures.append("Stage 3 split metadata final frame count changed")


def _validate_checkpoints(models_root, failures):
    if models_root is None or not models_root.is_dir():
        failures.append("Set SSD_MODELS_ROOT to the copied external models directory")
        return
    for model_name, relative in EXTERNAL_CHECKPOINT_RELATIVE_PATHS.items():
        external = models_root / relative
        canonical = ROOT / CHECKPOINT_RELATIVE_PATHS[model_name]
        if not external.is_file():
            failures.append(f"Selected {model_name} checkpoint is missing: {external}")
            continue
        if _sha256(external) != CHECKPOINT_SHA256[model_name]:
            failures.append(f"Selected {model_name} checkpoint SHA-256 mismatch: {external}")
        if not canonical.is_file():
            failures.append(f"Canonical checkpoint path/symlink is missing: {canonical}")
        elif canonical.resolve() != external.resolve():
            failures.append(f"Canonical checkpoint path does not resolve to SSD_MODELS_ROOT: {canonical}")
        elif _sha256(canonical) != CHECKPOINT_SHA256[model_name]:
            failures.append(f"Canonical selected checkpoint SHA-256 mismatch: {canonical}")


def _validate_writable_output(output_root, failures):
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".pi-preflight-", dir=output_root, delete=False
        ) as stream:
            test_path = Path(stream.name)
        test_path.unlink()
    except OSError as exc:
        failures.append(f"Output directory is not writable ({output_root}): {exc}")


def main():
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        print(__doc__)
        print("Required environment: SSD_DATASET_ROOT, SSD_DATASET_YAML, SSD_MODELS_ROOT, SSD_STAGE3_DIR, SSD_OUTPUT_ROOT")
        return 0

    failures = []
    print(f"Platform: {platform.platform()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")
    if sys.platform != "linux":
        failures.append("Preflight must run on Linux")
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        failures.append("Expected a 64-bit ARM (aarch64/arm64) Raspberry Pi OS install")

    module_versions = {}
    module_names = ("cv2", "numpy", "psutil", "torch", "torchvision", "yaml", "ultralytics", "PIL", "einops", "filterpy", "skimage")
    modules = {}
    for name in module_names:
        try:
            module = importlib.import_module(name)
            modules[name] = module
            module_versions[name] = getattr(module, "__version__", "importable")
        except ImportError as exc:
            failures.append(f"Required Python module {name!r} cannot be imported: {exc}")
    try:
        importlib.import_module("tools.helpers.pipeline")
    except ImportError as exc:
        failures.append(f"Project inference dependencies cannot be imported: {exc}")
    print(f"Python package versions: {json.dumps(module_versions, sort_keys=True)}")
    torch_module = modules.get("torch")
    if torch_module is not None:
        if torch_module.cuda.is_available():
            failures.append("CUDA is available; this experiment must use CPU only")
        if torch_module.empty(1).device.type != "cpu":
            failures.append("PyTorch did not allocate tensors on the CPU")
    try:
        import psutil

        if psutil.cpu_count(logical=False) is None or psutil.cpu_count(logical=True) is None:
            failures.append("Could not detect CPU core counts")
    except ImportError:
        pass

    dataset_root = env_path("SSD_DATASET_ROOT")
    dataset_yaml = env_path("SSD_DATASET_YAML")
    models_root = env_path("SSD_MODELS_ROOT")
    stage3 = env_path("SSD_STAGE3_DIR")
    output_root = env_path("SSD_OUTPUT_ROOT")
    for name, value in (
        ("SSD_DATASET_ROOT", dataset_root),
        ("SSD_DATASET_YAML", dataset_yaml),
        ("SSD_MODELS_ROOT", models_root),
        ("SSD_STAGE3_DIR", stage3),
        ("SSD_OUTPUT_ROOT", output_root),
    ):
        if value is None:
            failures.append(f"Environment variable {name} is required")
    if dataset_root is not None and not dataset_root.is_dir():
        failures.append(f"Dataset root does not exist: {dataset_root}")
    if dataset_yaml is not None and not dataset_yaml.is_file():
        failures.append(f"Dataset YAML does not exist: {dataset_yaml}")
    if stage3 is not None and not stage3.is_dir():
        failures.append(f"Frozen Stage 3 directory does not exist: {stage3}")
    if stage3 is not None:
        try:
            stage3.relative_to(ROOT)
        except ValueError:
            failures.append("SSD_STAGE3_DIR must be inside the repository for the frozen Stage 3 runner")
    if output_root is not None:
        if output_root.exists() and any(output_root.iterdir()):
            failures.append(f"Stage 4 output root is not fresh: {output_root}")
        try:
            output_root.relative_to(ROOT)
        except ValueError:
            pass
        else:
            failures.append("SSD_OUTPUT_ROOT must be outside the repository")
        _validate_writable_output(output_root, failures)
        if stage3 is not None and output_root.resolve() == stage3.resolve():
            failures.append("Stage 4 output root must be separate from frozen Stage 3 artifacts")
        if any(output_root.rglob("cost_profiles_final.json")):
            failures.append("Stage 4 output contains a pre-existing cost profile; use a fresh Pi-only output root")

    _validate_checkpoints(models_root, failures)
    if stage3 is not None and stage3.is_dir():
        for stale_name in (
            "cost_profiles_cpu.json",
            "cost_profiles_cpu.csv",
            "cost_measurements_cpu.csv",
            "metadata.json",
            "smoke_metadata.json",
        ):
            if (stage3 / stale_name).exists():
                failures.append(
                    f"Stage 3 directory contains stale output {stale_name}; copy only frozen inputs"
                )
        _validate_frozen_selections(stage3, failures)
        _validate_split_metadata(stage3, failures)
        report_path = stage3 / "pilot_report.md"
        if not report_path.is_file():
            failures.append(f"Frozen Stage 3 report is missing: {report_path}")
        else:
            report = report_path.read_text(encoding="utf-8")
            for marker in ("YOLO_PADDING_FINAL = 64", "ROISSD_MOBILENET_PADDING_FINAL = 128"):
                if marker not in report:
                    failures.append(f"Frozen Stage 3 report does not contain `{marker}`")

    audit_path = ROOT / "final_experiments" / "01_audit" / "val_manifest.jsonl"
    if audit_path.is_file():
        audit_sha = _sha256(audit_path)
        if audit_sha != EXPECTED_AUDIT_SHA256:
            failures.append("Audit manifest SHA-256 differs from the frozen source")
        try:
            audit_rows = _audit_rows(audit_path)
            audit_by_key = {_key(row): row for row in audit_rows}
            audit_order = {_key(row): index for index, row in enumerate(audit_rows)}
            if len(audit_by_key) != len(audit_rows):
                failures.append("Audit manifest contains duplicate frame IDs")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append(f"Could not validate audit manifest: {exc}")
            audit_rows, audit_by_key, audit_order = [], {}, {}
    else:
        failures.append(f"Audit manifest is missing: {audit_path}")
        audit_rows, audit_by_key, audit_order = [], {}, {}

    if stage3 is not None and dataset_root is not None and dataset_root.is_dir():
        pilot_path = stage3 / "pilot_manifest.txt"
        final_path = stage3 / "final_eval_manifest_candidate.txt"
        for path, expected_hash in (
            (pilot_path, EXPECTED_PILOT_SHA256),
            (final_path, EXPECTED_FINAL_SHA256),
        ):
            if not path.is_file():
                failures.append(f"Frozen manifest is missing: {path}")
            elif _sha256(path) != expected_hash:
                failures.append(f"Frozen manifest SHA-256 mismatch: {path}")
        if pilot_path.is_file() and final_path.is_file() and audit_rows:
            try:
                pilot_rows = _manifest(pilot_path)
                final_rows = _manifest(final_path)
                pilot_keys = _validate_manifest_paths(
                    pilot_rows, dataset_root, audit_by_key, audit_order, failures
                )
                final_keys = _validate_manifest_paths(
                    final_rows, dataset_root, audit_by_key, audit_order, failures
                )
                pilot_videos = {key[0] for key in pilot_keys}
                final_videos = {key[0] for key in final_keys}
                if len(pilot_rows) != EXPECTED_COUNTS["pilot_frames"]:
                    failures.append(f"Pilot frame count is {len(pilot_rows)}, expected 320")
                if len(pilot_videos) != EXPECTED_COUNTS["pilot_videos"]:
                    failures.append(f"Pilot video count is {len(pilot_videos)}, expected 32")
                if len(final_rows) != EXPECTED_COUNTS["final_frames"]:
                    failures.append(f"Final frame count is {len(final_rows)}, expected 4744")
                if len(final_videos) != EXPECTED_COUNTS["final_videos"]:
                    failures.append(f"Final video count is {len(final_videos)}, expected 476")
                if pilot_videos & final_videos:
                    failures.append("Pilot and final manifests overlap by video ID")
                if set(pilot_keys) & set(final_keys):
                    failures.append("Pilot and final manifests overlap by frame ID")
                if set(pilot_keys) | set(final_keys) != set(audit_by_key):
                    failures.append("Pilot/final frame IDs do not exactly cover the source audit manifest")
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                failures.append(f"Could not validate frozen manifests: {exc}")
    elif dataset_root is None:
        failures.append("Set SSD_DATASET_ROOT before checking manifest paths")

    if stage3 is not None and stage3.is_dir():
        try:
            import yaml

            if dataset_yaml is not None and dataset_yaml.is_file():
                parsed_yaml = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
                if dataset_root is None:
                    failures.append("Set SSD_DATASET_ROOT before validating dataset YAML splits")
                else:
                    portable_config = portable_dataset_config(parsed_yaml, dataset_root)
                    for split_name in ("train", "val", "test"):
                        values = portable_config.get(split_name)
                        if values is None:
                            continue
                        split_values = values if isinstance(values, list) else [values]
                        for value in split_values:
                            split_path = dataset_root / PurePosixPath(value)
                            if not split_path.exists():
                                failures.append(
                                    f"Dataset YAML {split_name} path does not resolve: {split_path}"
                                )
        except ImportError as exc:
            failures.append(f"PyYAML is unavailable: {exc}")
        except (OSError, AttributeError, TypeError, ValueError, yaml.YAMLError) as exc:
            failures.append(f"Could not validate dataset YAML: {exc}")

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip())
        print(f"Git commit: {commit}; dirty: {str(dirty).lower()}")
        if dirty:
            failures.append("Git worktree is dirty; commit the exact experiment source before running")
    except (OSError, subprocess.CalledProcessError) as exc:
        failures.append(f"Could not read Git commit/dirty status: {exc}")

    if failures:
        print("PI_PREFLIGHT_FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("CPU-only execution confirmed")
    print("Frozen operational selections: YOLO 0.10 / padding 64; ROI-SSD-MobileNet 0.50 / padding 128")
    print("No desktop cost profile selected")
    print("PI_PREFLIGHT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
