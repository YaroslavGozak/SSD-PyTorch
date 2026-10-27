"""Lock the Stage 3 video-disjoint pilot and candidate evaluation manifests."""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "final_experiments" / "pilot"
SOURCE_MANIFEST = ROOT / "final_experiments" / "01_audit" / "val_manifest.jsonl"
CANONICAL_MANIFEST = ROOT / "final_experiments" / "01_audit" / "canonical_validation.jsonl"
PILOT_VIDEO_COUNT = 32
SEED = 20261112

MODEL_CONFIGS = {
    "yolo26n": ROOT / "final_experiments" / "configs" / "yolo26n.yaml",
    "roi_ssd_mobilenet": (
        ROOT / "outputs" / "batch_infer" / "best-imagenet-vid-roissd-mobilenet"
        / "20260928-phase2-bn-freeze" / "best_map.pt" / "config.yaml"
    ),
}
MODEL_CHECKPOINTS = {
    "yolo26n": (
        ROOT / "trained_models" / "best-imagenet-vid-yolo" / "best.pt",
        "5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027",
    ),
    "roi_ssd_mobilenet": (
        ROOT / "trained_models" / "best-imagenet-vid-roissd-mobilenet"
        / "20260928-phase2-bn-freeze" / "best_map.pt",
        "7fa171ab669ac53f0d4f39586c08e61a60ca79fed98fd70874ef42bfd52356ce",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_manifest(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("video_id", "frame_index", "image", "label", "gt_json"))
        for row in rows:
            writer.writerow((
                str(row["video_id"]),
                int(row["frame_index"]),
                row["image"],
                row["label"],
                json.dumps(row["gt"], separators=(",", ":")),
            ))


def validate_source_rows(rows, dataset_root: Path):
    for row in rows:
        image = dataset_root / row["image"]
        label = dataset_root / row["label"]
        if not image.is_file():
            raise FileNotFoundError(f"Pilot source image is missing: {image}")
        if not label.is_file():
            raise FileNotFoundError(f"Pilot source label is missing: {label}")
        if sha256_file(label) != row["label_sha256"]:
            raise ValueError(f"Label hash mismatch: {label}")
        decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Image cannot be decoded: {image}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-video-count", type=int, default=PILOT_VIDEO_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.pilot_video_count < 1:
        raise ValueError("--pilot-video-count must be positive")

    source_rows = load_jsonl(SOURCE_MANIFEST)
    canonical_rows = load_jsonl(CANONICAL_MANIFEST)
    source_videos = sorted({str(row["video_id"]) for row in source_rows})
    canonical_videos = {str(row["video_id"]) for row in canonical_rows}
    eligible_videos = sorted(set(source_videos) - canonical_videos)
    if len(eligible_videos) < args.pilot_video_count:
        raise ValueError(
            f"Only {len(eligible_videos)} videos remain outside canonical validation; "
            f"requested {args.pilot_video_count}"
        )

    rng = random.Random(args.seed)
    pilot_videos = set(rng.sample(eligible_videos, args.pilot_video_count))
    pilot_rows = [row for row in source_rows if str(row["video_id"]) in pilot_videos]
    final_rows = [row for row in source_rows if str(row["video_id"]) not in pilot_videos]
    if not pilot_rows or not final_rows:
        raise ValueError("Pilot/final split must both contain frames")
    if {str(row["video_id"]) for row in pilot_rows} & {str(row["video_id"]) for row in final_rows}:
        raise RuntimeError("Pilot and final candidate overlap by video")

    config = yaml.safe_load(MODEL_CONFIGS["yolo26n"].read_text(encoding="utf-8"))
    dataset_yaml = Path(config["dataset_params"]["yolo_dataset_yaml"])
    dataset_config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(dataset_config["path"])
    validate_source_rows(pilot_rows, dataset_root)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    configs_dir = OUTPUT / "configs"
    configs_dir.mkdir(exist_ok=True)
    copied_configs = {}
    for name, source in MODEL_CONFIGS.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required selected model config is missing: {source}")
        destination = configs_dir / f"{name}.yaml"
        shutil.copy2(source, destination)
        copied_configs[name] = {
            "source": str(source.relative_to(ROOT)),
            "snapshot": str(destination.relative_to(ROOT)),
            "sha256": sha256_file(destination),
        }
    checkpoints = {}
    for name, (checkpoint, expected_hash) in MODEL_CHECKPOINTS.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
        actual_hash = sha256_file(checkpoint)
        if actual_hash != expected_hash:
            raise ValueError(f"Selected {name} checkpoint hash differs from the locked manifest")
        checkpoints[name] = {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": actual_hash,
        }

    pilot_path = OUTPUT / "pilot_manifest.txt"
    final_path = OUTPUT / "final_eval_manifest_candidate.txt"
    write_manifest(pilot_path, pilot_rows)
    write_manifest(final_path, final_rows)

    class_counts = Counter(
        int(obj[0])
        for row in pilot_rows
        for obj in row["gt"]
    )
    area_ratios = [
        float(obj[3]) * float(obj[4])
        for row in pilot_rows
        for obj in row["gt"]
    ]
    per_video_frame_counts = Counter(str(row["video_id"]) for row in pilot_rows)
    metadata = {
        "status": "PREPARED_NOT_BENCHMARKED",
        "selection": {
            "method": "seeded video-level sample, excluding every canonical-validation video",
            "seed": args.seed,
            "pilot_video_count_requested": args.pilot_video_count,
            "pilot_videos": sorted(pilot_videos),
            "pilot_frames": len(pilot_rows),
            "candidate_final_eval_videos": len(source_videos) - len(pilot_videos),
            "candidate_final_eval_frames": len(final_rows),
            "pilot_frame_counts_by_video": dict(sorted(per_video_frame_counts.items())),
            "pilot_class_counts": dict(sorted(class_counts.items())),
            "pilot_gt_area_ratio_mean": sum(area_ratios) / len(area_ratios) if area_ratios else None,
            "pilot_canonical_video_overlap": 0,
            "pilot_candidate_final_video_overlap": 0,
        },
        "provenance": {
            "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
            "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "canonical_manifest": str(CANONICAL_MANIFEST.relative_to(ROOT)),
            "canonical_manifest_sha256": sha256_file(CANONICAL_MANIFEST),
            "pilot_manifest_sha256": sha256_file(pilot_path),
            "final_eval_manifest_candidate_sha256": sha256_file(final_path),
            "dataset_yaml": str(dataset_yaml),
            "dataset_yaml_sha256": sha256_file(dataset_yaml),
            "dataset_root": str(dataset_root),
            "model_configs": copied_configs,
            "selected_checkpoints": checkpoints,
            "note": (
                "The candidate final evaluation is video-disjoint from this pilot but inherits "
                "the source dataset's prior validation/model-selection reuse; it is not an "
                "untouched external test set."
            ),
        },
    }
    (OUTPUT / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    report = f"""# Stage 3 CPU pilot

## Preparation status

`PREPARED_NOT_BENCHMARKED`. The full padding sweep, fresh CPU cost calibration,
and FF/SEG/CA/FULL-TECH runs have not been executed.

## Locked split

- Pilot: {len(pilot_videos)} video sequences / {len(pilot_rows)} labeled frames.
- Final-evaluation candidate: {len(set(source_videos) - pilot_videos)} videos / {len(final_rows)} frames.
- Selection: deterministic seeded sample (`seed={args.seed}`) from validation videos
  outside the canonical 256-frame validation videos; no prediction metrics were used.
- Video overlap: zero between pilot and final-evaluation candidate.
- Pilot manifest SHA-256: `{metadata['provenance']['pilot_manifest_sha256']}`.
- Candidate manifest SHA-256: `{metadata['provenance']['final_eval_manifest_candidate_sha256']}`.

The candidate is disjoint from this pilot, but the source validation dataset was
previously reused for model selection/diagnostics. It is therefore not an untouched
external test set. The original 5,064-frame source manifest remains unchanged.

## Selected models

- YOLO26n checkpoint SHA-256: `{checkpoints['yolo26n']['sha256']}`.
- ROI-SSD-MobileNet ROI checkpoint SHA-256: `{checkpoints['roi_ssd_mobilenet']['sha256']}`.
- Exact source configuration snapshots are in [`configs/`](./configs/).
- Inference thresholds follow the selected configs; YOLO output cap/NMS are locked
  to 200 detections and IoU 0.45. CPU threads are fixed to 8 (the i7-11700K core
  count); OpenCV threads to 1.

## Smoke check

The 20-frame CPU coordinate smoke is run separately. Its outcome is recorded in
[`smoke_metadata.json`](./smoke_metadata.json); smoke metrics are not pilot results.

## Remaining protocol

1. Run `python -m experiments.cpu_pilot.run_pilot --stage padding`.
2. Run `python -m experiments.cpu_pilot.run_pilot --stage cost`.
3. Review the locked padding decision and fresh CPU cost fits.
4. Run `python -m experiments.cpu_pilot.run_pilot --stage pipeline`.

No padding decision is final until the full sweep passes the predeclared accuracy/coverage rule.
CPU cost coefficients, when measured, remain preliminary until the pipeline acceptance checks pass.
"""
    (OUTPUT / "pilot_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata["selection"], indent=2))
    print(f"Pilot manifest SHA-256: {metadata['provenance']['pilot_manifest_sha256']}")
    print(f"Final candidate SHA-256: {metadata['provenance']['final_eval_manifest_candidate_sha256']}")


if __name__ == "__main__":
    main()
