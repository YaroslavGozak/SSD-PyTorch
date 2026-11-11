"""Run or analyze the isolated YOLO operational-confidence/padding diagnostic."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from experiments.cpu_pilot import run_pilot as pilot
from experiments.cpu_pilot.prepare import MODEL_CHECKPOINTS
from tools.trackers.static_padding_tracker import StaticPaddingTracker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_DIR = ROOT / "final_experiments" / "pilot_dual_conf_new"
DEFAULT_OUTPUT_DIR = ROOT / "final_experiments" / "yolo_conf_padding_diagnostic_20261109"
EXPECTED_MANIFEST_SHA256 = "817f5b145b774dc91ee0fa8d271e02c11d71d70bb50ff13082a0ba9487aa63cb"
EXPECTED_CHECKPOINT_SHA256 = "5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027"
EXPECTED_CONFIG_SHA256 = "2605281bb1fb0cf7d087f65642134322cda9d5b35a89903a40859792f210130e"
OPERATIONAL_CONFS = (0.01, 0.05, 0.10, 0.20, 0.35)
PADDINGS = (64, 96, 128, 160, 192, 224)
MAP_RETENTION_GATE = 0.95
COVERAGE_GATE = 0.98
AREA_RATIO_TIE_TOLERANCE = 0.01


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_id(value):
    text = str(value)
    return str(int(text)) if text.isdigit() else text


def _manifest_gt_count(rows):
    total = 0
    for row in rows:
        boxes = json.loads(row["gt_json"])
        total += len(boxes)
    return total


def _manifest_video_count(rows):
    return len({row["video_id"] for row in rows})


def _read_inputs(pilot_dir):
    pilot_dir = Path(pilot_dir).resolve()
    manifest_path = pilot_dir / "pilot_manifest.txt"
    split_path = pilot_dir / "split_metadata.json"
    snapshot_path = pilot_dir / "configs" / "yolo26n.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pilot manifest not found: {manifest_path}")
    if not split_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError(f"Prepared YOLO pilot metadata/config missing under {pilot_dir}")

    manifest_hash = _sha256(manifest_path)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Pilot manifest hash mismatch: {manifest_hash}")
    rows = pilot._read_tsv_manifest(manifest_path)
    counts = (len(rows), _manifest_video_count(rows), _manifest_gt_count(rows))
    if counts != (320, 32, 560):
        raise ValueError(f"Expected pilot subset (320 frames, 32 videos, 560 GT), found {counts}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    provenance = split["provenance"]
    locked_checkpoint = provenance["selected_checkpoints"]["yolo26n"]
    checkpoint_path, checkpoint_hash = MODEL_CHECKPOINTS["yolo26n"]
    config_hash = provenance["model_configs"]["yolo26n"]["sha256"]
    if locked_checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Prepared pilot does not use the already-selected YOLO26n checkpoint")
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Repository's Stage 3 checkpoint selection differs from the diagnostic lock")
    if _sha256(ROOT / checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("YOLO26n checkpoint SHA-256 does not match the diagnostic lock")
    if config_hash != EXPECTED_CONFIG_SHA256 or _sha256(snapshot_path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("Prepared YOLO26n config SHA-256 does not match the diagnostic lock")
    snapshot_config = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    dataset_yaml = Path(snapshot_config["dataset_params"]["yolo_dataset_yaml"]).resolve()
    expected_dataset_yaml = Path(provenance["dataset_yaml"]).resolve()
    if dataset_yaml != expected_dataset_yaml:
        raise ValueError("Prepared YOLO config does not use the dataset YAML locked by the pilot")
    if _sha256(dataset_yaml) != provenance["dataset_yaml_sha256"]:
        raise ValueError("Sampled YOLO dataset YAML SHA-256 does not match the pilot lock")

    return pilot_dir, rows, split, {
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": manifest_hash,
        "frame_count": counts[0],
        "video_count": counts[1],
        "ground_truth_count": counts[2],
        "config_path": str(snapshot_path.relative_to(ROOT)),
        "config_sha256": config_hash,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "dataset_yaml_path": str(dataset_yaml),
        "dataset_yaml_sha256": provenance["dataset_yaml_sha256"],
    }


def _output_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "root": output_dir,
        "results": output_dir / "results",
        "analysis": output_dir / "analysis",
        "resolved_config": output_dir / "resolved_config.yaml",
        "baseline": output_dir / "results" / "full_frame_baseline.json",
        "sweep": output_dir / "results" / "sweep.csv",
        "frames": output_dir / "results" / "frame_level.csv",
        "summary": output_dir / "analysis" / "summary.csv",
        "gate_matrix": output_dir / "analysis" / "gate_matrix.csv",
        "report": output_dir / "analysis" / "diagnostic_report.md",
    }


def _write_resolved_config(path, source_config, provenance, class_names):
    resolved = dict(source_config)
    resolved["diagnostic_protocol"] = {
        "purpose": "Isolated YOLO26n operational-confidence x static-padding diagnostic",
        "evaluation_conf": 0.01,
        "operational_confs": list(OPERATIONAL_CONFS),
        "paddings": list(PADDINGS),
        "mode": "SEG",
        "localization": "previous operational detections via StaticPaddingTracker",
        "hold_last_for_frames": 0,
        "periodic_full_frame_interval": pilot.KEY_FRAME_INTERVAL,
        "merge": "disabled",
        "device": "cpu",
        "pilot": provenance,
        "class_list": list(class_names),
        "gates": {
            "mAP50_retention_min": MAP_RETENTION_GATE,
            "gt_roi_coverage_mean_min": COVERAGE_GATE,
        },
        "recommendation": {
            "primary": "minimum tensor_area_ratio_mean among gate-passing configurations",
            "secondary": [
                "candidate_segments_per_frame",
                "smaller padding",
                "higher operational_conf",
            ],
            "near_best_tensor_area_relative_tolerance": AREA_RATIO_TIE_TOLERANCE,
        },
    }
    rendered = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(
                f"Resolved config already exists and differs: {path}. Use a fresh output directory."
            )
    else:
        path.write_text(rendered, encoding="utf-8")


def _validate_empty_results(paths):
    allowed_files = {
        paths["root"] / "README.md",
        paths["root"] / "RUNBOOK.md",
        paths["resolved_config"],
    }
    existing = [
        path for path in paths["root"].rglob("*")
        if path.is_file() and path not in allowed_files
    ]
    if existing:
        raise FileExistsError(
            "Diagnostic result files already exist; refusing to append/overwrite. "
            f"Use --analyze-only or a fresh --output directory: {existing[0]}"
        )


def _coverage_for_gt(gt, rois, full_frame):
    uncovered = []
    for class_name, boxes in gt.items():
        for box in boxes:
            x1, y1, x2, y2 = map(float, box)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            covered = bool(full_frame)
            if not covered and area > 0:
                for roi in rois:
                    rx1, ry1, rx2, ry2 = map(float, roi)
                    intersection = max(0.0, min(x2, rx2) - max(x1, rx1)) * max(
                        0.0, min(y2, ry2) - max(y1, ry1)
                    )
                    if intersection / area >= 0.5:
                        covered = True
                        break
            if not covered:
                uncovered.append({"class": class_name, "bbox": box})
    return uncovered


def _frame_records(rows, operational_conf, padding):
    records = []
    for row in rows:
        gt = json.loads(row["ground_truth_records"])
        rois = json.loads(row["segments_used"])
        uncovered = _coverage_for_gt(gt, rois, row["input_mode"] == "full")
        gt_count = sum(len(boxes) for boxes in gt.values())
        if gt_count - len(uncovered) != int(row["gt_covered"]):
            raise RuntimeError(
                f"Diagnostic GT coverage disagrees with Stage 3 on {row['frame_id']}"
            )
        records.append({
            "video_id": row["video_id"],
            "frame_id": row["frame_index"],
            "operational_conf": operational_conf,
            "padding": padding,
            "gt_count": gt_count,
            "evaluation_detection_count": row["evaluation_detection_count"],
            "operational_detection_count": row["operational_detection_count"],
            "candidate_segment_count": row["candidate_segments"],
            "next_candidate_segment_count": row["next_candidate_segments"],
            "covered_gt_count": int(row["gt_covered"]),
            "uncovered_gt_count": len(uncovered),
            "uncovered_gt": json.dumps(uncovered, separators=(",", ":")),
            "full_frame_fallback": row["full_frame_fallback"],
            "fallback_reason": row["fallback_reason"],
        })
    return records


def _run_summary(summary, frames, operational_conf, padding, baseline_map50):
    retention = summary["mAP50"] / baseline_map50 if baseline_map50 > 0 else None
    if retention is None:
        raise ValueError("Full-frame baseline mAP50 is zero; retention is undefined")
    map_pass = retention >= MAP_RETENTION_GATE
    coverage_pass = summary["gt_roi_coverage_mean"] >= COVERAGE_GATE
    per_video = defaultdict(list)
    for row in frames:
        if row["uncovered_gt_count"]:
            per_video[row["video_id"]].append(row)
    ranges = _temporal_ranges(frames)
    return {
        "operational_conf": operational_conf,
        "padding": padding,
        "mAP50": summary["mAP50"],
        "mAP50_95": summary["mAP50_95"],
        "recall50": summary["recall50"],
        "mAP50_retention": retention,
        "gt_roi_coverage_mean": summary["gt_roi_coverage_mean"],
        "gt_roi_coverage_p5": summary["gt_roi_coverage_p5"],
        "evaluation_detections_per_frame": sum(
            int(row["evaluation_detection_count"]) for row in frames
        ) / max(1, len(frames)),
        "operational_detections_per_frame": sum(
            int(row["operational_detection_count"]) for row in frames
        ) / max(1, len(frames)),
        "candidate_segments_per_frame": summary["candidate_segments_mean"],
        "next_candidate_segments_per_frame": summary["next_candidate_segments_mean"],
        "nn_calls_per_frame": summary["nn_calls_per_frame"],
        "processed_area_ratio_mean": summary["processed_area_ratio_mean"],
        "tensor_area_ratio_mean": summary["tensor_area_ratio_mean"],
        "full_frame_fraction": summary["full_frame_fraction"],
        "MAP_GATE_PASS": map_pass,
        "COVERAGE_GATE_PASS": coverage_pass,
        "FINAL_GATE_PASS": map_pass and coverage_pass,
        "uncovered_gt_count": sum(row["uncovered_gt_count"] for row in frames),
        "videos_with_uncovered_gt": len(per_video),
        "consecutive_miss_ranges": "; ".join(ranges),
    }


def _temporal_ranges(frames):
    by_video = defaultdict(list)
    for row in frames:
        if row["uncovered_gt_count"]:
            by_video[row["video_id"]].append(row)
    result = []
    for video_id, rows in sorted(by_video.items()):
        rows.sort(key=lambda item: int(item["frame_id"]))
        run = []
        for row in rows:
            if run and int(row["frame_id"]) != int(run[-1]["frame_id"]) + 1:
                result.append(_format_miss_range(video_id, run))
                run = []
            run.append(row)
        if run:
            result.append(_format_miss_range(video_id, run))
    return result


def _format_miss_range(video_id, rows):
    start = int(rows[0]["frame_id"])
    end = int(rows[-1]["frame_id"])
    missed = sum(row["uncovered_gt_count"] for row in rows)
    return (
        f"video {_video_id(video_id)} frames {start}-{end}: "
        f"{len(rows)} consecutive frame(s), {missed} missed GT"
    )


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _gate_matrix(rows):
    lookup = {(row["operational_conf"], row["padding"]): row for row in rows}
    matrix = []
    for confidence in OPERATIONAL_CONFS:
        matrix.append({
            "operational_conf": confidence,
            **{
                f"p{padding}": (
                    "PASS" if lookup[(confidence, padding)]["FINAL_GATE_PASS"] else "FAIL"
                )
                for padding in PADDINGS
            },
        })
    return matrix


def _recommend(rows):
    passing = [row for row in rows if row["FINAL_GATE_PASS"]]
    if not passing:
        return None
    best_area = min(row["tensor_area_ratio_mean"] for row in passing)
    near_best = [
        row for row in passing
        if row["tensor_area_ratio_mean"] <= best_area * (1 + AREA_RATIO_TIE_TOLERANCE)
    ]
    return min(
        near_best,
        key=lambda row: (
            row["candidate_segments_per_frame"],
            row["padding"],
            -row["operational_conf"],
            row["tensor_area_ratio_mean"],
        ),
    )


def _write_analysis(paths, sweep_rows, frame_rows):
    if not paths["sweep"].is_file() or not paths["frames"].is_file() or not paths["baseline"].is_file():
        raise FileNotFoundError("Analysis needs full_frame_baseline.json, sweep.csv, and frame_level.csv")
    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    baseline_map50 = baseline["mAP50"]
    summaries = []
    for row in sweep_rows:
        row["MAP_GATE_PASS"] = str(row["MAP_GATE_PASS"]).lower() == "true"
        row["COVERAGE_GATE_PASS"] = str(row["COVERAGE_GATE_PASS"]).lower() == "true"
        row["FINAL_GATE_PASS"] = str(row["FINAL_GATE_PASS"]).lower() == "true"
        confidence = float(row["operational_conf"])
        padding = int(row["padding"])
        subset = [
            item for item in frame_rows
            if float(item["operational_conf"]) == confidence and int(item["padding"]) == padding
        ]
        if len(subset) != 320:
            raise ValueError(
                f"Expected 320 frame records for confidence={confidence}, padding={padding}; "
                f"found {len(subset)}"
            )
        summary = dict(row)
        summary["uncovered_gt_count"] = sum(int(item["uncovered_gt_count"]) for item in subset)
        summary["videos_with_uncovered_gt"] = len({
            item["video_id"] for item in subset if int(item["uncovered_gt_count"])
        })
        summary["consecutive_miss_ranges"] = "; ".join(_temporal_ranges(subset))
        summaries.append(summary)
    _write_csv(paths["summary"], summaries)
    _write_csv(paths["gate_matrix"], _gate_matrix(summaries))
    _write_report(paths, baseline, summaries, frame_rows)


def _write_report(paths, baseline, summaries, frame_rows):
    lookup = {(float(row["operational_conf"]), int(row["padding"])): row for row in summaries}
    lines = [
        "# YOLO26n operational-confidence x padding diagnostic",
        "",
        "## Protocol",
        "",
        "- Evaluation confidence is fixed at 0.01 for the one full-frame baseline and all SEG runs.",
        "- Only operational detections seed the static previous-localization tracker; hold is 0.",
        "- Periodic full-frame interval is 10 frames; merge/CA/FULL-TECH are disabled.",
        "- The same pinned 32-video, 320-frame pilot manifest is used without shuffling.",
        f"- Full-frame baseline mAP50: `{baseline['mAP50']:.6f}`.",
        f"- Gates: mAP50 retention >= {MAP_RETENTION_GATE:.2f}; GT ROI coverage mean >= {COVERAGE_GATE:.2f}.",
        "",
        "## Gate matrix",
        "",
        "| operational_conf | 64 | 96 | 128 | 160 | 192 | 224 |",
        "|---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for confidence in OPERATIONAL_CONFS:
        cells = [
            "PASS" if lookup[(confidence, padding)]["FINAL_GATE_PASS"] else "FAIL"
            for padding in PADDINGS
        ]
        lines.append(f"| {confidence:.2f} | " + " | ".join(cells) + " |")

    recommendation = _recommend(summaries)
    lines.extend(["", "## Recommendation", ""])
    if recommendation is None:
        lines.append("No configuration passed both gates; do not promote a diagnostic setting.")
    else:
        lines.extend([
            f"Recommended `(operational_conf, padding)` = "
            f"`({float(recommendation['operational_conf']):.2f}, {int(recommendation['padding'])})`.",
            f"It has tensor area ratio `{float(recommendation['tensor_area_ratio_mean']):.4f}`, "
            f"`{float(recommendation['candidate_segments_per_frame']):.3f}` candidate segments/frame, "
            f"mAP50 retention `{float(recommendation['mAP50_retention']):.4f}`, and "
            f"coverage `{float(recommendation['gt_roi_coverage_mean']):.4f}`.",
            f"Selection minimizes tensor area among passing configurations; configurations within "
            f"{AREA_RATIO_TIE_TOLERANCE:.0%} of the minimum are ordered by fewer candidate segments, "
            "smaller padding, then higher confidence. This is diagnostic only and does not change "
            "the main Stage 3 configuration.",
        ])

    lines.extend([
        "",
        "## Coverage by threshold and padding",
        "",
        "Entries show `coverage / mAP50 retention / candidate segments per frame`.",
        "",
        "| Conf | Padding | Coverage | mAP50 retention | Candidate seg/frame | Tensor area ratio | Gate |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for confidence in OPERATIONAL_CONFS:
        for padding in PADDINGS:
            row = lookup[(confidence, padding)]
            lines.append(
                f"| {confidence:.2f} | {padding} | "
                f"{float(row['gt_roi_coverage_mean']):.4f} | "
                f"{float(row['mAP50_retention']):.4f} | "
                f"{float(row['candidate_segments_per_frame']):.3f} | "
                f"{float(row['tensor_area_ratio_mean']):.4f} | "
                f"{'PASS' if row['FINAL_GATE_PASS'] else 'FAIL'} |"
            )

    lines.extend(["", "## Factor comparisons", ""])
    coverage_pairs = [
        (confidence, padding)
        for confidence in OPERATIONAL_CONFS
        for padding in PADDINGS
        if lookup[(confidence, padding)]["COVERAGE_GATE_PASS"]
    ]
    high_conf_pass_paddings = [
        padding for padding in PADDINGS
        if lookup[(0.35, padding)]["COVERAGE_GATE_PASS"]
    ]
    low_conf_pass_paddings = [
        padding for padding in PADDINGS
        if lookup[(0.01, padding)]["COVERAGE_GATE_PASS"]
    ]
    if not coverage_pairs:
        lines.append("No tested pair passed the coverage gate; this grid cannot isolate a successful factor setting.")
    elif high_conf_pass_paddings:
        lines.append(
            "Larger padding alone can recover sufficient coverage at operational_conf=0.35 "
            f"(passing paddings: {high_conf_pass_paddings})."
        )
    elif low_conf_pass_paddings:
        lines.append(
            "Reducing operational confidence is sufficient at padding(s) "
            f"{low_conf_pass_paddings}; padding 224 alone does not rescue conf=0.35."
        )
    else:
        lines.append(
            "Neither conf=0.35 at any tested padding nor conf=0.01 at any tested padding "
            "passes coverage; inspect intermediate pairs below for an interaction."
        )
    if (
        coverage_pairs
        and not high_conf_pass_paddings
        and not low_conf_pass_paddings
    ):
        lines.append(
            "Coverage passes only at intermediate confidence/padding pairs, indicating both "
            "factors are needed within the tested grid."
        )
    for padding in PADDINGS:
        low = lookup[(0.01, padding)]
        high = lookup[(0.35, padding)]
        change = float(low["gt_roi_coverage_mean"]) - float(high["gt_roi_coverage_mean"])
        lines.append(
            f"- At padding {padding}, reducing operational confidence from 0.35 to 0.01 "
            f"changes coverage by {change:+.4f} "
            f"({float(high['gt_roi_coverage_mean']):.4f} -> "
            f"{float(low['gt_roi_coverage_mean']):.4f})."
        )
    for confidence in OPERATIONAL_CONFS:
        low = lookup[(confidence, 64)]
        high = lookup[(confidence, 224)]
        change = float(high["gt_roi_coverage_mean"]) - float(low["gt_roi_coverage_mean"])
        lines.append(
            f"- At operational confidence {confidence:.2f}, increasing padding from 64 to 224 "
            f"changes coverage by {change:+.4f} "
            f"({float(low['gt_roi_coverage_mean']):.4f} -> "
            f"{float(high['gt_roi_coverage_mean']):.4f})."
        )

    lines.extend(["", "## Temporal misses", ""])
    failed_coverage = [
        row for row in summaries if not row["COVERAGE_GATE_PASS"]
    ]
    if not failed_coverage:
        lines.append("All configurations passed the coverage gate; no failed-coverage runs to summarize.")
    else:
        for row in failed_coverage:
            ranges = row["consecutive_miss_ranges"] or "no uncovered GT objects"
            lines.append(
                f"- conf={float(row['operational_conf']):.2f}, padding={int(row['padding'])}: "
                f"{int(row['uncovered_gt_count'])} missed GT across "
                f"{int(row['videos_with_uncovered_gt'])} videos; {ranges}."
            )
    known_case = [
        row for row in frame_rows
        if _video_id(row["video_id"]) == "23010"
        and 83 <= int(row["frame_id"]) <= 89
    ]
    lines.extend(["", "### Known video 23010, frames 83-89", ""])
    if not known_case:
        lines.append("Those frames are not present in the pinned pilot subset.")
    else:
        for confidence in OPERATIONAL_CONFS:
            entries = []
            for padding in PADDINGS:
                selected = [
                    row for row in known_case
                    if float(row["operational_conf"]) == confidence and int(row["padding"]) == padding
                ]
                misses = sum(int(row["uncovered_gt_count"]) for row in selected)
                entries.append(f"p{padding}:{misses}")
            lines.append(f"- conf={confidence:.2f}: " + ", ".join(entries) + " missed GT.")

    lines.extend([
        "",
        "## Output notes",
        "",
        "- `sweep.csv` has one row per configuration and separate MAP_GATE_PASS, "
        "COVERAGE_GATE_PASS, and FINAL_GATE_PASS fields.",
        "- `frame_level.csv` records evaluation/operational detections, current and next candidate "
        "segments, fallback, covered/uncovered counts, and each uncovered GT class/bbox.",
        "- Timing is recorded by the shared Stage 3 pipeline but is not used to select a configuration.",
        "- This diagnostic does not update the frozen operational confidence or main Stage 3 outputs.",
    ])
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_model(pilot_dir, manifest_rows):
    pilot.PILOT_DIR = Path(pilot_dir)
    pilot.SPLIT_METADATA = pilot.PILOT_DIR / "split_metadata.json"
    pilot.MODEL_NAMES = ("yolo26n",)
    pilot._configure_runtime()
    pilot._load_session()
    model, dataset, config, evaluation_conf = pilot._load_model(
        "yolo26n", torch.device("cpu"), 0.01
    )
    indices = pilot._dataset_indices(dataset, manifest_rows, pilot._dataset_root(config))
    if len(indices) != len(manifest_rows):
        raise RuntimeError("Manifest selection and dataset indices are not one-to-one")
    names = model._yolo.names
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    return model, dataset, config, evaluation_conf, indices, names


def _run_pipeline(model, dataset, config, evaluation_conf, manifest_rows, indices,
                  operational_conf, padding, mode, run_name):
    return pilot._run_pipeline(
        "yolo26n",
        model,
        dataset,
        config,
        evaluation_conf,
        operational_conf,
        manifest_rows,
        indices,
        run_name,
        mode,
        padding,
        coefficients=None,
        debug=False,
    )


def _run_smoke(args, pilot_dir, manifest_rows, provenance, paths):
    if not 10 <= args.smoke_frames <= 20:
        raise ValueError("--smoke-frames must be between 10 and 20")
    if args.smoke_padding not in PADDINGS:
        raise ValueError(f"--smoke-padding must be one of {list(PADDINGS)}")
    model, dataset, config, evaluation_conf, indices, names = _prepare_model(
        pilot_dir, manifest_rows
    )
    _write_resolved_config(paths["resolved_config"], config, provenance, names)
    confidence = pilot._validate_confidence("operational_conf", args.smoke_operational_conf)
    if confidence < evaluation_conf:
        raise ValueError("operational confidence must be >= evaluation confidence")
    if confidence not in OPERATIONAL_CONFS:
        raise ValueError(f"--smoke-operational-conf must be one of {list(OPERATIONAL_CONFS)}")
    selected_rows = manifest_rows[:args.smoke_frames]
    selected_indices = indices[:args.smoke_frames]
    received = []

    class RecordingTracker(StaticPaddingTracker):
        def update(self, detections, *positional, **kwargs):
            received.append([
                {
                    "bbox": list(detection["bbox"]),
                    "class": detection["class"],
                    "confidence": float(detection["confidence"]),
                }
                for detection in detections
            ])
            return super().update(detections, *positional, **kwargs)

    with patch("experiments.cpu_pilot.run_pilot.StaticPaddingTracker", RecordingTracker):
        _, rows = _run_pipeline(
            model, dataset, config, evaluation_conf, selected_rows, selected_indices,
            confidence, args.smoke_padding, "SEG", "diagnostic_smoke",
        )
    if len(received) != len(rows):
        raise RuntimeError("Smoke test did not capture tracker inputs for every frame")
    for row, tracker_input in zip(rows, received):
        evaluation = json.loads(row["prediction_records"])
        operational = json.loads(row["operational_prediction_records"])
        if len(evaluation) < len(operational):
            raise RuntimeError(f"Evaluation count is below operational count at {row['frame_id']}")
        if any(float(item["confidence"]) < confidence for item in operational):
            raise RuntimeError(f"Below-threshold detection entered operational branch at {row['frame_id']}")
        canonical = lambda items: Counter(
            (item["class"], tuple(item["bbox"]), float(item["confidence"])) for item in items
        )
        if canonical(tracker_input) != canonical(operational):
            raise RuntimeError(
                f"Tracker input differs from operational detections at {row['frame_id']}"
            )
    print(
        f"Smoke passed: {len(rows)} frames; evaluation detections >= operational detections; "
        "tracker input exactly matched operational detections.",
        flush=True,
    )


def _run_sweep(args, pilot_dir, manifest_rows, provenance, paths):
    _validate_empty_results(paths)
    model, dataset, config, evaluation_conf, indices, names = _prepare_model(
        pilot_dir, manifest_rows
    )
    _write_resolved_config(paths["resolved_config"], config, provenance, names)
    baseline_conf = min(args.operational_confs)
    baseline, _ = _run_pipeline(
        model, dataset, config, evaluation_conf, manifest_rows, indices,
        baseline_conf, 0, "FF", "diagnostic_full_frame",
    )
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["analysis"].mkdir(parents=True, exist_ok=True)
    paths["baseline"].write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sweep_rows = []
    frame_rows = []
    for confidence in args.operational_confs:
        for padding in args.paddings:
            summary, pipeline_rows = _run_pipeline(
                model, dataset, config, evaluation_conf, manifest_rows, indices,
                confidence, padding, "SEG",
                f"diagnostic_conf{confidence:.2f}_padding{padding}",
            )
            config_frames = _frame_records(pipeline_rows, confidence, padding)
            sweep_rows.append(_run_summary(summary, config_frames, confidence, padding, baseline["mAP50"]))
            frame_rows.extend(config_frames)
            print(
                f"[diagnostic] conf={confidence:.2f} padding={padding} "
                f"coverage={sweep_rows[-1]['gt_roi_coverage_mean']:.4f} "
                f"retention={sweep_rows[-1]['mAP50_retention']:.4f}",
                flush=True,
            )
    _write_csv(paths["sweep"], sweep_rows)
    _write_csv(paths["frames"], frame_rows)
    _write_analysis(paths, sweep_rows, frame_rows)
    print(f"Diagnostic sweep complete. Report: {paths['report']}", flush=True)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--evaluation-conf", type=float, default=0.01)
    parser.add_argument("--operational-confs", type=float, nargs="+", default=list(OPERATIONAL_CONFS))
    parser.add_argument("--paddings", type=int, nargs="+", default=list(PADDINGS))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-frames", type=int, default=12)
    parser.add_argument("--smoke-operational-conf", type=float, default=0.05)
    parser.add_argument("--smoke-padding", type=int, default=128)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluation_conf != 0.01:
        parser.error("--evaluation-conf is fixed at 0.01 for this diagnostic")
    if sorted(args.operational_confs) != sorted(OPERATIONAL_CONFS):
        parser.error(f"--operational-confs must be exactly {list(OPERATIONAL_CONFS)}")
    if sorted(args.paddings) != sorted(PADDINGS):
        parser.error(f"--paddings must be exactly {list(PADDINGS)}")
    if args.analyze_only and args.smoke_only:
        parser.error("--analyze-only and --smoke-only cannot be combined")
    return args


def main(argv=None):
    args = _parse_args(argv)
    paths = _output_paths(args.output.resolve())
    pilot_dir, manifest_rows, _split, provenance = _read_inputs(args.pilot_dir)
    if paths["root"] == ROOT / "final_experiments" / "pilot":
        raise ValueError("Diagnostic output must not target final_experiments/pilot")
    if (
        paths["root"] == pilot_dir
        or pilot_dir in paths["root"].parents
        or paths["root"] in pilot_dir.parents
    ):
        raise ValueError("Diagnostic output must not overlap the source pilot")
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["analysis"].mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        _write_analysis(paths, _read_csv(paths["sweep"]), _read_csv(paths["frames"]))
        print(f"Analysis regenerated: {paths['report']}", flush=True)
        return

    if args.smoke_only:
        _run_smoke(args, pilot_dir, manifest_rows, provenance, paths)
        return

    _run_sweep(args, pilot_dir, manifest_rows, provenance, paths)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


if __name__ == "__main__":
    main()
