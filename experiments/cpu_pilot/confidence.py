"""Pilot-only operational confidence selection from evaluation-floor detections."""
import csv
import hashlib
import json


QUALITY_CONFIDENCE_FLOOR = 0.01
CONFIDENCE_GRID = (0.01,) + tuple(value / 100 for value in range(5, 100, 5))
SELECTION_RULE = (
    "maximum micro F1@IoU=0.5 on full-frame pilot using evaluation_conf detections; "
    "ties: higher recall, then lower threshold"
)


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def select_threshold(frame_rows):
    from .oracle_diagnosis import match_objects

    curves = []
    records = [
        (json.loads(row["prediction_records"]), json.loads(row["ground_truth_records"]))
        for row in frame_rows
    ]
    total_gt = sum(len(boxes) for _, gt in records for boxes in gt.values())
    if total_gt == 0:
        raise ValueError("Operational confidence cannot be selected without pilot GT")
    for threshold in CONFIDENCE_GRID:
        tp = fp = 0
        for predictions, gt in records:
            counts = match_objects(
                [det for det in predictions if det["confidence"] >= threshold], gt
            )
            tp += counts["tp"]
            fp += counts["fp"]
        fn = total_gt - tp
        curves.append(dict(
            threshold=threshold, tp=tp, fp=fp, fn=fn,
            precision=tp / (tp + fp) if tp + fp else 0.0,
            recall=tp / total_gt,
            f1=2 * tp / (2 * tp + fp + fn),
        ))
    selected = max(
        curves, key=lambda row: (row["f1"], row["recall"], -row["threshold"])
    )
    if selected["tp"] == 0:
        raise ValueError("No operational confidence candidate detects any pilot GT")
    return selected, curves


def load_frozen_threshold(directory, model_name, config):
    from pathlib import Path

    directory = Path(directory)
    path = directory / "operational_confidence.json"
    receipt = directory / "operational_confidence.sha256"
    if not path.is_file() or not receipt.is_file():
        raise FileNotFoundError(
            "Select and freeze operational confidence on the full pilot first: --stage confidence"
        )
    if digest(path) != receipt.read_text(encoding="utf-8").strip():
        raise ValueError("Frozen operational confidence was modified")
    decision = json.loads(path.read_text(encoding="utf-8"))
    selected = decision.get("models", {}).get(model_name)
    if selected is None:
        raise ValueError(
            f"Frozen operational_conf is missing {model_name}; run --stage confidence "
            "for all selected models or specify --operational-conf explicitly"
        )
    expected = {
        "checkpoint_sha256": config["audit_protocol"]["checkpoint_sha256"],
        "pilot_manifest_sha256": digest(directory / "pilot_manifest.txt"),
        "config_sha256": digest(directory / "configs" / f"{model_name}.yaml"),
    }
    if decision.get("selection_rule") != SELECTION_RULE or any(
        selected.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Operational confidence identity/protocol mismatch")
    for filename, expected_hash in selected["evidence_sha256"].items():
        expected_prefix = f"confidence_{model_name}_"
        if not filename.startswith(expected_prefix):
            raise ValueError("Unexpected operational confidence evidence file")
        if digest(directory / filename) != expected_hash:
            raise ValueError("Operational confidence evidence changed")
    candidates_path = directory / selected["evidence_files"]["candidates"]
    with candidates_path.open(encoding="utf-8", newline="") as stream:
        candidates = list(csv.DictReader(stream))
    best = max(
        candidates,
        key=lambda row: (
            float(row["f1"]), float(row["recall"]), -float(row["threshold"])
        ),
    )
    threshold = float(selected["operational_conf"])
    if threshold != float(best["threshold"]) or threshold not in CONFIDENCE_GRID:
        raise ValueError("Frozen operational_conf differs from the pilot selection")
    return threshold


def run_selection(split, manifest_rows, device, pilot_module=None):
    if pilot_module is None:
        from . import run_pilot as pilot_module
    pilot = pilot_module
    directory = pilot.PILOT_DIR
    decision_path = directory / "operational_confidence.json"
    receipt_path = directory / "operational_confidence.sha256"
    if decision_path.exists() or receipt_path.exists():
        raise FileExistsError(
            "Operational confidence is already frozen; use a fresh session to select again"
        )

    decisions = {}
    for name in pilot.MODEL_NAMES:
        model, dataset, config, _ = pilot._load_model(
            name, device, evaluation_conf=QUALITY_CONFIDENCE_FLOOR
        )
        indices = pilot._dataset_indices(
            dataset, manifest_rows, pilot._dataset_root(config)
        )
        summary, rows = pilot._run_pipeline(
            name, model, dataset, config,
            QUALITY_CONFIDENCE_FLOOR, QUALITY_CONFIDENCE_FLOOR,
            manifest_rows, indices, "confidence_full_frame_floor", "FF", 0,
        )
        selected, curve = select_threshold(rows)
        summary.update({
            "evaluation_conf": QUALITY_CONFIDENCE_FLOOR,
            "operational_conf": selected["threshold"],
        })
        filenames = {
            "candidates": f"confidence_{name}_candidates.csv",
            "frame_metrics": f"confidence_{name}_frame_metrics.csv",
            "summary": f"confidence_{name}_ff_summary.csv",
        }
        pilot._write_csv(directory / filenames["candidates"], curve)
        pilot._write_csv(directory / filenames["frame_metrics"], rows)
        pilot._write_csv(directory / filenames["summary"], [summary])
        config_path = directory / "configs" / f"{name}.yaml"
        decisions[name] = dict(
            {key: value for key, value in selected.items() if key != "threshold"},
            status="FROZEN_FROM_PILOT",
            operational_conf=selected["threshold"],
            evaluation_conf=QUALITY_CONFIDENCE_FLOOR,
            selection_rule=SELECTION_RULE,
            candidate_grid=list(CONFIDENCE_GRID),
            pilot_frames=len(rows),
            pilot_manifest_sha256=split["provenance"]["pilot_manifest_sha256"],
            checkpoint_sha256=config["audit_protocol"]["checkpoint_sha256"],
            config_sha256=digest(config_path),
            evidence_files=filenames,
            evidence_sha256={
                filename: digest(directory / filename) for filename in filenames.values()
            },
        )
        del model, dataset

    document = {
        "status": "FROZEN_FROM_PILOT",
        "evaluation_conf": QUALITY_CONFIDENCE_FLOOR,
        "selection_rule": SELECTION_RULE,
        "models": decisions,
    }
    with decision_path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(digest(decision_path) + "\n")
