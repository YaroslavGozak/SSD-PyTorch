"""Joint Stage 3 selection of operational confidence and segment padding."""

import ast
import csv
import json
import math
from pathlib import Path

import yaml


MAP_RETENTION_GATE = 0.95
COVERAGE_GATE = 0.98
AREA_RATIO_TIE_TOLERANCE = 0.01
SELECTION_RULE = (
    "Pass both fixed quality gates, then minimize tensor_area_ratio_mean; "
    "within 1% of the minimum, prefer fewer candidate segments/frame, smaller "
    "padding, then higher operational_conf"
)
LEGACY_CALIBRATION_IMPLEMENTATION_SHA256 = (
    "1598c6e19b4911dc4bd2fae4cdba9ae83cd3b4e66c137925b0a78b9931e04ca8"
)


def _read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def calibration_grid(config, model_name, evaluation_conf):
    values = config.get("calibration")
    if not isinstance(values, dict):
        raise ValueError(
            f"{model_name} config has no calibration grids; prepare a fresh output "
            "with calibration.operational_conf_values and calibration.padding_values"
        )
    operational_confs = [float(value) for value in values.get("operational_conf_values", [])]
    paddings = [int(value) for value in values.get("padding_values", [])]
    if (
        not operational_confs
        or len(operational_confs) != len(set(operational_confs))
        or any(
            not math.isfinite(value) or value < evaluation_conf or value > 1
            for value in operational_confs
        )
    ):
        raise ValueError(f"{model_name} operational confidence grid is invalid: {operational_confs}")
    if not paddings or len(paddings) != len(set(paddings)) or any(value <= 0 for value in paddings):
        raise ValueError(f"{model_name} padding grid is invalid: {paddings}")
    if any(
        not math.isfinite(float(value)) or float(value) != int(value)
        for value in values.get("padding_values", [])
    ):
        raise ValueError(f"{model_name} padding values must be integers: {values['padding_values']}")
    return operational_confs, paddings


def select_configuration(rows):
    eligible = [row for row in rows if row["FINAL_GATE_PASS"]]
    if not eligible:
        return None
    best_area = min(float(row["tensor_area_ratio_mean"]) for row in eligible)
    near_best = [
        row for row in eligible
        if float(row["tensor_area_ratio_mean"]) <= best_area * (1 + AREA_RATIO_TIE_TOLERANCE)
    ]
    return min(
        near_best,
        key=lambda row: (
            float(row["candidate_segments_per_frame"]),
            int(row["padding"]),
            -float(row["operational_conf"]),
        ),
    )


def _csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty calibration CSV: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _require_fresh_destination(directory):
    existing = [
        directory / name for name in (
            "calibration_sweep.csv",
            "calibration_gate_matrix.csv",
            "selected_operational_config.json",
            "selected_operational_config.sha256",
            "calibration_report.md",
            "full_frame_baseline.json",
        )
        if (directory / name).exists()
    ]
    if existing:
        raise FileExistsError(
            f"Calibration artifacts already exist at {directory}; "
            "validate and reuse the frozen result or use a fresh pilot directory"
        )


def _artifact_paths(directory):
    return {
        "sweep": directory / "calibration_sweep.csv",
        "matrix": directory / "calibration_gate_matrix.csv",
        "selected": directory / "selected_operational_config.json",
        "receipt": directory / "selected_operational_config.sha256",
        "report": directory / "calibration_report.md",
        "baseline": directory / "full_frame_baseline.json",
    }


def _hash(path, pilot):
    return pilot._sha256(path)


def _read_sweep(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("operational_conf", "padding", "evaluation_conf"):
            row[key] = float(row[key]) if key != "padding" else int(row[key])
        for key in ("MAP_GATE_PASS", "COVERAGE_GATE_PASS", "FINAL_GATE_PASS"):
            row[key] = row[key].strip().lower() == "true"
        row["implementation_sha256"] = ast.literal_eval(row["implementation_sha256"])
        for key in (
            "mAP50",
            "mAP50_95",
            "Recall",
            "mAP50_retention",
            "gt_roi_coverage_mean",
            "candidate_segments_per_frame",
            "tensor_area_ratio_mean",
        ):
            row[key] = float(row[key])
    return rows


def validate_selected_config(model_name, evaluation_conf, config, pilot_module):
    directory = pilot_module.PILOT_DIR / "calibration" / model_name
    paths = _artifact_paths(directory)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{model_name} joint calibration is incomplete; missing {missing[0]}"
        )
    selected_sha = _hash(paths["selected"], pilot_module)
    if paths["receipt"].read_text(encoding="utf-8").strip() != selected_sha:
        raise ValueError(f"{model_name} selected operational config receipt does not match")
    selected = json.loads(paths["selected"].read_text(encoding="utf-8"))
    expected = {
        "model": model_name,
        "checkpoint_sha256": config["audit_protocol"]["checkpoint_sha256"],
        "pilot_manifest_sha256": _hash(
            pilot_module.PILOT_DIR / "pilot_manifest.txt", pilot_module
        ),
        "config_sha256": _hash(
            pilot_module.PILOT_DIR / "configs" / f"{model_name}.yaml", pilot_module
        ),
        "evaluation_conf": float(evaluation_conf),
    }
    if selected.get("status") not in {"CALIBRATION_PASS", "CALIBRATION_GATE_FAILED"} or any(
        selected.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(
            f"{model_name} frozen calibration provenance/status mismatch; rerun calibration "
            "in a fresh pilot directory"
        )
    for key, artifact_key in (
        ("calibration_sweep_sha256", "sweep"),
        ("calibration_gate_matrix_sha256", "matrix"),
        ("full_frame_baseline_sha256", "baseline"),
    ):
        if selected.get(key) != _hash(paths[artifact_key], pilot_module):
            raise ValueError(f"{model_name} calibration artifact changed: {artifact_key}")
    operational_confs, paddings = calibration_grid(config, model_name, evaluation_conf)
    sweep_rows = _read_sweep(paths["sweep"])
    expected_pairs = {
        (confidence, padding)
        for confidence in operational_confs
        for padding in paddings
    }
    observed_pairs = {
        (row["operational_conf"], row["padding"])
        for row in sweep_rows
    }
    if len(sweep_rows) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise ValueError(f"{model_name} calibration sweep does not match the current config grid")
    for row in sweep_rows:
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{model_name} calibration sweep provenance mismatch")

    recorded_implementations = sweep_rows[0]["implementation_sha256"]
    if not recorded_implementations or any(
        row["implementation_sha256"] != recorded_implementations for row in sweep_rows
    ):
        raise ValueError(f"{model_name} calibration implementation hashes are inconsistent")
    current_implementations = pilot_module._implementation_hashes()
    implementation_mismatch = any(
        recorded_implementations.get(key) != value
        and not (
            key == "experiments/cpu_pilot/calibration.py"
            and recorded_implementations.get(key)
            == LEGACY_CALIBRATION_IMPLEMENTATION_SHA256
        )
        for key, value in current_implementations.items()
    )
    if implementation_mismatch:
        raise ValueError(f"{model_name} calibration execution implementation changed")
    if (
        selected.get("implementation_sha256") is not None
        and selected["implementation_sha256"] != recorded_implementations
    ):
        raise ValueError(f"{model_name} selected implementation hashes do not match its sweep")

    selected_row = select_configuration(sweep_rows)
    expected_status = "CALIBRATION_PASS" if selected_row else "CALIBRATION_GATE_FAILED"
    if selected["status"] != expected_status:
        raise ValueError(f"{model_name} selected status does not match its calibration sweep")
    for key in ("operational_conf", "padding"):
        expected_value = selected_row[key] if selected_row else None
        if selected.get(key) != expected_value:
            raise ValueError(f"{model_name} selected {key} does not match its calibration sweep")
    for selected_key, row_key in (
        ("mAP50_seg", "mAP50"),
        ("mAP50_retention", "mAP50_retention"),
        ("gt_roi_coverage_mean", "gt_roi_coverage_mean"),
        ("candidate_segments_per_frame", "candidate_segments_per_frame"),
        ("tensor_area_ratio_mean", "tensor_area_ratio_mean"),
    ):
        expected_value = selected_row[row_key] if selected_row else None
        if selected.get(selected_key) != expected_value:
            raise ValueError(f"{model_name} selected metrics do not match its calibration sweep")
    return selected


def load_selected_for_models(model_names, evaluation_conf, configs, pilot_module):
    selected = {}
    for model_name in model_names:
        selected[model_name] = validate_selected_config(
            model_name, evaluation_conf, configs[model_name], pilot_module
        )
    return selected


def _build_gate_matrix(rows, operational_confs, paddings):
    lookup = {
        (float(row["operational_conf"]), int(row["padding"])): row
        for row in rows
    }
    return [
        {
            "operational_conf": confidence,
            **{
                f"padding_{padding}": (
                    "PASS" if lookup[(confidence, padding)]["FINAL_GATE_PASS"] else "FAIL"
                )
                for padding in paddings
            },
        }
        for confidence in operational_confs
    ]


def _nearest_configurations(rows):
    def distance(row):
        retention = row["mAP50_retention"]
        map_shortfall = MAP_RETENTION_GATE if retention is None else max(
            0.0, MAP_RETENTION_GATE - retention
        )
        coverage_shortfall = max(0.0, COVERAGE_GATE - row["gt_roi_coverage_mean"])
        return (
            int(not row["MAP_GATE_PASS"]) + int(not row["COVERAGE_GATE_PASS"]),
            map_shortfall + coverage_shortfall,
            -row["tensor_area_ratio_mean"],
        )
    return sorted(rows, key=distance)[:3]


def _write_report(path, model_name, selected, rows, baseline, operational_confs, paddings):
    lines = [
        f"# Joint calibration: {model_name}",
        "",
        "Protocol: one FF baseline at evaluation_conf=0.01; SEG-only confidence × padding grid; "
        "Static/previous localization; hold=0; periodic full-frame interval=10; merge disabled.",
        "",
        f"- Full-frame mAP50: `{baseline['mAP50']:.6f}`",
        f"- Full-frame mAP50:0.95: `{baseline['mAP50_95']:.6f}`",
        f"- Full-frame Recall@0.5: `{baseline['recall50']:.6f}`",
        f"- Gates: mAP50 retention >= {MAP_RETENTION_GATE:.2f}; "
        f"GT coverage mean >= {COVERAGE_GATE:.2f}.",
        f"- Selection rule: {SELECTION_RULE}.",
        "",
        "## Gate matrix",
        "",
        "| operational_conf | " + " | ".join(map(str, paddings)) + " |",
        "|---:|" + "|".join("---:" for _ in paddings) + "|",
    ]
    lookup = {(row["operational_conf"], row["padding"]): row for row in rows}
    for confidence in operational_confs:
        entries = [
            "PASS" if lookup[(confidence, padding)]["FINAL_GATE_PASS"] else "FAIL"
            for padding in paddings
        ]
        lines.append(f"| {confidence:.2f} | " + " | ".join(entries) + " |")
    lines.extend([
        "",
        "## Sweep metrics",
        "",
        "| Conf | Padding | mAP50 | mAP50:95 | Recall | Retention | Coverage | Candidate seg/frame | Tensor area ratio | Gates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['operational_conf']:.2f} | {row['padding']} | {row['mAP50']:.4f} | "
            f"{row['mAP50_95']:.4f} | {row['Recall']:.4f} | "
            f"{row['mAP50_retention'] if row['mAP50_retention'] is not None else 'undefined'} | "
            f"{row['gt_roi_coverage_mean']:.4f} | "
            f"{row['candidate_segments_per_frame']:.3f} | {row['tensor_area_ratio_mean']:.4f} | "
            f"{'PASS' if row['FINAL_GATE_PASS'] else 'FAIL'} |"
        )
    if selected:
        lines.extend([
            "",
            "## Selected configuration",
            "",
            f"**PASS:** operational_conf=`{selected['operational_conf']}`, "
            f"padding=`{selected['padding']}`. "
            f"Retention=`{selected['mAP50_retention']:.4f}`, "
            f"coverage=`{selected['gt_roi_coverage_mean']:.4f}`, "
            f"tensor area ratio=`{selected['tensor_area_ratio_mean']:.4f}`.",
            "",
            "Frozen artifact: `selected_operational_config.json`.",
        ])
    else:
        lines.extend([
            "",
            "## CALIBRATION_GATE_FAILED",
            "",
            "No configuration passed both predeclared gates. No operational confidence or padding "
            "was selected; cost and pipeline stages are blocked for this model.",
            "",
            "Nearest configurations (ordered by number and magnitude of gate misses; then lower "
            "tensor area) are diagnostic only and are not auto-selected:",
            "",
            "| Conf | Padding | Retention | Coverage | MAP gate | Coverage gate |",
            "|---:|---:|---:|---:|:---:|:---:|",
        ])
        for row in _nearest_configurations(rows):
            lines.append(
                f"| {row['operational_conf']:.2f} | {row['padding']} | "
                f"{row['mAP50_retention'] if row['mAP50_retention'] is not None else 'undefined'} | "
                f"{row['gt_roi_coverage_mean']:.4f} | "
                f"{row['MAP_GATE_PASS']} | {row['COVERAGE_GATE_PASS']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_joint_calibration(split, manifest_rows, device, pilot_module):
    pilot = pilot_module
    for model_name in pilot.MODEL_NAMES:
        config_path = pilot.PILOT_DIR / "configs" / f"{model_name}.yaml"
        config = _read_yaml(config_path)
        operational_confs, paddings = calibration_grid(
            config, model_name, pilot.DEFAULT_EVALUATION_CONF
        )
        directory = pilot.PILOT_DIR / "calibration" / model_name
        paths = _artifact_paths(directory)
        if paths["selected"].exists():
            selected = validate_selected_config(
                model_name, pilot.DEFAULT_EVALUATION_CONF, config, pilot
            )
            print(
                f"[calibration] Reusing provenance-validated {model_name} result: "
                f"{selected['status']}, conf={selected['operational_conf']}, "
                f"padding={selected['padding']}",
                flush=True,
            )
            continue
        _require_fresh_destination(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model, dataset, config, evaluation_conf = pilot._load_model(
            model_name, device, pilot.DEFAULT_EVALUATION_CONF
        )
        indices = pilot._dataset_indices(dataset, manifest_rows, pilot._dataset_root(config))
        if len(indices) != len(manifest_rows):
            raise RuntimeError(f"{model_name} manifest rows do not map one-to-one to dataset indices")
        model_info = pilot._metadata(
            model_name, model, dataset, config, evaluation_conf, operational_confs[0], split
        )
        checkpoint_sha256 = model_info["checkpoint_sha256"]

        baseline, _ = pilot._run_pipeline(
            model_name, model, dataset, config, evaluation_conf, operational_confs[0],
            manifest_rows, indices, "calibration_full_frame", "FF", 0,
        )
        _json(paths["baseline"], {
            "model": model_name,
            "checkpoint_sha256": checkpoint_sha256,
            "pilot_manifest_sha256": split["provenance"]["pilot_manifest_sha256"],
            "evaluation_conf": evaluation_conf,
            **{
                key: baseline[key]
                for key in ("mAP50", "mAP50_95", "recall50", "num_frames")
            },
        })
        sweep_rows = []
        for operational_conf in operational_confs:
            for padding in paddings:
                summary, frame_rows = pilot._run_pipeline(
                    model_name, model, dataset, config, evaluation_conf, operational_conf,
                    manifest_rows, indices,
                    f"calibration_conf{operational_conf:.4f}_padding{padding}",
                    "SEG", padding,
                )
                if len(frame_rows) != len(manifest_rows):
                    raise RuntimeError(
                        f"{model_name} calibration run produced {len(frame_rows)} frames, "
                        f"expected {len(manifest_rows)}"
                    )
                retention = (
                    summary["mAP50"] / baseline["mAP50"]
                    if baseline["mAP50"] > 0 else None
                )
                map_pass = retention is not None and retention >= MAP_RETENTION_GATE
                coverage_pass = summary["gt_roi_coverage_mean"] >= COVERAGE_GATE
                record = {
                    "model": model_name,
                    "checkpoint_sha256": checkpoint_sha256,
                    "pilot_manifest_sha256": split["provenance"]["pilot_manifest_sha256"],
                    "config_sha256": model_info["config_sha256"],
                    "implementation_sha256": pilot._implementation_hashes(),
                    "evaluation_conf": evaluation_conf,
                    "operational_conf": operational_conf,
                    "padding": padding,
                    "mAP50": summary["mAP50"],
                    "mAP50_95": summary["mAP50_95"],
                    "Recall": summary["recall50"],
                    "mAP50_retention": retention,
                    "gt_roi_coverage_mean": summary["gt_roi_coverage_mean"],
                    "gt_roi_coverage_p5": summary["gt_roi_coverage_p5"],
                    "evaluation_detections_per_frame": sum(
                        int(row["evaluation_detection_count"]) for row in frame_rows
                    ) / len(frame_rows),
                    "operational_detections_per_frame": sum(
                        int(row["operational_detection_count"]) for row in frame_rows
                    ) / len(frame_rows),
                    "candidate_segments_per_frame": summary["candidate_segments_mean"],
                    "nn_calls_per_frame": summary["nn_calls_per_frame"],
                    "processed_area_ratio_mean": summary["processed_area_ratio_mean"],
                    "tensor_area_ratio_mean": summary["tensor_area_ratio_mean"],
                    "full_frame_fraction": summary["full_frame_fraction"],
                    "MAP_GATE_PASS": map_pass,
                    "COVERAGE_GATE_PASS": coverage_pass,
                    "FINAL_GATE_PASS": map_pass and coverage_pass,
                }
                sweep_rows.append(record)
                print(
                    f"[calibration/{model_name}] conf={operational_conf:.2f} padding={padding} "
                    f"retention={retention if retention is not None else 'undefined'} "
                    f"coverage={summary['gt_roi_coverage_mean']:.4f}",
                    flush=True,
                )
        matrix_rows = _build_gate_matrix(sweep_rows, operational_confs, paddings)
        _csv(paths["sweep"], sweep_rows)
        _csv(paths["matrix"], matrix_rows)
        selected_row = select_configuration(sweep_rows)
        selected_document = {
            "status": "CALIBRATION_PASS" if selected_row else "CALIBRATION_GATE_FAILED",
            "model": model_name,
            "checkpoint_sha256": checkpoint_sha256,
            "pilot_manifest_sha256": split["provenance"]["pilot_manifest_sha256"],
            "config_sha256": model_info["config_sha256"],
            "implementation_sha256": pilot._implementation_hashes(),
            "evaluation_conf": evaluation_conf,
            "operational_conf": selected_row["operational_conf"] if selected_row else None,
            "padding": selected_row["padding"] if selected_row else None,
            "mAP50_ff": baseline["mAP50"],
            "mAP50_seg": selected_row["mAP50"] if selected_row else None,
            "mAP50_retention": selected_row["mAP50_retention"] if selected_row else None,
            "gt_roi_coverage_mean": selected_row["gt_roi_coverage_mean"] if selected_row else None,
            "candidate_segments_per_frame": (
                selected_row["candidate_segments_per_frame"] if selected_row else None
            ),
            "tensor_area_ratio_mean": selected_row["tensor_area_ratio_mean"] if selected_row else None,
            "selection_rule": SELECTION_RULE,
            "area_ratio_tie_tolerance": AREA_RATIO_TIE_TOLERANCE,
            "operational_conf_values": operational_confs,
            "padding_values": paddings,
            "quality_gates": {
                "mAP50_retention_min": MAP_RETENTION_GATE,
                "gt_roi_coverage_mean_min": COVERAGE_GATE,
            },
            "calibration_sweep_sha256": _hash(paths["sweep"], pilot),
            "calibration_gate_matrix_sha256": _hash(paths["matrix"], pilot),
            "full_frame_baseline_sha256": _hash(paths["baseline"], pilot),
        }
        _json(paths["selected"], selected_document)
        paths["receipt"].write_text(_hash(paths["selected"], pilot) + "\n", encoding="utf-8")
        _write_report(
            paths["report"], model_name, selected_row, sweep_rows, baseline,
            operational_confs, paddings,
        )
        del model, dataset
        if not selected_row:
            print(f"[calibration/{model_name}] CALIBRATION_GATE_FAILED", flush=True)

    configurations = {
        model_name: validate_selected_config(
            model_name,
            pilot.DEFAULT_EVALUATION_CONF,
            _read_yaml(pilot.PILOT_DIR / "configs" / f"{model_name}.yaml"),
            pilot,
        )
        for model_name in pilot.MODEL_NAMES
        if (pilot.PILOT_DIR / "calibration" / model_name / "selected_operational_config.json").is_file()
    }
    return configurations
