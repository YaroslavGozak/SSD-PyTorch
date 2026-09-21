"""Analyze paired Experiment B timings and per-model decision quality."""

import argparse
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

from .common import file_sha256, write_json
from .models import control_predictions, load_latency_models
from .reproducibility import order_estimate, gate, canonical_hash
from .artifacts import load_calibration, read_reference, reference
from .validation_design import practical_label, control_diagnostics, repetitions_for_pair

# (rule_name, predicted_merge column, gain column, merged-cost column, separate-cost column)
RULES = (
    ("shape_lookup", "lookup_predicted_merge", "lookup_gain_ms", "lookup_merged_ms", "lookup_separate_ms"),
    ("shape_lookup_conservative", "conservative_predicted_merge", "conservative_gain_ms", "conservative_merged_ms", "conservative_separate_ms"),
    ("conservative_consensus", "consensus_predicted_merge", "consensus_gain_ms", "consensus_merged_ms", "consensus_separate_ms"),
    ("linear_tau", "linear_tau_predicted_merge", "linear_predicted_gain_ms", "linear_predicted_merged_cost_ms", "linear_predicted_separate_cost_ms"),
    ("quadratic_direct_cost", "quadratic_direct_predicted_merge", "quadratic_predicted_gain_ms", "quadratic_predicted_merged_cost_ms", "quadratic_predicted_separate_cost_ms"),
    ("piecewise_direct_cost", "piecewise_direct_predicted_merge", "piecewise_predicted_gain_ms", "piecewise_predicted_merged_cost_ms", "piecewise_predicted_separate_cost_ms"),
)


def _mean_ci(values, seed, iterations=2000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [float(np.mean(values)), float(np.mean(values))]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def _parse_optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_bool(value):
    text = str(value).strip().lower()
    return text == "true" if text in {"true", "false"} else None


def _classification_metrics(rows, rule_name):
    determinate = [row for row in rows if row["label"] != "ambiguous" and row["predictions"][rule_name]["predicted_merge"] is not None]
    actual_classes = {row["label"] for row in determinate}
    has_both_classes = actual_classes == {"merge_beneficial", "separate_beneficial"}
    actual = [row["label"] == "merge_beneficial" for row in determinate]
    predicted = [row["predictions"][rule_name]["predicted_merge"] for row in determinate]
    tp = sum(p and a for p, a in zip(predicted, actual)); fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(not p and a for p, a in zip(predicted, actual)); tn = sum(not p and not a for p, a in zip(predicted, actual))
    sensitivity = tp / max(tp + fn, 1); specificity = tn / max(tn + fp, 1); precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(len(determinate), 1) if determinate else None
    return {"determinate_pairs": len(determinate), "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "accuracy": accuracy, "balanced_accuracy": (sensitivity + specificity) / 2 if has_both_classes else None,
            "precision_merge": precision if tp + fp else None, "recall_merge": sensitivity if tp + fn else None,
            "f1_merge": 2 * precision * sensitivity / max(precision + sensitivity, np.finfo(float).eps) if tp + fp and tp + fn else None,
            "specificity": specificity if has_both_classes else None,
            "actual_classes": sorted(actual_classes), "classification_valid": has_both_classes,
            "accuracy_bootstrap_ci": _mean_ci([int(p == a) for p, a in zip(predicted, actual)], 17) if determinate else None}


def _regret(rows, rule_name):
    values = []
    for row in rows:
        prediction = row["predictions"][rule_name]["predicted_merge"]
        if prediction is None:
            continue
        chosen = row["measurements"]["merged_mean_ms"] if prediction else row["measurements"]["separate_mean_ms"]
        oracle = min(row["measurements"]["merged_mean_ms"], row["measurements"]["separate_mean_ms"])
        values.append(max(0.0, chosen - oracle))
    if not values:
        return {"mean": None, "median": None, "p95": None, "p99": None, "max": None, "n": 0}
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)), "n": len(values)}


def _rule_report(rows, rule_name):
    metrics = _classification_metrics(rows, rule_name)
    # Regret is reported both over every pair with a prediction and restricted to
    # determinate pairs, since ambiguous ground truth should not be blamed on the model.
    metrics["regret_ms_all_pairs"] = _regret(rows, rule_name)
    metrics["regret_ms_determinate_pairs"] = _regret([row for row in rows if row["label"] != "ambiguous"], rule_name)
    available = [r for r in rows if r["predictions"][rule_name]["predicted_merge"] is not None]
    metrics["expected_latency_gain_ms"] = float(np.mean([r["measurements"]["mean_difference_ms"] if r["predictions"][rule_name]["predicted_merge"] else 0 for r in available])) if available else None
    metrics["exploratory"] = rule_name == "conservative_consensus"
    return metrics


def _group_reports(rows, group_key):
    groups = sorted({row[group_key] for row in rows if row[group_key] is not None})
    report = {}
    for group in groups:
        scoped = [row for row in rows if row[group_key] == group]
        report[group] = {"pairs": len(scoped)}
        for rule_name, *_ in RULES:
            report[group][rule_name] = _rule_report(scoped, rule_name)
    return report


def _area_regime(a1, a2, au, breakpoint_area):
    if breakpoint_area is None:
        return None
    areas = (a1, a2, au)
    if all(area <= breakpoint_area for area in areas):
        return "below_breakpoint"
    if all(area > breakpoint_area for area in areas):
        return "above_breakpoint"
    return "spans_breakpoint"


def _load_calibration(input_path: str, calibration_path: str | None, metadata):
    """Prefer the recorded model snapshot; support portable and legacy run folders."""
    if calibration_path:
        path = Path(calibration_path)
        if metadata.get("schema_version",0)>=4 and file_sha256(path) != metadata["calibration_reference"]["content_hash"]:
            raise ValueError("Explicit calibration hash mismatch for frozen validation")
        return load_calibration(path), {"source": "explicit_fit", "path": str(path.resolve()), "content_hash": file_sha256(path)}
    snapshot = metadata.get("latency_models")
    if snapshot:
        envelope = metadata.get("calibration_envelope") or next(
            (model["calibration_envelope"] for model in snapshot.values() if model.get("calibration_envelope")), None)
        return {"latency_models": snapshot, "calibration_envelope": envelope}, {
            **metadata.get("calibration_reference", {}), "source": "experiment_b_metadata.latency_models"}
    directory = Path(input_path).resolve().parent
    reference = metadata.get("calibration_reference", {})
    reference_path = reference.get("path") or reference.get("path_or_id")
    candidates = []
    if reference_path:
        # Recorded legacy paths may be relative to the original working directory.
        portable = Path(reference_path.replace("\\", "/"))
        candidates.extend([directory / portable, portable, directory / portable.name])
    candidates.append(directory / "linear_fit.json")
    for path in candidates:
        if path.is_file():
            digest = file_sha256(path)
            if reference.get("content_hash") and digest != reference["content_hash"]:
                raise ValueError(f"Calibration hash mismatch: {path}; pass --fit explicitly to select another artifact")
            return load_calibration(path), {"source": "discovered_fit", "path": str(path.resolve()), "content_hash": digest}
    return {}, {"source": None}


def analyze(path: str, output: str | None = None, calibration_path: str | None = None):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Experiment B CSV is empty")

    metadata_file = Path(path).with_name("experiment_b_metadata.json")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
    if metadata.get("pair_specs_reference"):
        document = read_reference(metadata_file,metadata["pair_specs_reference"])
        metadata["pair_specs"] = document["payload"]
        if document["sha256"] != metadata["pair_specs_hash"]:
            raise ValueError("Pair specs hash mismatch")
    candidate_predictions = read_reference(metadata_file,metadata["candidate_predictions_reference"]) if metadata.get("candidate_predictions_reference") else {}
    if metadata.get("raw_observations_reference") and file_sha256(path) != metadata["raw_observations_reference"]["sha256"]:
        raise ValueError("Raw observations hash mismatch")
    calibration, calibration_reference = _load_calibration(path, calibration_path, metadata)
    breakpoint_area = calibration.get("latency_models", {}).get("piecewise", {}).get("breakpoint_area")
    if breakpoint_area is not None:
        breakpoint_area = float(breakpoint_area)
        if not np.isfinite(breakpoint_area) or breakpoint_area <= 0:
            raise ValueError("Piecewise breakpoint_area must be positive and finite")
    control_records = []
    controls = metadata.get("control_records", [])
    if controls and calibration.get("latency_models") and calibration.get("calibration_envelope"):
        models = load_latency_models(calibration)
        for record in controls:
            area = int(record["tensor_h"]) * int(record["tensor_w"])
            control_records.append({**record, "effective_area": area,
                                    "predictions": control_predictions(models, area, float(record["measured_ms"]))})
    else:
        control_records = controls

    if metadata.get("pair_specs") and canonical_hash(metadata["pair_specs"]) != metadata.get("pair_specs_hash"):
        raise ValueError("Recorded pair specs hash mismatch")
    if len({r.get("session_id", "legacy") for r in rows}) != 1 or len({r["timing_mode"] for r in rows}) != 1:
        raise ValueError("Mixed sessions or timing modes in Experiment B")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(row)

    options = metadata.get("config", {}).get("experiment_b", {})
    bootstrap_count = int(options.get("bootstrap_count", 2000))
    bootstrap_seed = int(options.get("bootstrap_seed", 0))
    confidence = float(options.get("confidence_level", .95))
    warnings = list(metadata.get("quality_warnings", []))
    if metadata.get("schema_version",0)<4:
        gate(not metadata.get("hardware_state_shift",False), "control_drift", metadata.get("config", {}), warnings)
    if len({r.get("evaluation_design","legacy") for r in rows})>1:
        raise ValueError("Representative and challenge measurements cannot be mixed")
    summaries = []
    for pair_id, pair_rows in grouped.items():
        first = pair_rows[0]
        if len({r["repetition"] for r in pair_rows}) != len(pair_rows):
            raise ValueError(f"Duplicate repetition in pair {pair_id}")
        if metadata.get("schema_version", 0) >= 3:
            gate(len(pair_rows) == repetitions_for_pair(metadata.get("config",{}),{"primary_stratum":first.get("primary_stratum") or first["boundary_bin"]}), "repetition_count", metadata.get("config",{}), warnings)
        differences = [float(row["difference_ms"]) for row in pair_rows]
        estimate = order_estimate(pair_rows, seed=bootstrap_seed+int(canonical_hash(pair_id)[:8],16), count=bootstrap_count, confidence=confidence)
        gate(estimate["balanced"], "order_balance", metadata.get("config", {}), warnings)
        ci = estimate["ci"]
        label = "merge_beneficial" if ci[0] > 0 else ("separate_beneficial" if ci[1] < 0 else "ambiguous")

        predictions = {
            rule_name: {
                "predicted_merge": _parse_optional_bool(first.get(merge_col)),
                "predicted_gain_ms": _parse_optional_float(first.get(gain_col)),
                "predicted_merged_cost_ms": _parse_optional_float(first.get(merged_cost_col)),
                "predicted_separate_cost_ms": _parse_optional_float(first.get(separate_cost_col)),
            }
            for rule_name, merge_col, gain_col, merged_cost_col, separate_cost_col in RULES
        }

        for name,decision in candidate_predictions.get(pair_id,{}).items():
            predictions[name] = dict(predicted_merge=decision["predicted_merge"],predicted_gain_ms=decision.get("predicted_gain_s")*1000 if decision.get("predicted_gain_s") is not None else None,
                                     predicted_merged_cost_ms=decision.get("predicted_merged_cost_s",0)*1000,
                                     predicted_separate_cost_ms=decision.get("predicted_separate_cost_s",0)*1000,
                                     gain_lcb_ms=decision.get("gain_lcb_s")*1000 if decision.get("gain_lcb_s") is not None else None,
                                     exploratory=name=="conservative_consensus")
        raw_observations = [
            {"repetition": int(row["repetition"]), "order": row["order"],
             "separate_ms": float(row["separate_ms"]), "merged_ms": float(row["merged_ms"]),
             "difference_ms": float(row["difference_ms"]), "order_block_id": row.get("order_block_id"), "position_in_order_block": row.get("position_in_order_block")}
            for row in pair_rows
        ]
        a1 = float(first["a1_effective"]); a2 = float(first["a2_effective"]); au = float(first["au_effective"])

        summaries.append({
            "pair_id": pair_id,
            "predicted_merge": predictions["linear_tau"]["predicted_merge"],  # backward-compatible alias
            "predictions": predictions,
            "label": label,
            "practical_label":practical_label(ci,float(options.get("minimum_worthwhile_gain_ms",0))),
            "effective_areas": {"a1": a1, "a2": a2, "merged": au},
            "delta_area": float(first["delta_effective_area"]),
            "tau_used": _parse_optional_float(first.get("tau_used")),
            "measurements": {
                "merged_mean_ms": float(np.mean([np.mean([obs["merged_ms"] for obs in raw_observations if obs["order"]==order]) for order in ("merged_first","separate_first")])),
                "separate_mean_ms": float(np.mean([np.mean([obs["separate_ms"] for obs in raw_observations if obs["order"]==order]) for order in ("merged_first","separate_first")])),
                "difference_by_order_ms": {order:float(np.mean([obs["difference_ms"] for obs in raw_observations if obs["order"]==order])) for order in ("merged_first","separate_first")},
                "mean_difference_ms": estimate["estimate"],
                "order_counts": estimate["order_counts"],
                "difference_ci_ms": ci,
                "n_paired_samples": len(raw_observations),
            },
            "raw_observations": raw_observations,
            "bootstrap_stability": next((p.get("bootstrap_stability", {}) for p in metadata.get("pair_specs", {}).get("pairs", []) if str(p["pair_id"]) == pair_id), {}),
            "primary_stratum": first.get("primary_stratum") or first["boundary_bin"],
            "boundary_bin": first["boundary_bin"],
            "boundary_side": first.get("boundary_side") or None,
            "geometry_type": first["geometry_type"],
            "area_regime": _area_regime(a1, a2, au, breakpoint_area),
            "computational_key": tuple(first[key] for key in ("r1_tensor_h", "r1_tensor_w", "r2_tensor_h", "r2_tensor_w", "union_tensor_h", "union_tensor_w")),
        })

    if metadata.get("pair_specs"):
        expected = {str(p["pair_id"]) for p in metadata["pair_specs"]["pairs"]}
        gate(set(grouped) == expected, "missing_pairs", metadata.get("config",{}), warnings)
    order_means = {
        order: float(np.mean([float(row["difference_ms"]) for row in rows if row["order"] == order]))
        for order in ("separate_first", "merged_first") if any(row["order"] == order for row in rows)
    }
    order_effect = order_means.get("separate_first", float("nan")) - order_means.get("merged_first", float("nan"))

    model_comparison = {rule_name: _rule_report(summaries, rule_name) for rule_name, *_ in RULES}
    metrics_by_boundary_bin = _group_reports(summaries, "boundary_bin")
    metrics_by_boundary_side = _group_reports(summaries, "boundary_side")
    metrics_by_stratum_and_boundary_side = {
        stratum:_group_reports([row for row in summaries if row["primary_stratum"] == stratum], "boundary_side")
        for stratum in sorted({row["primary_stratum"] for row in summaries if row["boundary_side"] is not None})
    }
    metrics_by_geometry_type = _group_reports(summaries, "geometry_type")
    metrics_by_area_regime = _group_reports(summaries, "area_regime") if breakpoint_area is not None else None

    cpu_temp_available = any(row.get("cpu_temp_c") not in (None, "") for row in rows)
    cpu_freq_available = any(row.get("cpu_freq_mhz") not in (None, "") for row in rows)

    result = {
        "schema_version": 4,
        "pair_specs_hash": metadata.get("pair_specs_hash"),
        "quality_warnings": warnings,
        "bootstrap_method": dict(method="stratified_by_order", count=bootstrap_count, seed=bootstrap_seed, confidence_level=confidence),
        "metrics_by_stratum": _group_reports(summaries,"primary_stratum"),
        "pairs": len(summaries),
        "ambiguous_fraction": sum(row["label"] == "ambiguous" for row in summaries) / max(len(summaries), 1),
        "sampling": {"sampling_basis": metadata.get("sampling_basis", "legacy_linear_tau"), "sampling_tau_pixels": summaries[0]["tau_used"] if summaries else None},
        "measurement_protocol": {
            "timing_mode": rows[0].get("timing_mode"),
            "order_design": "balanced_abba_baab",
            "cpu_temp_sensor_available": cpu_temp_available,
            "cpu_freq_sensor_available": cpu_freq_available,
        },
        "area_regime_available": breakpoint_area is not None,
        "area_regime_unavailable_reason": None if breakpoint_area is not None else "No piecewise breakpoint in the resolved calibration; provide --fit",
        "piecewise_breakpoint_area": breakpoint_area,
        "calibration_reference": calibration_reference,
        "control_records": control_records,
        "model_comparison": model_comparison,
        "order_means_ms": order_means,
        "order_effect_ms": order_effect,
        "order_effect_warning": bool(np.isfinite(order_effect) and abs(order_effect) > float(options.get("order_effect_warning_ms", .5))),
        "metrics_by_boundary_bin": metrics_by_boundary_bin,
        "metrics_by_boundary_side": metrics_by_boundary_side,
        "metrics_by_stratum_and_boundary_side": metrics_by_stratum_and_boundary_side,
        "metrics_by_geometry_type": metrics_by_geometry_type,
        "metrics_by_area_regime": metrics_by_area_regime,
        "unique_computational_configurations": len({row["computational_key"] for row in summaries}),
        "pair_summaries": summaries,
    }
    evaluation_design = metadata.get("evaluation_design", "challenge")
    result["evaluation_design"] = evaluation_design
    result["aggregate_scope"] = "reference_distribution" if evaluation_design=="representative" else "unweighted_challenge_average"
    result["policy_declaration"] = metadata.get("policy_declaration")
    result["primary_policy"] = metadata.get("policy_declaration",{}).get("primary","linear_tau")
    result["primary_metrics"] = model_comparison.get(result["primary_policy"])
    result["confirmatory"] = metadata.get("confirmatory",False)
    result["fallback_counts"] = metadata.get("fallback_counts",{})
    result["reference_distribution"] = metadata.get("pair_specs",{}).get("reference_distribution")
    result["control_diagnostics"] = control_diagnostics(calibration,controls,options) if calibration.get("calibration_envelope") else {}
    for flag,gate_name in (("hardware_state_shift","calibration_to_validation_shift"),("within_session_drift_detected","within_session_drift")):
        if metadata.get("schema_version",0)>=4:
            gate(not result["control_diagnostics"].get(flag,False),gate_name,metadata.get("config",{}),warnings)
    result["hardware_state_shift"] = result["control_diagnostics"].get("hardware_state_shift",metadata.get("hardware_state_shift",False))
    result["quality_warnings"] = list(dict.fromkeys(warnings))
    result["sampling_diagnostics"] = sampling_diagnostics(summaries,metadata.get("pair_specs",{}).get("sampling_diagnostics",{}))
    result["agreement_matrix"] = agreement_matrix(summaries)
    result["artifact_references"] = {key:metadata.get(key) for key in ("pair_specs_reference","candidate_predictions_reference","calibration_reference","raw_observations_reference")}
    result["weighted_reference_metrics"] = weighted_metrics(summaries,options.get("aggregation_weights"))
    if metadata.get("schema_version",0)>=4:
        for pair in summaries:
            pair.pop("raw_observations",None)
    if output:
        write_json(Path(output) / "decision_metrics.json", result)
    return result


def sampling_diagnostics(rows, generation):
    report = {}
    for name in sorted({r["primary_stratum"] for r in rows}):
        scoped = [r for r in rows if r["primary_stratum"]==name]
        report[name] = {**generation.get("by_stratum",{}).get(name,{}),
                        "generated_pairs":len(scoped),"determinate_pairs":sum(r["label"]!="ambiguous" for r in scoped),
                        "ambiguous_pairs":sum(r["label"]=="ambiguous" for r in scoped),
                        "actual_class_counts":dict(Counter(r["label"] for r in scoped)),
                        "prediction_counts_by_model":{rule:dict(Counter(str(r["predictions"][rule]["predicted_merge"]) for r in scoped)) for rule,*_ in RULES}}
    return dict(strata=report,generation_attempts=generation.get("generation_attempts"),rejection_reasons=generation.get("rejection_reasons",{}))


def agreement_matrix(rows):
    result = {}
    for left,*_ in RULES:
        result[left] = {}
        for right,*_ in RULES:
            common = [r for r in rows if r["predictions"][left]["predicted_merge"] is not None and r["predictions"][right]["predicted_merge"] is not None]
            result[left][right] = dict(pairs=len(common),agreement_fraction=float(np.mean([r["predictions"][left]["predicted_merge"]==r["predictions"][right]["predicted_merge"] for r in common])) if common else None)
    return result


def weighted_metrics(rows, declaration):
    if declaration is None:
        return None
    if not declaration.get("source") or declaration.get("derived_from_labels",False):
        raise ValueError("Weights require an external/reference source, never observed labels")
    weights = declaration["strata"]
    counts = Counter(r["primary_stratum"] for r in rows)
    if set(weights)!=set(counts) or any(not np.isfinite(v) or v<0 for v in weights.values()) or not np.isclose(sum(weights.values()),1):
        raise ValueError("External weights must cover strata and sum to one")
    report = {}
    for rule,*_ in RULES:
        if any(r["predictions"][rule]["predicted_merge"] is None for r in rows):
            report[rule] = None
            continue
        regret,gain,pair_weights = [],[],[]
        for r in rows:
            merge = r["predictions"][rule]["predicted_merge"]
            measured = r["measurements"]
            chosen = measured["merged_mean_ms"] if merge else measured["separate_mean_ms"]
            regret.append(chosen-min(measured["merged_mean_ms"],measured["separate_mean_ms"]))
            gain.append(measured["mean_difference_ms"] if merge else 0)
            pair_weights.append(weights[r["primary_stratum"]]/counts[r["primary_stratum"]])
        order = np.argsort(regret)
        cumulative = np.cumsum(np.asarray(pair_weights)[order])
        p95 = np.asarray(regret)[order][min(np.searchsorted(cumulative,.95),len(order)-1)]
        report[rule] = dict(mean_regret_ms=float(np.dot(regret,pair_weights)),p95_regret_ms=float(p95),expected_latency_gain_ms=float(np.dot(gain,pair_weights)))
    return dict(source=declaration["source"],stratum_weights=weights,metrics=report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--fit", help="Calibration artifact used to classify area regime relative to the piecewise breakpoint; "
                                      "defaults to the model snapshot in B metadata, its referenced fit, or adjacent linear_fit.json")
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.fit), indent=2))


if __name__ == "__main__":
    main()
