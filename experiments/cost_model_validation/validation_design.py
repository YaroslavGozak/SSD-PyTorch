"""Pre-measurement declarations, reuse guards, and observed control diagnostics."""
import json
from pathlib import Path

import numpy as np

from .common import write_json
from .reproducibility import canonical_hash
from .shape_model import policy_declaration
from .models import control_predictions, load_latency_models


def validation_declaration(config):
    experiment = config.get("experiment_b",{})
    design = experiment.get("evaluation_design","challenge")
    if design not in {"representative","challenge"}:
        raise ValueError("evaluation_design must be representative or challenge")
    return dict(policies=policy_declaration(config),evaluation_design=design,
                experiment_b=experiment,stopping_rule="fixed_predeclared_repetitions",
                minimum_worthwhile_gain_ms=float(experiment.get("minimum_worthwhile_gain_ms",0)))


def repetitions_for_pair(config, specification):
    options = config.get("experiment_b",{})
    count = int(options.get("repetitions",20))
    if options.get("evaluation_design","challenge") == "challenge" and specification["primary_stratum"] != "broad_random":
        count = int(options.get("challenge_repetitions",count))
    if count < 2 or count % 2:
        raise ValueError("Paired repetitions must be positive and even")
    return count


def check_freshness(path, document, config, override=False):
    warnings = []
    if not config.get("experiment_b",{}).get("confirmatory",False):
        return warnings
    payload = document["payload"]
    issues = []
    if payload.get("used_for_policy_development",False):
        issues.append("Pair file marked as used for policy development")
    if payload.get("validation_declaration") != validation_declaration(config):
        issues.append("Policy/design declaration differs from the frozen pair specification")
    history = Path(path).with_suffix(".usage.json")
    if history.exists():
        record = json.loads(history.read_text(encoding="utf-8"))
        if record.get("pair_specs_hash") == document["sha256"]:
            issues.append("Pair file already used for validation (including incomplete measurement)")
    if payload.get("schema_version",0) < 2:
        issues.append("Legacy pair file has no predeclared independent-validation design")
    if issues and not override:
        raise ValueError("; ".join(issues)+". Use --allow-reused-pairs only for explicitly non-confirmatory analysis.")
    return ["NON-CONFIRMATORY OVERRIDE: "+issue for issue in issues]


def mark_used(path, digest, output):
    write_json(Path(path).with_suffix(".usage.json"),dict(pair_specs_hash=digest,measurement_output=str(Path(output).resolve())))


def practical_label(ci, delta):
    if delta < 0:
        raise ValueError("minimum_worthwhile_gain_ms must be nonnegative")
    if ci[0]>delta:
        return "confident_merge"
    if ci[1]<-delta:
        return "confident_separate"
    if ci[0]>=-delta and ci[1]<=delta:
        return "practically_tied"
    return "uncertain"


def control_diagnostics(calibration, records, options):
    models = load_latency_models(calibration)
    baseline = calibration.get("control_records",[])
    within,between = {},{}
    for shape in sorted({(r["tensor_h"],r["tensor_w"]) for r in records}):
        key = f"{shape[0]}x{shape[1]}"
        scoped = sorted([r for r in records if (r["tensor_h"],r["tensor_w"])==shape],key=lambda r:r.get("elapsed_s",0))
        values = np.array([r["measured_ms"] for r in scoped])
        thirds = np.array_split(values,3)
        means = [float(v.mean()) if len(v) else None for v in thirds]
        drift = (means[2]-means[0])/means[0] if means[2] is not None and means[0] else None
        within[key] = dict(early_mean_ms=means[0],middle_mean_ms=means[1],late_mean_ms=means[2],
                           early_to_late_relative_drift=drift,drift_detected=drift is not None and abs(drift)>float(options.get("within_session_drift_threshold",.1)))
        observed = [r["measured_ms"] for r in baseline if (r["tensor_h"],r["tensor_w"])==shape]
        mean,median = float(values.mean()),float(np.median(values))
        base_mean = float(np.mean(observed)) if observed else None
        shift = mean-base_mean if base_mean is not None else None
        relative = shift/base_mean if base_mean else None
        predictions = control_predictions(models,shape[0]*shape[1],mean)
        lookup = calibration.get("latency_models",{}).get("shape_lookup",{}).get("table",{}).get(key)
        if lookup:
            predicted = lookup["point_s"]*1000
            predictions["shape_lookup"] = dict(available=True,predicted_ms=predicted,relative_error=(mean-predicted)/predicted)
        between[key] = dict(calibration_baseline_distribution_ms=observed,calibration_baseline_mean_ms=base_mean,
                            validation_mean_ms=mean,validation_median_ms=median,absolute_shift_ms=shift,relative_shift=relative,
                            baseline_available=bool(observed),shift_detected=relative is not None and abs(relative)>float(options.get("calibration_shift_threshold",.1)),
                            models={name:{**prediction,"prediction_residual_ms":mean-prediction["predicted_ms"] if prediction.get("available") else None} for name,prediction in predictions.items()})
    return dict(within_session_drift=within,calibration_to_validation_shift=between,
                hardware_state_shift=any(v["shift_detected"] for v in between.values()),
                within_session_drift_detected=any(v["drift_detected"] for v in within.values()))
