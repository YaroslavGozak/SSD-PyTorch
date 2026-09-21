"""Refit calibration on equally weighted session-level shape statistics."""
import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import file_sha256, write_json, system_metadata
from .models import load_latency_models
from .artifacts import load_calibration, export_calibration
from .shape_model import build_lookup
from .reproducibility import design, statistics, fit_models, bootstrap_models


def aggregate(sessions, output, count=2000, seed=0, pairs=None, method="mean"):
    if len(sessions) < 2:
        raise ValueError("At least two independent sessions required (3–5 recommended)")
    artifacts = [load_calibration(p) for p in sessions]
    if any(a.get("schema_version",0) < 3 or not a.get("provenance",{}).get("complete") for a in artifacts):
        raise ValueError("Aggregation requires schema-v3 artifacts with complete provenance")
    first = artifacts[0]
    keys = ("weights_sha256", "backend", "device", "preprocessing", "model_stride", "timing_mode", "model_config_sha256", "dtype", "shape_policy", "runtime_environment")
    identity = {k:first["provenance"].get(k) for k in keys}
    selector = first["primary_fit_selector"]
    shape_set = {(r["tensor_h"],r["tensor_w"]) for r in first["shape_statistics"]}
    ids = [a["provenance"]["session_id"] for a in artifacts]
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("Sessions must have distinct recorded session IDs")
    grouped = defaultdict(list)
    for artifact in artifacts:
        if ({k:artifact["provenance"].get(k) for k in keys} != identity or
            artifact["calibration_envelope"] != first["calibration_envelope"] or
            artifact["primary_fit_selector"] != selector or
            {(r["tensor_h"],r["tensor_w"]) for r in artifact["shape_statistics"]} != shape_set):
            raise ValueError("Incompatible calibration sessions")
        for row in artifact["shape_statistics"]:
            grouped[(row["tensor_h"],row["tensor_w"])].append(row[selector["statistic"]+"_s"])
    if method not in {"mean", "median"}:
        raise ValueError("Pooling method must be mean or median")
    support = int(first.get("config", {}).get("experiment_a", {}).get("breakpoint_min_support", 2))
    pooled = fit_models(*design(grouped,method,selector["level"],0),support)
    if not all(m["valid"] for m in pooled.values()):
        raise ValueError("Invalid pooled model; inspect session calibration quality")
    raw_sessions = []
    for source,artifact in zip(sessions,artifacts):
        raw_path = Path(source).with_name("experiment_a_raw.csv")
        if not raw_path.exists():
            raw_path = Path(artifact["raw_observations_reference"])
        if file_sha256(str(raw_path)) != artifact.get("raw_observations_sha256"):
            raise ValueError("Session raw observations hash mismatch")
        raw = defaultdict(list)
        with raw_path.open(newline="",encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                field = "inference_ms" if row["timing_mode"] == "inference_only" else "detector_call_ms"
                raw[(int(row["tensor_h"]),int(row["tensor_w"]))].append(float(row[field])/1000)
        if set(raw) != shape_set:
            raise ValueError("Session raw shape set mismatch")
        raw_sessions.append(raw)

    def sample_sessions(rng):
        selected = rng.choice(len(raw_sessions),len(raw_sessions),replace=True)
        sampled = defaultdict(list)
        for index in selected:
            for key,values in sorted(raw_sessions[index].items()):
                sample = rng.choice(values,len(values),replace=True)
                sampled[key].append(statistics(sample,selector["trim_fraction_each_tail"])[selector["statistic"]+"_s"])
        return sampled

    uncertainty = bootstrap_models(grouped,count,seed,method,selector["level"],0,support,sampler=sample_sessions)
    uncertainty["method"] = "hierarchical_sessions_then_within_shape"
    stability = {}
    for name in pooled:
        coefficients = {}
        for key in pooled[name]["coefficients"]:
            values = np.array([a["latency_models"][name]["coefficients"][key] for a in artifacts])
            coefficients[key] = dict(mean=float(values.mean()),median=float(np.median(values)),std=float(values.std(ddof=1)),
                                     cv=float(values.std(ddof=1)/abs(values.mean())) if values.mean() else None)
        stability[name] = dict(coefficients=coefficients, breakpoint_frequency=dict(Counter(str(a["latency_models"][name].get("breakpoint_area")) for a in artifacts)))
    models = [load_latency_models(a) for a in artifacts]
    areas = sorted({h*w for h,w in shape_set})
    prediction_differences = []
    for i,j in itertools.combinations(range(len(artifacts)),2):
        prediction_differences.append(dict(sessions=[ids[i],ids[j]], models={name:dict(
            mean_absolute_difference_ms=float(np.mean([abs(models[i][name].predict_seconds(a)-models[j][name].predict_seconds(a))*1000 for a in areas])),
            max_absolute_difference_ms=max(abs(models[i][name].predict_seconds(a)-models[j][name].predict_seconds(a))*1000 for a in areas)) for name in pooled}))
    agreement = None
    if pairs:
        specs = json.loads(Path(pairs).read_text(encoding="utf-8"))["payload"]["pairs"]
        agreement = {}
        for name in pooled:
            decisions = [[m[name].predict_seconds(p["effective_areas"][2]) < sum(m[name].predict_seconds(a) for a in p["effective_areas"][:2]) for m in models] for p in specs]
            agreement[name] = float(np.mean([len(set(d)) == 1 for d in decisions]))
    b0,b1 = pooled["linear"]["coefficients"]["b0"],pooled["linear"]["coefficients"]["b1"]
    result = dict(schema_version=3, artifact_type="pooled_calibration", session_ids=ids,
                  session_hashes=[file_sha256(p) for p in sessions], compatibility_result=True,
                  provenance=first["provenance"], aggregation_provenance=system_metadata(),
                  calibration_envelope=first["calibration_envelope"], primary_fit_selector=selector,
                  pooled_shape_statistics=[dict(tensor_h=h,tensor_w=w,effective_area=h*w,**statistics(v,0)) for (h,w),v in sorted(grouped.items())],
                  pooling_method=method, latency_models=pooled, K_t_s=b0,c_t_s_per_pixel=b1,tau_pixels=b0/b1,
                  bootstrap=uncertainty, between_session_stability=stability, pairwise_prediction_differences=prediction_differences,
                  frozen_pair_decision_agreement=agreement, config=first.get("config", {}),
                  per_session_summaries=[dict(session_id=i,models={name:a["latency_models"][name] for name in ("linear","quadratic","piecewise")}) for i,a in zip(ids,artifacts)],
                  quality_warnings=["Fewer than three sessions"] if len(ids)<3 else [])
    threshold = float(first.get("config",{}).get("experiment_a",{}).get("session_coefficient_cv_threshold",.2))
    for name,report in stability.items():
        if any(v["cv"] is not None and v["cv"]>threshold for v in report["coefficients"].values()):
            result["quality_warnings"].append(f"{name}: between-session coefficient CV exceeds {threshold}")
    result["stability_thresholds"] = dict(session_coefficient_cv_threshold=threshold)
    result["schema_version"] = 4
    result["shape_policy"] = first.get("shape_policy",{})
    result["policy_declaration"] = first.get("policy_declaration",{})
    result["lookup_coverage"] = first.get("lookup_coverage",{})
    result["latency_models"]["shape_lookup"] = build_lookup(grouped,method,0,count,seed)
    bootstrap_path = Path(output).with_name(Path(output).stem+"_bootstrap_models.json")
    write_json(bootstrap_path,uncertainty)
    from .artifacts import reference
    result["bootstrap_reference"] = reference(bootstrap_path)
    result["bootstrap"] = {k:v for k,v in uncertainty.items() if k != "replicates"}
    result["control_records"] = [r for a in artifacts for r in a.get("control_records",[])]
    write_json(Path(output),result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", required=True, help="Session linear_fit.json files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs")
    parser.add_argument("--pooling-method", choices=["mean","median"], default="mean")
    args = parser.parse_args()
    aggregate(args.sessions,args.output,args.bootstrap_count,args.seed,args.pairs,args.pooling_method)


if __name__ == "__main__":
    main()
