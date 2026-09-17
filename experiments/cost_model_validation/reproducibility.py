"""Persisted schedules, robust calibration and order-stratified uncertainty."""
import hashlib
import json
import random
from collections import Counter

import numpy as np

from .models import fit_linear_model, fit_quadratic_model, fit_piecewise_model


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def gate(condition, name, config, warnings):
    if condition:
        return
    policy = config.get("quality_gates", {}).get(name, config.get("quality_gates", {}).get("default", "fail"))
    message = f"Quality gate: {name}"
    if policy == "fail":
        raise ValueError(message)
    if policy != "warn":
        raise ValueError(f"Unknown quality policy {policy}")
    warnings.append(message)


def schedule(shapes, repetitions, seed):
    result, previous = [], None
    for block in range(repetitions):
        order = list(range(len(shapes)))
        derived = int(canonical_hash([seed, block])[:16], 16)
        random.Random(derived).shuffle(order)
        if len(order) > 1 and order == previous:
            order = order[1:] + order[:1]
        previous = order
        for position, index in enumerate(order):
            h, w = shapes[index]
            result.append(dict(global_position=len(result), block_index=block, position_in_block=position,
                               shape_id=f"h{h}_w{w}", tensor_h=h, tensor_w=w, effective_area=h*w))
    return result


def statistics(values, trim=.1):
    if not 0 <= trim < .5:
        raise ValueError("trim_fraction_each_tail must be in [0, .5)")
    x = np.sort(np.asarray(values, dtype=float))
    k = int(len(x)*trim)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.
    q05, q25, q75, q95 = np.quantile(x, [.05, .25, .75, .95])
    return dict(count=len(x), mean_s=mean, median_s=float(np.median(x)),
                trimmed_mean_s=float(x[k:len(x)-k].mean()), std_s=std, standard_error_s=std/len(x)**.5,
                coefficient_of_variation=std/mean if mean else None, min_s=float(x[0]), max_s=float(x[-1]),
                p05_s=float(q05), p25_s=float(q25), p75_s=float(q75), p95_s=float(q95),
                outlier_count=int(np.sum((x < q25-1.5*(q75-q25)) | (x > q75+1.5*(q75-q25)))))


def design(groups, statistic="trimmed_mean", level="shape_level", trim=.1):
    if statistic not in {"mean", "median", "trimmed_mean"} or level not in {"shape_level", "area_level"}:
        raise ValueError("Invalid primary statistic or fit level")
    x = np.array([h*w for h, w in sorted(groups)], dtype=float)
    y = np.array([statistics(groups[key], trim)[statistic+"_s"] for key in sorted(groups)])
    if level == "area_level":
        areas = np.unique(x)
        y = np.array([y[x == a].mean() for a in areas])
        x = areas
    return x, y


def fit_models(x, y, support=2):
    result = {}
    for name, fitter in (("linear", fit_linear_model), ("quadratic", fit_quadratic_model), ("piecewise", fit_piecewise_model)):
        try:
            fit = fitter(x, y, min_support=support) if name == "piecewise" else fitter(x, y)
            c = fit.coefficients
            valid = c["b1"] > 0 and (name != "piecewise" or c["b1"]+c["b2"] > 0)
            if name == "linear":
                valid = valid and c["b0"] > 0
            result[name] = dict(coefficients=c, breakpoint_area=fit.breakpoint, valid=bool(valid),
                                reason=None if valid else "nonpositive_slopes_or_intercept", r2=fit.r2,
                                rmse_s=fit.rmse, mae_s=fit.mae, aic=fit.aic, bic=fit.bic)
        except ValueError as exc:
            result[name] = dict(valid=False, reason=str(exc), coefficients={})
    return result


def bootstrap_models(groups, count, seed, statistic, level, trim, support=2, sampler=None):
    if count < 1:
        raise ValueError("bootstrap count must be positive")
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(count):
        sampled = sampler(rng) if sampler else {key: rng.choice(values, len(values), replace=True) for key, values in sorted(groups.items())}
        replicates.append(fit_models(*design(sampled, statistic, level, trim), support))
    summaries = {}
    for name in ("linear", "quadratic", "piecewise"):
        valid = [r[name] for r in replicates if r[name]["valid"]]
        parameters = {}
        for key in ("b0", "b1", "b2", "c2", "breakpoint_area", "tau"):
            values = []
            for r in valid:
                c = r["coefficients"]
                value = (c["b1"]+c["b2"] if key == "c2" and name == "piecewise" else
                         c["b0"]/c["b1"] if key == "tau" and name == "linear" else
                         r.get(key) if key == "breakpoint_area" else c.get(key))
                if value is not None:
                    values.append(value)
            if values:
                median = float(np.median(values))
                parameters[key] = dict(ci=np.quantile(values, [.025, .975]).tolist(), median=median,
                                       mad=float(np.median(np.abs(np.array(values)-median))))
        summaries[name] = dict(valid_count=len(valid), invalid_fraction=1-len(valid)/max(1,count), parameters=parameters,
                               breakpoint_frequency=dict(Counter(str(r.get("breakpoint_area")) for r in valid)) if name == "piecewise" else {})
    return dict(method="fixed_design_within_shape", seed=seed, count=count, confidence_level=.95,
                replicates=replicates, summaries=summaries)


def order_estimate(rows, seed=0, count=2000, confidence=.95):
    if count < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap count or confidence level")
    groups = [np.array([float(r["difference_ms"]) for r in rows if r["order"] == order])
              for order in ("merged_first", "separate_first")]
    if not all(len(g) for g in groups):
        raise ValueError("Both order conditions are required")
    rng = np.random.default_rng(seed)
    draws = sum(rng.choice(g, (count, len(g)), replace=True).mean(axis=1) for g in groups)/2
    alpha = (1-confidence)/2
    return dict(estimate=float(sum(g.mean() for g in groups)/2), ci=np.quantile(draws, [alpha, 1-alpha]).tolist(),
                balanced=len(groups[0]) == len(groups[1]), order_counts=[len(g) for g in groups])
