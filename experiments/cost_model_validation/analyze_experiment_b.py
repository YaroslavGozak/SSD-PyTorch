"""Analyze paired Experiment B timings."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import write_json


def _mean_ci(values, seed, iterations=2000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [float(np.mean(values)), float(np.mean(values))]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def _classification_metrics(rows):
    determinate = [row for row in rows if row["label"] != "ambiguous"]
    actual_classes = {row["label"] for row in determinate}
    has_both_classes = actual_classes == {"merge_beneficial", "separate_beneficial"}
    actual = [row["label"] == "merge_beneficial" for row in determinate]
    predicted = [row["predicted_merge"] for row in determinate]
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
            "accuracy_bootstrap_ci": _mean_ci([int(p == a) for p, a in zip(predicted, actual)], 17)}


def analyze(path: str, output: str | None = None):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(float(row["difference_ms"]))
    summaries = []
    for pair_id, values in grouped.items():
        ci = _mean_ci(values, seed=int(pair_id))
        row = next(r for r in rows if r["pair_id"] == pair_id)
        label = "merge_beneficial" if ci[0] > 0 else ("separate_beneficial" if ci[1] < 0 else "ambiguous")
        summaries.append({"pair_id": pair_id, "predicted_merge": row["predicted_merge"].lower() == "true", "label": label, "difference_ci_ms": ci, "mean_difference_ms": float(np.mean(values)), "boundary_bin": row["boundary_bin"], "geometry_type": row["geometry_type"], "computational_key": tuple(row[key] for key in ("r1_tensor_h", "r1_tensor_w", "r2_tensor_h", "r2_tensor_w", "union_tensor_h", "union_tensor_w"))})
    metrics = _classification_metrics(summaries)
    regrets = [max(0.0, -r["mean_difference_ms"] if r["predicted_merge"] else r["mean_difference_ms"]) for r in summaries]
    order_means = {order: float(np.mean([float(row["difference_ms"]) for row in rows if row["order"] == order])) for order in ("separate_first", "merged_first") if any(row["order"] == order for row in rows)}
    order_effect = order_means.get("separate_first", float("nan")) - order_means.get("merged_first", float("nan"))
    by_bin = {key: {"pairs": sum(r["boundary_bin"] == key for r in summaries), **_classification_metrics([r for r in summaries if r["boundary_bin"] == key])} for key in sorted({r["boundary_bin"] for r in summaries})}
    result = {"pairs": len(summaries), "ambiguous_fraction": sum(r["label"] == "ambiguous" for r in summaries) / max(len(summaries), 1), **metrics,
              "order_means_ms": order_means, "order_effect_ms": order_effect, "order_effect_warning": bool(np.isfinite(order_effect) and abs(order_effect) > .5), "metrics_by_boundary_bin": by_bin,
              "unique_computational_configurations": len({r["computational_key"] for r in summaries}),
              "regret_ms": {"mean": float(np.mean(regrets)) if regrets else float("nan"), "median": float(np.median(regrets)) if regrets else float("nan"), "p95": float(np.percentile(regrets, 95)) if regrets else float("nan"), "p99": float(np.percentile(regrets, 99)) if regrets else float("nan"), "max": float(np.max(regrets)) if regrets else float("nan")}, "pair_summaries": summaries}
    if output: write_json(Path(output) / "decision_metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output"); args = parser.parse_args(); print(json.dumps(analyze(args.input, args.output), indent=2))


if __name__ == "__main__": main()