"""Correct the geometric baseline from saved timings without model inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.cost_aware_merging.core import summarize
from tools.mergers.simple2 import simple_roi_merge_v2


def reanalyze(source):
    source = Path(source).resolve()
    raw_path = source / "pairs_raw.csv"
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    gamma = metadata["experiment_options"]["geometric_gamma"]
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    changed = 0
    for row in rows:
        for key in row:
            if key.endswith(("_decision", "_correct")):
                if row[key] not in ("True", "False"):
                    raise ValueError(f"Invalid boolean in {key}: {row[key]!r}")
                row[key] = row[key] == "True"
        for key in ("actual_separate_ms", "actual_merged_ms"):
            row[key] = float(row[key])
        boxes = [tuple(int(row[f"{prefix}_{axis}"]) for axis in ("x1", "y1", "x2", "y2"))
                 for prefix in ("r1", "r2")]
        decision = len(simple_roi_merge_v2(boxes, area_ratio_max=gamma)) == 1
        changed += decision != row["geometric_area_decision"]
        row["geometric_area_decision"] = decision
        row["geometric_area_correct"] = decision == row["oracle_decision"]
        row["geometric_area_ratio"] = float(row["merged_area"]) / (
            float(row["r1_area"]) + float(row["r2_area"]))
        if decision != (row["geometric_area_ratio"] <= gamma):
            raise RuntimeError("Geometric merger disagrees with the pairwise area criterion")
    summary = summarize(rows)
    # Keep original results and plots intact so the correction remains auditable.
    output = source / "geometric_corrected"
    output.mkdir(exist_ok=True)
    for name, records in (("pairs_raw", rows), ("summary", summary)):
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    provenance = dict(source=str(source), geometric_gamma=gamma, changed_decisions=changed,
                      pair_count=len(rows), timings_reused=True,
                      corrected_policy="geometric_area",
                      criterion="merged_area / (r1_area + r2_area) <= geometric_gamma",
                      source_pairs_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                      source_metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest())
    (output / "correction.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    geometric = next(item for item in summary if item["method"] == "geometric_area")
    print(f"{source.name}: {changed}/{len(rows)} decisions corrected; "
          f"TP={geometric['TP']}, FP={geometric['FP']}, "
          f"FN={geometric['FN']}, TN={geometric['TN']}; output: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing experiment output directory")
    reanalyze(parser.parse_args().source)
