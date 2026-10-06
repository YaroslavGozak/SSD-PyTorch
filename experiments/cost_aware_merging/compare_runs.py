"""Compare oracle merge decisions on identical geometry and frame content."""

import argparse
import csv
import json
from pathlib import Path

from .replay import validate_pairs


def compare(first_run, second_run, output):
    runs = [Path(first_run), Path(second_run)]
    payloads = [json.loads((run / "pairs.json").read_text(encoding="utf-8")) for run in runs]
    hashes = [validate_pairs(payload, payload["canvas_hw"]) for payload in payloads]
    if hashes[0] != hashes[1]:
        raise ValueError("Runs do not use the same ordered pair geometry and canvas")
    data = []
    metadata = [json.loads((run / "metadata.json").read_text(encoding="utf-8")) for run in runs]
    if metadata[0].get("timing_boundary") != metadata[1].get("timing_boundary"):
        raise ValueError("Runs use different timing boundaries")
    for run, payload in zip(runs, payloads):
        with (run / "pairs_raw.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        keyed = {str(row["pair_id"]): row for row in rows}
        if len(keyed) != len(rows) or set(keyed) != {str(pair["pair_id"]) for pair in payload["pairs"]}:
            raise ValueError("Run has missing or duplicate measured pairs")
        for pair in payload["pairs"]:
            row = keyed[str(pair["pair_id"])]
            for prefix in ("r1", "r2"):
                if [int(row[f"{prefix}_{axis}"]) for axis in ("x1", "y1", "x2", "y2")] != pair[prefix]:
                    raise ValueError("Measured pair geometry differs from pairs.json")
        data.append(keyed)
    comparisons = []
    for pair in payloads[0]["pairs"]:
        key = str(pair["pair_id"])
        a, b = data[0][key], data[1][key]
        if not a.get("frame_canvas_sha256") or not b.get("frame_canvas_sha256"):
            raise ValueError("Frame content hashes are missing; collect both runs with the updated runner")
        if (a["frame_canvas_sha256"], a["frame_key"]) != (b["frame_canvas_sha256"], b["frame_key"]):
            raise ValueError(f"Pair {key} uses different frame content")
        deltas = [float(row["actual_separate_ms"]) - float(row["actual_merged_ms"]) for row in (a, b)]
        decisions = [delta > 0 for delta in deltas]
        comparisons.append(dict(pair_id=pair["pair_id"], frame_key=a["frame_key"],
                                first_delta_ms=deltas[0], second_delta_ms=deltas[1],
                                first_merge=decisions[0], second_merge=decisions[1],
                                oracle_flip=decisions[0] != decisions[1]))
    same_model = (metadata[0].get("checkpoint_sha256") is not None and
                  metadata[0].get("checkpoint_sha256") == metadata[1].get("checkpoint_sha256") and
                  metadata[0].get("model", {}).get("stride") == metadata[1].get("model", {}).get("stride") and
                  metadata[0].get("preprocessing") == metadata[1].get("preprocessing"))
    summary = dict(first_run=str(runs[0].resolve()), second_run=str(runs[1].resolve()),
                   pairs_geometry_sha256=hashes[0], pair_count=len(comparisons),
                   oracle_flip_count=sum(row["oracle_flip"] for row in comparisons),
                   same_checkpoint_and_preprocessing=same_model,
                   comparison_kind="hardware_candidate" if same_model else "cross_model_or_preprocessing",
                   interpretation="Observed median decision flips; repeated runs are needed to distinguish noise")
    summary["oracle_flip_fraction"] = summary["oracle_flip_count"] / summary["pair_count"]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pair_flips.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    (output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_run", type=Path)
    parser.add_argument("second_run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    compare(args.first_run, args.second_run, args.output)
