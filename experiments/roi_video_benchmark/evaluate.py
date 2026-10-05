"""Re-evaluate saved detector outputs without loading models or running inference."""
import argparse
import json
from pathlib import Path

from tools.benchmarks.video_artifacts import detection_metrics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections", help="Path to detections.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.detections).read_text(encoding="utf-8"))
    metrics = detection_metrics(data["predictions"], data["ground_truths"], data["difficulties"])
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
