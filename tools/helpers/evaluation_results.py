"""Read complete sweep summaries without importing the inference pipeline."""
import csv
from pathlib import Path


def read_completed_results(path):
    if not Path(path).exists():
        return []
    rows = []
    with open(path, newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            try:
                if not row.get('evaluated_dataset') or None in row:
                    continue
                for key in ('mAP', 'mean_detector_recall', 'mean_latency_ms', 'mean_fps'):
                    float(row[key])
                if any(value is None for value in row.values()):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    return rows
