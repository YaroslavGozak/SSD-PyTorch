"""Convert benchmark comparison CSVs to a compact, ordered schema.

Usage:
	python -m tools.benchmarks.compact_csv \
		--input benchmark_results/vid/imagenet-vid/comparison2026_5_29.csv \
		--output benchmark_results/vid/imagenet-vid/comparison2026_5_29_compact.csv

Without --output, a sibling file ending with _compact.csv is generated.

Optional strict mode:
	python -m tools.benchmarks.compact_csv --input in.csv --output out.csv --strict
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Set


OUTPUT_COLUMNS: List[str] = [
	"model",
	"model_checkpoint",
	"tracker",
	"merger",
	"keyframe_interval",
	"experiment_name",
	"mAP50",
	"mAP95",
	"detector_recall50",
	"fps_total",
	"latency_mean_ms",
	"latency_full_frame_mean_ms",
	"latency_roi_mean_ms",
	"full_frame_fraction",
	"merge_latency_mean_ms",
	"processed_area_ratio_mean",
	"roi_count_pre_merge_mean",
	"roi_count_post_merge_mean",
	"dataset",
	"num_frames",
]


SOURCE_MAP: Dict[str, str] = {
	"model": "model",
	"model_checkpoint": "model_checkpoint",
	"experiment_name": "experiment_name",
	"mAP50": "mAP50",
	"mAP95": "mAP95",
	"detector_recall50": "detector_recall50",
	"fps_total": "fps_total",
	"latency_mean_ms": "latency_mean_ms",
	"latency_full_frame_mean_ms": "latency_full_frame_mean_ms",
	"latency_roi_mean_ms": "latency_roi_mean_ms",
	"full_frame_fraction": "full_frame_fraction",
	"merge_latency_mean_ms": "merge_latency_mean_ms",
	"processed_area_ratio_mean": "processed_area_ratio_mean",
	"roi_count_pre_merge_mean": "roi_count_pre_merge_mean",
	"roi_count_post_merge_mean": "roi_count_post_merge_mean",
	"dataset": "dataset",
	"num_frames": "num_frames",
}


TRACKER_FALLBACK_FIELDS: List[str] = ["benchmark_tracker_type", "tracker_type"]
MERGER_FALLBACK_FIELDS: List[str] = ["merger", "merge_fn"]
KEYFRAME_FALLBACK_FIELDS: List[str] = ["keyframe_interval", "key_frame_interval", "key_frame_interval.1"]


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert benchmark CSV into compact ordered comparison CSV."
	)
	parser.add_argument("--input", required=True, type=Path, help="Input CSV path.")
	parser.add_argument(
		"--output",
		type=Path,
		help="Output CSV path. If omitted, uses <input_stem>_compact.csv next to input.",
	)
	parser.add_argument(
		"--strict",
		action="store_true",
		help="Fail if any required input source columns are missing.",
	)
	return parser.parse_args()


def _derive_output_path(input_path: Path, output_path: Path | None) -> Path:
	if output_path is not None:
		return output_path

	if input_path.suffix:
		return input_path.with_name(f"{input_path.stem}_compact{input_path.suffix}")

	return input_path.with_name(f"{input_path.name}_compact.csv")


def _required_source_columns() -> Set[str]:
	return set(SOURCE_MAP.values())


def _validate_input_columns(fieldnames: Iterable[str], strict: bool) -> None:
	if not strict:
		return

	existing = set(fieldnames)
	missing = sorted(col for col in _required_source_columns() if col not in existing)
	has_tracker_source = any(col in existing for col in TRACKER_FALLBACK_FIELDS)
	has_merger_source = any(col in existing for col in MERGER_FALLBACK_FIELDS)
	has_keyframe_source = any(col in existing for col in KEYFRAME_FALLBACK_FIELDS)

	if not has_tracker_source:
		missing.append("benchmark_tracker_type|tracker_type")
	if not has_merger_source:
		missing.append("merger|merge_fn")
	if not has_keyframe_source:
		missing.append("keyframe_interval|key_frame_interval|key_frame_interval.1")

	if missing:
		raise ValueError(
			"Missing required columns in strict mode: " + ", ".join(missing)
		)


def _extract_tracker(row: Dict[str, str]) -> str:
	for key in TRACKER_FALLBACK_FIELDS:
		val = row.get(key, "")
		if val:
			return val
	return ""


def _extract_merger(row: Dict[str, str]) -> str:
	for key in MERGER_FALLBACK_FIELDS:
		val = row.get(key, "")
		if val:
			return val
	return ""


def _extract_keyframe_interval(row: Dict[str, str]) -> str:
	for key in KEYFRAME_FALLBACK_FIELDS:
		val = row.get(key, "")
		if val:
			return val
	return ""


def _transform_row(row: Dict[str, str]) -> Dict[str, str]:
	out: Dict[str, str] = {}
	for output_key in OUTPUT_COLUMNS:
		if output_key == "tracker":
			out[output_key] = _extract_tracker(row)
			continue
		if output_key == "merger":
			out[output_key] = _extract_merger(row)
			continue
		if output_key == "keyframe_interval":
			out[output_key] = _extract_keyframe_interval(row)
			continue
		source_key = SOURCE_MAP[output_key]
		out[output_key] = row.get(source_key, "")
	return out


def convert_csv(input_path: Path, output_path: Path, strict: bool = False) -> int:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	row_count = 0
	with input_path.open("r", newline="", encoding="utf-8") as f_in, output_path.open(
		"w", newline="", encoding="utf-8"
	) as f_out:
		reader = csv.DictReader(f_in)
		if not reader.fieldnames:
			raise ValueError(f"Input CSV has no header: {input_path}")

		_validate_input_columns(reader.fieldnames, strict=strict)

		writer = csv.DictWriter(f_out, fieldnames=OUTPUT_COLUMNS)
		writer.writeheader()

		for row in reader:
			writer.writerow(_transform_row(row))
			row_count += 1

	return row_count


def main() -> None:
	args = _parse_args()
	output_path = _derive_output_path(args.input, args.output)
	row_count = convert_csv(args.input, output_path, strict=args.strict)
	print(f"Wrote {row_count} rows to {output_path}")


if __name__ == "__main__":
	main()
