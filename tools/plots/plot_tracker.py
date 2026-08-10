from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Plot detector-wise tracker FPS vs mAP50 charts from benchmark CSV. "
			"Default behavior is interactive display."
		)
	)
	parser.add_argument("--input_csv", type=Path, required=True, help="Input benchmark CSV path")
	parser.add_argument(
		"--detectors",
		type=str,
		default=None,
		help="Optional comma-separated detector filter, e.g. yolo,roissd",
	)
	parser.add_argument(
		"--trackers",
		type=str,
		default=None,
		help="Optional comma-separated tracker filter, e.g. static_padding,kalman",
	)
	parser.add_argument(
		"--mergers",
		type=str,
		default=None,
		help="Optional comma-separated merger filter, e.g. none,simple",
	)
	parser.add_argument(
		"--key-intervals",
		type=str,
		default=None,
		help="Optional comma-separated key-frame intervals, e.g. 5,10",
	)
	parser.add_argument(
		"--no-show",
		action="store_true",
		help="Skip interactive display (useful for headless runs).",
	)
	parser.add_argument(
		"--fullframe-maps-out",
		type=Path,
		default=None,
		help=(
			"Optional output CSV path for full-frame baselines (tracker=none, k=1). "
			"Defaults to <input_csv_stem>_fullframe_map50.csv next to --input_csv."
		),
	)
	return parser.parse_args()


def _normalize_tracker_column(df: pd.DataFrame) -> pd.DataFrame:
	if "benchmark_tracker_type" in df.columns:
		df = df.assign(tracker=df["benchmark_tracker_type"])
	elif "tracker_type" in df.columns:
		df = df.assign(tracker=df["tracker_type"])
	else:
		raise KeyError("Missing tracker column: benchmark_tracker_type or tracker_type")
	return df


def _normalize_merger_column(df: pd.DataFrame) -> pd.DataFrame:
	if "merge_fn" in df.columns:
		out = df.assign(merger=df["merge_fn"])
	elif "roi_merge_strategy" in df.columns:
		out = df.assign(merger=df["roi_merge_strategy"])
	elif "merge_strategy" in df.columns:
		out = df.assign(merger=df["merge_strategy"])
	else:
		out = df.assign(merger="unknown")

	# Some CSVs may contain placeholder strings copied from header names.
	bad_merger_values = {"", "nan", "none", "merge_fn", "merger"}
	merged = out["merger"].astype(str).str.strip()
	mask_bad = merged.str.lower().isin(bad_merger_values - {"none"})
	out.loc[mask_bad, "merger"] = "unknown"
	out["merger"] = out["merger"].astype(str).str.strip()
	return out


def _normalize_key_interval_column(df: pd.DataFrame) -> pd.DataFrame:
	if "key_frame_interval" in df.columns:
		series = pd.to_numeric(df["key_frame_interval"], errors="coerce")
	elif "key_frame_interval.1" in df.columns:
		series = pd.to_numeric(df["key_frame_interval.1"], errors="coerce")
	else:
		series = pd.Series([pd.NA] * len(df), index=df.index)

	# Some CSVs may contain duplicated key-frame columns from schema changes.
	if "key_frame_interval.1" in df.columns:
		fallback = pd.to_numeric(df["key_frame_interval.1"], errors="coerce")
		series = series.fillna(fallback)

	return df.assign(key_frame_interval=series)


def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
	required = ["model_family", "mAP50", "fps_total", "experiment_name"]
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise KeyError(f"Missing required columns: {', '.join(missing)}")

	df = _normalize_tracker_column(df)
	df = _normalize_merger_column(df)
	df = _normalize_key_interval_column(df)

	out = df.copy()
	out["mAP50"] = pd.to_numeric(out["mAP50"], errors="coerce")
	out["fps_total"] = pd.to_numeric(out["fps_total"], errors="coerce")
	out = out.dropna(
		subset=[
			"mAP50",
			"fps_total",
			"tracker",
			"model_family",
			"experiment_name",
			"merger",
			"key_frame_interval",
		]
	)
	return out


def _parse_csv_filter(raw: str | None) -> List[str] | None:
	if not raw:
		return None
	values = [x.strip() for x in raw.split(",") if x.strip()]
	return values or None


def _annotation_from_experiment_name(experiment_name: str, tracker: str) -> str:
	exp = str(experiment_name)
	trk = str(tracker)
	prefix = f"{trk}_"
	if exp.startswith(prefix):
		return exp[len(prefix):]

	if "_" in exp:
		return exp.split("_", 1)[1]

	m = re.search(r"(\d+(?:p\d+)?)$", exp)
	if m:
		return m.group(1)
	return exp


def _format_key_interval(v: float | int) -> str:
	try:
		iv = int(float(v))
		return str(iv)
	except Exception:
		return str(v)


def _extract_fullframe_baseline(df_group: pd.DataFrame) -> float | None:
	baseline_df = df_group[
		(df_group["tracker"].astype(str).str.strip().str.lower() == "none")
		& (pd.to_numeric(df_group["key_frame_interval"], errors="coerce") == 1)
	]
	if baseline_df.empty:
		return None
	baseline_value = pd.to_numeric(baseline_df["mAP50"], errors="coerce").dropna()
	if baseline_value.empty:
		return None
	return float(baseline_value.mean())


def _extract_fullframe_rows(df: pd.DataFrame) -> pd.DataFrame:
	return df[
		(df["tracker"].astype(str).str.strip().str.lower() == "none")
		& (pd.to_numeric(df["key_frame_interval"], errors="coerce") == 1)
	].copy()


def _save_fullframe_maps(fullframe_df: pd.DataFrame, output_path: Path) -> Path:
	preferred_cols = [
		"model_family",
		"merger",
		"experiment_name",
		"mAP50",
		"mAP95",
		"fps_total",
		"tracker",
		"key_frame_interval",
		"dataset",
		"run_timestamp_utc",
	]
	cols = [col for col in preferred_cols if col in fullframe_df.columns]
	out = fullframe_df.loc[:, cols].copy()
	out = out.sort_values(["model_family", "merger", "mAP50"], ascending=[True, True, False])

	output_path.parent.mkdir(parents=True, exist_ok=True)
	out.to_csv(output_path, index=False)
	return output_path


def _plot_group(
	ax: plt.Axes,
	df_group: pd.DataFrame,
	detector: str,
	merger: str,
	baseline_source_df: pd.DataFrame | None = None,
) -> int:
	point_count = 0
	baseline_df = baseline_source_df if baseline_source_df is not None else df_group
	baseline_map50 = _extract_fullframe_baseline(baseline_df)
	plot_df = df_group[
		~(
			(df_group["tracker"].astype(str).str.strip().str.lower() == "none")
			& (pd.to_numeric(df_group["key_frame_interval"], errors="coerce") == 1)
		)
	]

	for (tracker_name, key_frame_interval), tracker_df in plot_df.groupby(["tracker", "key_frame_interval"]):
		tracker_df = tracker_df.sort_values("fps_total")
		label_name = f"{tracker_name} | k={_format_key_interval(key_frame_interval)}"
		ax.plot(
			tracker_df["fps_total"],
			tracker_df["mAP50"],
			marker="o",
			linewidth=1.5,
			markersize=4,
			label=label_name,
		)
		for _, row in tracker_df.iterrows():
			label = _annotation_from_experiment_name(row["experiment_name"], tracker_name)
			ax.annotate(
				label,
				(row["fps_total"], row["mAP50"]),
				textcoords="offset points",
				xytext=(3, 3),
				fontsize=8,
				alpha=0.9,
			)
			point_count += 1

	if baseline_map50 is not None:
		ax.axhline(
			y=baseline_map50,
			color="red",
			linestyle="--",
			linewidth=1,
			alpha=0.8,
			label=f"Full-frame baseline (tracker=none, k=1): {baseline_map50:.4f}",
		)

	ax.set_title(f"Detector: {detector} | Merger: {merger}")
	ax.set_xlabel("FPS (fps_total)")
	ax.set_ylabel("Accuracy (mAP50)")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=8)
	return point_count


def main() -> None:
	args = parse_args()

	df = pd.read_csv(args.input_csv)
	df = _validate_and_prepare(df)

	detector_filter = _parse_csv_filter(args.detectors)
	if detector_filter is not None:
		df = df[df["model_family"].astype(str).isin(detector_filter)]

	# Keep detector-level baseline source before merger/tracker/key filters.
	baseline_df = df.copy()
	fullframe_rows = _extract_fullframe_rows(baseline_df)
	if args.fullframe_maps_out is None:
		fullframe_out_path = args.input_csv.with_name(f"{args.input_csv.stem}_fullframe_map50.csv")
	else:
		fullframe_out_path = args.fullframe_maps_out
	if fullframe_rows.empty:
		print("No full-frame baseline rows found (tracker=none, key_frame_interval=1).")
	else:
		saved_path = _save_fullframe_maps(fullframe_rows, fullframe_out_path)
		print(f"Saved full-frame baseline mAP rows to: {saved_path}")

	merger_filter = _parse_csv_filter(args.mergers)
	if merger_filter is not None:
		df = df[df["merger"].astype(str).isin(merger_filter)]

	tracker_filter = _parse_csv_filter(args.trackers)
	if tracker_filter is not None:
		df = df[df["tracker"].astype(str).isin(tracker_filter)]

	key_interval_filter_raw = _parse_csv_filter(args.key_intervals)
	if key_interval_filter_raw is not None:
		allowed = {int(x) for x in key_interval_filter_raw}
		df = df[df["key_frame_interval"].astype(int).isin(allowed)]

	if df.empty and baseline_df.empty:
		raise RuntimeError("No rows left to plot after validation/filtering")

	detector_source_df = baseline_df if not baseline_df.empty else df
	detectors = sorted(detector_source_df["model_family"].astype(str).unique().tolist())
	if merger_filter is not None:
		mergers = merger_filter
	else:
		mergers = sorted(
			m for m in detector_source_df["merger"].astype(str).unique().tolist()
			if m and m != "unknown"
		)
		if not mergers:
			mergers = ["unknown"]
	plotted_summary: List[str] = []
	for detector in detectors:
		baseline_detector_df = baseline_df[
			baseline_df["model_family"].astype(str) == detector
		]
		for merger in mergers:
			group_df = df[
				(df["model_family"].astype(str) == detector)
				& (df["merger"].astype(str) == merger)
			]
			fig, ax = plt.subplots(figsize=(8, 5))
			if group_df.empty and baseline_detector_df.empty:
				ax.set_title(f"Detector: {detector} | Merger: {merger}")
				ax.set_xlabel("FPS (fps_total)")
				ax.set_ylabel("Accuracy (mAP50)")
				ax.grid(True, alpha=0.3)
				ax.text(0.5, 0.5, "No data for this combination", ha="center", va="center", transform=ax.transAxes)
				points = 0
			else:
				points = _plot_group(ax, group_df, detector, merger, baseline_source_df=baseline_detector_df)
			plotted_summary.append(f"{detector}/{merger}: {points} points")

	print("Plotted detector/merger groups:")
	for line in plotted_summary:
		print(f"- {line}")

	if not args.no_show:
		plt.show()
	else:
		plt.close("all")


if __name__ == "__main__":
	main()
