from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def read_csvs(folder: Path) -> pd.DataFrame:
    csv_files: List[Path] = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    dfs = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            df["source_file"] = csv_path.name
            dfs.append(df)
        except Exception as exc:
            print(f"Skipping {csv_path.name}: {exc}")

    if not dfs:
        raise RuntimeError("No readable CSV files found")

    df = pd.concat(dfs, ignore_index=True)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove fully empty rows if present.
    df = df.dropna(how="all")

    # Normalize tracker column.
    if "benchmark_tracker_type" in df.columns:
        df["tracker"] = df["benchmark_tracker_type"]
    elif "tracker_type" in df.columns:
        df["tracker"] = df["tracker_type"]
    else:
        df["tracker"] = "unknown"

    # Normalize model name.
    if "model" not in df.columns:
        df["model"] = "unknown"

    # Normalize latency column.
    if "latency_mean_ms" not in df.columns:
        raise KeyError("Expected column 'latency_mean_ms' not found")

    # Create a single padding column for plotting.
    df["pad_x"] = pd.NA
    df["pad_y"] = pd.NA

    if "static_pad_x" in df.columns:
        df["pad_x"] = df["pad_x"].fillna(df["static_pad_x"])
    if "static_pad_y" in df.columns:
        df["pad_y"] = df["pad_y"].fillna(df["static_pad_y"])

    if "kalman_pad_x" in df.columns:
        df["pad_x"] = df["pad_x"].fillna(df["kalman_pad_x"])
    if "kalman_pad_y" in df.columns:
        df["pad_y"] = df["pad_y"].fillna(df["kalman_pad_y"])

    # Oracle GT often has zero padding in file fields; keep it explicit.
    df["pad_x"] = pd.to_numeric(df["pad_x"], errors="coerce")
    df["pad_y"] = pd.to_numeric(df["pad_y"], errors="coerce")

    # For symmetric padding, create a single padding field.
    df["padding"] = df["pad_x"]
    same_pad = (df["pad_x"].notna()) & (df["pad_y"].notna()) & (df["pad_x"] == df["pad_y"])
    df.loc[~same_pad, "padding"] = pd.NA

    # Normalize dropout column if present.
    if "tracker_dropout_prob" in df.columns:
        df["dropout_prob"] = pd.to_numeric(df["tracker_dropout_prob"], errors="coerce")
    elif "dropout_prob" in df.columns:
        df["dropout_prob"] = pd.to_numeric(df["dropout_prob"], errors="coerce")
    else:
        df["dropout_prob"] = pd.NA

    # Normalize memory / hold column.
    if "static_hold_last_for_frames" in df.columns:
        df["hold_frames"] = pd.to_numeric(df["static_hold_last_for_frames"], errors="coerce")
    elif "hold_detections_frames" in df.columns:
        df["hold_frames"] = pd.to_numeric(df["hold_detections_frames"], errors="coerce")
    else:
        df["hold_frames"] = pd.NA

    # Normalize some useful columns.
    numeric_cols = [
        "mAP50",
        "mAP95",
        "fps_total",
        "latency_mean_ms",
        "latency_roi_mean_ms",
        "latency_full_frame_mean_ms",
        "processed_area_ratio_mean",
        "full_frame_fraction",
        "detector_recall50",
        "detector_recall95",
        "gt_roi_coverage_mean",
        "gt_roi_coverage_p5",
        "key_frame_interval",
        "roi_count_pre_merge_mean",
        "roi_count_post_merge_mean",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine benchmark CSV files into one normalized table")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with CSV files")
    parser.add_argument("--output_csv", type=Path, default=Path("all_benchmarks_combined.csv"))
    args = parser.parse_args()

    df = read_csvs(args.input_dir)
    df = normalize_columns(df)
    df.to_csv(args.output_csv, index=False)

    print(f"Saved combined table to: {args.output_csv}")
    print(f"Rows: {len(df)}")
    print("Columns:")
    print(sorted(df.columns.tolist()))


if __name__ == "__main__":
    main()