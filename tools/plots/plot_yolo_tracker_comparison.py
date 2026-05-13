from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_plot(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "tracker", "padding", "dropout_prob", "hold_frames"]
    group_cols = [c for c in group_cols if c in df.columns]

    metrics = [
        "mAP50",
        "fps_total",
        "latency_mean_ms",
        "processed_area_ratio_mean",
        "full_frame_fraction",
    ]
    metrics = [m for m in metrics if m in df.columns]

    return (
        df.groupby(group_cols, dropna=False)[metrics]
        .mean()
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare static vs kalman ROI generation for YOLO models")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("plots_yolo_tracker"))
    parser.add_argument("--dropout", type=float, default=None, help="Optional dropout filter, e.g. 0.7")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Keep YOLO rows only.
    # df = df[df["model"].astype(str).str.contains("yolo", case=False, na=False)]

    # Keep static + kalman only.
    df = df[df["tracker"].isin(["static_padding", "kalman"])]

    if args.dropout is not None and "dropout_prob" in df.columns:
        df = df[df["dropout_prob"] == args.dropout]

    df = aggregate(df)

    if df.empty:
        raise RuntimeError("No matching YOLO static/kalman rows found")

    # 1) Accuracy vs latency
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model_name, tracker_name), g in df.groupby(["model", "tracker"]):
        ax.scatter(g["latency_mean_ms"], g["mAP50"], s=70, label=f"{model_name}-{tracker_name}")
        for _, row in g.iterrows():
            padding = row["padding"]
            label = f"p{int(padding)}" if pd.notna(padding) else tracker_name
            ax.annotate(label, (row["latency_mean_ms"], row["mAP50"]), fontsize=8)

    ax.set_xlabel("Latency mean (ms)")
    ax.set_ylabel("mAP50")
    ax.set_title("YOLO: Static vs Kalman ROI")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_plot(fig, args.output_dir, "yolo_static_vs_kalman_accuracy_latency")

    # 2) Area vs accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model_name, tracker_name), g in df.groupby(["model", "tracker"]):
        ax.scatter(g["processed_area_ratio_mean"], g["mAP50"], s=70, label=f"{model_name}-{tracker_name}")
        for _, row in g.iterrows():
            padding = row["padding"]
            label = f"p{int(padding)}" if pd.notna(padding) else tracker_name
            ax.annotate(label, (row["processed_area_ratio_mean"], row["mAP50"]), fontsize=8)

    ax.set_xlabel("Processed area ratio mean")
    ax.set_ylabel("mAP50")
    ax.set_title("YOLO: Area vs Accuracy for Static and Kalman ROI")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_plot(fig, args.output_dir, "yolo_static_vs_kalman_area_accuracy")

    # 3) Full-frame fraction vs latency
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model_name, tracker_name), g in df.groupby(["model", "tracker"]):
        ax.scatter(g["full_frame_fraction"], g["latency_mean_ms"], s=70, label=f"{model_name}-{tracker_name}")
        for _, row in g.iterrows():
            padding = row["padding"]
            label = f"p{int(padding)}" if pd.notna(padding) else tracker_name
            ax.annotate(label, (row["full_frame_fraction"], row["latency_mean_ms"]), fontsize=8)

    ax.set_xlabel("Full-frame fraction")
    ax.set_ylabel("Latency mean (ms)")
    ax.set_title("YOLO: Full-frame Fallback vs Latency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_plot(fig, args.output_dir, "yolo_static_vs_kalman_fullframe_latency")


if __name__ == "__main__":
    main()