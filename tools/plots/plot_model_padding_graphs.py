from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def save_plot(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def aggregate_repeats(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "tracker", "padding"]
    optional_cols = ["dropout_prob", "hold_frames", "key_frame_interval"]
    for col in optional_cols:
        if col in df.columns:
            group_cols.append(col)

    metrics = [
        "mAP50",
        "fps_total",
        "latency_mean_ms",
        "processed_area_ratio_mean",
        "full_frame_fraction",
    ]
    metrics = [m for m in metrics if m in df.columns]

    out = (
        df.groupby(group_cols, dropna=False)[metrics]
        .mean()
        .reset_index()
        .sort_values(group_cols)
    )
    return out


def plot_padding_vs_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_name: str,
    output_dir: Path,
    tracker_filter: Optional[str] = None,
) -> None:
    plot_df = df.copy()
    plot_df = plot_df[plot_df["padding"].notna()]

    if tracker_filter is not None:
        plot_df = plot_df[plot_df["tracker"] == tracker_filter]

    if plot_df.empty:
        print(f"No data for {output_name}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, g in plot_df.groupby("model"):
        g = g.sort_values("padding")
        ax.plot(g["padding"], g[metric], marker="o", label=model_name)

    ax.set_xlabel("Padding (pixels)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_plot(fig, output_dir, output_name)


def plot_accuracy_vs_latency(
    df: pd.DataFrame,
    output_dir: Path,
    tracker_filter: Optional[str] = None,
    title_suffix: str = "",
) -> None:
    plot_df = df.copy()

    if tracker_filter is not None:
        plot_df = plot_df[plot_df["tracker"] == tracker_filter]

    plot_df = plot_df.dropna(subset=["mAP50", "latency_mean_ms"])
    if plot_df.empty:
        print("No data for accuracy_vs_latency")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, g in plot_df.groupby("model"):
        ax.scatter(g["latency_mean_ms"], g["mAP50"], s=70, label=model_name)
        for _, row in g.iterrows():
            padding = row["padding"]
            label = "full" if pd.isna(padding) else f"p{int(padding)}"
            ax.annotate(label, (row["latency_mean_ms"], row["mAP50"]), fontsize=8, alpha=0.8)

    ax.set_xlabel("Latency mean (ms)")
    ax.set_ylabel("mAP50")
    ax.set_title(f"Accuracy vs Latency{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    suffix = tracker_filter if tracker_filter else "all"
    save_plot(fig, output_dir, f"accuracy_vs_latency_{suffix}")


def plot_area_vs_accuracy(
    df: pd.DataFrame,
    output_dir: Path,
    tracker_filter: Optional[str] = None,
    title_suffix: str = "",
) -> None:
    plot_df = df.copy()

    if tracker_filter is not None:
        plot_df = plot_df[plot_df["tracker"] == tracker_filter]

    plot_df = plot_df.dropna(subset=["processed_area_ratio_mean", "mAP50"])
    if plot_df.empty:
        print("No data for area_vs_accuracy")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, g in plot_df.groupby("model"):
        ax.scatter(g["processed_area_ratio_mean"], g["mAP50"], s=70, label=model_name)
        for _, row in g.iterrows():
            padding = row["padding"]
            label = "full" if pd.isna(padding) else f"p{int(padding)}"
            ax.annotate(label, (row["processed_area_ratio_mean"], row["mAP50"]), fontsize=8, alpha=0.8)

    ax.set_xlabel("Processed area ratio mean")
    ax.set_ylabel("mAP50")
    ax.set_title(f"Accuracy vs Processed Area{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    suffix = tracker_filter if tracker_filter else "all"
    save_plot(fig, output_dir, f"accuracy_vs_area_{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model/padding comparison graphs")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("plots_models"))
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Optional tracker filter, e.g. oracle_gt, static_padding, kalman",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = aggregate_repeats(df)

    suffix = f" ({args.tracker})" if args.tracker else ""

    plot_padding_vs_metric(
        df,
        metric="mAP50",
        title=f"Padding vs mAP50{suffix}",
        output_name=f"padding_vs_mAP50_{args.tracker or 'all'}",
        output_dir=args.output_dir,
        tracker_filter=args.tracker,
    )

    plot_padding_vs_metric(
        df,
        metric="latency_mean_ms",
        title=f"Padding vs Latency{suffix}",
        output_name=f"padding_vs_latency_{args.tracker or 'all'}",
        output_dir=args.output_dir,
        tracker_filter=args.tracker,
    )

    plot_padding_vs_metric(
        df,
        metric="fps_total",
        title=f"Padding vs FPS{suffix}",
        output_name=f"padding_vs_fps_{args.tracker or 'all'}",
        output_dir=args.output_dir,
        tracker_filter=args.tracker,
    )

    plot_accuracy_vs_latency(df, args.output_dir, tracker_filter=args.tracker, title_suffix=suffix)
    plot_area_vs_accuracy(df, args.output_dir, tracker_filter=args.tracker, title_suffix=suffix)


if __name__ == "__main__":
    main()