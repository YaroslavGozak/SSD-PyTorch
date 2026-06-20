from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_X_METRIC = "latency_mean_ms"
DEFAULT_Y_METRIC = "mAP50"

METRIC_LABELS_UA = {
    "latency_mean_ms": "Середня затримка (мс)",
    "mAP50": "mAP50",
    "mAP95": "mAP95",
    "fps_total": "Швидкодія (FPS)",
    "processed_area_ratio_mean": "Середня частка обробленої площі",
    "static_padding": "Статичний відступ",
    "kalman": "Фільтр Калмана",
    "sort": "SORT",
    "greedy": "Жадібне злиття",
    "simple": "Злиття за просторовою близькістю",
    "simple_v2": "Злиття за співвідношенням площ",
    "none": "Без злиття",
    "невідомо": "Невідомо",
}

MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]


def metric_label(metric_name: str) -> str:
    return METRIC_LABELS_UA.get(metric_name, metric_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Pareto frontier chart from benchmark CSV data."
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument(
        "--x_metric",
        type=str,
        default=DEFAULT_X_METRIC,
        help="Metric on the X axis. Default: latency_mean_ms",
    )
    parser.add_argument(
        "--y_metric",
        type=str,
        default=DEFAULT_Y_METRIC,
        help="Metric on the Y axis. Default: mAP50",
    )
    parser.add_argument(
        "--x_goal",
        type=str,
        choices=["min", "max"],
        default="min",
        help="Whether lower or higher X values are better.",
    )
    parser.add_argument(
        "--y_goal",
        type=str,
        choices=["min", "max"],
        default="max",
        help="Whether lower or higher Y values are better.",
    )
    parser.add_argument(
        "--annotate",
        type=str,
        choices=["none", "frontier", "all"],
        default="none",
        help="Which points to annotate with experiment names.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Optional tracker filter, for example sort, kalman, or static_padding.",
    )
    parser.add_argument(
        "--non-dominated-points",
        action="store_true",
        help="If set, draw only non-dominated Pareto frontier points.",
    )
    return parser.parse_args()


def extract_padding_group(row: pd.Series) -> str:
    experiment_name = str(row.get("experiment_name", "")).strip()
    matches = re.findall(r"\d+", experiment_name)
    if matches:
        return matches[0]

    for column in ["static_pad_x", "static_pad_y", "oracle_gt_pad_x", "oracle_gt_pad_y"]:
        value = row.get(column)
        if pd.notna(value):
            try:
                return str(int(float(value)))
            except (TypeError, ValueError):
                continue

    return "невідомо"


def padding_sort_key(value: str) -> tuple[int, float | str]:
    if value == "невідомо":
        return (1, value)
    try:
        return (0, float(value))
    except ValueError:
        return (0, value)


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def filter_by_tracker(df: pd.DataFrame, tracker: str | None) -> pd.DataFrame:
    if tracker is None:
        return df
    if "tracker_type" not in df.columns:
        raise ValueError("CSV is missing required column for tracker filtering: ['tracker_type']")

    filtered = df[df["tracker_type"].astype(str) == tracker].copy()
    if filtered.empty:
        raise RuntimeError(f"No rows match tracker filter: {tracker}")
    return filtered


def aggregate_experiments(df: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    working = df.copy()
    working["padding_group"] = working.apply(extract_padding_group, axis=1)
    working["is_fullframe"] = (
        working["experiment_name"].astype(str).str.contains("fullframe", case=False, na=False)
    )

    group_columns = ["experiment_name", "padding_group", "is_fullframe"]
    for column in ["tracker_type", "merge_fn"]:
        if column in working.columns:
            group_columns.append(column)

    metric_columns = list(
        dict.fromkeys(
            column
            for column in [x_metric, y_metric, "fps_total", "processed_area_ratio_mean"]
            if column in working.columns
        )
    )

    aggregated = (
        working.groupby(group_columns, dropna=False)[metric_columns]
        .mean()
        .reset_index()
    )
    aggregated = aggregated.dropna(subset=[x_metric, y_metric])
    if aggregated.empty:
        raise RuntimeError("No rows remain after aggregating and dropping missing plot metrics")

    return aggregated


def is_better_or_equal(left: float, right: float, goal: str) -> bool:
    if goal == "min":
        return left <= right
    return left >= right


def is_strictly_better(left: float, right: float, goal: str) -> bool:
    if goal == "min":
        return left < right
    return left > right


def compute_pareto_frontier(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    x_goal: str,
    y_goal: str,
) -> pd.DataFrame:
    frontier_indices: list[int] = []

    for idx, candidate in df.iterrows():
        dominated = False
        for other_idx, other in df.iterrows():
            if idx == other_idx:
                continue

            x_better_or_equal = is_better_or_equal(other[x_metric], candidate[x_metric], x_goal)
            y_better_or_equal = is_better_or_equal(other[y_metric], candidate[y_metric], y_goal)
            x_strict = is_strictly_better(other[x_metric], candidate[x_metric], x_goal)
            y_strict = is_strictly_better(other[y_metric], candidate[y_metric], y_goal)

            if x_better_or_equal and y_better_or_equal and (x_strict or y_strict):
                dominated = True
                break

        if not dominated:
            frontier_indices.append(idx)

    frontier = df.loc[frontier_indices].copy()
    frontier = frontier.sort_values(by=x_metric, ascending=(x_goal == "min")).reset_index(drop=True)
    return frontier


def annotate_points(ax: plt.Axes, df: pd.DataFrame, x_metric: str, y_metric: str) -> None:
    for _, row in df.iterrows():
        ax.annotate(
            str(row["experiment_name"]),
            (row[x_metric], row[y_metric]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.85,
        )


def series_marker_label(row: pd.Series) -> str:
    if bool(row.get("is_fullframe", False)):
        return "Повний кадр"
    tracker = METRIC_LABELS_UA[str(row.get("tracker_type", "невідомо"))]
    merger = METRIC_LABELS_UA[str(row.get("merge_fn", "невідомо"))]
    return f"{tracker} / {merger}"


def build_marker_map(df: pd.DataFrame) -> dict[str, str]:
    labels = sorted({series_marker_label(row) for _, row in df.iterrows()})
    marker_map: dict[str, str] = {}
    marker_index = 0

    for label in labels:
        if label == "Повний кадр":
            marker_map[label] = "*"
            continue

        marker_map[label] = MARKERS[marker_index % len(MARKERS)]
        marker_index += 1

    return marker_map


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    validate_columns(df, ["experiment_name", args.x_metric, args.y_metric])
    df = filter_by_tracker(df, args.tracker)

    aggregated = aggregate_experiments(df, args.x_metric, args.y_metric)
    frontier = compute_pareto_frontier(
        aggregated,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        x_goal=args.x_goal,
        y_goal=args.y_goal,
    )
    plot_df = frontier.copy() if args.non_dominated_points else aggregated

    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("tab10")
    padding_groups = sorted(
        plot_df["padding_group"].astype(str).unique(),
        key=padding_sort_key,
    )
    color_map = {
        padding_group: cmap(index % cmap.N)
        for index, padding_group in enumerate(padding_groups)
    }
    marker_map = build_marker_map(plot_df)

    regular_points = plot_df[~plot_df["is_fullframe"]]
    fullframe_points = plot_df[plot_df["is_fullframe"]]

    for _, row in regular_points.iterrows():
        ax.scatter(
            row[args.x_metric],
            row[args.y_metric],
            s=80,
            color=color_map[str(row["padding_group"])],
            marker=marker_map[series_marker_label(row)],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

    for _, row in fullframe_points.iterrows():
        ax.scatter(
            row[args.x_metric],
            row[args.y_metric],
            s=130,
            color="gold",
            marker="*",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.95,
            zorder=4,
        )

    ax.plot(
        frontier[args.x_metric],
        frontier[args.y_metric],
        color="black",
        linewidth=1.5,
        linestyle="--",
        marker="o",
        markersize=4,
        label="Апроксимована межа Парето",
        zorder=3,
    )

    if args.annotate == "all":
        annotate_points(ax, plot_df, args.x_metric, args.y_metric)
    elif args.annotate == "frontier":
        annotate_points(ax, frontier, args.x_metric, args.y_metric)

    ax.set_xlabel(metric_label(args.x_metric))
    ax.set_ylabel(metric_label(args.y_metric))
    ax.grid(True, alpha=0.3)

    padding_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_map[padding_group],
            markeredgecolor="black",
            markersize=8,
            label=f"Відступ={padding_group} пкс",
        )
        for padding_group in padding_groups
        if padding_group != "невідомо"
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="gold" if label == "Повний кадр" else "white",
            markeredgecolor="black",
            markersize=10 if label == "Повний кадр" else 8,
            label=label,
        )
        for label, marker in marker_map.items()
    ]

    pareto_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        marker="o",
        markersize=4,
        label="Апроксимована межа Парето",
    )

    combined_handles = [pareto_handle] + padding_handles + marker_handles
    ax.legend(
        handles=combined_handles,
        fontsize=9,
        loc="lower left",
        ncol=2,
        borderaxespad=0.8,
        framealpha=0.9,
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()