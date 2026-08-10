import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DEFAULT_PATTERN = os.path.join(
    BASE_DIR,
    "benchmark_results",
    "**",
    "adaptive_tau_*.csv",
)


def _discover_latest_csv():
    candidates = glob.glob(DEFAULT_PATTERN, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No adaptive tau logs found with pattern: {DEFAULT_PATTERN}"
        )
    return max(candidates, key=os.path.getmtime)


def _to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_adaptive_tau(csv_path):
    frame_indices = []
    tau_current = []
    tau_min = []
    tau_max = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frame_idx", "tau_current", "tau_min", "tau_max"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV file is missing required columns: {sorted(missing)}"
            )

        for idx, row in enumerate(reader):
            frame_idx = row.get("frame_idx")
            t_cur = _to_float(row.get("tau_current"))
            t_min = _to_float(row.get("tau_min"))
            t_max = _to_float(row.get("tau_max"))
            if frame_idx is None or None in (t_cur, t_min, t_max):
                continue

            try:
                x_value = int(str(frame_idx).strip())
            except ValueError:
                x_value = idx

            frame_indices.append(x_value)
            tau_current.append(t_cur)
            tau_min.append(t_min)
            tau_max.append(t_max)

    if not tau_current:
        raise RuntimeError(f"No valid rows to plot in: {csv_path}")

    return frame_indices, tau_current, tau_min, tau_max


def _build_axis_label(_x_values):
    return "Індекс кадру"


def main():
    parser = argparse.ArgumentParser(
        description="Plot adaptive tau over time with min/max bounds."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to adaptive_tau_*.csv. If omitted, the latest file is auto-detected.",
    )
    parser.add_argument(
        "--fixed-size-tau",
        type=float,
        default=None,
        help=(
            "Optional fixed-size profiling tau value. "
            "If provided, it is shown as a gray dotted line."
        ),
    )
    args = parser.parse_args()

    csv_path = args.csv if args.csv else _discover_latest_csv()
    csv_path = os.path.abspath(csv_path)

    x_values, tau_current, tau_min, tau_max = _load_adaptive_tau(csv_path)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        x_values,
        tau_current,
        color="tab:blue",
        linewidth=2,
        label="Поточне tau",
    )
    ax.plot(
        x_values,
        tau_min,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Мінімальна межа τ",
    )
    ax.plot(
        x_values,
        tau_max,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Максимальна межа τ",
    )

    if args.fixed_size_tau is not None:
        ax.axhline(
            y=args.fixed_size_tau,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Попередньо профільоване τ",
        )

    ax.set_title("Динаміка адаптивного порогу злиття τ під час обробки відеопотоку")
    ax.set_xlabel(_build_axis_label(x_values))
    ax.set_ylabel("Значення τ")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
