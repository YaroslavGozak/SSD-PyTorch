"""Compare real ROI methods with a geometric Oracle reference (YOLO, K=10)."""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmark_results/vid/imagenet-vid/comparison2026_5_29.csv"
OUT = Path(__file__).resolve().parent
PADS = (16, 32, 64)
with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f)
            if r["model_family"] == "yolo"
            and r["key_frame_interval"] == "10"
            and r["merge_fn"] == "none"
            and r["tracker_dropout_enabled"].lower() == "false"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(9, 5.8), layout="constrained")
selected = []
for tracker, label, color, marker, style, offset in [
    ("static_padding", "Static", "#1764a0", "o", "-", (0, -18)),
    ("sort", "SORT", "#d57900", "s", "-", (-10, 10)),
    ("kalman", "Kalman", "#27824b", "^", "-", (10, 10)),
    ("oracle_gt", "Oracle GT (геометричний еталон)", "#777777", "D", "--", (0, 10)),
]:
    points = {}
    for r in rows:
        if r["benchmark_tracker_type"] != tracker:
            continue
        if tracker == "sort":
            p = int(r["experiment_name"].split("_")[1])
        else:
            col = {"static_padding": "static_pad_x", "kalman": "kalman_pmin",
                   "oracle_gt": "oracle_gt_pad_x"}[tracker]
            p = int(r[col])
        if p not in PADS:
            continue
        xy = (float(r["processed_area_ratio_mean"]), float(r["gt_roi_coverage_mean"]))
        assert all(math.isfinite(v) for v in xy), (tracker, p, xy)
        if p in points and points[p] != xy:
            raise ValueError(f"Conflicting records for {tracker}, p={p}")
        points[p] = xy
        selected.append({"method": tracker, "p": p, "experiment_name": r["experiment_name"],
                         "sweep_name": r["sweep_name"], "model": r["model"],
                         "key_frame_interval": 10, "merge_fn": "none",
                         "processed_area_ratio_mean": xy[0], "gt_roi_coverage_mean": xy[1]})
    assert sorted(points) == list(PADS), (tracker, points)
    ax.plot([points[p][0] for p in PADS], [points[p][1] for p in PADS],
            label=label, color=color, marker=marker, linestyle=style,
            linewidth=1.9, markersize=7, markerfacecolor="white" if tracker == "oracle_gt" else color)
    for p in PADS:
        ax.annotate(f"p={p}", points[p], xytext=offset, textcoords="offset points",
                    ha="center", color=color, fontsize=10)

ax.set_xlabel("Середня частка обробленої площі\n(processed_area_ratio_mean)")
ax.set_ylabel("Середня частка GT-об’єктів із достатнім покриттям ROI\n(gt_roi_coverage_mean)")
ax.set_title("YOLO · K=10 · без злиття ROI та dropout", pad=14)
ax.set_xlim(0.43, 1.01)
ax.set_ylim(0.90, 1.025)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.02))
ax.grid(alpha=0.22, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=True, edgecolor="0.85")
fig.supxlabel("p — padding у пікселях; для Kalman — pmin. Поріг покриття GT: 50%.", fontsize=10)
for extension in ("png", "svg", "pdf"):
    path = OUT / f"gt_coverage_real_methods.{extension}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(path)
with (OUT / "figure_data.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(selected[0]))
    writer.writeheader()
    writer.writerows(selected)
print(f"Validated {len(selected)} source records (4 methods × 3 padding values).")
