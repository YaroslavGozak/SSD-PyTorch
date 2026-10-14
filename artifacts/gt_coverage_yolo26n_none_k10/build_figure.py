"""Generate thesis figure from YOLO K=10, no merge, no dropout records."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmark_results/vid/imagenet-vid/comparison2026_5_29.csv"
OUT = Path(__file__).resolve().parent
with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f)
            if r["model_family"] == "yolo"
            and r["benchmark_tracker_type"] in ("static_padding", "oracle_gt")
            and r["key_frame_interval"] == "10"
            and r["merge_fn"] == "none"
            and r["tracker_dropout_enabled"].lower() == "false"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.2, 5.4), layout="constrained")
selected = []
for tracker, label, color, marker, style in [
    ("static_padding", "Static", "#1764a0", "o", "-"),
    ("oracle_gt", "Oracle GT", "#b65020", "s", "--"),
]:
    points = {}
    pad_col = "static_pad_x" if tracker == "static_padding" else "oracle_gt_pad_x"
    for r in rows:
        if r["benchmark_tracker_type"] != tracker:
            continue
        p = int(r[pad_col])
        xy = (float(r["processed_area_ratio_mean"]), float(r["gt_roi_coverage_mean"]))
        if p in points and points[p] != xy:
            raise ValueError(f"Conflicting records for {tracker}, p={p}")
        points[p] = xy  # Identical duplicated CSV records represent the same point.
    assert sorted(points) == [0, 16, 32, 64], points
    ax.plot([points[p][0] for p in sorted(points)],
            [points[p][1] for p in sorted(points)],
            label=label, color=color, marker=marker, linestyle=style,
            linewidth=1.8, markersize=7)
    for p, (x, y) in sorted(points.items()):
        ax.annotate(f"p={p}", (x, y), xytext=(0, 11 if tracker == "oracle_gt" else -19),
                    textcoords="offset points", ha="center", color=color, fontsize=10)
        selected.append({"method": label, "p": p,
                         "processed_area_ratio_mean": x, "gt_roi_coverage_mean": y})
ax.set_xlabel("Частка обробленої площі\n(processed_area_ratio_mean)")
ax.set_ylabel("Середнє покриття еталонних областей\n(gt_roi_coverage_mean)")
ax.set_title("YOLO26n · K=10 · без dropout · merge_fn=none", pad=13)
ax.set_xlim(0.32, 0.93)
ax.set_ylim(0.900, 1.022)
ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.02))
ax.grid(alpha=0.22, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=True, edgecolor="0.85")
for extension in ("png", "svg"):
    path = OUT / f"figure_4_X_gt_coverage.{extension}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(path)
with (OUT / "figure_data.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(selected[0]))
    writer.writeheader()
    writer.writerows(selected)
print(f"Validated {len(selected)} unique points from {len(rows)} source rows.")
