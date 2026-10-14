from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

models = ["YOLO26n", "ROI-SSD-MobileNet"]
all_changes = [25.1, 37.8]
stable_changes = [12.9, 25.9]
counts = [[123, 185], [63, 127]]
x = np.arange(2)
fig, ax = plt.subplots(figsize=(10.8, 6.5))
fig.patch.set_facecolor("white")
width = 0.30
for offset, values, color, label, nums in [
    (-width / 2, all_changes, "#3975A8", "Усі зміни оптимального рішення", counts[0]),
    (width / 2, stable_changes, "#E49B39", "Статистично стійкі зміни рішення", counts[1]),
]:
    bars = ax.bar(x + offset, values, width, color=color, label=label, zorder=3)
    for bar, value, count in zip(bars, values, nums):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.65,
                f"{value:.1f}%".replace(".", ",") + f"\n({count} із 490)",
                ha="center", va="bottom", fontsize=12, linespacing=1.4)

ax.set_xticks(x, models)
ax.set_ylabel("Частка пар сегментів, %", labelpad=12)
ax.set_ylim(0, 45)
ax.set_yticks(np.arange(0, 46, 5))
ax.set_xlim(-0.65, 1.65)
ax.grid(axis="y", color="#DCE1E5", linewidth=0.7, zorder=0)
ax.tick_params(axis="both", length=0, pad=9)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#AAB2BA")
fig.suptitle(
    "Частка пар сегментів зі зміною оптимального рішення\n"
    "про об’єднання між Core i7-11700K та Raspberry Pi 5",
    fontsize=16, y=0.97, linespacing=1.4,
)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False,
          ncol=1, fontsize=11)
fig.subplots_adjust(left=0.11, right=0.97, top=0.79, bottom=0.24)
for extension in ["png", "svg", "pdf"]:
    fig.savefig(OUT / f"platform_decision_changes.{extension}", dpi=300, facecolor="white")
plt.close(fig)
