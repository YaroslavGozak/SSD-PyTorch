"""Reproduce dissertation section 4.5.3 from saved pairwise measurements.

Run: .venv/Scripts/python.exe scripts/plot_merge_strategy_comparison.py
No inference or policy decisions are rerun. Pair timings are medians of repetitions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("ca_yolo26n_cpu", "YOLO26n", "Core i7"),
        ("ca_yolo26n_pi", "YOLO26n", "Raspberry Pi 5"),
        ("ca_roissd_mobilenet_cpu", "ROI-SSD-MobileNet", "Core i7"),
        ("ca_roissd_mobilenet_pi", "ROI-SSD-MobileNet", "Raspberry Pi 5")]
METHODS = ["no_merge", "existing_simple_merge", "geometric_area", "cost_affine", "cost_shape_lookup", "oracle"]
LABELS = ["Без об’єднання", "Просте об’єднання", "Геометричне", "Cost-affine", "Shape lookup", "Oracle"]
COLORS = ["#b5b5b5", "#9aabba", "#d0b58a", "#54878a", "#8b84a2", "#666666"]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError(f"Invalid decision: {value!r}")
    return value == "True"


def load_run(directory):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    rows = read_csv(directory / "pairs_raw.csv")
    observations = read_csv(directory / "pair_observations.csv")
    grouped = {}
    for item in observations:
        grouped.setdefault(item["pair_id"], []).append(item)
    assert len(rows) == metadata["pair_count"] == 500
    assert len({r["pair_id"] for r in rows}) == len(rows)
    assert set(grouped) == {r["pair_id"] for r in rows}
    sep = np.array([float(r["actual_separate_ms"]) for r in rows])
    merged = np.array([float(r["actual_merged_ms"]) for r in rows])
    assert np.isfinite(sep).all() and np.isfinite(merged).all()
    assert (sep > 0).all() and (merged > 0).all()
    for row, separate, merge in zip(rows, sep, merged):
        raw = grouped[row["pair_id"]]
        assert len(raw) == metadata["experiment_options"]["repetitions"]
        assert len({r["repetition"] for r in raw}) == len(raw)
        np.testing.assert_allclose([separate, merge],
            [np.median([float(r[k]) for r in raw]) for k in ("separate_ms", "merged_ms")], rtol=1e-12)
        ratio = float(row["merged_area"]) / (float(row["r1_area"]) + float(row["r2_area"]))
        assert boolean(row["geometric_area_decision"]) == (ratio <= metadata["experiment_options"]["geometric_gamma"])
        assert boolean(row["oracle_decision"]) == (merge < separate)
        assert not boolean(row["no_merge_decision"])
    decisions = np.array([[boolean(r[f"{m}_decision"]) for m in METHODS] for r in rows])
    effective = np.where(decisions, merged[:, None], sep[:, None])
    np.testing.assert_allclose(effective[:, -1], np.minimum(sep, merged), rtol=1e-12)
    assert (effective[:, -1, None] <= effective + 1e-12).all()
    assert decisions[:, 2].sum() > 0
    summary = {r["method"]: r for r in json.loads((directory / "summary.json").read_text(encoding="utf-8"))}
    assert set(summary) == set(METHODS)
    for j, method in enumerate(METHODS):
        np.testing.assert_allclose(effective[:, j].mean(), summary[method]["mean_effective_latency_ms"], rtol=1e-12)
        assert decisions[:, j].sum() == summary[method]["TP"] + summary[method]["FP"]
    return metadata, sep, effective, decisions


def save(fig, output, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=350, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "merge_strategy_comparison")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("Use at least 100 bootstrap samples")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records, audit = [], []
    for name, model, platform in RUNS:
        directory = ROOT / "outputs" / name
        metadata, sep, effective, decisions = load_run(directory)
        assert metadata["timing_boundary"] == "inference_only"
        assert metadata["architecture"] == ("AMD64" if platform == "Core i7" else "aarch64")
        assert (metadata["model"]["backend"] == "ultralytics") == (model == "YOLO26n")
        if model != "YOLO26n":
            assert "mobilenet" in metadata["model"]["model_config"]
        samples = []
        for start in range(0, args.bootstrap, 250):
            idx = rng.integers(0, len(sep), size=(min(250, args.bootstrap-start), len(sep)))
            samples.append(100 * (1 - effective[idx].mean(axis=1) / sep[idx].mean(axis=1)[:, None]))
        low, high = np.percentile(np.concatenate(samples), [2.5, 97.5], axis=0)
        for j, method in enumerate(METHODS):
            records.append(dict(model=model, platform=platform, strategy=method, n_pairs=len(sep),
                mean_effective_latency_ms=effective[:, j].mean(), no_merge_latency_ms=sep.mean(),
                reduction_percent=100*(1-effective[:, j].mean()/sep.mean()), ci95_low=low[j], ci95_high=high[j],
                source_experiment=f"outputs/{name}", n_merges=int(decisions[:, j].sum())))
        audit.append(dict(source=name, metadata=metadata,
            sha256={f: hashlib.sha256((directory/f).read_bytes()).hexdigest()
                    for f in ("metadata.json", "pairs_raw.csv", "pair_observations.csv", "summary.json")}))
    with (args.output / "merge_strategy_reduction.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "provenance.json").write_text(json.dumps(dict(seed=args.seed, bootstrap_samples=args.bootstrap,
        confidence_interval="paired percentile bootstrap, pairs resampled with replacement", sources=audit), indent=2), encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                        "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.1, 4.5), layout="constrained")
    x = np.arange(4)
    for j in range(1, 6):
        data = [records[i*6+j] for i in range(4)]
        y = np.array([r["reduction_percent"] for r in data])
        errors = np.array([y-[r["ci95_low"] for r in data], [r["ci95_high"] for r in data]-y])
        bars = ax.bar(x+(j-3)*.155, y, width=.145, color=COLORS[j], label=LABELS[j],
            edgecolor="#444444", linewidth=.4, hatch="///" if j == 5 else None,
            yerr=errors, error_kw=dict(elinewidth=.7, capsize=2))
        if j in (3, 5):
            for bar, row in zip(bars, data):
                ax.text(bar.get_x()+bar.get_width()/2, row["ci95_high"]+.18,
                    f'{row["reduction_percent"]:.2f}%', ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x, [f"{m}\n{p}" for _, m, p in RUNS], fontsize=9)
    ax.set_ylabel("Скорочення середнього часу обробки, %")
    ax.set_ylim(0, max(r["ci95_high"] for r in records)+2)
    ax.yaxis.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False, fontsize=9)
    save(fig, args.output, "merge_strategy_reduction")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4), layout="constrained")
    for ax, platform in zip(axes, ("Core i7", "Raspberry Pi 5")):
        for j in range(6):
            data = [r for r in records if r["platform"] == platform and r["strategy"] == METHODS[j]]
            ax.bar(np.arange(2)+(j-2.5)*.13, [r["mean_effective_latency_ms"] for r in data],
                width=.12, color=COLORS[j], label=LABELS[j], edgecolor="#444444", linewidth=.4,
                hatch="///" if j == 5 else None)
        ax.set_xticks([0, 1], ["YOLO26n", "ROI-SSD-\nMobileNet"], fontsize=9)
        ax.set_title(platform)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Середній ефективний час обробки, мс")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False, fontsize=9)
    save(fig, args.output, "merge_strategy_effective_latency")
    report = ["# Рисунок 4.5.3 — порівняння стратегій", "",
        "Використано повні evaluation sets: 500 пар × 30 повторень на конфігурацію. "
        "T_sep і T_merge — медіани вимірювань кожної пари; середній ефективний час — середнє по 500 парах. "
        "Медіани перевірено за pair_observations.csv, рішення взято з pairs_raw.csv, середні перевірено за summary.json.", "",
        f"95% percentile paired bootstrap: {args.bootstrap} вибірок, seed={args.seed}. "
        "Інтервали характеризують варіативність набору пар за фіксованих рішень і медіан; "
        "не враховують повторне навчання cost model або залежність пар одного кадру.", "",
        "Oracle — експериментальна верхня межа: min(T_sep, T_merge), а не придатна до розгортання стратегія.", "",
        "Джерела: " + ", ".join(f"outputs/{n}" for n, _, _ in RUNS) + ". "
        "Вибір підтверджено backend/config, архітектурою, inference_only, повними pairwise observations, "
        "узгодженням summaries і посиланнями first_run/second_run у ca_*_flips/comparison.json. "
        "CPU позначено Core i7 відповідно до постановки завдання; точну модель процесора metadata не містить. "
        "ca_roissd_mobilenet_cpu2 — окремий replay тих самих пар, не джерело опублікованого hardware comparison; "
        "ca_roissd_cpu — інша модель, без MobileNet.", "",
        "У вибраних наборах geometric_area має 84 merges і на кожній парі відповідає "
        "merged_area/(r1_area+r2_area) <= geometric_gamma. Наборів із нульовими geometric merges "
        "серед ca_* не знайдено. Результати не змінювалися; reanalyze_geometric.py не запускався.", "",
        "Розбіжність з описом lookup: lookup_exact_shape_fraction у metadata становить "
        "0.438667 для YOLO та 0.464667 для MobileNet, а не 1. "
        "Збережені lookup shapes іноді відрізняються від tensor shapes. Використано саме збережені "
        "рішення фінальних запусків; нових lookup/interpolation правил не додано.", "",
        "Отримані значення узгоджуються з усіма наведеними sanity checks після округлення. "
        "Автоматично перевірено oracle <= кожна стратегія на рівні пар і середніх.", "",
        "| Конфігурація | Стратегія | Час, мс | Скорочення, % | 95% CI, % | Merges |",
        "|---|---|---:|---:|---|---:|"]
    for r in records:
        report.append(f'| {r["model"]} / {r["platform"]} | {r["strategy"]} | {r["mean_effective_latency_ms"]:.3f} | {r["reduction_percent"]:.2f} | [{r["ci95_low"]:.2f}; {r["ci95_high"]:.2f}] | {r["n_merges"]} |')
    report += ["", "Основний рисунок: merge_strategy_reduction.{pdf,svg,png}; додатковий: "
        "merge_strategy_effective_latency.{pdf,svg,png}. PNG: 350 dpi. "
        "Агреговані дані: merge_strategy_reduction.csv; provenance.json містить metadata та SHA256 джерел.",
        "", "Відтворення: `.venv/Scripts/python.exe scripts/plot_merge_strategy_comparison.py`."]
    (args.output / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(f"Validated all four runs; saved figures, CSV and report to {args.output}")


if __name__ == "__main__":
    main()
