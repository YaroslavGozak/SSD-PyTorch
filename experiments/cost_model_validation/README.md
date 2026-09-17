# Cost model validation

For the current schema-v3 workflow (saved A schedules, robust bootstrap, pooled
sessions and frozen multi-model B pairs), use [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
The older B command examples below require generating `pair_specs.json` first;
the single-command `run_experiments` runner handles initial generation.

This package validates `T(A) = K_t + c_t A` independently of the video benchmark. `A` is always the spatial area read from the prepared model tensor. The two timing boundaries are explicit: `inference_only` and `detector_call`; never combine their rows in one fit.

## CPU smoke test

```bash
python -m unittest discover -s experiments/cost_model_validation/tests -v
python -m experiments.cost_model_validation.collect_experiment_a --config experiments/cost_model_validation/config.example.yaml --output outputs/cost_model_smoke --overwrite
python -m experiments.cost_model_validation.analyze_experiment_a --input outputs/cost_model_smoke/experiment_a_raw.csv --output outputs/cost_model_smoke
```

The example uses the deterministic fake adapter and needs no weights. For Raspberry Pi, set `model.backend: ultralytics`, provide the exact weights path and device, then run the same commands. The adapter imports Ultralytics lazily and does not modify the existing video benchmark.

Raw CSV is append-only and metadata records the model checksum, environment, timing boundary, stride, and warnings. A production run should use at least 30 repetitions and a warmed, stable hardware state.

## Full YOLO Experiment A and B

The repository configuration [config.voc-yolo26n.yaml](config.voc-yolo26n.yaml) uses `trained_models/voc-yolo26n/best.pt`, CPU inference, stride 32, and the `inference_only` timing boundary. Experiment B must use the fitted `tau` from Experiment A; do not enter a paper value manually unless using `--tau-override` explicitly.

Run the commands below from the repository root in PowerShell:

```powershell
python -m unittest discover -s experiments/cost_model_validation/tests -v

python -m experiments.cost_model_validation.collect_experiment_a `
	--config experiments/cost_model_validation/config.voc-yolo26n.yaml `
	--output outputs/cost_model_voc_yolo26n_full `
	--overwrite

python -m experiments.cost_model_validation.analyze_experiment_a `
	--input outputs/cost_model_voc_yolo26n_full/experiment_a_raw.csv `
	--output outputs/cost_model_voc_yolo26n_full

python -m experiments.cost_model_validation.collect_experiment_b `
	--config experiments/cost_model_validation/config.voc-yolo26n.yaml `
	--fit outputs/cost_model_voc_yolo26n_full/linear_fit.json `
	--output outputs/cost_model_voc_yolo26n_full `
	--overwrite

python -m experiments.cost_model_validation.analyze_experiment_b `
	--input outputs/cost_model_voc_yolo26n_full/experiment_b_raw.csv `
	--output outputs/cost_model_voc_yolo26n_full
```

The final command writes `decision_metrics.json`. The raw `experiment_b_raw.csv` contains the paired observations and confidence-aware pair summaries are included in `decision_metrics.json`. Experiment B measures each pair as two separate ROI calls versus one union-rectangle call, randomizing which is measured first on every repetition. A positive paired difference means that merging was faster. Pairs whose bootstrap confidence interval contains zero are reported as `ambiguous` rather than forced into a binary label.

Experiment B uses explicit `merge`, `near_low`, `near_high`, and `separate` quotas, rejects duplicate computational shape keys after preprocessing, and varies horizontal, vertical, diagonal, partial-overlap, and containment geometries. All pair/repetition trials are globally shuffled while each pair retains an exactly balanced first-order schedule. Periodic control shapes compare B latency with the A fit; `experiment_b_metadata.json` records `hardware_state_shift` when a control differs by more than 10%.

The full configuration collects 500 pairs with 20 paired repetitions. On CPU this can take a long time because each repetition performs three model calls. Use a separate output directory for every run; the raw CSV files are append-only and the `--overwrite` flag intentionally starts a new collection.

## ROI-SSD and ROI-SSD-MobileNet

The `roissd` backend supports both repository models. Set `model.model_config` to a training configuration whose `train_params.model` is `roissd` or `roissd-mobilenet`, and set `model.weights` to the checkpoint. Unlike YOLO, these models accept `(images, None)` and use stride 1 by default.

Example:

```powershell
python -m experiments.cost_model_validation.collect_experiment_a `
	--config experiments/cost_model_validation/config.voc-roi-ssd-mobilenet.yaml `
	--output outputs/cost_model_voc_roi_ssd_mobilenet `
	--overwrite
```

For plain ROI-SSD, use the matching example configuration:

```powershell
python -m experiments.cost_model_validation.collect_experiment_a `
	--config experiments/cost_model_validation/config.voc-roi-ssd.yaml `
	--output outputs/cost_model_voc_roi_ssd `
	--overwrite
```

## Experiment B v2 model comparison

Experiment A writes a versioned `linear_fit.json` calibration artifact containing the linear, quadratic, and piecewise models. Experiment B loads those coefficients without fitting on B and evaluates all available rules on the same measured pairs:

- `linear_tau`: direct linear cost comparison, retaining `tau = K_t / c_t` for compatibility;
- `quadratic_direct_cost`: direct comparison of the three quadratic predicted costs;
- `piecewise_direct_cost`: direct comparison using the fitted hinge model and breakpoint.

Run B only after A has completed all repetitions:

```powershell
python -m experiments.cost_model_validation.collect_experiment_b `
	--config experiments/cost_model_validation/config.voc-yolo26n.yaml `
	--fit outputs/cost_model_voc_yolo26n_full/linear_fit.json `
	--output outputs/cost_model_voc_yolo26n_full `
	--overwrite

python -m experiments.cost_model_validation.analyze_experiment_b `
	--input outputs/cost_model_voc_yolo26n_full/experiment_b_raw.csv `
	--output outputs/cost_model_voc_yolo26n_full `
	--fit outputs/cost_model_voc_yolo26n_full/linear_fit.json
```

The B metadata records `schema_version: 2`, the calibration artifact hash, domain counters, order/drift diagnostics, `sampling_basis`/`sampling_tau_pixels`, `measurement_protocol`, and the model metadata. The analyzer automatically resolves calibration from the model snapshot in `experiment_b_metadata.json`, then its referenced artifact, then `linear_fit.json` beside the input CSV. `--fit` explicitly overrides this selection. Recorded relative paths and relocated run folders are supported; discovered files are checked against the recorded hash when one is available.

A piecewise breakpoint enables `area_regime` (`below_breakpoint`/`spans_breakpoint`/`above_breakpoint`) and `metrics_by_area_regime` without requiring `--fit`. The report includes `piecewise_breakpoint_area`, `calibration_reference`, and an explicit `area_regime_unavailable_reason` if no breakpoint can be found. Areas equal to the breakpoint belong to the lower regime.

`decision_metrics.json` contains `model_comparison` for the three rules, each with classification metrics plus `regret_ms_all_pairs` and `regret_ms_determinate_pairs`. The same per-rule breakdown is repeated in `metrics_by_boundary_bin`, `metrics_by_geometry_type`, and, when a breakpoint is available, `metrics_by_area_regime`. Each pair summary includes `effective_areas`, `delta_area`, per-model `predicted_merged_cost_ms`/`predicted_separate_cost_ms`/`predicted_gain_ms`, and the raw per-repetition `M`/`S` observations with their measurement order. Missing quadratic or piecewise coefficients are reported as unavailable rather than estimated from B.

Control records store `predictions.linear`, `predictions.quadratic`, and `predictions.piecewise`, each with predicted latency and relative error `(measured_ms - predicted_ms) / predicted_ms`. Missing models have `available: false` and an explanation. The collector uses the effective shape returned by the timed call. Offline B analysis also reconstructs these predictions for legacy control records and includes them in `decision_metrics.json`; it leaves the original collection metadata unchanged.

## Calibration provenance

New Experiment A collections record weights SHA-256, actual device, backend and package versions, adapter identity, preprocessing (resize, normalization, layout, dtype, stride and postprocessing boundary), Git commit/dirty state, and seed. The analyzer copies this collection-time provenance into `linear_fit.json` and `experiment_a_summary.json`. Use `--metadata` to supply a metadata file stored elsewhere. Experiment B retains the calibration provenance in its metadata.

For historical runs, the analyzer preserves only recorded facts. It does not substitute current package versions or preprocessing for missing historical values. `provenance.complete`, `missing_fields`, and `warnings` identify incomplete records; a new collection is needed to capture values that were never recorded.

Experiment A's `linear_fit.json` reports each model's `fit_metrics` with an explicit `fit_level`: `shape_level` (primary, one point per unique effective shape) and `observation_level` (every raw repetition, kept for noise diagnostics) are both included so model comparison is not silently based on pseudo-replicated observations.
