# Cost model validation

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