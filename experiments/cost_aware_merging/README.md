# Cost-aware segment merging

Run from the repository root. The same command works on a desktop CPU or a
Raspberry Pi 5 CPU; each machine profiles its own model latency before making
predictions. The configured dataset supplies real ImageNet-VID frames, while
rectangle placement is synthetic. The experiment resizes each source frame to
the configured 640×640 canvas once, then crops rectangles without resizing
their content. The model adapters may round input dimensions to their stride;
both the crop and actual tensor shapes are recorded.

```powershell
python -m experiments.cost_aware_merging.run --config config/imagenet-vid-roissd.yaml --output outputs/cost_aware_roissd_cpu
python -m experiments.cost_aware_merging.run --config config/imagenet-vid-yolo.yaml --output outputs/cost_aware_yolo_cpu
```

The model checkpoint and dataset paths come from the training YAML. The command
loads the model before scanning the dataset. ROI-SSD uses its checkpoint under
`trained_models/<task_name>/<ckpt_name>`; YOLO uses `yolo_weights`, resolving it
under `trained_models/<task_name>` when needed. A missing checkpoint stops the
run immediately.

To tune the experiment without changing the training configuration, pass a
small YAML mapping with `--experiment-config`. For example:

```yaml
frame_count: 4
calibration_frames: 2
pair_count: 40
warmup: 30
calibration_repetitions: 30
repetitions: 30
timing_mode: inference_only
seed: 20261005
canvas_hw: [640, 640]
geometric_gamma: 1.4
simple_iou: 0.01
simple_distance: 40
progress_interval_s: 15
```

The command logs its seven stages to the console. Warmup, frame loading, and
calibration show completed counts, elapsed time, and estimated time remaining.
Pair measurement reports each completed pair and gives repetition progress
when a pair takes longer than `progress_interval_s`.

`shapes` controls the candidate rectangle shapes and `calibration_shapes`
controls the separately profiled tensor shapes. Each entry is `[width,height]`.
The default candidates include 32-pixel sides and the representative shapes
listed in the design document. The larger calibration shapes cover common
merged rectangles up to the default canvas. Override `calibration_shapes` when
changing the canvas. The nearest profiled tensor shape in log height/width
space supplies a lookup estimate when there is no exact entry.

`--pairs <previous-run>/pairs.json` replays the same rectangles on another
machine. Both runs must use the same canvas. Dataset frames are selected from
the configured split with the seed; use identical dataset/configuration and
seed to match frame content. Calibration and evaluation use disjoint frames
and separate timing calls. The oracle comes only from evaluation timings.

Outputs:

| File | Contents |
| --- | --- |
| `metadata.json` | Resolved config, paths, platform, frame IDs, model settings, timing boundary, preprocessing, seed |
| `calibration_raw.csv` | Independent profile timings and requested/actual shapes |
| `calibration.json` | Affine K, c, τ, fit metrics and shape lookup |
| `pairs.json` | Portable rectangle definitions |
| `pair_observations.csv` | Every timed separate/merged repetition and randomized order |
| `pairs_raw.csv` | Pair geometry, actual/predicted latencies, decisions and correctness |
| `summary.csv`, `summary.json` | Decision confusion matrices and effective latency by policy |
| `equal_area_shapes.json` | Same-area shapes and measured latency spread |
| `*.png` | Boundary, prediction, policy and shape heatmap plots |

`inference_only` times the existing adapter's model call on a prepared tensor;
`detector_call` includes adapter preprocessing and postprocessing. Disk I/O and
the full-frame canvas resize are excluded from both. The existing simple
merger is called with configurable IoU and center-distance thresholds. The
geometric-area policy uses the repository's `simple_roi_merge_v2` with its
configurable area ratio. All cost predictions are calibrated on the current
machine, and neither estimator reads pair evaluation latencies.

Run lightweight logic tests with:

```powershell
python -m unittest discover -s experiments/cost_aware_merging/tests -v
```
