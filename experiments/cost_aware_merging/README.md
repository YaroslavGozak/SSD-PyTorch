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

`--pairs <previous-run>/pairs.json` replays the same rectangles across models
and platforms. Both runs must use the same canvas. New pairs files also record
the calibration/evaluation frame manifest, source and resized-canvas SHA-256
hashes, and each pair's evaluation frame assignment. Replay resolves frames by
`parent-directory/filename`, independent of dataset root, OS path separators,
dataset order, and the current seed. The configured dataset/split must contain
those frames; missing, ambiguous, or changed content stops the run. Frame and
pair counts come from the replay file. Repetitions and calibration timings
remain local to the current run. Calibration and evaluation use disjoint frames;
the oracle comes only from evaluation timings.

Generate the pair set once, then copy its `pairs.json` to every target machine:

```powershell
python -m experiments.cost_aware_merging.run --config config/imagenet-vid-roissd.yaml --experiment-config experiments/cost_aware_merging/config.yaml --output outputs/merge_vgg_cpu
python -m experiments.cost_aware_merging.run --config config/imagenet-vid-roissd-mobilenet.yaml --experiment-config experiments/cost_aware_merging/config.yaml --pairs outputs/merge_vgg_cpu/pairs.json --output outputs/merge_mobilenet_cpu
# On Pi 5, use its model/dataset config and the copied pairs.json:
python -m experiments.cost_aware_merging.run --config config/raspberry/imagenet-vid-roissd-mobilenet.yaml --experiment-config experiments/cost_aware_merging/config.yaml --pairs pairs.json --output outputs/merge_mobilenet_pi5
```

Metadata records `pairs_geometry_sha256` and `workload_sha256`; matching workload
hashes certify the same geometry, frame content, and assignments. Sampling uses
the generating run's fitted tau and is frozen on replay. `source_boundary_region`
preserves its labels; `boundary_region` uses the current run's fitted tau.
Legacy pairs files remain supported for geometry replay but do not guarantee
identical frame content; the runner logs a warning.

Compare runs after transferring their output directories to the same machine:

```powershell
python -m experiments.cost_aware_merging.compare_runs outputs/merge_mobilenet_cpu outputs/merge_mobilenet_pi5 --output outputs/merge_hardware_flips
```

This checks matching geometry, frame content, pair coverage, and timing boundary,
then writes `pair_flips.csv` and `comparison.json` with oracle decisions and
latency differences (`separate - merged`). Same checkpoint and preprocessing
comparisons are marked `hardware_candidate`; other comparisons are marked
`cross_model_or_preprocessing`. Flips are observed median sign changes; repeat
runs to distinguish hardware effects from timing noise, especially near zero.

Outputs:

| File | Contents |
| --- | --- |
| `metadata.json` | Resolved config, paths, platform, frame IDs, model settings, timing boundary, preprocessing, seed |
| `calibration_raw.csv` | Independent profile timings, requested/actual shapes, active heads and feature maps |
| `calibration.json` | Affine K, c, τ, fit metrics and shape lookup |
| `pairs.json` | Portable rectangle definitions, frame assignments, frame manifest, geometry hash |
| `pair_observations.csv` | Every timed repetition, exact separate-call order, and execution trace for each r1/r2/merged invocation |
| `pairs_raw.csv` | Pair geometry, actual/predicted latencies, decisions, correctness, frame content hash, and first-repetition execution traces |
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

ROI-SSD execution logging is enabled automatically for both VGG16 and MobileNet.
The `r1_`, `r2_`, and `merged_` columns include `active_depth`, `active_head_count`,
`active_head_indices` (zero-based JSON list), and `feature_maps` (JSON list of
feature names, NCHW shapes, and classification/regression head paths). Depth is
the selected number of feature-map stages, not the backbone's convolution count.
These are captured from features actually consumed by the heads in each forward,
using the stride-rounded tensor. Each feature stage has a classification and a
regression head; `active_head_count` counts stages. Both models select depth
using the minimum tensor side: <=32: 1, <=64: 2, <=96: 3, <=140: 4, <=268: 5,
otherwise 6. Thus a large bounding-area increase may still leave depth unchanged
if the shorter side stays small.

The timed forward includes a small shape-tuple capture; JSON serialization and
file writes happen after timing. No tensor copies or diagnostic extra forwards
are used. `pair_observations.csv` contains each repetition's actual trace;
`pairs_raw.csv` copies the first repetition's trace alongside median timings.
Other model backends record `execution_trace_status=not_applicable` and empty
ROI-SSD fields. Old timings cannot reconstruct actual invocation traces.

For a pair, `geometric_area` merges exactly when
`merged_area / (r1_area + r2_area) <= geometric_gamma` (default 1.4).
`pairs_raw.csv` records this ratio as `geometric_area_ratio`.

Runs made before the `simple_roi_merge_v2` correction incorrectly reported
every geometric pair decision as false: merged clusters were appended inside
the search loop, producing duplicate output boxes. The merger now emits each
completed cluster once and updates its bounding area after each merge.
Recompute the baseline using existing measured timings with:

```powershell
python -m experiments.cost_aware_merging.reanalyze_geometric outputs/cost_aware_roissd_mobilenet_pi5
```

This writes corrected pair decisions, summaries, and source hashes under
`geometric_corrected/`. Original results and plots remain historical; use the
corrected summaries for comparisons. No new calibration or inference is needed.

Run lightweight logic tests with:

```powershell
python -m unittest discover -s experiments/cost_aware_merging/tests -v
```
