# ROI-SSD 128x128 diagnosis

This experiment implements `ai_docs/02_diagnose_roissd_128_anomaly.md` using the
updated stage-1 **ROI specialists**, never a filesystem search for weights:

- ROI-SSD: `best-imagenet-vid-roissd/imagenet26022026stage3.pt`.
- ROI-SSD-MobileNet: `best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt`.

Selections/configs are read from the role manifest linked by
`final_experiments/FINAL_EXPERIMENTS_MANIFEST.md`. Full-frame specialist weights
and penalized intermediate candidates are not substituted.

## Commands

From repository root:

```powershell
.venv/Scripts/python.exe -m unittest discover -s experiments/roissd_128_anomaly/tests -v
.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.run --prepare-only
.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.run --device auto --max-samples 4 --sizes 96 128 160 --output final_experiments/diagnostics_128_smoke
.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.run --device auto
.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.report
```

Use `--device cpu --threads 4` for CPU-only execution. `auto` selects CUDA when
available. The smoke output is explicitly marked insufficient for stage-3
clearance. A complete run uses the existing 256-image canonical subset and all
17 square sizes, 96..160 with step 4, for both checkpoints. Three additional
rectangular structural probes check H/W bookkeeping.

## Data and reproducibility

`data/selection_lock.json` binds the actual checkpoint bytes, stage-1 manifests,
config snapshots and class order. `data/subset.jsonl` preserves the 256 source
examples/GT, relative source paths, label hashes and image hashes. The external
dataset root comes from the audited dataset; image pixels are not duplicated
into this repository. Missing/changed images or labels fail instead of falling
back to a different dataset. `data/configs/` holds the original selected ROI
inference configs. `data/synthetic_example.json` supports dataset-free tests.

Subsequent runs use the lock and verify checkpoint/config/subset hashes; they do
not silently follow later recommendation changes. To study a different model
selection, archive this experiment's data and outputs and make a distinct lock.

Preprocessing uses the actual project `FixedSizeTestTransform`, not a separate
crop implementation. Per-image seeds are reset before each transform, giving
the same source and jitter draws for each model and size. Output tensors are
asserted to match requested sizes exactly. Hashes cover tensors and GT; both
models must agree. Several actual crop PNGs are stored for visual inspection.
GT visibility can change with size, so GT totals are recorded for every size.

## Observation and assertions

Scoped forward hooks observe active blocks and classification/regression heads.
Scoped wrappers observe the actual model's default-box generation, decoding and
NMS calls. Nothing changes production model source, weights, head selection,
regression, scoring or NMS. Observers are removed on exit, including failures.

Every forward checks localization slots = classification slots = generated
anchors before decoding. Finite outputs, label ranges, coordinate bounds and
requested/actual dimensions are checked too. A failure stops that model's sweep
and records the current heads/blocks and traceback; other family diagnostics can
still produce useful evidence.

The independent anchor reference checks center/size ordering and records
pre-clipping versus actual clamped geometry. Per-GT IoUs and positive counts are
computed before NMS without predicted confidence. Forced training assignment is
reported separately from simple pairwise IoU-threshold counts. Per-head maximum
foreground confidence and best anchor/decoded IoU help distinguish geometry
from prediction quality. Class AP uses the exact existing project metric
functions from `tools/infer.py`, avoiding import-time CLI/device side effects.

## Outputs

The full run writes `final_experiments/diagnostics_128/`:

- `diagnostic_report.md`, including 96/128/160 comparisons and all structural boundaries.
- `size_sweep_metrics.csv`, `internal_shapes.csv`, `anchor_geometry.csv`, `anchor_matching.csv`.
- `per_class_ap.csv`, `anchor_matching_per_gt.jsonl`, `per_example.jsonl`.
- `preprocessing.csv`, `rectangular_probes.csv`, `run_metadata.json`.
- `errors_or_assertions.log` (empty on success).
- Three PNG plots and sample crop images.

Raw decoded box slots differ from pre-NMS class candidates: the latter count is
after the model's confidence/top-k filtering. Both are recorded. Post-NMS count
is after the final detection cap; pre-cap NMS survivors are also recorded.

The generated classification is evidence-based and conservative: an accuracy
valley with consistent counts does not prove an implementation bug. No causal
depth/head intervention or automatic retraining is performed. If a bug is
localized, preserve these before-results before making a separately reviewed
minimal fix and writing after-results to another output directory.

## Stage 2b: forced-depth comparison

```powershell
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.forced_depth --device auto
```

This separate probe uses the same locked MobileNet ROI checkpoint and all 256
stage-2 examples at 128, 132, 136, 140 and 144 px. At each size it compares the
unmodified current policy with scoped instance-only overrides to L_R-1 and
L_R+1. No model source, architecture, checkpoint or production policy is changed.
Spatial convolution failures are recorded as physically impossible; unrelated
runtime failures stop the experiment rather than becoming feasibility results.

The probe verifies every tensor/GT hash against
`final_experiments/02_diagnostics_128/preprocessing.csv`, reproduces the five
current-policy stage-2 baselines within 1e-6, and checks prediction/anchor slots
on every forward. All modes at one size reuse the same preprocessed tensors.
Feature shapes include batch and channels; head IDs are zero-based.

Outputs in `final_experiments/diagnostics_128/` are `forced_depth_probe.csv`,
`forced_depth_metadata.json`, and the accompanying `forced_depth_report.md`.
The CSV uses [0,1] metrics, class-macro AP/recall from the existing evaluator,
pooled post-NMS/detection-cap mean/max confidence (blank if no detections),
and the mean per-image unthresholded foreground maximum for comparison with
stage 2. Best-anchor IoU statistics include every visible GT before confidence
filtering. The metadata remains RUNNING on failure; only all 15 completed or
explicitly impossible configurations produce COMPLETE.

## Stage 2c: boundary and rectangular validation

```powershell
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.depth_boundary --device auto
```

The separate stage-2c runner reuses the unchanged stage-2b evaluator, locked
checkpoint, all 256 source examples and exact preprocessing/settings. Shape
notation is **height x width**. It compares depths 4/5 at 129, 130 and 131 square
pixels and at 128x160, 160x128, 132x160, 160x132, 136x192, 192x136,
140x192, 192x140, 144x192 and 192x144. It also reruns depths 4/5 at the five
stage-2b square sizes and depth 6 at 144x144 as regression controls: 37
configurations total. Every control must reproduce the saved input hashes,
metrics and head/anchor structure within 1e-6 before the run can complete.

Outputs are `depth_boundary_validation.csv`, `depth_boundary_metadata.json`,
`depth_boundary_inputs.jsonl` (per-source tensor/GT hashes) and the accompanying
`depth_boundary_report.md` in `final_experiments/diagnostics_128/`.
All depth pairs reuse identical tensors and GT; every forward checks prediction
and anchor counts. Production/model sources and weights remain unchanged.
Source drift or runtime/structural failures stop the run, leaving metadata
RUNNING instead of silently skipping configurations.

## Stage 2d: all five depth transitions

```powershell
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.all_thresholds --device auto --threads 4
```

`all_thresholds.py` evaluates both adjacent depths at every requested local
square point around 32/64/96/140/268, with the extra 124/127/128/129 and
240..280 controls. Rectangles use aspect ratios 1.25 and 1.5, both orientations,
and nearest-integer-pixel rounding (half up, never stride rounding). Rectangular
minima are 31/32/33, 63/64/65, 95/96/97, 127/128/129/139/140/141 and
255/256/257/267/268/269. No production policy or earlier experiment file changes.

Accuracy reuses the immutable stage-2b evaluator on all 256 locked examples.
Earlier stage-2c shapes/depths serve as regression controls. Actual tensor/GT
hashes are saved per sample and shape. Only spatial convolution failures count
as INFEASIBLE; other errors stop the run.

Every feasible configuration has synchronized device wall-time inference
timing: batch 1, six warm-ups then 30 repeats, cycling sample IDs 0/85/170,
alternating the depth order each iteration. Shape-only loc/cls/anchor guards
remain enabled inside the timing (no duplicate softmax/IoU observer); crop,
H2D transfer and model construction are excluded. CPU additionally measures
square and all four rectangular formats at minima 32/64/96/128/140/256/268,
with four intra-op and one inter-op thread. Desktop CPU timings are not Pi
predictions; NMS/candidate counts depend on the timed inputs.

Outputs are written to `final_experiments/depth_policy_validation/`:
`all_thresholds_results.csv`, `all_thresholds_metadata.json`,
`latency_summary.csv`, `input_hashes.jsonl`, and accompanying
`boundary_summary.csv` / `all_thresholds_report.md`. Metadata includes the
entire shape plan, source/checkpoint/config hashes, timing settings and hardware.
COMPLETE is only recorded after every planned configuration is evaluated or
explicitly INFEASIBLE. A descriptive tolerance of 0.5 percentage points for
both AP and recall distinguishes practical equivalence from larger shifts;
it is not a significance test or an automatic policy approval.

Validate the completed experiment and print all paired accuracy/CPU evidence:

```powershell
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.threshold_evidence
```

This checks the full requested plan, per-forward counts, per-sample input
hashes, saved latency repetitions, result hashes and unchanged stage-2c
artifacts/sources. The report and boundary summary are evidence-based decisions,
not automatically applied production thresholds.

## Stage 2e: geometry-selected policy and object ratios

```powershell
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.feature_map_policy --device auto --threads 4
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.feature_map_analysis
```

This experiment evaluates legacy, maximum physically feasible depth, the first
actual 1x1 feature map, and the stage-2d adjacent reference on 20 requested
squares plus 16 oriented rectangles. For each shape all existing depths 1..6
are probed using real forwards; no pixel threshold is used for the automatic
policies. No additional blocks or heads are constructed. A spatial convolution
failure terminates feasible probing; other failures stop the experiment.

The user explicitly approved the intervening threshold-only change in the
production model source. The experiment leaves that source untouched: it
verifies the old Git blob against the stage-2d hash (including Windows CRLF
checkout normalization), proves AST equality outside the depth-selection
method, records both hashes and the diff, and scopes every selected depth.
`legacy` always means the original 32/64/96/140/268, not the edited thresholds.
Any additional source drift fails.

Adjacent reference uses stage-2d's predeclared 0.5-percentage-point AP/recall
practical-equivalence tolerance, preferring the lower depth on equivalence.
It is a retrospective local oracle, not a deployment rule or global upper
bound over all six depths. Each selected adjacent depth is reevaluated and
checked against stage 2d; all tensor/GT hashes must match.

Unique depth evaluations are shared by policies choosing the same depth.
Per-object geometry logs every retained GT after project crop/clipping.
Per-depth localization uses same-class post-NMS boxes and exact project IoU,
including one-to-one, score-ordered evaluator matching. Max same-class
confidence and best predicted IoU are separate diagnostics: a GT may have
high IoU but fail because another GT took that prediction. A class with no
detections has IoU/confidence zero, meaning actual absence, not missing data.
Predictions and per-depth per-object metrics are preserved as JSONL.

Paired comparisons separately log stage-2d adjacent depths, legacy-to-first-1x1,
and first-1x1-to-maximum. Ratio bins and correlation analyses include all
adjacent pairs, single added 1x1 heads, and repeated coarse-head additions.
Bin recall is micro-average one-to-one GT success, not macro evaluator recall.
Repeated source objects across shapes are exposures, not independent GT:
no independence-based p-values or causal claims are made. Within-shape-centered
Pearson and primary-GT-only sensitivity checks are included; empty bins and
undefined constant correlations are explicitly marked, not imputed.

CPU latency covers every unique selected/reference depth at every shape:
six warm-ups, 30 repeats, batch 1, sample IDs 0/85/170, alternating order,
four intra-op/one inter-op thread, shape-only prediction/anchor guard enabled.
Timing uses cached geometry decisions; it excludes the one-time depth probe
and preprocessing. This is not a runtime implementation of the geometry
selector and does not include uncached selection cost.

Outputs are in `final_experiments/feature_map_depth_policy/`: required policy
summary/per-shape/per-object/bins/latency/metadata/report files, five plots,
per-object paired comparisons, geometry probes, raw predictions, correlations
and analysis integrity metadata. Production policy and weights are not edited.

## Stage 2f: runtime depth by feature-map geometry

Run the focused post-edit checkpoint and representative Stage-2e regression
check (10 accuracy shapes, 4 CPU sanity shapes):

```powershell
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.quick_regression_check --device auto --threads 4
```

This writes `quick_regression_report.md`, `checkpoint_load_report.txt`,
`stage2e_reproduction.csv`, `cpu_latency_sanity.csv`, and `metadata.json` under
`final_experiments/feature_map_depth_policy_implementation_check/`. It compares
per-sample predictions against the saved Stage-2e outputs and fails on any
metric difference above 1e-6.

Run the checkpoint-compatibility, Stage-2e parity, and CPU profiling validation:

```powershell
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.implementation_validation --device auto --threads 4 --warmup 6 --repeats 30
```

`RoiSSDMobileNet` uses a single runtime policy: `until_1x1`. Its forward pass
computes a feature map, immediately applies the corresponding classification
and localization heads, then advances to the next existing block. The first
actual 1x1 map is processed by its heads and is the final level; no later block
or head is executed. The old threshold helper and legacy runtime mode have been
removed.

Set `record_execution_trace=True` to retrieve the last forward's depth, active
heads, feature shapes, next-level geometry, stop reason, and prediction/anchor
counts through `get_runtime_log()`. No trained layers, parameter shapes, or
state-dict keys are changed.

The Stage-2f validation reuses all 36 Stage-2e shapes and the exact 256
preprocessed sample/GT pairs. It checks strict checkpoint loading, prediction
values and accuracy metrics against Stage 2e within 1e-6, and records batch-1
CPU median/p95 for representative shapes with lightweight hooks that verify
which blocks and heads really ran, including their execution order. The legacy
accuracy and latency columns are frozen references from Stage 2e / the prior
Stage 2f validation; the runtime no longer switches into legacy mode. Results,
the implementation report, and test log are written to
`final_experiments/feature_map_depth_policy_implementation/`. This validation
does not update any trained weights. Stopping at 1x1 is a policy boundary;
some padded convolutions could mathematically produce another 1x1 output, but
those blocks are deliberately not evaluated.

## VGG16 runtime repair and paired regression

The VGG16 iterator uses `features` and `conv5_3_fc`, preserves the learned
conv4_3 L2 scale, and passes unnormalized conv4_3 to conv5/fc. Next-level
feasibility includes max-pooling geometry. Trace attributes are optional and
do not introduce checkpoint keys; construction preserves the caller's scales.

```powershell
.venv/Scripts/python.exe -m unittest discover -s experiments/roissd_128_anomaly/tests -v
.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.vgg16_fixed_regression
.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.vgg16_latency_regression
.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.vgg16_fullframe_checkpoint_regression
```

Results are written to `final_experiments/roissd_vgg16_fixed_regression/`.
The paired check executes historical VGG16 source at revision `06b0bf4` and
the current implementation using the same locked ROI checkpoint, 256 inputs,
GT, crop jitter, thresholds and evaluator. It replays exact saved Stage 2
metrics where available, covers the requested square/rectangular geometry,
and records CPU timings against the same old source. The additional check
loads the manifest's full-frame checkpoint and compares all 256 full-frame
outputs. The earlier failure evidence remains in
`final_experiments/roissd_vgg16_updated_regression/`; its diagnostic runner
never awards PASS after a repair.
