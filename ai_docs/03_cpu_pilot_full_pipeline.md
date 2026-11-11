# Stage 3 CPU pilot — joint segment calibration

## Purpose and prerequisites

Run Stage 3 only after Stages 1 and 2. Read
`final_experiments/FINAL_EXPERIMENTS_MANIFEST.md` and the current Stage 2
diagnostic report first.

The primary models are YOLO26n and ROI-SSD-MobileNetV3. ROI-SSD/VGG16 is an
optional experiment and must use its own prepared output directory. The current
desktop target is Intel Core i9-9900KF, but every run detects and records the
actual CPU and software environment instead of assuming a particular CPU model.

Stage 3 now selects `operational_conf` and padding jointly:

```text
smoke
  -> calibration (operational_conf x padding)
  -> freeze selected configuration
  -> fresh CPU cost calibration
  -> FF / SEG / CA / FULL-TECH
  -> review report and freeze Stage 3 parameters
```

The optional `confidence` and `padding` stages are diagnostics, not the final
selection procedure.

## Threshold semantics

`evaluation_conf` defaults to and, for final calibration, must equal `0.01`.
Detections at or above this floor feed mAP@0.5, mAP@0.5:0.95, and Recall
evaluation.

Each calibration cell has its own `operational_conf`. Only detections at or
above that threshold update previous-localization state and form next-frame
segments. A single detector invocation and shared NMS output feed both
branches; evaluation-only filtering and metric calculation are outside
`process_frame` timing.

The `confidence` stage remains available as an optional detector-level F1
diagnostic. Its F1-optimal value is **not** automatically the final Stage 3
operational threshold. The `padding` stage remains an optional fixed-confidence
padding diagnostic and does not freeze the final Stage 3 padding.

## Joint calibration

Each model's exact grids are stored in its source YAML under:

```yaml
calibration:
  operational_conf_values: [...]
  padding_values: [...]
```

The prepared config snapshots are hash-locked. Primary grids:

| Model | Operational confidence | Padding |
|---|---|---|
| YOLO26n | 0.01, 0.05, 0.10, 0.20, 0.35 | 64, 96, 128, 160, 192, 224 |
| ROI-SSD-MobileNetV3 | 0.01, 0.05, 0.10, 0.20, 0.25, 0.35, 0.50 | 64, 96, 112, 128, 144, 160, 192, 224 |

For each model, the runner uses the same pinned pilot manifest for:

1. One full-frame (FF) baseline at evaluation confidence 0.01.
2. One SEG-only run per operational-confidence/padding pair, using Static /
   previous localization, hold=0, periodic full-frame interval=10, and merge
   disabled.

The fixed gates are:

```text
mAP50_retention = mAP50_SEG / mAP50_FF >= 0.95
gt_roi_coverage_mean >= 0.98
```

Only configurations passing both gates are eligible. Select the minimum
`tensor_area_ratio_mean`. If candidates are within 1% of the minimum, break ties
in order: fewer `candidate_segments_per_frame`, smaller padding, then higher
operational confidence. Runtime is not a selection criterion.

If no cell passes, record `CALIBRATION_GATE_FAILED`, list the nearest candidates
as diagnostics only, and block cost/pipeline for that model. Do not select the
largest padding or adjust gates after viewing results.

Each model writes:

```text
calibration/<model>/
    calibration_sweep.csv
    calibration_gate_matrix.csv
    selected_operational_config.json
    selected_operational_config.sha256
    calibration_report.md
    full_frame_baseline.json
```

Reuse of a saved selection is allowed only after validating its receipt, sweep
and baseline hashes, checkpoint SHA-256, config SHA-256, pilot-manifest
SHA-256, evaluation threshold, complete config grid, and the selection recomputed
from the sweep. Execution-code hashes recorded on every sweep row must agree;
the selected file may omit those hashes in older runs because the hash-locked
sweep remains the source of that provenance.

## Cost calibration

Run cost calibration only after a passing joint calibration. It profiles actual
network input shapes and records K, c, tau, R², MAE, RMSE, warmups, repetitions,
evaluation/operational confidence, selected padding, checkpoint/config/manifest
hashes, and runtime environment.

The timing boundary is the model forward plus detector-internal postprocessing
at the fixed evaluation confidence. Operational thresholding, tracker updates,
and metrics are not timed. The operational threshold does not alter this
timed path. Pipeline execution rejects a cost profile whose frozen selection,
provenance, or detected CPU/software environment differs.

Recorded runtime metadata includes CPU model, physical/logical cores, PyTorch
intra/inter-op threads, OpenCV threads, OS, Python, and PyTorch. The current
desktop is expected to be an Intel Core i9-9900KF; the report always displays
the detected machine.

## Pipeline

`pipeline` runs FF, SEG, CA, and FULL-TECH using only the passing,
provenance-validated joint calibration plus the matching fresh cost profile.
It rejects a different evaluation confidence, operational confidence, padding,
checkpoint, config, manifest, or runtime environment.

- **FF:** full frame every frame.
- **SEG:** previous-localization segments, no merge.
- **CA:** same segment formation with `cost_affine` merge.
- **FULL-TECH:** periodic keyframes or empty segments fall back to FF;
  otherwise compare predicted merged-segment cost with predicted FF cost and
  process the cheaper option.

Frame logs retain candidate and post-merge segment counts, NN calls, source
processed area, actual tensor pixels/shapes, fallback reasons, predicted
selector costs/decision, and timing breakdowns. Reports distinguish the
FULL-TECH selector's logical choice from actual CA/FULL-TECH workload; equal
tensor pixels and NN calls must not be interpreted as selector savings based
only on timing differences.

Debug SEG exports must be reviewed for at least 20 frames. Do not proceed if
coordinate remapping, class IDs, clipping, or duplicate handling is incorrect.

## Preparation and commands

Use a fresh output directory; preparation refuses to overwrite a non-empty one.
The following commands assume the project virtual environment and repository
root in PowerShell. Change `--output-dir` consistently if you choose another
fresh location.

```powershell
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.prepare --output-dir final_experiments\pilot_joint_20261109 --models yolo26n roi_ssd_mobilenet
```

Run a short coordinate/integration smoke before calibration. The explicit
operational threshold here is for smoke only:

```powershell
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models yolo26n roi_ssd_mobilenet --stage smoke --max-frames 20 --operational-conf 0.10
```

Joint calibration may be run per model so the results can be reviewed
independently:

```powershell
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models yolo26n --stage calibration
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models roi_ssd_mobilenet --stage calibration
```

Only after both models pass:

```powershell
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models yolo26n roi_ssd_mobilenet --stage cost --cost-repetitions 8
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models yolo26n roi_ssd_mobilenet --stage pipeline
```

Alternatively, after a successful smoke, a fresh output directory can use one
command for calibration -> cost -> pipeline. A provenance-valid calibration
selection may be reused; cost is still measured afresh:

```powershell
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models yolo26n roi_ssd_mobilenet --stage all --cost-repetitions 8
```

Optional diagnostics:

```powershell
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models yolo26n roi_ssd_mobilenet --stage confidence
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models yolo26n --stage padding --operational-conf 0.10
```

The optional VGG16 experiment uses a distinct directory and does not gate the
primary models:

```powershell
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.prepare --output-dir final_experiments\pilot_roi_ssd_vgg16 --models roi_ssd
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_roi_ssd_vgg16 --models roi_ssd --stage smoke --max-frames 20 --operational-conf 0.10
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_roi_ssd_vgg16 --models roi_ssd --stage calibration
```

## Outputs and acceptance

The prepared output contains the pinned pilot and final-evaluation candidate
manifests plus exact config snapshots. Calibration artifacts are under
`calibration/<model>/`. Main result files include:

```text
pilot_report.md
calibration/<model>/selected_operational_config.json
calibration/<model>/calibration_sweep.csv
cost_profiles_cpu.csv
pipeline_summary.csv
frame_level_metrics.csv
debug_frames/
configs/
```

The pilot report records the final evaluation confidence, jointly selected
operational confidence and padding, fresh CPU K/c/tau, all four pipeline modes,
and detected runtime environment.

Stage 3 is complete only when each primary model has a passing joint
calibration, frozen operational confidence and padding, a matching fresh CPU
cost profile, completed FF/SEG/CA/FULL-TECH runs, correct coordinates/classes,
complete frame metrics, a clean full-pilot run, and a video-disjoint final
evaluation candidate. Any calibration gate failure blocks Stage 4. Do not
change pilot-selected parameters during Stage 4.
