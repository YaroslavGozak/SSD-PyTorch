# Stage 3 CPU pilot

## Preparation status

`PARTIAL_PIPELINE_SMOKE`. Completed stages: ['coordinate_smoke', 'cpu_cost_calibration', 'pipeline_modes']. Joint confidence × padding calibration remains pending. FF/SEG/CA/FULL-TECH pipeline runs are recorded.

## Locked split

- Pilot: 32 video sequences / 320 labeled frames.
- Final-evaluation candidate: 476 videos / 4744 frames.
- Selection: deterministic seeded sample (`seed=20261112`) from validation videos
  outside the canonical 256-frame validation videos; no prediction metrics were used.
- Video overlap: zero between pilot and final-evaluation candidate.
- Pilot manifest SHA-256: `817f5b145b774dc91ee0fa8d271e02c11d71d70bb50ff13082a0ba9487aa63cb`.
- Candidate manifest SHA-256: `4f36f3455ced1a344ef69329479cd09013934d71aafa8dc800d7f7e5ea6bf6d4`.

The candidate is disjoint from this pilot, but the source validation dataset was
previously reused for model selection/diagnostics. It is therefore not an untouched
external test set. The original 5,064-frame source manifest remains unchanged.

## Selected models

- yolo26n checkpoint SHA-256: `5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027`.
- roi_ssd_mobilenet checkpoint SHA-256: `1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804`.
- Exact source configuration snapshots are in [`configs/`](./configs/).
- `evaluation_conf` is fixed at `0.01` for final Stage 3 quality evaluation.
- Final `operational_conf` and padding are selected jointly by `--stage calibration`
  using each model's config-defined grids and fixed quality gates.
- `--stage confidence` is an optional detector-level diagnostic; its F1-optimal
  threshold does not automatically become the final operational threshold.
- YOLO output cap/NMS are locked to 200 detections and IoU 0.45.
- CPU thread counts are fixed by the runner, and actual CPU/runtime details are
  detected and recorded automatically.

## Smoke check

The 20-frame CPU coordinate smoke is run separately. Its outcome is recorded in
[`smoke_metadata.json`](./smoke_metadata.json); smoke metrics are not pilot results.

## Remaining protocol

1. Run `python -m experiments.cpu_pilot.run_pilot --stage calibration` after a successful smoke.

## Runtime environment

| Property | Value |
|---|---|
| CPU model | Intel64 Family 6 Model 158 Stepping 13, GenuineIntel |
| Physical cores | 8 |
| Logical cores | 16 |
| PyTorch intra-op threads | 8 |
| PyTorch inter-op threads | 1 |
| OpenCV threads | 1 |
| OS | Windows-11-10.0.26300-SP0 |
| Python | 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] |
| PyTorch | 2.12.0+cpu |

## Joint operational-confidence × padding calibration

Final Stage 3 selection requires both fixed quality gates, then minimizes tensor_area_ratio_mean. Within 1% of the minimum, ties use fewer candidate segments/frame, smaller padding, then higher operational confidence.

| Model | Status | evaluation_conf | operational_conf | Padding | Retention | GT coverage | Tensor area ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| yolo26n | CALIBRATION_PASS | 0.01 | 0.1 | 64 | 0.976283491559376 | 1.0 | 1.03040625 |

Artifacts: [yolo26n sweep](calibration/yolo26n/calibration_sweep.csv), [gate matrix](calibration/yolo26n/calibration_gate_matrix.csv), [decision](calibration/yolo26n/selected_operational_config.json), [report](calibration/yolo26n/calibration_report.md).
| roi_ssd_mobilenet | CALIBRATION_PASS | 0.01 | 0.5 | 128 | 0.9836732776019116 | 0.9964285714285714 | 1.1627283935546875 |

Artifacts: [roi_ssd_mobilenet sweep](calibration/roi_ssd_mobilenet/calibration_sweep.csv), [gate matrix](calibration/roi_ssd_mobilenet/calibration_gate_matrix.csv), [decision](calibration/roi_ssd_mobilenet/selected_operational_config.json), [report](calibration/roi_ssd_mobilenet/calibration_report.md).

## CPU smoke (not decision-grade)

Coordinate/class checks passed on 20 frames per model; 20 overlays per model were exported. These metrics are diagnostic only and must not select padding.

| Model | Mode | mAP50 | Recall50 |
|---|---|---:|---:|
| roi_ssd_mobilenet | FF | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | SEG | 0.9793 | 1.0000 |

Thresholds: evaluation_conf=0.01; operational_conf by model is recorded in smoke_metadata.json.

| Model | Evaluation detections | Operational detections | Candidate segments |
|---|---:|---:|---:|
| roi_ssd_mobilenet | 841 | 129 | 60 |

## Fresh CPU cost profiles

Timing boundary: model forward and detector-internal postprocessing at the fixed evaluation_conf. Operational thresholding, tracker updates, and metrics are outside this timing; the operational confidence and padding are recorded as frozen provenance.

| Model | evaluation_conf | operational_conf | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | 0.0100 | 0.1000 | 7.30096 | 0.00015598 | 46806.879 | 0.9917 | 0.3304 | 0.4395 |
| roi_ssd_mobilenet | 0.0100 | 0.5000 | 9.89278 | 0.000144553 | 68437.007 | 0.8760 | 1.5722 | 1.6785 |

## Frozen pilot parameters

YOLO_EVALUATION_CONF = 0.01
YOLO_OPERATIONAL_CONF_FINAL = 0.1
YOLO_PADDING_FINAL = 64
ROISSD_MOBILENET_EVALUATION_CONF = 0.01
ROISSD_MOBILENET_OPERATIONAL_CONF_FINAL = 0.5
ROISSD_MOBILENET_PADDING_FINAL = 128
ROISSD_EVALUATION_CONF = NOT SELECTED
ROISSD_OPERATIONAL_CONF_FINAL = NOT SELECTED
ROISSD_PADDING_FINAL = NOT SELECTED
YOLO_OPERATIONAL_CONF (confidence diagnostic) = NOT RUN
ROISSD_MOBILENET_OPERATIONAL_CONF (confidence diagnostic) = NOT RUN
EVALUATION_CONF_DEFAULT = 0.01
PIPELINE_OPERATIONAL_CONF = provenance-validated joint calibration
YOLO_PADDING_DIAGNOSTIC = NOT RUN
ROISSD_PADDING_DIAGNOSTIC = NOT RUN
ROISSD_MOBILENET_PADDING_DIAGNOSTIC = NOT RUN
YOLO_CPU_K = 7.300956148244171
YOLO_CPU_c = 0.00015598040772002173
YOLO_CPU_tau = 46806.8794982834
ROISSD_CPU_K = NOT RUN
ROISSD_CPU_c = NOT RUN
ROISSD_CPU_tau = NOT RUN
ROISSD_MOBILENET_CPU_K = 9.892776083211256
ROISSD_MOBILENET_CPU_c = 0.00014455301960633748
ROISSD_MOBILENET_CPU_tau = 68437.00747415959

Exact prepared configurations:

- `yolo26n`: [configs/yolo26n.yaml](configs/yolo26n.yaml)
- `roi_ssd_mobilenet`: [configs/roi_ssd_mobilenet.yaml](configs/roi_ssd_mobilenet.yaml)
