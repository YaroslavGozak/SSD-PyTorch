# Stage 3 CPU pilot

## Preparation status

`STAGE3_COMPLETE`. Completed stages: ['coordinate_smoke', 'cpu_cost_calibration', 'joint_calibration', 'pipeline_modes']. Reviewed acceptance checks pass for both models.

## Acceptance

```text
YOLO_CALIBRATION_GATE = PASS
YOLO_PIPELINE_QUALITY_GATE = PASS

ROISSD_MOBILENET_CALIBRATION_GATE = PASS
ROISSD_MOBILENET_PIPELINE_QUALITY_GATE = PASS

STAGE3_STATUS = PASS
```

Pipeline quality gates: `mAP50 retention >= 0.95` and `gt_roi_coverage_mean >= 0.98`, checked for SEG, CA, and FULL-TECH against each model's FF baseline. All modes ran on the full 320-frame pilot.

| Model | Mode | mAP50 retention | GT coverage | Result |
|---|---|---:|---:|:---:|
| yolo26n | SEG | 0.9763 | 1.0000 | PASS |
| yolo26n | CA | 0.9965 | 1.0000 | PASS |
| yolo26n | FULL-TECH | 0.9965 | 1.0000 | PASS |
| roi_ssd_mobilenet | SEG | 0.9837 | 0.9964 | PASS |
| roi_ssd_mobilenet | CA | 0.9748 | 0.9804 | PASS |
| roi_ssd_mobilenet | FULL-TECH | 0.9748 | 0.9804 | PASS |

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

Stage 3 acceptance is complete. Proceed to Stage 4 using the frozen parameters below.

## Runtime environment

| Property | Value |
|---|---|
| CPU model | Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz |
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
| yolo26n | FF | 1.0000 | 1.0000 |
| yolo26n | SEG | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | FF | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | SEG | 0.9401 | 1.0000 |

Thresholds: evaluation_conf=0.01; operational_conf by model is recorded in smoke_metadata.json.

| Model | Evaluation detections | Operational detections | Candidate segments |
|---|---:|---:|---:|
| yolo26n | 183 | 157 | 87 |
| roi_ssd_mobilenet | 1659 | 366 | 248 |

The low ROI-SSD-MobileNet smoke ratio is diagnostic only and was superseded by the full-pilot calibration and pipeline quality checks above.

## Fresh CPU cost profiles

Timing boundary: model forward and detector-internal postprocessing at the fixed evaluation_conf. Operational thresholding, tracker updates, and metrics are outside this timing; the operational confidence and padding are recorded as frozen provenance.

| Model | evaluation_conf | operational_conf | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | 0.0100 | 0.1000 | 8.07126 | 0.000194665 | 41462.304 | 0.9623 | 1.0647 | 1.1892 |
| roi_ssd_mobilenet | 0.0100 | 0.5000 | 10.7685 | 0.000158328 | 68014.119 | 0.8156 | 2.2190 | 2.3231 |

## FF / SEG / CA / FULL-TECH results

| Model | Mode | evaluation_conf | operational_conf | mAP50 | mAP50:95 | Recall50 | total p50 ms | total p95 ms | inference p50 ms | inference p95 ms | ROI coverage | Tensor area ratio | Full-frame fraction | NN calls/frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | FF | 0.0100 | 0.1000 | 0.8186 | 0.5760 | 0.8432 | 24.29 | 28.62 | 24.07 | 28.36 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| yolo26n | SEG | 0.0100 | 0.1000 | 0.7992 | 0.5532 | 0.8374 | 22.17 | 94.21 | 15.02 | 26.07 | 1.0000 | 1.0304 | 0.2188 | 2.006 |
| yolo26n | CA | 0.0100 | 0.1000 | 0.8157 | 0.5615 | 0.8393 | 21.06 | 28.52 | 20.55 | 28.03 | 1.0000 | 0.7552 | 0.2188 | 1.000 |
| yolo26n | FULL-TECH | 0.0100 | 0.1000 | 0.8157 | 0.5615 | 0.8393 | 20.94 | 27.42 | 20.24 | 25.92 | 1.0000 | 0.7552 | 0.3656 | 1.000 |
| roi_ssd_mobilenet | FF | 0.0100 | 0.5000 | 0.8377 | 0.4626 | 0.8816 | 21.80 | 26.91 | 21.51 | 26.56 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| roi_ssd_mobilenet | SEG | 0.0100 | 0.5000 | 0.8240 | 0.4431 | 0.9086 | 24.53 | 97.84 | 23.82 | 32.85 | 0.9964 | 1.1627 | 0.2781 | 1.400 |
| roi_ssd_mobilenet | CA | 0.0100 | 0.5000 | 0.8165 | 0.4562 | 0.8770 | 22.75 | 31.82 | 22.23 | 31.24 | 0.9804 | 0.9349 | 0.2781 | 1.000 |
| roi_ssd_mobilenet | FULL-TECH | 0.0100 | 0.5000 | 0.8165 | 0.4562 | 0.8770 | 23.44 | 32.35 | 22.83 | 31.76 | 0.9804 | 0.9349 | 0.5563 | 1.000 |

### CA and FULL-TECH selector/workload interpretation

- yolo26n: CA changed post-merge segment count, NN calls, tensor pixels versus SEG. FULL-TECH logical selector decisions `{'FF': 117, 'SEG': 203}`; CA tensor pixels/NN calls=`24748032`/`320`, FULL-TECH=`24748032`/`320`. CA and FULL-TECH produced identical evaluation predictions, total tensor pixels, and NN calls; any timing difference is not evidence of selector workload savings.
- roi_ssd_mobilenet: CA changed post-merge segment count, NN calls, tensor pixels versus SEG. FULL-TECH logical selector decisions `{'FF': 178, 'SEG': 142}`; CA tensor pixels/NN calls=`30634775`/`320`, FULL-TECH=`30634775`/`320`. CA and FULL-TECH produced identical evaluation predictions, total tensor pixels, and NN calls; any timing difference is not evidence of selector workload savings.

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
YOLO_CPU_K = 8.071262357983457
YOLO_CPU_c = 0.00019466507291216131
YOLO_CPU_tau = 41462.30362354448
ROISSD_CPU_K = NOT RUN
ROISSD_CPU_c = NOT RUN
ROISSD_CPU_tau = NOT RUN
ROISSD_MOBILENET_CPU_K = 10.76854700242187
ROISSD_MOBILENET_CPU_c = 0.00015832811179376522
ROISSD_MOBILENET_CPU_tau = 68014.11878421658

Exact prepared configurations:

- `yolo26n`: [configs/yolo26n.yaml](configs/yolo26n.yaml)
- `roi_ssd_mobilenet`: [configs/roi_ssd_mobilenet.yaml](configs/roi_ssd_mobilenet.yaml)
