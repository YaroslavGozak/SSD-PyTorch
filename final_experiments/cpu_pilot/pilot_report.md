# Stage 3 CPU pilot

## Preparation status

`PILOT_RUNS_COMPLETE_PENDING_REVIEW`. Completed stages: ['coordinate_smoke', 'cpu_cost_calibration', 'padding_sweep', 'pipeline_modes']. The full padding sweep is complete. FF/SEG/CA/FULL-TECH pipeline runs are recorded.

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

- YOLO26n checkpoint SHA-256: `5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027`.
- ROI-SSD-MobileNet ROI checkpoint SHA-256: `1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804`.
- Exact source configuration snapshots are in [`configs/`](./configs/).
- Inference thresholds follow the selected configs; YOLO output cap/NMS are locked
  to 200 detections and IoU 0.45. CPU threads are fixed to 8 (the i7-11700K core
  count); OpenCV threads to 1.

## Smoke check

The 20-frame CPU coordinate smoke is run separately. Its outcome is recorded in
[`smoke_metadata.json`](./smoke_metadata.json); smoke metrics are not pilot results.

## Remaining protocol

Review acceptance checks and final settings before Stage 4.

## CPU smoke (not decision-grade)

Coordinate/class checks passed on 20 frames per model; 20 overlays per model were exported. These metrics are diagnostic only and must not select padding.

| Model | Mode | mAP50 | Recall50 |
|---|---|---:|---:|
| yolo26n | FF | 1.0000 | 1.0000 |
| yolo26n | SEG | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | FF | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | SEG | 0.9793 | 1.0000 |

## Measured full-frame and padding results

All entries use the full locked 320-frame CPU pilot; image loading and metric computation are excluded from process-frame timing.

| Model | Mode | Padding | mAP50 | mAP50:95 | Recall50 | GT coverage | Mean latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | FF | 0 | 0.8186 | 0.5760 | 0.8432 | 1.0000 | 19.80 | 50.52 |
| yolo26n | SEG | 32 | 0.7492 | 0.5184 | 0.8423 | 0.9946 | 73.28 | 13.65 |
| yolo26n | SEG | 64 | 0.7977 | 0.5484 | 0.8582 | 1.0000 | 75.50 | 13.24 |
| yolo26n | SEG | 96 | 0.8147 | 0.5574 | 0.8640 | 1.0000 | 77.75 | 12.86 |
| yolo26n | SEG | 112 | 0.8206 | 0.5668 | 0.8632 | 1.0000 | 77.01 | 12.99 |
| yolo26n | SEG | 128 | 0.8176 | 0.5704 | 0.8552 | 1.0000 | 69.10 | 14.47 |
| yolo26n | SEG | 144 | 0.8170 | 0.5753 | 0.8503 | 1.0000 | 63.16 | 15.83 |
| yolo26n | SEG | 160 | 0.8204 | 0.5761 | 0.8524 | 1.0000 | 57.62 | 17.35 |
| roi_ssd_mobilenet | FF | 0 | 0.7410 | 0.4278 | 0.7605 | 1.0000 | 21.78 | 45.92 |
| roi_ssd_mobilenet | SEG | 32 | 0.6125 | 0.3703 | 0.6237 | 0.8375 | 23.82 | 41.97 |
| roi_ssd_mobilenet | SEG | 64 | 0.5951 | 0.3015 | 0.6267 | 0.9536 | 29.91 | 33.44 |
| roi_ssd_mobilenet | SEG | 96 | 0.6753 | 0.3645 | 0.7033 | 0.9661 | 31.78 | 31.46 |
| roi_ssd_mobilenet | SEG | 112 | 0.6860 | 0.3808 | 0.7139 | 0.9750 | 34.31 | 29.15 |
| roi_ssd_mobilenet | SEG | 128 | 0.7262 | 0.4118 | 0.7473 | 0.9964 | 33.48 | 29.87 |
| roi_ssd_mobilenet | SEG | 144 | 0.7334 | 0.4038 | 0.7580 | 0.9839 | 34.02 | 29.39 |
| roi_ssd_mobilenet | SEG | 160 | 0.7137 | 0.4116 | 0.7438 | 0.9804 | 31.51 | 31.74 |

Sources: [padding sweep](padding_sweep.csv), [frame-level metrics](frame_level_metrics.csv), [preflight](stage2_preflight.json).

## Locked padding decisions

| Model | Final padding | Criterion met | Eligible values |
|---|---:|:---:|---|
| yolo26n | 64 | True | [64, 96, 112, 128, 144, 160] |
| roi_ssd_mobilenet | 128 | True | [128, 144, 160] |

## Fresh CPU cost profiles

| Model | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) |
|---|---:|---:|---:|---:|---:|---:|
| yolo26n | 6.19895 | 0.00013877 | 44670.671 | 0.9751 | 0.5336 | 0.6837 |
| roi_ssd_mobilenet | 7.63317 | 0.000148618 | 51360.921 | 0.9136 | 1.2159 | 1.4105 |

## FF / SEG / CA / FULL-TECH results

| Model | Mode | mAP50 | mAP50:95 | Recall50 | total p50 ms | total p95 ms | inference p50 ms | inference p95 ms | ROI coverage | Tensor area ratio | Full-frame fraction | NN calls/frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | FF | 0.8186 | 0.5760 | 0.8432 | 19.22 | 21.39 | 18.95 | 21.11 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| yolo26n | SEG | 0.7977 | 0.5484 | 0.8582 | 22.79 | 325.76 | 11.77 | 20.53 | 1.0000 | 2.2231 | 0.1938 | 5.647 |
| yolo26n | CA | 0.8143 | 0.5586 | 0.8397 | 18.29 | 30.35 | 17.52 | 24.71 | 1.0000 | 0.7847 | 0.1938 | 1.000 |
| yolo26n | FULL-TECH | 0.8143 | 0.5586 | 0.8397 | 18.02 | 27.04 | 17.58 | 23.90 | 1.0000 | 0.7847 | 0.3844 | 1.000 |
| roi_ssd_mobilenet | FF | 0.7410 | 0.4278 | 0.7605 | 21.66 | 23.88 | 21.53 | 23.76 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| roi_ssd_mobilenet | SEG | 0.7262 | 0.4118 | 0.7473 | 23.07 | 91.28 | 22.91 | 33.51 | 0.9964 | 1.1627 | 0.2781 | 1.400 |
| roi_ssd_mobilenet | CA | 0.7092 | 0.4196 | 0.7294 | 22.09 | 31.92 | 21.87 | 31.69 | 0.9804 | 0.9349 | 0.2781 | 1.000 |
| roi_ssd_mobilenet | FULL-TECH | 0.7092 | 0.4196 | 0.7294 | 22.25 | 31.58 | 22.05 | 31.51 | 0.9804 | 0.9349 | 0.5563 | 1.000 |

## Frozen pilot parameters

YOLO_PADDING_FINAL = 64
ROISSD_MOBILENET_PADDING_FINAL = 128
YOLO_CPU_K = 6.198948738812571
YOLO_CPU_c = 0.00013876999377111665
YOLO_CPU_tau = 44670.67101722973
ROISSD_MOBILENET_CPU_K = 7.633166694570149
ROISSD_MOBILENET_CPU_c = 0.00014861818201253972
ROISSD_MOBILENET_CPU_tau = 51360.92092639175
