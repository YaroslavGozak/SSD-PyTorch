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

- yolo26n checkpoint SHA-256: `5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027`.
- roi_ssd_mobilenet checkpoint SHA-256: `1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804`.
- Exact source configuration snapshots are in [`configs/`](./configs/).
- Selected models use the locked inference confidence threshold
  `0.01`; YOLO output cap/NMS are locked to 200 detections and
  IoU 0.45. CPU threads are fixed to 8 (the i7-11700K core
  count); OpenCV threads to 1.

## Smoke check

The 20-frame CPU coordinate smoke is run separately. Its outcome is recorded in
[`smoke_metadata.json`](./smoke_metadata.json); smoke metrics are not pilot results.

## Remaining protocol

Review acceptance checks and final settings before Stage 4.

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

## CPU smoke (not decision-grade)

Coordinate/class checks passed on 20 frames per model; 20 overlays per model were exported. These metrics are diagnostic only and must not select padding.

| Model | Mode | mAP50 | Recall50 |
|---|---|---:|---:|
| yolo26n | FF | 1.0000 | 1.0000 |
| yolo26n | SEG | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | FF | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | SEG | 0.8053 | 1.0000 |

**STOP warning:** roi_ssd_mobilenet smoke segment/full-frame mAP50 ratio is 0.805. The sample is too small for a conclusion, but the full padding sweep must resolve this before CA/FULL-TECH is run.

## Measured full-frame and padding results

All entries use the locked pilot; image loading and metric computation are excluded from process-frame timing.

| Model | Mode | Padding | mAP50 | mAP50:95 | Recall50 | GT coverage | Mean latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | FF | 0 | 0.8186 | 0.5760 | 0.8432 | 1.0000 | 21.12 | 47.34 |
| yolo26n | SEG | 32 | 0.7492 | 0.5184 | 0.8423 | 0.9946 | 79.50 | 12.58 |
| yolo26n | SEG | 64 | 0.7977 | 0.5484 | 0.8582 | 1.0000 | 87.26 | 11.46 |
| yolo26n | SEG | 96 | 0.8147 | 0.5574 | 0.8640 | 1.0000 | 86.43 | 11.57 |
| yolo26n | SEG | 112 | 0.8206 | 0.5668 | 0.8632 | 1.0000 | 84.64 | 11.81 |
| yolo26n | SEG | 128 | 0.8176 | 0.5704 | 0.8552 | 1.0000 | 77.94 | 12.83 |
| yolo26n | SEG | 144 | 0.8170 | 0.5753 | 0.8503 | 1.0000 | 71.11 | 14.06 |
| yolo26n | SEG | 160 | 0.8204 | 0.5761 | 0.8524 | 1.0000 | 69.77 | 14.33 |
| yolo26n | SEG | 192 | 0.8174 | 0.5756 | 0.8461 | 1.0000 | 74.03 | 13.51 |
| yolo26n | SEG | 224 | 0.8184 | 0.5756 | 0.8441 | 1.0000 | 70.17 | 14.25 |
| roi_ssd_mobilenet | FF | 0 | 0.8377 | 0.4626 | 0.8816 | 1.0000 | 23.87 | 41.89 |
| roi_ssd_mobilenet | SEG | 32 | 0.6361 | 0.3473 | 0.9193 | 1.0000 | 14053.95 | 0.07 |
| roi_ssd_mobilenet | SEG | 64 | 0.6443 | 0.2924 | 0.9505 | 1.0000 | 12526.58 | 0.08 |
| roi_ssd_mobilenet | SEG | 96 | 0.7290 | 0.3657 | 0.9550 | 1.0000 | 7743.23 | 0.13 |
| roi_ssd_mobilenet | SEG | 112 | 0.7560 | 0.3888 | 0.9670 | 1.0000 | 5642.41 | 0.18 |
| roi_ssd_mobilenet | SEG | 128 | 0.7691 | 0.4127 | 0.9550 | 1.0000 | 4319.56 | 0.23 |
| roi_ssd_mobilenet | SEG | 144 | 0.7914 | 0.4109 | 0.9565 | 1.0000 | 241765.63 | 0.00 |
| roi_ssd_mobilenet | SEG | 160 | 0.7787 | 0.4173 | 0.9399 | 1.0000 | 2307.83 | 0.43 |
| roi_ssd_mobilenet | SEG | 192 | 0.8074 | 0.4402 | 0.9449 | 1.0000 | 1313.21 | 0.76 |
| roi_ssd_mobilenet | SEG | 224 | 0.8098 | 0.4458 | 0.9201 | 1.0000 | 666.75 | 1.50 |

Sources: [padding sweep](padding_sweep.csv), [frame-level metrics](frame_level_metrics.csv), [preflight](stage2_preflight.json).

## Locked padding decisions

| Model | Final padding | Criterion met | Eligible values |
|---|---:|:---:|---|
| yolo26n | 64 | True | [64, 96, 112, 128, 144, 160, 192, 224] |
| roi_ssd_mobilenet | 192 | True | [192, 224] |

## Fresh CPU cost profiles

| Model | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) |
|---|---:|---:|---:|---:|---:|---:|
| yolo26n | 6.87744 | 0.000143614 | 47888.313 | 0.9749 | 0.5505 | 0.7107 |
| roi_ssd_mobilenet | 8.69661 | 0.000131167 | 66301.590 | 0.9090 | 1.0836 | 1.2809 |

## FF / SEG / CA / FULL-TECH results

| Model | Mode | mAP50 | mAP50:95 | Recall50 | total p50 ms | total p95 ms | inference p50 ms | inference p95 ms | ROI coverage | Tensor area ratio | Full-frame fraction | NN calls/frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | FF | 0.7530 | 0.5480 | 0.7619 | 20.98 | 25.17 | 20.85 | 24.99 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| yolo26n | SEG | 0.7385 | 0.5265 | 0.7550 | 21.03 | 55.22 | 16.03 | 25.14 | 0.9625 | 0.8397 | 0.2938 | 1.419 |
| yolo26n | CA | 0.7504 | 0.5362 | 0.7587 | 21.33 | 27.83 | 20.88 | 27.22 | 0.9589 | 0.7574 | 0.2938 | 1.000 |
| yolo26n | FULL-TECH | 0.7504 | 0.5362 | 0.7587 | 20.12 | 27.50 | 19.54 | 26.13 | 0.9589 | 0.7574 | 0.3969 | 1.000 |
| roi_ssd_mobilenet | FF | 0.7410 | 0.4278 | 0.7605 | 21.55 | 25.51 | 21.42 | 25.26 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| roi_ssd_mobilenet | SEG | 0.7421 | 0.4207 | 0.7656 | 21.68 | 96.94 | 21.40 | 29.38 | 0.9857 | 1.3253 | 0.2625 | 1.391 |
| roi_ssd_mobilenet | CA | 0.7394 | 0.4238 | 0.7582 | 21.68 | 26.60 | 21.43 | 26.23 | 0.9857 | 0.9896 | 0.2656 | 1.000 |
| roi_ssd_mobilenet | FULL-TECH | 0.7394 | 0.4238 | 0.7582 | 21.62 | 25.43 | 21.28 | 24.96 | 0.9857 | 0.9896 | 0.9094 | 1.000 |

## Frozen pilot parameters

YOLO_PADDING_FINAL = 64
ROISSD_PADDING_FINAL = NOT RUN
ROISSD_MOBILENET_PADDING_FINAL = 192
YOLO_CPU_K = 6.8774393058650585
YOLO_CPU_c = 0.00014361414933227612
YOLO_CPU_tau = 47888.312800940774
ROISSD_CPU_K = NOT RUN
ROISSD_CPU_c = NOT RUN
ROISSD_CPU_tau = NOT RUN
ROISSD_MOBILENET_CPU_K = 8.696609392937296
ROISSD_MOBILENET_CPU_c = 0.0001311674338486362
ROISSD_MOBILENET_CPU_tau = 66301.58979074757
