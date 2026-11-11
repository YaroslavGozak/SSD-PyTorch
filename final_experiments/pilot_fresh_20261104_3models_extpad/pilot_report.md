# Stage 3 CPU pilot

## Preparation status

`PILOT_RUNS_COMPLETE_PENDING_REVIEW`. Completed stages: ['cpu_cost_calibration', 'padding_sweep', 'pipeline_modes']. The full padding sweep is complete. FF/SEG/CA/FULL-TECH pipeline runs are recorded.

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
- ROI-SSD checkpoint SHA-256: `f8f49132e08a7c5bf3744ce32e46e1b0fed6595aabd9cf59cf68699569503f76`.
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

## Measured full-frame and padding results

All entries use the locked pilot; image loading and metric computation are excluded from process-frame timing.

| Model | Mode | Padding | mAP50 | mAP50:95 | Recall50 | GT coverage | Mean latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | FF | 0 | 0.8186 | 0.5760 | 0.8432 | 1.0000 | 31.75 | 31.49 |
| yolo26n | SEG | 32 | 0.7492 | 0.5184 | 0.8423 | 0.9946 | 104.80 | 9.54 |
| yolo26n | SEG | 64 | 0.7977 | 0.5484 | 0.8582 | 1.0000 | 98.44 | 10.16 |
| yolo26n | SEG | 96 | 0.8147 | 0.5574 | 0.8640 | 1.0000 | 99.53 | 10.05 |
| yolo26n | SEG | 112 | 0.8206 | 0.5668 | 0.8632 | 1.0000 | 102.06 | 9.80 |
| yolo26n | SEG | 128 | 0.8176 | 0.5704 | 0.8552 | 1.0000 | 89.99 | 11.11 |
| yolo26n | SEG | 144 | 0.8170 | 0.5753 | 0.8503 | 1.0000 | 76.56 | 13.06 |
| yolo26n | SEG | 160 | 0.8204 | 0.5761 | 0.8524 | 1.0000 | 83.38 | 11.99 |
| yolo26n | SEG | 192 | 0.8174 | 0.5756 | 0.8461 | 1.0000 | 70.63 | 14.16 |
| yolo26n | SEG | 224 | 0.8184 | 0.5756 | 0.8441 | 1.0000 | 62.75 | 15.94 |
| roi_ssd | FF | 0 | 0.7568 | 0.4702 | 0.7651 | 1.0000 | 143.49 | 6.97 |
| roi_ssd | SEG | 32 | 0.6279 | 0.3680 | 0.6544 | 0.8929 | 133.26 | 7.50 |
| roi_ssd | SEG | 64 | 0.7168 | 0.4094 | 0.7431 | 0.9571 | 144.21 | 6.93 |
| roi_ssd | SEG | 96 | 0.7240 | 0.4142 | 0.7436 | 0.9518 | 171.50 | 5.83 |
| roi_ssd | SEG | 112 | 0.7258 | 0.4127 | 0.7370 | 0.9482 | 204.79 | 4.88 |
| roi_ssd | SEG | 128 | 0.7184 | 0.4255 | 0.7324 | 0.9643 | 192.63 | 5.19 |
| roi_ssd | SEG | 144 | 0.7176 | 0.4255 | 0.7444 | 0.9661 | 225.82 | 4.43 |
| roi_ssd | SEG | 160 | 0.7049 | 0.4219 | 0.7330 | 0.9679 | 212.62 | 4.70 |
| roi_ssd | SEG | 192 | 0.7368 | 0.4450 | 0.7444 | 0.9750 | 220.50 | 4.54 |
| roi_ssd | SEG | 224 | 0.7336 | 0.4517 | 0.7427 | 0.9875 | 328.72 | 3.04 |
| roi_ssd_mobilenet | FF | 0 | 0.7410 | 0.4278 | 0.7605 | 1.0000 | 23.82 | 41.99 |
| roi_ssd_mobilenet | SEG | 32 | 0.6125 | 0.3703 | 0.6237 | 0.8375 | 25.92 | 38.58 |
| roi_ssd_mobilenet | SEG | 64 | 0.5951 | 0.3015 | 0.6267 | 0.9536 | 31.62 | 31.63 |
| roi_ssd_mobilenet | SEG | 96 | 0.6753 | 0.3645 | 0.7033 | 0.9661 | 36.59 | 27.33 |
| roi_ssd_mobilenet | SEG | 112 | 0.6860 | 0.3808 | 0.7139 | 0.9750 | 39.11 | 25.57 |
| roi_ssd_mobilenet | SEG | 128 | 0.7262 | 0.4118 | 0.7473 | 0.9964 | 38.03 | 26.29 |
| roi_ssd_mobilenet | SEG | 144 | 0.7334 | 0.4038 | 0.7580 | 0.9839 | 36.40 | 27.47 |
| roi_ssd_mobilenet | SEG | 160 | 0.7137 | 0.4116 | 0.7438 | 0.9804 | 35.21 | 28.40 |
| roi_ssd_mobilenet | SEG | 192 | 0.7421 | 0.4207 | 0.7656 | 0.9857 | 34.67 | 28.84 |
| roi_ssd_mobilenet | SEG | 224 | 0.7466 | 0.4296 | 0.7660 | 1.0000 | 33.34 | 29.99 |

Sources: [padding sweep](padding_sweep.csv), [frame-level metrics](frame_level_metrics.csv), [preflight](stage2_preflight.json).

## Locked padding decisions

| Model | Final padding | Criterion met | Eligible values |
|---|---:|:---:|---|
| yolo26n | 64 | True | [64, 96, 112, 128, 144, 160, 192, 224] |
| roi_ssd | 224 | True | [224] |
| roi_ssd_mobilenet | 128 | True | [128, 144, 160, 192, 224] |

## Fresh CPU cost profiles

| Model | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) |
|---|---:|---:|---:|---:|---:|---:|
| yolo26n | 8.74918 | 0.00012535 | 69798.007 | 0.9287 | 0.9474 | 1.0720 |
| roi_ssd | 15.9228 | 0.00154761 | 10288.628 | 0.9958 | 2.2648 | 3.1134 |
| roi_ssd_mobilenet | 9.55172 | 0.00016249 | 58783.429 | 0.9333 | 1.0567 | 1.3406 |

## FF / SEG / CA / FULL-TECH results

| Model | Mode | mAP50 | mAP50:95 | Recall50 | total p50 ms | total p95 ms | inference p50 ms | inference p95 ms | ROI coverage | Tensor area ratio | Full-frame fraction | NN calls/frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | FF | 0.8186 | 0.5760 | 0.8432 | 24.95 | 29.35 | 24.75 | 29.10 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| yolo26n | SEG | 0.7977 | 0.5484 | 0.8582 | 26.65 | 387.27 | 13.88 | 24.00 | 1.0000 | 2.2231 | 0.1938 | 5.647 |
| yolo26n | CA | 0.8143 | 0.5586 | 0.8397 | 23.08 | 39.95 | 21.89 | 31.84 | 1.0000 | 0.7847 | 0.1938 | 1.000 |
| yolo26n | FULL-TECH | 0.8143 | 0.5586 | 0.8397 | 23.84 | 45.35 | 22.62 | 32.14 | 1.0000 | 0.7847 | 0.3844 | 1.000 |
| roi_ssd | FF | 0.7568 | 0.4702 | 0.7651 | 140.90 | 196.00 | 140.73 | 195.86 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| roi_ssd | SEG | 0.7336 | 0.4517 | 0.7427 | 138.27 | 608.42 | 132.53 | 189.73 | 0.9875 | 1.4111 | 0.2875 | 1.434 |
| roi_ssd | CA | 0.7338 | 0.4585 | 0.7416 | 134.16 | 173.78 | 133.56 | 173.66 | 0.9875 | 0.9960 | 0.2875 | 1.000 |
| roi_ssd | FULL-TECH | 0.7338 | 0.4585 | 0.7416 | 145.03 | 206.64 | 144.86 | 206.62 | 0.9875 | 0.9960 | 0.9563 | 1.000 |
| roi_ssd_mobilenet | FF | 0.7410 | 0.4278 | 0.7605 | 22.49 | 27.55 | 22.37 | 27.39 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| roi_ssd_mobilenet | SEG | 0.7262 | 0.4118 | 0.7473 | 24.84 | 94.01 | 24.36 | 34.04 | 0.9964 | 1.1627 | 0.2781 | 1.400 |
| roi_ssd_mobilenet | CA | 0.7092 | 0.4196 | 0.7294 | 25.27 | 34.65 | 24.87 | 34.46 | 0.9804 | 0.9349 | 0.2781 | 1.000 |
| roi_ssd_mobilenet | FULL-TECH | 0.7092 | 0.4196 | 0.7294 | 25.46 | 36.68 | 25.13 | 36.39 | 0.9804 | 0.9349 | 0.5563 | 1.000 |

## Frozen pilot parameters

YOLO_PADDING_FINAL = 64
ROISSD_PADDING_FINAL = 224
ROISSD_MOBILENET_PADDING_FINAL = 128
YOLO_CPU_K = 8.749180919149513
YOLO_CPU_c = 0.00012535001012166345
YOLO_CPU_tau = 69798.0072810337
ROISSD_CPU_K = 15.922756812554802
ROISSD_CPU_c = 0.0015476073516446731
ROISSD_CPU_tau = 10288.628311072167
ROISSD_MOBILENET_CPU_K = 9.551721143467546
ROISSD_MOBILENET_CPU_c = 0.0001624900298380981
ROISSD_MOBILENET_CPU_tau = 58783.429069369326
