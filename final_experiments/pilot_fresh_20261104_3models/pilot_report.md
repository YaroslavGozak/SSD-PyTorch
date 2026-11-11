# Stage 3 CPU pilot

## Preparation status

`BLOCKED_PIPELINE_ACCEPTANCE`. Completed stages: ['coordinate_smoke', 'cpu_cost_calibration', 'padding_sweep']. The full padding sweep is complete. CA/FULL-TECH pipeline runs have not been completed.

**Execution stop:** STOP: roi_ssd has no tested padding satisfying both segment/full-frame mAP50 >= 0.95 and GT coverage >= 0.98. Investigate before running CA/FULL-TECH.

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

Investigate failed segment-accuracy/coverage criteria for every model before rerunning CA/FULL-TECH. Do not relax thresholds or substitute checkpoints post hoc. Stage 4 / Raspberry Pi runs remain on hold.

## CPU smoke (not decision-grade)

Coordinate/class checks passed on 20 frames per model; 20 overlays per model were exported. These metrics are diagnostic only and must not select padding.

| Model | Mode | mAP50 | Recall50 |
|---|---|---:|---:|
| yolo26n | FF | 1.0000 | 1.0000 |
| yolo26n | SEG | 1.0000 | 1.0000 |
| roi_ssd | FF | 0.9500 | 0.9500 |
| roi_ssd | SEG | 0.9929 | 1.0000 |
| roi_ssd_mobilenet | FF | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | SEG | 0.9793 | 1.0000 |

## Measured full-frame and padding results

All entries use the locked pilot; image loading and metric computation are excluded from process-frame timing.

| Model | Mode | Padding | mAP50 | mAP50:95 | Recall50 | GT coverage | Mean latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | FF | 0 | 0.8186 | 0.5760 | 0.8432 | 1.0000 | 22.38 | 44.69 |
| yolo26n | SEG | 32 | 0.7492 | 0.5184 | 0.8423 | 0.9946 | 90.17 | 11.09 |
| yolo26n | SEG | 64 | 0.7977 | 0.5484 | 0.8582 | 1.0000 | 97.08 | 10.30 |
| yolo26n | SEG | 96 | 0.8147 | 0.5574 | 0.8640 | 1.0000 | 106.91 | 9.35 |
| yolo26n | SEG | 112 | 0.8206 | 0.5668 | 0.8632 | 1.0000 | 108.48 | 9.22 |
| yolo26n | SEG | 128 | 0.8176 | 0.5704 | 0.8552 | 1.0000 | 93.64 | 10.68 |
| yolo26n | SEG | 144 | 0.8170 | 0.5753 | 0.8503 | 1.0000 | 95.56 | 10.47 |
| yolo26n | SEG | 160 | 0.8204 | 0.5761 | 0.8524 | 1.0000 | 81.05 | 12.34 |
| roi_ssd | FF | 0 | 0.7568 | 0.4702 | 0.7651 | 1.0000 | 143.76 | 6.96 |
| roi_ssd | SEG | 32 | 0.6279 | 0.3680 | 0.6544 | 0.8929 | 110.89 | 9.02 |
| roi_ssd | SEG | 64 | 0.7168 | 0.4094 | 0.7431 | 0.9571 | 145.68 | 6.86 |
| roi_ssd | SEG | 96 | 0.7240 | 0.4142 | 0.7436 | 0.9518 | 167.49 | 5.97 |
| roi_ssd | SEG | 112 | 0.7258 | 0.4127 | 0.7370 | 0.9482 | 180.89 | 5.53 |
| roi_ssd | SEG | 128 | 0.7184 | 0.4255 | 0.7324 | 0.9643 | 188.24 | 5.31 |
| roi_ssd | SEG | 144 | 0.7176 | 0.4255 | 0.7444 | 0.9661 | 189.63 | 5.27 |
| roi_ssd | SEG | 160 | 0.7049 | 0.4219 | 0.7330 | 0.9679 | 201.92 | 4.95 |
| roi_ssd_mobilenet | FF | 0 | 0.7410 | 0.4278 | 0.7605 | 1.0000 | 25.14 | 39.77 |
| roi_ssd_mobilenet | SEG | 32 | 0.6125 | 0.3703 | 0.6237 | 0.8375 | 25.84 | 38.71 |
| roi_ssd_mobilenet | SEG | 64 | 0.5951 | 0.3015 | 0.6267 | 0.9536 | 31.07 | 32.18 |
| roi_ssd_mobilenet | SEG | 96 | 0.6753 | 0.3645 | 0.7033 | 0.9661 | 35.95 | 27.82 |
| roi_ssd_mobilenet | SEG | 112 | 0.6860 | 0.3808 | 0.7139 | 0.9750 | 36.88 | 27.12 |
| roi_ssd_mobilenet | SEG | 128 | 0.7262 | 0.4118 | 0.7473 | 0.9964 | 34.54 | 28.95 |
| roi_ssd_mobilenet | SEG | 144 | 0.7334 | 0.4038 | 0.7580 | 0.9839 | 35.66 | 28.04 |
| roi_ssd_mobilenet | SEG | 160 | 0.7137 | 0.4116 | 0.7438 | 0.9804 | 32.98 | 30.32 |

Sources: [padding sweep](padding_sweep.csv), [frame-level metrics](frame_level_metrics.csv), [preflight](stage2_preflight.json).

## Locked padding decisions

| Model | Final padding | Criterion met | Eligible values |
|---|---:|:---:|---|
| yolo26n | 64 | True | [64, 96, 112, 128, 144, 160] |
| roi_ssd | 160 | False | [] |
| roi_ssd_mobilenet | 128 | True | [128, 144, 160] |

## Fresh CPU cost profiles

| Model | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) |
|---|---:|---:|---:|---:|---:|---:|
| yolo26n | 8.15417 | 0.000187031 | 43597.948 | 0.9814 | 0.6337 | 0.7948 |
| roi_ssd | 13.0456 | 0.00170421 | 7654.914 | 0.9986 | 1.4511 | 1.9599 |
| roi_ssd_mobilenet | 9.64153 | 0.000163613 | 58929.069 | 0.8765 | 1.5511 | 1.8954 |

## Frozen pilot parameters

YOLO_PADDING_FINAL = 64
ROISSD_PADDING_FINAL = 160
ROISSD_MOBILENET_PADDING_FINAL = 128
YOLO_CPU_K = 8.154166423140088
YOLO_CPU_c = 0.00018703096644783555
YOLO_CPU_tau = 43597.94839329107
ROISSD_CPU_K = 13.045614340181984
ROISSD_CPU_c = 0.001704214296486716
ROISSD_CPU_tau = 7654.914271682776
ROISSD_MOBILENET_CPU_K = 9.641534138989767
ROISSD_MOBILENET_CPU_c = 0.0001636125320014494
ROISSD_MOBILENET_CPU_tau = 58929.06870302793
