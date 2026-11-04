# Stage 3 CPU pilot

## Preparation status

`BLOCKED_PIPELINE_ACCEPTANCE`. Completed stages: ['coordinate_smoke', 'cpu_cost_calibration', 'padding_sweep']. The full padding sweep is complete. CA/FULL-TECH pipeline runs have not been completed.

**Execution stop:** STOP: ROI-SSD-MobileNet p=96 retains only 0.68805965 of its full-frame mAP50 (required >=0.95), with GT coverage 0.95892857 (required >=0.98). No tested padding satisfies the predeclared criteria. CA/FULL-TECH were not run; Stage 4/Raspberry Pi remain on hold.

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
- ROI-SSD-MobileNet ROI checkpoint SHA-256: `7fa171ab669ac53f0d4f39586c08e61a60ca79fed98fd70874ef42bfd52356ce`.
- Exact source configuration snapshots are in [`configs/`](./configs/).
- Inference thresholds follow the selected configs; YOLO output cap/NMS are locked
  to 200 detections and IoU 0.45. CPU threads are fixed to 8 (the i7-11700K core
  count); OpenCV threads to 1.

## Smoke check

The 20-frame CPU coordinate smoke is run separately. Its outcome is recorded in
[`smoke_metadata.json`](./smoke_metadata.json); smoke metrics are not pilot results.

## Remaining protocol

Investigate the failed MobileNet segment-accuracy/coverage criteria before rerunning CA/FULL-TECH. Do not relax thresholds or substitute checkpoints post hoc. Stage 4 / Raspberry Pi runs remain on hold.

## CPU smoke (not decision-grade)

Coordinate/class checks passed on 20 frames per model; 20 overlays per model were exported. These metrics are diagnostic only and must not select padding.

| Model | Mode | mAP50 | Recall50 |
|---|---|---:|---:|
| yolo26n | FF | 1.0000 | 1.0000 |
| yolo26n | SEG | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | FF | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | SEG | 0.1410 | 0.3167 |

**STOP warning:** MobileNet smoke segment/full-frame mAP50 ratio is 0.141. The sample is too small for a conclusion, but the full padding sweep must resolve this before CA/FULL-TECH is run.

## Measured full-frame and padding results

All entries use the full locked 320-frame CPU pilot; image loading and metric computation are excluded from process-frame timing.

| Model | Mode | Padding | mAP50 | mAP50:95 | Recall50 | GT coverage | Mean latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | FF | 0 | 0.8186 | 0.5760 | 0.8432 | 1.0000 | 19.68 | 50.83 |
| yolo26n | SEG | 32 | 0.7492 | 0.5184 | 0.8423 | 0.9946 | 74.30 | 13.46 |
| yolo26n | SEG | 64 | 0.7977 | 0.5484 | 0.8582 | 1.0000 | 73.39 | 13.62 |
| yolo26n | SEG | 96 | 0.8147 | 0.5574 | 0.8640 | 1.0000 | 75.88 | 13.18 |
| roi_ssd_mobilenet | FF | 0 | 0.7391 | 0.4583 | 0.7549 | 1.0000 | 22.09 | 45.27 |
| roi_ssd_mobilenet | SEG | 32 | 0.4484 | 0.2027 | 0.4971 | 0.8643 | 29.24 | 34.20 |
| roi_ssd_mobilenet | SEG | 64 | 0.3894 | 0.1600 | 0.4924 | 0.9446 | 38.49 | 25.98 |
| roi_ssd_mobilenet | SEG | 96 | 0.5086 | 0.1926 | 0.5893 | 0.9589 | 37.57 | 26.62 |

Sources: [padding sweep](padding_sweep.csv), [frame-level metrics](frame_level_metrics.csv), [preflight](stage2_preflight.json).

## Locked padding decisions

| Model | Final padding | Criterion met | Eligible values |
|---|---:|:---:|---|
| yolo26n | 64 | True | [64, 96] |
| roi_ssd_mobilenet | 96 | False | [] |

## Fresh CPU cost profiles

| Model | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) |
|---|---:|---:|---:|---:|---:|---:|
| yolo26n | 6.33534 | 0.000128433 | 49328.017 | 0.9850 | 0.3552 | 0.4896 |
| roi_ssd_mobilenet | 8.09516 | 0.000154142 | 52517.638 | 0.9489 | 0.8915 | 1.1035 |

## Frozen pilot parameters

YOLO_PADDING_FINAL = 64
ROISSD_MOBILENET_PADDING_FINAL = 96
YOLO_CPU_K = 6.335343954672639
YOLO_CPU_c = 0.0001284329742175617
YOLO_CPU_tau = 49328.017148779494
ROISSD_MOBILENET_CPU_K = 8.095160321754358
ROISSD_MOBILENET_CPU_c = 0.0001541417439743473
ROISSD_MOBILENET_CPU_tau = 52517.63807149851
