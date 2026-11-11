# Stage 3 CPU pilot

## Preparation status

`COORDINATE_SMOKE_PASS_FULL_PILOT_PENDING`. Completed stages: ['coordinate_smoke', 'padding_sweep']. The full padding sweep is complete. CA/FULL-TECH pipeline runs have not been completed.

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

2. Run `python -m experiments.cpu_pilot.run_pilot --stage cost`.

## CPU smoke (not decision-grade)

Coordinate/class checks passed on 20 frames per model; 20 overlays per model were exported. These metrics are diagnostic only and must not select padding.

| Model | Mode | mAP50 | Recall50 |
|---|---|---:|---:|
| yolo26n | FF | 1.0000 | 1.0000 |
| yolo26n | SEG | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | FF | 1.0000 | 1.0000 |
| roi_ssd_mobilenet | SEG | 0.9793 | 1.0000 |

## Measured full-frame and padding results

All entries use the locked pilot; image loading and metric computation are excluded from process-frame timing.

| Model | Mode | Padding | mAP50 | mAP50:95 | Recall50 | GT coverage | Mean latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | FF | 0 | 0.8186 | 0.5760 | 0.8432 | 1.0000 | 25.84 | 38.70 |
| yolo26n | SEG | 32 | 0.7492 | 0.5184 | 0.8423 | 0.9946 | 98.05 | 10.20 |
| yolo26n | SEG | 64 | 0.7977 | 0.5484 | 0.8582 | 1.0000 | 94.06 | 10.63 |
| yolo26n | SEG | 96 | 0.8147 | 0.5574 | 0.8640 | 1.0000 | 90.51 | 11.05 |
| yolo26n | SEG | 112 | 0.8206 | 0.5668 | 0.8632 | 1.0000 | 89.14 | 11.22 |
| yolo26n | SEG | 128 | 0.8176 | 0.5704 | 0.8552 | 1.0000 | 80.25 | 12.46 |
| yolo26n | SEG | 144 | 0.8170 | 0.5753 | 0.8503 | 1.0000 | 71.68 | 13.95 |
| yolo26n | SEG | 160 | 0.8204 | 0.5761 | 0.8524 | 1.0000 | 68.46 | 14.61 |
| roi_ssd_mobilenet | FF | 0 | 0.7410 | 0.4278 | 0.7605 | 1.0000 | 22.37 | 44.70 |
| roi_ssd_mobilenet | SEG | 32 | 0.6125 | 0.3703 | 0.6237 | 0.8375 | 25.61 | 39.05 |
| roi_ssd_mobilenet | SEG | 64 | 0.5951 | 0.3015 | 0.6267 | 0.9536 | 38.73 | 25.82 |
| roi_ssd_mobilenet | SEG | 96 | 0.6753 | 0.3645 | 0.7033 | 0.9661 | 37.22 | 26.87 |
| roi_ssd_mobilenet | SEG | 112 | 0.6860 | 0.3808 | 0.7139 | 0.9750 | 41.01 | 24.39 |
| roi_ssd_mobilenet | SEG | 128 | 0.7262 | 0.4118 | 0.7473 | 0.9964 | 38.55 | 25.94 |
| roi_ssd_mobilenet | SEG | 144 | 0.7334 | 0.4038 | 0.7580 | 0.9839 | 36.91 | 27.09 |
| roi_ssd_mobilenet | SEG | 160 | 0.7137 | 0.4116 | 0.7438 | 0.9804 | 38.23 | 26.16 |

Sources: [padding sweep](padding_sweep.csv), [frame-level metrics](frame_level_metrics.csv), [preflight](stage2_preflight.json).

## Locked padding decisions

| Model | Final padding | Criterion met | Eligible values |
|---|---:|:---:|---|
| yolo26n | 64 | True | [64, 96, 112, 128, 144, 160] |
| roi_ssd_mobilenet | 128 | True | [128, 144, 160] |

## Frozen pilot parameters

YOLO_PADDING_FINAL = 64
ROISSD_VGG_PADDING_FINAL = NOT RUN
ROISSD_MOBILENET_PADDING_FINAL = 128
YOLO_CPU_K = NOT RUN
YOLO_CPU_c = NOT RUN
YOLO_CPU_tau = NOT RUN
ROISSD_MOBILENET_CPU_K = NOT RUN
ROISSD_MOBILENET_CPU_c = NOT RUN
ROISSD_MOBILENET_CPU_tau = NOT RUN
