# Stage 3 CPU pilot

## Preparation status

`PIPELINE_ERROR`. Completed stages: ['coordinate_smoke', 'cpu_cost_calibration', 'padding_sweep']. The full padding sweep is complete. CA/FULL-TECH pipeline runs have not been completed.

**Execution stop:** Cost-aware merging requires measured effective tensor shapes

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

Investigate the failed MobileNet segment-accuracy/coverage criteria before rerunning CA/FULL-TECH. Do not relax thresholds or substitute checkpoints post hoc. Stage 4 / Raspberry Pi runs remain on hold.

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
| yolo26n | FF | 0 | 0.8186 | 0.5760 | 0.8432 | 1.0000 | 25.02 | 39.97 |
| yolo26n | SEG | 32 | 0.7492 | 0.5184 | 0.8423 | 0.9946 | 94.50 | 10.58 |
| yolo26n | SEG | 64 | 0.7977 | 0.5484 | 0.8582 | 1.0000 | 91.92 | 10.88 |
| yolo26n | SEG | 96 | 0.8147 | 0.5574 | 0.8640 | 1.0000 | 93.99 | 10.64 |
| yolo26n | SEG | 112 | 0.8206 | 0.5668 | 0.8632 | 1.0000 | 96.78 | 10.33 |
| yolo26n | SEG | 128 | 0.8176 | 0.5704 | 0.8552 | 1.0000 | 86.63 | 11.54 |
| yolo26n | SEG | 144 | 0.8170 | 0.5753 | 0.8503 | 1.0000 | 77.88 | 12.84 |
| yolo26n | SEG | 160 | 0.8204 | 0.5761 | 0.8524 | 1.0000 | 72.90 | 13.72 |
| roi_ssd_mobilenet | FF | 0 | 0.7410 | 0.4278 | 0.7605 | 1.0000 | 26.34 | 37.96 |
| roi_ssd_mobilenet | SEG | 32 | 0.6125 | 0.3703 | 0.6237 | 0.8375 | 30.15 | 33.17 |
| roi_ssd_mobilenet | SEG | 64 | 0.5951 | 0.3015 | 0.6267 | 0.9536 | 37.44 | 26.71 |
| roi_ssd_mobilenet | SEG | 96 | 0.6753 | 0.3645 | 0.7033 | 0.9661 | 39.53 | 25.30 |
| roi_ssd_mobilenet | SEG | 112 | 0.6860 | 0.3808 | 0.7139 | 0.9750 | 43.52 | 22.98 |
| roi_ssd_mobilenet | SEG | 128 | 0.7262 | 0.4118 | 0.7473 | 0.9964 | 41.90 | 23.87 |
| roi_ssd_mobilenet | SEG | 144 | 0.7334 | 0.4038 | 0.7580 | 0.9839 | 43.44 | 23.02 |
| roi_ssd_mobilenet | SEG | 160 | 0.7137 | 0.4116 | 0.7438 | 0.9804 | 40.61 | 24.62 |

Sources: [padding sweep](padding_sweep.csv), [frame-level metrics](frame_level_metrics.csv), [preflight](stage2_preflight.json).

## Locked padding decisions

| Model | Final padding | Criterion met | Eligible values |
|---|---:|:---:|---|
| yolo26n | 64 | True | [64, 96, 112, 128, 144, 160] |
| roi_ssd_mobilenet | 128 | True | [128, 144, 160] |

## Fresh CPU cost profiles

| Model | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) |
|---|---:|---:|---:|---:|---:|---:|
| yolo26n | 8.55664 | 0.000160886 | 53184.555 | 0.9667 | 0.8062 | 0.9217 |
| roi_ssd_mobilenet | 9.67932 | 0.000182902 | 52920.680 | 0.8926 | 1.6343 | 1.9583 |

## Frozen pilot parameters

YOLO_PADDING_FINAL = 64
ROISSD_MOBILENET_PADDING_FINAL = 128
YOLO_CPU_K = 8.556641460408814
YOLO_CPU_c = 0.00016088583232070242
YOLO_CPU_tau = 53184.55538926758
ROISSD_MOBILENET_CPU_K = 9.67931572057076
ROISSD_MOBILENET_CPU_c = 0.00018290232983781814
ROISSD_MOBILENET_CPU_tau = 52920.68028413598
