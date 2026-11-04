# Stage 3 CPU pilot

## Preparation status

`BLOCKED_PIPELINE_ACCEPTANCE`. Completed stages: ['coordinate_smoke', 'cpu_cost_calibration', 'padding_sweep']. The full padding sweep is complete. CA/FULL-TECH pipeline runs have not been completed.

**Execution stop:** STOP: roi_ssd_mobilenet has no tested padding satisfying both segment/full-frame mAP50 >= 0.95 and GT coverage >= 0.98. Investigate before running CA/FULL-TECH.

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
| yolo26n | FF | 0 | 0.8186 | 0.5760 | 0.8432 | 1.0000 | 19.62 | 50.98 |
| yolo26n | SEG | 32 | 0.7492 | 0.5184 | 0.8423 | 0.9946 | 74.04 | 13.51 |
| yolo26n | SEG | 64 | 0.7977 | 0.5484 | 0.8582 | 1.0000 | 75.04 | 13.33 |
| yolo26n | SEG | 96 | 0.8147 | 0.5574 | 0.8640 | 1.0000 | 77.16 | 12.96 |
| roi_ssd_mobilenet | FF | 0 | 0.7410 | 0.4278 | 0.7605 | 1.0000 | 24.26 | 41.22 |
| roi_ssd_mobilenet | SEG | 32 | 0.6125 | 0.3703 | 0.6237 | 0.8375 | 28.00 | 35.72 |
| roi_ssd_mobilenet | SEG | 64 | 0.5951 | 0.3015 | 0.6267 | 0.9536 | 31.19 | 32.06 |
| roi_ssd_mobilenet | SEG | 96 | 0.6753 | 0.3645 | 0.7033 | 0.9661 | 32.96 | 30.34 |

Sources: [padding sweep](padding_sweep.csv), [frame-level metrics](frame_level_metrics.csv), [preflight](stage2_preflight.json).

## Locked padding decisions

| Model | Final padding | Criterion met | Eligible values |
|---|---:|:---:|---|
| yolo26n | 64 | True | [64, 96] |
| roi_ssd_mobilenet | 96 | False | [] |

## Fresh CPU cost profiles

| Model | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) |
|---|---:|---:|---:|---:|---:|---:|
| yolo26n | 7.97816 | 0.000128515 | 62079.679 | 0.8477 | 1.1219 | 1.6807 |
| roi_ssd_mobilenet | 8.05867 | 0.000149489 | 53908.189 | 0.8926 | 1.2866 | 1.6002 |

## Frozen pilot parameters

YOLO_PADDING_FINAL = 64
ROISSD_MOBILENET_PADDING_FINAL = 96
YOLO_CPU_K = 7.978164064036682
YOLO_CPU_c = 0.0001285149061155904
YOLO_CPU_tau = 62079.67857721398
ROISSD_MOBILENET_CPU_K = 8.05867302587487
ROISSD_MOBILENET_CPU_c = 0.00014948884775687645
ROISSD_MOBILENET_CPU_tau = 53908.188783294536
