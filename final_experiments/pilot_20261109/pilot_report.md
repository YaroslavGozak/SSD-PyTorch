# Stage 3 CPU pilot

## Preparation status

`BLOCKED_PIPELINE_ACCEPTANCE`. Completed stages: ['cpu_cost_calibration', 'operational_confidence_selection', 'padding_sweep']. The full padding sweep is complete. CA/FULL-TECH pipeline runs have not been completed.

**Execution stop:** STOP: yolo26n has no tested padding satisfying both segment/full-frame mAP50 >= 0.95 and GT coverage >= 0.98. Investigate before running CA/FULL-TECH.

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
- Selected models use default evaluation_conf
  `0.01`; choose per-model operational_conf with `--stage confidence`.
  YOLO output cap/NMS are locked to 200 detections and
  IoU 0.45. CPU threads are fixed to 8 (the i7-11700K core
  count); OpenCV threads to 1.

## Smoke check

The 20-frame CPU coordinate smoke is run separately. Its outcome is recorded in
[`smoke_metadata.json`](./smoke_metadata.json); smoke metrics are not pilot results.

## Remaining protocol

Investigate failed segment-accuracy/coverage criteria for every model before rerunning CA/FULL-TECH. Do not relax thresholds or substitute checkpoints post hoc. Stage 4 / Raspberry Pi runs remain on hold.

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

## Frozen operational confidence

Evaluation floor: 0.01.
maximum micro F1@IoU=0.5 on full-frame pilot using evaluation_conf detections; ties: higher recall, then lower threshold
Candidate grid: [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].

| Model | Operational confidence | Micro F1@0.5 | Recall |
|---|---:|---:|---:|
| yolo26n | 0.3500 | 0.828654 | 0.764286 |
| roi_ssd_mobilenet | 0.2500 | 0.835629 | 0.812500 |

Candidates are scored using detections retained at evaluation_conf=0.01; ties resolve by higher recall, then lower threshold.
Decision and evidence hashes: [operational_confidence.json](operational_confidence.json).

## Measured full-frame and padding results

All entries use the locked pilot; image loading and metric computation are excluded from process-frame timing.

| Model | Mode | Padding | evaluation_conf | operational_conf | mAP50 | mAP50:95 | Recall50 | GT coverage | Mean latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | FF | 0 | 0.0100 | 0.3500 | 0.8186 | 0.5760 | 0.8432 | 1.0000 | 21.70 | 46.08 |
| yolo26n | SEG | 32 | 0.0100 | 0.3500 | 0.7577 | 0.5233 | 0.8075 | 0.9304 | 23.23 | 43.05 |
| yolo26n | SEG | 64 | 0.0100 | 0.3500 | 0.8018 | 0.5533 | 0.8373 | 0.9625 | 26.59 | 37.60 |
| yolo26n | SEG | 96 | 0.0100 | 0.3500 | 0.8172 | 0.5698 | 0.8410 | 0.9625 | 28.26 | 35.39 |
| yolo26n | SEG | 112 | 0.0100 | 0.3500 | 0.8242 | 0.5703 | 0.8462 | 0.9625 | 30.38 | 32.92 |
| yolo26n | SEG | 128 | 0.0100 | 0.3500 | 0.8124 | 0.5730 | 0.8363 | 0.9625 | 30.20 | 33.11 |
| yolo26n | SEG | 144 | 0.0100 | 0.3500 | 0.8159 | 0.5728 | 0.8367 | 0.9625 | 30.74 | 32.54 |
| yolo26n | SEG | 160 | 0.0100 | 0.3500 | 0.8163 | 0.5736 | 0.8367 | 0.9625 | 30.58 | 32.70 |
| yolo26n | SEG | 192 | 0.0100 | 0.3500 | 0.8173 | 0.5724 | 0.8373 | 0.9696 | 29.90 | 33.45 |
| yolo26n | SEG | 224 | 0.0100 | 0.3500 | 0.8181 | 0.5739 | 0.8373 | 0.9750 | 30.82 | 32.45 |
| roi_ssd_mobilenet | FF | 0 | 0.0100 | 0.2500 | 0.8377 | 0.4626 | 0.8816 | 1.0000 | 22.45 | 44.54 |
| roi_ssd_mobilenet | SEG | 32 | 0.0100 | 0.2500 | 0.7251 | 0.4039 | 0.8306 | 0.9089 | 33.00 | 30.30 |
| roi_ssd_mobilenet | SEG | 64 | 0.0100 | 0.2500 | 0.7688 | 0.3629 | 0.9012 | 0.9839 | 42.46 | 23.55 |
| roi_ssd_mobilenet | SEG | 96 | 0.0100 | 0.2500 | 0.7777 | 0.3984 | 0.8971 | 1.0000 | 46.38 | 21.56 |
| roi_ssd_mobilenet | SEG | 112 | 0.0100 | 0.2500 | 0.7943 | 0.4179 | 0.8940 | 0.9946 | 45.54 | 21.96 |
| roi_ssd_mobilenet | SEG | 128 | 0.0100 | 0.2500 | 0.8207 | 0.4408 | 0.9222 | 1.0000 | 43.48 | 23.00 |
| roi_ssd_mobilenet | SEG | 144 | 0.0100 | 0.2500 | 0.8175 | 0.4300 | 0.9092 | 1.0000 | 42.75 | 23.39 |
| roi_ssd_mobilenet | SEG | 160 | 0.0100 | 0.2500 | 0.8120 | 0.4466 | 0.9002 | 1.0000 | 45.94 | 21.77 |
| roi_ssd_mobilenet | SEG | 192 | 0.0100 | 0.2500 | 0.8352 | 0.4585 | 0.9088 | 0.9929 | 40.39 | 24.76 |
| roi_ssd_mobilenet | SEG | 224 | 0.0100 | 0.2500 | 0.8370 | 0.4651 | 0.9073 | 1.0000 | 42.43 | 23.57 |

Sources: [padding sweep](padding_sweep.csv), [frame-level metrics](frame_level_metrics.csv), [preflight](stage2_preflight.json).

## Locked padding decisions

| Model | evaluation_conf | operational_conf | Final padding | Criterion met | Eligible values |
|---|---:|---:|---:|:---:|---|
| yolo26n | 0.0100 | 0.3500 | 224 | False | [] |
| roi_ssd_mobilenet | 0.0100 | 0.2500 | 128 | True | [128, 144, 160, 192, 224] |

## Fresh CPU cost profiles

Timing mode: shared detector call at evaluation_conf; quality-only filtering/metrics are not timed.

| Model | evaluation_conf | operational_conf | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | 0.0100 | 0.3500 | 6.79567 | 0.000139568 | 48690.756 | 0.9812 | 0.4538 | 0.5954 |
| roi_ssd_mobilenet | 0.0100 | 0.2500 | 8.57737 | 0.000136997 | 62609.743 | 0.8854 | 1.2029 | 1.5208 |

## Frozen pilot parameters

YOLO_OPERATIONAL_CONF = 0.35
ROISSD_OPERATIONAL_CONF = NOT SELECTED
ROISSD_MOBILENET_OPERATIONAL_CONF = 0.25
PADDING_EVALUATION_CONF = evaluation_conf (default 0.01)
PIPELINE_OPERATIONAL_CONF = per-model frozen value above
YOLO_PADDING_FINAL = 224
ROISSD_PADDING_FINAL = NOT RUN
ROISSD_MOBILENET_PADDING_FINAL = 128
YOLO_CPU_K = 6.7956694048755955
YOLO_CPU_c = 0.0001395679592423991
YOLO_CPU_tau = 48690.75568464105
ROISSD_CPU_K = NOT RUN
ROISSD_CPU_c = NOT RUN
ROISSD_CPU_tau = NOT RUN
ROISSD_MOBILENET_CPU_K = 8.577367902962315
ROISSD_MOBILENET_CPU_c = 0.00013699733352571174
ROISSD_MOBILENET_CPU_tau = 62609.74343236038
