# Stage 3 CPU pilot

## Preparation status

`OPERATIONAL_CONFIDENCE_FROZEN`. Completed stages: ['operational_confidence_selection']. The full padding sweep remains pending. CA/FULL-TECH pipeline runs have not been completed.

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

1. Run `python -m experiments.cpu_pilot.run_pilot --stage padding`.
2. Run `python -m experiments.cpu_pilot.run_pilot --stage cost`.

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

## Frozen pilot parameters

YOLO_OPERATIONAL_CONF = 0.35
ROISSD_OPERATIONAL_CONF = NOT SELECTED
ROISSD_MOBILENET_OPERATIONAL_CONF = 0.25
PADDING_EVALUATION_CONF = evaluation_conf (default 0.01)
PIPELINE_OPERATIONAL_CONF = per-model frozen value above
YOLO_PADDING_FINAL = NOT RUN
ROISSD_PADDING_FINAL = NOT RUN
ROISSD_MOBILENET_PADDING_FINAL = NOT RUN
YOLO_CPU_K = NOT RUN
YOLO_CPU_c = NOT RUN
YOLO_CPU_tau = NOT RUN
ROISSD_CPU_K = NOT RUN
ROISSD_CPU_c = NOT RUN
ROISSD_CPU_tau = NOT RUN
ROISSD_MOBILENET_CPU_K = NOT RUN
ROISSD_MOBILENET_CPU_c = NOT RUN
ROISSD_MOBILENET_CPU_tau = NOT RUN
