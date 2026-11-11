# Stage 3 CPU pilot

## Preparation status

`PREPARED_NOT_BENCHMARKED`. The full padding sweep, fresh CPU cost calibration,
and FF/SEG/CA/FULL-TECH runs have not been executed.

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
- All three models use the locked inference confidence threshold
  `0.01`; YOLO output cap/NMS are locked to 200 detections and
  IoU 0.45. CPU threads are fixed to 8 (the i7-11700K core
  count); OpenCV threads to 1.

## Smoke check

The 20-frame CPU coordinate smoke is run separately. Its outcome is recorded in
[`smoke_metadata.json`](./smoke_metadata.json); smoke metrics are not pilot results.

## Remaining protocol

1. Run `python -m experiments.cpu_pilot.run_pilot --stage padding`.
2. Run `python -m experiments.cpu_pilot.run_pilot --stage cost`.
3. Review the locked padding decision and fresh CPU cost fits.
4. Run `python -m experiments.cpu_pilot.run_pilot --stage pipeline`.

No padding decision is final until the full sweep passes the predeclared accuracy/coverage rule.
CPU cost coefficients, when measured, remain preliminary until the pipeline acceptance checks pass.
