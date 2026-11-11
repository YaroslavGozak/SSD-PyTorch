# YOLO26n operational-confidence x padding diagnostic

This isolated diagnostic measures how YOLO26n operational confidence and static
ROI padding affect Stage 3 segment evaluation and temporal GT coverage. It uses
the already-selected Stage 3 checkpoint and the locked pilot manifest in
`final_experiments/pilot_dual_conf_new/`. It does not create a new sample, alter
the main Stage 3 configuration, or write to `final_experiments/pilot/`.

The evaluation confidence is fixed at `0.01`. Only operational detections feed
the previous-localization tracker. The sweep uses SEG mode, zero tracker hold,
a periodic full-frame interval of 10, and disabled merge. One full-frame
baseline is evaluated once, followed by 30 operational-confidence/padding
combinations.

Run the short smoke test and the complete manual sweep using the commands in
[RUNBOOK.md](./RUNBOOK.md). The full sweep is intentionally not run as part of
preparing this diagnostic. If any result files already exist, the sweep
refuses to overwrite or silently resume them; use a fresh output directory.

Expected output files:

- `resolved_config.yaml` — locked model, checkpoint, class list, manifest, and
  diagnostic protocol.
- `results/full_frame_baseline.json` — the single evaluation-confidence 0.01
  full-frame baseline.
- `results/sweep.csv` — one row per configuration, including separate mAP,
  coverage, and final gate flags.
- `results/frame_level.csv` — per-frame detection counts, candidate segments,
  fallback details, and uncovered GT class/bboxes.
- `analysis/summary.csv` — per-configuration temporal miss summaries.
- `analysis/gate_matrix.csv` — PASS/FAIL matrix indexed by operational
  confidence and padding.
- `analysis/diagnostic_report.md` — gate matrix, factor comparisons, temporal
  failure ranges, and a gate-constrained recommendation.

After a completed sweep, rerun report generation without inference using the
`--analyze-only` command in the runbook.
