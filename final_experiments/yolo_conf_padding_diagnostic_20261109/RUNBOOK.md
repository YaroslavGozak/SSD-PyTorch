# Runbook

Run these commands from the repository root on the target Intel Core i9-9900KF
using CPU only. The pinned source pilot must remain unchanged:

`final_experiments\pilot_dual_conf_new\pilot_manifest.txt`

Its expected SHA-256 is
`817f5b145b774dc91ee0fa8d271e02c11d71d70bb50ff13082a0ba9487aa63cb`
(32 videos, 320 frames, 560 GT objects). The script also verifies the
checkpoint and model-config hashes before inference.

## 1. Short smoke test

This tests 12 manifest-order frames with operational confidence 0.05 and
padding 128. It checks that evaluation detections are a superset of operational
detections and that tracker inputs match operational detections exactly:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_yolo_conf_padding_diagnostic --device cpu --pilot-dir final_experiments\pilot_dual_conf_new --output final_experiments\yolo_conf_padding_diagnostic_20261109 --evaluation-conf 0.01 --smoke-only --smoke-frames 12 --smoke-operational-conf 0.05 --smoke-padding 128
```

The smoke test does not write sweep results. It writes or verifies
`resolved_config.yaml`.

## 2. Full 30-configuration sweep (manual)

Do not start this command until the smoke test passes. It runs one FF baseline
and then the five confidence thresholds crossed with six paddings:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_yolo_conf_padding_diagnostic --device cpu --pilot-dir final_experiments\pilot_dual_conf_new --output final_experiments\yolo_conf_padding_diagnostic_20261109 --evaluation-conf 0.01 --operational-confs 0.01 0.05 0.10 0.20 0.35 --paddings 64 96 128 160 192 224
```

The result directory is protected from accidental reruns. If result files
already exist, the command stops instead of overwriting or resuming; choose a
fresh `--output` directory for another full run.

## 3. Regenerate analysis only

After a successful sweep, regenerate the summary, gate matrix, and report
without loading the model or rerunning inference:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_yolo_conf_padding_diagnostic --device cpu --pilot-dir final_experiments\pilot_dual_conf_new --output final_experiments\yolo_conf_padding_diagnostic_20261109 --analyze-only
```

## Runtime artifacts

The sweep creates:

- `results\full_frame_baseline.json`
- `results\sweep.csv`
- `results\frame_level.csv`
- `analysis\summary.csv`
- `analysis\gate_matrix.csv`
- `analysis\diagnostic_report.md`

The report includes the 5-by-6 gate matrix, coverage and mAP retention for all
configurations, temporal ranges of consecutive frames with uncovered GT,
miss counts for video 23010 frames 83-89 when present, factor comparisons, and
a recommendation based first on passing both gates and then on tensor area,
candidate segments, padding, and operational confidence. The recommendation
is diagnostic only; it does not modify frozen confidence values or the main
Stage 3 outputs.
