# Stage 3 CPU pilot

The final Stage 3 workflow jointly calibrates each model's operational
confidence and segment padding, then profiles CPU cost and runs FF/SEG/CA/
FULL-TECH using the hash-locked selected configuration:

```powershell
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.prepare --output-dir final_experiments\pilot_joint_20261109 --models yolo26n roi_ssd_mobilenet
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models yolo26n roi_ssd_mobilenet --stage smoke --max-frames 20 --operational-conf 0.10
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models yolo26n roi_ssd_mobilenet --stage calibration
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models yolo26n roi_ssd_mobilenet --stage cost --cost-repetitions 8
.\.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_joint_20261109 --models yolo26n roi_ssd_mobilenet --stage pipeline
```

By default the primary model pair is YOLO26n and ROI-SSD-MobileNetV3. Config
snapshots include each model's `calibration.operational_conf_values` and
`calibration.padding_values`. Calibration runs one FF baseline and the
config-defined SEG grid per model with evaluation confidence 0.01, hold=0,
periodic full frame every 10 frames, and merge disabled.

The fixed quality gates are mAP50 retention >= 0.95 and mean GT ROI coverage
>= 0.98. Among passing configurations, selection minimizes tensor area ratio;
values within 1% of the minimum are ordered by fewer candidate segments/frame,
smaller padding, then higher operational confidence. No passing cell blocks
cost and pipeline for that model. Frozen decisions are stored under
`calibration/<model>/selected_operational_config.json` with provenance and
artifact hashes.

`cost` requires a passing, provenance-valid calibration and records the actual
CPU/software environment. `pipeline` requires the matching fresh cost profile
and uses the frozen evaluation threshold, operational threshold, and padding.
`all` runs calibration -> cost -> pipeline after a successful smoke; it never
runs smoke or confidence automatically.

The optional `confidence` stage is a detector-level F1 diagnostic only. Its
threshold is not the final operational confidence. The optional `padding`
stage is a fixed-confidence padding diagnostic and does not freeze final
padding. The CLI help describes all stage semantics.

Use `--max-frames` only for smoke/debug, never calibration or `all`. For
independent results, pass one model to `--models`, then use the same prepared
pilot output directory for its smoke, calibration, cost, and pipeline stages.
ROI-SSD/VGG16 is optional and should be prepared in its own output directory.

Full protocol, outputs, stop conditions, and exact PowerShell commands:
[`ai_docs/03_cpu_pilot_full_pipeline.md`](../../ai_docs/03_cpu_pilot_full_pipeline.md).
