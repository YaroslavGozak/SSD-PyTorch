# Stage 3 CPU pilot

Run commands from the repository root in the activated project environment:

```powershell
.venv\Scripts\python.exe -m experiments.cpu_pilot.prepare
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage confidence
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage smoke --max-frames 20
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage padding
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage cost
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage pipeline
```

By default, the Stage 3 CPU pilot runs YOLO26n and ROI-SSD-MobileNet on the same
locked video-disjoint pilot. Preparation snapshots those configs and verifies
their checkpoint hashes. `evaluation_conf` defaults to 0.01. Run `confidence`
to select and freeze an `operational_conf` separately for every model before
smoke, padding, cost, pipeline, or all. ROI-SSD with the
VGG16 backbone is optional and should be run separately in its own output
directory, so its longer CPU run cannot overwrite the paired experiment:

```powershell
.venv\Scripts\python.exe -m experiments.cpu_pilot.prepare --output-dir final_experiments\pilot_roi_ssd_vgg16 --models roi_ssd
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_roi_ssd_vgg16 --models roi_ssd --stage confidence
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_roi_ssd_vgg16 --models roi_ssd --stage smoke --max-frames 20
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_roi_ssd_vgg16 --models roi_ssd --stage padding
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_roi_ssd_vgg16 --models roi_ssd --stage cost
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --output-dir final_experiments\pilot_roi_ssd_vgg16 --models roi_ssd --stage pipeline
```

The smoke stage is short and writes
diagnostic overlays only. `padding` tests p=32/64/96/112/128/144/160/192/224 and selects
the smallest padding satisfying both accuracy/coverage criteria, falling back to
the largest tested padding if none qualify. `cost` measures
the actual network input shapes with the selected project model wrappers and
fits `T(A) = K + cA`. The CPU target is the Intel Core i7-11700K; the runner
uses 8 PyTorch intra-op threads, 1 inter-op thread, and 1 OpenCV thread.
Smoke, padding sweep, and cost calibration use `evaluation_conf` (default
`0.01`) for evaluation, while operational state and segment formation always
use each model's frozen `operational_conf`. The `confidence` stage selects
that value by maximum micro F1@IoU=0.5 over the candidate grid, with ties
broken by higher recall then lower threshold. The grid is 0.01, then 0.05 to
0.95 in steps of 0.05. Pipeline modes FF/SEG/CA/FULL-TECH
also use that same operational threshold. Override the operational value only
for an explicitly requested diagnostic/run:

```powershell
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage padding --evaluation-conf 0.02
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage smoke --operational-conf 0.35
```

For evaluation metrics, detections are retained from `evaluation_conf` and
scored separately; only detections at or above `operational_conf` update the
tracker and form next-frame segments. Each frame and summary records both
thresholds and both detection counts. The same single detector invocation and
shared NMS output feed both branches; evaluation-only filtering and metric
calculation are excluded from process-frame timing. `--stage all` requires
operational confidence already frozen by `--stage confidence` (or an explicit
`--operational-conf` override).
The run metadata, report, and benchmark CSVs record CPU model, physical/logical
cores, PyTorch/OpenCV thread counts, OS, Python, and PyTorch versions.
`pipeline` requires both prior stages and executes FF,
SEG, CA, and FULL-TECH.

Use `--max-frames N` only for a smoke/debug run. Do not treat partial-run
summaries as final pilot results. `--stage all` is the full run; use it only
after reviewing the outputs of the preceding stages.

The locked manifests, copied model configs, outputs, and reports are stored in
the selected output directory (default `final_experiments/pilot/`). The candidate
evaluation split excludes pilot video IDs and is not an untouched test set
because the source validation data was previously reused for model selection
and diagnostics.

## Stage 3a oracle diagnosis

The failed segment-accuracy gate remains in effect. Diagnose it without running
CA/FULL-TECH or Pi:

```powershell
.venv\Scripts\python.exe -m unittest discover -s experiments/cpu_pilot/tests -v
.venv\Scripts\python.exe -m experiments.cpu_pilot.oracle_diagnosis
.venv\Scripts\python.exe -m experiments.cpu_pilot.summarize_oracle_diagnosis
```

The diagnosis replays FF and PRED_SEG against all 640 saved pilot prediction
records, then evaluates GT-generated crops on every frame at p=96. Padding uses
the pilot's 320x320 tensor units and identical coordinate conversions. The short
p=32/64/128 context probes and a no-global-NMS variant are diagnostic contrasts;
they do not update the pilot padding choice or acceptance thresholds.

Artifacts, per-object geometry, frame records, and 40 comparison overlays are
written exclusively to `final_experiments/pilot_oracle_diagnosis/`. Original
pilot artifacts and production sources are hashed before and after execution.

## Stage 3b controlled crop-domain study

```powershell
.venv\Scripts\python.exe -m experiments.cpu_pilot.crop_domain
.venv\Scripts\python.exe -m experiments.cpu_pilot.crop_training_proxy
.venv\Scripts\python.exe -m experiments.cpu_pilot.crop_domain_validate
.venv\Scripts\python.exe -m experiments.cpu_pilot.crop_domain_report
```

Results are in `final_experiments/crop_domain_diagnosis/`: five paired paddings,
neutral outside-frame padding, fixed-ROI preprocessing controls, per-head
anchor/score/decoded geometry, and eight plots. The historical training sample
is explicitly a reconstruction from September source: the checkpoint does not
embed an exact training revision or crop manifest. It must not be presented as
the actual 1,000 crops observed during training. Current October sampler code
is not substituted as checkpoint-training provenance.

The preprocessing arms are declared before evaluation: source crop resized to
current H/W, to 224x224 (historical large-ROI mode), and to 300x300 (historical
full-output-size control). Their results do not select a production resize or
padding. All existing pipeline gates remain in effect.
