# Stage 3 CPU pilot

Run commands from the repository root in the activated project environment:

```powershell
.venv\Scripts\python.exe -m experiments.cpu_pilot.prepare
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage smoke --max-frames 20
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage padding
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage cost
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage pipeline
```

The smoke stage is short and writes diagnostic overlays only. `padding` runs the
predeclared 32/64/96 sweep and selects the smallest padding satisfying both
accuracy/coverage criteria, falling back to 96 if none qualify. `cost` measures
the actual network input shapes with the selected project model wrappers and
fits `T(A) = K + cA`. The CPU target is the Intel Core i7-11700K; the runner
uses 8 PyTorch intra-op threads, 1 inter-op thread, and 1 OpenCV thread.
`pipeline` requires both prior stages and executes FF,
SEG, CA, and FULL-TECH.

Use `--max-frames N` only for a smoke/debug run. Do not treat partial-run
summaries as final pilot results. `--stage all` is the full run; use it only
after reviewing the outputs of the preceding stages.

The locked manifests, copied model configs, outputs, and reports are stored in
`final_experiments/pilot/`. The candidate evaluation split excludes pilot video
IDs and is not an untouched test set because the source validation data was
previously reused for model selection and diagnostics.

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
