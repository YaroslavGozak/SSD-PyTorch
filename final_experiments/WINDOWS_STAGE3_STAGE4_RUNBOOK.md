# Windows Stage 3 / Stage 4 Runbook

This runbook runs the frozen Stage 3 and Stage 4 CPU experiments on Windows.
It preserves the selected checkpoints, manifests, thresholds, and padding.
The dataset, checkpoints, and outputs may be on different drives.

The Stage 3 launcher (`experiments.cpu_pilot.pi_runtime`) is portable despite
its historical module name. `experiments.pi_preflight` is Raspberry Pi-only
and explicitly rejects Windows; do not use it as the Windows preflight.

## 1. Set local paths

Use stable, local drive letters. The repository, dataset, models, and outputs
do not need to be on the same drive. Avoid removable or cloud-synced drives
during a run.

```powershell
$Repo = (Resolve-Path 'D:\Repositories\Other\SSD-PyTorch').Path
$DatasetRoot = 'D:\Repositories\Other\imagenet_vid_yolo_sampled'
$DatasetYaml = Join-Path $DatasetRoot 'imagenet_vid.yaml'
$ModelsRoot = 'D:\Repositories\Other\SSD-PyTorch\trained_models'
$Stage3Dir = Join-Path $Repo 'final_experiments\pilot_windows'
$OutputRoot = 'D:\Repositories\Other\SSD-PyTorch\outputs\stage4_windows'

$env:SSD_DATASET_ROOT = $DatasetRoot
$env:SSD_DATASET_YAML = $DatasetYaml
$env:SSD_MODELS_ROOT = $ModelsRoot
$env:SSD_STAGE3_DIR = $Stage3Dir
$env:SSD_OUTPUT_ROOT = $OutputRoot
$env:SSD_INFERENCE_DEVICE = 'cpu'

Set-Location $Repo
Test-Path $DatasetRoot
Test-Path $DatasetYaml
Test-Path $ModelsRoot
```

`SSD_DATASET_ROOT` must be the dataset root, not the drive root. The portable
runtime rewrites absolute Windows split paths in the dataset YAML into paths
relative to this root. Do not edit frozen configs or manifests to change drive
letters.

## 2. Record the source revision and check Python

Use the exact source revision intended for the experiment. Record the commit
and any local changes; do not discard unrelated work. Uncommitted changes will
be reflected in Stage 4 provenance.

```powershell
git rev-parse HEAD
git status --short
python --version
python -c "import torch, torchvision, cv2, yaml, ultralytics; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available()); assert not torch.cuda.is_available()"
python -m pip check
```

Use 64-bit Windows Python and CPU-only PyTorch. The repository requirements
files conflict, so do not install either wholesale without verifying its
versions. A prior Windows pilot recorded Python 3.12.10 and CPU-only
PyTorch 2.12.0 / torchvision 0.27.0. Confirm all project imports and CPU
execution before proceeding.

## 3. Export frozen Stage 3 inputs

Use a new Stage 3 directory. The exporter validates the frozen inputs and
copies only what Stage 3/4 need; desktop timing profiles and smoke outputs are
not included.

```powershell
if (Test-Path $Stage3Dir) {
    if ((Get-ChildItem -Force $Stage3Dir | Measure-Object).Count -ne 0) {
        throw "Stage 3 directory must be new or empty: $Stage3Dir"
    }
} else {
    New-Item -ItemType Directory -Path $Stage3Dir | Out-Null
}

python -m experiments.cpu_pilot.export_pi_inputs --output-dir $Stage3Dir
```

The exporter is named for the Pi workflow but can be run on Windows. It does
not make the Pi preflight Windows-compatible. Keep generated outputs out of
source control; if needed, add `/final_experiments/pilot_windows/` to
`.git/info/exclude`.

## 4. Place and verify the selected checkpoints

Put the exact checkpoint files here:

```text
<ModelsRoot>\yolo26n\best.pt
<ModelsRoot>\roi_ssd_mobilenet\best_map.pt
```

Verify their hashes:

```powershell
Get-FileHash (Join-Path $ModelsRoot 'yolo26n\best.pt') -Algorithm SHA256
Get-FileHash (Join-Path $ModelsRoot 'roi_ssd_mobilenet\best_map.pt') -Algorithm SHA256
```

Expected SHA-256:

| Model | SHA-256 |
|---|---|
| YOLO26n | `5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027` |
| ROI-SSD-MobileNetV3 | `1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804` |

The portable launcher also requires the canonical repository checkpoint paths
to resolve to these files. If they are not already linked, create file
symlinks as below. Windows may require Developer Mode or an elevated
PowerShell session. If a destination already exists, inspect it; do not
overwrite it.

```powershell
$YoloCanonical = Join-Path $Repo 'trained_models\best-imagenet-vid-yolo\best.pt'
$SsdCanonical = Join-Path $Repo 'trained_models\imagenet-vid-roissd-mobilenet-20261027\stage3\best_map.pt'

New-Item -ItemType Directory -Force (Split-Path $YoloCanonical) | Out-Null
New-Item -ItemType Directory -Force (Split-Path $SsdCanonical) | Out-Null

New-Item -ItemType SymbolicLink -Path $YoloCanonical `
  -Target (Join-Path $ModelsRoot 'yolo26n\best.pt')
New-Item -ItemType SymbolicLink -Path $SsdCanonical `
  -Target (Join-Path $ModelsRoot 'roi_ssd_mobilenet\best_map.pt')
```

## 5. Set and preserve the thread policy

Choose thread counts before starting and keep them unchanged through Stage 3
and Stage 4. These are example values:

```powershell
$env:SSD_CPU_THREADS = '8'
$env:SSD_INTEROP_THREADS = '1'
$env:SSD_OPENCV_THREADS = '1'
```

If you change a count, regenerate cost profiles and repeat the pipeline and
benchmark measurements. Preserve earlier results separately.

## 6. Run Stage 3 smoke, cost profile, and pipeline check

Run from the repository root. Save each model's smoke output before the next
invocation because the launcher writes the same report filenames each time.
The selected operational confidences are 0.10 for YOLO26n and 0.50 for
ROI-SSD-MobileNetV3.

```powershell
python -m experiments.cpu_pilot.pi_runtime --stage smoke --max-frames 20 `
  --models yolo26n --operational-conf 0.10 --output-dir $Stage3Dir
Copy-Item "$Stage3Dir\smoke_metadata.json" "$Stage3Dir\smoke_metadata_yolo26n.json"
Copy-Item "$Stage3Dir\smoke_summary.csv" "$Stage3Dir\smoke_summary_yolo26n.csv"

python -m experiments.cpu_pilot.pi_runtime --stage smoke --max-frames 20 `
  --models roi_ssd_mobilenet --operational-conf 0.50 --output-dir $Stage3Dir
Copy-Item "$Stage3Dir\smoke_metadata.json" "$Stage3Dir\smoke_metadata_roi_ssd_mobilenet.json"
Copy-Item "$Stage3Dir\smoke_summary.csv" "$Stage3Dir\smoke_summary_roi_ssd_mobilenet.csv"
```

After both smokes pass, profile both models on this Windows machine, then run
the short pipeline validation:

```powershell
python -m experiments.cpu_pilot.pi_runtime --stage cost `
  --cost-repetitions 8 --output-dir $Stage3Dir

python -m experiments.cpu_pilot.pi_runtime --stage pipeline --max-frames 20 `
  --no-debug --output-dir $Stage3Dir
```

Review the cost profiles, smoke reports, and `pipeline_smoke_summary.csv`.
Stop if either model fails or the recorded thread counts differ from the
selected policy. Do not reuse cost profiles from another host.

## 7. Validate and run Stage 4

Use a fresh output directory outside the repository. Do not use a `raspberry`
device label: Stage 4 treats it as a Raspberry Pi label. Two timing repeats
are only permitted for Pi labels, so use at least three repeats on Windows.
Use `windows-core-i9-9900kf` only if that is the actual CPU; otherwise choose
a stable Windows-specific label.

```powershell
$DeviceLabel = 'windows-core-i9-9900kf'

python -m experiments.final_end_to_end_benchmark --dry-run `
  --output-root $OutputRoot `
  --dataset-yaml $DatasetYaml `
  --device-label $DeviceLabel `
  --timing-repeats 3
```

Only continue if the dry run validates the frozen inputs. Generate fresh
Stage 4 profiles:

```powershell
python -m experiments.final_end_to_end_benchmark --stage calibrate `
  --output-root $OutputRoot `
  --dataset-yaml $DatasetYaml `
  --device-label $DeviceLabel `
  --timing-repeats 3 `
  --calibration-repetitions 8
```

Inspect `cost_profiles_final.json` and confirm both models and the expected
runtime metadata before benchmarking. Then run all four modes:

```powershell
python -m experiments.final_end_to_end_benchmark --stage benchmark `
  --output-root $OutputRoot `
  --dataset-yaml $DatasetYaml `
  --device-label $DeviceLabel `
  --timing-repeats 3 `
  --system-load-note "Windows CPU run; 8 PyTorch threads"
```

Use the same output root and device label for calibration and benchmarking.
Use a new output root if calibration must be repeated; do not overwrite frozen
profiles.

## 8. Windows measurement notes and output review

The current hardware helpers read temperature and CPU frequency from Linux
sysfs and Raspberry Pi throttling through `vcgencmd`. Those fields may be
blank on Windows. The active power scheme is queried with `powercfg`. This
does not prevent the benchmark, but note the limitation and independently
record temperature/frequency observations if required for comparison.

Review the device output folder, especially `manifest_provenance.json`,
`cost_profiles_final.json`, `hardware_software_environment.md`,
`final_summary.csv`, `timing_runs.csv`, `frame_level_metrics.csv`, and any
crash/error evidence. Keep Stage 3 artifacts and Stage 4 outputs separate.
