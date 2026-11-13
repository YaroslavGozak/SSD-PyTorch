# Raspberry Pi 5 Stage 3 / Stage 4 Runbook

This runbook moves the frozen experiment to a 64-bit, CPU-only Raspberry Pi 5.
It does not change the selected checkpoints, manifests, operational thresholds,
or padding. Stage 3 confidence-by-padding tuning is not run on the Pi.

## 1. Portable directory layout

Use this layout; the project root may be anywhere because paths are supplied
through local environment variables:

```text
~/object-detection/
  repo/                                  # cloned Git repository
    final_experiments/pilot_pi5/         # frozen inputs + fresh Pi Stage 3 outputs
  data/imagenet_vid_yolo_sampled/
    imagenet_vid.yaml
    images/
      train/
      val/
    labels/
      train/
      val/
  models/
    yolo26n/best.pt
    roi_ssd_mobilenet/best_map.pt
  outputs/stage4_pi5/                    # fresh, Pi-only Stage 4 outputs
```

## Path audit and portability behavior

| Surface | Desktop-specific or fragile path | Pi handling |
|---|---|---|
| Frozen Stage 3 YAML snapshots | `yolo_dataset_yaml` contains the Windows desktop YAML path | `SSD_DATASET_YAML` overrides it into a temporary runtime config; the frozen YAML and its hash are not changed |
| Stage 3 `split_metadata.json` | selected checkpoint identities use backslashes | The launcher checks each identity, normalizes separators in the staged copy only, and records the original file SHA in runtime metadata |
| Model loading | `tools/helpers/pipeline.py` normally resolves some checkpoints relative to the process working directory | Checkpoints live under `SSD_MODELS_ROOT`; verified symlinks expose them at canonical repository-relative names, and runtime configs pass absolute paths to the loader |
| Dataset YAML split values | May be absolute Windows paths or contain backslashes | Runtime YAML sets its dataset root from `SSD_DATASET_ROOT` and converts absolute split paths to dataset-relative `images/...` paths |
| Pilot/final manifests | Could break if absolute paths were embedded | Current manifests are relative and are checked against the source audit IDs, order, GT, and label hashes |
| Outputs | A desktop output/profile tree must not be reused | Stage 3 uses a fresh repository-local `SSD_STAGE3_DIR`; Stage 4 uses a separate fresh `SSD_OUTPUT_ROOT` outside the repository |

Keep the copied dataset's directory names and tree intact. The pilot and final
TSV manifests already use dataset-relative paths such as
`images/val/...JPEG` and `labels/val/...txt`; they contain no absolute image or
annotation paths. `SSD_DATASET_ROOT` is their common root. The frozen model
configs retain Windows paths for provenance; the launcher rewrites temporary
runtime configs. Only the staged Pi copy of `split_metadata.json` has its
checkpoint path separators normalized; the source frozen artifacts remain
unchanged and the source split-metadata SHA is recorded.

## 2. Exact selected checkpoints

| Model | Desktop source | Pi path | Expected filename | SHA-256 |
|---|---|---|---|---|
| YOLO26n | `D:\Repositories\Other\SSD-PyTorch\trained_models\best-imagenet-vid-yolo\best.pt` | `$MODELS_ROOT/yolo26n/best.pt` | `best.pt` | `5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027` |
| ROI-SSD-MobileNetV3 | `D:\Repositories\Other\SSD-PyTorch\trained_models\imagenet-vid-roissd-mobilenet-20261027\stage3\best_map.pt` | `$MODELS_ROOT/roi_ssd_mobilenet/best_map.pt` | `best_map.pt` | `1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804` |

These are the only accepted checkpoints. The runner and preflight fail on a
missing file or hash mismatch; they do not choose another checkpoint.

## 3. Freeze and clone the exact source

On the Windows desktop, commit and push the source changes required for this
run, then record the commit:

```powershell
git status --short
git add experiments/cpu_pilot/portable_paths.py experiments/cpu_pilot/prepare.py experiments/cpu_pilot/run_pilot.py experiments/cpu_pilot/pi_runtime.py experiments/cpu_pilot/export_pi_inputs.py experiments/cpu_pilot/tests/test_portable_paths.py experiments/pi_preflight.py experiments/final_end_to_end_benchmark.py ai_docs/04_final_end_to_end_benchmark.md final_experiments/RASPBERRY_PI_STAGE3_STAGE4_RUNBOOK.md
git commit -m "Prepare portable Raspberry Pi benchmark workflow"
git push
$COMMIT_SHA = (git rev-parse HEAD).Trim()
$COMMIT_SHA
```

On the Pi, set `COMMIT_SHA` to that printed value:

```bash
COMMIT_SHA=<printed-commit-sha>
mkdir -p "$HOME/object-detection"
git clone https://github.com/YaroslavGozak/SSD-PyTorch.git "$HOME/object-detection/repo"
git -C "$HOME/object-detection/repo" fetch origin
git -C "$HOME/object-detection/repo" checkout "$COMMIT_SHA"
test "$(git -C "$HOME/object-detection/repo" rev-parse HEAD)" = "$COMMIT_SHA"
git -C "$HOME/object-detection/repo" status --short
```

The checkout must be clean. The Stage 3 implementation files are hash-bound
to the accepted Stage 3 calibration, so do not edit them or the selected
calibration JSON/receipts. The portable Stage 3 launcher keeps path handling
outside the hash-bound runner and logs its own SHA-256 in `metadata.json`.

## 4. Create and validate the Python environment

First record the actual Pi platform and Python:

```bash
uname -m
getconf LONG_BIT
python3 --version
```

Use 64-bit Raspberry Pi OS (`aarch64`, 64-bit). The repository has conflicting
dependency sets: `requirements.txt` pins torch/torchvision `2.3.1/0.18.1`,
while `requirements313.txt` pins `2.12.0/0.27.0`; their scikit-image and LAP
pins also differ. Repository reports include separate Python 3.12.10 CPU and
Python 3.13.5 CUDA environments, but there is no Pi-tested Python/ARM wheel
lock. Do not install either file wholesale before verifying the interpreter
and aarch64 wheels.

Create a venv and ask pip to resolve the repository's recent PyTorch pair as
binary CPU wheels for the Pi's actual Python/architecture. This is a
compatibility check; if it fails, stop and choose a PyTorch/torchvision pair
documented as compatible by PyTorch before continuing:

```bash
cd "$HOME/object-detection/repo"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip download --only-binary=:all: \
  --dest /tmp/ssd-pytorch-wheel-check \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.12.0 torchvision==0.27.0
```

Only if that check succeeds, install the matched CPU pair and the remaining
`requirements313.txt` packages without allowing pip to replace the pair:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.12.0 torchvision==0.27.0
grep -Ev '^(torch|torchvision)=' requirements313.txt > /tmp/ssd-requirements-pi.txt
python -m pip install -r /tmp/ssd-requirements-pi.txt
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); assert not torch.cuda.is_available()"
```

If the exact pair has no wheel for the Pi's installed Python, do not silently
fall back to the older `requirements.txt` pair. Resolve and record a compatible
CPU pair first, then pin it in the local environment instructions. The
[official PyTorch installer](https://pytorch.org/get-started/locally/) and
[Raspberry Pi OS Python documentation](https://www.raspberrypi.com/documentation/computers/os.html#use-python-on-a-raspberry-pi)
are the authoritative starting points.

## 5. Copy models and dataset

Recommended transfer is `rsync` over SSH, run from WSL on the Windows desktop.
It preserves the directory tree and can resume partial transfers. Replace
`PI_USER` and `PI_HOST` with the Pi login and host name:

Create the destination directories on the Pi first:

```bash
mkdir -p "$HOME/object-detection/data/imagenet_vid_yolo_sampled"
mkdir -p "$HOME/object-detection/models/yolo26n"
mkdir -p "$HOME/object-detection/models/roi_ssd_mobilenet"
```

Then transfer the dataset and two model files:

```bash
rsync -avh --partial --append-verify --info=progress2 \
  /mnt/d/ImageNet-VID/ImageNet/data/ImageNet2015/imagenet_vid_yolo_sampled/ \
  PI_USER@PI_HOST:/home/PI_USER/object-detection/data/imagenet_vid_yolo_sampled/

rsync -avh --partial --append-verify --info=progress2 \
  /mnt/d/Repositories/Other/SSD-PyTorch/trained_models/best-imagenet-vid-yolo/best.pt \
  PI_USER@PI_HOST:/home/PI_USER/object-detection/models/yolo26n/best.pt

rsync -avh --partial --append-verify --info=progress2 \
  /mnt/d/Repositories/Other/SSD-PyTorch/trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt \
  PI_USER@PI_HOST:/home/PI_USER/object-detection/models/roi_ssd_mobilenet/best_map.pt
```

Two alternatives are
`scp -r` over SSH (simple, but resume is manual) or copying to an external SSD
and using `rsync` from the mounted drive. After transfer, an `rsync -avhnc`
dry-run from WSL can compare file contents without changing either side.

On the Pi, expose the external model files at the canonical repository paths
used by the frozen model configs. These symlinks must resolve to the external
files above; do not overwrite an existing regular file:

```bash
BASE="$HOME/object-detection"
REPO="$BASE/repo"
MODELS_ROOT="$BASE/models"
mkdir -p "$REPO/trained_models/best-imagenet-vid-yolo"
mkdir -p "$REPO/trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3"
ln -s "$MODELS_ROOT/yolo26n/best.pt" \
  "$REPO/trained_models/best-imagenet-vid-yolo/best.pt"
ln -s "$MODELS_ROOT/roi_ssd_mobilenet/best_map.pt" \
  "$REPO/trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt"
```

If `ln` reports that a destination already exists, inspect it; do not replace
it automatically.

## 6. Configure local paths and stage the frozen inputs

Run these commands in the Pi shell; they are local environment settings, not
shared Git configuration:

```bash
export BASE="$HOME/object-detection"
export REPO="$BASE/repo"
export SSD_DATASET_ROOT="$BASE/data/imagenet_vid_yolo_sampled"
export SSD_DATASET_YAML="$SSD_DATASET_ROOT/imagenet_vid.yaml"
export SSD_MODELS_ROOT="$BASE/models"
export SSD_STAGE3_DIR="$REPO/final_experiments/pilot_pi5"
export SSD_OUTPUT_ROOT="$BASE/outputs/stage4_pi5"
export SSD_INFERENCE_DEVICE=cpu
```

Stage 3 writes outputs beside its frozen inputs, so use a new ignored,
repository-local directory. Stage 4 outputs stay outside the repository. On
the Windows desktop, export a curated frozen-input bundle; it includes only
the files Stage 3/4 need and validates their hashes. It deliberately omits
desktop cost profiles, timing CSVs, smoke outputs, and Stage 4 profiles:

```powershell
$bundle = "$env:TEMP\ssd_pi_inputs_$COMMIT_SHA"
python -m experiments.cpu_pilot.export_pi_inputs --output-dir $bundle
Compress-Archive -Path "$bundle\*" -DestinationPath "$bundle.zip"
scp "$bundle.zip" PI_USER@PI_HOST:/home/PI_USER/object-detection/stage3_inputs.zip
```

On the Pi, unpack into the fresh Stage 3 output location. This archive is
transferred separately because the frozen selection, split metadata, and
calibration evidence are intentionally not all tracked in Git:

```bash
printf '\n/final_experiments/pilot_pi5/\n' >> "$REPO/.git/info/exclude"
mkdir -p "$SSD_STAGE3_DIR"
unzip -q "$BASE/stage3_inputs.zip" -d "$SSD_STAGE3_DIR"
mkdir -p "$SSD_OUTPUT_ROOT"
```

The bundle includes each model's `calibration_report.md`; the frozen cost and
pipeline stages require it alongside the selected-config receipt, sweep,
gate matrix, and baseline. If an earlier bundle omitted these reports, export
a new bundle from the same desktop frozen inputs and copy only the reports
into the existing Pi Stage 3 directory:

```powershell
$bundle = "$env:TEMP\ssd_pi_inputs_$COMMIT_SHA-repaired"
python -m experiments.cpu_pilot.export_pi_inputs --output-dir $bundle
Compress-Archive -Path "$bundle\*" -DestinationPath "$bundle.zip"
scp "$bundle.zip" PI_USER@PI_HOST:/home/PI_USER/object-detection/stage3_inputs_repaired.zip
```

```bash
unzip -o "$BASE/stage3_inputs_repaired.zip" \
  'calibration/yolo26n/calibration_report.md' \
  'calibration/roi_ssd_mobilenet/calibration_report.md' \
  -d "$SSD_STAGE3_DIR"
test -s "$SSD_STAGE3_DIR/calibration/yolo26n/calibration_report.md"
test -s "$SSD_STAGE3_DIR/calibration/roi_ssd_mobilenet/calibration_report.md"
```

## 7. Checkpoint checksums and preflight

Verify the two exact files on the Pi:

```bash
sha256sum "$SSD_MODELS_ROOT/yolo26n/best.pt"
sha256sum "$SSD_MODELS_ROOT/roi_ssd_mobilenet/best_map.pt"
```

Expected SHA-256 values are in the checkpoint table above. Then, from the
repository root:

```bash
cd "$REPO"
git status --short
python -m experiments.pi_preflight
```

Preflight checks Linux/aarch64/CPU, importable project dependencies, dataset
and YAML paths, the exact checkpoint hashes and symlink targets, all image and
annotation paths in both manifests, label checksums and frame/video IDs
against the source audit manifest, frozen selection hashes/thresholds/padding,
absence of copied desktop cost profiles, a fresh writable output root, and a
clean Git commit. It prints `PI_PREFLIGHT_PASS` only if every check succeeds.
Resolve every reported failure before starting inference.

The selected frozen settings are:

| Model | evaluation_conf | operational_conf | padding |
|---|---:|---:|---:|
| YOLO26n | 0.01 | 0.10 | 64 |
| ROI-SSD-MobileNetV3 | 0.01 | 0.50 | 128 |

## 8. Stage 3 smoke, fresh Pi cost profile, and short pipeline check

The smoke uses 20 pilot frames per model and the corresponding frozen
operational confidence. Save each per-model smoke report because the runner
writes one report per invocation:

```bash
python -m experiments.cpu_pilot.pi_runtime --stage smoke --max-frames 20 \
  --models yolo26n --operational-conf 0.10 --output-dir "$SSD_STAGE3_DIR"
cp "$SSD_STAGE3_DIR/smoke_metadata.json" "$SSD_STAGE3_DIR/smoke_metadata_yolo26n.json"
cp "$SSD_STAGE3_DIR/smoke_summary.csv" "$SSD_STAGE3_DIR/smoke_summary_yolo26n.csv"

python -m experiments.cpu_pilot.pi_runtime --stage smoke --max-frames 20 \
  --models roi_ssd_mobilenet --operational-conf 0.50 --output-dir "$SSD_STAGE3_DIR"
cp "$SSD_STAGE3_DIR/smoke_metadata.json" "$SSD_STAGE3_DIR/smoke_metadata_roi_ssd_mobilenet.json"
cp "$SSD_STAGE3_DIR/smoke_summary.csv" "$SSD_STAGE3_DIR/smoke_summary_roi_ssd_mobilenet.csv"
```

After both smoke runs pass, collect a **new Pi-only Stage 3 cost profile**.
This profiles model inference at evaluation confidence and does not repeat the
confidence-by-padding sweep:

```bash
python -m experiments.cpu_pilot.pi_runtime --stage cost \
  --cost-repetitions 8 --output-dir "$SSD_STAGE3_DIR"
```

Review `cost_profiles_cpu.json` for Pi hardware/software metadata and the
frozen thresholds/padding. Then exercise the short pipeline on 20 pilot frames:

```bash
python -m experiments.cpu_pilot.pi_runtime --stage pipeline --max-frames 20 \
  --no-debug --output-dir "$SSD_STAGE3_DIR"
```

Review `pipeline_smoke_summary.csv`, `metadata.json`, and any error/crash
evidence. Do not continue if a model failed or a profile reports a different
runtime. The `SSD_STAGE3_DIR` output is Pi-only; never copy desktop K/c/tau into
it.

## 9. Stage 4 final benchmark

Only after preflight, both smoke runs, fresh Pi cost calibration, and the short
pipeline validation pass, run the final benchmark. It calibrates a separate
Stage 4 cost profile under the empty Pi-only output root before processing the
4,744-frame final manifest:

```bash
python -m experiments.final_end_to_end_benchmark \
  --stage all \
  --output-root "$SSD_OUTPUT_ROOT" \
  --dataset-yaml "$SSD_DATASET_YAML" \
  --device-label raspberry-pi-5 \
  --timing-repeats 2 \
  --calibration-repetitions 8 \
  --system-load-note "Dedicated Raspberry Pi 5 CPU run"
```

Review the device directory under `outputs/stage4_pi5/`, especially
`manifest_provenance.json`, `cost_profiles_final.json`,
`hardware_software_environment.md`, `final_summary.csv`, crash evidence, and
the per-run timing/frame files. Stage 4 validates cost-profile runtime
signatures and refuses to reuse profiles from another host.

## 10. Frozen inputs and known path facts

- Pilot manifest: 320 frames / 32 videos; SHA-256
  `817f5b145b774dc91ee0fa8d271e02c11d71d70bb50ff13082a0ba9487aa63cb`.
- Final candidate: 4,744 frames / 476 videos; SHA-256
  `4f36f3455ced1a344ef69329479cd09013934d71aafa8dc800d7f7e5ea6bf6d4`.
- The checked-in source audit manifest currently hashes to
  `6daa585af9a75d9fc390be20157e850d62c24e03f970bfc32b16e70e3cdd8041`,
  matching Stage 3 split metadata. `FINAL_EXPERIMENTS_MANIFEST.md` still
  documents the older `a375...` hash; do not edit the frozen source manifest
  during transfer. Preflight validates the actual audit file, full frame IDs,
  order, annotations, and labels.
- The manifests use relative `images/val/...` and `labels/val/...` paths. Their
  order, annotations, IDs, and label hashes are checked against
  `final_experiments/01_audit/val_manifest.jsonl`.
- Frozen Stage 3 configs/split metadata contain desktop-specific Windows YAML
  paths and backslashes in provenance fields. The manifests stay unchanged. The
  launcher normalizes checkpoint separators in the *staged Pi copy* of
  `split_metadata.json`, recording the source file hash in `metadata.json`;
  it uses local environment paths and temporary runtime configs for inference.
- The Stage 3 calibrated implementation hashes are validated by preflight.
  Keep the recorded Stage 3 source unchanged; path portability is provided by
  the separate launcher and Stage 4 runtime config.
- The repository requirements files conflict. The Pi's exact binary wheel
  compatibility must be established before experiments; do not guess a
  package version or install an x86_64 wheel.

No full Raspberry Pi Stage 3 or Stage 4 experiment is executed as part of this
runbook preparation.
