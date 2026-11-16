# Experiment B: fixed ROI merging on ImageNet-VID

This experiment compares `no_merge`, `geometric`, `affine`, `piecewise`, `lookup`,
`conservative_lookup` and retrospective `oracle` with identical, frozen,
GT-derived candidate rectangles. It is a new fixed-workload experiment, not a
rerun of the old closed-loop tracker benchmark. No model is retrained.

Read [the repository audit](docs/experiment_b_inventory.md),
[the protocol](docs/protocol.md), [transfer instructions](TRANSFER.md) and
[current execution blockers](BLOCKERS.md).

## Assets and execution controls

Run from the repository root with a compatible project virtual environment.
Install the repository dependencies plus `threadpoolctl`, `scipy` and
`matplotlib`. The selected YOLO path requires `ultralytics==8.4.60`.
Install the experiment extras into that environment:

```bash
python -m pip install -r experiments/roi_merge_b/requirements.txt
```

Only CPU float32, batch 1 is implemented; GPU results must not be inferred.
No model downloads or alternate checkpoint selection occur.

Default checkpoint locations are relative to the repository, exactly as in the
Pi workflow:

| Detector | Checkpoint | SHA-256 |
|---|---|---|
| YOLO26n | `trained_models/best-imagenet-vid-yolo/best.pt` | `5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027` |
| ROI-SSD-MobileNetV3 | `trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt` | `1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804` |

Existing verified symlinks are supported. `--checkpoint-root ROOT` is an explicit
alternative root containing that same `trained_models/...` tree. All identities
are checked before loading. No `SSD_MODELS_ROOT` inference or filename search.
The architecture settings are committed inside `config.example.yaml`; old
absolute Windows dataset paths are not used.

Bash (Pi example; freeze a thread policy in a separate pilot before measuring):

```bash
export SSD_DATASET_ROOT=/local/path/imagenet_vid_yolo_sampled
export SSD_DATASET_YAML="$SSD_DATASET_ROOT/imagenet_vid.yaml"
export SSD_CPU_THREADS=4
export SSD_INTEROP_THREADS=1
export SSD_OPENCV_THREADS=1
```

PowerShell (laptop/i9; the number is a local experimental choice):

```powershell
$env:SSD_DATASET_ROOT = 'E:\local\imagenet_vid_yolo_sampled'
$env:SSD_DATASET_YAML = "$env:SSD_DATASET_ROOT\imagenet_vid.yaml"
$env:SSD_CPU_THREADS = '4'
$env:SSD_INTEROP_THREADS = '1'
$env:SSD_OPENCV_THREADS = '1'
```

All phases accept `--dataset-root`; model phases accept `--cpu-threads`,
`--interop-threads`, `--opencv-threads`, `--affinity` with explicit CLI values
preceding environment values, then config defaults. Choose the same controls
for calibration and test on each host. Changed threads/affinity, weights,
software, host, power policy, source or config require a new calibration run.

## Prepare once on the laptop

```bash
python -m experiments.roi_merge_b inspect --config experiments/roi_merge_b/config.example.yaml --output outputs/merge_b/preflight
python -m experiments.roi_merge_b prepare --config experiments/roi_merge_b/config.example.yaml --output artifacts/roi_merge_b/frozen
```

`inspect` decodes and checks every indexed image and label, reports actual
counts/IDs/order, verifies both checkpoints, and runs each model smoke in a
separate process. `--skip-inference` is explicitly an assets-only check, not a
complete preflight. Failed preflight produces actionable `preflight.json`.

The default source is the canonical audit index, not an assumed count of 4,744.
An optional `--cohort-manifest path/to/final_eval_manifest_candidate.txt` on
`prepare`/`inspect` restricts it to that exact Stage 3 cohort before the new
sequence split. Do not describe all audit frames or the resulting held-out
subset as the Stage 3 final cohort. Actual split counts are in manifest metadata.

For a compact real workload, explicitly version the manifest and metadata
(the repository globally ignores JSON):

```bash
git add -f artifacts/roi_merge_b/frozen/manifest.jsonl artifacts/roi_merge_b/frozen/manifest.meta.json
```

Copy the prepared manifest byte-for-byte to every host. Never run `prepare`
again on Pi/i9. It hashes compressed image bytes and labels (one full pass),
stores dimensions and effective shapes, and rejects a nonempty frozen manifest
destination. The exact same source and controls generate identical manifest
bytes; no timestamp or local absolute path enters the manifest.

## Run independently on each host, both detectors

Use a fresh run ID when changing controls. Example for one model:

```bash
python -m experiments.roi_merge_b inspect --manifest artifacts/roi_merge_b/frozen/manifest.jsonl --output outputs/merge_b/host_run/preflight
python -m experiments.roi_merge_b calibrate --manifest artifacts/roi_merge_b/frozen/manifest.jsonl --model yolo26n --dry-run --output outputs/merge_b/host_run/yolo26n/calibration
python -m experiments.roi_merge_b calibrate --manifest artifacts/roi_merge_b/frozen/manifest.jsonl --model yolo26n --output outputs/merge_b/host_run/yolo26n/calibration
python -m experiments.roi_merge_b evaluate --manifest artifacts/roi_merge_b/frozen/manifest.jsonl --model yolo26n --models-dir outputs/merge_b/host_run/yolo26n/calibration --subset smoke --dry-run --output outputs/merge_b/host_run/yolo26n/smoke
python -m experiments.roi_merge_b evaluate --manifest artifacts/roi_merge_b/frozen/manifest.jsonl --model yolo26n --models-dir outputs/merge_b/host_run/yolo26n/calibration --subset smoke --output outputs/merge_b/host_run/yolo26n/smoke
```

```bash
python -m experiments.roi_merge_b inspect --manifest artifacts/roi_merge_b/frozen/manifest.jsonl --output outputs/merge_b/host_run/preflight
python -m experiments.roi_merge_b calibrate --manifest artifacts/roi_merge_b/frozen/manifest.jsonl --model roi_ssd_mobilenet --dry-run --output outputs/merge_b/host_run/roi_ssd_mobilenet/calibration
python -m experiments.roi_merge_b calibrate --manifest artifacts/roi_merge_b/frozen/manifest.jsonl --model roi_ssd_mobilenet --output outputs/merge_b/host_run/roi_ssd_mobilenet/calibration
python -m experiments.roi_merge_b evaluate --manifest artifacts/roi_merge_b/frozen/manifest.jsonl --model roi_ssd_mobilenet --models-dir outputs/merge_b/host_run/roi_ssd_mobilenet/calibration --subset smoke --dry-run --output outputs/merge_b/host_run/roi_ssd_mobilenet/smoke
python -m experiments.roi_merge_b evaluate --manifest artifacts/roi_merge_b/frozen/manifest.jsonl --model roi_ssd_mobilenet --models-dir outputs/merge_b/host_run/roi_ssd_mobilenet/calibration --subset smoke --output outputs/merge_b/host_run/roi_ssd_mobilenet/smoke
```

Repeat with `--model roi_ssd_mobilenet` and corresponding directories. Then
use `--subset pilot` or `--subset full` and fresh evaluation directories.
Smoke/pilot are immutable seeded selections of held-out frame IDs; their hashes
must match on every host. `--limit-frames` is an explicitly identified partial
run, not a full result. Never reduce the Pi cohort silently. Compare the same
completed subset separately from full laptop/i9 results.

Dry-run reports all possible union shapes and invocation counts before timing.
`profiling.max_heldout_shapes` prevents infeasible jobs. Do not bypass it by
changing a test policy; freeze a smaller common cohort/config and rerun all
hosts instead. Profiling cost can be large on Pi. Warmups are outside each
measurement and repeated in every randomized block.

## Resume and transfer

Reissue the same command and output directory to resume. Each shape trial and
frame/policy/repetition is atomically checkpointed under a stable content key.
An identity mismatch fails rather than mixing sessions. A completed calibration
is immutable and reused only after receipt and identity validation. Keep
unrelated workloads away during measurement. Thermals/frequency are logged,
not automatically normalized. Resume across thermal sessions is visible in raw
observations and must be considered in analysis.

```bash
python -m experiments.roi_merge_b bundle --manifest artifacts/roi_merge_b/frozen/manifest.jsonl --output outputs/merge_b/inputs.zip
python -m experiments.roi_merge_b aggregate --results-root outputs/merge_b --output outputs/merge_b_report
```

Aggregate only one manifest/test cohort at a time: point `--results-root` at a
collection of matching full runs, matching pilot runs, or matching smoke runs.
Mixing full and smoke evaluations fails loudly. Transfer calibration directories
alongside evaluations to generate all supporting calibration figures.
Outputs include `main_results.csv`, `heldout_fit.csv`, `paper_results.md` and
standalone PNGs, plus per-evaluation JSONL traces, sequence bootstrap intervals,
coverage/geometry/boundary strata and frame replay summaries.

## Tests and explicit synthetic smoke

```bash
python -m unittest discover -s experiments/roi_merge_b/tests -v
python -m experiments.roi_merge_b fixture --output outputs/merge_b_fixture
python -m experiments.roi_merge_b calibrate --config outputs/merge_b_fixture/config.yaml --manifest outputs/merge_b_fixture/workload/manifest.jsonl --model yolo26n --synthetic-smoke --output outputs/merge_b_fixture/results/yolo26n/calibration
python -m experiments.roi_merge_b evaluate --config outputs/merge_b_fixture/config.yaml --manifest outputs/merge_b_fixture/workload/manifest.jsonl --model yolo26n --synthetic-smoke --dataset-root outputs/merge_b_fixture/data --models-dir outputs/merge_b_fixture/results/yolo26n/calibration --output outputs/merge_b_fixture/results/yolo26n/evaluation
```

Repeat the last two commands for MobileNet; aggregate `outputs/merge_b_fixture/results`.
The fixture uses generated images and deterministic synthetic latency, **not
real weights or scientific measurements**. Both the manifest and runner must
explicitly opt in; synthetic execution is prohibited for real manifests.
A compact fixture manifest is committed under `fixtures/synthetic/`; regenerate
its image assets with `fixture`, never treat it as ImageNet-VID.
