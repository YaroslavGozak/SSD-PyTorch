# Recursive checkpoint evaluation

From the repository root, using the environment with PyTorch and Ultralytics:

```powershell
.venv/Scripts/python.exe -m tools.batch_infer --dry-run
.venv/Scripts/python.exe -m tools.batch_infer --resume
```

The runner discovers `.pt` and `.pth` files at every depth under `trained_models/`,
excluding `epoch.pth` resume counters. It runs `tools.infer` sequentially with
`pad-loop` and `fixed-size-loop`, without sample-image inference. Each checkpoint
schedules 21 padding evaluations and 100 fixed-size evaluations.

Outputs are isolated by relative checkpoint path, including its extension:

```text
outputs/batch_infer/<run>/<nested-run>/<checkpoint.pt>/
  config.yaml
  pad-loop.log
  pad-loop/results.csv
  fixed-size-loop.log
  fixed-size-loop/fixed_size_results.csv
```

Per-transform artifacts live alongside each mode's CSV. `batch_report.json`
records failures and resolved config sources. Failed modes do not prevent other
modes/checkpoints from running; the launcher exits nonzero if any fail.
`--resume` skips successfully completed modes when checkpoint size, modification
time and generated config still match. Within an interrupted mode, completed
padding values or height/width pairs are skipped; the interrupted setting runs
again from the beginning. Summary rows are committed atomically after metrics
finish, preventing partial rows and duplicate results on resume.
Per-mode `.progress.json` files bind these rows to the checkpoint and generated
config. With `--resume`, changing worker count, pinning, fixed-size batch size,
or model reuse preserves completed models, methods, and individual sweep values.
Only unfinished values use the new execution settings. Existing markers from
earlier versions are supported even if their execution settings differ.
Changed checkpoints/configs or runs without `--resume` start fresh.
To continue an existing output folder with six workers:

```powershell
.venv/Scripts/python.exe -m tools.batch_infer --resume --num-workers 6 --output outputs/batch_infer
```

Results can therefore contain measurements from different execution settings.
Use a separate output folder when comparing timing across those settings.
Older CSVs without markers are adopted when the saved config matches, their rows
identify the same model/checkpoint/dataset, and the checkpoint is no newer than
the CSV. Existing identity markers take precedence; mismatched results are not
adopted. A complete CSV skips the entire sweep even without a completion marker.
Use repeatable `--resume-from` options to recover missing values from other batch
output folders. Current output rows take precedence, and source files stay intact:

```powershell
.venv/Scripts/python.exe -m tools.batch_infer --resume --resume-from outputs/batch_infer_v2 --num-workers 6 --pin-memory --fixed-size-batch-size 6
```

Add `--dry-run` to preview the number of recoverable values without importing rows
or starting inference.

Saved run snapshots take precedence over inferred repository defaults. SSD head
dimensions establish class counts; known run names distinguish SSD from ROI-SSD
when their state layouts are identical. YOLO uses embedded dataset metadata.
Inference defaults cannot reconstruct every historical training configuration.
Review the dry-run report and override ambiguous/custom runs with a YAML map:

```yaml
my-run: config/imagenet-vid-roissd.yaml
my-run/nested/special-name.pth: config/voc.yaml
```

```powershell
.venv/Scripts/python.exe -m tools.batch_infer --config-map batch_configs.yaml --resume
```

Map keys are relative to `--models-root`; the most specific path wins. Config
paths are relative to the working directory. The runner substitutes the actual
checkpoint path regardless of the original config's filename or task name.
Custom architectures still need support in the existing inference pipeline.

Use `--match '*best*'` to limit checkpoint paths, `--models-root
trained_models/my-run` to limit a subtree, and `--output outputs/my_batch` to
choose another output directory. Dataset roots come from the resolved config;
this does not force VOC and ImageNet models onto a common evaluation dataset.
