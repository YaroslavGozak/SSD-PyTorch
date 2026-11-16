# Transfer and cross-platform procedure

Use user-chosen local roots and host/user placeholders. Checkpoints and dataset
already present on a machine need not be copied again. Verify their hashes using
`inspect`; file names alone are not sufficient.

1. Review and commit the experiment source, portable YAML, tests, protocol and
   compact manifests on the laptop. Do not add `outputs/`, dataset frames,
   checkpoints or raw latency traces. Record the commit. On i9/Pi fetch and check
   out the exact same commit. Keep the working tree clean for final runs.
2. Prepare the real ImageNet workload once, run preflight, and create its bundle:

```bash
python -m experiments.roi_merge_b bundle \
  --manifest artifacts/roi_merge_b/frozen/manifest.jsonl \
  --output outputs/merge_b/inputs.zip
scp outputs/merge_b/inputs.zip USER@HOST:/LOCAL/TRANSFER/inputs.zip
```

3. On the target unpack into a fresh directory (do not overwrite a manifest):

```bash
mkdir -p artifacts/roi_merge_b/frozen
unzip /LOCAL/TRANSFER/inputs.zip -d artifacts/roi_merge_b/frozen
```

Verify `checksums.json` before proceeding:

```bash
python - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('artifacts/roi_merge_b/frozen')
for name,expected in json.loads((root/'checksums.json').read_text()).items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual==expected,(name,expected,actual)
print('Transfer checksums match')
PY
```

4. Set local `SSD_DATASET_ROOT`, `SSD_DATASET_YAML` and thread policy as in README.
   Default model paths are relative to the repository. If assets are external,
   expose the exact canonical model locations with verified symlinks, or supply
   `--checkpoint-root ROOT_WITH_TRAINED_MODELS_TREE`. Never substitute weights.
5. Use `--config artifacts/roi_merge_b/frozen/config.example.yaml` with every
   phase if the exported config differs from the repository example. Run
   `inspect --manifest ...` on the target and compare manifest/candidate/content
   hashes with the laptop. Run the same smoke cohort, then both model pilots.
   Freeze one appropriate thread policy per host before full calibration/test.
6. Transfer result/calibration directories back for aggregation. For example:

```bash
scp -r USER@HOST:/LOCAL/REPO/outputs/merge_b/HOST_RUN outputs/merge_b_collection/
python -m experiments.roi_merge_b aggregate \
  --results-root outputs/merge_b_collection \
  --output outputs/merge_b_report
```

Keep smoke, pilot and full results in separate collections. Aggregation rejects
mixed cohort/manifest hashes. Cost artifacts are specific to a host and must
never be transferred to another host for inference decisions. Transferring them
back for analysis/figures is valid.

If Pi cannot finish the full cohort, aggregate only the same explicitly
completed subset across all hosts; keep larger laptop/i9 runs separate.
A resumed run must use identical source/config/software/runtime. All code changes
invalidate the source-bound calibration; commit before collecting final data.
