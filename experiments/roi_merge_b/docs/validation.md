# Implementation validation (2026-11-16)

## Executed

* `.venv/Scripts/python.exe -m unittest discover -s experiments/roi_merge_b/tests -q`:
  **18 tests passed**. Tests cover clipping/actual crop shapes, strict affine tau
  equivalence, deterministic grouping/ties, geometric equivalence, ID conservation,
  oracle/calibration separation, lookup fallback/uncertainty, sequence split and
  manifest/checkpoint failures, cost receipts, resume identities and metrics.
* MobileNet construction with `pretrained_backbone=False` made no pretrained
  download. Randomly initialized architecture forwards at 32x32, 32x320, 96x128,
  320x320 confirmed valid `until_1x1` execution traces. This is not checkpoint
  verification or a detector accuracy measurement.
* Explicit synthetic end-to-end runs completed for both detector shape rules:
  16 generated-image frames, 8 sequences; calibration/validation/test frame counts
  4/4/8. Seven policies, 9 original ROI pairs, 224 frame-policy-repetition records
  per detector, no duplicate trial IDs.
* Fixture manifest SHA-256:
  `866850696ba3daa1df43ce978b9d2be181b00751799194b2e1844d58b313bdd9`.
  This is a **synthetic fixture digest**, not the real ImageNet manifest digest.
* Completed calibration resume reused the frozen receipt-checked artifact;
  evaluation resume reused persisted trials without duplicate records.
* Bundle archive content matched every checksum. Aggregation generated two
  CSV tables, machine-readable aggregates, `paper_results.md`, and 10 standalone
  figures. Figures explicitly mark synthetic fixture data and were visually
  checked for layout. `compileall`, CLI help and `git diff --check` passed.

Local ignored output paths:

* `outputs/roi_merge_b_verified/final_results/`
* `outputs/roi_merge_b_verified/final_report/paper_results.md`
* `outputs/roi_merge_b_verified/final_transfer.zip`
* `outputs/roi_merge_b_preflight/preflight.json`

## Not executed / publication blockers

No real Stage 1 weights or dataset images were available. Their expected
identities are configured, but successful real checkpoint verification,
real-model calibration, real ImageNet geometry preparation, real detector replay,
i9/Pi execution and scientific comparisons were **not executed**.
The available audit index has 5,064 records / 508 video IDs; actual image files
were not verified and this must not be called the frozen 4,744-frame final cohort.

The implementation venv contains Python 3.10.11, torch 2.3.1+cu121,
torchvision 0.18.1+cu121, NumPy 1.26.4, Pillow 10.4.0, Ultralytics 8.3.77,
OpenCV 4.10.0.84 and no installed threadpoolctl. Real YOLO requires the configured
8.4.60 and the runtime controls require threadpoolctl. Dependencies were not
silently upgraded. Preflight returned BLOCKED with exact missing asset paths.
See `../BLOCKERS.md` for restoration and execution commands.

Synthetic fixture costs and their policy rankings are software-test outputs,
not publication findings. The committed fixture metadata is portable; its
synthetic image assets are regenerated with the documented fixture command.
