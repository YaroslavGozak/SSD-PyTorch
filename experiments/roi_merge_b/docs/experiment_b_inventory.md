# Repository audit and implementation plan

Audit date: 2026-11-16. This audit precedes implementation. Experiment B is a
fixed, annotation-informed ROI replay, not the historical closed-loop video run.

| Interface | Discovered behavior | Reuse / adaptation |
|---|---|---|
| `cpu_pilot/portable_paths.py` | Two canonical repository-relative checkpoint locations and exact SHA-256 identities; optional external model root | Reuse identities and defaults; expose explicit root override |
| `final_experiments/configs/*.yaml` | Stage 1 detector architecture configs; contain obsolete absolute dataset paths | Copy architecture settings into portable experiment config; never load their dataset paths |
| `final_experiments/01_audit/val_manifest.jsonl` | Canonical ordered video/frame IDs, relative paths, normalized YOLO GT and label SHA | Reuse as dataset index; choose sequence-disjoint cohort explicitly |
| `cpu_pilot/prepare.py` | Stage 3 pilot/final sequence split; final TSV is not available in this checkout | Support optional final TSV cohort; do not claim all audit rows are the 4,744-frame final cohort |
| `pipeline.convert_crop_to_input_tensor` | Crop existing resized tensor; YOLO snaps window to 32 inside canvas, SSD grid 1 | Reuse directly for manifest shape derivation and actual replay |
| `dataset/transforms/ssd_transform.py` | Torchvision square resize and ImageNet normalization | Reuse for the base 320x320 tensor; undo ImageNet normalization only for YOLO |
| `model/model_adapters.py` | YOLO predict API and raw-forward paths differ | Primary timing explicitly uses existing raw-forward profiling path; not Stage 3/4 predict-API timings |
| `cost_model_validation/ultralytics_adapter.py` | Loads exact YOLO checkpoint; raw `measurek.run_forward`; RGB /255 | Reuse loading/infer on already prepared real tensor; no second resize |
| `cost_model_validation/roissd_adapter.py` | Loads MobileNet state dict; forward includes internal detection processing | Reuse loading/infer; never use legacy execution_metadata tuple unpacking |
| `model/roissd_mobilenet.py` | `until_1x1`, feature feasibility checks, dictionary execution trace | Verify trace and actual network input shape; fail infeasible shapes rather than guessing stride 32 |
| `cost_model_validation/models.py` | Affine and hinge piecewise area fits, positive-slope validation | Reuse fits; reject invalid affine, constrain hinge branches using calibration only |
| `cost_model_validation/shape_model.py` | Exact-shape lookup, SE-sum conservative lower bound, separate fallback | Reuse documented safeguard semantics; new portable artifact and nearest-neighbor baseline kept distinct |
| `tools/mergers/simple.py` | Sequential center-distance OR IoU test; default IoU=0 accepts disjoint rectangles | Not the primary geometric baseline; document inconsistency |
| `tools/mergers/simple2.py` | Sequential bounding area / sum area <= 1.4 baseline | Reuse rule/order, add source-ID groups and trace, regression-test against original |
| `tools/mergers/calibrated.py` | Iterative best positive gain, first tie, remove pair then append union | Reuse enumeration/tie semantics across all cost policies |
| `cpu_pilot/runtime_threads.py` | Explicit intra/inter/OpenCV/BLAS controls and Ultralytics override | Reuse control installer via lightweight runtime facade; record actual settings |
| `cost_model_validation/common.py` | CPU power/frequency/thermal metadata and SHA helpers | Reuse sensors; bind cost artifacts to host, source, config, weights and threads |

## Implementation plan

1. Portable config, source index/cohort reader, immutable GT-derived manifest,
   sequence splits and separately identified synthetic test fixture.
2. Exact canonical checkpoint verification and dataset content/decode preflight;
   workload estimates and portable transfer bundle.
3. Independent randomized calibration/validation timing blocks and test-only
   held-out shape trials; resume journal bound to immutable runtime identity.
4. Seven policies, pairwise metrics, fixed-workload frame replay, sequence
   bootstrap, boundary/geometry strata, held-out exact small-n and explicitly
   approximate large-n oracle; no policy access to test timings.
5. Reports, fit diagnostics, figures and transfer/run documentation. Real model
   execution is blocked until the selected weights and dataset are accessible.

## Initial asset blockers

Neither canonical checkpoint exists in this checkout. Neither dataset location
referenced by the older configs exists here. The frozen Stage 3 final TSV is
also absent. Do not silently substitute VOC weights, another dataset, or a
different frame cohort. See `../BLOCKERS.md` for executable recovery commands.

The implementation adds an optional `pretrained_backbone=False` path to the
MobileNet constructor/adapter to avoid downloading unrelated initialization
weights before strict full-checkpoint loading. Legacy defaults are preserved.
