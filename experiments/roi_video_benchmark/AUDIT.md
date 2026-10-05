# Pre-change audit (2026-09-28)

| Requirement | Existing implementation/file (pre-change lines) | Gap | Minimal change |
|---|---|---|---|
| Entry point/config | `tools/benchmarks/benchmark_framework_vid.py:258,432`; `config/benchmark-vid-roissd.yaml:6`; batch runner `tools/benchmarks/batch_benchmark_vid.py:123` | Tracker sweeps, not a locked four-policy protocol | Experiment launcher using `VideoSequenceBenchmark` and inherited PC/Pi configs |
| Actual data | `config/imagenet-vid-yolo-base.yaml:5`; `tools/helpers/pipeline.py:213`; `dataset/yolo_imagenet_vid.py:68,174` | Current data is **YOLO-converted sampled ImageNet-VID**, not presumed voc-vid | Persist ordered loader manifest with annotations and dimensions |
| Split/root | `D:/ImageNet-VID/ImageNet/data/ImageNet2015/imagenet_vid_yolo_sampled/imagenet_vid.yaml` exists; its `path` points to that directory | Loader's `test` maps to YAML `val` (`images/val`, `labels/val`); release/version/FPS unspecified | Record actual split, classes, counts and frame indices; do not assume temporal sampling rate |
| GT completeness | `dataset/yolo_imagenet_vid.py:190` parses normalized YOLO txt labels and drops frames without detections | Existing protocol excludes missing/empty annotations; not complete raw-video evaluation | Document retained-frame protocol and audit missing/empty labels separately |
| Baseline | `tools/mergers/greedy.py:49` | Not the plain linear tau rule: beta(area)-weighted delta **greater than tau**, area inflation <=100, plus tau>image/canvas shortcuts | Preserve baseline unchanged; PC base tau=17000, Pi base=225677; current sweep overrides tau=225000 |
| Calibrated merge | `tools/mergers/calibrated.py:11`; `experiments/cost_model_validation/shape_model.py:108` | Rectangles are rounded before actual coordinate roundtrip/crop; sparse lookup fallback only logs | Supply effective shapes from actual crop function; per-frame decisions/reasons/counters |
| Ordering | Both mergers enumerate combinations, choose maximum gain with first tie, remove pair and append union | Baseline has additional shortcuts | Retain pair-selection algorithm; clearly separate scoring and shortcut behavior |
| Tracking/ROI | `tools/helpers/pipeline.py:683`; `tools/benchmarks/benchmark_framework_vid.py:480` | PC base uses oracle; no fixed subset/warmup | Explicit non-oracle tracker, fixed ordered subset, reset per video/policy; no proposal replay exists |
| Shapes/remap/NMS | `tools/helpers/pipeline.py:334,370,472,508` | Crop alignment is 32 for every model; full tensor may be 300 | Record actual tensor dimensions; never silently round observed shape into calibration |
| Quality | `tools/benchmarks/benchmark_framework_vid.py:615` uses existing `compute_map` | mAP95 is AP at .95, not mAP50:95 | Mean AP over .50:.05:.95, reuse evaluator and serialize its inputs |
| Time | `tools/helpers/pipeline.py:723`; benchmark `:673` | Frame timer excludes loader/read; no CUDA synchronization/warmup | Synchronized model timer and separate loaded-frame pipeline versus decode/preprocess-inclusive totals |
| Costs/coverage | `benchmark_framework_vid.py:574` | Source-pixel area rather than effective tensor area, no call totals or per-video results | Persist frame records, inference calls, tensor pixels, merges, coverage and uncovered GT counts |
| Reproducibility | Experiment A/B `common.py:151,222`, `artifacts.py:24` | Video CSV lacks manifest, hashes and runtime validation | Reuse runtime controls/hash-verified artifacts; store config, command, provenance and detections |
| Available runtime | `.venv/Scripts/python.exe`: torch 2.12.0+cu126, CUDA available | Pi hardware not available here | CPU first with existing CPU calibration; Pi configs supplied, not claimed tested |
| Calibration | `outputs/cost_model_voc_roissdmobilenet_v7_constrained/calibration/experiment_a_summary.json` | VOC MobileNet weights, CPU/4 threads/affinity 0..3/stride32; cannot apply to ImageNet-trained weights | Use its exact VOC checkpoint and existing label-remap mechanism on current VID data; explicitly report unsupported classes and preprocessing/runtime differences |

Before changes: `python -m pytest ...` unavailable (pytest not installed).
` .venv/Scripts/python.exe -m unittest experiments.cost_model_validation.tests.test_core -q`: **11 passed**.
Pre-existing user changes in cost-model configs/README are left intact.

Calibration A uses generated image inputs (not video GT); no evidence establishes that historical parameter selection never used these videos. Record that limitation rather than claiming an independent held-out study. No tuning on benchmark videos is permitted.
