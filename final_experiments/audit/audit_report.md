# Stage 1 model and dataset audit

## Result
Three checkpoints selected using fresh canonical evaluation; see [manifest](../FINAL_EXPERIMENTS_MANIFEST.md). All nine candidate checkpoints loaded and passed real-image CPU and decoded-output checks. Models, training configs and external datasets were not changed.

## Reproduction
Run from repository root:

```powershell
.venv/Scripts/python.exe -u final_experiments/audit/run_audit.py
.venv/Scripts/python.exe final_experiments/audit/build_report.py
```

The runner scans checkpoints across the repository (excluding epoch counters and environments), audits all labels in both common splits, writes immutable-order manifests, and compares the fixed shortlists. Stored predictions support metric rechecking without another inference pass.

## Protocol
256 frames selected from label occurrences and uniform fill, before inspecting prediction metrics. 228 loader videos, all 27 available GT classes. CPU, four torch threads, batch=1. RGB, OpenCV bilinear squash to 320x320, divide by 255; ImageNet normalization only for SSD. Confidence 0.01, class-aware output NMS 0.45, maximum 200 predictions, SSD pre-NMS top-k 400. Existing tools.infer.compute_map area-integral AP at IoU 0.50:0.05:0.95 is shared by every model. Scores are not the old Ultralytics validation scores.

## Ranking
| Family | Checkpoint | mAP50 | mAP50:95 | Recall50 | Selected |
|---|---|---:|---:|---:|---|
| YOLO26n | [trained_models/best-imagenet-vid-yolo/best.pt](../../trained_models/best-imagenet-vid-yolo/best.pt) | 0.804438 | 0.557575 | 0.850659 | yes |
| YOLO26n | [trained_models/imagenet-vid-yolo/best.pt](../../trained_models/imagenet-vid-yolo/best.pt) | 0.269252 | 0.193494 | 0.314313 | no |
| YOLO26n | [trained_models/imagenet-vid-yolo-val/best.pt](../../trained_models/imagenet-vid-yolo-val/best.pt) | 0.759027 | 0.526438 | 0.797480 | no |
| ROI-SSD | [trained_models/best-imagenet-vid-roissd/imagenet26022026stage3.pt](../../trained_models/best-imagenet-vid-roissd/imagenet26022026stage3.pt) | 0.818881 | 0.444104 | 0.872259 | no |
| ROI-SSD | [trained_models/imagenet-vid-roissd-best-v3/imagenet-vid200712stage3.pth](../../trained_models/imagenet-vid-roissd-best-v3/imagenet-vid200712stage3.pth) | 0.806710 | 0.427610 | 0.868163 | no |
| ROI-SSD | [trained_models/imagenet-vid-roissd-20261002/best_map_stage2.pt](../../trained_models/imagenet-vid-roissd-20261002/best_map_stage2.pt) | 0.850117 | 0.569247 | 0.893262 | yes |
| ROI-SSD-MobileNet | [trained_models/best-imagenet-vid-roissd-mobilenet/best_map.pt](../../trained_models/best-imagenet-vid-roissd-mobilenet/best_map.pt) | 0.832970 | 0.513597 | 0.917004 | no |
| ROI-SSD-MobileNet | [trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms/best_map.pt](../../trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms/best_map.pt) | 0.836082 | 0.526433 | 0.892203 | yes |
| ROI-SSD-MobileNet | [trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase3-preserved-forms/best_map.pt](../../trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase3-preserved-forms/best_map.pt) | 0.830415 | 0.523041 | 0.884123 | no |

## Shortlist rationale
YOLO: three 30-class YOLO26n final/best checkpoints with distinct training-dataset provenance; best50 is a predecessor when best.pt exists, 13-class and VOC checkpoints are incompatible. SSD: completed stage-3 runs with strongest available saved metrics and stage-2 best with auditable epoch metadata; penalized wrappers do not establish a separate better trained state, and October stage3 has only 3/20 epochs and is excluded. MobileNet: preserved stage1 best and preserved-forms stage2/stage3 best; older runs have lower logged mAP, unfreezed phase2 has worse metrics, and phase3-bn-freeze peaks at 0.6692. Old logs do not bind every peak to every saved checkpoint; fresh common evaluation is the selection evidence.

## Provenance and dataset checks
- Checkpoints inventoried: 82. [CSV](model_candidates.csv), [full JSON](model_candidates.json).
- Epoch.pth files are resume counters, not models. Embedded YOLO names and architecture are inspected; SSD class counts are checked against actual classification convolution output dimensions. State-dictionary readability is distinguished from strict instantiated-model loading.
- Historical benchmark weight hashes and paths: outputs/ca_yolo26n_cpu/metadata.json, outputs/ca_roissd_cpu/metadata.json, outputs/ca_roissd_mobilenet_cpu/metadata.json. Names can be stale after folder renames; hashes remain authoritative.
- Common validation: 5,064 frames, 508 videos, no missing/empty/invalid labels. Common train: 64,265 frames. Source IDs include split/shard prefixes; do not infer leakage from reused numeric IDs.
- Dataset subset variants and train/validation overlaps: [alternative_datasets.json](alternative_datasets.json).
- Background-inclusive 31 outputs and explicit YAML ordering are required; old 34-output or VOC checkpoints cannot cover the same foreground schema.

## Limitations
- mAP values below are newly measured on the locked 256-frame control subset, not on the entire 5,064-frame split. This is a short selection audit; no claim of statistical significance or global-best exhaustive search.
- There is no distinct test split in the audited YAML: loader test aliases validation. The final 5,064-frame manifest includes the 256 selection frames, so subsequent results are validation benchmarks, not an untouched held-out test.
- The class schema has 30 foreground entries, but domestic_cat (9), giant_panda (12), red_panda (20) have zero supplied GT in both train and validation. AP is averaged over the 27 present classes; the three absent-class APs are undefined, not zero.
- The common 5,064-frame validation set can support a locked sampled-frame end-to-end benchmark. It contains 508 short source clips (consecutive index gaps of 1 within each clip); original FPS, conversion version, negative-frame completeness and representativeness against full VID are unknown. Do not claim complete raw-video or full-VID generalization.
- Current dataset selection uses paths, class order, hashes, loader semantics and historical artifacts, not folder names alone. May comparison2026_5_29.csv records 5,064 frames and the same dataset; September audit matches YAML hash, count, video count and the exact 256-image prefix. Complete historical label-content hashes are unavailable.
- Other YOLO data copies contain train/validation overlap; their embedded historical metrics are not directly comparable. See alternative_datasets.json. Historical training/tuning overlap cannot be fully reconstructed from stripped weights.
- SSD checkpoints do not embed class-name lists or training git hashes. Foreground order is recovered from yolo-imagenet-vid training/evaluation provenance and the loader YAML; 31-way heads corroborate dimensions, and fresh class-wise evaluation checks decoding. Legacy 34-way VID, 21-way VOC, 13-class YOLO and 80-class COCO models are excluded.
- The old label_spaces.IMAGENET_VID_CLASSES constant is a different 33-foreground schema. Final configs must use yolo-imagenet-vid and the audited YAML; never replace it with that constant or silently claim the schemas are identical.
- Current training configs point to renamed/missing runs. Final configs snapshot the actually tested inference settings and explicit checkpoints. They are reconstructed inference configs, not assertions of original training settings.
- Historical ca_yolo26n_cpu weights hash exactly matches the selected YOLO checkpoint. Historical ca_roissd_mobilenet_cpu phase3.pt hash exactly matches the selected MobileNet phase2 best; the phase3 filename does not establish stage-3 training. See benchmark_provenance.json.
- YOLO26 is end-to-end: iou passed to predict alone does not ensure class-aware NMS. This audit applies torchvision batched_nms at 0.45 to all families after decoding; SSD also has internal NMS at the same threshold. YOLO internal top-k and SSD pre_nms_topK remain architecture-specific.
- Native Raspberry Pi inference was not tested. Windows CUDA wheels are not a Pi package lock. The tested package versions are recorded; Pi needs CPU/aarch64 builds and another load/forward smoke before long experiments.

## Smoke evidence
[canonical_results.json](canonical_results.json) contains per-class AP50, full-frame load/forward status, checkpoint SHA256 and all four SSD variable-size checks for every candidate. Variable-size smoke checks feasibility only; no conclusion about the 128x128 anomaly is made.

## Outputs
- [class compatibility CSV](class_compatibility.csv)
- [canonical result CSV](canonical_results.csv)
- [dataset audit](dataset_audit.json)
- [software versions](software.json)
- [machine-readable selections](selected_models.json)
- [execution log](execution.log)
