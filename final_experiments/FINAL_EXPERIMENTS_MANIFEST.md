# Final experiments manifest

<!-- checkpoint-role-recommendations:start -->
## Updated individual models and same-family pairs (2026-10-23)

**This section supersedes the original single-checkpoint recommendation for ROI processing.** The canonical full-frame results below remain historical evidence for that role. Penalized ROI-SSD candidates are excluded at the user's request.

Saved batch results currently cover **51 checkpoints; 48 have both complete sweeps**. Partial or conflicting sweeps, penalized wrappers, legacy class spaces, and other datasets are excluded from the ROI shortlist. Only non-penalized 31-output models on the sampled 30-foreground-class VID schema support the recommendations below.

### Individual best models by role

There is no established universal winner for both roles. Full-frame specialists come from the original 256-frame canonical evaluation at confidence 0.01. ROI specialists come from the saved 5,064-frame crop sweeps at configured confidence 0.5. These scores measure different protocols and must not be compared directly.

| Family | Role | Checkpoint | Evidence |
|---|---|---|---|
| ROI-SSD | Best full-frame candidate in canonical shortlist | [imagenet-vid-roissd-20261002/best_map_stage2.pt](../trained_models/imagenet-vid-roissd-20261002/best_map_stage2.pt) | mAP50 85.01%; mAP50:95 56.92%. [Canonical results](audit/canonical_results.json) |
| ROI-SSD | Recommended individual ROI checkpoint | [best-imagenet-vid-roissd/imagenet26022026stage3.pt](../trained_models/best-imagenet-vid-roissd/imagenet26022026stage3.pt) | Mean padding mAP50 64.17%; mean fixed-size mAP50 33.40%. [Padding CSV](../outputs/batch_infer/best-imagenet-vid-roissd/imagenet26022026stage3.pt/pad-loop/results.csv); [size CSV](../outputs/batch_infer/best-imagenet-vid-roissd/imagenet26022026stage3.pt/fixed-size-loop/fixed_size_results.csv) |
| ROI-SSD-MobileNet | Best full-frame candidate in canonical shortlist | [best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms/best_map.pt](../trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms/best_map.pt) | mAP50 83.61%; mAP50:95 52.64%. [Canonical results](audit/canonical_results.json) |
| ROI-SSD-MobileNet | Recommended individual ROI checkpoint | [best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt](../trained_models/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt) | Mean padding mAP50 51.53%; mean fixed-size mAP50 24.72%. [Padding CSV](../outputs/batch_infer/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt/pad-loop/results.csv); [size CSV](../outputs/batch_infer/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt/fixed-size-loop/fixed_size_results.csv) |

The ROI averages are descriptive: padding averages all 21 settings from 0 to 200 px; fixed-size averages all 100 height/width combinations from 32 to 320 px in 32-pixel steps. They are not a pooled mAP and do not represent deployment-frequency weights.

### What changed relative to the original selection

| Family | Previous full-frame selection: mean padding / size mAP50 | Recommended ROI specialist: mean padding / size mAP50 |
|---|---:|---:|
| ROI-SSD | 53.93% / 18.40% | 64.17% / 33.40% |
| ROI-SSD-MobileNet | 43.55% / 13.21% | 51.53% / 24.72% |

### MobileNet alternatives

- For smaller crops, retain [best-imagenet-vid-roissd-mobilenet/20260928-best-phase3/best_map.pt](../trained_models/best-imagenet-vid-roissd-mobilenet/20260928-best-phase3/best_map.pt): at 128x128 its mAP50 is 46.82%, versus 40.76% for the recommended phase-2 checkpoint. Across the 25 shapes with both sides <=160 px, its mean is 6.56%, versus 5.64%. [Size CSV](../outputs/batch_infer/best-imagenet-vid-roissd-mobilenet/20260928-best-phase3/best_map.pt/fixed-size-loop/fixed_size_results.csv).
- The phase2-bn-freeze `last.pt` has a slightly higher fixed-size mean (24.85%) than `best_map.pt` (24.72%), but lower padding mean (49.88% versus 51.53%). This does not establish a universal winner or statistical significance; the recommended checkpoint favors padding robustness, with the phase-3 alternative retained for small crops.

### Recommended pairs

| Family | Full-frame / discovery / fallback | ROI processing between full-frame calls |
|---|---|---|
| ROI-SSD | [imagenet-vid-roissd-20261002/best_map_stage2.pt](../trained_models/imagenet-vid-roissd-20261002/best_map_stage2.pt) | [best-imagenet-vid-roissd/imagenet26022026stage3.pt](../trained_models/best-imagenet-vid-roissd/imagenet26022026stage3.pt) |
| ROI-SSD-MobileNet | [best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms/best_map.pt](../trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms/best_map.pt) | [best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt](../trained_models/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt) |

Each pair uses two checkpoints from the same architecture family, the same foreground class order, and SSD background index 0. Both outputs must be converted into the same source-frame coordinates and update the same tracker. Full-frame calls initialize detection, discover objects, and handle fallback; ROI calls process intervening frames. This is role switching, not prediction averaging.

**Pair status: recommended for validation, not yet demonstrated superior end to end.** Compare full-frame/full-frame, ROI/ROI, full-frame/ROI, and the reversed pair on identical videos, tracker settings, ROI generation, full-frame schedule and fallback rules. Measure mAP, recall, end-to-end latency, full-frame invocation rate, and peak memory with both checkpoints resident. No pair accuracy or Pi memory/latency is inferred from separate crop tests.

Explicit role paths and evidence: [machine-readable role recommendations](audit/model_role_recommendations.json). Original full-frame configs remain unchanged. ROI configs linked there are the saved batch configs (confidence 0.5), not a claim that a mixed pipeline has been implemented.

### Remaining limitations

- Full-frame winners were selected among three canonical candidates per family, not every checkpoint on full frames. The later ROI sweeps do not expand that full-frame comparison.
- Near ties can reflect crop jitter because fixed-size crops use randomness without a frozen per-image geometry manifest. Use shared deterministic crops before resolving small differences.
- The updated YOLO adapter fixes preprocessing and prediction-setting propagation, but corrected sweeps must finish before a new YOLO ranking is published. YOLO is outside the two-family paired recommendation here.
- Penalized checkpoints are excluded; no rerun or ranking of those intermediate experiments is required.
- Existing dataset limitations still apply: 30 schema classes, only 27 with supplied GT, and validation reused for selection rather than an untouched test split.

<!-- checkpoint-role-recommendations:end -->
Audit date: 2026-10-14. Stage 1 only; models and datasets were not modified or retrained.

## Common dataset
- Dataset name: YOLO-converted sampled ImageNet-VID (`yolo-imagenet-vid`).
- Dataset root: `D:\ImageNet-VID\ImageNet\data\ImageNet2015\imagenet_vid_yolo_sampled`.
- Train manifest: [train_manifest.jsonl](audit/train_manifest.jsonl).
- Validation manifest: [val_manifest.jsonl](audit/val_manifest.jsonl); loader `test` and `val` both resolve YAML `val`.
- Final evaluation manifest: [val_manifest.jsonl](audit/val_manifest.jsonl), frozen ordered 5,064-frame split.
- Canonical selection manifest: [canonical_validation.jsonl](audit/canonical_validation.jsonl), 256 frames / 228 loader video IDs.
- Number of videos: validation 508; train 6472 source IDs (3710 numeric loader IDs).
- Number of frames: train 64265; validation/final 5064.
- Foreground class count: 30 schema entries; 27 GT-supported entries in supplied splits.
- Exact class list: 0: airplane; 1: antelope; 2: bear; 3: bicycle; 4: bird; 5: bus; 6: car; 7: cattle; 8: dog; 9: domestic_cat; 10: elephant; 11: fox; 12: giant_panda; 13: hamster; 14: horse; 15: lion; 16: lizard; 17: monkey; 18: motorcycle; 19: rabbit; 20: red_panda; 21: sheep; 22: snake; 23: squirrel; 24: tiger; 25: train; 26: turtle; 27: watercraft; 28: whale; 29: zebra.
- Annotation format: YOLO TXT, zero-based foreground ID + normalized center-x/center-y/width/height; manifests preserve GT, relative paths and label SHA256.
- Notes: validation manifest SHA256 `a375ebf0a2fb4c4aaac8cb810f5fef0d3579ad3f283484c536bb8904e89b1ac0`; canonical SHA256 `59c38606d753c5679297c7f7e932fc5e69c9f9dd0375d24ab5e5eed860869127`. No missing, empty or invalid label files. See [dataset audit](audit/dataset_audit.json).

## YOLO26n
- Selected run: `trained_models/best-imagenet-vid-yolo`.
- Checkpoint: [best.pt](../trained_models/best-imagenet-vid-yolo/best.pt).
- Checkpoint SHA256: `5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027`.
- Config: [locked inference YAML](configs/yolo26n.yaml); recovered from [config/imagenet-vid-yolo.yaml](../config/imagenet-vid-yolo.yaml).
- Training dataset: `D:/ImageNet-VID/ImageNet/data/ImageNet2015/imagenet_vid_yolo_sampled/imagenet_vid.yaml`.
- Validation dataset: common sampled ImageNet-VID validation, fixed 256-frame canonical control subset.
- Foreground class count: 30.
- Exact class list: 0: airplane; 1: antelope; 2: bear; 3: bicycle; 4: bird; 5: bus; 6: car; 7: cattle; 8: dog; 9: domestic_cat; 10: elephant; 11: fox; 12: giant_panda; 13: hamster; 14: horse; 15: lion; 16: lizard; 17: monkey; 18: motorcycle; 19: rabbit; 20: red_panda; 21: sheep; 22: snake; 23: squirrel; 24: tiger; 25: train; 26: turtle; 27: watercraft; 28: whale; 29: zebra.
- Validation mAP@0.5: **0.80443761**.
- Validation mAP@0.5:0.95: **0.55757464** (mean of ten area-integral AP metrics; not COCO 101-point interpolation).
- Validation Recall@0.5: **0.85065865**, mean class recall at confidence 0.01.
- Selection reason: winner among the three compatible shortlisted checkpoints on identical control frames, pixel resize, confidence, output NMS, class set and metric implementation; mAP50 primary, mAP50:95 for differences <0.005, then recall.
- Smoke: strict checkpoint load; real full-frame image at native source resolution and at 320x320, batch=1 CPU, finite decoded predictions, valid IDs and source-coordinate bounds; SSD variable inputs 96x96, 128x128, 160x160, 160x192 passed. [Native evidence](audit/native_full_frame_smoke.json).

## ROI-SSD
- Selected run: `trained_models/imagenet-vid-roissd-20261002`.
- Checkpoint: [best_map_stage2.pt](../trained_models/imagenet-vid-roissd-20261002/best_map_stage2.pt).
- Checkpoint SHA256: `e6f15e01032ce2f2d509f72f96efff6b0ed447d125b858fb099e44e4e23f79f3`.
- Config: [locked inference YAML](configs/roi_ssd.yaml); recovered from [trained_models/imagenet-vid-roissd-20261002/mixed_sampling_eval.yaml](../trained_models/imagenet-vid-roissd-20261002/mixed_sampling_eval.yaml).
- Training dataset: `D:\ImageNet-VID\ImageNet\data\ImageNet2015\imagenet_vid_yolo_sampled\imagenet_vid.yaml`.
- Validation dataset: common sampled ImageNet-VID validation, fixed 256-frame canonical control subset.
- Foreground class count: 30.
- Background handling: 31 outputs including background index 0; dataset foreground ID i becomes internal SSD ID i+1.
- Exact foreground class list: 0: airplane; 1: antelope; 2: bear; 3: bicycle; 4: bird; 5: bus; 6: car; 7: cattle; 8: dog; 9: domestic_cat; 10: elephant; 11: fox; 12: giant_panda; 13: hamster; 14: horse; 15: lion; 16: lizard; 17: monkey; 18: motorcycle; 19: rabbit; 20: red_panda; 21: sheep; 22: snake; 23: squirrel; 24: tiger; 25: train; 26: turtle; 27: watercraft; 28: whale; 29: zebra.
- Validation mAP@0.5: **0.85011720**.
- Validation mAP@0.5:0.95: **0.56924727** (mean of ten area-integral AP metrics; not COCO 101-point interpolation).
- Validation Recall@0.5: **0.89326247**, mean class recall at confidence 0.01.
- Selection reason: winner among the three compatible shortlisted checkpoints on identical control frames, pixel resize, confidence, output NMS, class set and metric implementation; mAP50 primary, mAP50:95 for differences <0.005, then recall.
- Smoke: strict checkpoint load; real full-frame image at native source resolution and at 320x320, batch=1 CPU, finite decoded predictions, valid IDs and source-coordinate bounds; SSD variable inputs 96x96, 128x128, 160x160, 160x192 passed. [Native evidence](audit/native_full_frame_smoke.json).

## ROI-SSD-MobileNet
- Selected run: `trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms`.
- Checkpoint: [best_map.pt](../trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms/best_map.pt).
- Checkpoint SHA256: `29a861219da99dbde13ba0c8004f945f3ece316d62c332f729712e370e79cd45`.
- Config: [locked inference YAML](configs/roi_ssd_mobilenet.yaml); recovered from [trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms/mixed_sampling_eval.yaml](../trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms/mixed_sampling_eval.yaml).
- Training dataset: `D:\ImageNet-VID\ImageNet\data\ImageNet2015\imagenet_vid_yolo_sampled\imagenet_vid.yaml`.
- Validation dataset: common sampled ImageNet-VID validation, fixed 256-frame canonical control subset.
- Foreground class count: 30.
- Background handling: 31 outputs including background index 0; dataset foreground ID i becomes internal SSD ID i+1.
- Exact foreground class list: 0: airplane; 1: antelope; 2: bear; 3: bicycle; 4: bird; 5: bus; 6: car; 7: cattle; 8: dog; 9: domestic_cat; 10: elephant; 11: fox; 12: giant_panda; 13: hamster; 14: horse; 15: lion; 16: lizard; 17: monkey; 18: motorcycle; 19: rabbit; 20: red_panda; 21: sheep; 22: snake; 23: squirrel; 24: tiger; 25: train; 26: turtle; 27: watercraft; 28: whale; 29: zebra.
- Validation mAP@0.5: **0.83608191**.
- Validation mAP@0.5:0.95: **0.52643326** (mean of ten area-integral AP metrics; not COCO 101-point interpolation).
- Validation Recall@0.5: **0.89220331**, mean class recall at confidence 0.01.
- Selection reason: winner among the three compatible shortlisted checkpoints on identical control frames, pixel resize, confidence, output NMS, class set and metric implementation; mAP50 primary, mAP50:95 for differences <0.005, then recall.
- Smoke: strict checkpoint load; real full-frame image at native source resolution and at 320x320, batch=1 CPU, finite decoded predictions, valid IDs and source-coordinate bounds; SSD variable inputs 96x96, 128x128, 160x160, 160x192 passed. [Native evidence](audit/native_full_frame_smoke.json).

## Class compatibility
- Status: `BACKGROUND_OFFSET_ONLY` for all 30 foreground schema entries.
- Mapping required: no semantic class mapping; convert SSD internal labels back to zero-based dataset IDs by subtracting 1.
- Background offset: +1 inside SSD; YOLO has no background output. The existing loader adds 1; existing YOLO adapters also add 1 for project-format output.
- Full compatibility table: [class_compatibility.csv](audit/class_compatibility.csv).

## Software
- commit: `f012b5b9dca5a917f761324be4a457d4ec64ea4b`.
- python: `3.13.5`.
- torch: `2.12.0+cu126`.
- torchvision: `0.27.0+cu126`.
- ultralytics: `8.4.60`.
- opencv: `4.10.0`.
- numpy: `1.26.4`.
- yaml: `6.0.1`.
- cpu_threads: `8`.
- device: `cpu`.

## Unresolved issues
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

## Stage gate
- Stage-1 selection and local CPU smoke: PASS. One explicit checkpoint per family is locked.
- Stop if a checkpoint/hash, dataset YAML/class order or frame manifest differs; never rediscover best.pt by name.
- Stage 2 must consume these files and preserve the documented scope. Pi smoke is still required before Pi experiments. No stage-2 anomaly diagnosis or long benchmark was run.
