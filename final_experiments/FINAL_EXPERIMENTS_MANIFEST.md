# Final experiments manifest

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
