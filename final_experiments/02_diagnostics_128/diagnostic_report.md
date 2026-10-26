# ROI-SSD 128x128 diagnostic report

Run status: **COMPLETE**. Complete stage-2 dense diagnostic sweep.

## Checkpoints and protocol

Checkpoints are the ROI-role specialists explicitly selected in the updated stage-1 manifest; full-frame specialists are not substituted. Penalized candidates are excluded.

- ROI-SSD: `trained_models/best-imagenet-vid-roissd/imagenet26022026stage3.pt`; SHA256 `e63200377a07c497c31d859b0a2dbbeabf6a1c1c7cafc0a7e795f70070d9a3fa`.
- ROI-SSD-MobileNet: `trained_models/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt`; SHA256 `7fa171ab669ac53f0d4f39586c08e61a60ca79fed98fd70874ef42bfd52356ce`.

256 fixed canonical source examples, shared across both models and all sizes. Confidence 0.5, NMS IoU 0.45, max detections 200. RGB project FixedSizeTestTransform, ImageNet normalization, per-image seed reused across sizes and families. Requested dimensions are fed exactly: no stride rounding or silent padding. Foreground IDs 1..30 correspond to dataset IDs 0..29; background is index 0.

The transform crops around annotated GT and can retain a different number of visible GT objects at different sizes. This is an object-centered ROI diagnostic, not whole-frame downsampling or a tracker benchmark. Tensor/GT hashes verify the same inputs across families.

mAP50:95 is the mean of the existing project area-integral AP at ten IoU thresholds, not COCO 101-point interpolation. Absent-class AP is undefined. Pre-NMS counts are class-wise candidates after confidence/top-k filtering; decoded box slots are recorded separately. Max-confidence mean uses unthresholded foreground softmax maxima; zero post-NMS scores are recorded separately.

## Structural boundaries

| Family | Input size | Active depth | Heads (zero-based) | Feature maps | Anchors | mAP50 |
|---|---:|---:|---|---|---:|---:|
| ROI-SSD | 96 | 3 | 0, 1, 2 | 12x12, 6x6, 3x3 | 846 | 0.6555 |
| ROI-SSD | 100 | 4 | 0, 1, 2, 3 | 13x13, 6x6, 3x3, 2x2 | 970 | 0.6557 |
| ROI-SSD | 104 | 4 | 0, 1, 2, 3 | 13x13, 6x6, 3x3, 2x2 | 970 | 0.6434 |
| ROI-SSD | 108 | 4 | 0, 1, 2, 3 | 14x14, 7x7, 4x4, 2x2 | 1198 | 0.5009 |
| ROI-SSD | 112 | 4 | 0, 1, 2, 3 | 14x14, 7x7, 4x4, 2x2 | 1198 | 0.4076 |
| ROI-SSD | 116 | 4 | 0, 1, 2, 3 | 15x15, 7x7, 4x4, 2x2 | 1314 | 0.2801 |
| ROI-SSD | 120 | 4 | 0, 1, 2, 3 | 15x15, 7x7, 4x4, 2x2 | 1314 | 0.2032 |
| ROI-SSD | 124 | 4 | 0, 1, 2, 3 | 16x16, 8x8, 4x4, 2x2 | 1528 | 0.0794 |
| ROI-SSD | 128 | 4 | 0, 1, 2, 3 | 16x16, 8x8, 4x4, 2x2 | 1528 | 0.0694 |
| ROI-SSD | 132 | 4 | 0, 1, 2, 3 | 17x17, 8x8, 4x4, 2x2 | 1660 | 0.0653 |
| ROI-SSD | 136 | 4 | 0, 1, 2, 3 | 17x17, 8x8, 4x4, 2x2 | 1660 | 0.0640 |
| ROI-SSD | 140 | 4 | 0, 1, 2, 3 | 18x18, 9x9, 5x5, 3x3 | 1986 | 0.6206 |
| ROI-SSD | 144 | 5 | 0, 1, 2, 3, 4 | 18x18, 9x9, 5x5, 3x3, 1x1 | 1990 | 0.6973 |
| ROI-SSD | 148 | 5 | 0, 1, 2, 3, 4 | 19x19, 9x9, 5x5, 3x3, 1x1 | 2138 | 0.7080 |
| ROI-SSD | 152 | 5 | 0, 1, 2, 3, 4 | 19x19, 9x9, 5x5, 3x3, 1x1 | 2138 | 0.7095 |
| ROI-SSD | 156 | 5 | 0, 1, 2, 3, 4 | 20x20, 10x10, 5x5, 3x3, 1x1 | 2408 | 0.7323 |
| ROI-SSD | 160 | 5 | 0, 1, 2, 3, 4 | 20x20, 10x10, 5x5, 3x3, 1x1 | 2408 | 0.7224 |
| ROI-SSD-MobileNet | 96 | 3 | 0, 1, 2 | 6x6, 3x3, 2x2 | 222 | 0.0346 |
| ROI-SSD-MobileNet | 100 | 4 | 0, 1, 2, 3 | 7x7, 4x4, 2x2, 1x1 | 322 | 0.5905 |
| ROI-SSD-MobileNet | 104 | 4 | 0, 1, 2, 3 | 7x7, 4x4, 2x2, 1x1 | 322 | 0.6010 |
| ROI-SSD-MobileNet | 108 | 4 | 0, 1, 2, 3 | 7x7, 4x4, 2x2, 1x1 | 322 | 0.5918 |
| ROI-SSD-MobileNet | 112 | 4 | 0, 1, 2, 3 | 7x7, 4x4, 2x2, 1x1 | 322 | 0.5806 |
| ROI-SSD-MobileNet | 116 | 4 | 0, 1, 2, 3 | 8x8, 4x4, 2x2, 1x1 | 382 | 0.3942 |
| ROI-SSD-MobileNet | 120 | 4 | 0, 1, 2, 3 | 8x8, 4x4, 2x2, 1x1 | 382 | 0.4115 |
| ROI-SSD-MobileNet | 124 | 4 | 0, 1, 2, 3 | 8x8, 4x4, 2x2, 1x1 | 382 | 0.4415 |
| ROI-SSD-MobileNet | 128 | 4 | 0, 1, 2, 3 | 8x8, 4x4, 2x2, 1x1 | 382 | 0.4274 |
| ROI-SSD-MobileNet | 132 | 4 | 0, 1, 2, 3 | 9x9, 5x5, 3x3, 2x2 | 552 | 0.0237 |
| ROI-SSD-MobileNet | 136 | 4 | 0, 1, 2, 3 | 9x9, 5x5, 3x3, 2x2 | 552 | 0.0389 |
| ROI-SSD-MobileNet | 140 | 4 | 0, 1, 2, 3 | 9x9, 5x5, 3x3, 2x2 | 552 | 0.0288 |
| ROI-SSD-MobileNet | 144 | 5 | 0, 1, 2, 3, 4 | 9x9, 5x5, 3x3, 2x2, 1x1 | 556 | 0.5960 |
| ROI-SSD-MobileNet | 148 | 5 | 0, 1, 2, 3, 4 | 10x10, 5x5, 3x3, 2x2, 1x1 | 632 | 0.5219 |
| ROI-SSD-MobileNet | 152 | 5 | 0, 1, 2, 3, 4 | 10x10, 5x5, 3x3, 2x2, 1x1 | 632 | 0.5507 |
| ROI-SSD-MobileNet | 156 | 5 | 0, 1, 2, 3, 4 | 10x10, 5x5, 3x3, 2x2, 1x1 | 632 | 0.5636 |
| ROI-SSD-MobileNet | 160 | 5 | 0, 1, 2, 3, 4 | 10x10, 5x5, 3x3, 2x2, 1x1 | 632 | 0.5815 |

## ROI-SSD

| Input | mAP50 | mAP50:95 | Recall50 | GT | Candidates pre-NMS | Detections/image | Mean max confidence | Mean best anchor IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.6555 | 0.4617 | 0.6651 | 331 | 1.02 | 0.95 | 0.9021 | 0.3577 |
| 128 | 0.0694 | 0.0124 | 0.1425 | 330 | 2.42 | 0.98 | 0.7997 | 0.3958 |
| 160 | 0.7224 | 0.4866 | 0.7343 | 330 | 4.88 | 1.04 | 0.9077 | 0.7347 |

Per-head evidence (each entry is an average over examples or GT, not a confidence-selected matching statistic):

| Input | Head | Feature map | Mean max foreground confidence | Mean best anchor IoU | Mean best decoded IoU before confidence/NMS |
|---:|---:|---|---:|---:|---:|
| 96 | 0 | 12x12 | 0.0137 | 0.0665 | 0.2809 |
| 96 | 1 | 6x6 | 0.0083 | 0.1719 | 0.5426 |
| 96 | 2 | 3x3 | 0.9021 | 0.3446 | 0.7536 |
| 128 | 0 | 16x16 | 0.0114 | 0.0713 | 0.2443 |
| 128 | 1 | 8x8 | 0.0209 | 0.1905 | 0.4446 |
| 128 | 2 | 4x4 | 0.0855 | 0.3343 | 0.6500 |
| 128 | 3 | 2x2 | 0.7898 | 0.3246 | 0.4928 |
| 160 | 0 | 20x20 | 0.0122 | 0.0796 | 0.2405 |
| 160 | 1 | 10x10 | 0.0407 | 0.2110 | 0.4155 |
| 160 | 2 | 5x5 | 0.1898 | 0.3997 | 0.6908 |
| 160 | 3 | 3x3 | 0.7408 | 0.5428 | 0.7621 |
| 160 | 4 | 1x1 | 0.7898 | 0.5780 | 0.6548 |

128 uses depth 4; that same depth is used at tested sizes 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140. A 128 accuracy valley is therefore not evidence of a depth switch occurring exactly at 128. Feature-grid resolution and head confidence must be checked separately.

**128x128 valley reproduced on this checkpoint.**
Classification: **C, consistent with feature/head or training-distribution sensitivity; a causal learned-feature explanation is not proven**.
All observed localization/classification/anchor box-slot counts agree. The mean best-anchor IoU does not show a corresponding >0.05 valley relative to both controls. This rules out a gross slot-count mismatch; it does not by itself prove anchor matching is optimal.
Compare 124/128/132 as well as the 96->100 and 140->144 boundaries in the table above. Depth selection is size-dependent; correlation with a feature-map/head boundary is evidence of sensitivity, not proof that the adaptive-depth code is incorrect.
The per-head table separates two mechanisms: which head has confidence high enough to survive filtering, and whether that head has anchors/regressed boxes that cover GT well. A shift toward a coarser confident head while a better-localizing head loses confidence supports joint head-specialization/feature-grid sensitivity. Anchor coverage can still constrain quality even when counts and the anchor-reference implementation agree.
**No implementation fix is justified solely by this accuracy valley.** No production model or checkpoint was changed. A causal head/depth intervention or training-distribution study is a follow-up if needed.

## ROI-SSD-MobileNet

| Input | mAP50 | mAP50:95 | Recall50 | GT | Candidates pre-NMS | Detections/image | Mean max confidence | Mean best anchor IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.0346 | 0.0071 | 0.0367 | 331 | 0.21 | 0.18 | 0.3035 | 0.2570 |
| 128 | 0.4274 | 0.1557 | 0.4551 | 330 | 1.58 | 1.06 | 0.6720 | 0.5262 |
| 160 | 0.5815 | 0.2861 | 0.5990 | 330 | 3.25 | 0.95 | 0.8483 | 0.6869 |

Per-head evidence (each entry is an average over examples or GT, not a confidence-selected matching statistic):

| Input | Head | Feature map | Mean max foreground confidence | Mean best anchor IoU | Mean best decoded IoU before confidence/NMS |
|---:|---:|---|---:|---:|---:|
| 96 | 0 | 6x6 | 0.0091 | 0.0598 | 0.1048 |
| 96 | 1 | 3x3 | 0.1072 | 0.1621 | 0.2308 |
| 96 | 2 | 2x2 | 0.2881 | 0.2284 | 0.4432 |
| 128 | 0 | 8x8 | 0.0164 | 0.0661 | 0.1145 |
| 128 | 1 | 4x4 | 0.0728 | 0.1663 | 0.2715 |
| 128 | 2 | 2x2 | 0.3483 | 0.2250 | 0.3333 |
| 128 | 3 | 1x1 | 0.5974 | 0.4746 | 0.5283 |
| 160 | 0 | 10x10 | 0.0218 | 0.0765 | 0.1317 |
| 160 | 1 | 5x5 | 0.0876 | 0.2009 | 0.3438 |
| 160 | 2 | 3x3 | 0.3482 | 0.3695 | 0.5435 |
| 160 | 3 | 2x2 | 0.1768 | 0.3201 | 0.4168 |
| 160 | 4 | 1x1 | 0.7678 | 0.5780 | 0.5682 |

128 uses depth 4; that same depth is used at tested sizes 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140. A 128 accuracy valley is therefore not evidence of a depth switch occurring exactly at 128. Feature-grid resolution and head confidence must be checked separately.

**D: the historical 128x128 valley did not reproduce on this selected checkpoint.**
Do not invent a cause for the old point. Use these checkpoint/protocol-specific measurements when replacing historical figures. This does not establish good accuracy at every tested size.

**Additional neighboring-size sensitivity:** 132, 136, 140 px have mAP50 below half of both the 128 and 160 control values. This is a descriptive screening rule, not a significance test or an acceptable-accuracy threshold.
Do not interpret a non-reproduced 128 valley as proof of robust arbitrary-size inference. Inspect the same-depth feature-grid transition, per-head confidence and anchor matching at these sizes. Counts and reference agreement can pass while the learned detector is ineffective on real segment sizes. An explicit padding/size policy would need a separate accuracy check; no silent rounding or model fix was applied.

## Anchor geometry and normalization

The independent center/size reference checks anchor order and normalized xyxy geometry against actual generated anchors. Statistics include both the returned clipped anchors and the reconstructed pre-clipping boxes; returned anchors may all be in bounds because the model clamps them. Normalized widths/heights are also converted using actual W/H into pixel dimensions. Rectangular probes distinguish width/height bookkeeping from square-only evidence.
Positive matches per GT at IoU >=0.5 are confidence-independent. A separate assigned-positive count reproduces one-GT-per-anchor matching and forced best-anchor assignment, which need not equal the simple pairwise threshold count.

## Stage-3 gate

**Structurally eligible on the tested MobileNet sizes and rectangular probes.** No observed slot-count, anchor-reference, finite-output, class-ID or coordinate-bound failure. This is not a guarantee of every possible segment shape or acceptable accuracy below the useful ROI-size range.
The two-checkpoint video pipeline itself has not been benchmarked here. Retraining was not performed or assumed necessary.

**Unrestricted MobileNet stage-3 runs remain on hold for size-policy validation:** the dense sweep found severe neighboring-size accuracy sensitivity even though structural checks passed. This is an accuracy limitation, not an identified structural/anchor implementation bug.

## Evidence and reproduction

- [Size sweep](size_sweep_metrics.csv)
- [Internal shapes and per-head confidence/IoU](internal_shapes.csv)
- [Anchor geometry](anchor_geometry.csv)
- [Anchor matching](anchor_matching.csv)
- [Per-GT matching](anchor_matching_per_gt.jsonl)
- [Per-class AP](per_class_ap.csv)
- [Per-example predictions and GT](per_example.jsonl)
- [Preprocessing hashes](preprocessing.csv)
- [Rectangular probes](rectangular_probes.csv)
- [Errors/assertions](errors_or_assertions.log)
- [Runtime metadata](run_metadata.json)

```powershell
.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.run --device auto
```

![mAP@0.5](map50_vs_input_size.png)

![Total anchors](total_anchors_vs_input_size.png)

![Mean best anchor IoU](max_anchor_iou_vs_input_size.png)
