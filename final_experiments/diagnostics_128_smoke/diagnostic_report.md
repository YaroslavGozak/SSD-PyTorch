# ROI-SSD 128x128 diagnostic report

Run status: **COMPLETE**. **SMOKE/PARTIAL evidence only; stage 2 is not complete.**

## Checkpoints and protocol

Checkpoints are the ROI-role specialists explicitly selected in the updated stage-1 manifest; full-frame specialists are not substituted. Penalized candidates are excluded.

- ROI-SSD: `trained_models/best-imagenet-vid-roissd/imagenet26022026stage3.pt`; SHA256 `e63200377a07c497c31d859b0a2dbbeabf6a1c1c7cafc0a7e795f70070d9a3fa`.
- ROI-SSD-MobileNet: `trained_models/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt`; SHA256 `7fa171ab669ac53f0d4f39586c08e61a60ca79fed98fd70874ef42bfd52356ce`.

4 fixed canonical source examples, shared across both models and all sizes. Confidence 0.5, NMS IoU 0.45, max detections 200. RGB project FixedSizeTestTransform, ImageNet normalization, per-image seed reused across sizes and families. Requested dimensions are fed exactly: no stride rounding or silent padding. Foreground IDs 1..30 correspond to dataset IDs 0..29; background is index 0.

The transform crops around annotated GT and can retain a different number of visible GT objects at different sizes. This is an object-centered ROI diagnostic, not whole-frame downsampling or a tracker benchmark. Tensor/GT hashes verify the same inputs across families.

mAP50:95 is the mean of the existing project area-integral AP at ten IoU thresholds, not COCO 101-point interpolation. Absent-class AP is undefined. Pre-NMS counts are class-wise candidates after confidence/top-k filtering; decoded box slots are recorded separately. Max-confidence mean uses unthresholded foreground softmax maxima; zero post-NMS scores are recorded separately.

## Structural boundaries

| Family | Input size | Active depth | Heads (zero-based) | Feature maps | Anchors | mAP50 |
|---|---:|---:|---|---|---:|---:|
| ROI-SSD | 96 | 3 | 0, 1, 2 | 12x12, 6x6, 3x3 | 846 | 1.0000 |
| ROI-SSD | 128 | 4 | 0, 1, 2, 3 | 16x16, 8x8, 4x4, 2x2 | 1528 | 0.2500 |
| ROI-SSD | 160 | 5 | 0, 1, 2, 3, 4 | 20x20, 10x10, 5x5, 3x3, 1x1 | 2408 | 1.0000 |
| ROI-SSD-MobileNet | 96 | 3 | 0, 1, 2 | 6x6, 3x3, 2x2 | 222 | 0.2500 |
| ROI-SSD-MobileNet | 128 | 4 | 0, 1, 2, 3 | 8x8, 4x4, 2x2, 1x1 | 382 | 0.5000 |
| ROI-SSD-MobileNet | 160 | 5 | 0, 1, 2, 3, 4 | 10x10, 5x5, 3x3, 2x2, 1x1 | 632 | 1.0000 |

## ROI-SSD

| Input | mAP50 | mAP50:95 | Recall50 | GT | Candidates pre-NMS | Detections/image | Mean max confidence | Mean best anchor IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 1.0000 | 0.8750 | 1.0000 | 4 | 1.00 | 1.00 | 0.9561 | 0.2464 |
| 128 | 0.2500 | 0.0250 | 0.2500 | 4 | 1.75 | 1.00 | 0.8644 | 0.3660 |
| 160 | 1.0000 | 0.7125 | 1.0000 | 4 | 3.50 | 1.00 | 0.8231 | 0.6880 |

**128x128 valley reproduced on this checkpoint.**
Classification: **C, consistent with feature/head or training-distribution sensitivity; a causal learned-feature explanation is not proven**.
All observed localization/classification/anchor box-slot counts agree. The mean best-anchor IoU does not show a corresponding >0.05 valley relative to both controls. This rules out a gross slot-count mismatch; it does not by itself prove anchor matching is optimal.
Compare 124/128/132 as well as the 96->100 and 140->144 boundaries in the table above. Depth selection is size-dependent; correlation with a feature-map/head boundary is evidence of sensitivity, not proof that the adaptive-depth code is incorrect.
**No implementation fix is justified solely by this accuracy valley.** No production model or checkpoint was changed. A causal head/depth intervention or training-distribution study is a follow-up if needed.

## ROI-SSD-MobileNet

| Input | mAP50 | mAP50:95 | Recall50 | GT | Candidates pre-NMS | Detections/image | Mean max confidence | Mean best anchor IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.2500 | 0.0750 | 0.2500 | 4 | 0.25 | 0.25 | 0.2730 | 0.2490 |
| 128 | 0.5000 | 0.0500 | 0.5000 | 4 | 2.00 | 1.25 | 0.7552 | 0.4531 |
| 160 | 1.0000 | 0.3312 | 1.0000 | 4 | 4.25 | 1.00 | 0.9814 | 0.6880 |

**D: the historical 128x128 valley did not reproduce on this selected checkpoint.**
Do not invent a cause for the old point. Use these checkpoint/protocol-specific measurements when replacing historical figures. This does not establish good accuracy at every tested size.

## Anchor geometry and normalization

The independent center/size reference checks anchor order and normalized xyxy geometry against actual generated anchors. Statistics include both the returned clipped anchors and the reconstructed pre-clipping boxes; returned anchors may all be in bounds because the model clamps them. Normalized widths/heights are also converted using actual W/H into pixel dimensions. Rectangular probes distinguish width/height bookkeeping from square-only evidence.
Positive matches per GT at IoU >=0.5 are confidence-independent. A separate assigned-positive count reproduces one-GT-per-anchor matching and forced best-anchor assignment, which need not equal the simple pairwise threshold count.

## Stage-3 gate

**BLOCKED:** full diagnostic scope or structural checks are incomplete/failed. Inspect errors_or_assertions.log. Do not infer clearance from a smoke run.
The two-checkpoint video pipeline itself has not been benchmarked here. Retraining was not performed or assumed necessary.

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
