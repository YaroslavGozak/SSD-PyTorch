# Stage 3a: ROI-SSD-MobileNet oracle segment diagnosis

**B — crop-domain robustness dominates**

## Locked protocol

32 videos / 320 frames, original pilot manifest, checkpoint, CPU (8 threads), confidence 0.5, NMS IoU 0.45, existing area-AP evaluator. FF and PRED_SEG were rerun and every saved prediction reproduced within 1e-6. PRED_SEG uses previous localization, hold=0, interval=10, no merge. ORACLE uses current-frame GT on every frame, with no keyframe/fallback/tracker memory.

**Padding coordinates:** pilot p=96 means 96 pixels in the base 320×320 model tensor, scaled independently to source-frame X/Y. ORACLE uses the same scaling and source→model→source round trip as PRED_SEG, rather than silently substituting 96 native-image pixels. Inference uses slices of the already normalized/resized full-frame tensor, exactly as the pilot.

| Mode | mAP50 | mAP50:95 | Recall50 | Coverage | Det/frame | FP/frame | Duplicates/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| FF | 0.739147 | 0.458323 | 0.754915 | 1.000000 | 1.387 | 0.131 | 0.000 |
| ORACLE_GT_SEG | 0.487592 | 0.170886 | 0.563414 | 1.000000 | 1.656 | 0.688 | 0.013 |
| PRED_SEG | 0.508578 | 0.192555 | 0.589260 | 0.958929 | 1.587 | 0.588 | 0.034 |
| ORACLE_GT_SEG_NO_GLOBAL_NMS | 0.467714 | 0.165029 | 0.577642 | 1.000000 | 2.366 | 1.341 | 0.319 |
| ORACLE_GT_SEG_P32 | 0.367422 | 0.164313 | 0.421486 | 1.000000 | 1.306 | 0.700 | 0.016 |
| ORACLE_GT_SEG_P64 | 0.403758 | 0.133529 | 0.477457 | 1.000000 | 1.716 | 1.056 | 0.009 |
| ORACLE_GT_SEG_P128 | 0.637146 | 0.340777 | 0.693074 | 1.000000 | 1.722 | 0.575 | 0.022 |

## Practical loss decomposition

| Metric | FF−PRED | FF−ORACLE (crop) | ORACLE−PRED (formation) | Crop share | Formation share |
|---|---:|---:|---:|---:|---:|
| mAP50 | 0.230570 | 0.251556 | -0.020986 | 109.1% | -9.1% |
| recall50 | 0.165654 | 0.191501 | -0.025847 | 115.6% | -15.6% |

ORACLE mAP50 retention versus FF: **65.97%**. These are practical contrasts, not independent causal effects. ORACLE also changes proposal count, geometry and refresh behavior. Negative components/shares, if present, are retained.

## Object geometry and misses

Each GT contributes exactly once at p=96, linked to its own GT-generated crop. Recall bins use unique one-to-one, class-correct matches after global NMS; IoU uses the best class-correct final prediction. Confidence is the maximum class-correct returned confidence in that object’s own crop (zero when absent). These are thresholded detector outputs, not raw logits. Crop dimensions/areas are in source pixels; actual network H/W are logged separately.

| Feature | Bin | N GT | Recall50 micro | Mean best IoU | Mean own-crop max confidence | Miss rate |
|---|---|---:|---:|---:|---:|---:|
| gt_area_ratio | [0,0.25) | 484 | 0.5083 | 0.4054 | 0.6672 | 0.4917 |
| gt_area_ratio | [0.25,0.5) | 59 | 0.7966 | 0.7205 | 0.9684 | 0.2034 |
| gt_area_ratio | [0.5,0.7) | 17 | 1.0000 | 0.8901 | 0.9960 | 0.0000 |
| gt_max_side_ratio | [0,0.25) | 115 | 0.4000 | 0.3772 | 0.5552 | 0.6000 |
| gt_max_side_ratio | [0.25,0.5) | 231 | 0.5325 | 0.4096 | 0.6629 | 0.4675 |
| gt_max_side_ratio | [0.5,0.7) | 158 | 0.5506 | 0.4448 | 0.7985 | 0.4494 |
| gt_max_side_ratio | [0.7,0.8) | 38 | 0.9737 | 0.7842 | 0.9469 | 0.0263 |
| gt_max_side_ratio | [0.8,0.9) | 18 | 0.9444 | 0.8765 | 0.9951 | 0.0556 |
| center_distance_ratio | [0,0.05) | 173 | 0.6590 | 0.4632 | 0.7338 | 0.3410 |
| center_distance_ratio | [0.05,0.1) | 78 | 0.7051 | 0.5261 | 0.8183 | 0.2949 |
| center_distance_ratio | [0.1,0.2) | 167 | 0.5749 | 0.5162 | 0.6864 | 0.4251 |
| center_distance_ratio | [0.2,0.3) | 60 | 0.3833 | 0.3558 | 0.7161 | 0.6167 |
| center_distance_ratio | [0.3,0.4) | 62 | 0.3548 | 0.3664 | 0.6427 | 0.6452 |
| center_distance_ratio | [0.4,0.5) | 17 | 0.0000 | 0.1366 | 0.4381 | 1.0000 |
| center_distance_ratio | [0.5,inf) | 3 | 0.0000 | 0.0313 | 0.4485 | 1.0000 |

Empty bins remain in CSV and support no conclusion. Bin associations are descriptive and confounded by class, source size, clipping and context; no monotonic or causal size effect is assumed.

## Duplicates and global NMS

The normal pipeline already performs global class-wise NMS at IoU 0.45 after remapping. The no-global-NMS variant retains detector-internal per-crop NMS and the same confidence/clipping. At p=96, global NMS changes oracle mAP50 from 0.467714 to 0.487592 (Δ=+0.019878). Duplicate counts mean repeated predictions matching an already matched GT; pair counts at IoU>0.5/>0.7 before/after NMS are separately provided in duplicate_nms_analysis.csv. FP counts include predictions from classes absent in a frame.

## Frame-level evidence

Twenty largest FF→ORACLE drops and twenty largest ORACLE→PRED drops are ranked by per-frame mAP50, then TP loss, with stable manifest order for ties. Each comparison overlay shows GT, actual snapped crop boundaries, predictions/confidence, selected L_R and feature-map shapes. Raw detections, crops and traces are retained in frame_records.jsonl.

## Next action

Next: inspect training crop distribution, context and anchor/head specialization; do not start with tracker tuning.
Production weights, thresholds, pipeline and padding decision are unchanged. CA/FULL-TECH and Raspberry Pi remain on hold; this diagnostic does not waive pilot acceptance.

## Explicit interpretation of the measured contrasts


1. **Oracle deficit:** FF−ORACLE = 0.251556 mAP50; ORACLE retains 65.97% of FF despite 100% GT coverage.
2. **Temporal contrast:** ORACLE−PRED = -0.020986. ORACLE is slightly worse, not better. PRED uses full-frame inference on 101/320 frames (31.6%); ORACLE always crops. The negative formation contrast does not establish beneficial temporal errors or exclude missed tracks.
3. **Dominant cause:** crop-domain robustness (Case B). The measured 109.1% crop / −9.1% formation mAP-loss shares are retained as signed practical contrasts, not interpreted as independent causal percentages.
4. **Object area:** the observed direction contradicts a large-object-collapse explanation. Area ratio <0.25: recall 50.83%, N=484; area ratio ≥0.25: 84.21%, N=76. There are no GT with area ratio ≥0.70, so extreme object filling is untested. Small objects, source resolution, class and head/anchor geometry remain plausible contributors.
5. **Position:** normalized center distance <0.1: recall 67.33%, N=251; distance ≥0.2: 31.69%, N=142; distance ≥0.4: 0.00%, N=20. This supports an association with edge displacement/clipped context, not an isolated causal effect. Center distance is sqrt(center_dx_ratio²+center_dy_ratio²).
6. **NMS:** global class-wise NMS is present. Removing it increases duplicate detections/frame from 0.0125 to 0.31875 and FP/frame from 0.6875 to 1.340625. Restoring it adds 0.019878 mAP50, yet leaves the much larger 0.251556 FF deficit. It slightly lowers macro Recall50 (0.577642→0.563414), consistent with a precision/recall tradeoff; this does not explain the main accuracy loss.
7. **Next:** prioritize crop/context distribution and head/anchor diagnostics over tracker tuning. Oracle mAP50 improves monotonically across p=32/64/96/128: 0.367422 → 0.403758 → 0.487592 → 0.637146. At p=128 it still trails FF by 0.102001 and retains only 86.20%. This supports context/domain sensitivity but does not separate context from changes in network shape, active depth or anchor sampling. Do not select p=128 post hoc or proceed to CA/FULL-TECH/Pi. A next controlled study should hold source objects fixed, vary crop context, log L_R/anchor IoU and compare the training crop-size distribution before changing training or production.

Macro evaluator Recall50 in mode_summary.csv and micro GT recall in geometry bins use different averaging; they should not be compared as identical metrics.

Reproduce:

```powershell
.venv/Scripts/python.exe -m experiments.cpu_pilot.oracle_diagnosis
.venv/Scripts/python.exe -m experiments.cpu_pilot.summarize_oracle_diagnosis
```
