# Stage 3b: controlled crop-domain diagnosis

**Conclusion: context, network input scale/head regime and historical preprocessing distribution interact. Crop-first alone is not the explanation. No production change or retraining was performed.**

## Protocol and evidence

Locked 32 videos / 320 frames / 560 GT, unchanged MobileNet checkpoint, CPU 8 threads, until_1x1, confidence 0.5, internal/global class-wise NMS 0.45 and the original evaluator. No tracker, merge or refresh. p=32/64/96/128 exactly replay Stage 3a predictions. Padding remains in base 320×320 tensor units. Each GT has five paired context observations.

Frame metrics combine all current-frame GT crops with global NMS. Object trajectories, border comparisons and head attribution primarily use detection in that GT’s own crop: this isolates the controlled object from help/suppression by another object’s crop. Both own-crop and after-global-NMS outcomes are saved. These recalls differ from the class-macro evaluator recall and must not be equated. Target-only FP counts in long object rows are not frame FP counts; valid neighboring GT may be present in a crop.

## A. Paired context trajectories

| p | Frame mAP50 | Macro Recall50 | Own-crop Recall50 | Mean GT area ratio | Best anchor IoU | Confidence | Prediction IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.367422 | 0.421486 | 0.341071 | 0.302402 | 0.625478 | 0.536020 | 0.317940 |
| 64 | 0.403758 | 0.477457 | 0.335714 | 0.179475 | 0.593111 | 0.636345 | 0.338484 |
| 96 | 0.487592 | 0.563414 | 0.467857 | 0.134235 | 0.627039 | 0.708960 | 0.398098 |
| 128 | 0.637146 | 0.693074 | 0.589286 | 0.113757 | 0.606943 | 0.805467 | 0.491203 |
| 160 | 0.701172 | 0.748650 | 0.651786 | 0.104388 | 0.603530 | 0.817119 | 0.531615 |

At p=160, mAP50 retention versus FF is 94.86%; the remaining mAP50 gap is 0.037976. 145/560 GT (25.9%) are MISS at p96 and TP at p128/p160 (48.7% of p96 own-crop misses). 195/560 (34.8%) are still MISS at p160; 128/560 (22.9%) are MISS at every padding. The wide trajectories include first TP, confidence, prediction/anchor IoU, depth, final shape, and nonmonotonic outcomes; larger context is not assumed to improve every object.

| Padding transition | MISS→TP | Fraction of all GT | Fraction of prior misses | TP→MISS |
|---|---:|---:|---:|---:|
| 32→64 | 58 | 10.4% | 15.7% | 61 |
| 64→96 | 113 | 20.2% | 30.4% | 39 |
| 96→128 | 91 | 16.2% | 30.5% | 23 |
| 128→160 | 66 | 11.8% | 28.7% | 31 |

## B. Clipped versus outside-frame padding at p96

504/560 nominal symmetric ROIs cross a source boundary. The padded arm retains the nominal tensor-grid ROI and fills outside with exactly zero after ImageNet normalization (equivalent to ImageNet mean RGB before normalization). No reflection/replication is used. Remapped outputs are clipped to the real source frame before scoring.

| Variant | N GT | Own-crop Recall50 | Mean prediction IoU | Mean confidence | Mean anchor IoU |
|---|---:|---:|---:|---:|---:|
| CLIPPED | 504 | 0.462302 | 0.393370 | 0.714591 | 0.629470 |
| PADDED_OUTSIDE_FRAME | 504 | 0.519841 | 0.402587 | 0.680025 | 0.634678 |

| Clipped center distance | N GT | Clipped recall | Padded recall |
|---|---:|---:|---:|
| [0,0.1) | 195 | 0.6667 | 0.6256 |
| [0.1,0.2) | 167 | 0.4431 | 0.5269 |
| [0.2,0.3) | 60 | 0.2833 | 0.4333 |
| [0.3,0.4) | 62 | 0.1935 | 0.4194 |
| [0.4,inf) | 20 | 0.0000 | 0.0000 |

For the full set (unchanged crops for non-border GT), PADDED mAP50 is 0.488478, macro Recall50 0.548930; clipped p96 is 0.487592/0.563414. Thus own-crop recovery must not be mistaken for an overall AP/recall improvement. The border intervention changes position, valid context fraction, H/W and head/anchor geometry together. Edge position is associated with failure but is not established as an independent cause.

## C. Training evidence and preprocessing comparison

**Exact training revision is unavailable.** The checkpoint contains stage=2, epoch=18 (zero based), best_map=0.693501514842189, matching training_metrics.csv epoch 19. It has no source hash, launch command or crop manifest. The saved YAML is an evaluation snapshot. September Git source is used only as a contemporaneous reconstruction; an uncommitted training working tree cannot be ruled out.

Historical source `03042cc32ad3daefdbc958a9283e78dbaa8ca797` uses native RGB CHW float input, GT-reference center/scale jitter and random source-pixel padding, clips a source ROI, resizes each crop independently with bilinear interpolation (align_corners=False), then applies ImageNet normalization. No photometric transform or RandomIoUCrop is passed by that mixed-training entry point. Stage-2 output sides are 300/224/160/96/64, with probabilities 0.5/0.2/0.1/0.1/0.1; they are not automatically scaled to YAML im_size=320 in that revision. The historical processor builds a map from stages 2 then 3: repeated mode names are overwritten by stage-3 jitter/context settings. The reconstruction executes that behavior and records the effective mode map. October’s rewritten sampler is not substituted as September-training evidence.

CURRENT uses the exact Stage 3a tensor slice. CROP_FIRST_MATCHED_HW crops the same snapped source ROI and resizes it to exactly the current tensor H/W, isolating order/interpolation at fixed geometry. CROP_FIRST_224 was declared as the primary historical large-ROI mode; CROP_FIRST_300 is a declared historical full-output-size control, not a post-hoc production choice. Both resize to a square and therefore additionally change sampling density, aspect distortion, active depth and head geometry. Every arm uses the exact same source ROI/GT.

| Variant | Frame mAP50 | Macro Recall50 | Own-crop Recall50 | Mean confidence | Pred IoU | Anchor IoU |
|---|---:|---:|---:|---:|---:|---:|
| CURRENT_PIPELINE | 0.487592 | 0.563414 | 0.467857 | 0.708960 | 0.398098 | 0.627039 |
| CROP_FIRST_MATCHED_HW | 0.445637 | 0.521193 | 0.442857 | 0.704943 | 0.388113 | 0.627039 |
| CROP_FIRST_224 | 0.386460 | 0.455946 | 0.469643 | 0.726351 | 0.372926 | 0.611635 |
| CROP_FIRST_300 | 0.750849 | 0.795957 | 0.660714 | 0.804918 | 0.519143 | 0.641573 |

| CROP_FIRST variant (own crop) | TP both | TP only CURRENT | TP only CROP_FIRST | MISS both |
|---|---:|---:|---:|---:|
| CROP_FIRST_MATCHED_HW | 241 | 21 | 7 | 291 |
| CROP_FIRST_224 | 165 | 97 | 98 | 200 |
| CROP_FIRST_300 | 226 | 36 | 144 | 154 |

Matched-H/W crop-first preprocessing does not recover accuracy. This control also changes interpolation/pixel alignment (historical bilinear without antialiasing versus the pilot full-frame torchvision resize); it is not a normalization-only intervention. Resizing to 224 also underperforms CURRENT; resizing to 300 improves mAP50/Recall50. Therefore these measurements implicate input resolution/head regime and preprocessing distribution together, not a simple normalization-order bug. The best 300 result still has much lower mAP50:95 than FF, so its mAP50 advantage is not equivalent to restoring full localization quality.

## D. Head/anchor transitions

Per-head rows include spatial shape, anchor count, target anchor IoU, raw class-correct foreground confidence, best decoded IoU and own-crop matched head. Scores are softmax probabilities; decoded IoU scans all anchors before score filtering. Matched-head provenance is recovered from returned box+class+score against decoded anchors at tolerance 1e-6 (first matching source if a tie). This is descriptive attribution, not a head ablation.

The table uses unique GT rescued from p96 to their first successful p128/p160 own crop. Classification is exclusive in table order: a newly active head must supply the match; otherwise unchanged head set and anchor IoU gain ≥0.05; otherwise |anchor delta|<0.05 and raw confidence gain ≥0.10 with |decoded-IoU delta|<0.10; otherwise unchanged head set, |anchor delta|<0.05, |raw confidence delta|≤0.05 and decoded-IoU gain ≥0.10; remaining cases are mixed/unclear. “Same anchor regime” here is a coarse diagnostic label, not identical anchor coordinates. Thresholds are diagnostic bins, not inference thresholds.

| Mechanism | N MISS→TP | Fraction |
|---|---:|---:|
| New active head appeared | 0 | 0.0% |
| Same heads, anchor IoU improved | 35 | 24.1% |
| Same anchor regime, confidence improved | 5 | 3.4% |
| Same confidence, localization improved | 0 | 0.0% |
| Mixed/unclear | 105 | 72.4% |

57/145 rescued GT change selected depth; 0/145 change final feature shape. Although 57 rescues change depth, none is directly supplied by a newly activated head under the recovered matched-head attribution. Existing-head feature/score/regression changes therefore deserve attention alongside the geometry changes. Mean best-anchor IoU does not rise monotonically with context while confidence/prediction quality generally recover. Anchor overlap alone is not a sufficient explanation.

## E. Reconstructed training distribution versus pilot

A fixed-seed sample of 1,000 distinct training-manifest frames was processed by the historical mixed sampler, without training the model. One reference GT is measured per sampled crop; full-mode crops use one random GT. This is a representative reconstruction under available code/config, not the actual crops seen by the checkpoint. Both all-mode and ROI-only distributions are shown to avoid hiding the 50% full-frame mixture. Classes/reference-object sampling and source-frame distributions differ, so this is descriptive domain-shift evidence.

| Feature | Domain | N | Median | p05 | p95 | Descriptive KS vs pilot |
|---|---|---:|---:|---:|---:|---:|
| gt_area_ratio | historical_training_proxy_all | 1000 | 0.3948 | 0.0106 | 0.9377 | 0.4665 |
| gt_area_ratio | historical_training_proxy_ROI_only | 489 | 0.7551 | 0.1570 | 0.9541 | 0.7828 |
| gt_area_ratio | pilot_oracle_p96 | 560 | 0.0724 | 0.0169 | 0.4723 | 0.0000 |
| center_distance_ratio | historical_training_proxy_all | 1000 | 0.0752 | 0.0126 | 0.3456 | 0.1466 |
| center_distance_ratio | historical_training_proxy_ROI_only | 489 | 0.0379 | 0.0086 | 0.1065 | 0.4889 |
| center_distance_ratio | pilot_oracle_p96 | 560 | 0.1093 | 0.0021 | 0.3819 | 0.0000 |
| network_aspect_ratio | historical_training_proxy_all | 1000 | 1.0000 | 1.0000 | 1.0000 | 0.7036 |
| network_aspect_ratio | historical_training_proxy_ROI_only | 489 | 1.0000 | 1.0000 | 1.0000 | 0.7036 |
| network_aspect_ratio | pilot_oracle_p96 | 560 | 0.8902 | 0.5118 | 1.2158 | 0.0000 |
| min_normalized_boundary_distance | historical_training_proxy_all | 1000 | 0.0338 | 0.0000 | 0.3167 | 0.2926 |
| min_normalized_boundary_distance | historical_training_proxy_ROI_only | 489 | 0.0036 | 0.0000 | 0.2445 | 0.4585 |
| min_normalized_boundary_distance | pilot_oracle_p96 | 560 | 0.1506 | 0.0000 | 0.3936 | 0.0000 |

Historical network crops are square at fixed scheduled sizes, whereas pilot inputs are variable rectangles cut from a full-frame 320×320 resize. Source-pixel training padding also differs from pilot padding scaled from tensor units. Area/center/context histograms quantify the additional shifts; they do not prove the checkpoint saw this exact distribution. In this reconstruction, ROI-only training has median GT area ratio 0.7551 versus pilot 0.0724, and median center distance 0.0379 versus pilot 0.1093: pilot objects are typically much smaller in the crop and farther from its center.

## Answers and next decision

1. More context substantially restores accuracy for the same GT, but changes input H/W and heads at the same time. p160 reaches 94.86% of FF mAP50 on this oracle study; it is not a new production padding selection.
2. Edge position is not isolated as an independent factor: clipping, centering, context and network geometry change jointly. Paired border bins provide the measured effect.
3. Outside-frame padding improves own-crop recall in aggregate but barely changes full-set mAP50 and lowers macro recall. It is not a demonstrated standalone fix.
4. Available historical training code differs from CURRENT; exact run provenance remains unverified. No assumption is made that current training source describes the old checkpoint.
5. CROP_FIRST is not uniformly better. Fixed-H/W and 224 comparisons are worse, while 300 is better at mAP50. This points to a resolution/head/preprocessing interaction.
6. The rescue table measures new-head attribution and mixed changes; it does not establish causality without head/geometry-controlled ablations.
7. Reconstructed training and pilot geometry distributions differ, especially fixed square versus variable rectangular network inputs. Historical identity cannot be established.
8. Most likely: a combination of context loss and input-scale/head specialization under a shifted preprocessing/crop distribution; edge sensitivity contributes but zero padding alone does not remove the frame-level loss. Anchor IoU alone is insufficient.
9. A no-retraining inference candidate exists (crop-first 300), but this oracle result does not demonstrate a production fix: normal PRED_SEG, localization quality and CPU cost have not been validated for that candidate. No automatic rollout is justified.
10. If a separate fixed-protocol validation cannot meet the original acceptance/cost criteria, test training on the deployment path: full-frame resize then variable rectangular slices, matched context/offset/border distributions, coverage across head transition sizes, and a controlled mixture retaining full-frame behavior. Compare this against crop-first training with a deliberately matched inference path. Recover exact source/config/seed and save crop manifests in that future run; no retraining is performed here.

Production weights, anchors, loss, thresholds, NMS, tracker, merge, padding and acceptance criteria are unchanged. CA/FULL-TECH/Pi remain on hold. All claims are limited to this validation pilot; multiple frames per video are correlated. No statistical independence or untouched-test generalization is claimed.

## Figures

![Recall versus padding](recall_vs_padding.png)

![Confidence versus padding](confidence_vs_padding.png)

![Prediction IoU versus padding](pred_iou_vs_padding.png)

![Border intervention by center offset](clipped_vs_padded_by_center_offset.png)

![Reconstructed training versus pilot area ratio](train_vs_pilot_area_ratio.png)

![Reconstructed training versus pilot center offset](train_vs_pilot_center_offset.png)

![Network aspect ratio](train_vs_pilot_aspect_ratio.png)

![Minimum normalized context](train_vs_pilot_context_to_edge.png)

Reproduce:

```powershell
.venv/Scripts/python.exe -m experiments.cpu_pilot.crop_domain
.venv/Scripts/python.exe -m experiments.cpu_pilot.crop_training_proxy
.venv/Scripts/python.exe -m experiments.cpu_pilot.crop_domain_validate
.venv/Scripts/python.exe -m experiments.cpu_pilot.crop_domain_report
```
