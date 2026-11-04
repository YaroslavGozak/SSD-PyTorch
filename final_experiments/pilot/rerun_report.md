# Standalone CPU pilot report: locked October 2026 checkpoints

## 1. Outcome and execution scope

The pilot evaluated YOLO26n and the newly locked ROI-SSD-MobileNet checkpoint on the same 320 labeled frames from 32 videos. **MobileNet was run at paddings 32, 64, and 96**, as well as full frame. Each setting has 320 distinct frame records covering all 32 videos. These were completed in the rerun; this report verifies and consolidates those measurements rather than presenting a second execution as new evidence.

Coordinate smoke and CPU cost calibration completed. YOLO meets the predeclared accuracy/coverage criteria at padding 64. MobileNet meets neither criterion at any tested padding. Consequently, execution status is **BLOCKED_PIPELINE_ACCEPTANCE**: FF and SEG measurements are complete, while CA and FULL-TECH were not benchmarked. Padding 96 is MobileNet's prescribed largest-tested fallback, not an accepted production setting. No Raspberry Pi or final evaluation was performed.

The non-MobileNet ROI-SSD checkpoint `trained_models/imagenet-vid-roissd-20261028/stage3/best_map.pt` remains selected for other tests but was not evaluated in this stage. Stage 3 specifies YOLO26n and ROI-SSD-MobileNet.

## 2. Locked models and inference configuration

### YOLO26n

- Checkpoint: `trained_models/best-imagenet-vid-yolo/best.pt`.
- Checkpoint SHA-256: `5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027`.
- Exact configuration: `final_experiments/pilot/configs/yolo26n.yaml`.
- Configuration SHA-256: `585fa26b5c5612f1cc9942d05e84fb0b86ffc15ed472aac9a5b042d06290bd72`.

### ROI-SSD-MobileNet

- Checkpoint: `trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt`.
- Checkpoint SHA-256: `1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804`.
- Exact configuration: `final_experiments/pilot/configs/roi_ssd_mobilenet.yaml`.
- Configuration SHA-256: `6ca85c1c7238fc60490b63d670cc4b1baf68a34d384d0c51b2754597aafea932`.

| Parameter | YOLO26n | ROI-SSD-MobileNet |
|---|---|---|
| Architecture | YOLO26n via project YOLO adapter | roissd-mobilenet |
| Base frame input | 320 x 320 RGB | 320 x 320 RGB |
| Confidence threshold | 0.01 | 0.5 |
| Output NMS IoU | 0.45 | 0.45 |
| Maximum detections per frame | 200 | 200 |
| ROI dimension alignment | 32 pixels | 1 pixel |
| Foreground classes | 30 schema classes | 30 schema classes |
| Background handling | No native background output; adapter exposes project labels 1-30 | 31 outputs; background 0, foreground 1-30 |
| Merge during padding sweep | Disabled | Disabled |

The dataset uses zero-based foreground IDs 0-29. The common project output uses foreground IDs 1-30, so compatibility requires only the existing background offset, with no semantic remapping. The loader performs full-frame squash resizing to 320 x 320 and ImageNet normalization. The YOLO adapter reverses that normalization to obtain the expected RGB [0,1] input. Its ROI tensors are stride aligned. Actual network shapes are observed; requested crop area is not substituted for network tensor area.

MobileNet model settings: scales [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]; aspect ratios [[1.0, 2.0, 0.5], [1.0, 2.0, 3.0, 0.5, 0.333], [1.0, 2.0, 3.0, 0.5, 0.333], [1.0, 2.0, 3.0, 0.5, 0.333], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]; pre-NMS top-K 400; model low-score threshold is set from the inference confidence threshold. The selected checkpoint is `best_map.pt`, not the run configuration's original `last.pt`. The model configuration was copied from the selected run; no training was performed.

## 3. Dataset and split

- Dataset: sampled, YOLO-converted ImageNet-VID validation.
- Dataset root: `D:\ImageNet-VID\ImageNet\data\ImageNet2015\imagenet_vid_yolo_sampled`.
- Dataset YAML: `D:\ImageNet-VID\ImageNet\data\ImageNet2015\imagenet_vid_yolo_sampled\imagenet_vid.yaml`.
- Full source validation: 5,064 labeled frames from 508 source videos.
- Pilot: 320 frames from 32 videos, 10 frames per video.
- Final-evaluation candidate: 4,744 frames from 476 videos; not evaluated here.
- Pilot selection: seeded video-level sample, seed 20261112, excluding videos in the earlier canonical control subset.
- Pilot/candidate video overlap: 0. Pilot/canonical video overlap: 0.
- Pilot manifest SHA-256: `817f5b145b774dc91ee0fa8d271e02c11d71d70bb50ff13082a0ba9487aa63cb`.
- Candidate manifest SHA-256: `4f36f3455ced1a344ef69329479cd09013934d71aafa8dc800d7f7e5ea6bf6d4`.
- Dataset YAML SHA-256: `29ea3cd7d5f6acbca59decce76ef2c1491d8c1c7661eeff45263c8a26d45ba22`.

Exact foreground class order: 0: airplane; 1: antelope; 2: bear; 3: bicycle; 4: bird; 5: bus; 6: car; 7: cattle; 8: dog; 9: domestic_cat; 10: elephant; 11: fox; 12: giant_panda; 13: hamster; 14: horse; 15: lion; 16: lizard; 17: monkey; 18: motorcycle; 19: rabbit; 20: red_panda; 21: sheep; 22: snake; 23: squirrel; 24: tiger; 25: train; 26: turtle; 27: watercraft; 28: whale; 29: zebra.

The pilot contains 560 annotated objects across 17 represented classes. Class/object counts: airplane: 70, antelope: 20, bear: 10, bicycle: 110, bus: 10, car: 60, cattle: 10, dog: 70, elephant: 40, fox: 10, horse: 10, monkey: 10, motorcycle: 20, rabbit: 10, squirrel: 30, train: 30, whale: 40. AP is averaged over GT-present classes, not all 30 schema classes. Mean annotated object area / source-frame area is 0.098067.

Pilot video IDs: `00007008, 00007016, 00021002, 00023010, 00026002, 00028002, 00037004, 00039001, 00041011, 00041012, 00042000, 00042017, 00044003, 00044006, 00050000, 00053002, 00063002, 00069003, 00075001, 00084000, 00087000, 00104000, 00118004, 00118007, 00118008, 00135002, 00144000, 00150003, 00160000, 00165001, 00173003, 00174000`.

Frames within a video are temporally correlated. Although the pilot and candidate split are video disjoint, the source validation dataset was previously reused for training validation, model selection and diagnostics. The candidate is not an untouched external test set. No full-VID generalization or statistical significance is established by this pilot.

## 4. Pipeline and measurement protocol

FF processes the complete resized frame on every frame. SEG uses detections from the previous processed frame to form static padded regions. Tracking state resets at video boundaries. Padding extends each side of the detection box by p pixels in the 320 x 320 model coordinate system, then converts to source coordinates and clips to the source frame. It is not a fixed count of native source-image pixels.

SEG uses periodic full-frame inference when the source frame index is divisible by 10, hold=0, and a full-frame fallback when no usable segments exist. Overlapping segments are processed separately during this sweep. Crops use the base frame tensor, with local detections mapped back to the source frame; global duplicate handling/NMS follows detection. No motion prediction or merge sweep was added.

For each model, choose the smallest p in {32,64,96} satisfying both:

```text
mAP50_SEG / mAP50_FF >= 0.95
gt_roi_coverage_mean >= 0.98
```

If none qualifies, record p=96 as the largest-tested fallback and mark criterion not met. The pipeline gate now checks both requirements for both models before CA/FULL-TECH. Thresholds and the tested padding set were not changed after viewing results.

One measured 320-frame series was performed per model/setting, preceded by two detector warmups. Batch size is 1. CPU controls are 8 PyTorch intra-op threads, 1 inter-op thread and 1 OpenCV thread. OMP_NUM_THREADS=8, MKL_NUM_THREADS=8 and OPENBLAS_NUM_THREADS=1. No GPU forward is used in the measured runs; a CUDA availability message printed by an imported helper does not describe the pilot execution device.

Process-frame timing includes formation/tracking, crop/preprocess work, neural inference and postprocessing. Dataset image loading and metric computation are excluded. Debug exports are outside measured process-frame timing. FPS is frames / total measured process-frame time; it is not video-file decoding or display throughput. There is no repeated-series confidence interval or thermal-control claim.

Metric definitions: mAP50 uses IoU=0.5 and area-integral precision-recall AP; mAP50:95 averages ten AP values at IoUs 0.50,0.55,...,0.95 and is not COCO 101-point interpolation. Recall50 is mean class recall at the configured confidence threshold. GT coverage counts a GT object as covered if at least one submitted region covers at least 50% of its area; full-frame inference covers all. Coverage p5 is the fifth percentile of per-frame covered-object fractions.

## 5. Complete quality results: all models and paddings

| Model | Mode / padding | Frames | mAP50 | mAP50:95 | Recall50 | mAP50 / FF | GT coverage | Coverage p5 | Both criteria |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26n | FF | 320 | 0.818573 | 0.575973 | 0.843156 | 1.000000 | 1.000000 | 1.000000 | Baseline |
| YOLO26n | SEG p=32 | 320 | 0.749158 | 0.518369 | 0.842291 | 0.915200 | 0.994643 | 1.000000 | FAIL |
| YOLO26n | SEG p=64 | 320 | 0.797707 | 0.548352 | 0.858193 | 0.974509 | 1.000000 | 1.000000 | PASS |
| YOLO26n | SEG p=96 | 320 | 0.814665 | 0.557426 | 0.863987 | 0.995226 | 1.000000 | 1.000000 | PASS |
| ROI-SSD-MobileNet | FF | 320 | 0.741016 | 0.427778 | 0.760466 | 1.000000 | 1.000000 | 1.000000 | Baseline |
| ROI-SSD-MobileNet | SEG p=32 | 320 | 0.612503 | 0.370310 | 0.623739 | 0.826571 | 0.837500 | 0.500000 | FAIL |
| ROI-SSD-MobileNet | SEG p=64 | 320 | 0.595105 | 0.301519 | 0.626719 | 0.803093 | 0.953571 | 0.750000 | FAIL |
| ROI-SSD-MobileNet | SEG p=96 | 320 | 0.675346 | 0.364520 | 0.703266 | 0.911378 | 0.966071 | 1.000000 | FAIL |

MobileNet p=32 achieves mAP50 0.612503 with 83.75% GT coverage. At p=64, coverage improves to 95.36%, but mAP50 falls to 0.595105. At p=96, mAP50 rises to 0.675346 and coverage reaches 96.61%. Padding therefore does not produce monotonic accuracy in this measured pipeline. Changing crop context, shape, previous detections and fallback behavior are coupled; these aggregate results do not establish which factor causes the p=64 drop.

YOLO p=32 fails accuracy retention despite passing coverage. Both p=64 and p=96 pass; the predetermined smallest-padding rule selects p=64. MobileNet has no eligible padding. Its best retention among these settings is 91.14% at p=96, 3.86 percentage points below the retention requirement; coverage is 1.39 percentage points below its requirement. The fallback must not be presented as an accepted final configuration.

## 6. Runtime and work performed

### Process-frame latency and throughput

| Model | Mode / padding | Mean ms | p50 ms | p95 ms | FPS | FF / SEG latency speedup |
|---|---|---|---|---|---|---|
| YOLO26n | FF | 19.615 | 19.528 | 21.737 | 50.98 | 1.000x |
| YOLO26n | SEG p=32 | 74.036 | 21.985 | 340.445 | 13.51 | 0.265x |
| YOLO26n | SEG p=64 | 75.040 | 22.905 | 342.887 | 13.33 | 0.261x |
| YOLO26n | SEG p=96 | 77.161 | 23.589 | 319.858 | 12.96 | 0.254x |
| ROI-SSD-MobileNet | FF | 24.262 | 23.509 | 29.866 | 41.22 | 1.000x |
| ROI-SSD-MobileNet | SEG p=32 | 27.997 | 24.228 | 52.487 | 35.72 | 0.867x |
| ROI-SSD-MobileNet | SEG p=64 | 31.194 | 24.171 | 72.442 | 32.06 | 0.778x |
| ROI-SSD-MobileNet | SEG p=96 | 32.963 | 24.122 | 76.351 | 30.34 | 0.736x |

A speedup below 1 means SEG is slower than the same model's FF baseline. Every tested SEG setting is slower than FF on this CPU. These are measured SEG results with merge disabled; they do not establish CA or FULL-TECH performance.

### Detector inference latency per NN call

| Model | Mode / padding | Mean ms/call | p50 ms/call | p95 ms/call |
|---|---|---|---|---|
| YOLO26n | FF | 19.390 | 19.310 | 21.557 |
| YOLO26n | SEG p=32 | 10.191 | 8.974 | 18.987 |
| YOLO26n | SEG p=64 | 13.068 | 11.961 | 21.346 |
| YOLO26n | SEG p=96 | 15.448 | 14.387 | 23.463 |
| ROI-SSD-MobileNet | FF | 24.133 | 23.408 | 29.705 |
| ROI-SSD-MobileNet | SEG p=32 | 23.724 | 22.441 | 35.937 |
| ROI-SSD-MobileNet | SEG p=64 | 22.856 | 22.233 | 32.479 |
| ROI-SSD-MobileNet | SEG p=96 | 23.987 | 22.917 | 32.946 |

Inference latency is per call, not per frame. It includes the project detector adapter/network forward and detector-internal decoding/postprocessing. SEG can make several calls in one frame, so its per-call latency is not directly comparable to total process-frame latency.

### Mean per-frame time by recorded component

| Model | Mode / padding | NN ms/frame | Crop/preprocess ms | Postprocess ms | Formation ms | Merge ms |
|---|---|---:|---:|---:|---:|---:|
| YOLO26n | FF | 19.390 | 0.000 | 0.131 | 0.014 | 0.000 |
| YOLO26n | SEG p=32 | 72.611 | 0.322 | 0.229 | 0.027 | 0.082 |
| YOLO26n | SEG p=64 | 73.792 | 0.257 | 0.203 | 0.024 | 0.071 |
| YOLO26n | SEG p=96 | 76.034 | 0.222 | 0.193 | 0.022 | 0.064 |
| ROI-SSD-MobileNet | FF | 24.133 | 0.000 | 0.070 | 0.011 | 0.000 |
| ROI-SSD-MobileNet | SEG p=32 | 27.802 | 0.038 | 0.067 | 0.011 | 0.030 |
| ROI-SSD-MobileNet | SEG p=64 | 30.998 | 0.041 | 0.060 | 0.010 | 0.029 |
| ROI-SSD-MobileNet | SEG p=96 | 32.757 | 0.042 | 0.064 | 0.010 | 0.030 |

NN ms/frame sums inference calls before averaging across frames. Components are separately instrumented and may not sum exactly to the complete process-frame timer because of timer boundaries and other bookkeeping. Disabled merge has no cost-aware merging work; any recorded merge time is the call/dispatch overhead of the disabled strategy.

### Processed area, segment counts and fallback

| Model | Mode / padding | Geometric area ratio | Actual tensor area ratio | Full-frame fraction | Candidates/frame | Segments after merge/frame | NN calls/frame |
|---|---|---|---|---|---|---|---|
| YOLO26n | FF | 1.0000 | 1.0000 | 1.0000 | 0.000 | 0.000 | 1.000 |
| YOLO26n | SEG p=32 | 1.5631 | 1.5604 | 0.2031 | 8.375 | 6.922 | 7.125 |
| YOLO26n | SEG p=64 | 2.2252 | 2.2231 | 0.1938 | 6.400 | 5.453 | 5.647 |
| YOLO26n | SEG p=96 | 2.7440 | 2.7428 | 0.1938 | 5.434 | 4.728 | 4.922 |
| ROI-SSD-MobileNet | FF | 1.0000 | 1.0000 | 1.0000 | 0.000 | 0.000 | 1.000 |
| ROI-SSD-MobileNet | SEG p=32 | 0.5505 | 0.5502 | 0.2906 | 0.988 | 0.881 | 1.172 |
| ROI-SSD-MobileNet | SEG p=64 | 0.7947 | 0.7943 | 0.3375 | 1.137 | 1.019 | 1.356 |
| ROI-SSD-MobileNet | SEG p=96 | 0.9751 | 0.9748 | 0.2906 | 1.206 | 1.075 | 1.366 |

Geometric area ratio is the mean per-frame sum of submitted source-region areas / source-frame area. Tensor area ratio is actual input pixels across all NN calls / full-frame input pixels. Overlap counts once per call, so ratios can exceed 1. YOLO's tensor ratio exceeds 1 at every SEG padding because multiple overlapping/aligned crops are processed independently. MobileNet reduces tensor area most at p=32, but the call count and per-call overhead prevent a measured speedup.

| Model | Mode / padding | NN calls total | Tensor pixels total | No-segments FF | Periodic FF | Forced FF |
|---|---|---|---|---|---|---|
| YOLO26n | FF | 320 | 32768000 | 0 | 0 | 320 |
| YOLO26n | SEG p=32 | 2280 | 51130368 | 35 | 30 | 0 |
| YOLO26n | SEG p=64 | 1807 | 72847360 | 32 | 30 | 0 |
| YOLO26n | SEG p=96 | 1575 | 89877504 | 32 | 30 | 0 |
| ROI-SSD-MobileNet | FF | 320 | 32768000 | 0 | 0 | 320 |
| ROI-SSD-MobileNet | SEG p=32 | 375 | 18027991 | 64 | 29 | 0 |
| ROI-SSD-MobileNet | SEG p=64 | 434 | 26029031 | 83 | 25 | 0 |
| ROI-SSD-MobileNet | SEG p=96 | 437 | 31941064 | 66 | 27 | 0 |

Fallback categories are mutually exclusive in the recorded decision order: an empty segment list is recorded as no_segments before the periodic-keyframe check. Thus periodic counts can differ between settings even on the same source frames. A periodic full-frame fraction of exactly 10% is not guaranteed: short clips start at different source indices and each video begins without segments. No cost-driven full-frame selections were measured because FULL-TECH was not run.

## 7. Fresh CPU affine cost calibration

Model: T(A)=K+c*A milliseconds, where A is the actual network input height times width. Tau=K/c in pixels. Eleven shapes were profiled per model with two warmups followed by eight repetitions per shape (88 measured calls and 22 warmup calls per model). Coefficients are least-squares fits to per-shape median latency. Requested and actual shapes were recorded; these profiles use actual tensor dimensions, not raw source bbox area.

| Model | K ms | c ms/pixel | Tau pixels | R-squared | MAE ms | RMSE ms |
|---|---|---|---|---|---|---|
| YOLO26n | 7.978164 | 0.000128514906 | 62079.679 | 0.847748 | 1.121940 | 1.680749 |
| ROI-SSD-MobileNet | 8.058673 | 0.000149488848 | 53908.189 | 0.892607 | 1.286563 | 1.600178 |

Both fitted intercepts and slopes are positive. Fit quality is imperfect (R-squared approximately 0.85/0.89); these are single-series pilot estimates without an independent fit-validation run. Coefficients remain preliminary because the downstream pipeline acceptance gate failed.

| Actual H x W | YOLO median ms | YOLO p95 ms | MobileNet median ms | MobileNet p95 ms |
|---|---|---|---|---|
| 32 x 32 | 5.959 | 8.322 | 6.103 | 7.201 |
| 64 x 64 | 12.910 | 52.462 | 7.019 | 7.877 |
| 64 x 128 | 8.232 | 8.671 | 9.824 | 11.085 |
| 96 x 96 | 9.327 | 10.668 | 9.008 | 10.642 |
| 128 x 64 | 6.922 | 7.836 | 9.647 | 11.163 |
| 128 x 128 | 9.713 | 9.927 | 12.220 | 12.663 |
| 160 x 320 | 14.684 | 16.021 | 19.284 | 21.432 |
| 192 x 192 | 12.723 | 17.951 | 13.795 | 15.583 |
| 256 x 256 | 16.863 | 19.184 | 16.556 | 19.056 |
| 320 x 160 | 15.564 | 19.914 | 16.375 | 17.499 |
| 320 x 320 | 20.396 | 26.501 | 21.779 | 24.622 |

## 8. Correctness, completeness and stage status

The 20-frame smoke per model exercised FF and SEG. It passed model loading, class ordering, detector CPU execution, crop-local to global remapping, and bounds checks. Forty overlays with companion coordinate records were exported, 20 per model. One MobileNet overlay was visually inspected with no obvious systematic box offset; the report does not claim exhaustive visual inspection of every frame.

| Model | Smoke FF mAP50 | Smoke SEG mAP50 | Smoke FF Recall50 | Smoke SEG Recall50 |
|---|---|---|---|---|
| YOLO26n | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| ROI-SSD-MobileNet | 1.000000 | 0.979287 | 1.000000 | 1.000000 |

These smoke metrics cover only the first 20 frames and were not used to select padding. The full 320-frame sweep supersedes their accuracy impression.

The consolidated report verified eight complete summaries and 2,560 frame records. Every run contains exactly the same 320 unique (video_id, frame_index) pairs across 32 videos. Checkpoint hashes match the locked files; recorded NN call totals and tensor-pixel totals equal the corresponding sums from frame records. Every summary records CPU execution, eight PyTorch threads and one OpenCV thread. The three runner helper tests passed, including the regression that blocks either model when coverage fails despite accuracy passing.

| Stage / mode | Status | Interpretation |
|---|---|---|
| Current checkpoint real-image load/size smoke | PASS | Both selected SSD families were checked at selection time |
| Fresh Stage-2 structural regression | PASS | 114 CPU forwards, 19 boundary shapes/model, 3 crops/shape |
| Dense Stage-2 accuracy / historical sweep attribution | NOT VERIFIED | Saved 100-shape CSVs declare last.pt; selected best_map.pt weights differ |
| Pilot coordinate smoke | PASS | 20 frames/model; FF and SEG |
| FF baselines | COMPLETE | 320 frames/model |
| SEG p=32 | COMPLETE | 320 frames/model |
| SEG p=64 | COMPLETE | 320 frames/model |
| SEG p=96 | COMPLETE | 320 frames/model |
| CPU cost profiles | COMPLETE, PRELIMINARY | 11 actual shapes/model |
| CA: cost-aware merge | NOT RUN | Acceptance gate stopped execution |
| FULL-TECH: merge plus cost selector | NOT RUN | No automatic cost-selection performance claim |
| Final evaluation / Raspberry Pi | NOT RUN | Pilot acceptance not met |

At the time of the Stage-3 rerun, fresh Stage-2 diagnostics were not available. A subsequent fresh structural regression now passed on both locked checkpoints: 19 shapes per model, three canonical real-image crops per shape, 114 CPU forwards. Square sizes were 95/96/97, 127/128/129, 139/140/141, 255/256/257 and 267/268/269; rectangular inputs were 128x160, 160x128, 256x320 and 320x256. Observed feature-block calls and active heads agree with until_1x1, and localization/classification/anchor counts and independent anchor geometry agree. Spatially infeasible next blocks are explicitly distinguished from first-1x1 stops. All 37 diagnostic/runtime tests passed. This structural PASS does not establish accuracy across those shapes or waive the MobileNet pilot gate. The saved 100-shape CSVs for both new training runs declare last.pt, contain no execution-time checkpoint hash, and cannot be attributed to best_map.pt: current last.pt and selected best_map.pt tensor hashes differ in both families. Thus a full fresh accuracy sweep remains unverified, and historical metrics have not been relabeled.

## 9. Reproduction settings and commands

Recorded execution environment: Windows-11-10.0.22000-SP0; processor Intel64 Family 6 Model 167 Stepping 1, GenuineIntel; logical CPU count 16. The pilot targets the Core i7-11700K environment. Recorded Python: 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)].

Dependency versions inspected while consolidating this report: torch 2.12.0+cu126, torchvision 0.27.0+cu126, ultralytics 8.4.60, numpy 1.26.4, opencv-python 4.10.0.84, PyYAML 6.0.1. These are the current local environment versions; no separately frozen package lock or Raspberry Pi compatibility check is claimed.

Recorded repository commit: `c5cdb772c9f576b4f0f3313cac4a4d8627ba9bc9`; working tree dirty: True. This commit alone does not reproduce uncommitted source changes. Configuration and checkpoint hashes above identify the evaluated model artifacts.

Run from the repository root using the project environment:

```powershell
# Smoke: FF and SEG for each model
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage smoke --max-frames 20

# Runs FF plus p=32, p=64 and p=96 for BOTH models; no padding-specific argument needed
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage padding

# Fresh CPU shape calibration
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage cost --cost-repetitions 8

# Attempts FF / SEG / CA / FULL-TECH; currently stops at acceptance gate
.venv\Scripts\python.exe -m experiments.cpu_pilot.run_pilot --stage pipeline
```

The runner reads the prepared YAML snapshots and manifest files under `final_experiments/pilot/`; it does not accept `--config` or an individual padding argument. Preparation has already been performed. Archive a completed session before regenerating it so old and new results cannot be mixed. Measured settings are static previous localization, hold=0, full-frame interval=10, confidence 0.01/0.5, NMS=0.45, 200 detections, base size=320, ROI alignment=32/1, and merge disabled for the sweep.

For audit/reproduction, raw files are named `padding_sweep.csv`, `frame_level_metrics.csv`, `padding_decision.json`, `cost_profiles_cpu.csv`, `cost_profiles_cpu.json`, `smoke_metadata.json`, `split_metadata.json` and `metadata.json`. Debug artifacts are in `debug_frames/`. Stage logs are `padding_execution.log`, `cost_execution.log` and `pipeline_execution.log`. All are under `final_experiments/pilot/`. No external report is required to interpret this document.

## 10. Interpretation and next work

MobileNet p=32 and p=64 are complete measured runs, not omitted settings. Their failures show that reducing padding does not resolve the current acceptance problem. The p=96 setting gives the best tested MobileNet accuracy but still falls short of both thresholds and is slower than FF. YOLO passes the quality gate at p=64, but SEG also remains slower than FF with merge disabled.

Before claiming the full technology is validated, investigate MobileNet's remaining segment quality/coverage loss with the locked checkpoint, resolve the remaining sweep-to-checkpoint provenance or perform an explicitly hash-bound accuracy evaluation, and declare any revised experiment before measuring it. The p=64 accuracy dip is a useful diagnostic target, but no cause is established here. Do not relax acceptance thresholds, add a padding post hoc as if it were predeclared, or reuse earlier-checkpoint cost profiles. Only after a valid acceptance path is established should CA/FULL-TECH and final hardware evaluation proceed.

