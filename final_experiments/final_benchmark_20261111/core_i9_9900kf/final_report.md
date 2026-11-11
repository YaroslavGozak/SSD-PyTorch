# Final end-to-end benchmark

## 1. Dataset and compatibility
- Final-evaluation manifest: `final_experiments\pilot_20261110\final_eval_manifest_candidate.txt`
- SHA-256: `4f36f3455ced1a344ef69329479cd09013934d71aafa8dc800d7f7e5ea6bf6d4`; 4,744 ordered frames from 476 videos.
- Dataset YAML: `D:\Repositories\Other\imagenet_vid_yolo_sampled\imagenet_vid.yaml` (SHA-256 `5496fbf035ffc1ab5ce59b4799a31b375e17ddd4f6433febb1ba7ba6954878c2`).
- Stage 3 pilot frames are checked for video-level disjointness from final frames.
- The source validation split was previously used for model selection/diagnostics; it is not an untouched external test set.
- WARNING: the current source audit-manifest checksum differs from the checksum recorded in the Stage 1 manifest; the frozen Stage 3 candidate checksum is verified, and every candidate row was cross-checked against current source content.

- Stage 3 reported `PIPELINE_ERROR` before CA/FULL-TECH completion; this runner executes the frozen modes and records any failures without replacing them with placeholder results.

## 2. Selected models
- YOLO26n and ROI-SSD-MobileNet use the Stage 3 selected checkpoints and config snapshots.

## 3. Fixed pipeline configuration
- Static/previous localization; hold=0; keyframe interval=10; empty segments fall back to full frame.
- Frozen Stage 3 padding: YOLO26n=64, ROI-SSD-MobileNet=128.
- Evaluation confidence=0.01; operational confidence is read from the hash-validated Stage 3 calibration selections.
- Stage 3 operational confidence (yolo26n): 0.1 (selection SHA-256 `4961610e1a9e68ae89f56a09d2736ada417804ac6a3e4c572b2b83cb0e56d468`).
- Stage 3 operational confidence (roi_ssd_mobilenet): 0.5 (selection SHA-256 `8b5e3ca4905a4872bd09b71bd977a40ae5714d2e24cb92ef4f73d7c022336b30`).
- No final-set parameter tuning. Online/adaptive tau is disabled.

## 4. Device-specific cost calibration
- Each profile is freshly measured on this device using the separate Stage 3 pilot manifest and actual observed network tensor shapes.
- The affine rule is T(A)=K+cA, with tau=K/c; coefficients remain frozen for all final runs.

| Model | Device | Operational conf | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) | Shapes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| yolo26n | core_i9_9900kf | 0.1000 | 7.05588 | 0.000145045001 | 48646.110 | 0.9861 | 0.4496 | 0.5316 | 11 |
| roi_ssd_mobilenet | core_i9_9900kf | 0.5000 | 9.70358 | 0.000167565086 | 57909.313 | 0.9324 | 1.1721 | 1.3919 | 11 |

## 5. Full-frame baselines

| Model | Device | Mode | FPS | mAP50 | Recall50 |
| --- | --- | --- | --- | --- | --- |
| yolo26n | core_i9_9900kf | FF | 42.8885 | 0.8311 | 0.8726 |
| yolo26n | core_i9_9900kf | SEG | 27.6429 | 0.7911 | 0.8609 |
| yolo26n | core_i9_9900kf | CA | 48.5023 | 0.8200 | 0.8625 |
| yolo26n | core_i9_9900kf | FULL-TECH | 46.7319 | 0.8200 | 0.8625 |
| roi_ssd_mobilenet | core_i9_9900kf | FF | 42.8270 | 0.8349 | 0.8924 |
| roi_ssd_mobilenet | core_i9_9900kf | SEG | 30.2078 | 0.8133 | 0.8902 |
| roi_ssd_mobilenet | core_i9_9900kf | CA | 41.5691 | 0.8246 | 0.8839 |
| roi_ssd_mobilenet | core_i9_9900kf | FULL-TECH | 41.6392 | 0.8246 | 0.8839 |

## 6. Segment formation only

| Model | Device | Mode | FPS | Speedup vs FF | ΔmAP50 | Tensor ratio |
| --- | --- | --- | --- | --- | --- | --- |
| yolo26n | core_i9_9900kf | FF | 42.8885 | 1.0188 | 0.0000 | 1.0000 |
| yolo26n | core_i9_9900kf | SEG | 27.6429 | 0.6567 | -0.0400 | 1.0748 |
| yolo26n | core_i9_9900kf | CA | 48.5023 | 1.1522 | -0.0111 | 0.7789 |
| yolo26n | core_i9_9900kf | FULL-TECH | 46.7319 | 1.1101 | -0.0111 | 0.7789 |
| roi_ssd_mobilenet | core_i9_9900kf | FF | 42.8270 | 1.0370 | 0.0000 | 1.0000 |
| roi_ssd_mobilenet | core_i9_9900kf | SEG | 30.2078 | 0.7314 | -0.0216 | 1.1581 |
| roi_ssd_mobilenet | core_i9_9900kf | CA | 41.5691 | 1.0065 | -0.0103 | 0.9479 |
| roi_ssd_mobilenet | core_i9_9900kf | FULL-TECH | 41.6392 | 1.0082 | -0.0103 | 0.9479 |

## 7. Cost-aware merge

| Model | Device | Mode | FPS | Speedup vs FF | NN calls/frame | Mean latency ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo26n | core_i9_9900kf | FF | 42.8885 | 1.0188 | 1.0000 | 23.3204 |
| yolo26n | core_i9_9900kf | SEG | 27.6429 | 0.6567 | 2.0586 | 36.3260 |
| yolo26n | core_i9_9900kf | CA | 48.5023 | 1.1522 | 1.0000 | 20.6363 |
| yolo26n | core_i9_9900kf | FULL-TECH | 46.7319 | 1.1101 | 1.0000 | 21.4530 |
| roi_ssd_mobilenet | core_i9_9900kf | FF | 42.8270 | 1.0370 | 1.0000 | 23.3690 |
| roi_ssd_mobilenet | core_i9_9900kf | SEG | 30.2078 | 0.7314 | 1.3290 | 33.1142 |
| roi_ssd_mobilenet | core_i9_9900kf | CA | 41.5691 | 1.0065 | 1.0000 | 24.0572 |
| roi_ssd_mobilenet | core_i9_9900kf | FULL-TECH | 41.6392 | 1.0082 | 1.0000 | 24.0180 |

## 8. Full technology

| Model | Device | FF FPS | FULL-TECH FPS | Speedup | FF mAP50 | FULL-TECH mAP50 | ΔmAP50 | FF Recall | FULL-TECH Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roi_ssd_mobilenet | core_i9_9900kf | 42.8270 | 41.6392 | 0.9723 | 0.8349 | 0.8246 | -0.0103 | 0.8924 | 0.8839 |
| yolo26n | core_i9_9900kf | 42.8885 | 46.7319 | 1.0896 | 0.8311 | 0.8200 | -0.0111 | 0.8726 | 0.8625 |

## 9. Ablation: FF -> SEG -> CA -> FULL-TECH

### roi_ssd_mobilenet / core_i9_9900kf

- FF -> SEG: FPS 42.8270 -> 30.2078; mAP50 0.8349 -> 0.8133; tensor ratio 1.0000 -> 1.1581; NN calls/frame 1.000 -> 1.329.
- SEG -> CA: FPS 30.2078 -> 41.5691; mAP50 0.8133 -> 0.8246; tensor ratio 1.1581 -> 0.9479; NN calls/frame 1.329 -> 1.000.
- CA -> FULL-TECH: FPS 41.5691 -> 41.6392; mAP50 0.8246 -> 0.8246; tensor ratio 0.9479 -> 0.9479; NN calls/frame 1.000 -> 1.000.

### yolo26n / core_i9_9900kf

- FF -> SEG: FPS 42.8885 -> 27.6429; mAP50 0.8311 -> 0.7911; tensor ratio 1.0000 -> 1.0748; NN calls/frame 1.000 -> 2.059.
- SEG -> CA: FPS 27.6429 -> 48.5023; mAP50 0.7911 -> 0.8200; tensor ratio 1.0748 -> 0.7789; NN calls/frame 2.059 -> 1.000.
- CA -> FULL-TECH: FPS 48.5023 -> 46.7319; mAP50 0.8200 -> 0.8200; tensor ratio 0.7789 -> 0.7789; NN calls/frame 1.000 -> 1.000.

## 10. YOLO26n: Core i9-9900KF vs Raspberry Pi 5
- Cross-device comparison is pending runs on both target platforms.

## 11. ROI-SSD-MobileNet: Core i9-9900KF vs Raspberry Pi 5
- Cross-device comparison is pending runs on both target platforms.

## 12. Accuracy-speed trade-off
- See the numeric mode-by-mode table above; negative or neutral speed/accuracy changes are retained.

## 13. Negative or unexpected findings
- Inspect `timing_runs.csv`, `frame_level_metrics.csv`, and crash/error records. Thermal-throttling status and temperatures are recorded for every run.

## 14. Dissertation-ready conclusions
- The FULL-TECH vs FF numbers above are direct measurements for this model/device/data protocol only.
- Timing repeats: 3 per mode. Bootstrap confidence intervals were not computed; adjacent frames are not treated as independent samples.

## Timing and metric definitions
- mAP50: project area-integral AP at IoU 0.50; mAP50:95 is mean AP at ten IoU thresholds 0.50..0.95.
- FPS and frame latency use the Stage 3 process-frame timing boundary; model/data initialization and evaluation are excluded.
- Tensor ratio is actual network-input pixels relative to full-frame tensor pixels. Geometric area ratio is recorded per frame in the frame CSV.
- Segmentation formation time is represented by tracker-update/formation time; preprocessing, inference, merge, postprocess and total process time are separate.
- Timing summary reports mean ± sample standard deviation of run FPS and mean of run-level latency percentiles.
