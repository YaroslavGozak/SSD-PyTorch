# Final end-to-end benchmark

## 1. Dataset and compatibility
- Final-evaluation manifest: `final_experiments\pilot\final_eval_manifest_candidate.txt`
- SHA-256: `4f36f3455ced1a344ef69329479cd09013934d71aafa8dc800d7f7e5ea6bf6d4`; 4,744 ordered frames from 476 videos.
- Dataset YAMLs by device: `{"core_i9_9900kf": {"path": "D:\\Repositories\\Other\\imagenet_vid_yolo_sampled\\imagenet_vid.yaml", "sha256": "5496fbf035ffc1ab5ce59b4799a31b375e17ddd4f6433febb1ba7ba6954878c2"}}`.
- Stage 3 pilot frames are checked for video-level disjointness from final frames.
- The source validation split was previously used for model selection/diagnostics; it is not an untouched external test set.
- WARNING: the current source audit-manifest checksum differs from the checksum recorded in the Stage 1 manifest; the frozen Stage 3 candidate checksum is verified, and every candidate row was cross-checked against current source content.

- Stage 3 reported `PIPELINE_ERROR` before CA/FULL-TECH completion; this runner executes the frozen modes and records any failures without replacing them with placeholder results.

## 2. Selected models
- YOLO26n and ROI-SSD-MobileNet use the Stage 3 selected checkpoints and config snapshots.

## 3. Fixed pipeline configuration
- Static/previous localization; hold=0; keyframe interval=10; empty segments fall back to full frame.
- Frozen Stage 3 padding: YOLO26n=64, ROI-SSD-MobileNet=128.
- Evaluation confidence=0.01; operational confidence comes from the frozen Stage 3 configs.
- No final-set parameter tuning. Online/adaptive tau is disabled.

## 4. Device-specific cost calibration
- Each profile is freshly measured on this device using the separate Stage 3 pilot manifest and actual observed network tensor shapes.
- The affine rule is T(A)=K+cA, with tau=K/c; coefficients remain frozen for all final runs.

| Model | Device | K (ms) | c (ms/pixel) | tau (pixels) | R² | MAE (ms) | RMSE (ms) | Shapes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| yolo26n | core_i9_9900kf | 7.0812 | 0.000132603365 | 53401.324 | 0.9876 | 0.3739 | 0.4590 | 11 |
| roi_ssd_mobilenet | core_i9_9900kf | 10.3855 | 0.000150976438 | 68789.147 | 0.8681 | 1.4401 | 1.8164 | 11 |

## 5. Full-frame baselines

| Model | Device | Mode | FPS | mAP50 | Recall50 |
| --- | --- | --- | --- | --- | --- |
| yolo26n | core_i9_9900kf | FF | 41.7439 | 0.8311 | 0.8726 |
| yolo26n | core_i9_9900kf | SEG | 9.5624 | 0.7636 | 0.8799 |
| yolo26n | core_i9_9900kf | CA | 42.0499 | 0.8258 | 0.8711 |
| yolo26n | core_i9_9900kf | FULL-TECH | 38.9773 | 0.8258 | 0.8711 |
| roi_ssd_mobilenet | core_i9_9900kf | FF | 44.6192 | 0.8349 | 0.8924 |
| roi_ssd_mobilenet | core_i9_9900kf | SEG | 29.3108 | 0.8133 | 0.8902 |
| roi_ssd_mobilenet | core_i9_9900kf | CA | 41.2653 | 0.8246 | 0.8839 |
| roi_ssd_mobilenet | core_i9_9900kf | FULL-TECH | 41.8638 | 0.8246 | 0.8839 |

## 6. Segment formation only

| Model | Device | Mode | FPS | Speedup vs FF | ΔmAP50 | Tensor ratio |
| --- | --- | --- | --- | --- | --- | --- |
| yolo26n | core_i9_9900kf | FF | 41.7439 | 0.9633 | 0.0000 | 1.0000 |
| yolo26n | core_i9_9900kf | SEG | 9.5624 | 0.2207 | -0.0675 | 2.5907 |
| yolo26n | core_i9_9900kf | CA | 42.0499 | 0.9703 | -0.0053 | 0.8080 |
| yolo26n | core_i9_9900kf | FULL-TECH | 38.9773 | 0.8994 | -0.0053 | 0.8080 |
| roi_ssd_mobilenet | core_i9_9900kf | FF | 44.6192 | 1.0224 | 0.0000 | 1.0000 |
| roi_ssd_mobilenet | core_i9_9900kf | SEG | 29.3108 | 0.6716 | -0.0216 | 1.1581 |
| roi_ssd_mobilenet | core_i9_9900kf | CA | 41.2653 | 0.9456 | -0.0103 | 0.9479 |
| roi_ssd_mobilenet | core_i9_9900kf | FULL-TECH | 41.8638 | 0.9593 | -0.0103 | 0.9479 |

## 7. Cost-aware merge

| Model | Device | Mode | FPS | Speedup vs FF | NN calls/frame | Mean latency ms |
| --- | --- | --- | --- | --- | --- | --- |
| yolo26n | core_i9_9900kf | FF | 41.7439 | 0.9633 | 1.0000 | 23.9746 |
| yolo26n | core_i9_9900kf | SEG | 9.5624 | 0.2207 | 6.9269 | 104.7732 |
| yolo26n | core_i9_9900kf | CA | 42.0499 | 0.9703 | 1.0004 | 23.8058 |
| yolo26n | core_i9_9900kf | FULL-TECH | 38.9773 | 0.8994 | 1.0004 | 25.6793 |
| roi_ssd_mobilenet | core_i9_9900kf | FF | 44.6192 | 1.0224 | 1.0000 | 22.4175 |
| roi_ssd_mobilenet | core_i9_9900kf | SEG | 29.3108 | 0.6716 | 1.3290 | 34.3113 |
| roi_ssd_mobilenet | core_i9_9900kf | CA | 41.2653 | 0.9456 | 1.0000 | 24.2876 |
| roi_ssd_mobilenet | core_i9_9900kf | FULL-TECH | 41.8638 | 0.9593 | 1.0000 | 23.8994 |

## 8. Full technology

| Model | Device | FF FPS | FULL-TECH FPS | Speedup | FF mAP50 | FULL-TECH mAP50 | ΔmAP50 | FF Recall | FULL-TECH Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roi_ssd_mobilenet | core_i9_9900kf | 44.61917524936175 | 41.8638016233326 | 0.9382468723227113 | 0.8348805815130833 | 0.824588117025724 | -0.01029246448735932 | 0.8923727685049293 | 0.8839458475307269 |
| yolo26n | core_i9_9900kf | 41.743891160857444 | 38.97733054130332 | 0.9337253777111156 | 0.8311271947505224 | 0.825817251330423 | -0.005309943420099383 | 0.8725802109393567 | 0.8710896835414684 |

## 9. Ablation: FF -> SEG -> CA -> FULL-TECH

### roi_ssd_mobilenet / core_i9_9900kf

- FF -> SEG: FPS 44.6192 -> 29.3108; mAP50 0.8349 -> 0.8133; tensor ratio 1.0000 -> 1.1581; NN calls/frame 1.000 -> 1.329.
- SEG -> CA: FPS 29.3108 -> 41.2653; mAP50 0.8133 -> 0.8246; tensor ratio 1.1581 -> 0.9479; NN calls/frame 1.329 -> 1.000.
- CA -> FULL-TECH: FPS 41.2653 -> 41.8638; mAP50 0.8246 -> 0.8246; tensor ratio 0.9479 -> 0.9479; NN calls/frame 1.000 -> 1.000.

### yolo26n / core_i9_9900kf

- FF -> SEG: FPS 41.7439 -> 9.5624; mAP50 0.8311 -> 0.7636; tensor ratio 1.0000 -> 2.5907; NN calls/frame 1.000 -> 6.927.
- SEG -> CA: FPS 9.5624 -> 42.0499; mAP50 0.7636 -> 0.8258; tensor ratio 2.5907 -> 0.8080; NN calls/frame 6.927 -> 1.000.
- CA -> FULL-TECH: FPS 42.0499 -> 38.9773; mAP50 0.8258 -> 0.8258; tensor ratio 0.8080 -> 0.8080; NN calls/frame 1.000 -> 1.000.

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
