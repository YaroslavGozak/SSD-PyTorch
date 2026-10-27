# ROI-SSD-MobileNet quick regression check

**Result: PASS**

- Existing checkpoint loaded with `strict=True`; key sets and tensor shapes match.
- Locked Stage 2e inputs and GT were reproduced for the 10 requested shapes.
- All per-sample predictions, detections, selected depths, feature maps, anchors,
  mAP@0.5 and Recall@0.5 match Stage 2e; metric tolerance is 1e-6.
- Each checked forward ends at its first 1x1 map; execution hooks verify no
  later block/head runs and verify `N_loc == N_cls == N_anchors`.
- CPU batch-1 timing and early-exit checks passed on four requested shapes.

## Stage 2e comparison

| Input HxW | L_R | mAP@0.5 | Recall@0.5 | Detections | Anchors | Final map |
|---|---:|---:|---:|---:|---:|---|
| 96x96 | 4 | 0.492519 | 0.506523 | 215 | 228 | [1, 1] |
| 128x128 | 4 | 0.427359 | 0.455115 | 271 | 382 | [1, 1] |
| 128x160 | 5 | 0.328956 | 0.374639 | 237 | 492 | [1, 1] |
| 129x129 | 5 | 0.584698 | 0.598297 | 253 | 556 | [1, 1] |
| 140x140 | 5 | 0.603422 | 0.622562 | 255 | 556 | [1, 1] |
| 144x144 | 5 | 0.596018 | 0.615206 | 254 | 556 | [1, 1] |
| 256x256 | 5 | 0.603260 | 0.618318 | 257 | 1532 | [1, 1] |
| 256x320 | 6 | 0.610493 | 0.634962 | 277 | 1928 | [1, 1] |
| 257x257 | 6 | 0.496167 | 0.527521 | 240 | 1866 | [1, 1] |
| 268x268 | 6 | 0.592639 | 0.614231 | 267 | 1866 | [1, 1] |

## CPU latency sanity check

| Input HxW | L_R | Median (ms) | p95 (ms) | Executed blocks |
|---|---:|---:|---:|---|
| 128x128 | 4 | 11.716 | 13.110 | ['features_stage1', 'conv_to_672', 'features_stage2', 'conv8_2', 'conv9_2'] |
| 140x140 | 5 | 13.204 | 15.127 | ['features_stage1', 'conv_to_672', 'features_stage2', 'conv8_2', 'conv9_2', 'conv10_2'] |
| 256x256 | 5 | 18.747 | 20.503 | ['features_stage1', 'conv_to_672', 'features_stage2', 'conv8_2', 'conv9_2', 'conv10_2'] |
| 268x268 | 6 | 22.585 | 25.903 | ['features_stage1', 'conv_to_672', 'features_stage2', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2'] |

Checkpoint compatibility, inference parity, structural checks,
and runtime sanity support proceeding to the Stage 3 CPU pilot.
