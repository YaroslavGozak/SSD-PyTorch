# Fixed ROI-SSD VGG16 regression

**Result: PASS**

Strict checkpoint compatibility, finite/bounded outputs, independent anchor geometry and prediction counts verified. Old and new run on the identical frozen 256-frame tensors/GT. Historical Stage 2 CSV metrics are replayed at tolerance 1e-6 where available. Full-frame comparison uses the same ROI checkpoint at confidence 0.01, resize 320x320, ImageNet normalization, project area-AP evaluator and NMS 0.45. Crop confidence remains 0.5.

| Mode/input H×W | Old depth | New depth | Old mAP50 | New mAP50 | ΔmAP50 | ΔRecall50 |
|---|---:|---:|---:|---:|---:|---:|
| crop 64×64 | 2 | 4 | 0.554677592 | 0.565453117 | +0.010775524 | +0.011309524 |
| crop 88×88 | 3 | 4 | 0.614792950 | 0.622100596 | +0.007307646 | +0.008968773 |
| crop 96×96 | 3 | 4 | 0.655548265 | 0.655548265 | +0.000000000 | +0.000000000 |
| crop 104×104 | 4 | 4 | 0.643408203 | 0.643408203 | +0.000000000 | +0.000000000 |
| crop 108×108 | 4 | 4 | 0.500871844 | 0.500871844 | +0.000000000 | +0.000000000 |
| crop 112×112 | 4 | 4 | 0.407621918 | 0.407621918 | +0.000000000 | +0.000000000 |
| crop 116×116 | 4 | 4 | 0.280059835 | 0.280059835 | +0.000000000 | +0.000000000 |
| crop 120×120 | 4 | 4 | 0.203174532 | 0.203174532 | +0.000000000 | +0.000000000 |
| crop 124×124 | 4 | 4 | 0.079417241 | 0.079417241 | +0.000000000 | +0.000000000 |
| crop 128×128 | 4 | 4 | 0.069360771 | 0.069360771 | +0.000000000 | +0.000000000 |
| crop 132×132 | 4 | 4 | 0.065259334 | 0.065259334 | +0.000000000 | +0.000000000 |
| crop 136×136 | 4 | 4 | 0.064039656 | 0.064039656 | +0.000000000 | +0.000000000 |
| crop 140×140 | 4 | 5 | 0.620585258 | 0.664639423 | +0.044054164 | +0.042258014 |
| crop 144×144 | 5 | 5 | 0.697251357 | 0.697251357 | +0.000000000 | +0.000000000 |
| crop 160×160 | 5 | 5 | 0.722413334 | 0.722413334 | +0.000000000 | +0.000000000 |
| crop 256×256 | 5 | 5 | 0.626590528 | 0.626590528 | +0.000000000 | +0.000000000 |
| crop 268×268 | 5 | 6 | 0.653410908 | 0.657917081 | +0.004506173 | +0.004629630 |
| crop 300×300 | 6 | 6 | 0.627670893 | 0.627670893 | +0.000000000 | +0.000000000 |
| crop 128×160 | 4 | 4 | 0.496787513 | 0.496787513 | +0.000000000 | +0.000000000 |
| crop 160×128 | 4 | 4 | 0.635102121 | 0.635102121 | +0.000000000 | +0.000000000 |
| crop 140×192 | 4 | 5 | 0.616920048 | 0.666356360 | +0.049436313 | +0.051043423 |
| crop 192×140 | 4 | 5 | 0.647206336 | 0.679841152 | +0.032634816 | +0.032745012 |
| full_frame 320×320 | 6 | 6 | 0.818881492 | 0.818881492 | +0.000000000 | +0.000000000 |

CPU latency: 4 threads, batch 1, 3 warmups + 10 repeats per shape/model; see latency_sanity.csv. Hook traces verify each active block runs once and no deeper block runs. This is a short sanity measurement, not a deployment benchmark. Results establish regression behavior on this locked validation subset, not all possible inputs.

## Additional checkpoint and runtime verification


Second checkpoint: `trained_models/imagenet-vid-roissd-20261002/best_map_stage2.pt`. Strict load and unchanged weights: PASS. All 256 full-frame prediction arrays match the old source within 1e-6. mAP50 0.850125701, Recall50 0.893262466; both deltas are zero. These paired results use the same checkpoint/protocol; the manifest historical canonical score additionally used its audit postprocessing and is not asserted to match bit for bit.

The known 108–136 px accuracy collapse is unchanged, not repaired by this forward fix. At 140 px the feasible fifth level improves mAP50 from 0.620585258 to 0.664639423. Heads retain their original parameters, scales, aspect ratios and class ordering.

The initial sequential CPU timing is retained in `latency_sanity.csv`. Interleaved timing below alternates model order, disables optional tracing for production behavior, and uses 5 warmups + 30 repeats. It reduces ordering and thermal bias; p95 still has short-run noise.

| Input | Old median ms | New median ms | New/old | Old p95 ms | New p95 ms |
|---:|---:|---:|---:|---:|---:|
| 96 | 32.257 | 29.801 | 0.924 | 39.324 | 40.992 |
| 128 | 39.091 | 38.377 | 0.982 | 55.079 | 52.407 |
| 140 | 51.189 | 50.571 | 0.988 | 57.908 | 63.786 |
| 256 | 131.024 | 131.666 | 1.005 | 156.449 | 155.071 |
| 300 | 192.889 | 195.769 | 1.015 | 210.388 | 219.828 |
