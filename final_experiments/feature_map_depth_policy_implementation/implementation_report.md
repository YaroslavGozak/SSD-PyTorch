# Stage 2f: sequential until_1x1 implementation

## Implementation

- `RoiSSDMobileNet.forward` now computes a feature level, applies its classification and localization heads, then advances to the next block.
- The first 1x1 feature map is included with its heads and is the final level. No deeper block or head executes afterward.
- Some padded convolutions could mathematically map 1x1 to another 1x1; the runtime stops at the first 1x1 by policy. The trace records such a next map as physically feasible while marking the stop reason as `STOPPED_AT_1X1`.
- Runtime policy is fixed to `until_1x1`; legacy selection and the old threshold helper have been removed from the model.
- No trainable layers, tensor shapes, or state-dict keys were changed. The selected checkpoint loaded with `strict=True`.

## Validation

- Exact locked inputs and GT hashes verified: True.
- Until-1x1 Stage 2e accuracy and prediction parity at 1e-6: True / True (9216 per-sample outputs).
- Prediction/anchor slot checks passed: True.
- CPU timings verify sequential block/head execution using forward hooks; Prior legacy CPU median across 19 shapes: 12.717 ms; current until_1x1 median across 19 shapes: 13.769 ms.

## Accuracy comparison

| Reference | Macro mAP@0.5 | Macro Recall@0.5 |
|---|---:|---:|
| Stage 2e legacy reference | 0.325360 | 0.353919 |
| Sequential until_1x1 | 0.474089 | 0.505772 |

Legacy figures are frozen Stage 2e measurements; legacy inference is intentionally no longer supported by the runtime model. The until_1x1 results reproduce Stage 2e on the same checkpoint, subset, preprocessing, evaluator and per-sample predictions. This supports using the policy in the Stage 3 CPU pilot, subject to pilot-level validation.
