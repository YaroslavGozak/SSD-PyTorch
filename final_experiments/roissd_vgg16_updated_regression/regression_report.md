# Updated ROI-SSD VGG16 regression check

**Result: FAIL**

Checkpoint: `trained_models/best-imagenet-vid-roissd/imagenet26022026stage3.pt`.
SHA256: `e63200377a07c497c31d859b0a2dbbeabf6a1c1c7cafc0a7e795f70070d9a3fa`.

The existing checkpoint loads with `strict=True`, no missing/unexpected keys or shape mismatches. Every loaded tensor matches the checkpoint and remains unchanged after all attempted forwards.

All 27/27 attempted forwards fail before any feature block or detection head runs. `RoiSSD._iter_feature_levels()` calls `self.features_stage1(x)`, but VGG16 defines `self.features`. The new iterator also references MobileNet-only `conv_to_672` and `features_stage2`. The constructor does not initialize `record_execution_trace` or `last_execution_trace` either. These are source findings; the observed first exception is shown below.

```text
Traceback (most recent call last):
  File "H:\Projects\University\SSD-PyTorch\experiments\roissd_128_anomaly\vgg16_updated_regression.py", line 114, in main
    model(tensor, None)
    ~~~~~^^^^^^^^^^^^^^
  File "H:\Projects\University\SSD-PyTorch\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "H:\Projects\University\SSD-PyTorch\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "H:\Projects\University\SSD-PyTorch\model\roissd.py", line 752, in forward
    for i, features in self._iter_feature_levels(x):
                       ~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "H:\Projects\University\SSD-PyTorch\model\roissd.py", line 662, in _iter_feature_levels
    features_stage1 = self.features_stage1(x)
                      ^^^^^^^^^^^^^^^^^^^^
  File "H:\Projects\University\SSD-PyTorch\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1967, in __getattr__
    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'"
    )
AttributeError: 'RoiSSD' object has no attribute 'features_stage1'

```

All requested squares and rectangles were attempted, plus 108/112/116 for the unstable range and a real full-frame image resized to 320x320. The frozen 256-frame image and label hashes, checkpoint/config/subset hashes, seed, thresholds and shared preprocessing/evaluator source were verified against the saved Stage 2 provenance.

## Accuracy and unstable range

Exact historical CSV values are used below. New metrics and deltas are unavailable, not zero: the first forward fails independently of image contents. The 256-frame accuracy sweep and CPU latency repetitions cannot produce meaningful results. No claim of preserved full-frame accuracy, anchor alignment, finite predictions, early exit, or reduced 108–136 collapse can be made.

| Input | Old depth | New depth | Old mAP50 | New mAP50 | ΔmAP50 | Final feature shape |
|---:|---:|---|---:|---|---|---|
| 108 | 4 | unavailable | 0.500871844 | unavailable | unavailable | unavailable |
| 112 | 4 | unavailable | 0.407621918 | unavailable | unavailable | unavailable |
| 116 | 4 | unavailable | 0.280059835 | unavailable | unavailable | unavailable |
| 120 | 4 | unavailable | 0.203174532 | unavailable | unavailable | unavailable |
| 124 | 4 | unavailable | 0.079417241 | unavailable | unavailable | unavailable |
| 128 | 4 | unavailable | 0.069360771 | unavailable | unavailable | unavailable |
| 132 | 4 | unavailable | 0.065259334 | unavailable | unavailable | unavailable |
| 136 | 4 | unavailable | 0.064039656 | unavailable | unavailable | unavailable |
| 140 | 4 | unavailable | 0.620585258 | unavailable | unavailable | unavailable |
| 144 | 5 | unavailable | 0.697251357 | unavailable | unavailable | unavailable |

## Decision

Checkpoint compatibility: PASS. Structural/inference checks: FAIL. Accuracy, full-frame non-regression and latency: NOT EVALUABLE. Overall regression-safe decision: **FAIL**. The model was not modified, retrained or retuned. Correcting the forward is outside this task. Rerun complete accuracy and timing checks after a separately authorized fix.

Reproduce: `.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.vgg16_updated_regression`. This failure-diagnostic runner does not award PASS if the forward is subsequently repaired.
