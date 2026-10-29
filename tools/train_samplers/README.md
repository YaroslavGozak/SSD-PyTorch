# Deployment-matched mixed sampling

`train_mixed_sampling.py` uses the same deployment implementation for Stage 2
and Stage 3. Set `train_params.roi_mixed_stage` and supply
`train_params.roi_sampling.stage2` / `stage3` as in
`config/imagenet-vid-roissd-mobilenet.yaml`. Missing stage config fails before
checkpoint side effects. Stage 1 continues to use the separate full-frame script.

Each batch has one mode (`full` / `roi`) and one integer H×W. ROI area is
sampled continuously within a YAML bin; W/H is log-uniform over the feasible
intersection with aspect/minimum-side/frame constraints. H/W are not aligned
to a stride. A request carries a deterministic per-sample seed derived from
seed/stage/epoch/batch/slot. Call `sampler.set_epoch(epoch)` before iteration;
the training entry point already does this on resume as well.

The request-aware dataset resamples unsuitable raw items deterministically.
Reference GT must meet its visibility threshold. Both stages allow 30 attempts.
If item attempts are exhausted, both crop dimensions increase once by 10 pixels
(capped at the deployment frame size). The entire batch retries at that shape,
with up to 30 item attempts per slot. If an enlarged item still fails, an error
is logged and training skips the batch. Crops are never stretched or padded.
Metadata records `shape_enlarged` and the requested and effective dimensions. Eligible reference selection changes image/GT sampling probabilities;
resampling counts and selected source indices are logged.

Raw RGB is resized once to the deployment full-frame square. The processor
recovers uint8 from the existing raw datasets' float/255 decoding to match
deployment resize rounding, then takes an unscaled rectangular slice and
normalizes it. Targets clip to the crop; x coordinates divide by W and y by H.
Collate only stacks identical shapes and rejects inconsistent definitions.

Per-epoch batch/sample CSVs and a summary JSON log full/ROI shares, conditional
and absolute area-bin shares, H/W/aspect distributions, geometry and resampling.
`reference_gt_area_ratio` is the visible reference target area / crop area;
`reference_original_gt_area_ratio` retains its pre-clipping area. Reference
center offset is measured from the original reference center in crop units.
Full batches have no selected reference, so reference geometry fields are null.

Sanity checks (no training loop or optimizer steps):

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p '*mixed_sampling*.py' -v
.venv/Scripts/python.exe -m experiments.cpu_pilot.deployment_sampling_check
```

Artifacts, YAML diff, real B=2 forward/backward checks and per-module BN audit:
`final_experiments/deployment_mixed_sampling_check/implementation_report.md`.
Legacy `roi_mixed_sampling.py` remains for historical experiments and previews.

## Fixed-size accuracy evaluation

`python -m tools.infer --config <evaluation.yaml> --infer-samples false --eval-mode fixed-size-loop --summary-path <new-output-directory>`

The fixed-size sweep now resizes the full source frame once to `dataset_params.im_size`
(the deployment square), then extracts each requested H/W without object-dependent
rescaling, stretching, padding, jitter, or enlargement. Each frame uses the first
annotation whose centered, boundary-clamped crop meets the reference visibility
threshold. Other boxes are clipped and filtered by visible area fraction. The
active `roi_mixed_stage` configuration supplies reference/other visibility thresholds;
if unavailable, defaults are 0.9/0.3. Crop dimensions must not exceed the deployment
frame. Both SSD and YOLO variants use this geometry; only normalization differs.

Images without an eligible reference (including unannotated images) are excluded
before inference. `fixed_size_results_v2.csv` contains conditional accuracy and
`fixed_size_coverage_v2.csv` records eligible/ineligible/total image counts and coverage.
Always inspect coverage: different sizes can evaluate different images/objects.
Zero-eligible settings have undefined (NaN) accuracy. The crops use ground truth,
so this evaluates crop detection rather than the deployment ROI proposal pipeline.
Use a new output directory when comparing against the former resize-to-fit protocol;
resume requires a matching new-protocol coverage record.
