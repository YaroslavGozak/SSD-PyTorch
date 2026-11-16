# Protocol: Experiment B v1

## Question and scope

How accurately do hardware/detector-specific latency estimators decide whether
fixed image regions should be processed separately or as enclosing rectangles?
Primary pair results use shape-conditioned prepared-forward microbenchmarks.
The second primary analysis actually replays the chosen partitions on fixed
image content. These are separate estimands. No tracking recovery, real-time
input scheduling, cross-ROI quality/NMS or detection mAP is claimed.

All deployable policies see calibration artifacts and fixed geometry only.
Ground truth generates proposals, not online predictions. Adding object context
by merging can change detections even with equal nominal coverage. Detection
quality is optional and is not implemented in this version; report it as absent.

## Immutable workload and splits

Read canonical audit IDs/order, relative image/label paths and YOLO annotations.
Optionally intersect the exact frozen final TSV. Never guess the final cohort
from an approximate frame count. Decode every image, validate class IDs/YOLO
coordinates/annotation hashes, record W/H and SHA-256 of compressed image bytes.
The dataset content digest hashes ordered relative names and content hashes.
This is a full byte hash policy, not a metadata-only or random sample check.

Sort by video ID and frame index; assign whole sequences to calibration,
validation and test using a seeded permutation of sorted IDs. Defaults reserve
32 sequences each for calibration and validation; all remaining sequences form
test. Source counts, split counts and candidate multiplicities are reported.
Retained sampled frame indices do not imply continuous real-time video.

Proposals: normalized GT boxes expanded by 16 pixels per side in a conceptual
320x320 canvas, minimum canvas side 32. Map that padding into original pixels,
move/clamp windows inside the original image, deduplicate exact rectangles and
retain source annotation IDs. Store exclusive xyxy original-pixel boxes.
Candidates are canonically sorted before assigning local ROI IDs. No policy or
hardware timings influence generation. No stress-set oversampling is pooled
with the natural workload. Near-boundary test strata are descriptive only.

Empty candidate sets are retained; actual replay uses one full-frame call.
Pair metrics naturally contain only frames with at least two proposals.
Frame summaries explicitly separate all frames and the merge-eligible subset.
Singletons and zero-ROI frames remain visible in counts. A seeded frozen test
selection defines smoke/pilot/full cohorts; explicit limits are identified by
frame IDs and cohort digest, never described as the full benchmark.

## Executed tensors and checkpoints

Use the canonical Stage 1 SHA identities and architecture parameters in the
versioned YAML. Strict state-dict loading is required. MobileNet's optional
`pretrained_backbone=False` constructor path skips an unnecessary pretrained
backbone download because the complete selected checkpoint overwrites every
parameter; the legacy default stays True. This does not change architecture or
checkpoint contents. YOLO requires Ultralytics 8.4.60, verifies names/order
against the frozen dataset mapping, and runs raw `measurek.run_forward` via the
existing profiling adapter. It is not the Stage 3/4 predict-API wrapper.

For actual-content replay, use the existing `SsdTransform` test square resize
and ImageNet normalization. YOLO reverses that normalization to [0,1]. Reuse
`pipeline.convert_crop_to_input_tensor` both when freezing effective dimensions
and executing crops: SSD grid 1; YOLO grid 32 expands/shifts a crop window inside
the 320x320 base tensor. Crops are never resized back to 320x320. Both models
receive the identical original rectangles for each policy; model-specific
snapping is explicit. No assumed multiple-of-32 constraint is applied to SSD.
A forward prehook verifies exactly one actual network invocation and its shape.
SSD traces expose `until_1x1`, active heads, and feature feasibility. A failing
shape blocks the run; no fabricated shape cost or silent skipped frame.

Prepared-forward scope: pre-created CPU float32 tensor, batch 1, gradients off.
YOLO includes its raw detector forward (no external NMS or prediction API);
SSD includes its own decoder/NMS inside forward. Bookkeeping/hook checks are
inside the same timing wrapper. Crop/resize/base-image decoding are excluded
from shape costs. Compare policy savings within one detector/platform; these
boundaries are not an assertion of equivalent postprocessing across detectors.
Actual serial frame replay separately records crop overhead, per-call times,
wall time, pixels, groups and planning overhead. Base-image decoding/resize is
outside this replay boundary and not included in reported FPS-equivalent.

## Calibration, validation and test isolation

Per detector/host, choose calibration entries from a predeclared square/non-square
side grid plus original/all-subset-union shapes in calibration sequences only.
The configured budget caps the table. Test shapes never choose training entries.
Validation shapes come from validation sequences, measured in independent blocks
after fitting, without refitting. The same shape may occur in different splits;
timed trials and RNG inputs remain independent. This is held-out measurement
validation, not a claim of disjoint shape supports.

Synthetic seeded RGB content is prepared by the existing detector adapter.
Calibration means fit the affine and piecewise estimators. Lookup stores mean,
median, repetition count and mean SE. The affine fit reuses repository code and
requires positive intercept/slope. A continuous hinge fit selects a breakpoint
by calibration RSS with positive branch slopes and intercept enforced by bounded
least squares. The unconstrained repository fit is retained as a diagnostic;
no breakpoint, margin, geometric threshold or uncertainty constant is selected
on test latency. Invalid fits block, prompting a separate calibration pilot.

Each timing stage uses balanced randomized shape order per repetition, warmups
per shape/block and thermal/frequency snapshots. Full-canvas grid entries serve
as repeat controls when present. Fixed CPU affinity is optional but recorded.
Intra/inter/OpenCV and BLAS limits are explicit. Temperature/frequency are logged
rather than silently corrected. A separate device-appropriate thread pilot is
needed before freezing settings (Pi is not required to use desktop threads).

Calibration identity binds source hashes, Git, checkpoint/config/manifest,
host/CPU/software, observed threads/affinity and governor/power metadata. A cost
artifact from another host or settings cannot be reused. Raw trials are atomic
per shape/block; resumes require identical identities. Frozen completed cost
artifacts are immutable and receipt-checked. Fresh directories are needed for
new calibration sessions/settings.

## Policies and ties

* `no_merge`: retain all source groups.
* `geometric`: faithful `simple_roi_merge_v2` sequential greedy baseline with
  gamma=1.4: bounding-box area / (current area + candidate area) <= gamma. It can
  merge disjoint boxes. It is not the buggy IoU=0 default in `simple.py`.
* `affine`: T=K+cA, positive K,c. Choose the maximum strictly positive predicted
  gain above the configured minimum; for zero margin this is delta area < K/c.
* `piecewise`: same pair enumeration, strict gain and stable tie rule using the
  continuous constrained hinge predictor.
* `lookup`: exact entry if available, otherwise deterministic Euclidean nearest
  neighbor in ordered effective (H,W), ties by canonical shape key. Extrapolation
  is allowed and flagged as not exact. No test timing enters this table.
* `conservative_lookup`: only exact measured calibration entries; absent shape
  or absent uncertainty separates. Merge iff gain - z*(SE_left+SE_right+SE_union)
  > predeclared margin, using the worst-case correlation bound. Defaults z=1.645,
  margin=0. No unmeasured-oracle query or nearest fallback.
* `oracle`: retrospective only, using first-half held-out test shape trials.
  Pair oracle chooses cheaper measured separate/merged cost. For <=6 original
  ROIs, subset DP finds the exact minimum-cost partition under the additive
  shape-cost assumption. Above the cap use a clearly labeled greedy measured
  reference with the same enumeration. Neither is a global optimum of actual
  image-content wall time.

Cost-aware policies enumerate pairs in current group order, retain the first
maximum gain, remove the chosen pair, append its union, and repeat. Equality
separates. Source IDs are conserved without duplication; grouping preserves
original proposed regions. Geometric intentionally has its established distinct
sequential search. Its behavior is regression-tested against repository code.

All deployable multi-ROI plans are frozen before test costs are read/collected.
All possible subset enclosing-box shapes are computed from geometry, with a
budget gate before inference. Reference timing and scoring timing use disjoint
halves of independent held-out blocks. Independent scoring avoids choosing and
scoring the oracle using the same noisy samples. No online method can construct
an OracleCosts object from a calibration artifact.

## Metrics and uncertainty

Pair labels use reference mean costs; score every chosen action on disjoint
scoring means. Report TP/TN/FP/FN, accuracy, merge precision/recall/F1, false merge
FP/(FP+TN), false separate FN/(FN+TP), mean/median/summed costs, savings, speedup,
realized oracle gain, regret and normalized regret. A zero/nonpositive oracle
saving denominator yields N/A, never a fabricated percentage.

Reference-selected oracle may be worse on independent scoring from noise.
Retain negative reference regrets, reference/scoring decision disagreements and
oracle-gain exceptions rather than clipping. Also report nonnegative regret
relative to the scoring sample minimum separately (optimistically selected on
that sample). Confidence intervals are paired sequence-cluster percentiles,
with repetitions kept inside sequence clusters. Small sequence counts limit
interpretation. Trial counts/SE are retained, not treated as independent frames.

Boundary bins and minimum-gain sensitivity thresholds are predeclared in config;
report sizes, errors, false merges and regret, plus overlap/diagonal/separated,
empty-bounding-area and exact-lookup-domain strata. Near-noise-boundary status
uses the configured z times scoring gain SE; it is descriptive, not tuning.

Frame results report raw repeated serial timing, mean/median/p90/p95 of per-frame
repeat means, bootstrap mean CI, calls/processed pixels/planning/crop overhead,
all/eligible counts and FPS-equivalent=1000/mean serial ms. This is not real-time
video throughput. Policies are shuffled within each frame/repetition block and
share the exact same content/proposals. No silent retries/exclusions: failure
stops and is resumable. Partial cohorts have their own digest.

## Limits and publication gate

Synthetic shape cost ignores content-dependent detector/postprocess variability;
actual-content replay tests its practical consequences. GT proposals are an
annotation-informed condition. mAP and closed-loop tracking are not implemented.
No real checkpoint/dataset measurements were available in the implementation
checkout. Synthetic fixtures validate software only. Do not report their latency,
accuracy or rankings as findings. Real preflight, both detector pilots, a common
cohort on all three machines and independent calibrations remain publication gates.
