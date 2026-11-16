# Agent task: Reproducible Experiment B — Hardware-Aware ROI Merging on ImageNet-VID

## Mission
Implement and document a publication-grade **Experiment B** comparing ROI merging decisions for **YOLO26n** and **ROI-SSD-MobileNetV3** on the **`imagenet_vid_yolo_sampled`** subset (~4,744 frames), using the existing Stage 1 model checkpoints and existing merging/latency scripts wherever possible. Prepare the full experiment on a laptop, commit code/configuration/small manifests through Git, and make it reproducible on (1) laptop, (2) Core i9 desktop, and (3) Raspberry Pi 5. Large assets (checkpoints, dataset, raw measurements) can be transferred with SCP/file sharing; the dataset and checkpoints are reported to already exist on all machines.

**Do not assume that any paths, architectures, preprocessing functions, checkpointer formats, or existing benchmark implementations match the names below. Discover and document the actual repository interfaces before coding. Do not retrain models. Do not silently replace checkpoints or detector versions.**

Core methods to compare under identical conditions:

1. `geometric` — clearly defined, repository-existing geometric heuristic (name the exact rule, thresholds, and whether overlapping-only, proximity, bounding-box expansion etc.); if multiple candidates exist, designate the established simple baseline and optionally include additional heuristic baselines.
2. `affine` — cost model based on effective area: `T(A) = K + c*A` and merge if predicted `T(merged) < T(left)+T(right)`; for equal fixed overhead and positive slope this recovers the earlier threshold `tau = K/c`.
3. `piecewise` — calibrated piecewise latency estimator; compare predicted costs at the *effective* shapes, not raw proposed ROI areas.
4. `lookup` — empirically calibrated `T(H_eff,W_eff)` lookup with explicitly documented interpolation/nearest-neighbor policy and no unmeasured-oracle access.
5. `conservative_lookup` — same lookup with strict in-domain coverage and explicit out-of-domain/fallback/uncertainty safeguards; do not assume a specific rule exists until code has been inspected.
6. `oracle` — retrospective reference using held-out, *measured* processing costs for the same candidate shapes, not available to a deployable algorithm; must not be included as an online method. Clarify whether the oracle is pairwise or globally optimal for the multi-ROI case.
7. `no_merge` — mandatory control.

## Research question and clean experimental separation

**RQ:** How well do detector/hardware-specific latency models choose between separate and merged inference on realistically distributed video ROI configurations, and what fraction of the measured oracle saving can they achieve?

Implement three distinguishable layers, not one confounded benchmark:

- **A. Deterministic workload / ROI geometry generation (platform independent):** create fixed candidate ROI sets per frame *before* selecting a merging method. Persist them to a manifest. This is the **primary reproducible paired comparison** and must be identical for every method and hardware platform. Where reasonable, share geometry between detectors too.
- **B. Platform/model calibration + held-out shape timing:** fit models separately on every detector/hardware pair. Calibration observations must be independent of the evaluation/test measurements. No fitting on the evaluation frame latency, using the oracle timings as policy inputs, or tuning thresholds on test outcomes.
- **C. Policy evaluation / optional real detection:** evaluate the decisions and end-to-end execution against the same fixed workload. If running model inference on cropped images and computing mAP, report that as a **secondary** evaluation with explicit treatment of full-frame fallback, object context, and prediction remapping. Do not imply that fixed ROI replay measures closed-loop tracking recovery or actual detector-guided ROI generation.

## Phase 0 — mandatory repository audit (before implementation)

Search the codebase (e.g., `rg`) for:

- Existing Stage 1 benchmark entrypoints, checkpoint loading, YOLO26n inference interface, ROI-SSD-MobileNetV3 model creation, `until_1x1` dynamic-depth behavior, model-specific image preprocessing, output postprocessing and NMS.
- Merging variants (simple/geometric, overlap, greedy cost-aware), `shape_lookup_conservative`, the area/latency cost model, profile sampling, ROI shape normalization/quantization, padding and maximum-size/crop handling.
- Dataset metadata reader for `imagenet_vid_yolo_sampled`, frame ordering/sequence IDs, annotations, class mapping (ImageNet-VID to shared YOLO/ROI-SSD labels), earlier `cpu_pilot`, `experiment_a`, `experiment_b`, Stage 1 artifact paths and configuration.
- Actual runtime environments (OS, Python, PyTorch, Ultralytics or other detector versions, threading, CPU affinity, GPU setting, Pi-specific runtime constraints).

Create `docs/experiment_b_inventory.md` containing a table of discovered modules/functions/files, their role, and whether they are reusable or need adaptation. **Prefer reusing existing validated implementations; avoid creating subtly different crop/padding or inference code.** Record any inconsistencies found between old benchmark and current inference path before comparing results. Preserve legacy behavior behind explicit flags as needed.

Do not hard-code unsupported assumptions such as "both models require multiples of 32" or "ROI-SSD accepts any shape regardless of feature-pyramid constraints". Determine effective tensor shape using the **real preprocessing actually run on each model**. Check that ROI-SSD-MobileNetV3's current depth policy and checkpoint are compatible.

## Phase 1 — portability and asset verification

Create a versioned experiment configuration (YAML/JSON) with the dataset logical root, checkpoint logical IDs, detectors, calibrated/evaluated shapes, ROI-generation mode, random seed, frame selection/ordering, and run repetitions. Paths must be CLI/environment overrides; **absolute machine-specific paths must not be committed**.

Implement an `inspect` / `preflight` command that:

- Locates the dataset on the current machine and inventories sequences/frames, verifies that all expected files exist and can be decoded, verifies annotation/class mappings, checks image dimensions and sequence order. Do **not** rely on "~4,744" being exact; report the actual count.
- Locates the exact Stage 1 checkpoint for each model; reports checkpoint SHA-256, model architecture, input-shape rules, active-head policy, class count and mapping, and a small inference smoke test. Fail on checkpoint mismatch, unless explicitly acknowledged in a non-comparable run.
- Reports Git commit, config digest, manifest digest, dataset index/content hashes (practical hash policy documented), installed package versions, device, CPU model, threads and CPU governor/power state where accessible.
- Verifies that every manifest frame resolves on all devices through portable *relative* paths, not machine-local absolute paths.

Use a minimal expected directory convention, adapting to the real repo:

```text
experiments/roi_merge_b/
  README.md
  config.example.yaml
  prepare_workload.py
  calibrate.py
  evaluate_decisions.py
  evaluate_runtime.py
  evaluate_detection.py       # optional/secondary
  aggregate.py
  tests/
  docs/experiment_b_inventory.md
  docs/protocol.md
artifacts/roi_merge_b/<experiment_id>/
  manifest.jsonl             # fixed geometry, portable
  manifest.meta.json         # provenance / hashes / split definitions
  calibration/<device_model_run>/...
  evaluations/<device_model_run>/...
  reports/...
```

**Git:** commit the experiment code, schemas, small configuration files, hashes, and compact manifests when feasible. Do not commit full ImageNet-VID frames, model weights or large raw latency traces. Generate a transfer bundle (archive) of the manifest, metadata, config and artifact-checksum index for SCP/copy when needed. All outputs should use stable run IDs and be mergeable across machines.

## Phase 2 — freeze exactly the same ROI geometry

**Primary approach:** generate a single portable, deterministic **manifest of proposed ROI rectangles** from the dataset on the laptop, then reuse byte-for-byte on all machines and all policies.

Make ROI definitions independent of policy selection and of processing-time order. Store image-relative pixel coordinates or normalized coordinates **plus original image dimensions**, with a precise inclusive/exclusive coordinate convention (`x1,y1` inclusive, `x2,y2` exclusive recommended). Store per-frame: sequence ID, frame index, relative image path, W/H, candidate `roi_id`s, ROI pixel boxes, generation source, input annotation/teacher version, context/padding settings, and optional object IDs. Sort records canonically and hash the manifest. No random ROI generation during evaluation.

**Choose and clearly name the ROI-generation methodology after auditing repository capabilities:**

- Best for a *detector-independent paired cost benchmark*: fixed geometry from ground-truth annotations or a frozen, platform-independent reference detector/tracker pass. If GT is used, label it **GT-derived candidate ROIs**, not tracker-generated online ROIs. Do not leak GT to measured detection accuracy interpretation or call it realistic deployed detection.
- If the repo already contains a reliable Stage 1 tracker-generated ROI manifest, reuse it when applicable. Otherwise create the reference pass *once on the laptop*, freeze outputs, and record the originating checkpoint, tracker settings and frame sequence. Never regenerate geometry on Pi/i9, since detector numerical differences and control-flow feedback would confound cross-platform comparisons.
- Require **multi-object scenes** and preserve the actual distribution of 0, 1, 2, 3+ ROIs. Deduplication, class filtering, minimum context padding and clipping are to be deterministic and documented. Handle no-ROI frames via a clearly specified fallback (e.g. full-frame) or exclude them from decision-only analysis with separate counts.
- To avoid a trivial dataset dominated by single-ROI/no-merge frames, report metrics both on **all frames** and the **merge-eligible subset (>=2 candidates)**. Extract additional stratified subsets based on geometry (overlap, near, diagonal, separated, high empty-area ratio). Do **not** silently alter the natural distribution and still describe it as representative. Optional oversampled hard subset must be identified as a separate controlled stress test.
- Assign frames/sequences to predefined `calibration`/`validation`/`test` splits. Prefer splitting by **video sequence**, never random frames from the same sequence across splits when temporal correlation matters. Calibration for timing may use synthetic tensor shapes, but all tuning (including uncertainty margins/lookup fallback/geometry thresholds) must use calibration/validation, **not the held-out test results**.

A standardized manifest is the source of truth. Shape normalization depends on the model, but on any one device each model must receive **identical candidate image pixel rectangles** across policies. Store both raw ROI geometry and derived effective tensor `(H_eff,W_eff)` for each detector/preprocessing configuration, and validate they match what the runtime actually executes.

## Phase 3 — hardware-specific latency profiling

Each platform must independently calibrate YOLO26n and ROI-SSD-MobileNetV3; a checkpoint can be common but latency parameters **must never be copied between hardware**.

- Replicate the actual production execution path and report exactly which stages are timed: at minimum **network inference-only**, optionally full ROI pipeline (crop, resize/pad, tensor conversion, model, NMS, remapping). Do not mix these timing scopes within the same decision cost model. If the decision objective is pipeline latency, train/profile the *same pipeline scope*.
- Warm up each effective shape and overall runtime; set reproducible execution controls (device, intra-op/inter-op threads, CPU affinity where reasonable, batch size, inference mode, disabled gradients, power settings, thermal stabilization). **Raspberry Pi 5 has 4 Cortex-A76 CPU cores**: do not set threads to 8 merely for parity with an 8-thread desktop without explicit controlled evidence. Test reasonable Pi thread counts in a separate pilot and freeze one for comparisons.
- Prefer calibration at shapes actually encountered in the manifest, plus a designed grid covering possible merged bounding rectangles. Include min/max, boundary shapes, non-square aspect ratios, and model-specific tensor strides; avoid generating an infeasible dense full 2-D lookup on Pi.
- Quantization is a **cost-model and inference-path property**: e.g., YOLO may require model-specific effective shape alignment; ROI-SSD-MobileNetV3 may have different constraints. Do not confuse lookup table granularity (e.g. 4-pixel grid) with executed input-shape quantization. Document their separate effects on memory and accuracy.
- Profile with repeats, warmups, randomized/block-balanced shape execution order, robust latency summaries (median, trimmed mean, optional confidence intervals); log thermal/frequency state if available. Repeat controls at intervals to detect drift. Record whether elapsed time includes host/device synchronization.
- Fit `affine` with bounds/validity checks (positive slope, plausible intercept) and `piecewise` with branch/breakpoint chosen solely on calibration/validation, guarding against overfit. Build shape lookup and implement an explicit covered-domain/fallback rule for conservative lookup. Include table size in entries/bytes, calibration time, and decision overhead.
- Produce a self-contained cost-model artifact per detector/device/version (parameters + shape domain + policy rules + provenance + hashes + fitting diagnostics), so its predictions can be loaded without recalibration during the test pass.
- **No unfair oracle leakage:** the lookup must be trained on calibration timings, not evaluation measurements of the same candidate cases. Independent timed replicates on the same shape can be used for held-out validation. Test result timing can be used to compute a retrospective reference/oracle only.

## Phase 4 — merge policies and fairness

Define a common interface such as:

```python
merge_policy(rois, predict_latency, metadata, config) -> list[merged_roi_groups]
```

The interface returns groups of original `roi_id`s and their enclosing bounding rectangles. No policy can alter the initial ROI set or image contents. Use the exact same underlying pair candidate enumeration and stable tie-breaking across cost-aware variants; differences should come only from predicted costs. Document whether merging is iterative greedy, connected-components, or global optimization.

For `affine`, `piecewise`, `lookup`, `conservative_lookup`, use the same greedy pair-selection rule wherever possible: at each step evaluate the positive predicted gain and choose the best pair, repeat until no positive gain. Include timing for the planner. Store the ordered merge trace and estimated gain for reproducibility. `geometric` must implement a fully documented independent non-cost baseline; any hyperparameters chosen on validation only.

For pairwise analysis, each manifest ROI pair receives decisions from every policy on equal original inputs. For full-frame multi-ROI analysis, choices of one merge change possible later merges; therefore **pairwise decision accuracy must not be misrepresented as global partition accuracy**.

**Oracle definition:**

- Pairwise oracle: independently measure `T_sep` vs `T_merge` using held-out trials; choose lower measured cost. Decisions near noise-level differences are uncertain; report sensitivity to a minimum meaningful gain threshold.
- Multi-ROI oracle: for small `n` (e.g., capped at 8–10 ROIs), evaluate exact minimum-cost grouping using cached held-out measured cost of every feasible subset enclosing rectangle, with explicit inference-time/cost assumptions. For larger n, use a documented approximate/greedy measured-cost oracle and **do not call it global optimum**. If exhaustive shape timing is too expensive on Pi, calculate a faithful, limited subset and report coverage; do not fabricate oracle results.
- The oracle may include pipeline overhead only if all methods' objective includes the same overhead. The oracle reference must use independently measured means with confidence intervals or repeated trials and never be treated as deterministic truth near zero delta.

## Phase 5 — required evaluation outputs

### 5.1 Decision-only evaluation (primary)

For **pairwise** ROI comparisons on held-out test shapes and frame geometry, output for each detector/platform:

- TP, TN, FP (false **merge**), FN (missed beneficial merge), decision accuracy, precision/recall/F1 for the merge class, false-merge rate `FP/(FP+TN)`, false-separate rate `FN/(FN+TP)` (be explicit about denominators).
- Measured mean/median per-example cost under each selected policy, baseline no-merge cost, reference oracle cost, absolute time saved, relative speedup and **realized oracle gain** `sum(T_no_merge-T_policy) / sum(T_no_merge-T_oracle)` when the denominator is strictly positive and comparably measured; report N/A otherwise.
- Per-sample regret `r_i=T_chosen,i-min(T_separate,i,T_merge,i)` and mean/sum regret; normalized regret `sum(r_i)/sum(T_oracle,i)`. Compute from **held-out** timing and use independent repeated measurements to quantify statistical uncertainty.
- Boundary analysis: bin by `abs(T_sep-T_merge)` relative to fixed bins in milliseconds and/or relative to measured timing noise; report bin sizes, decision errors, false merges and regret. Include geometric strata (diagonal, overlapping, separated, extra bounding-area ratio).
- Distinguish naturally sampled test-set results from an explicitly designated **hard/near-boundary stress subset**. Avoid pooled headline percentages when sampling intentionally overweights difficult cases.
- Provide paired confidence intervals via sequence-cluster bootstrap where video sequence correlation exists, and/or repeated-run confidence intervals as appropriate. Do not assert meaningful model superiority solely from small differences in mean latency.

### 5.2 Multi-ROI frame-level merging (primary or secondary, but required)

For every test frame, fixed candidate geometry and each policy:

- number of candidate ROI vs final segments; processed effective pixels; number of detector calls; per-frame predicted total cost and held-out measured runtime; planning overhead and pipeline overhead (separately); percentage with 0/1/2+ ROI and percentage merge-eligible.
- Mean/median/p90/p95 runtime and FPS-equivalent derived from actual serial execution; do not report a true real-time video FPS unless input scheduling, drops and backpressure are modeled/measured.
- Compare `no_merge`, geometric, affine, piecewise, lookup, conservative lookup, oracle/reference where feasible. Use **the exact same manifest** for every method, checkpoint and host.
- Produce a summary table stratified by platform × detector and detailed per-frame JSONL/Parquet/CSV traces. Specify frame retries/errors/exclusions.

### 5.3 Real detection quality (secondary, recommended)

If resources permit, run model inference on fixed ROI configurations and map predictions back to full-image coordinates. Use the shared ImageNet-VID category mapping, test annotations and the exact same postprocessing/evaluation implementation per detector; carefully handle duplicate predictions from overlapping ROIs and cross-ROI NMS. Compute mAP50 and preferably mAP50:95 where applicable, plus precision/recall. Report latency vs accuracy Pareto tradeoffs and keyframe/fallback policy explicitly. Fixed manifest that uses GT-derived ROIs creates **annotation-informed candidate proposals**, so detection accuracy here represents an oracle-proposal condition, not an online deployable system.

The primary contribution is the cost-decision model, so absence of end-to-end mAP must not block decision-only results; label scope/limitations transparently. If comparing end-to-end mAP, keep the ROI set fixed across policies and verify that merging can change image context and hence predictions, despite equal nominal object coverage.

## Phase 6 — reproducible orchestration

Implement CLI entry points (exact module names may change after the repository audit) with behavior equivalent to:

```bash
python -m experiments.roi_merge_b.inspect --config configs/roi_merge_b.yaml
python -m experiments.roi_merge_b.prepare --config configs/roi_merge_b.yaml --output artifacts/roi_merge_b/frozen_manifest
python -m experiments.roi_merge_b.calibrate --config configs/roi_merge_b.yaml --manifest artifacts/roi_merge_b/frozen_manifest/manifest.jsonl --model yolo26n --output results/$(hostname)/yolo26n/calibration
python -m experiments.roi_merge_b.evaluate --config configs/roi_merge_b.yaml --manifest artifacts/roi_merge_b/frozen_manifest/manifest.jsonl --model yolo26n --models-dir results/$(hostname)/yolo26n/calibration --output results/$(hostname)/yolo26n/evaluation
python -m experiments.roi_merge_b.aggregate --results-root results --output reports/experiment_b
```

Run both detectors on each host; generate the manifest **once only**. Never modify a manifest silently during evaluation. Output structured machine-readable artifacts plus a concise researcher-readable markdown report.

Recommend a results structure with metadata fields:

- `experiment_id`, `run_id`, `git_commit`, `manifest_sha256`, `dataset_index_hash`, `checkpoint_sha256`, `detector`, `hardware_id`, `software_versions`, `calibration_id`, `shape_rule_id`, `timing_scope`, `seed`, `threads`, `temperature/frequency` (if available), `policy`.
- Record per-ROI and per-frame actual rectangles, effective shapes, merge trace, predicted and measured latency, reference/oracle and grouping. Include units consistently (seconds vs milliseconds; pixels vs effective pixels).
- Include resumable atomic output/checkpointing, failure recovery and a `--dry-run` / `--limit-frames` smoke mode. Retrying a test must not reselect ROIs or train models on test data.

**Portability test:** generate manifest on laptop, run `inspect` and first 10–20 frames on another computer using only Git-tracked code/manifest plus local dataset/checkpoints. Verify all frame IDs, ROI coordinates, effective-shape computations, and candidate-list hashes match laptop exactly. Compare checkpoint hashes where weights are intended to be identical.

If filenames/classes or dataset order differ across hosts, resolve via canonical relative paths and immutable dataset-index mapping; do not rely on directory traversal order.

## Phase 7 — feasibility and runtime controls

A full exhaustive pair-oracle with many ROI shapes on Raspberry Pi can be prohibitively expensive. Build in an estimated-work report before execution:

- count of frames, ROI count per frame, unique effective shapes for original and merged candidates, expected number of network invocations under each stage, timing repetitions and rough runtime based on pilot results;
- deduplicate shape timings when the timing scope permits; separate microbench profiling from actual content-dependent end-to-end runs;
- selectable test subsets (`smoke`, `pilot`, `full`) with immutable sampled frame IDs and explicit sampling protocol;
- cache warmed timing per shape while preserving independent calibration/test splits and repeated measurements;
- prevent thermal-run drift from biasing method comparisons by block-randomizing policy order / interleaving controls;
- ensure device-appropriate threads, memory limits and lookup sizes.

Do not compromise comparability by silently reducing the test set only on Pi. If Pi cannot run the whole job, state the completed common subset and report it **separately** from laptop/i9 full-set results. Cross-platform headline comparisons must be based on the same frame IDs.

## Verification / unit tests

At minimum test:

1. Canonical manifest reproducibility (same seed/source => identical bytes and SHA-256) and coordinate clipping.
2. Deterministic ROI ordering and deterministic tie-breaking.
3. Exact mapping of proposed rectangles to executed effective tensor shapes for each model; no mismatched rounding or extra preprocessing.
4. For affine `T(A)=K+cA`, a pair's decision agrees algebraically with the `tau=K/c` criterion for `c>0` (taking effective area and any model constraints into account).
5. Pair with equal costs / almost equal costs: documented threshold and tie convention.
6. Non-overlap, partial overlap, diagonal separation, nested ROI, empty/no ROI, full-frame-sized ROI and extreme aspect ratios.
7. Merging preserves the union of source ROIs, stays within image bounds, and outputs no duplicate/missing ROI IDs.
8. Lookup out-of-domain behavior (including conservative abstention/fallback) and memory report; no test-data-derived interpolation.
9. Oracle costs use held-out measurement files only; policies cannot access those files during selection.
10. Split boundaries by sequence, checkpoint SHA mismatch, manifest hash mismatch and missing assets fail loudly.
11. Decision metric consistency: TP/TN/FP/FN, regret >=0 up to numerical tolerance, oracle realized-gain bounds under consistent measurement assumptions (report exceptions caused by noise).
12. Small deterministic cross-platform smoke case produces identical ROI groups given identical cost-model artifacts (when running hardware-specific models, results **may** legitimately differ and that difference must be documented).

## Expected publication-ready outputs

Produce:

1. Working experiment code + tests + example config + portable fixed ROI manifest.
2. Detailed inventory and protocol, exact setup/transfer/run commands for laptop, Core i9 and Raspberry Pi 5; a small `TRANSFER.md` with `git` and `scp` examples based on **user-configurable placeholders**, not guessed actual paths.
3. Per-device/per-detector calibration artifacts and held-out evaluation reports; common test cohort, with hashes.
4. Main results table: `device × detector × policy` — decision accuracy, false merges/separates, mean regret, measured cost, speedup, oracle gain and latency overhead, with uncertainty where possible.
5. Supporting tables: model fit (RMSE, MAE, R² on held-out shape measurements), shape-domain in/out performance, table size/calibration time, geometry strata, timing/profiling settings.
6. Figures: cost vs area, latency heatmap or selected shape slices, predicted-vs-measured merge gain, regret by boundary distance, and mean runtime / realized oracle gain by policy and hardware.
7. A compact `paper_results.md` containing numerical findings, limitations, exact methods, figure/table paths and **explicit distinction between new findings and previously published video results**.

## Acceptance criteria / success gates

- A single identical manifest/test frame cohort is reproducibly evaluated across all 3 machines and both detectors (or documented common complete subset with no silent deviations).
- The same Stage 1 weights and preprocessing rules are used and verified by hash/version metadata, independently of host.
- All 7 policies including no-merge and retrospective oracle are implemented and run; missing/undefined historical policies are clearly documented and created without relabeling their behavior.
- Independent calibration of all deployable models per detector/platform; no oracle/test leakage into merge decisions.
- Pairwise and frame-level evaluations, uncertainty/near-boundary reporting and a complete analysis of exceptions.
- An honest conclusion is possible even if `geometric` outperforms advanced models on the natural subset. **Do not optimize the benchmark to guarantee a desired ranking.**

## Agent workflow / stop conditions

1. **Audit and write an implementation plan** with discovered modules and specific intended reuse, then implement without asking for confirmation unless critical assets are inaccessible or checkpoint/model identity cannot be established.
2. Prioritize manifest + smoke test, then 1 detector × laptop pilot, then both laptop models, then portability and other hardware, then full evaluation.
3. If a prerequisite is missing, create an actionable blocker report containing the exact file/command/mismatch and continue with independent tasks; do not invent values or silently use a substitute.
4. At the end provide a concise completion report: changed files, CLI commands, validated checkpoints/hashes, dataset counts, manifest digest, smoke-test output, tests run, results and remaining blockers. Clearly distinguish **implemented**, **executed**, and **not yet executed**.

## Non-goals

- Retraining YOLO26n or ROI-SSD-MobileNetV3.
- Claiming GT-derived ROI workload represents a deployable tracker-based solution.
- Reproducing the entire old ROI formation/uncertainty paper.
- Treating calibration fit R² as evidence of merge-decision accuracy.
- Reusing a fixed `tau` calibrated on another hardware platform as a main baseline without labeling it cross-device transfer.
- Treating a greedy measured-cost oracle as globally optimal without an exact optimization guarantee.
