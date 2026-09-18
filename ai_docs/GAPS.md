# Gaps to revisit after Experiments A and B

## Scope and baseline

Recorded on **2026-09-15**, against commit `406b606` and the then-current uncommitted Experiment B v2 changes.

This is a follow-up backlog from a repository review, not an instruction to interrupt or redesign the ongoing experiments. Recheck each finding against the code and results that exist when A and B are established. No fixes were made as part of this review.

The review read all project Markdown files, inspected the main pipeline and experiment implementation, and examined saved results. The cost-model unit suite passed **10 tests**:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments/cost_model_validation/tests -v
```

Training and full benchmarks were not rerun. Passing this small suite does not establish that every acceptance criterion in the experiment specifications is covered.

## G1. Video merge rule differs from the rule validated by A and B

**Status:** Confirmed implementation mismatch; intended behavior needs reconciliation.

**Evidence:** [tools/mergers/greedy.py](tools/mergers/greedy.py), functions `greedy_roi_merge` and `compute_delta`; [experiments/cost_model_validation/models.py](experiments/cost_model_validation/models.py), function `decide_merge`.

The experiments compare predicted merged cost with predicted separate cost. For the documented linear model:

```text
T(A) = K + c*A
merge iff A_union - A_1 - A_2 < K/c
```

The video merger instead defines:

```text
delta = beta_1*A_1 + beta_2*A_2 - beta_union*A_union
merge iff delta > tau
```

If the intended weighted cost is `K + c*beta*A`, the corresponding condition is `delta > -tau`, not `delta > tau`. The video implementation additionally has depth-dependent weights, an area-inflation constraint, and shortcuts based on canvas/image area. Consequently, changing the sign alone would not establish equivalence with the synthetic experiment.

**Why it matters:** A good decision result in Experiment B does not directly validate the policy currently used by the video benchmark.

**Follow-up:**

- [ ] Decide which cost model the video merger is intended to implement.
- [ ] Reconcile the sign, depth weights, area constraint, and special shortcuts with that model.
- [ ] Test overlapping, separated, containment, exact-tie, and large-tau cases against explicit separate/merged predicted costs.
- [ ] Establish whether the video pipeline should consume the shared latency-model interface from the experiments.

## G2. Balanced order counts do not preserve the advertised ABBA blocks

**Status:** Confirmed implementation/protocol mismatch.

**Evidence:** [collect_experiment_b.py](experiments/cost_model_validation/collect_experiment_b.py), functions `balanced_orders` and `collect`.

The collector constructs ABBA/BAAB-style order schedules, then globally shuffles individual `(pair_id, repetition)` trials. Each pair retains balanced counts of `merged_first` and `separate_first`, but the scheduled blocks are not preserved as contiguous measurements. Metadata describes the order design as `balanced_abba_baab`.

**Why it matters:** Equal order counts and locally counterbalanced blocks are different controls for temporal drift. The protocol description should match actual execution.

**Follow-up:**

- [ ] Choose and document the intended timing design.
- [ ] If contiguous blocks are required, randomize blocks while preserving their internal execution order.
- [ ] Test the actual execution sequence, not only the number of occurrences of each order label.

## G3. Latest saved results still show a substantial order effect

**Status:** Observed in a saved run; cause is not established.

**Evidence:** Local artifacts under `outputs/cost_model_voc_yolo26n_v3/`, especially `decision_metrics.json` and `experiment_b_metadata.json`. These files are ignored by Git and may not exist in another checkout.

The reviewed run reports:

- 500 unique computational pairs, including 200 near-boundary pairs;
- 20 paired repetitions per pair;
- 372 determinate pairs and 25.6% ambiguous pairs;
- an execution-order effect of approximately **6.51 ms**;
- `order_effect_warning: true` and `hardware_state_shift: true`.

Saved decision metrics were:

| Rule | Accuracy on determinate pairs | Mean regret over all pairs |
| --- | ---: | ---: |
| Linear threshold | 87.10% | 0.970 ms |
| Quadratic direct cost | 98.66% | 0.186 ms |
| Piecewise direct cost | 98.12% | 0.220 ms |

These are historical observations, not target values or acceptance criteria. Regret is the measured extra latency from choosing the slower alternative.

**Why it matters:** Model ranking looks promising, but timing order and system-state variation may affect labels and regret. Also, the `hardware_state_shift` flag is based on deviation from model predictions; it does not by itself prove a physical hardware-state change, because model error can also cause deviation.

**Follow-up:**

- [ ] Reassess order effects after the measurement protocol is finalized.
- [ ] Compare independent sessions and inspect raw control measurements over time.
- [ ] Distinguish model prediction error from temporal drift when interpreting control warnings.
- [ ] Retain ambiguous outcomes and report the scope of each accuracy/regret metric.

## G4. Experiment timing boundaries are not fully equivalent to production inference

**Status:** Confirmed implementation limitations; impact depends on backend and timing mode.

**Evidence:** [timing.py](experiments/cost_model_validation/timing.py), [ultralytics_adapter.py](experiments/cost_model_validation/ultralytics_adapter.py), [roissd_adapter.py](experiments/cost_model_validation/roissd_adapter.py), and [pipeline.py](tools/helpers/pipeline.py).

- The shared experiment timer does not synchronize CUDA before and after inference. GPU execution is asynchronous, so host-side elapsed time alone is insufficient for a completed-forward latency measurement. This does not invalidate CPU timing on that basis.
- The Ultralytics experiment adapter's `postprocess` returns its input unchanged. Its `detector_call` therefore does not implement the documented normal postprocessing/NMS stage.
- The ROI-SSD adapter invokes the repository model's complete forward path. The boundary called `inference_only` should be checked for any decoding/postprocessing performed inside that forward path before comparing it with YOLO raw-forward timing.
- The video pipeline currently returns a grid size of 32 for all models in `_model_roi_grid`, whereas the ROI-SSD experiment adapter defaults to stride 1. Supported arbitrary input sizes and the actual shapes used by the video pipeline are different concepts.

**Follow-up:**

- [ ] Document exactly what is timed for each backend and mode.
- [ ] Implement and verify device synchronization before relying on CUDA experiment results.
- [ ] Match production preprocessing/postprocessing where claiming end-to-end detector-call equivalence.
- [ ] Verify that calibration shapes and effective areas represent inputs that the video pipeline actually processes.

## G5. Reproducibility and test coverage lag behind the specifications

**Status:** Confirmed repository-level limitations; detailed acceptance coverage still needs an audit.

**Evidence:** [.gitignore](.gitignore), [test_core.py](experiments/cost_model_validation/tests/test_core.py), [exp_context.md](exp_context.md), [experiment_fix_suggestions.md](experiment_fix_suggestions.md), and [agent_instructions_experiment_b_v2.md](agent_instructions_experiment_b_v2.md).

All CSV and JSON files are ignored globally, including raw measurements, calibration artifacts, metadata, and derived summaries. The latest experiment implementation also had nine modified tracked files at review time, so the recorded commit alone does not identify the exact code used. Several configurations contain machine-specific paths.

The ten passing tests cover useful basics, but do not establish all of the specifications' requirements, such as end-to-end same-pair evaluation, calibration identity validation, resume compatibility, block-preserving execution, and bootstrap behavior under controlled noise.

**Follow-up:**

- [ ] Preserve the exact source revision or patch alongside each retained experiment session.
- [ ] Define an explicit artifact-retention policy: selected versioned fixtures, an external archive, or another reproducible storage mechanism.
- [ ] Record hardware, dependencies, weights identity, resolved configuration, seeds, and timing boundaries with results.
- [ ] Audit specification acceptance criteria against meaningful automated tests and integration checks.
- [ ] Provide portable configuration examples with documented local path overrides.

## G6. Documentation mixes current behavior with historical plans

**Status:** Confirmed documentation drift.

**Evidence:** [README.md](README.md), [CONTEXT.md](CONTEXT.md), [IMAGENET_VID_INTEGRATION_GUIDE.md](IMAGENET_VID_INTEGRATION_GUIDE.md), and [adaptive_tau_context.md](adaptive_tau_context.md).

- The root README still leads with the original educational SSD project, while the repository now centers on ROI video-detection research.
- The README invokes `tools.build_imagenet_vid_voc_subset`; the current module is `tools.imagenet.build_imagenet_vid_voc_subset`.
- The ImageNet integration guide references `config/imagenet-vid.yaml`, which is absent from the reviewed tree.
- Adaptive-tau notes describe planned implementation even though an `OnlineCostModel` now exists in the video benchmark.
- `CONTEXT.md` presents `tau` partly as an absolute ROI-area benefit threshold. For the documented pairwise linear merge rule, its precise meaning is the allowable **additional merged area**, `A_union - A_1 - A_2`, that trades against one saved invocation.

**Follow-up:**

- [ ] Separate current user documentation from historical design notes and experimental observations.
- [ ] Update executable commands and configuration references.
- [ ] Explain the merge threshold using additional effective area and saved invocation cost.
- [ ] Date empirical claims and link them to retained runs rather than treating them as universal detector properties.

## Suggested revisit order

After A and B are established, first reconcile the video policy with the validated model (**G1**), then verify protocol and timing semantics (**G2–G4**) before using synthetic results to support video-level conclusions. Preserve the final implementation and evidence (**G5**) and refresh the documentation (**G6**) around the established behavior.
