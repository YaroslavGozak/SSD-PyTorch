# Fix suggestions for the synthetic ROI cost-model experiments

## Purpose

This document describes the changes required before the results of Experiment A and Experiment B can be used in the paper or in a response to the reviewer.

The current implementation is a successful smoke test, but several issues make the reported values unsuitable as final evidence:

- Experiment A and Experiment B do not appear to use the same effective tensor-shape convention.
- Experiment B frequently evaluates areas outside the range calibrated in Experiment A.
- The generated configurations contain computational duplicates that inflate the effective sample size.
- The reported bootstrap confidence interval for $\tau$ is implausibly narrow and is not supported by the raw observations.
- The paired timing experiment has a statistically significant execution-order effect.
- The current run was performed on Windows AMD64 rather than the target Raspberry Pi 5.

The goal of the fixes is not to force the linear model to look better. The code must accurately show where the area-only model works, where tensor shape matters, and whether the resulting threshold still makes useful merge decisions.

## Current results that must remain reproducible

Experiment A currently reports:

$$
K_t=0.0123146\text{ s},
\qquad
c_t=1.46265\cdot10^{-7}\text{ s/pixel},
$$

$$
\tau=\frac{K_t}{c_t}=84\,194\text{ pixels}.
$$

The raw-observation linear fit has:

- $R^2=0.5229$;
- adjusted $R^2=0.5225$;
- RMSE = 3.605 ms;
- MAE = 2.879 ms.

Experiment B currently reports:

- 500 pairs and 20 repetitions per pair;
- 5.8% ambiguous pairs;
- 95.54% accuracy on 471 determinate pairs;
- 100% precision and 77.42% recall for the `merge` class;
- 88.71% balanced accuracy;
- mean regret = 0.198 ms;
- total latency overhead relative to the per-pair oracle = approximately 0.63%.

These values are useful regression targets for the current implementation. They are not target values that a corrected implementation must reproduce exactly.

## P0 fixes required before another full run

### 1. Use one detector adapter and one tensor-shape definition

#### Current problem

In Experiment A, all logged tensor dimensions are multiples of stride 32. In Experiment B, none of the 500 pairs has all three inputs aligned to stride 32. Examples include `64×132` and `766×132`.

Therefore, either:

- Experiment B bypasses the preprocessing and stride-rounding used by Experiment A; or
- the `*_tensor_w` and `*_tensor_h` fields in Experiment B contain raw geometric dimensions rather than actual model-input dimensions.

The $\tau$ fitted in Experiment A cannot be applied rigorously in Experiment B unless both experiments use the same invocation path and the same definition of area.

#### Required implementation change

Create one shared adapter method that is used by both experiments:

```python
@dataclass(frozen=True)
class PreparedInput:
    tensor: Any
    requested_h: int
    requested_w: int
    crop_h: int
    crop_w: int
    tensor_h: int
    tensor_w: int

    @property
    def effective_area(self) -> int:
        return self.tensor_h * self.tensor_w


class DetectorAdapter(Protocol):
    stride: int

    def prepare(
        self,
        image: np.ndarray,
        requested_hw: tuple[int, int],
    ) -> PreparedInput:
        ...

    def infer(self, prepared: PreparedInput) -> Any:
        ...
```

Both Experiment A and every `R1`, `R2`, and `RU` invocation in Experiment B must call the same `prepare()` and `infer()` implementations.

Do not calculate `tensor_h` and `tensor_w` from geometry alone. Read them from the prepared tensor immediately before inference.

#### Required assertions

For every timed invocation:

```python
assert prepared.tensor_h == int(prepared.tensor.shape[-2])
assert prepared.tensor_w == int(prepared.tensor.shape[-1])
assert prepared.effective_area == prepared.tensor_h * prepared.tensor_w
```

If the backend requires stride-aligned inputs:

```python
assert prepared.tensor_h % adapter.stride == 0
assert prepared.tensor_w % adapter.stride == 0
```

If arbitrary shapes are intentionally supported, log that fact explicitly and use arbitrary shapes in both experiments. Do not combine a stride-rounded calibration with an unrounded decision test.

#### Acceptance test

For an identical requested `HxW`, Experiment A and Experiment B must produce exactly the same:

- prepared tensor shape;
- effective area;
- timing boundary;
- model invocation function.

Add an integration test using a fake adapter and, when weights are available, a short real-model smoke test.

### 2. Eliminate extrapolation between experiments

#### Current problem

Experiment A was calibrated only up to:

$$
320\times320=102\,400\text{ pixels}.
$$

In Experiment B:

- 56.8% of merged inputs exceed 102,400 pixels;
- the maximum merged input is 1,330,566 pixels;
- approximately 8% of the individual ROI inputs also exceed the Experiment A range.

More than half of the merge decisions therefore use an extrapolated cost model.

#### Required implementation change

Support one of two explicit policies in configuration:

```yaml
domain_policy: expand_calibration  # or reject_out_of_domain
```

`expand_calibration`:

1. Generate or load the planned Experiment B geometries without running inference.
2. Pass all unique `R1`, `R2`, and `RU` requested shapes through the shared preprocessing shape resolver.
3. Add representative effective shapes covering the entire required range to Experiment A.
4. Fit the model only after this extended calibration is complete.

`reject_out_of_domain`:

1. Fit Experiment A within its configured limits.
2. Reject any Experiment B pair for which `R1`, `R2`, or `RU` lies outside the calibrated area and dimension range.
3. Log the rejected count and reasons.

The default for the paper should be `expand_calibration`, with limits matching the input sizes that can occur in the real ROI pipeline.

#### Domain checks

Area alone is not sufficient for the check. A shape may have a familiar area but an unseen extreme dimension or aspect ratio. Store the calibration envelope:

```json
{
  "min_effective_area": 0,
  "max_effective_area": 0,
  "min_tensor_h": 0,
  "max_tensor_h": 0,
  "min_tensor_w": 0,
  "max_tensor_w": 0,
  "min_aspect_ratio": 0.0,
  "max_aspect_ratio": 0.0
}
```

Flag a pair as out of domain if any input violates the configured envelope.

#### Acceptance test

The final Experiment B metadata must report:

```text
out_of_calibration_pair_count = 0
out_of_calibration_invocation_count = 0
```

unless an explicit extrapolation experiment is requested and labeled separately.

### 3. Deduplicate Experiment A configurations before timing

#### Current problem

The current generator produces 36 entries per run but only 31 unique requested shapes and 25 unique effective shapes. In particular, effective shape `320×160` is measured 240 times, while most shapes are measured 40 times.

The duplicate was apparently created when several requested configurations were clipped to the same maximum dimensions. It gives that shape six times the weight of a normal shape in the regression.

#### Required implementation change

Generate the full candidate list first, preprocess or resolve every candidate shape, and deduplicate before timed collection.

Use one of these keys depending on the intended analysis:

```python
requested_key = (requested_h, requested_w)
effective_key = (tensor_h, tensor_w)
```

The default benchmark should contain each `effective_key` exactly once per `run_id`.

Keep provenance separately:

```python
effective_shape_sources: dict[tuple[int, int], list[RequestedShape]]
```

This preserves information about collisions without repeating the timed configuration.

#### Acceptance test

After each run:

```python
counts = rows.groupby(["run_id", "tensor_h", "tensor_w"]).size()
assert counts.max() == 1
assert counts.min() == 1
```

### 4. Generate unique near-boundary computational configurations

#### Current problem

The current Experiment B contains 222 `near` pairs, but 200 of them have the same computational shape tuple:

```text
R1 = 64×132
R2 = 64×132
RU = 766×132
delta_effective_area = 84216
```

The geometric positions differ, but positions do not create different computation in an `inference_only` benchmark when input shapes and content policy are otherwise equivalent.

This pseudo-replication raises the reported accuracy from approximately 92.3% on unique determinate computational configurations to 95.5% on the duplicated dataset.

#### Required implementation change

Define the computational identity of a pair:

```python
def computational_pair_key(pair: PreparedPair) -> tuple:
    return (
        pair.r1.tensor_h,
        pair.r1.tensor_w,
        pair.r2.tensor_h,
        pair.r2.tensor_w,
        pair.union.tensor_h,
        pair.union.tensor_w,
    )
```

Reject duplicate keys during pair generation.

The `near` stratum must contain at least 100–200 unique keys after preprocessing. Generate candidates using diverse combinations of:

- small, medium, and large `R1`/`R2`;
- square, wide, and tall ROI;
- horizontal, vertical, and diagonal separation;
- partial overlap and containment where applicable;
- different union aspect ratios.

Calculate the boundary ratio only after preprocessing:

$$
r=\frac{A_U^{eff}-A_1^{eff}-A_2^{eff}}{\hat\tau}.
$$

Recommended bins:

```yaml
boundary_bins:
  strong_merge: [-1.0, 0.5]
  merge: [0.5, 0.8]
  near_low: [0.8, 1.0]
  near_high: [1.0, 1.2]
  separate: [1.2, 2.0]
  strong_separate: [2.0, null]
```

Use configurable quotas per bin and a maximum generation-attempt limit. Fail with diagnostics if the quota cannot be achieved instead of silently filling the bin with duplicates.

#### Acceptance tests

```python
assert pair_keys.is_unique
assert near_pair_count >= configured_minimum
assert near_unique_pair_count == near_pair_count
```

Also report unique counts by `boundary_bin` and `geometry_type`.

### 5. Correct the bootstrap confidence intervals

#### Current problem

The reported confidence interval is:

$$
CI_{95\%}(\tau)=[84\,098,84\,281].
$$

This interval is not credible given the observed timing variation. Recalculation from the raw file gives approximately:

- run-block bootstrap with the current weighting: 79,973–88,853 pixels;
- bootstrap across effective shapes: 66,879–107,590 pixels;
- per-run equal-shape estimates range roughly from 58,000 to 114,000 pixels.

The exact corrected interval may change after the other fixes. The implementation must nevertheless be repaired before rerunning.

#### Required implementation change

Implement explicit block bootstrap. Each bootstrap iteration must:

1. Draw blocks with replacement using a single RNG initialized once outside the loop.
2. Reconstruct the bootstrap dataframe from the selected blocks.
3. Refit both $K_t$ and $c_t$ from that dataframe.
4. Calculate a new $\tau_b=K_{t,b}/c_{t,b}$.
5. Store the individual bootstrap parameters.

Example structure:

```python
rng = np.random.default_rng(seed)
bootstrap_rows = []

for bootstrap_id in range(n_bootstrap):
    sampled_run_ids = rng.choice(run_ids, size=len(run_ids), replace=True)
    sample = concatenate_runs_with_new_block_ids(data, sampled_run_ids)
    fit = fit_linear(sample)

    valid = fit.intercept_s > 0 and fit.slope_s_per_pixel > 0
    bootstrap_rows.append({
        "bootstrap_id": bootstrap_id,
        "K_t_s": fit.intercept_s,
        "c_t_s_per_pixel": fit.slope_s_per_pixel,
        "tau_pixels": fit.intercept_s / fit.slope_s_per_pixel if valid else None,
        "valid": valid,
    })
```

At minimum, provide:

- run-block bootstrap;
- effective-shape bootstrap as a sensitivity analysis;
- invalid-fit fraction;
- bootstrap distributions for $K_t$, $c_t$, and $\tau$.

Do not initialize the RNG inside the bootstrap loop. Do not reuse the original fit for bootstrap samples. Do not calculate a confidence interval from repeated copies of one estimate.

#### Required outputs

```text
bootstrap_fits.csv
bootstrap_summary.json
figures/tau_bootstrap_distribution.png
```

#### Unit test

Generate synthetic timing data with known $K_t$ and $c_t$, multiple run-level offsets, and random noise. Verify that:

- bootstrap estimates are not all identical;
- the interval width increases when noise increases;
- run-block CI is wider than an incorrectly treated deterministic dataset;
- the known $\tau$ is covered in most repeated test simulations.

### 6. Counterbalance the measurement order in Experiment B

#### Current problem

The within-pair difference between execution orders is approximately:

$$
-1.08\text{ ms},
\qquad
CI_{95\%}=[-1.23,-0.92]\text{ ms}.
$$

This is material near the decision boundary. Pure randomization produced between 3 and 16 repetitions of an order per pair instead of an exact balance.

#### Required implementation change

Use an exactly counterbalanced schedule for every pair. For 20 repetitions:

```python
orders = ["separate_first", "merged_first"] * 10
rng.shuffle(orders)
```

An ABBA schedule is also acceptable. The important condition is an equal number of both orders for every pair.

Retain the `order` field and calculate:

- mean difference by order;
- within-pair order-effect estimate;
- confidence interval and p-value for the order effect;
- decision metrics both pooled and adjusted for order.

Consider a short untimed neutral invocation between alternatives only if that behavior also exists in the real pipeline. Do not hide an order effect by adding unrealistic delays.

#### Acceptance tests

```python
order_counts = raw.groupby(["pair_id", "order"]).size().unstack(fill_value=0)
assert (order_counts["separate_first"] == order_counts["merged_first"]).all()
```

The analysis must emit a warning if the absolute order effect exceeds a configurable practical threshold, for example 0.5 ms.

## P1 analysis and reporting fixes

### 7. Report both observation-level and configuration-level fit

The cost model predicts expected computation time for a configuration. Repeated timings from the same tensor shape are not independent geometric evidence.

Produce two clearly labeled analyses:

1. **Observation-level fit** using all timing observations. This quantifies runtime noise.
2. **Configuration-level fit** using the mean and median for each unique effective shape. This quantifies how well area explains expected latency across shapes.

Do not present only the higher $R^2$. The current data illustrate why both are needed:

- raw-observation linear $R^2\approx0.523$;
- effective-shape mean linear $R^2\approx0.841$;
- effective-area mean linear $R^2\approx0.875$.

Use cluster-robust uncertainty or block bootstrap where repeated observations are retained.

Add these files:

```text
experiment_a_shape_summary.csv
experiment_a_area_summary.csv
fit_observation_level.json
fit_shape_level.json
```

### 8. Diagnose the tensor-shape effect

The current data contain a large orientation effect for the same area:

```text
160×320: approximately 23.59 ms
320×160: approximately 17.92 ms
difference: approximately 5.68 ms
```

Add diagnostic models. They do not have to replace the cost-aware threshold model:

```text
M_area:       T = b0 + b1*A
M_hw:         T = b0 + b1*A + b2*H + b3*W
M_aspect:     T = b0 + b1*A + b2*log(W/H)
M_piecewise:  T = b0 + b1*A + b2*max(0, A-A0)
```

Report whether the added terms materially improve grouped-CV RMSE. Use grouped cross-validation by `(tensor_h, tensor_w)`, so repeats of one shape cannot occur in both training and validation.

Required plots:

- measured latency vs area, colored by aspect-ratio category;
- residual vs area;
- residual vs `H`;
- residual vs `W`;
- residual vs aspect ratio;
- paired comparison for transposed shapes with equal area.

Do not automatically replace the paper's linear model with a more complex model. The analysis should support one of these honest conclusions:

- area-only linear model is adequate;
- area-only model is a useful operational approximation but not a precise latency model;
- the merge policy should use a shape-aware cost function.

### 9. Correct model-comparison statistics

The current piecewise model has lower AIC than the linear model, but AIC calculated from 1,440 repeated observations may overstate the evidence because observations are clustered by run and shape.

Compare linear, quadratic, and piecewise models at both levels:

- raw observations with cluster-aware uncertainty;
- unique effective-shape means;
- grouped cross-validation by effective shape.

For the piecewise model, select the breakpoint only on the training fold during cross-validation. Count the selected breakpoint as an estimated parameter in AIC/BIC.

Report absolute and relative improvement, not only $\Delta$AIC:

```text
delta_RMSE_ms
relative_RMSE_improvement_percent
delta_MAE_ms
delta_AIC
delta_BIC
```

### 10. Expand Experiment B metrics

The current summary includes only overall accuracy, ambiguous fraction, mean regret, and pair summaries. Add:

- confusion matrix;
- accuracy with 95% pair-level bootstrap or Wilson interval;
- balanced accuracy;
- precision, recall, and F1 for `merge`;
- specificity;
- fraction of ambiguous pairs;
- metrics by `boundary_bin`;
- metrics by `geometry_type`;
- metrics on unique computational configurations;
- mean, median, p95, p99, and maximum regret;
- normalized regret;
- total-latency overhead relative to oracle;
- comparison with `always_merge` and `always_separate`.

For the current data, the analysis should be able to reproduce approximately:

```text
determinate pairs: 471
TP: 72
FP: 0
FN: 21
TN: 378
accuracy: 0.9554
balanced accuracy: 0.8871
merge precision: 1.0000
merge recall: 0.7742
merge F1: 0.8727
unique-configuration accuracy: 0.9231
mean regret: 0.198 ms
p95 regret: 0.924 ms
max regret: 8.690 ms
oracle overhead: 0.632%
```

The `merge` class is the positive class.

### 11. Preserve ambiguous decisions correctly

For each pair, calculate a confidence interval for the mean paired difference:

$$
d=T_{separate}-T_{merged}.
$$

The interval must be a confidence interval for the mean paired difference, not the 2.5% and 97.5% quantiles of the raw repetitions.

Preferred implementation:

- bootstrap repetitions within the pair, preserving the paired trial definition; or
- Student $t$ interval for the mean as a documented fallback.

Labels:

```python
if ci_low > 0:
    label = "merge_beneficial"
elif ci_high < 0:
    label = "separate_beneficial"
else:
    label = "ambiguous"
```

Store the CI method and bootstrap count in metadata.

### 12. Complete the metadata

Add:

- exact CPU model;
- physical and logical core counts;
- configured thread count;
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and relevant framework thread settings;
- Python, torch, torchvision, Ultralytics, ONNX Runtime, NumPy, SciPy, and scikit-learn versions where applicable;
- device and backend;
- weights filename and checksum;
- model precision/dtype;
- quantization status;
- batch size;
- model stride;
- warm-up procedure;
- timing-boundary definition;
- process affinity, if set;
- OS and kernel;
- governor/power mode when available;
- temperature and frequency availability;
- wall-clock start/end and valid elapsed time;
- git commit and dirty-worktree state;
- resolved configuration and seed.

The current `elapsed_s` field is zero throughout Experiment A and missing throughout Experiment B. Fix it using a monotonic clock measured from session start.

Process RSS should normally be available on Windows and Linux. If it cannot be collected, store the exception text in metadata warnings.

## P2 final experimental protocol

### 13. Run a corrected desktop smoke test

After implementing P0 and P1:

1. Run a small Experiment A with 5 repetitions per unique shape.
2. Verify unique shapes and calibration-domain coverage.
3. Run Experiment B with 30–50 unique pairs and 4 balanced repetitions.
4. Run all unit and integration tests.
5. Inspect summaries for the required fields.
6. Only then run the full desktop benchmark.

The purpose of the desktop run is to validate the code and analysis. Label it as a Windows CPU result, not a Raspberry Pi result.

### 14. Run the final benchmark on Raspberry Pi 5

The final paper-facing benchmark should use:

- Raspberry Pi 5;
- the same YOLO weights and model wrapper used by the video pipeline;
- CPU, batch size 1;
- explicit thread configuration;
- at least 50 global warm-up invocations;
- at least 30 repetitions per unique effective shape in Experiment A;
- 500–1000 unique pairs in Experiment B;
- 20 exactly counterbalanced repetitions per pair;
- at least three independent sessions.

Do not automatically change the CPU governor or system power settings. Record them. If temperature or frequency varies materially, report results per session and use session-block uncertainty.

### 15. Run `inference_only` and `detector_call` separately

Keep two independent timing modes:

- `inference_only`: prepared tensor to raw model output;
- `detector_call`: preprocessing + inference + normal postprocessing/NMS.

Never combine them in one regression. Fit and report separate:

```text
K_t_inference_only
c_t_inference_only
tau_inference_only

K_t_detector_call
c_t_detector_call
tau_detector_call
```

The primary value for the paper must use the same timing boundary as the actual greedy-merging pipeline.

## Recommended implementation sequence

1. Add the shared `PreparedInput` and detector adapter path.
2. Add shape logging assertions.
3. Refactor both collectors to use the shared invocation function.
4. Add calibration-domain planning and checks.
5. Deduplicate Experiment A configurations.
6. Deduplicate Experiment B computational pair keys.
7. Replace the near-boundary generator with quota-based unique sampling.
8. Counterbalance the Experiment B order.
9. Repair block bootstrap and save bootstrap samples.
10. Add observation-level and configuration-level analyses.
11. Add shape-effect diagnostics and grouped cross-validation.
12. Expand decision and regret metrics.
13. Complete metadata collection and elapsed-time logging.
14. Add regression tests using the current result files.
15. Run the short desktop smoke test, then the full Raspberry Pi benchmark.

## Required automated tests

### Shape and preprocessing tests

- Same requested shape produces the same actual tensor shape in A and B.
- Logged tensor dimensions equal the actual tensor dimensions.
- Effective area equals `tensor_h * tensor_w`.
- Stride assertions follow the backend configuration.
- Duplicate effective shapes are removed before timing.
- Out-of-domain pairs are rejected or trigger calibration expansion.

### Pair-generation tests

- Computational pair keys are unique.
- Every requested boundary bin reaches its quota.
- Boundary classification uses effective areas.
- Near-boundary pairs are diverse in individual and union dimensions.
- Generator terminates with a diagnostic if a quota is impossible.

### Statistical tests

- Linear fit recovers known synthetic $K_t$ and $c_t$ within tolerance.
- Bootstrap produces distinct fits.
- Bootstrap CI widens as simulated noise increases.
- Grouped CV never splits repetitions of one effective shape across folds.
- Piecewise breakpoint is selected only from training data.
- Confidence intervals for pair differences are intervals for the mean.
- Ambiguous labels are assigned when the interval contains zero.

### Timing-protocol tests

- Each pair has exactly equal counts of `separate_first` and `merged_first`.
- Timed blocks exclude input generation and file writing.
- CUDA synchronization is applied only when required.
- Resume does not duplicate completed runs or pairs.
- A resumed run preserves the original resolved configuration and seed.

### Output regression tests

Using the current attached files as a frozen fixture, the analysis code should reproduce the current point estimates within a documented numerical tolerance. This test verifies analysis compatibility only; it must not preserve known generator or bootstrap bugs.

## Final acceptance criteria

The corrected experiment is ready for a paper-facing run when all conditions below are true:

- A and B use exactly the same preprocessing, inference function, and timing boundaries.
- Every logged effective shape is verified from the actual model input.
- Experiment B contains no unmarked out-of-calibration inputs.
- Experiment A gives equal planned weight to each unique effective shape.
- Experiment B contains unique computational pair configurations.
- At least 40% of pairs are unique near-boundary cases.
- Every pair has an exactly counterbalanced execution order.
- Bootstrap intervals respond realistically to run and shape variability.
- Observation-level and configuration-level fit metrics are both reported.
- Shape effects are quantified rather than hidden in residual noise.
- Decision accuracy, balanced accuracy, class-specific metrics, ambiguity, and regret are reported.
- Metadata are sufficient to reproduce the hardware and software configuration.
- The corrected smoke test passes before the long Raspberry Pi run begins.

## Intended scientific interpretation

Do not require the corrected experiment to prove that the linear model is exact. A defensible outcome may be:

> The area-only linear model captures the mean latency trend but does not explain all variation associated with tensor geometry and system state. Despite this limitation, its analytically derived threshold provides a computationally inexpensive and conservative merge rule with low decision regret.

If the corrected results continue to show no false-positive merges but some missed beneficial merges, describe the method as conservative. If a shape-aware model materially reduces grouped-CV error and decision regret, report it as an extension or sensitivity analysis rather than silently changing the original formulation.
