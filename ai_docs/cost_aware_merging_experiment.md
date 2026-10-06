# Experiment: Cost-Aware Segment Merging Validation

## Goal

Implement a controlled experiment for validating a method of merging image segments based on the **predicted computational cost of neural-network processing**.

The experiment must be added to the existing `experiments/` directory and should reuse the repository's existing infrastructure as much as possible.

Do **not** create a separate standalone inference framework if equivalent functionality already exists in the repository.

Before implementing anything, inspect the repository and determine:

- how experiments in `experiments/` are structured;
- how YAML configuration files are loaded;
- how datasets are instantiated;
- how ImageNet-VID frames are loaded;
- how ROI-SSD and YOLO models are instantiated;
- how model checkpoints are loaded;
- how variable-size image segments are passed to each model;
- how inference latency is currently measured;
- whether profiling utilities already exist;
- whether segment/ROI merging utilities already exist;
- whether simple/geometric merging is already implemented;
- whether fixed-size / fixed-padding experiments already contain reusable code;
- whether there are existing utilities for CSV/JSON output, plotting, reproducibility, device selection, logging, etc.

Examples of existing configurations include:

```text
imagenet-vid-roissd.yaml
imagenet-vid-yolo.yaml

These names are examples only. Do not hardcode assumptions about their location or schema. Inspect the repository and use the existing configuration mechanism.
The resulting implementation should fit naturally into the existing project architecture.
1. Research question
The experiment evaluates the following general merging rule.
For two candidate image segments
\[
R_1,\;R_2,
\]
construct the smallest bounding rectangle containing both:
\[
R_{12}=\operatorname{BBox}(R_1\cup R_2).
\]
Let
\[
\hat T(R)
\]
denote an estimate of the neural-network processing cost of segment \(R\).
The proposed cost-aware method merges the segments if
\[
\hat T(R_{12})
<
\hat T(R_1)+\hat T(R_2).
\]
The main purpose of the experiment is not to validate only one particular formula for \(\hat T\).
The main purpose is to validate the general principle:
merge segments when the predicted cost of processing the merged segment is lower than the predicted cost of processing the segments separately.

Different cost estimators must therefore be compared.
2. Important methodological distinction
Keep the following concepts separate in the implementation and in the results.
Merging method
The general decision rule is:
\[
\hat T(R_{12})
<
\hat T(R_1)+\hat T(R_2).
\]
Cost estimator
The function
\[
\hat T(R)
\]
can be implemented in different ways.
At minimum, implement:
1. affine area-based cost model;
2. shape-based lookup table.
The code should make it reasonably easy to add other cost estimators later.
Do not tightly couple the merging algorithm to the affine model.
A suitable abstraction could be something conceptually equivalent to:
estimated_cost = cost_estimator(width, height, ...)


but follow the architectural conventions already used in the repository rather than introducing unnecessary abstractions.
3. Target hardware
The primary experiment is intended for:
- x86 CPU;
- Raspberry Pi 5 CPU.
GPU benchmarking is not required for this experiment.
The reason is methodological: on the previously tested GPUs the estimated threshold
\[
\tau=\frac{K}{c}
\]
can be comparable to or larger than the entire usable input image area. In that case it is difficult to generate meaningful configurations on both sides of the merging boundary.
CPU and Raspberry Pi measurements are therefore more informative for this experiment.
The implementation must not depend on CUDA.
It should be possible to run the same experiment code on a regular development machine and on Raspberry Pi.
Reuse the project's existing device/model configuration mechanism.
4. Dataset usage
Use the dataset specified by the existing YAML configuration.
For example:
imagenet-vid-roissd.yaml
imagenet-vid-yolo.yaml

The experiment must not contain hardcoded ImageNet-VID paths.
The dataset configuration already present in the repository is the source of truth.
The experiment should obtain real image frames through the existing dataset/data-loading infrastructure.
Ground-truth object annotations are not required for the primary experiment.
The dataset is used as a source of real image content.
Candidate segments are generated synthetically on top of real frames.
Therefore, segments are allowed to contain:
- an object;
- part of an object;
- several objects;
- background only.
This is intentional.
The experiment measures computational cost rather than localization accuracy.
5. Synthetic segment generation
For every selected real frame, generate controlled pairs
\[
(R_1,R_2).
\]
Vary at least:
- segment width;
- segment height;
- horizontal displacement;
- vertical displacement;
- diagonal displacement;
- overlap;
- aspect ratio;
- relative sizes of the two segments.
All generated segments must remain inside the image boundaries.
If a requested configuration cannot fit inside a frame, either generate another valid placement or skip it and record the reason.
Use a deterministic random seed.
6. Segment sizes
The generated test set must explicitly include segments having a side of 32 pixels.
Use the following values as the default candidate side lengths:
32
64
96
128
160
192
256
320

Do not assume that every Cartesian combination must be benchmarked if that would make the experiment unnecessarily large.
However, the experiment must include representative shapes such as:
32 x 32
32 x 64
64 x 32
32 x 96
96 x 32
32 x 128
128 x 32
32 x 256
256 x 32

64 x 64
64 x 128
128 x 64

96 x 96
96 x 192
192 x 96

128 x 128
128 x 256
256 x 128

160 x 160
192 x 192
256 x 256
320 x 320

The exact grid should be configurable.
Do not assume square segments.
7. Important check for model preprocessing
Before trusting any timing results, verify what spatial dimensions are actually processed by the neural network.
This is especially important for models that may internally resize or letterbox their inputs.
For every benchmarked segment, distinguish between:
requested crop shape
actual model input shape

If the existing inference pipeline always resizes every crop to the same fixed input resolution, then raw crop size is not a meaningful predictor of neural-network processing cost.
In that case:
1. inspect how previous variable-size experiments in this repository were implemented;
2. reuse the correct variable-input inference path if one exists;
3. record the actual tensor shape processed by the network;
4. do not silently report crop dimensions as network dimensions.
ROI-SSD and YOLO may have different preprocessing/inference paths. Reuse the appropriate existing implementation for each model.
8. Spatial configurations
Generate several classes of segment pairs.
Horizontal
Example:
+---------+          +---------+
|   R1    | <- gap ->|   R2    |
+---------+          +---------+

Vertical
+---------+
|   R1    |
+---------+
     |
    gap
     |
+---------+
|   R2    |
+---------+

Diagonal
+---------+
|   R1    |
+---------+

                 +---------+
                 |   R2    |
                 +---------+

Diagonal configurations are particularly important because the bounding rectangle may contain a large amount of empty space.
Overlapping
Include several overlap levels.
The generator should cover cases ranging from:
strong overlap
small overlap
touching
small gap
medium gap
large gap

Prefer a parameterized generator instead of hardcoded coordinates.
9. Geometric quantities to record
For every pair record at least:
\[
A_1=A(R_1),
\]
\[
A_2=A(R_2),
\]
\[
A_{12}=A(R_{12}),
\]
and
\[
A_{\text{extra}}
=
A(R_{12})-A(R_1)-A(R_2).
\]
Also record:
- widths and heights;
- coordinates;
- horizontal gap/displacement;
- vertical gap/displacement;
- IoU between the original segments;
- aspect ratios;
- frame dimensions.
For overlapping segments, A_extra can be negative. This is valid.
10. Actual processing cost
For every pair measure actual latency for:
Separate processing
\[
T_{\mathrm{sep}}
=
T(R_1)+T(R_2).
\]
Merged processing
\[
T_{\mathrm{merge}}
=
T(R_{12}).
\]
Define:
\[
\Delta T_{\mathrm{actual}}
=
T_{\mathrm{sep}}-T_{\mathrm{merge}}.
\]
The actual optimal decision is:
\[
M^*=
\begin{cases}
\text{merge}, & T_{\mathrm{merge}}<T_{\mathrm{sep}},\\
\text{separate}, & \text{otherwise}.
\end{cases}
\]
This actual measured decision acts as the oracle for evaluating other merging criteria.
The oracle is an evaluation reference, not a deployable algorithm.
11. Timing methodology
Reuse existing benchmark/timing utilities if available.
Do not implement a second timing framework unless necessary.
At minimum support configurable:
warmup iterations
measurement repetitions
number of frames
random seed

Reasonable defaults can be approximately:
warmup: 30-100
repetitions: 30-50

but inspect existing repository conventions first.
For CPU timing, use a high-resolution monotonic timer or the timing utility already used by the project.
Dataset loading from disk should normally not be included in neural-network inference latency.
The primary metric should represent the model-processing cost used by the merging decision.
If preprocessing/postprocessing is already considered part of inference in existing project benchmarks, preserve that convention and document it.
If practical, report both:
model-only latency
end-to-end segment inference latency

but do not duplicate large parts of the inference pipeline just to obtain both.
For Raspberry Pi, benchmark duration and thermal throttling can matter. Avoid a benchmark order that systematically favors one strategy.
Prefer randomized or interleaved execution order where practical.
Record enough platform information to reproduce the experiment.
12. Cost estimator A: affine area model
Implement or reuse:
\[
\hat T_{\mathrm{affine}}(R)
=
K+cA(R).
\]
Estimate \(K\) and \(c\) from profiling measurements.
Do not hardcode previously measured values.
Fit the parameters on the current:
model + hardware + execution mode

combination.
The corresponding merging decision is:
\[
\hat T_{\mathrm{affine}}(R_{12})
<
\hat T_{\mathrm{affine}}(R_1)
+
\hat T_{\mathrm{affine}}(R_2).
\]
Also calculate:
\[
\tau=\frac{K}{c}.
\]
For analysis, the equivalent condition is:
\[
A(R_{12})
<
A(R_1)+A(R_2)+\tau.
\]
or
\[
A_{\mathrm{extra}}<\tau.
\]
Record:
K
c
tau
R^2
MAE
RMSE

for the fitted model where appropriate.
13. Cost estimator B: shape lookup
Implement a cost estimator based on empirical input shape:
\[
\hat T_{\mathrm{shape}}(R)
=
LUT(w_R,h_R).
\]
The lookup table should be created by profiling the neural model on representative input shapes.
The important property is that it distinguishes shapes having equal area but different geometry.
For example:
\[
32\times256
\]
and
\[
64\times128
\]
have equal area:
\[
8192\;\text{pixels},
\]
but they may have different actual processing latency.
The shape lookup experiment should determine whether accounting for this difference improves merging decisions.
Before implementing a new profiler or lookup-table abstraction, inspect the repository for existing fixed-size experiments and profiling code. Reuse them if possible.
If the exact requested shape is not present in the lookup table, use a simple documented strategy such as:
- nearest measured shape;
- interpolation;
- another method already present in the repository.
Do not introduce a complicated machine-learning latency predictor for this experiment unless the repository already contains one.
14. Avoid evaluation leakage
Do not construct a shape lookup table from the exact same timing samples that are subsequently treated as unseen evaluation measurements.
Separate conceptually:
profiling/calibration
evaluation

The profiling stage produces:
K, c
shape lookup table

The evaluation stage tests merging decisions on generated segment pairs.
This separation may use:
- separate frames;
- separate timing runs;
- separate shape configurations where appropriate.
The exact implementation can follow the existing experiment infrastructure.
The purpose is to avoid making the shape lookup method equivalent to the oracle.
15. Methods to compare
At minimum compare the following.
A. No merge
Always process:
\[
R_1,\;R_2
\]
separately.
This is a latency baseline, not a merging criterion.
B. Existing simple/geometric merging
Find the geometric/simple merging strategy already implemented in the repository and reuse it.
Do not invent a replacement if the project already has the method used in previous experiments.
Record exactly which criterion and threshold it uses.
C. Geometric area-inflation baseline
If no equivalent baseline already exists, implement a simple configurable geometric criterion based on relative area growth, for example:
\[
\frac{A(R_{12})}
{A(R_1)+A(R_2)}
<
\gamma.
\]
The threshold \(\gamma\) must be configurable.
If an equivalent criterion already exists in the repository, reuse it instead.
This baseline is useful because it uses geometry but has no knowledge of measured neural-network latency.
D. Cost-aware affine
Use:
\[
\hat T(R)=K+cA(R).
\]
Decision:
\[
\hat T(R_{12})
<
\hat T(R_1)+\hat T(R_2).
\]
E. Cost-aware shape lookup
Use:
\[
\hat T(R)=LUT(w_R,h_R).
\]
Use the same general cost-aware merging rule.
F. Oracle
Use actual measured latency:
\[
T(R_{12})
<
T(R_1)+T(R_2).
\]
The oracle is used only for evaluation.
16. Do not confuse shape lookup with a different merging method
Both
cost-aware affine

and
cost-aware shape lookup

implement the same general cost-aware merging method.
They differ only in how they estimate
\[
\hat T(R).
\]
The experiment should make this distinction explicit in code, result names, plots, and documentation.
17. Primary evaluation metric
The most important metric is whether a criterion makes the same merge/separate decision as the measured oracle.
For each pair:
actual_decision
predicted_decision_affine
predicted_decision_shape
predicted_decision_geometric
...

Calculate decision accuracy:
\[
Accuracy_{\mathrm{decision}}
=
\frac{\text{correct decisions}}
{\text{all evaluated pairs}}.
\]
Also calculate a confusion matrix:
	Actually merge	Actually separate
Predict merge	TP	FP
Predict separate	FN	TN


Pay particular attention to:
- false positive: merging was predicted to be beneficial but actually increased latency;
- false negative: separate processing was selected even though merging would have reduced latency.
18. Effective latency metric
Decision accuracy alone is not sufficient.
Also calculate the actual processing latency that would result from following each policy.
For policy \(P\):
\[
T_P=
\begin{cases}
T_{\mathrm{merge}}, & P(R_1,R_2)=\text{merge},\\
T_{\mathrm{sep}}, & P(R_1,R_2)=\text{separate}.
\end{cases}
\]
Aggregate over the evaluation set.
Report at least:
mean latency
median latency
total accumulated latency
relative difference from no-merge
relative difference from oracle

This is important because two policies may have similar decision accuracy but very different costs for their mistakes.
19. Affine-model boundary analysis
For the affine estimator separately analyze the theoretical boundary:
\[
A_{\mathrm{extra}}=\tau.
\]
Generate enough cases in three regions:
\[
A_{\mathrm{extra}}<\tau,
\]
\[
A_{\mathrm{extra}}\approx\tau,
\]
\[
A_{\mathrm{extra}}>\tau.
\]
Do not sample only uniformly over the entire geometry space.
Ensure sufficient density near the decision boundary.
A useful configurable interval is:
\[
A_{\mathrm{extra}}
\in
[0.5\tau,\;1.5\tau],
\]
with additional samples near:
\[
[0.8\tau,\;1.2\tau].
\]
Because \(\tau\) is platform-dependent, generate or select these configurations after profiling the current platform.
20. Cross-platform experiment
The same code should be runnable independently on:
desktop/laptop CPU
Raspberry Pi 5 CPU

Do not require both platforms in one execution.
Each machine can generate its own result directory.
The output must contain enough metadata to compare results later.
A particularly important research question is whether the same geometric pair:
\[
(R_1,R_2)
\]
can have different optimal decisions on different hardware.
For example:
CPU A: merge
Raspberry Pi: separate

or vice versa.
If practical, support exporting the generated pair definitions separately from the timing results so that exactly the same synthetic geometry can be replayed on another platform.
For example:
pairs.json

or
pairs.csv

This is preferable to independently generating unrelated pairs on each machine.
21. Reproducibility
A run should store at least:
resolved configuration
original config path
model identifier
checkpoint identifier/path if appropriate
dataset/split
device
platform information
random seed
timestamp
number of frames
number of generated pairs
warmup count
repetition count
profiling parameters
K
c
tau
shape lookup data

Follow existing repository conventions for run directories and metadata if they exist.
22. Raw result format
Prefer CSV for pair-level results unless the repository already uses another tabular format.
Each evaluated pair should contain fields conceptually equivalent to:
frame_id
pair_id

frame_width
frame_height

r1_x1
r1_y1
r1_x2
r1_y2
r1_width
r1_height
r1_area

r2_x1
r2_y1
r2_x2
r2_y2
r2_width
r2_height
r2_area

merged_x1
merged_y1
merged_x2
merged_y2
merged_width
merged_height
merged_area

area_extra
gap_x
gap_y
iou

actual_r1_ms
actual_r2_ms
actual_separate_ms
actual_merged_ms
actual_delta_ms

affine_r1_ms
affine_r2_ms
affine_merged_ms
affine_delta_ms

shape_r1_ms
shape_r2_ms
shape_merged_ms
shape_delta_ms

oracle_decision
affine_decision
shape_decision
geometric_decision
simple_merge_decision

affine_correct
shape_correct
geometric_correct
simple_merge_correct

Adapt names to repository style.
Do not add meaningless duplicate columns merely to match this example.
23. Summary output
Generate a compact summary table containing at least:
method
decision_accuracy
TP
FP
FN
TN
mean_effective_latency_ms
median_effective_latency_ms
total_effective_latency_ms
relative_to_no_merge_percent
relative_to_oracle_percent

Methods should include:
no_merge
existing_simple_merge
geometric_area
cost_affine
cost_shape_lookup
oracle

where applicable.
24. Plots
Reuse existing plotting infrastructure if available.
Generate at least the following plots.
Actual latency difference vs extra area
X:
\[
A_{\mathrm{extra}}
\]
Y:
\[
\Delta T_{\mathrm{actual}}
=
T_{\mathrm{sep}}-T_{\mathrm{merge}}.
\]
For the affine model mark:
\[
A_{\mathrm{extra}}=\tau.
\]
Also mark:
\[
\Delta T=0.
\]
This is one of the most important figures.
Predicted vs actual latency difference
For each cost estimator compare:
\[
\Delta T_{\mathrm{pred}}
\]
with
\[
\Delta T_{\mathrm{actual}}.
\]
Decision accuracy by method
Compare:
geometric
simple merge
cost-aware affine
cost-aware shape lookup

against the oracle.
Effective latency by method
Compare the actual latency that would result from following each policy.
Shape latency map
For shape lookup, produce a heatmap:
X = width
Y = height
color = measured latency

This should make equal-area / different-shape behavior visible.
25. Equal-area shape analysis
Explicitly include analysis of shapes with equal or approximately equal area but different aspect ratios.
Examples:
\[
32\times256
\quad\text{vs}\quad
64\times128,
\]
and other equivalent pairs available in the configured shape grid.
Compare actual latency:
\[
T(w_1,h_1)
\]
and
\[
T(w_2,h_2).
\]
This analysis answers whether area alone is sufficient for estimating processing cost.
If differences are negligible, report that result.
Do not assume shape lookup must outperform the affine model.
26. Expected experiment workflow
The intended high-level workflow is:
load existing YAML config
        |
        v
reuse dataset loader
        |
        v
reuse model/inference implementation
        |
        v
select real frames
        |
        +-------------------------+
        |                         |
        v                         v
profile model shapes       generate synthetic pairs
        |                         |
        v                         |
fit affine model                  |
build shape LUT                   |
        |                         |
        +------------+------------+
                     |
                     v
        benchmark R1, R2, R12
                     |
                     v
            determine oracle
                     |
                     v
       evaluate merging policies
                     |
                     v
           CSV + summary + plots

27. Configuration
Do not hardcode experiment parameters throughout the Python code.
Follow the project's existing configuration approach.
Ideally the experiment should be runnable using an existing model/dataset config plus experiment-specific parameters.
For example, conceptually:
python experiments/<experiment_script>.py \
    --config <existing-yaml-config>

Additional CLI/config options may control:
number of frames
number of pairs
seed
warmup
repetitions
shape grid
gap grid
output directory
profiling/evaluation split
enabled baselines

However, inspect existing CLI/config conventions before deciding the exact interface.
If the repository normally uses config inheritance, config sections, command-line overrides, Hydra, argparse, or another mechanism, follow that convention.
Do not introduce a second configuration system.
28. ROI-SSD and YOLO support
The implementation should preferably work with both existing configurations such as:
imagenet-vid-roissd.yaml
imagenet-vid-yolo.yaml

but do not duplicate the experiment code for each model.
Reuse the repository's existing model abstraction if possible.
If ROI-SSD and YOLO currently expose incompatible inference APIs, implement only the smallest adapter necessary.
Before adding adapters, inspect previous experiments that already benchmark both models. They are likely the best source of reusable code.
Do not refactor unrelated parts of the repository merely to make the APIs aesthetically identical.
29. Scope: pairwise validation first
The primary experiment is deliberately pairwise.
It validates the fundamental decision:
\[
R_1,R_2
\rightarrow
\{\text{merge},\text{separate}\}.
\]
Do not immediately turn this into a large end-to-end tracking experiment.
Do not require object annotations.
Do not require mAP evaluation.
Do not require SORT/Kalman tracking.
Do not require full video pipeline execution.
Those experiments already answer different questions.
The purpose here is to isolate the merging decision and determine whether computational-cost-aware merging predicts actual runtime better than purely geometric merging.
30. Optional extension: multi-segment greedy merging
Only after the pairwise experiment works correctly, consider adding an optional mode that evaluates the existing greedy multi-segment merging algorithm.
Do this only if the repository already contains reusable greedy merging code or if the extension is trivial.
The pairwise experiment is the priority.
Do not delay completion of the primary experiment for this extension.
31. Tests
Add lightweight tests for logic that can be tested without loading a neural network.
At minimum test:
- bounding rectangle calculation;
- area calculation;
- A_extra;
- synthetic pair generation;
- image-boundary handling;
- affine merge decision;
- shape lookup decision;
- oracle decision;
- deterministic generation with fixed seed.
Reuse the project's test framework and conventions.
Do not create heavyweight tests requiring ImageNet-VID or model checkpoints unless such integration-test infrastructure already exists.
32. Sanity checks
The experiment should fail loudly or warn when:
- no valid pairs were generated;
- requested crop sizes exceed frame dimensions;
- actual model input shape differs unexpectedly from requested shape;
- timing measurements are zero/invalid;
- affine fitting produces physically nonsensical parameters;
- \(c\le0\);
- \(\tau\) cannot be calculated;
- shape lookup coverage is insufficient;
- evaluation accidentally uses calibration timing as oracle timing;
- the model is silently running on a different device than requested.
Do not silently continue with invalid profiling results.
33. Performance considerations
This experiment can require many inference calls, especially on Raspberry Pi.
Avoid unnecessary repeated work.
Examples:
- load model once;
- load each frame only as often as necessary;
- cache generated crop coordinates;
- cache profiling results;
- allow resuming an interrupted experiment if consistent with existing project infrastructure;
- flush raw results periodically so a long Raspberry Pi run is not lost;
- do not retain unnecessary model outputs if only latency is required.
However, do not optimize the benchmark in a way that changes the actual inference path being measured.
34. Scientific integrity
The implementation must not be designed to make cost-aware merging look better.
In particular:
- do not tune thresholds on the evaluation pairs;
- do not use evaluation latency directly inside affine or shape predictions;
- do not discard cases where cost-aware merging fails;
- do not select only favorable segment geometries;
- do not assume the affine model is correct;
- do not assume shape lookup is better;
- do not assume geometric merging is worse;
- preserve raw measurements.
Unexpected results are valid results.
35. Expected scientific questions
The final outputs should allow us to answer:
1. Does computational-cost-aware merging predict the actually faster merge/separate decision?
2. Is it more reliable than the existing geometric/simple merging strategy?
3. How often does the affine model
\[
K+cA
\]
make the correct decision?
4. Does a shape-based latency lookup improve the decision?
5. Is image area alone sufficient to estimate processing latency?
6. Do equal-area segments with different aspect ratios have measurably different latency?
7. Does the theoretical affine boundary
\[
A_{\mathrm{extra}}=\tau
\]
correspond to the empirical merge/separate boundary?
8. Does the optimal decision change between x86 CPU and Raspberry Pi?
9. How much actual latency is lost by incorrect merge/separate decisions?
10. How close do practical estimators get to the oracle?
36. Deliverables
When implementation is complete, provide:
1. experiment code under experiments/;
2. any minimal reusable helper code required;
3. experiment-specific config additions only if necessary;
4. lightweight tests;
5. a short README or docstring explaining how to run the experiment;
6. example commands using at least:
   - an ROI-SSD ImageNet-VID config;
   - a YOLO ImageNet-VID config, if supported by the existing code;
7. description of generated output files;
8. summary of which existing repository components were reused;
9. list of any assumptions that had to be made.
Do not commit generated benchmark results unless that is already the repository convention.
37. Implementation strategy for the coding agent
The repository is the source of truth for implementation details.
If any instruction in this document conflicts with the actual architecture of the repository:
1. inspect existing experiments and utilities;
2. preserve existing project conventions;
3. reuse working code rather than duplicating it;
4. make the smallest reasonable extension;
5. document the deviation.
Do not guess APIs, paths, class names, config schemas, model wrappers, dataset loaders, or checkpoint formats from this document.
Find them in the repository.
In particular, inspect existing code related to:
fixed-size experiments
fixed-padding experiments
ROI/segment experiments
inference benchmarking
ROI-SSD inference
YOLO inference
ImageNet-VID loading
segment merging
hardware profiling
cost estimation

These existing implementations should be preferred over new code.
38. Definition of done
The task is complete when one experiment command can:
1. load an existing repository configuration;
2. instantiate the configured dataset and model;
3. profile processing cost on the current hardware;
4. fit the affine estimator;
5. build the shape lookup estimator;
6. generate reproducible synthetic segment pairs on real dataset frames;
7. measure separate and merged processing latency;
8. determine the measured oracle decision;
9. evaluate the available geometric and cost-aware merging policies;
10. save raw measurements;
11. generate summary metrics;
12. generate the core plots;
13. run without CUDA on x86 CPU;
14. be runnable with the same code on Raspberry Pi 5 CPU.
Prioritize correctness, reproducibility, and reuse of the existing repository over creating a large new experiment framework.