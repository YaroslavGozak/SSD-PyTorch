# Відтворюване калібрування A та валідація B

## Змінені файли та результат перевірки

Нові: `reproducibility.py`, `aggregate_sessions.py`, `pair_specs.py`,
`tests/test_reproducibility.py`, `REPRODUCIBILITY.md`.

Оновлені: `collect_experiment_a.py`, `analyze_experiment_a.py`,
`collect_experiment_b.py`, `analyze_experiment_b.py`, `models.py`, `common.py`,
`run_experiments.py`, `config.voc-yolo26n.yaml`, `README.md`.

Перевірено 29 unit/integration tests: усі пройшли. Також пройшли
`python -m compileall -q experiments/cost_model_validation` і `git diff --check`.
Повний hardware/publication run із clean commit ще не виконувався.

## Потік даних

```text
config → saved A schedule → raw A + independent controls
       → per-shape statistics → primary shape-/area-level fit
       → within-shape bootstrap (linear/quadratic/piecewise)
3–5 independent A artifacts + raw CSVs → compatibility checks
       → equal-weight pooled shape values → refitted pooled models
       → hierarchical session/observation bootstrap
pooled calibration → generate pair_specs.json once
pair_specs.json + same calibration → repeated B measurements
       → order-stratified bootstrap → decision_metrics.json
```

B measurements and labels never enter calibration fitting, breakpoint search,
stability probabilities, or primary statistic/model selection. The model set is
predeclared; no automatic model selection is performed.

## Конфігурація

Повний приклад: `config.voc-yolo26n.yaml`. Основні параметри A:

```yaml
experiment_a:
  timing_mode: inference_only
  max_requested_hw: [640, 640]
  repetitions_per_effective_shape: 40
  schedule_seed: 20260913
  global_warmup_iterations: 50
  prewarm_passes: 2
  primary_shape_statistic: trimmed_mean
  trim_fraction_each_tail: 0.10
  primary_fit_level: shape_level
  breakpoint_min_support: 2
  bootstrap_seed: 0
  max_invalid_bootstrap_fraction: 0.20
  control_every_blocks: 2
  control_shapes: [[160, 160], [160, 320], [320, 320]]
  control_drift_threshold: 0.10
  fit_level_relative_difference_threshold: 0.20
  session_coefficient_cv_threshold: 0.20
```

`breakpoint_min_support` counts **unique effective areas** on each side:
`A <= B` and `A > B`. Candidate breakpoints are observed effective areas.
Piecewise replicates with nonpositive lower/upper slopes are invalid and retain
a reason. Bootstrap intervals are percentiles of fitted replicates, not a CI
for the mean of those replicates. The single-session bootstrap keeps every
shape and resamples its observations independently. Pooled bootstrap resamples
whole session identities, then observations within each selected session/shape.

Control drift is `abs(late_mean / early_mean - 1) > control_drift_threshold`.
Thirds are formed chronologically for each control shape; fewer than three
observations cannot establish this drift diagnostic. Controls are never fit
observations and never correct measurements. Their predictions are added by A
analysis, after models are fitted. Control order rotates between checkpoints.

Quality policies use `quality_gates.default: fail` and named overrides such as
`control_drift: warn`, `invalid_bootstrap: warn`, `order_balance: warn`,
`repetition_count: fail`, `block_shapes: fail`, `stride_alignment: fail`,
`invalid_piecewise: fail`. Integrity failures (corrupt hashes, incompatible
provenance, duplicate geometry, impossible quotas) always stop: a warning cannot
make a corrupt artifact usable. Thresholds and policies are retained in config
metadata. `publication_run: true` requires a clean Git working tree for A.

## Команди PowerShell із кореня репозиторію

Одна calibration session:

```powershell
python -m experiments.cost_model_validation.collect_experiment_a --config experiments/cost_model_validation/config.voc-yolo26n.yaml --output outputs/calibration/session1
python -m experiments.cost_model_validation.analyze_experiment_a --input outputs/calibration/session1/experiment_a_raw.csv --output outputs/calibration/session1 --bootstrap-count 2000
```

Повторити незалежно для `session2`, `session3` (рекомендовано 3–5 сесій).
Кожний каталог має власний session ID. Повторний запуск collector без
`--overwrite` відновлює ту саму сесію; це не нова незалежна сесія.

Об’єднання:

```powershell
python -m experiments.cost_model_validation.aggregate_sessions --sessions outputs/calibration/session1/linear_fit.json outputs/calibration/session2/linear_fit.json outputs/calibration/session3/linear_fit.json --output outputs/calibration/pooled_calibration.json --pooling-method mean --bootstrap-count 2000 --seed 0
```

The pooling method is fixed before validation (`mean` or `median`). Sessions
have equal weight per shape, regardless of repetition count. Raw CSVs must be
adjacent to their artifacts or available at their recorded paths, and must
match their recorded hashes. Optional `--pairs existing_pair_specs.json` reports
cross-session decision agreement without using B timings. Without frozen pairs
this field is `null`; cross-session prediction differences are still reported
on the calibration area grid.

For a post-generation agreement report, write aggregation output to a separate
report file. Do not replace the frozen calibration: adding report fields changes
its artifact hash even when the coefficients are unchanged.

Одноразова генерація геометрій:

```powershell
python -m experiments.cost_model_validation.pair_specs --config experiments/cost_model_validation/config.voc-yolo26n.yaml --calibration outputs/calibration/pooled_calibration.json --output outputs/calibration/pair_specs.json
```

`--regenerate-pairs` explicitly replaces an existing specification. Quotas are
fractions summing to one; primary tags use priority disagreement → piecewise →
quadratic → linear → broad. Candidate rectangles are sampled without using
measured B timings. All matched tags are retained. Each boundary primary quota
reserves half its slots per decision side (the extra slot goes to merge).
Generation fails with obtained/required counts if quotas or both sides are
unattainable. `boundary_width_ms`, `linear_boundary_width_pixels` and
`max_generation_attempts` are explicit configuration options. The default
linear width is 20% of calibration tau. A failure is not permission to silently
replace boundary/disagreement examples with easier pairs.

Повторні вимірювання B:

```powershell
python -m experiments.cost_model_validation.collect_experiment_b --config experiments/cost_model_validation/config.voc-yolo26n.yaml --fit outputs/calibration/pooled_calibration.json --pairs outputs/calibration/pair_specs.json --output outputs/validation/run1
python -m experiments.cost_model_validation.analyze_experiment_b --input outputs/validation/run1/experiment_b_raw.csv --output outputs/validation/run1
```

Для наступного B змінити лише output на `run2`. Не повторювати calibration або
генерацію пар. Metadata містить той самий ordered список pair IDs, pair hash і
calibration hash. `--overwrite` замінює B observations, але не pair specs.
The collector never invokes the generator. Calibration provenance, canvas,
stride, actual prepared tensor shapes, geometry, quotas and hashes are checked.

The paired estimator is the equally weighted mean of the two order-condition
means. CI uses independent resampling within each order condition; metadata
records count, seed and confidence. ABBA/BAAB blocks stay contiguous while blocks
across pairs are shuffled. An even repetition count is required; counts not
divisible by four end with a balanced two-trial block. Labels use strict CI
signs; zero-crossing intervals remain ambiguous.

Однокомандний запуск A → generation → B:

```powershell
python -m experiments.cost_model_validation.run_experiments --config experiments/cost_model_validation/config.voc-yolo26n.yaml --output outputs/cost_model_new --overwrite
```

Runner creates specs only if missing. After deliberately recollecting A into a
directory with old specs, add `--regenerate-pairs` to explicitly replace them.
For repeated validation against one frozen calibration use the B commands above.

## Schemas та artifacts

`experiment_a_schedule.json`:

```json
{"settings_hash":"sha256", "hash":"sha256", "items":[
  {"global_position":0,"block_index":0,"position_in_block":0,
   "shape_id":"h160_w320","tensor_h":160,"tensor_w":320,"effective_area":51200}
]}
```

Raw A CSV retains legacy timing columns and adds `shape_id`, `block_index`,
`position_in_block`, `global_position`, `latency_s`. `metadata.json` is written
before measurement, with session/provenance/schedule and warmup schedule.
`experiment_a_controls.json` checkpoints controls independently of raw CSV.

`linear_fit.json` is a backward-compatible filename for the **three-model**
schema-v3 calibration. It includes `primary_fit_selector`, `alternative_fits`,
`latency_models`, `shape_statistics`, envelope, controls, raw reference/hash,
config, warnings, provenance and bootstrap. `bootstrap_models.json` contains:

```json
{"method":"fixed_design_within_shape", "seed":0, "count":2000,
 "confidence_level":0.95, "replicates":[
   {"piecewise":{"valid":true,"reason":null,
     "coefficients":{"b0":0.01,"b1":0.000001,"b2":0.000002},
     "breakpoint_area":51200}}
 ], "summaries":{"piecewise":{"invalid_fraction":0.0,
     "parameters":{"breakpoint_area":{"ci":[51200,61440],"median":51200,"mad":0}},
     "breakpoint_frequency":{"51200.0":1800,"61440.0":200}}}}
```

The example omits the other models/metrics for brevity. Shape CSV includes mean,
median, trimmed mean, sample SD, SE, CV, quantiles, IQR outlier count and first/last
position. Area multiplicity and all mean/robust shape-/area-level fits remain in
the calibration. Observation-level fits remain diagnostics.

`pair_specs.json` wraps `{ "payload": {...}, "sha256": "..." }`. The digest
covers canonical sorted-key JSON of `payload` (excluding its enclosing digest).
Each pair records coordinates, requested/actual shapes, areas/delta, domain,
computational key, primary/all tags, generator seed, calibration hash, predicted
gains and per-model bootstrap stability:

```json
{"merge_probability":0.97,"predicted_gain_ms_ci":[0.1,0.8],
 "decision_stable":true,"valid_model_count":1980}
```

Stability thresholds default to <=0.10 / >=0.90 and are configurable under
`decision_stability`. No observed B label is involved. B raw CSV adds
`order_block_id`, `position_in_order_block`, `primary_stratum`, `stratum_tags`.
`decision_metrics.json` includes order-aware estimates/CIs, stability, overall
and stratum/geometry/area-regime metrics and all/determinate regret.

## Migration та перевірка

Historical CSV analysis remains supported, with missing provenance reported
explicitly. Legacy A collections without saved schedules cannot be resumed as
schema v3; start a new output directory. B collection now requires generated
specs; the old `boundary_quotas`/`minimum_near_pairs` are superseded by
`strata_quotas`, and `--tau-override` is rejected for frozen validation.
Existing datasets are not rewritten by this change. Multi-session aggregation
requires schema-v3 artifacts and raw observations. Calibration hash changes
require an explicit new pair-generation operation.

```powershell
python -m unittest discover -s experiments/cost_model_validation/tests -v
```

Tests cover schedule determinism/counts, interrupted resume, stride/domain,
robust statistics, fixed-design bootstrap, invalid slopes/support, merge
probability, incompatible sessions, pooled refitting rather than coefficient
averaging, pair hash/duplicates, quota failures, order stratification and a
deterministic end-to-end three-session/two-validation-run smoke test.
The smoke test injects synthetic timings; it does not claim hardware accuracy.
A real publication run must be performed after committing changes and setting
`publication_run: true`; no real YOLO benchmark is run by the unit tests.

## Rare-boundary generation fix

Pair generation first draws up to `random_generation_attempts` random candidates
(default 10000, capped by `max_generation_attempts`). It then searches the legal
stride-aligned tensor grid for missing boundary and disagreement slots. Broad
coverage remains random; if its quota is short, the remaining random budget is
used. `tensor_grid_search: false` disables this extra search. Neither widths,
quotas, priority nor side balance change. The frozen artifact records
`generation_method`. Existing frozen files remain valid and are not rewritten.

The grid constructs real rectangles at opposite corners of each feasible union.
This deliberately targets computational coverage, rather than claiming uniformly
random geometry for boundary/disagreement strata. Failed searches now report
accepted merge/separate counts for each boundary.
