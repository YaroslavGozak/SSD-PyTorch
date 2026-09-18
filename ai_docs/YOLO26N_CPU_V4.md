# YOLO26n CPU: shape-aware calibration та незалежна validation

## Причина змін

Area-only fit не враховує різницю між `(H,W)` та `(W,H)`. Merge gain — різниця
трьох прогнозів, тому менша абсолютна RMSE не гарантує правильного знаку gain.
Крім того, попередній equal-strata набір був challenge distribution, а його pooled
accuracy не оцінювала deployment prevalence. Попередній hardware-shift flag
порівнював controls із linear prediction і змішував похибку моделі зі зміною CPU.

`linear_tau` залишається default calibrated baseline. Area linear/quadratic/
piecewise збережено; ніякого вибору policy за їх R²/AIC/BIC або старими B labels
немає. `shape_lookup` та `shape_lookup_conservative` — кандидати для незалежного
порівняння. `conservative_consensus` має `exploratory=true`.

## Canonical shape path і lookup

Єдина формула — `geometry.stride_rounded_shape(H,W,stride)`. Її використовують
fake, Ultralytics і ROI-SSD adapters, calibration grid, pair generator та
calibrated runtime policy. Requested rectangles спочатку об’єднуються; лише
потім округлюються H і W та обчислюється effective area. Measurement перевіряє
фактичний tensor shape. Lookup keys ordered: `160x320` та `320x160` різні.

Поточний YAML описує повний domain: сторони 128..640, stride 32, площа
25600..409600, W/H 0.5..2.0. Це **200 shapes**, з deterministic grid hash.
`shape_design: full_grid` вимагає coverage 200/200; `strict_lookup: true` у B
забороняє missing entry і fallback. Publication mode також вимагає lookup.

Shape point estimate — predeclared trimmed mean. SE та CI отримуються
bootstrap-ом саме цієї статистики всередині shape. Conservative gain:

```text
gain = lookup(H1,W1) + lookup(H2,W2) - lookup(Hm,Wm)
se_bound = se1 + se2 + sem
gain_lcb = gain - NormalQuantile(confidence_level) * se_bound
merge iff gain_lcb > decision_margin_s
```

Сума SE — верхня межа SD для довільних кореляцій. Не робиться необґрунтоване
припущення про незалежність shapes. Normal LCB є наближенням, а не finite-sample
або sequentially valid гарантією. У поточному predeclared config confidence=0.95,
додатковий margin=0 s. Margin не підбирався на виміряних 500 парах. За відсутності
SE conservative rule повертає separate. Bootstrap не виконується в runtime.

`piecewise_area_limit(a1,a2,model)` — helper для умовної межі area; він не створює
універсальний tau. Усі завантажені area models перевіряються на монотонність;
для quadratic перевіряється derivative на обох кінцях domain.

## Заморожений дизайн B

Representative config: `config.voc-yolo26n.representative.yaml`. Це явно
`synthetic_reference_distribution`, не production trace: дві requested ROI
мають незалежні рівномірні цілочислові H/W у canvas та рівномірні допустимі
позиції. Distribution умовна на domain acceptance, унікальний computational key
та виключення prior keys. Межі моделей не балансуються. Scope:
`reference_distribution`. Primary outputs — mean/p95 regret та expected latency
gain проти separate.

Challenge config: `config.voc-yolo26n.yaml`. Збережені п’ять strata, priority,
квоти, обидві сторони boundary і targeted tensor-grid search. Scope:
`unweighted_challenge_average`. Per-stratum diagnostics показують requested/
generated/determinate/ambiguous counts, actual classes, predictions та rejection
reasons. Generation attempts рахують concrete candidates після grid prefilter.

`aggregation_weights` необов’язкові та зафіксовані в pair declaration:

```yaml
experiment_b:
  aggregation_weights:
    source: external_reference_trace_v1
    derived_from_labels: false
    strata:
      broad_random: 0.6
      linear_boundary: 0.1
      piecewise_boundary: 0.1
      quadratic_boundary: 0.1
      model_disagreement: 0.1
```

Ваги мають покривати всі представлені strata, бути невід’ємними й сумуватися до
1. JSON зберігає source та weights. Код не оцінює їх із labels; походження source
залишається відповідальністю автора external distribution.

Fixed repetitions: representative/broad — 20; challenge boundary/disagreement —
40. ABBA/BAAB blocks залишаються цілими; bootstrap окремо resamples дві order
conditions. Немає adaptive stopping за звичайним 95% CI. Statistical label і
`practical_label` розділені. `minimum_worthwhile_gain_ms` явно заданий до run;
поточний neutral default 0, щоб не вводити емпіричну 0.5/1 ms з попередніх B.

Pair specs v2 містять policy/design declaration і hash. Config змінити між
generation та confirmatory measurement не можна. `prior_pair_specs` задає старі
артефакти: їх hashes перевіряються, seed має бути іншим, їх computational keys
виключаються. При measurement створюється `.usage.json` поруч із pair file;
повторне використання, policy-development flag або legacy pair file потребують
`--allow-reused-pairs` і отримують warning та `confirmatory=false`. Для переносного
reuse guard слід переносити pair file разом із його usage sidecar. Це захист від
випадкового reuse, не зовнішній registry проти навмисного видалення історії.

## Control diagnostics

`within_session_drift`: early/middle/late та early-to-late relative change
окремо для кожного control shape. `calibration_to_validation_shift`: observed A
control distribution проти B mean/median, absolute/relative shift. Hardware flag
походить від другого порівняння, не від residual area model. Residuals усіх
area models і lookup зберігаються окремо. За відсутності observed baseline
позначається `baseline_available=false`; baseline не вигадується з прогнозу.

Пороги `within_session_drift_threshold` і `calibration_shift_threshold` незалежні.
Warnings дедуплікуються. `near_pair_count` видалено у schema v4;
`boundary_pair_count` рахує пари, у яких **будь-який matched tag** закінчується
на `_boundary`, включно з primary disagreement.

## Artifacts v4 та migration

```text
calibration_grid.json + experiment_a_schedule.json + metadata.json
  → experiment_a_raw.csv + experiment_a_controls.json
  → bootstrap_models.json                  # sole full replicate artifact
  → shape_statistics.json + control_records.json
  → linear_fit.json                        # compact calibration + references
  → experiment_a_summary.json              # diagnostics, no replicates
  → representative_pair_specs.json / challenge_pair_specs.json
  → each B: raw CSV, pair_specs.json, candidate_predictions.json,
             validation_declaration.json, experiment_b_metadata.json
  → each B: decision_metrics.json
```

`linear_fit.json` містить ordered lookup table, area baselines, envelope,
shape policy (`numeric_dtype` окремо від stride alignment), primary selector,
policy declaration, provenance та hash references. Area model keys залишені
`linear`, `quadratic`, `piecewise` для сумісності; додано `shape_lookup`.
`bootstrap_reference` має `path` і `sha256`. Full replicates не embed-яться у
calibration/summary/B metadata. B отримує локальну compact calibration copy та
маленькі statistics/control references для переносного offline analysis, але
не копіює calibration bootstrap replicates. Candidate predictions frozen до
вимірювань. Raw observations залишаються у CSV із hash, не дублюються в кожному
pair summary. JSON містить `null` замість non-standard NaN/Infinity.

Reader підтримує старі calibration і B artifacts для offline аналізу. Legacy A
можна проаналізувати, але partial lookup не можна використати як strict full-grid
validation. Старий `run_experiments` працює з одним design; для нового workflow
використовуйте `run_fresh_validation`. Старі outputs не мігруються in-place.

## Команди PowerShell

Використовуйте `.venv`: у перевіреному середовищі там Ultralytics 8.4.60, тоді як
системний Python має іншу версію. Не запускайте calibration і validation inference
паралельно.

Повний workflow однією командою:

```powershell
.venv/Scripts/python.exe -m experiments.cost_model_validation.run_fresh_validation --output outputs/yolo26n_cpu_full_v4
```

Окремий Experiment A:

```powershell
.venv/Scripts/python.exe -m experiments.cost_model_validation.collect_experiment_a --config experiments/cost_model_validation/config.voc-yolo26n.yaml --output outputs/yolo26n_cpu_full_v4/calibration
.venv/Scripts/python.exe -m experiments.cost_model_validation.analyze_experiment_a --input outputs/yolo26n_cpu_full_v4/calibration/experiment_a_raw.csv --output outputs/yolo26n_cpu_full_v4/calibration --bootstrap-count 2000
```

Representative B:

```powershell
.venv/Scripts/python.exe -m experiments.cost_model_validation.pair_specs --config experiments/cost_model_validation/config.voc-yolo26n.representative.yaml --calibration outputs/yolo26n_cpu_full_v4/calibration/linear_fit.json --output outputs/yolo26n_cpu_full_v4/representative_pairs.json
.venv/Scripts/python.exe -m experiments.cost_model_validation.collect_experiment_b --config experiments/cost_model_validation/config.voc-yolo26n.representative.yaml --fit outputs/yolo26n_cpu_full_v4/calibration/linear_fit.json --pairs outputs/yolo26n_cpu_full_v4/representative_pairs.json --output outputs/yolo26n_cpu_full_v4/representative
.venv/Scripts/python.exe -m experiments.cost_model_validation.analyze_experiment_b --input outputs/yolo26n_cpu_full_v4/representative/experiment_b_raw.csv --output outputs/yolo26n_cpu_full_v4/representative
```

Challenge B:

```powershell
.venv/Scripts/python.exe -m experiments.cost_model_validation.pair_specs --config experiments/cost_model_validation/config.voc-yolo26n.yaml --calibration outputs/yolo26n_cpu_full_v4/calibration/linear_fit.json --output outputs/yolo26n_cpu_full_v4/challenge_pairs.json
.venv/Scripts/python.exe -m experiments.cost_model_validation.collect_experiment_b --config experiments/cost_model_validation/config.voc-yolo26n.yaml --fit outputs/yolo26n_cpu_full_v4/calibration/linear_fit.json --pairs outputs/yolo26n_cpu_full_v4/challenge_pairs.json --output outputs/yolo26n_cpu_full_v4/challenge
.venv/Scripts/python.exe -m experiments.cost_model_validation.analyze_experiment_b --input outputs/yolo26n_cpu_full_v4/challenge/experiment_b_raw.csv --output outputs/yolo26n_cpu_full_v4/challenge
```

Новий hardware smoke зі вже завершеним повним A:

```powershell
.venv/Scripts/python.exe -m experiments.cost_model_validation.run_fresh_validation --calibration outputs/cost_model_voc_yolo26n_v7/calibration/linear_fit.json --output outputs/cost_model_voc_yolo26n_v7/fresh_smoke --smoke-pairs 20
```

Ця команда вже виконана під час перевірки; для ще одного незалежного run потрібні
нові generator seeds, output і prior-pair exclusions. Не перезаписуйте ці pair
files, щоб обійти reuse guard. Для повного confirmatory run на 500+500 pairs
скопіюйте configs, задайте нові seeds **до перегляду outcomes** і додайте обидва
smoke pair files у `prior_pair_specs`.

## Opt-in runtime integration

У video config:

```yaml
roi_merge:
  strategy: calibrated
  calibration: outputs/yolo26n_cpu_full_v4/calibration/linear_fit.json
  policies:
    primary: linear_tau
    confidence_level: 0.95
    decision_margin_s: 0.0
    fallback: error
```

Щоб явно дослідити lookup runtime, `primary: shape_lookup_conservative`.
Production lookup може мати явний `fallback: conservative_area` (all-three
consensus, потім separate за OOD) або `fallback: separate`. Кожний fallback
логуються та лічиться. Strict validation не використовує жодного fallback.
Merger metadata містить calibration hash і policy declaration. Існуючі legacy
video strategies не перемикаються автоматично. Результати optional runtime
candidate ще не є рекомендацією змінити production default.

## Змінені файли та перевірки

Нові: `shape_model.py`, `validation_design.py`, `artifacts.py`,
`run_fresh_validation.py`, `config.voc-yolo26n.representative.yaml`,
`tests/test_shape_validation.py`, `tools/mergers/calibrated.py`, цей документ.

Оновлені: усі три experiment adapters, `models.py`, `common.py`,
`reproducibility.py`, collectors/analyzers A/B, `aggregate_sessions.py`,
`pair_specs.py`, YOLO YAML, `tests/test_reproducibility.py`, README,
`tools/helpers/pipeline.py`, `tools/infer_framework_vid.py`,
`tools/benchmarks/benchmark_framework_vid.py`.

```powershell
.venv/Scripts/python.exe -m unittest discover -s experiments/cost_model_validation/tests -v
.venv/Scripts/python.exe -m compileall -q experiments/cost_model_validation tools/mergers/calibrated.py
git diff --check
```

Hardware results і остаточна кількість тестів наведені в `V4_VERIFICATION.md`.
Жодні outcomes v6 не використано для fit, налаштування margin чи confirmatory
policy selection. Prior pair files використовуються лише для виключення reuse.

## Обмеження та exploratory findings

Lookup залежить від weights, inference path, dtype, device, CPU state і domain.
Synthetic reference не замінює deployment trace. Один calibration session і
малий B smoke не дають точного production ranking; вони перевіряють pipeline.
SE-bound conservative policy може втрачати корисні merge opportunities;
normal approximation також потребує незалежної статистичної перевірки.
Повний publication run потребує clean commit і контрольованого CPU навантаження.

Exploratory: all-three consensus і shape-lookup conservative — реалізовані
кандидати; кращий fit будь-якої area model не є доказом кращого merge policy.
Жодний кандидат не оголошується переможцем за старим challenge set або новим smoke.
