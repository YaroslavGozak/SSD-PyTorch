# Інструкція для coding agent: стабілізація Experiment A та відтворювана валідація Experiment B

## 1. Мета

Потрібно змінити код cost-model calibration та pairwise validation для YOLO26n так, щоб:

1. калібрування не залежало від порядку shapes, нагріву CPU, cache state і timing spikes;
2. bootstrap охоплював breakpoint та коефіцієнти piecewise-моделі;
3. підтримувалося об’єднання 3–5 незалежних Experiment A sessions;
4. усі повторні Experiment B використовували однаковий набір геометрій;
5. validation set містив broad/random pairs, межі всіх моделей і model-disagreement cases.

Experiment B не можна використовувати для fit коефіцієнтів, breakpoint, threshold або автоматичного вибору primary model.

## 2. Інваріанти, які треба зберегти

Для requested shape `(h, w)`:

```python
tensor_h = ceil(h / stride) * stride
tensor_w = ceil(w / stride) * stride
effective_area = tensor_h * tensor_w
```

Для YOLO26n зараз `stride = 32`.

Вимоги:

* округлювати окремо висоту і ширину;
* не округлювати лише площу;
* для merged ROI спочатку будувати bounding rectangle, потім квантувати його висоту і ширину;
* для stride 32 effective areas і `delta_area` мають бути кратними 1024;
* domain validation виконувати на actual tensor shapes;
* використовувати batch size 1 і той самий inference path для calibration та validation.

Latency models:

```text
Linear:
T(A) = b0 + b1*A

Piecewise:
T(A) = b0 + b1*A + b2*max(0, A-B)

Quadratic:
T(A) = b0 + b1*A + b2*A^2
```

Загальне рішення:

```text
merge, якщо T(A_merged) < T(A_1) + T(A_2)
```

## 3. Experiment A: interleaved measurement schedule

Не вимірювати один shape 40 разів поспіль.

Для `N` shapes і `R` repetitions створити `R` blocks. У кожному block кожний shape вимірюється рівно один раз у новому псевдовипадковому порядку:

```text
block 0: permutation_0(all_shapes)
block 1: permutation_1(all_shapes)
...
block R-1: permutation_R-1(all_shapes)
```

Вимоги:

* schedule детермінований за `schedule_seed`;
* кожний block використовує derived seed;
* сусідні permutations не повинні повторюватися;
* кожний shape має рівно `R` observations;
* не сортувати measurements за area, height, width або aspect ratio;
* зберігати schedule до запуску;
* при resume завантажувати збережений schedule, а не генерувати новий;
* обчислювати SHA-256 canonical schedule.

Schedule item:

```json
{
  "global_position": 0,
  "block_index": 0,
  "position_in_block": 0,
  "shape_id": "h160_w320",
  "tensor_h": 160,
  "tensor_w": 320,
  "effective_area": 51200
}
```

## 4. Warmup

* Зберегти global backend/model warmup.
* Після нього виконати однакову кількість prewarm passes для всіх shapes.
* Порядок shapes у prewarm passes також рандомізувати.
* Warmup observations не включати до fit.
* Зберігати warmup schedule, seed і кількість passes у metadata.

## 5. Calibration controls

Додати control shapes між Experiment A blocks:

```yaml
experiment_a:
  control_every_blocks: 2
  control_shapes:
    - [160, 160]
    - [160, 320]
    - [320, 320]
```

Controls повинні:

* бути всередині calibration envelope;
* проходити через загальну stride-quantization function;
* не входити до fit dataset;
* вимірюватися у збалансованому порядку;
* містити timestamp, elapsed time, block index і measured latency;
* містити predictions linear, piecewise та quadratic models.

Рахувати:

* early/middle/late mean;
* median, min, max;
* relative change між першою й останньою третиною;
* trend latency vs elapsed time;
* `hardware_state_shift` за configurable documented rule.

Controls використовуються як diagnostics. Не коригувати ними observations автоматично без окремого явно ввімкненого режиму.

## 6. Raw observations та per-shape statistics

Зберігати кожний measured inference Experiment A:

```json
{
  "session_id": "...",
  "shape_id": "h160_w320",
  "requested_h": 0,
  "requested_w": 0,
  "tensor_h": 160,
  "tensor_w": 320,
  "effective_area": 51200,
  "block_index": 0,
  "position_in_block": 0,
  "global_position": 0,
  "timestamp_utc": "...",
  "elapsed_s": 0.0,
  "latency_s": 0.0
}
```

Raw observations можна зберігати як JSONL, CSV або Parquet.

Для кожного computational shape обчислювати:

* count;
* mean;
* median;
* symmetric trimmed mean;
* sample standard deviation з `ddof=1`;
* standard error;
* coefficient of variation;
* min/max;
* p05/p25/p75/p95;
* outlier count;
* first/last measurement position.

Config:

```yaml
experiment_a:
  primary_shape_statistic: trimmed_mean
  trim_fraction_each_tail: 0.10
```

Для 40 observations 10% trimming відкидає чотири найменші та чотири найбільші значення.

Зберігати fits і для mean, і для robust statistic. Primary statistic має бути явно заданий до запуску. Не вибирати його за Experiment B accuracy.

## 7. Shape-level та area-level fits

Зберегти:

* shape-level fit: одна observation на унікальний `(tensor_h, tensor_w)`;
* area-level fit: shapes з однаковою effective area агрегуються перед fit.

Додати:

```yaml
experiment_a:
  primary_fit_level: shape_level
```

або:

```yaml
experiment_a:
  primary_fit_level: area_level
```

Не перемикати primary fit автоматично за результатом Experiment B.

Зберігати:

* fit level;
* primary shape statistic;
* count unique shapes;
* count unique areas;
* multiplicity кожної area;
* coefficients та fit metrics для всіх альтернативних fits.

Якщо coefficients або breakpoint суттєво залежать від fit level, створити warning, але не змінювати primary model.

## 8. Bootstrap усіх моделей

Реалізувати fixed-design hierarchical bootstrap:

1. Зберегти повний designed shape set.
2. Для кожного bootstrap replicate окремо ресемплювати raw observations кожного shape із поверненням.
3. Повторно обчислити primary shape statistic.
4. Побудувати selected shape- або area-level dataset.
5. Повторно fit-ити linear, quadratic і piecewise models.
6. Зберегти coefficients, breakpoint, fit status і diagnostics.

Не змішувати observations різних shapes.

### Piecewise breakpoint

* Шукати breakpoint тільки на quantized grid.
* Для stride 32 breakpoint має бути кратним 1024.
* Використовувати observed unique effective areas або явно визначені допустимі межі.
* Вимагати configurable minimum support з обох боків breakpoint.
* Перевіряти:

```text
b1 > 0
b1 + b2 > 0
```

Invalid replicates позначати як invalid, зберігати причину й рахувати invalid fraction.

### Bootstrap output

Piecewise:

* CI для `b0`, `b1`, `b2`;
* CI для `c2 = b1 + b2`;
* breakpoint distribution;
* breakpoint CI;
* frequency table quantized breakpoint values;
* median/MAD coefficients;
* invalid replicate fraction.

Quadratic:

* CI для всіх coefficients.

Linear:

* наявні CI для `K`, `c`, `tau`.

## 9. Bootstrap stability merge decisions

Для кожної frozen pair specification застосувати всі valid bootstrap models Experiment A.

Зберігати:

```json
{
  "pair_id": "...",
  "model": "piecewise",
  "merge_probability": 0.0,
  "predicted_gain_ms_ci": [0.0, 0.0],
  "decision_stable": true
}
```

```text
merge_probability =
fraction of valid bootstrap models predicting merge
```

Пороги stability задавати в config, наприклад:

```yaml
decision_stability:
  stable_separate_max_probability: 0.10
  stable_merge_min_probability: 0.90
```

Не використовувати ground-truth labels Experiment B для визначення decision stability.

## 10. Multi-session Experiment A

Додати окремий command для об’єднання 3–5 незалежних calibration sessions.

Перед aggregation перевіряти однаковість:

* weights SHA-256;
* backend;
* device;
* dtype;
* preprocessing;
* stride і shape policy;
* timing mode;
* calibration shape set;
* calibration envelope.

Несумісні sessions не об’єднувати.

Pooled model:

1. вирівняти observations за computational shape;
2. отримати session-level primary statistic для кожного shape;
3. побудувати pooled per-shape value за заздалегідь визначеним методом;
4. fit-ити pooled models заново.

Не отримувати pooled model простим усередненням готових coefficients або breakpoints.

Звітувати:

* per-session coefficients;
* coefficient mean/median/std/CV;
* breakpoint frequency;
* per-session RMSE/MAE/AIC/BIC;
* pairwise differences predictions між sessions;
* частку frozen-pair decisions, однакових у всіх sessions;
* pooled coefficients;
* stability warnings.

## 11. Заморожений `pair_specs.json`

Повністю відокремити pair generation від measurement.

Generate mode:

```text
generate-pairs \
  --calibration pooled_calibration.json \
  --output pair_specs.json
```

Для кожної пари зберігати:

* stable `pair_id`;
* coordinates/source geometry;
* geometry type;
* requested shapes;
* actual tensor shapes;
* effective areas;
* `delta_area`;
* domain result;
* computational key;
* primary stratum;
* усі matched stratum tags;
* calibration artifact hash;
* generator seed;
* schema version.

Після генерації обчислити SHA-256 canonical `pair_specs.json`.

Measure mode:

```text
measure-pairs \
  --pairs pair_specs.json \
  --calibration pooled_calibration.json
```

Вимоги:

* не регенерувати pairs;
* перевіряти schema, hash, canvas, domain і stride;
* відхиляти duplicate computational keys;
* записувати `pair_specs_hash` у metadata;
* `--regenerate-pairs` має бути окремою явною операцією.

Усі наступні Experiment B runs повинні використовувати той самий ordered список `pair_id`.

## 12. Strata Experiment B

Quotas повинні задаватися в config.

### `broad_random`

Model-independent coverage calibration domain.

### `linear_boundary`

Пари з малим:

```text
abs(delta_area - tau_linear)
```

Зберігати signed distance та сторону threshold.

### `piecewise_boundary`

Пари з малим:

```text
abs(predicted_gain_piecewise_ms)
```

Включати обидва боки нуля.

### `quadratic_boundary`

Пари з малим:

```text
abs(predicted_gain_quadratic_ms)
```

Включати обидва боки нуля.

### `model_disagreement`

Пари, для яких хоча б дві models дають різні boolean decisions.

Одна пара може мати кілька tags. Primary stratum призначати за детермінованим priority:

```text
model_disagreement
piecewise_boundary
quadratic_boundary
linear_boundary
broad_random
```

Priority і quotas зберігати в metadata.

Якщо quota або обидві сторони boundary недоступні в calibration domain, завершувати generation із diagnostic, а не заповнювати quota іншими парами мовчки.

## 13. Experiment B: order-aware estimate та bootstrap

Зберегти balanced ABBA/BAAB protocol.

Додати:

* `order_block_id`;
* position усередині ABBA/BAAB block;
* перевірку однакової кількості `merged_first` і `separate_first`.

Pair estimate:

```text
D_pair =
0.5 * (
    mean(D | merged_first)
    +
    mean(D | separate_first)
)
```

де:

```text
D = separate_ms - merged_ms
```

CI отримувати:

* stratified bootstrap окремо всередині order conditions; або
* resampling цілих ABBA/BAAB blocks.

Не використовувати naive bootstrap observations, який може порушити order balance.

Metadata повинна містити:

* bootstrap method;
* number of replicates;
* confidence level;
* seed.

Labels:

```text
merge_beneficial     if CI_low > 0
separate_beneficial  if CI_high < 0
ambiguous            otherwise
```

## 14. Result artifacts

Experiment A session:

* provenance;
* schedule config/hash;
* raw observations reference;
* per-shape statistics;
* control diagnostics;
* shape-/area-/observation-level fits;
* primary fit selector;
* bootstrap distributions;
* calibration envelope;
* quality warnings.

Pooled calibration:

* session IDs і hashes;
* compatibility result;
* per-session summaries;
* pooled coefficients;
* between-session stability;
* bootstrap uncertainty;
* provenance.

Experiment B:

* pair specs hash;
* calibration artifact hash;
* raw order-aware observations;
* order-adjusted pair estimates;
* model predictions та gains;
* bootstrap merge probabilities;
* metrics overall та за strata/geometry/area regime;
* regret для all/determinate pairs;
* control/order/hardware diagnostics.

## 15. Quality gates

Додати configurable warning/fail policy для:

* неправильного repetition count;
* пропущеного або повторного shape у block;
* schedule mismatch при resume;
* OOD shapes;
* non-stride-aligned tensors;
* високого invalid bootstrap ratio;
* `b1 <= 0` або `b1+b2 <= 0`;
* недостатньої підтримки з одного боку breakpoint;
* сильного control drift;
* provenance mismatch;
* pair specs hash mismatch;
* duplicate computational keys;
* невиконаних strata quotas;
* `git_dirty=true` для final/publication runs.

Числові thresholds повинні бути config parameters і зберігатися в metadata.

## 16. Обов’язкові тести

1. Кожний shape з’являється один раз у кожному block.
2. Кожний shape має рівно `R` observations.
3. Однаковий seed відтворює schedule byte-for-byte.
4. Інший seed змінює permutations.
5. Resume використовує той самий schedule hash.
6. Height і width окремо округлюються до stride.
7. Для stride 32 area і delta area кратні 1024.
8. Domain validation використовує actual tensor shape.
9. Mean/median/trimmed mean/std/CV відповідають fixture.
10. Bootstrap не змішує observations різних shapes.
11. Breakpoint належить quantized grid.
12. Invalid slopes правильно позначаються.
13. Bootstrap детермінований за seed.
14. `merge_probability` правильно рахується.
15. Несумісні sessions не агрегуються.
16. Pooled model повторно fit-иться на pooled data.
17. Pair specs hash відтворюється.
18. Measure mode не викликає pair generator.
19. Duplicate computational keys відхиляються.
20. Quotas і обидві boundary sides перевіряються.
21. Повторні runs використовують однакові `pair_id`.
22. Order balance перевіряється для кожної пари.
23. Block/stratified bootstrap зберігає баланс.
24. Synthetic order effect компенсується в `D_pair`.
25. End-to-end smoke test створює всі artifacts.

## 17. Acceptance criteria

Завдання завершене, коли:

* Experiment A використовує interleaved schedule;
* зберігаються raw observations і robust statistics;
* controls виконуються між blocks;
* bootstrap охоплює piecewise breakpoint та coefficients;
* доступний multi-session aggregation command;
* pair generation відокремлений від measurement;
* повторні Experiment B використовують один `pair_specs_hash`;
* присутні broad, boundary і disagreement strata;
* Experiment B використовує order-aware bootstrap;
* Experiment B не впливає на calibration fit або model selection;
* тести проходять;
* final run відтворюється з clean commit і повним provenance.

Не використовувати конкретні accuracy, breakpoint або regret як pass/fail criteria. Мета — відтворюваність і контроль похибки, а не підгонка під попередній run.

## 18. Звіт coding agent

Після реалізації надати:

1. перелік змінених файлів;
2. схему нового data flow;
3. приклад Experiment A config;
4. команду однієї calibration session;
5. команду multi-session aggregation;
6. команду генерації `pair_specs.json`;
7. команду повторного Experiment B;
8. приклади нових schemas;
9. команди тестів і результати;
10. migration/compatibility notes;
11. підтвердження, що Experiment B не використовувався для fit або автоматичного model selection.
