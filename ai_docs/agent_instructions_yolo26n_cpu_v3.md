# Інструкція для coding agent: виправлення latency-моделі та Experiment B після оновленого YOLO26n CPU run

## 1. Мета завдання

Потрібно оновити код калібрування та валідації ROI merge для `YOLO26n` на CPU з урахуванням фактичних результатів нового run.

Головний висновок нового run: **piecewise area-only модель краще описує середню залежність latency від площі в Experiment A, але гірше приймає merge-рішення в Experiment B**. Тому не можна автоматично замінювати `linear_tau` на `piecewise_direct_cost` як production/default policy.

Потрібно:

1. зберегти всі три area-only моделі як наукові baselines;
2. додати shape-aware latency model, яка враховує фактичні `tensor_h` та `tensor_w` після вирівнювання до stride 32;
3. розділити representative validation та навмисно складний challenge set;
4. додати conservative uncertainty-aware merge policy;
5. виправити remaining issues у diagnostics та JSON artifacts;
6. повторити Experiment A і провести **новий незалежний** Experiment B, не підбираючи production policy на вже виміряних парах.

Ця інструкція замінює попередні рекомендації там, де вони суперечать новому run.

## 2. Зафіксований baseline нового run

Використовувати наведені числа лише як regression/sanity reference, не hardcode-ити їх у production logic або тести.

### 2.1. Experiment A

- backend: `ultralytics 8.4.60`;
- model: `YOLO26n`, `float32`, CPU;
- preprocessing shape policy: `ceil_to_stride`;
- stride: `32`;
- primary fit: `shape_level`, statistic `trimmed_mean`;
- observations: `1240`;
- effective shapes: `31`;
- unique effective areas: `24`;
- calibration envelope: `128..640` по кожній стороні, aspect ratio `0.5..2.0`, area `25600..409600`;
- Experiment A не позначив hardware shift.

Primary shape-level fits:

| Model | R² | RMSE | MAE | AIC |
|---|---:|---:|---:|---:|
| linear | 0.9586 | 2.707 ms | 2.232 ms | -362.53 |
| quadratic | 0.9831 | 1.728 ms | 1.503 ms | -388.37 |
| piecewise | 0.9901 | 1.324 ms | 1.079 ms | -402.87 |

Linear coefficients:

```text
b0 = 0.0221570700 s
b1 = 1.1982093945e-7 s/pixel
tau = 184918.18 pixels
bootstrap 95% CI(tau) = [178319.77, 191995.15]
```

Piecewise coefficients:

```text
T(A) = b0 + b1*A + b2*max(0, A-B)
b0 = 0.0161030548 s
b1 = 2.0147385424e-7 s/pixel
b2 = -1.0982155482e-7 s/pixel
B  = 123904 pixels
post-breakpoint slope = b1+b2 = 9.1652299422e-8 s/pixel
```

Bootstrap breakpoint is not perfectly stable:

```text
95% CI(B) = [93184, 165888]
frequency:
  93184  -> 468 / 2000
  123904 -> 1403 / 2000
  131072 -> 18 / 2000
  165888 -> 111 / 2000
```

### 2.2. Experiment B

- pairs: `500`;
- unique computational configurations: `500`;
- determinate pairs: `345`;
- ambiguous fraction: `31%`;
- sampling: five frozen strata, по 100 pairs у кожній;
- accepted OOD pairs: `0`;
- domain policy: `reject_out_of_domain`;
- order design: balanced `ABBA/BAAB`;
- paired repetitions: `20`;
- order effect: `0.691 ms`, warning threshold `0.5 ms`;
- `hardware_state_shift = true` через `control_drift`;
- Experiment B використав `180` різних effective shapes і `101` різну effective area.

Overall unweighted pooled metrics на determinate pairs:

| Rule | Accuracy | Balanced accuracy | Mean regret, all pairs | FP | FN |
|---|---:|---:|---:|---:|---:|
| linear_tau | 81.74% | 85.42% | 0.597 ms | 60 | 3 |
| quadratic_direct_cost | 77.68% | 82.89% | 0.725 ms | 77 | 0 |
| piecewise_direct_cost | 71.59% | 78.22% | 0.945 ms | 98 | 0 |

У `model_disagreement` stratum на 61 determinate pairs:

| Rule | Accuracy | Balanced accuracy | Mean regret, all 100 stratum pairs |
|---|---:|---:|---:|
| linear_tau | 72.13% | 37.93% | 0.834 ms |
| quadratic_direct_cost | 49.18% | 73.28% | 1.472 ms |
| piecewise_direct_cost | 14.75% | 55.17% | 2.574 ms |

Цей stratum сильно незбалансований: 58 determinate `separate_beneficial` проти 3 `merge_beneficial`. Тому окремо показувати accuracy, balanced accuracy, confusion matrix і regret; жодна одна метрика не є достатньою.

## 3. Інтерпретація, яку код і звіт мають відображати

### 3.1. Кращий fit не гарантує кращого merge-рішення

Experiment A оцінює абсолютну функцію `T`. Experiment B використовує різницю трьох прогнозів:

```text
gain = T(shape_1) + T(shape_2) - T(shape_merged)
```

Невелика систематична помилка кожного прогнозу може скластися у неправильний знак `gain`. Тому вибирати production policy лише за `R²`, RMSE, AIC або BIC заборонено.

### 3.2. Area-only модель недостатня для CPU

У Experiment A однакова площа, але транспоновані shapes мають різну latency. Для trimmed mean різниця становить приблизно:

| Effective area | Shapes | Різниця |
|---:|---|---:|
| 28,672 | 128×224 vs 224×128 | 0.818 ms |
| 46,080 | 160×288 vs 288×160 | 0.969 ms |
| 67,584 | 192×352 vs 352×192 | 1.583 ms |
| 93,184 | 224×416 vs 416×224 | 1.340 ms |
| 131,072 | 256×512 vs 512×256 | 1.937 ms |
| 165,888 | 288×576 vs 576×288 | 1.739 ms |
| 204,800 | 320×640 vs 640×320 | 0.992 ms |

Це порівнюване або більше за `boundary_width_ms = 0.5`. Отже, `A = H*W` не є достатньою ознакою для точного рішення на CPU.

### 3.3. Просторове вирівнювання до 32 — не numeric quantization

У назвах коду та metadata використовувати терміни:

```text
stride alignment
shape quantization
ceil_to_stride
```

Не називати це просто `model quantization`, щоб не сплутати з INT8/FP16 quantization. Поточний run працює у `float32`.

## 4. P0: не змінювати default policy на piecewise

До незалежної перевірки нової shape-aware моделі:

- `linear_tau` залишити default baseline;
- `piecewise_direct_cost` і `quadratic_direct_cost` залишити experimental/offline rules;
- не підставляти емпіричний threshold із Experiment B;
- не навчати коефіцієнти на Experiment B;
- не використовувати результати цього run як acceptance threshold для нового run.

Якщо runtime код уже перемкнено на piecewise, повернути default на `linear_tau` або на явно конфігурований policy. Зміна policy має бути видима у config та provenance.

## 5. P0: єдина stride-aware функція effective shape

У репозиторії повинна бути рівно одна канонічна функція, яку використовують:

- Experiment A;
- pair generator;
- Experiment B measurement;
- offline analyzer;
- production merge decision.

Концептуально:

```python
def effective_shape(requested_h: int, requested_w: int, stride: int = 32) -> tuple[int, int]:
    if requested_h <= 0 or requested_w <= 0:
        raise ValueError("shape dimensions must be positive")
    h = ((requested_h + stride - 1) // stride) * stride
    w = ((requested_w + stride - 1) // stride) * stride
    return h, w
```

Але не копіювати цю функцію, якщо adapter уже реалізує точну preprocessing policy. Винести/перевикористати реальну adapter logic.

Обов’язково:

1. спочатку побудувати requested ROI або merged bounding box;
2. окремо округлити `H` і `W` вгору до stride;
3. лише потім рахувати `effective_area = tensor_h * tensor_w`;
4. не округляти raw area до кратності `32` або `1024` замість округлення сторін;
5. зберігати requested shape та effective tensor shape окремо.

## 6. P0: додати shape lookup latency model

Оскільки domain дискретний, найнадійніша primary candidate для цього setup — повна таблиця latency для всіх допустимих stride-aligned shapes.

За поточного envelope можливі `H,W`:

```text
128, 160, 192, ..., 640
```

Після застосування area та aspect-ratio constraints у domain є приблизно 200 shapes. Це невелика скінченна таблиця.

### 6.1. Новий calibration grid

Згенерувати всі shapes, для яких одночасно:

```python
128 <= H <= 640
128 <= W <= 640
25600 <= H * W <= 409600
0.5 <= W / H <= 2.0
H % 32 == 0
W % 32 == 0
```

Не використовувати лише 31 sampled shape. Зберегти grid deterministically та його hash.

Для кожного shape виконати однакову кількість вимірювань за randomized/block-balanced schedule. Зберігати raw observations або окремий raw artifact із hash.

Primary point estimate на shape:

```text
trimmed_mean або median
```

Не змішувати observations різних `(H,W)`, навіть якщо `H*W` однакова.

### 6.2. Інтерфейс

```python
@dataclass(frozen=True)
class ShapeLatencyEstimate:
    point_s: float
    standard_error_s: float | None
    ci_s: tuple[float, float] | None
    repetitions: int


class ShapeLookupLatencyModel:
    def predict_seconds(self, tensor_h: int, tensor_w: int) -> float:
        ...

    def estimate(self, tensor_h: int, tensor_w: int) -> ShapeLatencyEstimate:
        ...

    def is_in_domain(self, tensor_h: int, tensor_w: int) -> bool:
        ...
```

Lookup key має бути ordered pair `(tensor_h, tensor_w)`. Не сортувати сторони, бо `T(H,W)` і `T(W,H)` на CPU не однакові.

### 6.3. Fallback

Для primary publication run повна domain table повинна мати запис для кожного accepted shape. Missing lookup entry — fail fast, а не тихе area-only fallback.

Для production можна явно конфігурувати fallback:

```text
shape_lookup -> conservative area model -> separate
```

Кожне використання fallback рахувати й логувати. У validation run fallback count має бути нуль.

## 7. Area-only моделі залишити як baselines

### 7.1. Linear

```text
T(A) = b0 + b1*A
merge iff A_m - A_1 - A_2 < b0/b1
```

Direct-cost та scalar-`tau` реалізації мають бути еквівалентні.

### 7.2. Piecewise

```text
T(A) = b0 + b1*A + b2*max(0, A-B)
```

Практичне рішення:

```python
gain_s = T(a1) + T(a2) - T(am)
predicted_merge = gain_s > 0
```

Piecewise не має одного universal `tau`. Для фіксованих `a1,a2` можна обчислити умовний максимально допустимий `am` шляхом інверсії монотонної функції:

```python
def piecewise_area_limit(a1: int, a2: int, m: PiecewiseModel) -> float:
    separate_cost = m.predict_seconds(a1) + m.predict_seconds(a2)
    cost_at_breakpoint = m.b0 + m.b1 * m.breakpoint_area
    post_slope = m.b1 + m.b2

    if m.b1 <= 0 or post_slope <= 0:
        raise ValueError("piecewise latency model must be strictly increasing")

    if separate_cost <= cost_at_breakpoint:
        return (separate_cost - m.b0) / m.b1

    return (
        m.breakpoint_area
        + (separate_cost - cost_at_breakpoint) / post_slope
    )
```

Тоді для fixed `a1,a2`:

```text
merge iff am < piecewise_area_limit(a1, a2, model)
```

Це diagnostic/API helper, а не заміна direct-cost rule. Через stride 32 можливі лише дискретні `(H,W)`, тому runtime все одно має порівнювати costs для фактичного merged tensor shape.

### 7.3. Online complexity

Piecewise decision — `O(1)`: три виклики функції з одним `max`. Shape lookup decision — три dictionary/array lookup та кілька арифметичних операцій. Їх overhead має бути на порядки меншим за inference latency у десятки мілісекунд.

Не виконувати bootstrap із 2000 models у production hot path. Bootstrap потрібний offline для побудови uncertainty metadata або для попереднього вибору margin/policy.

## 8. P1: uncertainty-aware conservative policy

Новий run показує, що агресивні nonlinear rules мають дуже багато false merge. Для latency objective false merge у цьому run дорожчий за conservative fallback.

Додати candidate policy:

```text
merge тільки якщо прогнозована перевага статистично/практично достатня;
інакше separate.
```

Для shape lookup бажаний варіант:

```python
gain_point_s = t1.point_s + t2.point_s - tm.point_s
gain_se_s = sqrt(t1.se_s**2 + t2.se_s**2 + tm.se_s**2)
gain_lcb_s = gain_point_s - z * gain_se_s

predicted_merge = gain_lcb_s > decision_margin_s
```

Використати confidence level і `decision_margin_s`, задані **до нового Experiment B**. Не підбирати margin на вже виміряних 500 pairs.

Якщо незалежність shape estimates не обґрунтована, отримувати gain distribution offline із calibration bootstrap та зберігати компактні параметри/квантілі, але не тягнути всі 2000 replicate models у runtime.

### 8.1. Exploratory ensemble

На поточному run post-hoc policy `merge only if all three area models agree and have sufficient gain` виглядає значно краще, але це результат на тих самих даних і **не є підтвердженим production rule**.

Дозволено реалізувати його як окремий candidate:

```text
conservative_consensus
```

але:

- позначити `exploratory = true`;
- не робити default;
- не hardcode-ити `0.5 ms` або `1.0 ms` з цього run;
- перевірити лише на новому frozen pair set;
- звітувати окремо від pre-registered primary policy.

## 9. P0: розділити representative set і challenge set

Поточний набір має рівно по 100 pairs у:

```text
broad_random
linear_boundary
piecewise_boundary
quadratic_boundary
model_disagreement
```

Це корисний challenge design, але unweighted pooled accuracy не оцінює production distribution.

Зробити два незалежні artifacts/runs.

### 9.1. Representative validation

- sampling має відтворювати реальний або явно заданий deployment distribution ROI size, aspect ratio, overlap і distance;
- якщо реального trace немає, назвати його `synthetic_reference_distribution`, а не production distribution;
- не балансувати штучно model boundaries;
- використовувати цей set для primary expected regret і expected latency gain.

### 9.2. Challenge validation

- зберегти frozen multi-model strata;
- використовувати для stress testing boundaries, disagreements та failure modes;
- звітувати per-stratum metrics;
- pooled metrics називати `unweighted_challenge_average`;
- не переносити їх напряму на deployment prevalence.

Якщо потрібна одна weighted aggregate metric, weights мають походити з deployment/reference distribution і бути записані у JSON. Не визначати weights за observed labels.

## 10. P0: покращити pair sampling diagnostics

Поточний `broad_random` дав 100/100 determinate `merge_beneficial`; він не тестує обидва класи. Це не помилка саме по собі, але треба показувати class coverage.

Для кожного stratum записувати:

```json
{
  "requested_pairs": 100,
  "generated_pairs": 100,
  "determinate_pairs": 0,
  "ambiguous_pairs": 0,
  "actual_class_counts": {},
  "prediction_counts_by_model": {},
  "generation_attempts": 0,
  "rejection_reasons": {}
}
```

Виправити або видалити застаріле поле `near_pair_count`. У поточному artifact воно дорівнює `0`, хоча існують boundary strata по 100 pairs. Поле має або рахувати всі boundary pairs із чіткою семантикою, або бути перейменоване/прибране у новій schema version.

## 11. P0: timing, order effect та ambiguous pairs

Balanced `ABBA/BAAB` значно покращив попередню ситуацію, але `order_effect_ms = 0.691` все ще перевищує warning threshold `0.5`.

### 11.1. Зберегти balanced order

- однакова кількість `merged_first` і `separate_first`;
- paired difference `D = separate_ms - merged_ms`;
- bootstrap має зберігати order strata або resample-ити paired blocks;
- звітувати `D` окремо за order condition.

### 11.2. Labels

Primary label залишається:

```text
merge_beneficial     if CI_low > 0
separate_beneficial  if CI_high < 0
ambiguous            otherwise
```

31% ambiguous — багато. Для challenge strata збільшити repetitions або використати pre-registered sequential design.

Безпечний fixed варіант:

```text
broad/reference pairs: 20 paired repetitions
boundary/disagreement pairs: 40 або 60 paired repetitions
```

Якщо використовувати adaptive extension:

- наперед задати checkpoints і maximum repetitions;
- застосувати sequentially valid CI або alpha-spending correction;
- не зупинятися за звичайним 95% CI без корекції;
- записати stopping rule у metadata.

### 11.3. Practical equivalence zone

Окрім significance label, додати practical label із наперед заданим `minimum_worthwhile_gain_ms`:

```text
confident_merge      if CI_low > +delta_ms
confident_separate   if CI_high < -delta_ms
practically_tied     if CI entirely inside [-delta_ms, +delta_ms]
uncertain            otherwise
```

Не змішувати statistical ambiguity та practical equivalence в одному полі.

## 12. P0: control diagnostics

Зберегти два різні поняття:

1. `within_session_drift` — early/middle/late зміна controls усередині Experiment B;
2. `calibration_to_validation_shift` — відмінність controls між Experiment A та B.

Не називати обидва одним `control_drift`.

Для кожного control shape і кожної latency model зберігати:

```text
calibration baseline distribution
validation mean/median
absolute shift ms
relative shift
prediction residual
early-to-late relative drift
```

Hardware shift визначати за observed calibration control baseline, а не лише за residual однієї area model. Поточний run показує, що model residual залежить від вибраної моделі.

Дедуплікувати warnings: у `decision_metrics.json` рядок `Quality gate: control_drift` зараз записаний двічі.

## 13. P1: компактні та недубльовані artifacts

Поточні artifacts дублюють великий список із 2000 bootstrap replicate models у кількох JSON, через що `experiment_b_metadata`, `linear_fit` і summary займають мегабайти.

Змінити схему:

- `bootstrap_models.json` або compressed binary/JSON.gz — єдине місце з replicate models;
- primary summary files містять лише bootstrap method, count, seed, parameter summaries, CI, invalid fraction та reference/hash на full artifact;
- `experiment_b_metadata` не повинен embed-ити calibration bootstrap replicates;
- `linear_fit.json` не повинен дублювати весь Experiment A summary;
- великі raw observations винести в окремий artifact із hash;
- canonical content hash має дозволяти перевірити, що analyzer використав правильний calibration artifact.

Не ламати можливість повністю відтворити метрики офлайн.

## 14. Schema additions

Рекомендована структура calibration artifact:

```json
{
  "schema_version": 4,
  "shape_policy": {
    "name": "ceil_to_stride",
    "stride": 32,
    "numeric_dtype": "float32"
  },
  "calibration_envelope": {},
  "latency_models": {
    "shape_lookup": {
      "statistic": "trimmed_mean",
      "trim_fraction_each_tail": 0.1,
      "table": {
        "160x160": {
          "point_s": 0.0,
          "standard_error_s": 0.0,
          "ci_s": [0.0, 0.0],
          "repetitions": 0
        }
      }
    },
    "linear_area": {},
    "quadratic_area": {},
    "piecewise_area": {}
  },
  "bootstrap_reference": {
    "path": "bootstrap_models.json",
    "sha256": "..."
  },
  "provenance": {}
}
```

Рекомендовані additions у decision result:

```json
{
  "evaluation_design": "representative|challenge",
  "aggregate_scope": "reference_distribution|unweighted_challenge_average",
  "model_comparison": {},
  "metrics_by_stratum": {},
  "agreement_matrix": {},
  "control_diagnostics": {
    "within_session_drift": {},
    "calibration_to_validation_shift": {}
  },
  "sampling_diagnostics": {},
  "artifact_references": {}
}
```

## 15. Tests

Додати або оновити тести.

### 15.1. Shape/stride

1. `effective_shape(161, 319, 32) == (192, 320)`.
2. Округлюються сторони, не area.
3. `(H,W)` і `(W,H)` залишаються різними lookup keys.
4. Усі generated accepted shapes кратні 32.
5. Усі accepted shapes є у shape lookup table.
6. Missing lookup entry у strict validation mode спричиняє fail fast.

### 15.2. Models

7. Linear direct-cost еквівалентний `delta_area < tau`.
8. Piecewise неперервний у breakpoint.
9. Piecewise area limit збігається з direct-cost decision по обидва боки breakpoint.
10. Усі area models monotonic на calibration domain; invalid slope відхиляється.
11. Shape lookup повертає точне значення для ordered shape.
12. Tie та insufficient margin дають `separate`.

### 15.3. Evaluation

13. Representative і challenge metrics не змішуються.
14. Unweighted challenge metric має явний scope.
15. Weighted metric використовує лише задані external/reference weights.
16. Ambiguous pairs виключені з confusion matrix, але можуть входити до `regret_all_pairs`.
17. Bootstrap resampling зберігає paired/order-block structure.
18. `quality_warnings` не містить дублікатів.
19. `near_pair_count` має валідну задокументовану семантику або відсутній у schema v4.
20. Full bootstrap replicates не embed-яться у metadata/summary JSON.

### 15.4. Reproducibility

21. Calibration grid deterministically містить усі in-domain stride-aligned shapes.
22. Однаковий seed дає однаковий schedule та pair specs.
23. Pair specs hash перевіряється до measurement та analysis.
24. Fresh Experiment B не може випадково використати pair file, на якому підбирали policy/margin, без explicit override і warning.

## 16. План повторного експерименту

### Phase A — повна shape calibration

1. Згенерувати повний in-domain stride-32 grid.
2. Запустити randomized schedule з controls.
3. Побудувати `shape_lookup` та area-only baselines.
4. Перевірити table coverage, uncertainty та control drift.
5. Заморозити calibration artifact і hashes.

### Phase B — policy declaration

До перегляду нового pair outcome зафіксувати:

- primary policy;
- candidate policies;
- confidence level;
- decision margin;
- representative sampling distribution;
- challenge strata/quotas;
- repetitions/stopping rule;
- primary metric: бажано mean latency regret плюс p95;
- secondary classification metrics.

### Phase C — fresh validation

1. Згенерувати новий `pair_specs` з іншим seed.
2. Не повторно використовувати поточні 500 pairs для confirmatory result.
3. Запустити representative та challenge sets окремо.
4. Порахувати всі policies на тих самих measurements у межах кожного set.
5. Окремо звітувати session shift, order effect та ambiguous fraction.

## 17. Acceptance criteria

Завдання виконане, коли:

- piecewise не використовується як default лише через кращий fit у Experiment A;
- є повна stride-32 shape lookup table для всього primary domain;
- production decision використовує ordered `(H,W)`, а не тільки `H*W`;
- validation set не має missing lookup або silent fallback;
- representative і challenge evaluations розділені;
- conservative policy та margin зафіксовані до fresh validation;
- order-aware paired CI реалізовано коректно;
- warnings не дублюються;
- `near_pair_count` виправлено/прибрано;
- bootstrap/raw artifacts не дублюються у великих summary JSON;
- усі тести проходять;
- coding agent надає команди запуску, перелік змінених файлів, migration notes і результати fresh smoke test.

## 18. Що не робити

- Не оголошувати piecewise переможцем за Experiment A fit metrics.
- Не оголошувати linear остаточним переможцем лише за pooled metrics цього challenge run.
- Не підбирати margin або ensemble rule на поточних 500 pairs і потім звітувати їх як незалежну accuracy.
- Не зводити piecewise до одного universal `tau`.
- Не округляти лише area замість `H` і `W`.
- Не вважати `H×W` і `W×H` еквівалентними на CPU.
- Не запускати 2000 bootstrap models у runtime hot path.
- Не змішувати stride alignment із INT8 quantization у назвах або документації.
- Не приховувати `hardware_state_shift`, `order_effect_warning` або high ambiguous fraction.

## 19. Очікуваний звіт coding agent

Після змін надати:

1. короткий root-cause summary;
2. перелік змінених файлів;
3. опис canonical effective-shape path;
4. розмір і coverage shape lookup table;
5. формати нових artifacts та migration behavior;
6. точні команди Experiment A, representative B і challenge B;
7. команди тестів та результати;
8. підтвердження, що current Experiment B не використовувався для confirmatory tuning;
9. відомі обмеження;
10. окремий список exploratory findings, які ще потребують fresh validation.
