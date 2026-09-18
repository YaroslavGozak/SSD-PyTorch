# Інструкція для coding agent: оновлення latency-based ROI merge та Experiment B v2

## 1. Роль і мета

Потрібно проаналізувати наявний репозиторій і внести зміни до коду калібрування та парної перевірки ROI merge. Головна мета — замінити практичне рішення, прив’язане лише до скалярного порога `tau = K / c`, на загальне порівняння прогнозованої вартості обчислень:

```text
merge, якщо T_hat(A_merged) < T_hat(A_1) + T_hat(A_2)
```

Після цього на **одному й тому самому наборі виміряних пар** треба порівняти три незалежно відкалібровані правила:

1. `linear_tau` — поточна лінійна модель та еквівалентний поріг `tau = K / c`;
2. `quadratic_direct_cost` — пряма оцінка двох варіантів квадратичною latency model;
3. `piecewise_direct_cost` — пряма оцінка двох варіантів piecewise-linear latency model.

Одночасно треба усунути неоднозначність domain counters і зменшити вплив порядку запусків, прогріву, кешу та апаратного drift на Experiment B.

## 2. Наукові обмеження, які не можна порушувати

- Experiment A є **єдиним джерелом коефіцієнтів latency models**.
- Experiment B є незалежною валідацією. Заборонено підбирати на ньому коефіцієнти моделей, breakpoint або робочий threshold.
- Не зашивати в код емпіричний поріг близько `200–210 kpx`. Він є діагностичним результатом попереднього Experiment B, а не параметром нової моделі.
- Усі decision rules мають оцінюватися на тих самих `pair_id`, геометріях, tensor shapes і виміряних latency samples.
- Основні classification metrics рахувати тільки за determinate pairs. Пари з label `ambiguous` не переводити штучно в один із класів.
- Значення площі має відповідати **фактичній обчислювальній площі tensor після всіх resize/padding/stride-rounding операцій**, а не сирій площі ROI у вихідному кадрі.
- Не змінювати preprocessing або спосіб inference між альтернативами `merged` і `separate`, крім кількості та розмірів tensor, що є предметом порівняння.

## 3. Спочатку дослідити репозиторій

Перед редагуванням:

1. Знайти entry points Experiment A та Experiment B, структури конфігурації, серіалізацію JSON і поточну реалізацію `tau`/`predicted_merge`.
2. Знайти єдине місце, де requested ROI shape перетворюється на actual tensor shape. Не дублювати цю логіку.
3. Знайти код warmup, синхронізації пристрою, таймера, bootstrap CI, генерації пар і control measurements.
4. Запустити наявні тести до змін і зафіксувати baseline. Не виправляти сторонні проблеми, не пов’язані з цим завданням.
5. Зберегти сумісність із поточними calibration/result JSON настільки, наскільки це можливо. Якщо потрібна міграція схеми, додати `schema_version` і явний compatibility path.

Назви модулів у цій інструкції концептуальні. Адаптувати їх до реальної архітектури репозиторію, не створюючи паралельних дубльованих реалізацій.

## 4. Уніфікований інтерфейс latency model

Створити невеликий спільний інтерфейс/протокол для моделей. Мінімально він має надавати:

```python
predict_seconds(effective_area: int) -> float
is_in_domain(tensor_h: int, tensor_w: int) -> bool
metadata() -> dict
```

За можливості використати наявні класи або dataclass. Валідувати коефіцієнти під час завантаження: усі обов’язкові поля мають бути присутніми, числовими та finite; площа повинна бути додатною; прогноз часу також має бути finite.

### 4.1. Linear model

```text
T_hat(A) = K + c * A
```

Правило direct-cost:

```text
T_hat(A_merged) < T_hat(A_1) + T_hat(A_2)
```

має бути математично еквівалентне:

```text
A_merged - A_1 - A_2 < K / c
```

Залишити `tau_pixels = K / c` у metadata та звітах для інтерпретації й backward compatibility, але реалізація порівняння бажано має проходити через спільний direct-cost API. Для точного tie використовувати детерміноване рішення `separate`; не вводити довільний великий epsilon.

### 4.2. Quadratic model

Якщо Experiment A вже зберігає валідну quadratic model, реалізувати:

```text
T_hat(A) = b0 + b1 * A + b2 * A^2
```

Для quadratic model не виводити й не використовувати один універсальний `tau`: рішення залежить від усіх трьох площ. Якщо calibration artifact не містить потрібних quadratic coefficients, **не вигадувати їх і не навчати на Experiment B**. У такому разі позначити правило як `unavailable` із чіткою причиною в результаті або додати збереження quadratic coefficients у Experiment A.

### 4.3. Piecewise-linear model

Реалізувати hinge form:

```text
T_hat(A) = b0 + b1 * A + b2 * max(0, A - breakpoint_area)
```

Тобто slope до breakpoint дорівнює `b1`, а після breakpoint — `b1 + b2`. Не замінювати piecewise model одним наближеним threshold.

Попередні коефіцієнти YOLO26n можна використати лише у unit-test fixture або sanity check, але не як production constants:

```text
b0 = 0.0156894 s
b1 = 1.91548e-7 s/pixel
b2 = -9.38511e-8 s/pixel
breakpoint_area = 123904 pixels
```

Production run повинен завантажувати коефіцієнти з конкретного calibration artifact Experiment A і записувати ідентифікатор/хеш цього artifact у результат Experiment B.

## 5. Єдина функція прийняття рішення

Реалізувати одну pure function, яку використовують усі моделі:

```python
def decide_merge(model, shape_1, shape_2, merged_shape):
    a1 = effective_area(shape_1)
    a2 = effective_area(shape_2)
    am = effective_area(merged_shape)

    merged_cost_s = model.predict_seconds(am)
    separate_cost_s = model.predict_seconds(a1) + model.predict_seconds(a2)
    predicted_gain_s = separate_cost_s - merged_cost_s

    return {
        "predicted_merge": merged_cost_s < separate_cost_s,
        "predicted_merged_cost_s": merged_cost_s,
        "predicted_separate_cost_s": separate_cost_s,
        "predicted_gain_s": predicted_gain_s,
        "effective_areas": {"a1": a1, "a2": a2, "merged": am},
    }
```

Обчислення `shape -> effective_area` має використовувати фактичні `tensor_h` і `tensor_w`. Для linear rule додатково зберігати:

```text
delta_area = A_merged - A_1 - A_2
tau_pixels
```

## 6. Calibration artifact Experiment A

Розширити або нормалізувати результат Experiment A так, щоб Experiment B міг без повторного fit завантажити:

- linear coefficients, `tau_pixels` і bootstrap CI;
- quadratic coefficients, якщо модель підтримується;
- piecewise coefficients і `breakpoint_area`;
- calibration envelope;
- одиниці вимірювання;
- назву моделі, weights/version, device, precision, preprocessing parameters;
- timestamp, git commit, dirty flag, random seed;
- `schema_version`.

Бажана структура:

```json
{
  "schema_version": 2,
  "latency_models": {
    "linear": {
      "formula": "b0 + b1*A",
      "coefficients": {"b0_s": 0.0, "b1_s_per_pixel": 0.0},
      "tau_pixels": 0.0,
      "fit_metrics": {}
    },
    "quadratic": {
      "formula": "b0 + b1*A + b2*A^2",
      "coefficients": {
        "b0_s": 0.0,
        "b1_s_per_pixel": 0.0,
        "b2_s_per_pixel2": 0.0
      },
      "fit_metrics": {}
    },
    "piecewise": {
      "formula": "b0 + b1*A + b2*max(0,A-B)",
      "coefficients": {
        "b0_s": 0.0,
        "b1_s_per_pixel": 0.0,
        "b2_s_per_pixel": 0.0,
        "breakpoint_area": 0
      },
      "fit_metrics": {}
    }
  },
  "calibration_envelope": {},
  "provenance": {}
}
```

Не обов’язково дослівно копіювати цю схему, якщо в репозиторії вже є стабільний формат. Важливо, щоб значення були однозначними, з одиницями і без повторного fit у Experiment B.

## 7. Calibration domain і правильна семантика counters

Створити одну функцію перевірки envelope. Вона має враховувати:

- `min/max_tensor_h`;
- `min/max_tensor_w`;
- `min/max_effective_area`;
- `min/max_aspect_ratio`, де aspect ratio визначений однозначно і задокументовано.

Для поточного calibration envelope YOLO26n перевірити, зокрема:

```text
128 <= H,W <= 640
25600 <= H*W <= 409600
0.5 <= W/H <= 2.0
```

За `domain_policy = reject_out_of_domain`:

- не допускати OOD shape до прийнятих pair records;
- `out_of_calibration_pair_count` у фінальному наборі має бути `0`;
- відхилені candidates рахувати окремо як `rejected_out_of_domain_candidate_count`;
- `out_of_calibration_pair_count` має означати кількість **pair records**, у яких хоча б одна з трьох shapes OOD, а не кількість repetitions або inference calls;
- за потреби окремо зберігати `out_of_calibration_shape_count` і `out_of_calibration_inference_sample_count`, але не змішувати ці сутності;
- для unique counters використовувати множину стабільних shape/pair keys, а не збільшувати лічильник у циклі repetitions.

Якщо потрібен окремий robustness run поза envelope, він повинен мати `domain_policy = allow_and_flag`, окрему назву/секцію результату і не змішуватися з primary validation.

Прибрати `64x64` із primary control shapes, оскільки площа `4096` лежить поза наведеним calibration envelope. Мінімальний control shape має одночасно задовольняти всі обмеження; наприклад, `160x160`. Перед запуском перевіряти весь список control shapes тією самою функцією domain validation.

## 8. Генерація спільного набору пар

1. Згенерувати і зберегти список pair specifications один раз із фіксованим seed.
2. Виміряти кожну пару один раз за єдиним measurement protocol.
3. Застосувати всі decision rules офлайн до того самого pair record.
4. Не генерувати окремі «вигідні» набори під кожну модель.
5. Зберігати для кожної пари requested shapes, actual tensor shapes, effective areas, geometry type, seed-derived `pair_id` і computational key.

Поточні boundary bins (`merge`, `near_low`, `near_high`, `separate`) можна залишити як stratification відносно **linear tau**, щоб порівняння з попереднім run було зрозумілим. Тоді обов’язково записати:

```text
sampling_basis = linear_tau
sampling_tau_pixels = ...
```

Додатково бажано звітувати метрики по geometry type, area regime (усі площі до breakpoint / перетин breakpoint / усі після breakpoint) і model-disagreement region. Це допоможе показати, де саме нелінійні моделі кращі.

## 9. Стабілізація timing protocol

Попередній run показав великий order effect (`~3.95 ms`) і `hardware_state_shift = true`, тому Experiment B v2 має використовувати збалансований paired protocol.

### 9.1. Вимірювальна одиниця

- `M` (`merged`) — один inference merged tensor.
- `S` (`separate`) — сума двох послідовних inference для tensor 1 і tensor 2.
- Один paired sample містить одне вимірювання `M` та одне вимірювання `S`.
- Зберігати raw paired differences `D = S - M`. Позитивне `D` означає, що merge швидший.

### 9.2. Порядок

Використати збалансовані блоки, наприклад `M,S,S,M` і `S,M,M,S` (ABBA/BAAB), та випадково перемішувати порядок блоків із фіксованим seed. Кількість paired samples за кожним first-order condition повинна відрізнятися не більше ніж на один.

Не вимірювати спочатку всі merged repetitions, а потім усі separate repetitions. Не робити різний warmup для альтернатив.

### 9.3. Warmup і синхронізація

- Залишити global warmup.
- Додати симетричний warmup для нових effective shapes, якщо backend має значний first-call/JIT/cache effect.
- Warmup samples не включати до statistics.
- Для GPU синхронізувати device до старту і після inference перед зупинкою таймера.
- Для CPU використовувати monotonic high-resolution timer, наприклад `time.perf_counter_ns()`.
- Виконувати inference у відповідному no-grad/inference mode.
- Зафіксувати thread settings, precision, device, power/performance mode, preprocessing і `timing_mode` у metadata.

### 9.4. Control measurements і drift

- Виконувати control block через фіксований інтервал pair blocks, а не тільки на початку/кінці.
- Усі primary control shapes мають бути всередині calibration envelope.
- Зберігати raw control observations із `pair_completed`, shape, measured latency і prediction кожної latency model.
- Окремо рахувати drift/order diagnostics для in-domain controls.
- Якщо CPU frequency або temperature недоступні, залишити `null` і додати явний availability flag; не підставляти оцінені значення.
- `hardware_state_shift` має визначатися задокументованим правилом, а не вручну.

## 10. Ground-truth label для пари

Для кожної пари обчислити bootstrap confidence interval для середньої paired difference:

```text
D = measured_separate_ms - measured_merged_ms
```

Label:

```text
merge_beneficial     if CI_low > 0
separate_beneficial  if CI_high < 0
ambiguous            otherwise
```

Зберігати `mean_difference_ms`, `difference_ci_ms`, кількість paired samples і bootstrap parameters. Bootstrap має ресемплювати paired observations, не окремі merged і separate samples незалежно.

## 11. Метрики для трьох decision rules

Для кожної доступної моделі на однакових determinate pairs обчислити:

- confusion matrix `tp`, `tn`, `fp`, `fn`, де positive class — `merge_beneficial`;
- accuracy та bootstrap CI;
- balanced accuracy;
- precision/recall/F1 для merge;
- specificity;
- метрики за boundary bin;
- метрики за geometry type;
- метрики за area regime відносно piecewise breakpoint.

Для latency regret використати:

```text
chosen_ms = measured_merged_ms, якщо predicted_merge, інакше measured_separate_ms
oracle_ms = min(measured_merged_ms, measured_separate_ms)
regret_ms = max(0, chosen_ms - oracle_ms)
```

Звітувати щонайменше `mean`, `median`, `p95`, `p99`, `max` окремо:

- на всіх парах за pair mean;
- на determinate pairs.

Не використовувати sign label для обчислення regret: regret визначається безпосередньо виміряними mean latency. Чітко позначити scope кожної aggregate metric.

Aggregate bootstrap CI отримувати ресемплінгом pair records, а не окремих inference samples з різних пар.

## 12. Формат результату Experiment B v2

Зберегти raw pair data достатньо детально, щоб усі decision metrics можна було перерахувати без нового inference. Приклад:

```json
{
  "schema_version": 2,
  "calibration_reference": {
    "path_or_id": "...",
    "content_hash": "...",
    "experiment_a_session_id": "..."
  },
  "sampling": {
    "seed": 0,
    "sampling_basis": "linear_tau",
    "sampling_tau_pixels": 0.0,
    "boundary_quotas": {}
  },
  "measurement_protocol": {
    "timing_mode": "inference_only",
    "order_design": "balanced_abba_baab",
    "warmup": {},
    "bootstrap": {}
  },
  "pair_summaries": [
    {
      "pair_id": "...",
      "tensor_shapes": {
        "first": [0, 0],
        "second": [0, 0],
        "merged": [0, 0]
      },
      "effective_areas": {"first": 0, "second": 0, "merged": 0},
      "delta_area": 0,
      "measurements": {
        "merged_mean_ms": 0.0,
        "separate_mean_ms": 0.0,
        "mean_difference_ms": 0.0,
        "difference_ci_ms": [0.0, 0.0]
      },
      "label": "ambiguous",
      "predictions": {
        "linear_tau": {},
        "quadratic_direct_cost": {},
        "piecewise_direct_cost": {}
      }
    }
  ],
  "model_comparison": {},
  "domain_diagnostics": {},
  "order_and_drift_diagnostics": {},
  "provenance": {}
}
```

Для backward compatibility поле `predicted_merge` у pair summary можна тимчасово залишити як alias до `predictions.linear_tau.predicted_merge`, але новий код аналізу повинен використовувати словник `predictions`.

## 13. CLI/config

Додати або оновити параметри так, щоб запуск явно приймав:

- шлях до calibration artifact Experiment A;
- перелік decision rules;
- domain policy;
- seed;
- кількість pairs і repetitions/paired samples;
- boundary quotas;
- control shapes та control interval;
- bootstrap repetitions і confidence level;
- output path.

Fail fast з чітким повідомленням, якщо calibration artifact несумісний, не містить обраної моделі або має іншу model/device/preprocessing identity, ніж поточний inference run. Якщо mismatch дозволяється окремим override, записати `override_used = true` і точну причину.

## 14. Тести

Додати щонайменше такі unit/integration tests:

1. **Linear equivalence:** direct-cost decision дорівнює `delta_area < K/c` для багатьох випадкових додатних площ.
2. **Strict tie:** при рівних predicted costs рішення — `separate`.
3. **Piecewise formula:** значення до, на і після breakpoint збігаються з ручним розрахунком; функція неперервна в breakpoint.
4. **Quadratic formula:** direct-cost використовує всі три площі й не звертається до scalar tau.
5. **Effective area:** використовується actual tensor shape після rounding, не requested/raw ROI area.
6. **Domain boundaries:** значення на межі приймаються; кожне порушення H/W/area/aspect ratio правильно відхиляється або позначається.
7. **Strict domain counters:** за `reject_out_of_domain` accepted OOD pair count дорівнює нулю, rejected candidate counter зростає один раз на candidate, не на repetition.
8. **Same-pair evaluation:** усі правила отримують ідентичний ordered список `pair_id`.
9. **Determinism:** однаковий seed дає однакові pair specs і order schedule.
10. **Order balance:** кількість `merged_first` і `separate_first` paired samples відрізняється не більше ніж на один.
11. **Paired bootstrap:** ресемплюються індекси парних observations.
12. **Ambiguous exclusion:** ambiguous pairs не входять до confusion matrix, але можуть входити до явно позначеного `regret_all_pairs`.
13. **Backward compatibility:** старий calibration/result JSON або коректно читається, або відхиляється з конкретним migration message.
14. **Small end-to-end smoke test:** короткий Experiment B створює валідний JSON, який повторно аналізується без inference.

## 15. Acceptance criteria

Завдання виконане, коли:

- усі три правила (за наявності коефіцієнтів) працюють через спільний direct-cost interface;
- linear rule зберігає аналітичний `tau`, але piecewise і quadratic не редукуються до нього;
- coefficients не fit-яться на Experiment B;
- один measured pair dataset використовується для всіх моделей;
- primary run із `reject_out_of_domain` не містить accepted OOD pairs;
- control shapes primary run лежать у calibration envelope;
- порядок вимірювань збалансований та відтворюваний;
- JSON містить per-model predictions, comparison metrics, regret, domain/order/drift diagnostics і provenance;
- тести проходять;
- старі поля не змінюють семантику мовчки.

Не використовувати конкретне покращення accuracy як pass/fail criterion: новий run може відрізнятися через hardware state. Для sanity check попередній аналіз давав приблизно:

```text
linear_tau:              accuracy 78.6%, balanced accuracy 82.2%, mean regret 1.51 ms
piecewise_direct_cost:   accuracy 93.4%, balanced accuracy 95.1%, mean regret ~0.53 ms
```

Ці числа потрібні лише для виявлення грубих помилок у формулах або sign convention. Їх не можна hardcode у тестах і не можна гарантувати для нового вимірювання.

## 16. Очікуваний звіт coding agent

Після реалізації надати:

1. короткий опис архітектурних змін;
2. перелік змінених файлів;
3. точні команди запуску Experiment A та Experiment B v2;
4. точні команди тестів і їх результат;
5. приклад нового JSON output;
6. пояснення compatibility/migration behavior;
7. відомі обмеження або невирішені питання;
8. підтвердження, що Experiment B не використовувався для fit параметрів.

Не робити великих сторонніх рефакторингів. Якщо структура репозиторію робить якийсь пункт недоречним, реалізувати найближчий семантично еквівалентний варіант і явно пояснити відхилення у фінальному звіті.
