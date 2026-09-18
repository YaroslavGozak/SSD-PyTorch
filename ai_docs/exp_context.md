# Контекст для реалізації експерименту з валідації cost model

## 1. Завдання агента

Потрібно написати відтворюваний Python-код для двох синтетичних експериментів:

1. **Experiment A — validation of the inference-time cost model.** Перевірити, наскільки добре фактичний час одного запуску детектора описується моделлю

   $$
   T(A)=K_t+c_tA,
   $$

   де $A$ — фактична площа вхідного тензора після всіх resize, letterbox і stride-rounding, $K_t$ — стала вартість окремого запуску, $c_t$ — вартість обробки одного пікселя.

2. **Experiment B — validation of the merge decision.** Перевірити, чи правильно оцінена модель вибирає швидший спосіб обробки двох ROI: два окремі запуски або один запуск для охоплювального прямокутника.

Експеримент не потребує відео, датасету, анотацій, трекера або розрахунку точності детекції. Використовуються синтетичні тензори або синтетичні зображення. Досліджується лише обчислювальний час.

Початковий результат роботи агента має бути кодом, конфігурацією, тестами та коротким README з командами запуску. Не змінювати код основного відеобенчмарку, якщо для інтеграції достатньо окремого модуля в `experiments/`.

## 2. Науковий контекст

Стаття пропонує cost-aware merging ROI. Для $n$ окремих ROI:

$$
T_{sep}=\sum_{i=1}^{n}(K_t+c_tA_i).
$$

Для одного охоплювального прямокутника з площею $A_U$:

$$
T_{merge}=K_t+c_tA_U.
$$

Злиття вважається доцільним, якщо

$$
A_U < \sum_{i=1}^{n}A_i+(n-1)\tau,
\qquad
\tau=\frac{K_t}{c_t}.
$$

Для пари ROI критерій еквівалентний:

$$
\Delta A=A_U-A_1-A_2<\tau.
$$

Рецензент вказав, що стаття не подає достатньої валідації цієї моделі: потрібні goodness-of-fit, residual analysis, confidence intervals, перевірка різних розмірів ROI та hardware states, а також порівняння з piecewise-linear або nonlinear моделями.

Наявні числа зі статті слід використовувати лише як sanity check, а не як очікувану відповідь:

- YOLO26n, Raspberry Pi 5: $c_t=0.81\cdot10^{-6}$ s/pixel, $K_t=0.0186$ s, $\tau\approx22\,935$ pixels;
- незалежне профілювання fixed input sizes: $\tau\approx22\,700$ pixels;
- online adaptive estimate: переважно $20\,000$–$27\,000$ pixels;
- в іншій таблиці основного відеобенчмарку greedy merging запускався з $\tau=3100$.

Остання розбіжність не повинна бути прихована або автоматично «виправлена». Новий код має незалежно оцінити $K_t$, $c_t$ і $\tau$ для кожної чітко визначеної межі таймінгу. Не використовувати жодне з наведених значень як ground truth.

## 3. Пріоритетна платформа і інтеграція

Основна цільова конфігурація — **YOLO на Raspberry Pi 5, CPU, batch size 1**. У рукописі модель названа `YOLO26n`, але код не повинен мовчки припускати назву weights або версію бібліотеки.

Перед реалізацією агент має:

1. Переглянути наявний репозиторій і знайти поточний model wrapper, preprocessing та код вимірювання inference time.
2. Повторно використати той самий шлях виклику моделі, що застосовується для online-оцінювання $\tau$, якщо він доступний.
3. Винести конкретні model weights, device, thread count, image limits і timing mode у конфігурацію або CLI.
4. Якщо репозиторій не містить потрібного wrapper, створити невеликий adapter interface та реалізацію для Ultralytics YOLO. Не прив'язувати весь аналіз до Ultralytics API.

Рекомендований мінімальний інтерфейс адаптера:

```python
class DetectorAdapter(Protocol):
    stride: int

    def prepare(self, image: np.ndarray, requested_hw: tuple[int, int]) -> PreparedInput:
        """Return preprocessed input and the actual tensor HxW."""

    def infer(self, prepared: PreparedInput) -> Any:
        """Run model forward pass without file I/O or plotting."""

    def postprocess(self, raw_output: Any) -> Any:
        """Optional postprocessing/NMS for end-to-end timing."""
```

Якщо фактичний pipeline має інший API, зберегти його, але забезпечити ті самі логічні межі таймінгу.

## 4. Критично важливе визначення площі

Незалежна змінна — не геометрична площа crop до preprocessing, а

$$
A_{effective}=H_{tensor}\cdot W_{tensor},
$$

де `H_tensor` і `W_tensor` — фактичні просторові розміри тензора, переданого моделі.

Код повинен логувати одночасно:

- requested ROI width/height and area;
- crop width/height перед preprocessing;
- actual tensor width/height після resize, letterbox та stride-rounding;
- effective area;
- model stride.

Не можна лише обчислити `ceil(size / stride) * stride` і вважати це фактичним розміром, якщо preprocessing може масштабувати або letterbox-ити зображення. Розмір треба зчитати з підготовленого тензора. Якщо всі конфігурації несподівано дають однаковий tensor shape, benchmark повинен завершитися з чіткою помилкою: такий запуск не перевіряє залежність часу від площі.

## 5. Межі таймінгу

Потрібно підтримати щонайменше два режими та ніколи не змішувати їх в одній регресії:

1. `inference_only`: лише model forward на вже підготовленому тензорі;
2. `detector_call`: preprocessing + model forward + штатний postprocessing/NMS для одного ROI.

Основним режимом для висновку має бути той, що точно відповідає timing boundary чинного online estimator. Якщо це неможливо встановити з репозиторію, обидва режими є обов'язковими, а README має явно зазначити невизначеність.

Додатково логувати компоненти окремо, коли це можливо:

- `preprocess_ms`;
- `inference_ms`;
- `postprocess_ms`;
- `detector_call_ms`;
- `wall_clock_trial_ms`.

Використовувати `time.perf_counter_ns()`. Для CPU не потрібна синхронізація пристрою. Якщо adapter згодом запускається на CUDA, перед початком і завершенням timed block потрібна відповідна device synchronization.

Не включати у timed block:

- генерацію синтетичного зображення;
- запис CSV/JSON;
- побудову графіків;
- імпорт модулів і завантаження weights;
- збір системних метаданих, якщо вони не є частиною реального detector call.

## 6. Experiment A — дизайн збору даних

### 6.1. Конфігурації ROI

Взяти 20–30 рівнів площі, які рівномірно покривають робочий діапазон від приблизно 5% до 100% максимальної дозволеної площі. Максимальні width/height мають задаватися конфігурацією та відповідати реальному pipeline; не хардкодити 300×300, 640×640 або розмір відеокадру без перевірки.

Для кожного рівня площі сформувати щонайменше три aspect ratios:

- `1:1`;
- `2:1`;
- `1:2`.

Крайні значення мають бути обрізані до конфігурованих меж. Після фактичного preprocessing видалити дублікати за ключем `(tensor_h, tensor_w)`, але зберегти інформацію, скільки requested shapes дали кожен effective shape.

Бажано дозволити два способи задання сітки:

- `area_fractions`: автоматична генерація за частками максимальної площі;
- `explicit_shapes`: точний список `HxW` для відтворення fixed-size profiling.

### 6.2. Синтетичний вміст

За фіксованим seed створити детерміноване RGB-зображення або тензор з псевдовипадковою текстурою. Не генерувати нові random pixels усередині timed block. Для `inference_only` допустимо кешувати prepared inputs усіх effective shapes, якщо це не створює неприйнятного використання RAM.

Для `detector_call` передавати синтетичне зображення відповідного requested shape через реальний preprocessing. Через можливу залежність NMS від вмісту використовувати однаковий тип текстури для всіх shapes і логувати кількість predictions до та після NMS, якщо wrapper це дозволяє.

### 6.3. Warm-up і повтори

Значення за замовчуванням:

- `global_warmup_iterations: 50`;
- `repetitions_per_effective_shape: 40`;
- не менше 30 повторів у фінальному запуску;
- batch size 1;
- `model.eval()` і режим без градієнтів.

Збір організувати блоками: кожен `run_id` містить по одному вимірюванню кожної effective shape, а порядок shapes усередині блоку рандомізується. Не вимірювати всі малі, а потім усі великі ROI, оскільки temperature/DVFS drift може штучно корелювати з площею.

Перед першим timed measurement кожного нового shape дозволити один untimed shape-specific warm-up, щоб не включити одноразову компіляцію або allocation. Якщо backend має shape-dependent compilation/cache, логувати це в metadata.

### 6.4. Hardware states

Мінімальний науково коректний запуск — один стабільний warmed-up state. Бажано підтримати кілька незалежних sessions, наприклад:

- `warm_stable`;
- `long_run_or_hot`;
- інший governor/power mode, лише якщо користувач задає його вручну.

Код не повинен сам змінювати governor, CPU affinity, power mode або системні ліміти. Він лише читає і логує доступні параметри:

- timestamp and elapsed time;
- CPU temperature;
- current/mean CPU frequency;
- CPU governor, якщо доступний;
- logical/physical CPU count;
- thread-related environment variables;
- process RSS;
- device and OS information.

Відсутність окремого системного сенсора не повинна зупиняти benchmark; записати `null` і warning.

## 7. Experiment A — статистичний аналіз

Аналіз не повинен відкидати outliers без прозорого правила. Зберігати raw observations завжди. Якщо надається filtered analysis, він має бути лише додатковим, із зафіксованим наперед правилом і кількістю відкинутих точок.

### 7.1. Основна лінійна модель

Для кожного `session_id × timing_mode` окремо fit:

$$
T=K_t+c_tA.
$$

Звітувати:

- $K_t$ та $c_t$;
- $\tau=K_t/c_t$;
- $R^2$ і adjusted $R^2$;
- RMSE і MAE;
- 95% confidence intervals для $K_t$ і $c_t$;
- 95% bootstrap CI для $\tau$;
- кількість observations та unique effective shapes.

Для uncertainty використовувати block bootstrap за `run_id` або session-level blocks, а не наївний bootstrap окремих correlated observations. Для $\tau$ зберігати лише bootstrap fits із додатними $K_t$ і $c_t$, але обов'язково повідомляти частку invalid bootstrap fits. Не маскувати нестабільність ratio.

Додати grouped cross-validation за effective shape, наприклад `GroupKFold`, і звітувати out-of-sample RMSE/MAE. Група — `(tensor_h, tensor_w)`, щоб повтори того самого shape не потрапляли одночасно в train і validation.

### 7.2. Альтернативні моделі

На тих самих observations порівняти:

1. Linear:

   $$T=b_0+b_1A.$$

2. Quadratic:

   $$T=b_0+b_1A+b_2A^2.$$

3. Continuous piecewise-linear with one breakpoint:

   $$
   T=b_0+b_1A+b_2\max(0,A-A_0).
   $$

Breakpoint $A_0$ обирати прозорим grid search лише серед внутрішніх 10–90% unique areas за мінімальним RSS. Під час розрахунку AIC/BIC врахувати вибір breakpoint як параметр. Не використовувати test data для вибору breakpoint у grouped cross-validation.

Для кожної моделі зберегти:

- $R^2$ і adjusted $R^2$;
- RMSE, MAE;
- AIC, BIC;
- grouped-CV RMSE/MAE;
- fitted parameters;
- breakpoint для piecewise model.

Не вводити довільний критерій на кшталт `R² >= 0.95 = модель доведена`. Код повинен чесно показати результат незалежно від того, яка модель краща.

### 7.3. Residual analysis

Згенерувати щонайменше:

- measured time vs effective area із linear fit і 95% CI;
- residuals vs effective area;
- residuals vs fitted value;
- residual distribution/QQ plot;
- latency by aspect ratio for comparable effective areas;
- residuals or latency vs temperature and CPU frequency, якщо ці дані доступні.

У summary додати кореляцію residuals із `tensor_h/tensor_w`, temperature, frequency та elapsed time. Це допоможе виявити shape effect і thermal/DVFS drift.

## 8. Experiment B — синтетичні пари ROI

### 8.1. Геометрія

Після отримання $\hat\tau$ з Experiment A згенерувати 500–1000 пар прямокутних ROI на конфігурованому canvas. Варіювати:

- розмір кожного ROI;
- aspect ratio;
- overlap;
- horizontal separation;
- vertical separation;
- diagonal separation;
- containment/partial overlap, якщо geometry generator це підтримує.

Для кожної пари обчислити найменший axis-aligned bounding rectangle $R_U$ та фактичні effective areas трьох model inputs:

$$
A_1^{eff},\quad A_2^{eff},\quad A_U^{eff}.
$$

Decision feature:

$$
\Delta A^{eff}=A_U^{eff}-A_1^{eff}-A_2^{eff}.
$$

Прогноз моделі:

```text
merge iff delta_effective_area < tau_hat
```

Важливо: для критерію використовувати effective tensor areas кожного окремого виклику, а не лише raw geometric areas.

### 8.2. Покриття decision boundary

Розподіл випадкових пар часто дасть забагато очевидних прикладів. Генератор має стратифікувати або rejection-sample пари за

$$
r=\Delta A^{eff}/\hat\tau.
$$

Щонайменше 40% пар повинні бути поблизу межі, наприклад $0.8\le r\le1.2$. Решту рівномірно розподілити між очевидно вигідним і невигідним злиттям. Зберегти `geometry_type` і `boundary_bin` для подальшого аналізу.

Якщо $\hat\tau$ не є додатним або його bootstrap CI патологічно нестабільний, Experiment B не повинен мовчки продовжуватися. Завершити з діагностикою або дозволити явний `--tau-override`, записавши override у metadata.

### 8.3. Фактичне вимірювання рішення

Для кожної пари порівняти однакову timing boundary:

$$
T_{sep}^{actual}=T(R_1)+T(R_2)
$$

та

$$
T_{merge}^{actual}=T(R_U).
$$

Зробити 20–30 paired repetitions. Усередині кожного repetition рандомізувати, що вимірюється першим — `separate` чи `merged`. Не генерувати input та не писати файл усередині timed blocks.

Для кожної пари розрахувати paired difference:

$$
d=T_{sep}^{actual}-T_{merge}^{actual}.
$$

Позитивне $d$ означає, що merge швидший. Ground-truth label вважати визначеним лише тоді, коли 95% CI paired difference не містить нуль:

- `merge_beneficial`, якщо нижня межа CI > 0;
- `separate_beneficial`, якщо верхня межа CI < 0;
- `ambiguous`, якщо CI містить 0.

Не змушувати noisy near-boundary cases мати штучний binary ground truth.

### 8.4. Метрики рішення

На determinate pairs подати:

- confusion matrix;
- accuracy;
- balanced accuracy;
- precision/recall/F1 для класу `merge`;
- 95% bootstrap CI за pairs;
- accuracy залежно від `boundary_bin` і `geometry_type`;
- частку `ambiguous` pairs.

Додатково подати практично важливу метрику regret:

$$
regret = T_{chosen}-\min(T_{sep},T_{merge}).
$$

Звітувати mean, median, p95 і maximum regret у ms, а також normalized regret. Це показує не лише кількість неправильних рішень, а й їхню фактичну ціну.

## 9. Формати даних

### 9.1. Raw observations Experiment A

Файл `experiment_a_raw.csv` повинен містити щонайменше:

```text
session_id,run_id,order_index,timing_mode,seed,
requested_w,requested_h,requested_area,
crop_w,crop_h,crop_area,
tensor_w,tensor_h,effective_area,aspect_ratio,model_stride,
preprocess_ms,inference_ms,postprocess_ms,detector_call_ms,
prediction_count_before_nms,prediction_count_after_nms,
cpu_temp_c,cpu_freq_mhz,process_rss_mb,elapsed_s,timestamp_utc
```

### 9.2. Raw observations Experiment B

Файл `experiment_b_raw.csv` повинен містити щонайменше:

```text
session_id,pair_id,repetition,order,timing_mode,geometry_type,boundary_bin,
r1_x1,r1_y1,r1_x2,r1_y2,r2_x1,r2_y1,r2_x2,r2_y2,
union_x1,union_y1,union_x2,union_y2,
r1_tensor_w,r1_tensor_h,r2_tensor_w,r2_tensor_h,
union_tensor_w,union_tensor_h,
a1_effective,a2_effective,au_effective,delta_effective_area,
tau_used,predicted_merge,separate_ms,merged_ms,difference_ms,
cpu_temp_c,cpu_freq_mhz,elapsed_s,timestamp_utc
```

Записувати raw CSV інкрементально після кожного run/pair, щоб довгий запуск можна було відновити. Resume має пропускати лише повністю завершені одиниці роботи. Не перезаписувати наявні observations без явного `--overwrite`.

### 9.3. Метадані

`metadata.json` або `metadata.yaml`:

- timestamp UTC;
- git commit і dirty-worktree flag;
- повна CLI command/config;
- random seed;
- Python, framework, Ultralytics/torch/ONNX Runtime versions;
- OS/kernel/architecture;
- CPU model, core counts, thread settings;
- device;
- model filename and checksum або інший стабільний model identifier;
- model stride, precision, quantization status;
- warm-up/repetitions;
- timing boundary definitions;
- max requested dimensions/canvas;
- governor/frequency/temperature availability;
- warnings.

## 10. Очікувана структура результатів

Рекомендована структура, якщо репозиторій не диктує іншу:

```text
experiments/cost_model_validation/
  README.md
  config.example.yaml
  adapters.py
  geometry.py
  timing.py
  collect_experiment_a.py
  analyze_experiment_a.py
  collect_experiment_b.py
  analyze_experiment_b.py
  common.py
  tests/
outputs/<run_name>/
  config.resolved.yaml
  metadata.json
  experiment_a_raw.csv
  experiment_a_shape_summary.csv
  model_comparison.csv
  linear_fit.json
  experiment_b_raw.csv
  experiment_b_pair_summary.csv
  decision_metrics.json
  figures/
```

Допускається менша кількість Python-файлів, якщо модулі залишаються тестованими й зрозумілими. Не створювати один монолітний notebook як єдиний спосіб запуску. Notebook може бути лише додатковим засобом огляду результатів.

## 11. CLI і відтворюваність

Бажані команди:

```bash
python -m experiments.cost_model_validation.collect_experiment_a \
  --config experiments/cost_model_validation/config.pi5.yaml \
  --output outputs/cost_model_pi5

python -m experiments.cost_model_validation.analyze_experiment_a \
  --input outputs/cost_model_pi5/experiment_a_raw.csv

python -m experiments.cost_model_validation.collect_experiment_b \
  --config experiments/cost_model_validation/config.pi5.yaml \
  --fit outputs/cost_model_pi5/linear_fit.json \
  --output outputs/cost_model_pi5

python -m experiments.cost_model_validation.analyze_experiment_b \
  --input outputs/cost_model_pi5/experiment_b_raw.csv
```

Якщо існуючий репозиторій використовує інший entry-point style, адаптувати команди до нього. Усі випадкові генератори повинні отримувати seed із конфігурації. Усі графіки мають будуватися лише з уже збережених raw CSV, без повторного запуску моделі.

## 12. Мінімальні автоматичні тести

Потрібні unit tests для:

1. union bounding rectangle;
2. raw та effective area calculations;
3. stride/shape handling на fake adapter;
4. виявлення ситуації, коли всі actual tensor shapes однакові;
5. pair generator і покриття boundary bins;
6. linear fit на синтетичних даних із відомими $K_t$ і $c_t$;
7. quadratic/piecewise comparison на контрольованих даних;
8. merge decision для відомих геометрій;
9. resume logic без дублювання rows;
10. metadata serialization.

Тести не повинні вимагати Raspberry Pi або реальних model weights. Для цього використати deterministic fake adapter.

## 13. Перевірки якості та fail-fast умови

Зупинити запуск із чітким повідомленням, якщо:

- weights/model не завантажено;
- model inference не працює на requested device;
- усі requested shapes перетворюються на один tensor shape;
- зібрано менше мінімальної кількості unique effective areas;
- у fit $c_t\le0$ або $K_t\le0$;
- Experiment B отримав invalid $\tau$ без явного override;
- output directory містить несумісний незавершений запуск і не задано resume/overwrite.

Warning, але не crash:

- temperature/frequency/RSS sensor недоступний;
- для деяких requested shapes виникли дублікати effective shapes;
- aspect-ratio coverage неповне через configured limits;
- prediction counts недоступні в adapter;
- bootstrap має невелику частку invalid ratio fits, але основний fit є додатним.

## 14. Критерії готовності реалізації

Робота завершена, коли:

- Experiment A запускається однією документованою командою та створює raw data без змішування timing modes;
- actual tensor shapes і effective areas перевіряються, а не припускаються;
- linear, quadratic і one-breakpoint piecewise models аналізуються на тих самих даних;
- отримуються $R^2$, adjusted $R^2$, RMSE, MAE, AIC, BIC, grouped-CV metrics і confidence intervals;
- створюються основні residual plots;
- Experiment B генерує контрольовані пари, особливо біля $\Delta A=\tau$;
- separate/merged вимірювання є paired та мають randomized execution order;
- неоднозначні через timing noise пари позначаються `ambiguous`;
- обчислюються classification metrics і regret;
- raw data, resolved config, metadata та derived results збережені окремо;
- unit tests проходять без model weights;
- README пояснює, як виконати короткий smoke test на звичайному ПК та фінальний запуск на Raspberry Pi 5.

## 15. Що не входить у це завдання

Не реалізовувати в межах цього експерименту:

- відеобенчмарк;
- tracking або ROI prediction у часі;
- mAP, precision, recall чи інші accuracy metrics;
- ImageNet-VID або інший dataset loader;
- greedy merging для довільної кількості об'єктів у реальному відео;
- порівняння з SAHI, PaD, batching або dynamic resolution;
- автоматичне керування governor/power mode;
- редагування статті або формулювання відповіді рецензенту.

Головна мета коду — ізольовано й статистично коректно відповісти на два питання:

1. Чи є лінійна модель $T(A)=K_t+c_tA$ достатньою апроксимацією на цільовій платформі?
2. Наскільки добре отриманий поріг $\tau=K_t/c_t$ передбачає фактично швидший спосіб обробки синтетичних пар ROI?
