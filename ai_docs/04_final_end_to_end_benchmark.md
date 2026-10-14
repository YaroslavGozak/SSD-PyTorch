# Етап 4. Фінальний end-to-end benchmark інформаційної технології

## Мета

Виконати фінальну контрольовану оцінку повної інформаційної технології, яка включає:

1. формування пріоритетних сегментів з використанням міжкадрової інформації;
2. cost-aware об'єднання сформованих сегментів;
3. нейромережеву локалізацію об'єктів;
4. вибір між segment mode та full-frame mode за прогнозованою обчислювальною вартістю.

Потрібно отримати результат, на основі якого в дисертації можна кількісно і чесно сформулювати:

- чи прискорює повна технологія YOLO26n;
- чи прискорює повна технологія ROI-SSD-MobileNet;
- як змінюється точність;
- як результат залежить від Core i7 та Raspberry Pi 5;
- який внесок дають окремо segment formation і cost-aware merge.

## Передумови

Обов'язково прочитати:

- `final_experiments/FINAL_EXPERIMENTS_MANIFEST.md`
- `final_experiments/diagnostics_128/diagnostic_report.md`
- `final_experiments/pilot/pilot_report.md`

Використовувати **тільки** зафіксовані там:

- checkpoints;
- dataset;
- class list;
- final evaluation manifest;
- padding;
- configs.

Не підбирати параметри після перегляду final results.

## 1. Моделі та платформи

Обов'язкові комбінації:

| Model | Core i7-11700K | Raspberry Pi 5 |
|---|---|---|
| YOLO26n | required | required |
| ROI-SSD-MobileNet | required | required |

Якщо Raspberry Pi фізично не може виконати одну з моделей через memory/runtime incompatibility:

- не замінювати модель іншою;
- зафіксувати технічну причину;
- зберегти crash/error evidence;
- завершити доступні конфігурації.

## 2. Dataset

Використовувати **лише final evaluation manifest**, зафіксований у stage 3.

Вимоги:

- однакові кадри для всіх моделей;
- однаковий порядок кадрів;
- однакові GT annotations;
- однаковий class subset;
- жоден video із final set не використовувався для padding tuning у stage 3.

Зберегти checksum або копію manifest разом із результатами.

## 3. Formation configuration

Використовувати:

- Static / previous localization;
- `hold = 0`;
- periodic full-frame interval = 10;
- model-specific padding із `pilot_report.md`;
- empty segments → full frame.

Не запускати додаткове tuning Static/SORT/Kalman у final set.

## 4. Cost calibration для кожної device × model

Для фінального benchmark потрібен **окремий свіжий profile на кожній платформі**:

- YOLO26n / Core i7;
- YOLO26n / Raspberry Pi 5;
- ROI-SSD-MobileNet / Core i7;
- ROI-SSD-MobileNet / Raspberry Pi 5.

Не копіювати K/c/tau між платформами.

### Calibration dataset

Використовувати окремі profiling inputs / development inputs, не final evaluation labels.

### Профіль

Оцінити:

`T(A) = K + cA`

та:

`tau = K/c`

Зберегти:

- K;
- c;
- tau;
- R²;
- MAE;
- RMSE;
- shapes;
- repetitions;
- warmup;
- exact software/hardware info.

### Важливо

`A` має відповідати actual effective NN input area.

Для YOLO врахувати реальний stride-compatible tensor shape.

Для ROI-SSD-MobileNet — фактичний variable-size tensor.

Після calibration **заморозити K/c/tau** для всіх final runs цієї model × device комбінації.

Adaptive online tau у фінальному benchmark не використовувати: він уже є окремим дослідженням.

## 5. Чотири обов'язкові режими

Для кожної model × device виконати рівно такі основні режими.

### FF

Full-frame inference на кожному кадрі.

### SEG

- segment formation;
- merge disabled;
- periodic full frame = 10.

### CA

- segment formation;
- `cost_affine` merge;
- periodic full frame = 10;
- без cost-based mode selector: якщо segments існують, обробити merged segments.

### FULL-TECH

Повна технологія:

1. periodic keyframe → full frame;
2. сформувати segments;
3. empty → full frame;
4. cost-aware merge;
5. обчислити predicted total segment cost;
6. обчислити predicted full-frame cost;
7. якщо segment cost менший → segment inference;
8. інакше → full frame.

## 6. Необов'язковий reference mode

Якщо незначно збільшує час експерименту, можна додати:

`SIMPLE_MERGE`

лише як reference до старого system experiment.

Але цей режим **не є обов'язковим** для фінального висновку. Основні режими — FF / SEG / CA / FULL-TECH.

## 7. Повторення timing runs

### Quality

Якщо inference детермінований, достатньо одного повного quality run для кожної конфігурації.

### Timing

Для кожної конфігурації виконати **3 повні повтори**.

Звіт:

- mean FPS;
- std FPS;
- mean total latency;
- p50;
- p95;
- min/max run FPS.

Якщо повний повтор на Raspberry Pi надто довгий, мінімально допустимо:

- 2 repeats;

але це треба зафіксувати, і однаковий protocol використовувати для всіх режимів цієї моделі на Pi.

## 8. Raspberry Pi monitoring

Для кожного run логувати:

- temperature before;
- temperature after;
- maximum temperature, якщо доступно;
- CPU frequency;
- throttling status;
- power/performance mode;
- number of threads;
- system load background note.

Між runs забезпечити однакові стартові умови настільки, наскільки практично можливо.

Якщо виник thermal throttling:

- не приховувати;
- позначити run;
- за можливості повторити після cooldown.

## 9. Frame-level log

Кожен run має створити frame-level CSV із полями не менше:

- `run_id`
- `model`
- `device`
- `video_id`
- `frame_id`
- `pipeline_mode`
- `used_full_frame`
- `fallback_reason`
- `num_segments_pre_merge`
- `num_segments_post_merge`
- `nn_calls`
- `processed_geometric_pixels`
- `processed_area_ratio`
- `tensor_pixels`
- `tensor_area_ratio`
- `formation_ms`
- `merge_ms`
- `preprocess_ms`
- `inference_ms`
- `postprocess_ms`
- `total_process_frame_ms`

Якщо tracker time окремий:

- `tracking_ms`

## 10. Основні aggregate metrics

Для кожної конфігурації:

### Якість

- mAP@0.5
- mAP@0.5:0.95
- Recall@0.5
- per-class AP, якщо вже підтримується

### Формування сегментів

- `gt_roi_coverage_mean`
- `gt_roi_coverage_p5`
- `processed_area_ratio_mean`
- `full_frame_fraction`

### Об'єднання

- pre-merge segments/frame
- post-merge segments/frame
- fraction of frames where merge occurred
- merge decisions count

### Фактичний compute

- NN calls/frame
- tensor pixels/frame
- tensor area ratio
- inference ms/frame
- merge overhead ms/frame
- total process latency
- FPS

## 11. Основні derived metrics

Для FULL-TECH відносно FF тієї самої model × device:

### Speedup

`speedup = FPS_FULL_TECH / FPS_FF`

### Latency reduction

`latency_reduction = 1 - latency_FULL_TECH / latency_FF`

### Accuracy change

`delta_map50 = mAP50_FULL_TECH - mAP50_FF`

`delta_recall50 = Recall50_FULL_TECH - Recall50_FF`

### Compute reduction

`nn_call_reduction`

`tensor_pixel_reduction`

`geometric_area_reduction`

## 12. Ablation interpretation

Потрібно окремо відповісти:

### FF → SEG

Що дає сам метод формування сегментів?

- area reduction;
- accuracy change;
- number of NN calls;
- speedup або slowdown.

### SEG → CA

Що дає cost-aware merge?

- зміна segment count;
- зміна calls/frame;
- merge overhead;
- зміна total latency.

### CA → FULL-TECH

Що дає cost-based mode selector?

- додаткові full-frame fallbacks;
- чи усуває він невигідні segment configurations;
- вплив на FPS;
- вплив на mAP/recall.

## 13. Головна фінальна таблиця

Обов'язково сформувати таблицю:

| Model | Device | Mode | FPS | Speedup vs FF | mAP50 | ΔmAP50 | Recall50 | Area ratio | Tensor ratio | NN calls/frame | Full-frame fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Окремо коротку підсумкову таблицю лише FULL-TECH vs FF:

| Model | Device | FF FPS | FULL-TECH FPS | Speedup | FF mAP50 | FULL-TECH mAP50 | ΔmAP50 | FF Recall | FULL-TECH Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

Ця таблиця повинна бути придатною для прямого перенесення в дисертацію.

## 14. Статистична невизначеність

Для timing:

- report mean ± std across repeated runs.

Для frame-level latency додатково:

- p50;
- p95.

Якщо можливо без значного ускладнення, для speedup FULL-TECH vs FF побудувати bootstrap 95% CI по відео або по frame-level paired measurements.

Перевага — bootstrap **по відео**, щоб не вважати сусідні кадри незалежними спостереженнями.

Якщо bootstrap не реалізований — не блокувати весь експеримент, але вказати це у звіті.

## 15. Інтерпретація негативного результату

Не оптимізувати post hoc лише для отримання speedup.

### Якщо YOLO прискорюється, а ROI-SSD-MobileNet — ні

Це валідний результат.

Перевірити:

- calls/frame;
- tensor area;
- merge overhead;
- full-frame fraction;
- actual ROI inference scaling.

Пояснити, який компонент нівелює виграш.

### Якщо ROI-SSD-MobileNet втрачає accuracy

Порівняти:

- `gt_roi_coverage_mean`;
- FF accuracy;
- segment accuracy;
- context/padding.

Якщо coverage високе, але mAP падає — проблема не в formation, а у моделі/контекстній чутливості.

### Якщо FULL-TECH повільніша за CA

Перевірити:

- selector overhead;
- cost prediction;
- кількість переключень у full frame;
- accuracy benefit.

Негативний результат не приховувати.

## 16. Обов'язкові outputs

Створити:

`final_experiments/final_benchmark/`

У ній:

- `final_report.md`
- `final_summary.csv`
- `final_fulltech_vs_fullframe.csv`
- `frame_level_metrics.csv`
- `timing_runs.csv`
- `cost_profiles_final.csv`
- `hardware_software_environment.md`
- exact config snapshots
- final evaluation manifest copy/checksum
- plots:
  - `fps_vs_map50.png`
  - `speedup_by_model_device.png`
  - `area_tensor_calls_comparison.png`
  - `latency_breakdown.png`

Не створювати графік, якщо для нього немає коректних даних.

## 17. Обов'язкова структура `final_report.md`

```md
# Final end-to-end benchmark

## 1. Dataset and compatibility
## 2. Selected models
## 3. Fixed pipeline configuration
## 4. Device-specific cost calibration
## 5. Full-frame baselines
## 6. Segment formation only
## 7. Cost-aware merge
## 8. Full technology
## 9. Ablation: FF -> SEG -> CA -> FULL-TECH
## 10. YOLO26n: Core i7 vs Raspberry Pi 5
## 11. ROI-SSD-MobileNet: Core i7 vs Raspberry Pi 5
## 12. Accuracy-speed trade-off
## 13. Negative or unexpected findings
## 14. Dissertation-ready conclusions
```

У `Dissertation-ready conclusions` не писати маркетингових формулювань. Дати тільки твердження, які безпосередньо підтверджуються отриманими числами.

## 18. Критерій завершення фінального етапу

Експеримент завершений, коли для кожної доступної model × device комбінації є:

1. full-frame baseline;
2. SEG;
3. CA;
4. FULL-TECH;
5. однаковий final dataset;
6. frozen padding;
7. device-specific frozen cost profile;
8. quality metrics;
9. full runtime metrics;
10. repeatable timing;
11. підсумкова speedup/accuracy таблиця;
12. пояснення, **чому** технологія дала або не дала прискорення.

Головний результат має дозволяти сформулювати висновок виду:

> Для моделі X на платформі Y повна інформаційна технологія змінила швидкодію з ... до ... кадрів/с (у ... раза / на ...%) при зміні mAP@0.5 з ... до ..., середній частці геометрично обробленої площі ... та ... викликах нейронної моделі на кадр.

Таке формулювання допустиме лише після отримання фактичних результатів.
