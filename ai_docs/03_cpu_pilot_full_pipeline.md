# Етап 3. CPU pilot повного сегменто-орієнтованого відеоконвеєра

## Передумови

Запускати лише після завершення етапів 1 і 2.

Обов'язково прочитати:

- `final_experiments/FINAL_EXPERIMENTS_MANIFEST.md`
- `final_experiments/diagnostics_128/diagnostic_report.md`

Для фінального pipeline використовувати:

- selected YOLO26n;
- selected ROI-SSD-MobileNet.

ROI-SSD без MobileNet у цьому етапі не є обов'язковою: він використовується в етапі 2 для діагностики архітектурної поведінки.

## Мета

Перед довгими фінальними benchmark runs:

1. перевірити, що YOLO26n і ROI-SSD-MobileNet працюють в одному й тому самому відеоконвеєрі;
2. перевірити координати, класи, cropping, merge, metric calculation;
3. підібрати **за наперед визначеним правилом**, а не post hoc, робочий padding для кожної моделі;
4. повторно відкалібрувати cost model на Core i7;
5. перевірити cost-aware merge;
6. перевірити автоматичний вибір між segment mode та full-frame mode;
7. зафіксувати всі параметри, які потім будуть заморожені у фінальному етапі.

## 1. Dataset

Використовувати спільний ImageNet-VID dataset із:

`FINAL_EXPERIMENTS_MANIFEST.md`

Не використовувати `voc-vid`, якщо manifest не встановив його як canonical common dataset.

### Development/pilot subset

Для tuning/pilot вибрати **окремі відеопослідовності**, які не будуть використовуватися у фінальному stage 4 evaluation.

Бажано:

- декілька відео;
- сумарно приблизно 300–1000 кадрів;
- різна кількість об'єктів;
- різні розміри об'єктів;
- різний рух.

Розділяти **за відео**, а не випадковими кадрами одного відео, щоб зменшити temporal leakage.

Зберегти:

`final_experiments/pilot/pilot_manifest.txt`

і окремо:

`final_experiments/pilot/final_eval_manifest_candidate.txt`

Не змінювати final evaluation manifest після перегляду фінальних результатів.

## 2. Єдина реалізація формування сегментів

Для CPU pilot використати одну основну реалізацію методу формування:

`Static / previous localization`

Причина: у попередньому контрольованому дослідженні це найпростіша реалізація з хорошим співвідношенням coverage / processed area / системні витрати.

Не запускати знову повний sweep Static × SORT × Kalman — це вже окремо досліджено.

Параметри:

- `hold = 0`;
- periodic full-frame interval = `10`;
- при відсутності придатних сегментів → full frame.

## 3. Development sweep padding

Для кожної моделі окремо:

- YOLO26n;
- ROI-SSD-MobileNet;

перевірити:

- `p = 32`;
- `p = 64`;
- `p = 96`.

Якщо для ROI-SSD-MobileNet після етапу 2 є обґрунтована причина додати ще один padding — дозволено, але це треба пояснити в pilot report.

### Режим для padding sweep

На цьому підетапі:

- merge вимкнено;
- periodic full frame = 10;
- hold = 0;
- однаковий pilot subset.

### Вибір padding

Вибрати **найменший** padding, який задовольняє обидві умови відносно full-frame baseline тієї самої моделі:

```text
mAP50_segment / mAP50_full >= 0.95
gt_roi_coverage_mean >= 0.98
```

Якщо `mAP50_full` дуже малий або критерій стає беззмістовним — явно повідомити.

Якщо жоден padding не задовольняє обидві умови:

- не підбирати інший threshold post hoc;
- вибрати найбільший протестований padding;
- зафіксувати, що criterion not met.

Зберегти обрані значення як:

- `YOLO_PADDING_FINAL`
- `ROISSD_MOBILENET_PADDING_FINAL`

у `pilot_report.md`.

## 4. Full-frame baselines

Для кожної моделі на pilot subset виконати full-frame baseline.

Логувати:

- mAP@0.5;
- mAP@0.5:0.95;
- Recall@0.5;
- total processing latency;
- inference latency;
- FPS;
- p50 latency;
- p95 latency.

## 5. Свіжа cost calibration на Core i7

Не використовувати старі `tau` автоматично.

Для кожної моделі:

- YOLO26n;
- ROI-SSD-MobileNet;

виконати профілювання на тому самому CPU environment, де буде pilot.

### Важливо

Cost model має отримувати **фактичний effective input shape**, який реально обробляє нейронна модель.

Для YOLO:

- врахувати stride constraints;
- letterbox/padding;
- actual tensor `H×W`.

Для ROI-SSD-MobileNet:

- використовувати actual variable input shape;
- якщо preprocessing змінює shape — логувати requested і actual окремо.

Оцінити:

`T(A) = K + c A`

і:

`tau = K / c`

Зберегти:

- K;
- c;
- tau;
- R²;
- MAE;
- RMSE;
- profiling shapes;
- repetitions;
- warmup protocol.

Зберегти в:

`final_experiments/pilot/cost_profiles_cpu.csv`

## 6. Режими pipeline, які треба перевірити

Після вибору padding і cost calibration для **кожної моделі** виконати:

### FF — Full frame

Повний кадр на кожному кадрі.

### SEG — Segment formation only

- periodic full frame = 10;
- Static;
- selected padding;
- hold=0;
- merge disabled.

### CA — Segment formation + cost-aware merge

Те саме, але ввімкнути `cost_affine`.

### FULL-TECH — повна інформаційна технологія

На кожному кадрі:

1. якщо це periodic keyframe → full frame;
2. інакше сформувати priority segments;
3. якщо segments empty → full frame;
4. виконати cost-aware merge;
5. оцінити predicted total cost для merged segments;
6. оцінити predicted full-frame cost;
7. якщо segment cost < full-frame cost → обробити segments;
8. інакше → full frame.

Це має бути реальна реалізація технології, а не просто `CA` з іншою назвою.

## 7. Перевірка coordinate correctness

Для segment inference обов'язково перевірити:

- crop coordinates у source image;
- model input coordinates;
- decoded local bbox;
- transform назад у global frame coordinates;
- clipping;
- NMS/duplicate handling після кількох segments.

Зробити debug export мінімум для 20 кадрів:

- original frame;
- segments;
- local detections;
- global detections;
- GT.

Зберегти в:

`final_experiments/pilot/debug_frames/`

Якщо bbox явно зміщені, не продовжувати performance benchmark до виправлення.

## 8. Обов'язкові runtime metrics

Для кожного кадру логувати:

- frame id;
- video id;
- mode: full / segments;
- number of candidate segments;
- number of segments after merge;
- number of NN calls;
- geometric processed area;
- actual tensor pixels processed;
- tracker/formation time;
- merge decision time;
- crop/preprocess time;
- NN inference time;
- postprocessing time;
- total `process_frame` time;
- whether full-frame fallback was triggered;
- reason for fallback.

### Нові ключові метрики

#### `nn_calls_per_frame`

Кількість фактичних викликів detector forward/inference.

#### `tensor_pixels_per_frame`

Сума:

`actual_input_h * actual_input_w`

для всіх NN calls цього кадру.

#### `tensor_area_ratio`

Відношення `tensor_pixels_per_frame` до кількості tensor pixels full-frame inference тієї ж моделі.

Не підмінювати `processed_area_ratio_mean` цим показником. Обидва потрібні.

## 9. Quality metrics

Для кожного режиму:

- mAP@0.5;
- mAP@0.5:0.95;
- Recall@0.5;
- `gt_roi_coverage_mean`;
- `gt_roi_coverage_p5`, якщо реалізація вже підтримує;
- `processed_area_ratio_mean`;
- `tensor_area_ratio_mean`;
- `full_frame_fraction`;
- pre-merge segment count mean;
- post-merge segment count mean;
- NN calls/frame mean.

## 10. Timing protocol

На CPU:

- зафіксувати number of threads;
- зафіксувати OpenCV threads;
- однакові environment variables;
- warm-up перед кожною серією;
- image loading не включати в `process_frame`, якщо старий benchmark теж його не включав;
- metric calculation не включати;
- preprocessing/cropping/postprocess, які є частиною runtime pipeline, включати.

Для pilot достатньо 1–3 repeats, але всі режими однієї моделі мають мати однаковий protocol.

## 11. Acceptance checks перед stage 4

Для YOLO26n і ROI-SSD-MobileNet окремо перевірити:

- FF metrics адекватні;
- SEG не має coordinate bug;
- CA реально зменшує або принаймні змінює segment count;
- `cost_affine` використовує свіжі K/c/tau;
- FULL-TECH реально іноді обирає full frame через cost, якщо така ситуація виникає;
- full-frame fallback reason логуються;
- final padding зафіксований;
- pipeline проходить весь pilot subset без crash;
- ROI-SSD-MobileNet не має критичної втрати accuracy на segments.

## 12. Обов'язкові outputs

Створити:

`final_experiments/pilot/`

і зберегти:

- `pilot_report.md`
- `pilot_manifest.txt`
- `final_eval_manifest_candidate.txt`
- `padding_sweep.csv`
- `cost_profiles_cpu.csv`
- `pipeline_summary.csv`
- `frame_level_metrics.csv`
- `debug_frames/`
- exact configs used.

У `pilot_report.md` обов'язково записати:

```text
YOLO_PADDING_FINAL =
ROISSD_MOBILENET_PADDING_FINAL =

YOLO_CPU_K =
YOLO_CPU_c =
YOLO_CPU_tau =

ROISSD_MOBILENET_CPU_K =
ROISSD_MOBILENET_CPU_c =
ROISSD_MOBILENET_CPU_tau =
```

і relative links до final configs.

## STOP CONDITIONS

Не переходити до Raspberry Pi / final benchmark, якщо:

- ROI-SSD-MobileNet segment inference має некоректні global coordinates;
- class IDs не збігаються;
- mAP collapse не пояснений;
- FULL-TECH не відрізняється логічно від CA;
- cost selector порівнює raw bbox area замість actual model input cost;
- benchmark silently resize/pad inputs без logging;
- final dataset overlap із pilot/tuning videos не визначений.

## Критерій завершення

Етап завершений, якщо:

1. для кожної моделі вибраний і зафіксований padding;
2. CPU cost profiles отримані;
3. FF / SEG / CA / FULL-TECH коректно працюють;
4. всі runtime та quality metrics логуються;
5. існує зафіксований final evaluation manifest;
6. конфігурацію можна без ручного переписування перенести на Raspberry Pi.
