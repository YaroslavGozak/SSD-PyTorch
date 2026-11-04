# Ручний запуск повторної CPU-валідації

## Середовище для всіх команд

- Working directory: `H:\Projects\University\SSD-PyTorch`.
- Python: `.venv\Scripts\python.exe` (перевірений Python 3.13.5).
- Основний module: `experiments.cpu_revalidation`; він запускає subprocess-и
  тим самим Python у корені репозиторію, строго послідовно, без паралельного CPU profiling.
- Root результатів: `final_experiments/cpu_revalidation_20261102` (далі `R`).
- Device: `cpu` для кожного subprocess, detector adapter і calibration.
- PyTorch intra-op/inter-op: 8/1; OpenCV: 1; batch=1; workers=0.
- Environment: `CUDA_VISIBLE_DEVICES=''`, `SSD_INFERENCE_DEVICE=cpu`,
  `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=1`,
  `NUMEXPR_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE`, `MKL_DYNAMIC=FALSE`, `PYTHONHASHSEED=0`.
  Runner задає їх до імпорту torch у дочірньому процесі.

```powershell
Set-Location -LiteralPath 'H:\Projects\University\SSD-PyTorch'
```

## Моделі, конфіги та ваги

| CLI model | Quality config відносно R | Checkpoint відносно репозиторію | Quality conf_floor |
|---|---|---|---:|
| `yolo26n` | `configs/yolo26n.yaml` | `trained_models/best-imagenet-vid-yolo/best.pt` | 0.01 |
| `roi_ssd_vgg16` | `configs/roi_ssd_vgg16.yaml` | `trained_models/imagenet-vid-roissd-20261028/stage3/best_map.pt` | 0.01 |
| `roi_ssd_mobilenet` | `configs/roi_ssd_mobilenet.yaml` | `trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt` | 0.01 |

No checkpoint copying is required. Existing manifest paths are used directly.
Ніяких копій `last.pt`, пошуку кращих ваг або навчання немає.
У `pilot/<model>/configs/` лежать окремі operational config templates;
Stage 03 обирає відповідний config за `--models` і завантажує threshold із
зафіксованого pilot-рішення. Початковий `infer_conf_threshold` template не є
обраним operational threshold і не використовується як fallback.

## Два призначення confidence

Для full-frame accuracy, pad-loop і fixed-size-loop **усі три моделі мають
conf_floor=0.01**. Для SSD це задано як `train_params.infer_conf_threshold` і
`model_params.low_score_threshold`; для YOLO — як predict confidence floor.
mAP обчислюється на всіх повернутих detections із ранжуванням за confidence
наявною реалізацією `compute_map`; operational threshold до quality AP не застосовується.
Наявний area-integral AP не замінюється іншим evaluator/COCO interpolation.

Stage 03 має окремий обов'язковий етап **confidence** для кожної моделі:

1. FF inference при floor=0.01 на всіх 320 кадрах frozen pilot manifest.
   NMS/cap/coordinates — ті самі, що в Stage03 FF; final-candidate frames не читаються.
2. На збережених FF predictions перевіряється наперед задана сітка:
   `{0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95}`.
   Повторний inference для кожного threshold не потрібний.
3. Обирається максимум **micro F1@IoU=0.5** за сумарними TP/FP/FN усіх pilot frames
   і класів: `F1=2TP/(2TP+FP+FN)`. Використовується наявний class-aware matcher
   `oracle_diagnosis.match_objects`, із confidence ordering та обліком дублікатів як FP.
   За рівності F1 — вищий recall; за подальшої рівності — нижчий threshold.
   Без GT або за TP=0 для всіх кандидатів selection зупиняється без фіксації порога.
4. Threshold фіксується в `pilot/<model>/operational_confidence.json` з receipt
   `operational_confidence.sha256`, hashes checkpoint/config/pilot manifest та
   evidence CSV. Обрані значення ще не відомі: selection не запускалася під час підготовки.
5. Padding sweep, smoke, cost calibration і FF/SEG/CA/FULL-TECH завантажують це
   рішення. Воно задає detector cutoff, output filtering і tracker confidence.
   Відсутнє/змінене рішення або evidence спричиняє помилку; між stages
   перевіряється той самий decision hash. До фінального benchmark переносити
   зафіксовані operational thresholds разом із їх provenance, без повторного tuning.

Пороги якості та operational thresholds записуються окремо у config/audit/report.
Source configs із `final_experiments/configs/`, архітектури та ваги не змінюються.

## Виявлені відмінності протоколів

Quality floor уніфіковано до 0.01 за останнім уточненням користувача.
Operational operating points для pipeline обираються окремо на pilot.

За дозволеним уточненням YOLO pad-loop більше не розширює ROI до квадрата.
Геометричний ROI для всіх моделей: перший об'єкт в annotation order,
padding p у координатах кадру 320×320, clipping до меж кадру. Для YOLO після
crop виконується лише симетричне neutral padding незалежно по H та W до
`ceil(H/32)*32`, `ceil(W/32)*32`; пікселі crop не розтягуються. Новий protocol
не тотожний старим квадратним YOLO pad-loop результатам.

**STOP для тверджень про ідентичну вибірку/постобробку:** в існуючих scripts є
ще дві відмінності, які тут не змінювалися:

1. Fixed-size crop reference visibility: YOLO default 0.9, selected SSD stage3
   configs 0.8; other-object visibility 0.3. Тому однаковий requested H×W
   не гарантує однакову eligible підмножину моделей. Coverage CSV/JSON обов'язкові.
2. Accuracy `tools.infer` використовує наявний YOLO predict adapter (default
   max_det=300); `audit_protocol.max_detections=200` сам по собі не є параметром
   цього loader. Stage 03 явно задає YOLO max_det=200 та global class-aware NMS
   0.45 у `process_frame`. SSD має internal NMS 0.45 і cap=200. Для end-to-end
   YOLO параметр predict `iou=0.45` не гарантує зовнішній NMS.

Це повторення чинних методик зі збереженими параметрами. Не робити прямого
висновку про різницю якості між accuracy loops і Stage 03 як про ефект лише
геометрії. Спільний протокол додаткового NMS/visibility/cap потребує окремого
рішення перед таким порівнянням; цей запуск не змінює їх мовчки.
ImageNet normalization для SSD і відновлення [0,1] для YOLO adapter збережені.

## 0. Перевірка без inference

Підготовка вже виконана. Повторно `--stage prepare` для цього root не запускати.

```powershell
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage check
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage all --dry-run
```

Config/checkpoint/output для кожної команди беруться з таблиці вище.
`check` перевіряє SHA-256 ваг, конфігів, коду, manifests і dataset YAML,
наявність 5064 source images і SHA-256 їх labels. `dry-run` друкує всі 26
дочірніх команд без inference, calibration або створення результатів.

Опційна ручна перевірка координат після вибору operational thresholds:

```powershell
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage confidence
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage smoke
```

Викликає `experiments.cpu_pilot.run_pilot --stage smoke --max-frames 20`
для кожної моделі. Outputs: `R/pilot/<model>/smoke_metadata.json`,
`smoke_summary.csv`, `debug_frames/<model>/coordinate_smoke_seg/*.jpg` та
`*.json` (20 кадрів/модель). JSON містить GT, segments, local detections,
global detections до NMS, final detections після NMS і шлях оригіналу.
Переглянути overlays, особливо VGG16. При помилці координат зупинити довгий запуск.
Smoke не обирає padding і не підставляється замість pilot.
`confidence` тут — повний FF pilot, який користувач запускає вручну.
Після цього продовжувати окремими `accuracy`, `padding`, `cost`, `pipeline`,
`aggregate`; `--stage all`/`stage03` повторили б уже зафіксований confidence stage.

## 1. Пріоритетний варіант — одна команда для всього

```powershell
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage all
```

Послідовність: full-frame accuracy YOLO → VGG16 → MobileNet;
pad-loop YOLO → VGG16 → MobileNet; fixed-size-loop YOLO → VGG16 →
MobileNet; окрема ROI-SSD/VGG16 confidence-діагностика на п'яти порогах →
operational confidence selection/freeze усіх трьох →
Stage03 padding усіх трьох → автоматичний запис final paddings →
fresh cost усіх трьох → FF/SEG/CA/FULL-TECH кожної моделі → combined reports.
Усі дочірні команди використовують configs/checkpoints із таблиці і device CPU.
Очікувані outputs перелічені нижче. Після цієї команди окремі stages не повторювати.

## 2. Альтернатива — дві об'єднані команди

Запускати замість `--stage all`, у зазначеному порядку:

```powershell
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage accuracy --with-confidence-diagnostic
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage stage03
```

Перша команда покриває full-frame accuracy та обидва accuracy-loop для всіх трьох
моделей і окрему confidence-діагностику ROI-SSD/VGG16.
Друга покриває confidence selection → padding → calibration → pipeline для всіх
трьох і формує reports з нових результатів.
Stage03 використовує manifests із `R/pilot/<model>/` і свіжі coefficients
цієї конкретної моделі з того самого каталогу.

### Окремо — тільки точність на повному кадрі трьох моделей

```powershell
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage accuracy --accuracy-modes full-frame
```

Це одна команда для YOLO26n, ROI-SSD/VGG16 та ROI-SSD-MobileNetV3 послідовно.
Використовуються ті самі configs/checkpoints із таблиці, `device=cpu` і весь
`R/manifests/val_manifest.jsonl` — 5064 кадри для кожної моделі. Кожен повний
кадр масштабується до зафіксованого input 320×320 без ROI crop; це перевірка
всієї сцени при configured resolution, а не inference у native resolution.
Викликається наявний `tools.infer --eval-mode default` з transform `ssd`.
Quality confidence floor: 0.01 для всіх трьох моделей.

Outputs: `R/accuracy_sweeps/<model>/full_frame/results.csv`, `ssd_audit.json`,
`job_accuracy.json`, `accuracy_execution.log`. JSON містить mAP@0.5,
mAP@0.5:0.95, Detector Recall, кількість кадрів, coverage=1, actual input shape
та latency/FPS. Агрегація також створює `R/combined_reports/full_frame_accuracy.csv`.
FF baseline Stage03 на 320 pilot frames залишається окремим експериментом.

### ROI-SSD/VGG16 при conf = 0.01, 0.05, 0.1, 0.25, 0.5

Одна команда послідовно виконує всі п'ять повнокадрових перевірок:

```powershell
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage confidence-diagnostic
```

Модель — саме ROI-SSD/VGG16 із зафіксованим `best_map.pt` із таблиці. Для кожного
threshold оцінюються всі 5064 validation frames, вся сцена при input 320×320,
device CPU; cutoff однаково задається в `infer_conf_threshold` і
`model_params.low_score_threshold`. Решта параметрів, включно з NMS, незмінна.
Використовується наявний `tools.infer --eval-mode default`.

Конфіги: `R/configs/confidence_diagnostic/roi_ssd_vgg16/conf_0p01.yaml`,
`conf_0p05.yaml`, `conf_0p1.yaml`, `conf_0p25.yaml`, `conf_0p5.yaml`.
Результати кожного порога — в однойменних підкаталогах
`R/confidence_diagnostics/roi_ssd_vgg16/`: `results.csv`, `ssd_audit.json`,
`job_accuracy.json`, `accuracy_execution.log`. Після п'яти успішних запусків
автоматично створюється `summary.csv` із mAP50, mAP50:95, Detector Recall,
latency/FPS, evaluated/total frames, cutoff, config і checkpoint SHA-256.
Під час загальної агрегації таблиця також потрапляє в
`R/combined_reports/roi_ssd_confidence_diagnostic.csv`.

Це окрема діагностика впливу score cutoff. Основне quality-порівняння моделей
залишається при floor=0.01; його CSV не доповнюється рядками інших cutoffs.
Validation-діагностика не переобирає operational threshold: це робить лише
320-frame pilot за зафіксованим F1/recall правилом. Усі п'ять значень також
уже входять до candidate grid цього pilot selection.

`--stage all` включає діагностику автоматично. Щоб об'єднати її лише з accuracy,
використати `--stage accuracy --with-confidence-diagnostic`. Звичайний
`--stage accuracy` виконує тільки основні quality-оцінювання при floor=0.01.
Не запускати діагностику повторно, якщо її вже виконано через `all` або цей flag;
після окремої діагностики продовжувати звичайними `accuracy`, `stage03`.

Якщо full-frame вже виконано окремо, запустити решту без його повторення:

```powershell
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage accuracy --accuracy-modes pad-loop fixed-size-loop
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage confidence-diagnostic
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage stage03
```

## 3. Детальний контроль етапів

Це альтернатива двом варіантам вище; не дублювати вже виконані етапи.

```powershell
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage accuracy
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage confidence-diagnostic
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage confidence
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage padding
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage cost
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage pipeline
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage aggregate
```

Кожна команда охоплює всі три моделі. Для запуску частини можна додати
`--models yolo26n roi_ssd_vgg16` або одну модель, а для accuracy —
`--accuracy-modes full-frame`, `--accuracy-modes pad-loop` чи `--accuracy-modes fixed-size-loop`.
Наприклад, одна команда для full-frame та обох loops VGG16:

```powershell
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage accuracy --models roi_ssd_vgg16
```

Фактичні script/module/config/output можна переглянути через `--dry-run` із
тими самими аргументами. Output окремих моделей залишається в їхніх підкаталогах.

## Grids, manifests і gates

- Accuracy dataset: `R/manifests/val_manifest.jsonl`, усі 5064 validation frames
  у frozen order. `canonical_validation.jsonl` із 256 selection frames збережено
  для provenance; воно не замінює повну validation вибірку.
- Full-frame accuracy: 5064 кадри кожної моделі, input 320×320 без crop,
  жодних eligibility exclusions; запускається перед обома loops за замовчуванням.
- Pad-loop: `0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200`.
- Fixed-size: H,W ∈ `{32,64,96,128,160,192,224,256,288,320}`; 100 requested shapes
  на модель. SSD передається прямокутний tensor. YOLO dimensions у grid уже
  кратні 32; actual shapes записуються forward hook, а не вгадуються з request.
- Pilot: незмінені `pilot_manifest.txt` та `final_eval_manifest_candidate.txt`
  з останнього `final_experiments/cpu_pilot/` (320 кадрів / 32 відео pilot).
  SHA-256 є в `experiment_manifest.md`; pilot/final video overlap=0.
- Stage03 padding: `{32,64,96,112,128,144,160}` плюс FF baseline для кожної моделі.
  Static/previous localization; hold=0; periodic FF=10; merge disabled.
- Обирається найменший p, де mAP50_SEG/mAP50_FF ≥0.95 та GT coverage ≥0.98.
  Якщо жоден не пройшов: p=160, статус `PADDING_GATE_FAILED`. Наступні режими
  виконуються як діагностика; failed gate залишається в summary/report.
  Старий default Stage03 поза цим новим session продовжує зупинятися на gate.
- Final paddings автоматично зберігаються у `padding_decision.json` і report:
  `YOLO_PADDING_FINAL`, `ROISSD_VGG_PADDING_FINAL`, `ROISSD_MOBILENET_PADDING_FINAL`.
- Cost shapes: `32×32,64×64,96×96,128×128,192×192,256×256,320×320,64×128,128×64,160×320,320×160`;
  2 warmups + 8 repetitions/shape. T(A)=K+cA, tau=K/c. Actual network shape
  спостерігається hook; записуються також requested shape, R²/MAE/RMSE.
  Нефізичний fit (K≤0 або c≤0) зупиняє запуск; старі coefficients не підтягуються.

## Очікувані outputs

| Етап | Каталог відносно R | Файли |
|---|---|---|
| full-frame accuracy | `accuracy_sweeps/<model>/full_frame/` | `results.csv` (1 row), `ssd_audit.json` (5064 evaluated frames), `job_accuracy.json`, `accuracy_execution.log` |
| pad-loop | `accuracy_sweeps/<model>/pad_loop/` | `results.csv` (21 rows), `*_audit.json` (21), `job_accuracy.json`, `accuracy_execution.log` |
| fixed-size | `accuracy_sweeps/<model>/fixed_size_loop/` | `fixed_size_results_v2.csv` (100 rows), `fixed_size_coverage_v2.csv`, `*_audit.json` (100), job/log |
| ROI-SSD confidence diagnostic | `confidence_diagnostics/roi_ssd_vgg16/` | `conf_0p01/`, `conf_0p05/`, `conf_0p1/`, `conf_0p25/`, `conf_0p5/`; `summary.csv` (5 rows) |
| Operational confidence | `pilot/<model>/` | `confidence_candidates.csv` (20 candidates), `confidence_frame_metrics.csv`, `confidence_ff_summary.csv`, `operational_confidence.json`, `operational_confidence.sha256`, `confidence_execution.log` |
| Stage03 padding | `pilot/<model>/` | `padding_sweep.csv` (FF + 7 paddings), `padding_decision.json`, `frame_level_metrics.csv`, `padding_execution.log` |
| CPU calibration | `pilot/<model>/` | `cost_profiles_cpu.json`, `cost_profiles_cpu.csv`, `cost_measurements_cpu.csv`, `cost_execution.log` |
| FF/SEG/CA/FULL-TECH | `pilot/<model>/` | `pipeline_summary.csv` (4 rows), `frame_level_metrics.csv`, `metadata.json`, `pilot_report.md`, `debug_frames/`, `pipeline_execution.log` |
| Aggregation | `combined_reports/` | `accuracy_sweeps.csv`, `full_frame_accuracy.csv`, `roi_ssd_confidence_diagnostic.csv`, `operational_confidence.json`, `pipeline_summary.csv`, `ca_vs_full_tech.csv`, `completeness.json`, `report.md` |

`*_audit.json` додає evaluated/total frames, applicability, mAP50/mAP50:95,
Detector Recall, `confidence_floor`, latency/FPS, requested tensor shapes, geometric crop shapes
до YOLO stride padding, actual network shapes, active depth/heads SSD та threads.
Checkpoint path/hash прив'язано через `session.json`/`job_accuracy.json` і
додано до combined CSV. Coverage тут — частка eligible frames; це не GT area coverage.
Stage03 має окремі GT coverage mean/p5.

Frame-level CSV зберігає padding, frame/video IDs, mode, candidate/merged counts,
NN calls, geometric/tensor areas, requested/actual shapes, formation/merge/crop/
NN/postprocess/total timings, fallback reason, predicted costs і selector decision.
Для FF/SEG cost поля порожні, бо cost selector у цих режимах не використовується.
На periodic/empty-segment fallback candidate cost може бути порожній: selector
не обчислюється. `total_process_frame_time_s` FULL-TECH включає selector overhead.
NN timing включає adapter та detector-internal postprocessing; це не isolated kernel timing.

`ca_vs_full_tech.csv` зіставляє однакові frame IDs, actual tensors і predictions.
Якщо selector логічно обрав FF, але CA вже виконував еквівалентний tensor workload,
report явно пояснює це; різниця timing не приписується selector.
Prediction-dependent tracker trajectories можуть різнитися; різні inputs не
трактуються автоматично як bug чи causal speedup.

## Чистота результатів і повторні спроби

Runner не має silent resume або overwrite. Перед запуском перевіряються всі
target stages; існуючий result directory/job marker/основний output зупиняє запуск.
При child error наступні jobs не запускаються, `job_<stage>.json` має FAILED,
подробиці — в execution log. Не видаляти marker для домішування нових даних.
Агрегація читає лише цей root і позначає неповний scope в `completeness.json`.

Для справді нового повтору створити інший root (не редагуючи Python):

```powershell
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage prepare --output final_experiments/cpu_revalidation_20261102_run2
.venv\Scripts\python.exe -m experiments.cpu_revalidation --stage all --output final_experiments/cpu_revalidation_20261102_run2
```

Якщо код/config/hash змінено після підготовки, check зупиниться: потрібен новий
root і нова конфігураційна ідентичність, а не використання старих CSV.

## Що виконано під час підготовки

Перевірки imports/CLI/config/hash, короткі unit tests та технічні crop/pipeline
приклади зберігаються в `validation_checks/` і `artifacts/cpu_revalidation_smoke.log`.
Жодних повних pad-loop, 100-shape inference sweeps, pilot або calibration grid
під час підготовки не виконано. Mock-based tests sweep control flow не запускають detector.
