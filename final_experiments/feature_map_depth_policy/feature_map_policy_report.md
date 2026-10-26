# Stage 2e: depth policy за геометрією feature maps

**COMPLETE. 36 форм x 4 policies = 144 policy-конфігурації; 96 унікальних
shape/depth evaluations на всіх 256 samples. Ваги й production code не змінено.**

## Висновок для Stage 3

**Рекомендується `until_1x1` як кандидат для контрольованого CPU-pilot на
перевірених формах, після окремого затвердження.** Вона обходиться без ручних
pixel thresholds, усуває великі legacy-провали, наближається до локального
adjacent reference та зазвичай дешевша за максимальну фізично можливу depth.
Це не сертифікація довільних H/W, не гарантія ідентичності максимальній
Recall і не реалізована production policy.

Користь додавання coarse 1x1 head **пов'язана з великою часткою GT у crop**;
найсильніший із перевірених описових показників — **area ratio**.
Додавання ще однієї/кількох 1x1 heads після першої 1x1 дає значно менший
виграш. Кореляція не доводить причину, а геометрія GT недоступна inference
і **не використовується для автоматичного вибору depth**.

## 1. Модель, provenance та дозволена відмінність source

- Checkpoint:
  [20260928-phase2-bn-freeze/best_map.pt](../../trained_models/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt).
  SHA256 `7fa171ab669ac53f0d4f39586c08e61a60ca79fed98fd70874ef42bfd52356ce`.
- Той самий [locked subset](../../experiments/roissd_128_anomaly/data/subset.jsonl):
  256 source samples; SHA256
  `576e8a3eb9b27b05abe1f20eef3da2bb4b4c87bcd20b4bf5f192225606bf54d2`.
- Config snapshot SHA256
  `81ec5a55754462f2db9ad6630ee9a37fe2f79d00d43648cdbe0701e734c813bd`.
  Project FixedSizeTestTransform, RGB/ImageNet normalization, seed `20261023`,
  confidence `0.5`, NMS IoU `0.45`, cap `200`, batch=1.
- Accuracy — той самий project area-integral AP@0.5 і class-macro
  operating-point Recall@0.5. Mean raw max confidence — середнє за crops
  максимумів foreground softmax **до** confidence/NMS. Anchor metrics —
  за всіма видимими GT без confidence-фільтра.
- Перед Stage 2e у worktree вже були змінені production thresholds:
  `<63 / <88 / <127 / <255`. Користувач **явно дозволив** scoped legacy
  override без відновлення/редагування цього файла.
  AST перевірка підтвердила ідентичність усієї моделі поза методом вибору
  depth; Git blob із Windows CRLF checkout відтворює Stage-2d source hash.
- Model source Stage 2d:
  `634708db126185b807e86609e6f7ef0811b4551d42e68b0712d5abf018abd315`.
  Поточний дозволений source:
  `664452ed17f797d415880af20ba4a578040ee82335e2445dfcaa43de5bb0c8fe`.
  Обидва hashes, line endings, точний diff і всі залежні source hashes
  записані в [metadata.json](metadata.json).
- **Legacy завжди означає старі 32/64/96/140/268**, не відредаговані пороги.
  Усі policies використовують scoped instance override; після запусків
  відновлюється наявний метод. До/після перевірено незмінність sources/ваг.
- Для кожного з 9216 shape/sample inputs hash tensors+GT збігся зі Stage 2d.
  **72/72 adjacent-depth controls** відтворили збережені метрики й структуру
  з допуском `1e-6`. Labels/images перевірено за locked hashes.

Accuracy device: RTX 3060 Ti, PyTorch `2.12.0+cu126`, deterministic algorithms.
CPU: Intel Core i7-11700K @ 3.60 GHz, Windows, 4 intra-op / 1 inter-op thread.
Це **не Raspberry Pi**, CPU accuracy parity на всьому subset не оцінювалася.

## 2. Набір форм і визначення policies

20 squares: **32,64,88,96,104,124,127,128,129,136,140,144,
240,252,255,256,257,264,268,280**.

16 oriented rectangles: **96x144/144x96, 127x159/159x127,
128x160/160x128, 128x192/192x128, 140x175/175x140,
256x320/320x256, 256x384/384x256, 268x335/335x268**.
Усюди notation **H x W**, без stride-rounding.

| Policy | Реалізація лише в експерименті |
|---|---|
| `legacy` | Fixed original thresholds 32/64/96/140/268 |
| `max_feasible_depth` | Реальні forced forwards L1..L6; наступна depth додається, поки обчислення просторово коректне; жодних pixel thresholds |
| `until_1x1` | Перша активна feature map, для якої фактичний H=W=1; далі не додавати heads; якщо 1x1 не досягнуто, остання feasible depth |
| `best_adjacent_reference` | Retrospective reference за Stage-2d tolerance: upper, якщо AP або Recall виграє >0.5 в.п. і жодна не втрачає >0.5 в.п.; інакше lower |

Reference свідомо віддає перевагу дешевшій depth на практичній
еквівалентності. Це локальна оцінка на вже використаних даних, **не production
policy, не незалежний test result і не глобальна upper bound серед L1..L6**.

Усі шість depths фізично feasible для всіх 36 форм: `max_feasible_depth`
всюди обрала L6. Padded stride-2 blocks можуть залишатися 1x1; тому
фізична feasible-ність сама по собі не визначає корисність heads.

Усі policy evaluations із однаковим selected depth ділять **точно той самий**
результат, predictions і latency; вони не рахуються як незалежні запуски.
Додатково оцінено обидві Stage-2d adjacent depths для per-object pairs.

## 3. Підсумкова якість і CPU

**Рівновагове середнє за 36 формами**, а не pooled mAP, не deployment-weighted
score і не оцінка швидкодії pipeline з реальним розподілом ROI.

| Policy | Mean mAP50, % | Mean Recall50, % | Raw max conf | Best-anchor IoU | GT coverage, % | Mean CPU median, ms | Mean CPU p95, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 32.54 | 35.39 | 0.5817 | 0.4947 | 49.31 | 14.755 | 17.428 |
| max_feasible_depth | 47.38 | 50.80 | 0.7786 | 0.7180 | 84.81 | 16.004 | 18.774 |
| until_1x1 | **47.41** | 50.58 | 0.7451 | 0.6318 | 76.16 | **15.220** | **17.912** |
| best_adjacent_reference | 46.47 | 49.65 | 0.7335 | 0.6083 | 72.79 | 15.115 | 17.787 |

При tolerance 0.5 в.п. за AP/Recall:

- `until_1x1` vs legacy: **22 material gains, 14 equivalent, 0 material losses**.
- `until_1x1` vs adjacent reference: **2 gains, 33 equivalent, 1 loss**
  (255x255, Recall -0.74 в.п., AP -0.34 в.п.).
- `max_feasible_depth` vs adjacent reference: 7 gains, 29 equivalent,
  0 material losses.
- `max_feasible_depth` vs `until_1x1`: **6 material gains, 29 equivalent,
  1 material loss**. Mean mAP max−until **-0.027 в.п.**, mean Recall
  **+0.227 в.п.**: максимум depth не є безумовним accuracy-виграшем.

### Selected depth distribution, число форм

| Policy | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 1 | 1 | 4 | 15 | 14 | 1 |
| max_feasible_depth | 0 | 0 | 0 | 0 | 0 | 36 |
| until_1x1 | 0 | 1 | 1 | 6 | 18 | 10 |
| best_adjacent_reference | 1 | 1 | 0 | 8 | 17 | 9 |

## 4. Геометрія, відомі провали та винятки

`until_1x1` зупиняється на 1x1 в усіх 36 формах.
Повні active heads, NCHW/spatial shapes, наступна potential map,
feasible status та counts збережено для кожної policy у
[per_shape_results.csv](per_shape_results.csv).
Для L6 наступного блоку немає: `next_head_status=PYRAMID_EXHAUSTED`,
shape порожня; це не невдала convolution.

| Shape | Legacy L_R | Until L_R | mAP legacy / until, % | Коментар |
|---|---:|---:|---:|---|
| 32x32 | 1 | 2 | 0.00 / 0.00 | Tiny quality не вирішено |
| 64x64 | 2 | 3 | 0.00 / 0.00 | Tiny quality не вирішено |
| 88x88 | 3 | 4 | 2.94 / 46.65 | Усунено великий legacy-провал |
| 96x96 | 3 | 4 | 3.46 / 49.25 | Усунено великий legacy-провал |
| 96x144 | 3 | 5 | 3.25 / 43.31 | Adjacent reference L4 дає лише 20.92% |
| 144x96 | 3 | 5 | 8.29 / 31.37 | Adjacent reference L4 дає лише 19.88% |
| 128x128 | 4 | 4 | 42.74 / 42.74 | L4 map вже 1x1; зайва depth не потрібна |
| 128x160 | 4 | 5 | 20.82 / 32.90 | L4 map 1x2; L5 досягає 1x1 |
| 160x128 | 4 | 5 | 11.68 / 21.96 | L4 map 2x1; L5 досягає 1x1 |
| 129x129 | 4 | 5 | 0.89 / 58.47 | Square legacy-провал усунено |
| 140x140 | 4 | 5 | 2.88 / 60.34 | Square legacy-провал усунено |
| 256x256 | 5 | 5 | 60.33 / 60.33 | L5 map вже 1x1 |
| 256x320 | 5 | 6 | 60.27 / 61.05 | L5 map 1x2 |
| 257x257 | 5 | 6 | 44.09 / 49.62 | Покращено, але не повернуто рівень square 256 |
| 268x268 | 5 | 6 | 49.45 / 59.26 | Стара межа запізнювалася |

Geometry rule адаптується до форми без fitted pixel threshold. Це не означає,
що абсолютна точність стала незалежною від aspect ratio/orientation:
наприклад, 128x192/192x128 із until дають 30.31/13.39% mAP.
Політика не має матеріальних втрат проти legacy на цих формах, проте
наявна orientation sensitivity моделі/preprocessing нікуди не зникає.

### Чи потрібні голови після першої 1x1?

У 26 формах max додає heads після першої 1x1; в інших 10 until вже досягла L6.
Матеріальні зміни max проти until:

| Shape | Until→max depth | ΔmAP, в.п. | ΔRecall, в.п. | CPU median until / max, ms |
|---|---|---:|---:|---:|
| 64x64 | 3→6 | +0.53 | +0.78 | 7.001 / 9.064 |
| 96x96 | 4→6 | -0.06 | +0.74 | 9.762 / 10.926 |
| 255x255 | 5→6 | +0.34 | +0.74 | 19.243 / 20.220 |
| 159x127 | 5→6 | -0.13 | +0.74 | 12.478 / 13.156 |
| 160x128 | 5→6 | -0.12 | +0.74 | 12.414 / 13.389 |
| 128x192 | 5→6 | +0.04 | +0.74 | 13.421 / 14.387 |
| 96x144 | 5→6 | **-0.67** | -0.37 | 11.894 / 12.788 |

Отже, **додатковий gain після 1x1 існує**, переважно невеликий Recall gain,
але не універсальний: max також може погіршувати AP.
Для 26 пар це **25 gained / 3 lost GT-exposures** із 8750,
micro-Recall gain **+0.251 в.п.**, mean best same-class predicted IoU
**+0.00316**. Macro Recall може змінюватися сильніше через малочисельні
класи; це не суперечність між таблицями.

## 5. CPU latency і scope автоматичного вибору

Виміряно **96 унікальних CPU shape/depth конфігурацій**, batch=1,
**6 warm-up + 30 repeats**, циклічно samples 0/85/170; depth order
чергується. Використано той самий lightweight shape-only slot guard,
що у Stage 2d, без повторного softmax/IoU observer усередині timing.
Forward включає backbone/heads/decode/confidence/NMS/cap;
crop, transfer, construction виключено.

За всіма 36 формами until має **на 0.784 ms / 4.90% нижче** mean median,
ніж max. Для 26 форм, де вона зупиняється раніше:

- медіана парного додаткового overhead max: **+0.962 ms / +7.00%**;
- медіана економії until від часу max: **6.54%**;
- діапазон overhead **+0.431..+2.608 ms**, **+2.50..+44.44%**;
- на tiny ROI відносна економія більша, на великих — скромніша.

Це помірний виграш, не заявка про кратне прискорення.
Форми з already-L6 мають точно спільний timing, а не випадкову різницю.
Mean p95 у summary — середнє 36 per-shape p95, не pooled p95.

**Важливо:** timing використовує cached H/W→selected depth, що отримана
реальним геометричним probing. Вартість початкових L1..L6 probes не входить.
Експеримент не реалізує dynamic production selector. Якщо в production
пробувати всі depths для кожного ROI, наведені savings не застосовуються.
Наступна реалізація має або зупиняти реальний forward після першої 1x1,
або мати обмежений geometry cache; її overhead слід заміряти окремо.
CPU не ізольовано від OS/shared-machine noise, affinity/turbo не фіксувалися.

## 6. Per-object geometry і localization

Збережено **12 412 retained GT-exposures** у 36 формах, усі об'єкти
multi-GT crops, не лише перший. Для кожного є crop H/W, GT w/h/area,
crop area, width/height/area/max-side/min-side ratios.
Це геометрія **після фактичного crop/resize/clipping**, у pixels і ratios.

Парні записи:

- `stage2d_adjacent`: 36 пар / 12 412 GT-exposures;
- `until_vs_max`: 26 пар / 8750 GT-exposures;
- `legacy_vs_until`: 26 пар / 8975 GT-exposures.

Для кожного GT логуються lower/upper/delta:
one-to-one localization success@0.5, max confidence серед **same-class
returned detections**, best same-class predicted IoU, confidence найкраще
локалізованої prediction, best-anchor IoU, crop raw max confidence.
Відсутність prediction означає дійсні нулі IoU/confidence/success, не
підстановку замість missing data. Одну prediction не можна зарахувати
двом GT; exact project IoU та score-ordered matching відтворюють evaluator.
Перевірено збіг per-class success recall із project Recall.

AP не декомпонується адитивно по samples: замість некоректного «per-GT ΔAP»
використано localization success і IoU. Per-object bin Recall нижче —
**micro GT success**, не macro metric policy summary.

### Усі adjacent pairs, за area ratio

N — повторні shape/GT-exposures, **не незалежні унікальні об'єкти**.
Δconfidence — max same-class post-NMS confidence; ΔIoU — best same-class
predicted IoU, не anchor IoU.

| GT area ratio | N GT | Lower Recall, % | Upper Recall, % | ΔRecall, в.п. | Mean Δconfidence | Mean ΔIoU |
|---|---:|---:|---:|---:|---:|---:|
| <0.25 | 3954 | 19.52 | 19.50 | -0.03 | +0.0809 | +0.0098 |
| 0.25–0.50 | 2583 | 31.67 | 42.24 | +10.57 | +0.1536 | +0.0914 |
| 0.50–0.70 | 2798 | 43.21 | 68.66 | +25.45 | +0.2466 | +0.2021 |
| 0.70–0.85 | 1774 | 40.19 | 72.21 | +32.02 | +0.2969 | +0.2631 |
| 0.85–0.95 | 1025 | 39.41 | 77.76 | +38.34 | +0.3626 | +0.3246 |
| >=0.95 | 278 | 41.37 | 70.14 | +28.78 | +0.2457 | +0.2465 |

### Лише adjacent pairs, де додається одна head із map 1x1

34 форми / **11 738 GT-exposures**; виключено 96x144 і 144x96, у яких
Stage-2d upper L4 map ще не 1x1. Сюди входять як перша 1x1, так і
повторна 1x1; окремий `until_vs_max` нижче ізолює post-first-1x1 additions.

| GT area ratio | N GT | Lower Recall, % | Upper Recall, % | ΔRecall, в.п. | Mean Δconfidence | Mean ΔIoU |
|---|---:|---:|---:|---:|---:|---:|
| <0.25 | 3795 | 20.34 | 20.32 | -0.03 | +0.0818 | +0.0098 |
| 0.25–0.50 | 2408 | 33.14 | 44.06 | +10.92 | +0.1573 | +0.0933 |
| 0.50–0.70 | 2636 | 45.11 | 71.24 | +26.14 | +0.2546 | +0.2085 |
| 0.70–0.85 | 1670 | 42.16 | 73.59 | +31.44 | +0.2949 | +0.2631 |
| 0.85–0.95 | 959 | 41.61 | 79.56 | +37.96 | +0.3583 | +0.3254 |
| >=0.95 | 270 | 42.59 | 71.11 | +28.52 | +0.2444 | +0.2470 |

Не строго монотонно: крайній bin >=0.95 не має найбільшого gain.
Це суперечить надто сильному правилу «чим ближче до 100%, тим завжди краще».

### Спеціальні діапазони і додавання heads після першої 1x1

| Порівняння / population | N GT | Lower Recall, % | Upper Recall, % | ΔRecall, в.п. | Mean ΔIoU |
|---|---:|---:|---:|---:|---:|
| Adjacent +single 1x1, area <0.8 | 9921 | 32.31 | 45.04 | +12.73 | +0.1081 |
| Adjacent +single 1x1, area >=0.8 | 1817 | 42.49 | 78.48 | **+35.99** | **+0.3067** |
| Adjacent +single 1x1, max-side <0.9 | 4775 | 29.01 | 31.27 | +2.26 | +0.0253 |
| Adjacent +single 1x1, max-side >=0.9 | 6963 | 37.23 | 63.21 | **+25.98** | **+0.2167** |
| Until→max, area <0.8 | 7249 | 38.89 | 39.18 | +0.29 | +0.00326 |
| Until→max, area >=0.8 | 1501 | 77.28 | 77.35 | +0.07 | +0.00265 |

Великі GT явно асоційовані з великим gain у coarse adjacent comparison,
але **великий area ratio не означає, що потрібно додавати heads без кінця**.
Після першої 1x1 виграш для area>=0.8 набагато менший, ніж у загальній
coarse-head таблиці. Це узгоджується з насиченням корисності coarse geometry,
але не доводить конкретний механізм learned features/anchor specialization.

Аналогічні bins для **width, height та max-side ratios**, для всіх трьох
pair kinds і coarse subsets, збережено в
[object_ratio_bins.csv](object_ratio_bins.csv). Empty bins позначені EMPTY,
undefined correlation — окремим status; їх не заповнювали нулями.

## 7. Який показник краще пов'язаний із gain?

Для adjacent comparison, що додає одну 1x1 head, усі 11 738 GT-exposures:

| Predictor | Pearson r із ΔIoU | Spearman rho із ΔIoU | Within-shape centered r із ΔIoU | Pearson r із Δsuccess |
|---|---:|---:|---:|---:|
| **Area ratio** | **0.3891** | **0.3798** | **0.4201** | **0.3581** |
| Width ratio | 0.3000 | 0.2517 | 0.3167 | 0.2701 |
| Height ratio | 0.3069 | 0.2964 | 0.3138 | 0.2878 |
| Max-side ratio | 0.3037 | 0.2401 | 0.2896 | 0.2779 |

Area ratio також має найбільший зв'язок із Δconfidence:
Pearson **0.2576**, Spearman **0.3103**.
Sensitivity check лише на primary GT (gt_index=0, **8704 exposures**):
area ratio vs ΔIoU Pearson **0.3181**, Spearman **0.3482**,
within-shape centered **0.3390**; сильніше, ніж width/height/max-side.
Отже, результат не зникає лише через включення малих secondary GT.

**Area ratio — найкращий описовий predictor серед перевірених, не доведений
causal explanation і не валідована модель прогнозування.** Помірні r
далекі від повного пояснення variance. GT клас, aspect ratio, clipping,
confidence threshold, scales/anchors, learned head specialization та source
повторюються/змінюються разом; within-shape centering не усуває всі confounders.
Немає незалежного test split, randomized crop-size intervention чи
source-cluster bootstrap; p-values/статистична значущість не заявляються.

## 8. Прямі відповіді на шість питань

1. **Чи можна відмовитися від manual pixel thresholds?**
   Для цього checkpoint на перевірених 36 формах — так, geometry-кандидат
   `until_1x1` успішно їх замінює. Для довільних ROI і production це ще не
   доведено; зміна потребує окремого review і реального selector benchmark.
2. **Чи достатньо `until_1x1`?**
   Достатньо як рекомендованого pilot-кандидата за accuracy/CPU компромісом:
   0 material losses vs legacy, 33/36 equivalent і 2 кращі vs local reference.
   Не ідентична oracle: при 255x255 пропускає Recall +0.74 в.п.;
   tiny 32/64 залишаються практично непридатними.
3. **Чи max дає gain після 1x1?**
   Так, невеликий localization/Recall gain у частині форм (6 material gains),
   але 29 equivalent і 1 AP loss. Середній mAP не кращий, CPU дорожчий.
   Тому max не рекомендовано як default; якщо pilot віддає пріоритет навіть
   малим Recall gains, це окрема альтернатива для затвердження.
4. **Яка policy для Stage 3?**
   `until_1x1`, із закритим перевіреним scope, явним tiny-ROI
   fallback-рішенням, моніторингом Recall exceptions і окремим CPU parity/
   end-to-end benchmark. Fallback тут не реалізовувався і не оцінювався.
5. **Гіпотеза про великий об'єкт?**
   Підтримана **асоціацією**, особливо area>=0.8 і max-side>=0.9.
   Причинність і достатність цього чинника не доведені.
6. **Кращий показник?**
   **Area ratio** серед чотирьох перевірених: сильніші raw/within-shape/
   primary-GT associations із localization gain. Не універсальна формула.

Пропоноване правило для майбутньої реалізації без pixel thresholds:

```text
Під час одного послідовного feature-extraction forward:
  обчислити наступний існуючий блок і його heads;
  перевірити spatial feasibility та loc/cls/anchor slots;
  якщо actual feature H == 1 і W == 1 — зупинитися;
  інакше продовжити до останнього існуючого feasible блоку.
```

Це **не** функція лише від min_dim і не рекомендація робити шість
повторних forwards на кожному runtime ROI. Жодної нової production
реалізації чи автоматичного перемикання в цьому етапі немає.

## 9. Артефакти та перевірки

Обов'язкові:
[policy_summary.csv](policy_summary.csv),
[per_shape_results.csv](per_shape_results.csv),
[per_object_crop_geometry.csv](per_object_crop_geometry.csv),
[object_ratio_bins.csv](object_ratio_bins.csv),
[latency_summary.csv](latency_summary.csv),
[metadata.json](metadata.json).

Детальні:
[per_object_pair_comparisons.csv](per_object_pair_comparisons.csv),
[object_ratio_correlations.csv](object_ratio_correlations.csv),
[per_depth_results.csv](per_depth_results.csv),
[per_depth_object_metrics.jsonl](per_depth_object_metrics.jsonl),
[predictions.jsonl](predictions.jsonl),
[geometry_probes.json](geometry_probes.json),
[analysis_metadata.json](analysis_metadata.json),
[execution.log](execution.log).

Графіки:
[mAP за policy](map_by_policy.png),
[CPU latency](latency_by_policy.png),
[selected depths](selected_depth_distribution.png),
[gain vs area ratio](gain_vs_gt_area_ratio.png),
[gain vs max-side ratio](gain_vs_gt_max_side_ratio.png).

На **24 576 accuracy + 216 geometry probes + 3456 CPU warm-up/timing =
28 248 forwards** перевірено
`N_loc_predictions == N_cls_predictions == N_anchors`. Усі structural
checks PASS; feasibility failures немає. **26 focused tests PASS**.
Усі required output shapes/metrics, geometry ratios, deltas, timing repeats,
source/output hashes і Stage-2d controls перевірено.

```powershell
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.feature_map_policy --device auto --threads 4
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.feature_map_analysis
.\.venv\Scripts\python.exe -m unittest discover -s experiments\roissd_128_anomaly\tests -v
```
