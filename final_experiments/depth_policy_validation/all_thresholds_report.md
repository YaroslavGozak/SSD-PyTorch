# Stage 2d: усі пороги адаптивної глибини ROI-SSD-MobileNet

**COMPLETE: 140 форм, 280 adjacent-depth конфігурацій, 71 680 оцінювань
accuracy. 280/280 PASS, 0 INFEASIBLE. Production policy і ваги не змінено.**

## Головне рішення

**Поточна функція лише від `min_dim` недостатня для оптимального вибору глибини.**
Не рекомендується замінювати старі пороги одним новим scalar-набором для
довільних ROI. Пороги 32/64/96/140/268 не отримали автоматичного схвалення:
32 не дає достатніх accuracy-свідчень; 64, 140 і 268 мають shape dependence;
96 потрібно знижувати, але точний onset нижче найменшої перевіреної точки
88 ще не встановлено.

Це експериментальні висновки для конкретного checkpoint і crop protocol,
не зміна production code та не твердження про оптимальність серед усіх шести
глибин: за завданням кожен перехід порівнював лише дві сусідні глибини.

## 1. Модель, дані та provenance

- ROI-specialist із [manifest](../FINAL_EXPERIMENTS_MANIFEST.md):
  [20260928-phase2-bn-freeze/best_map.pt](../../trained_models/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt).
  SHA256 `7fa171ab669ac53f0d4f39586c08e61a60ca79fed98fd70874ef42bfd52356ce`.
- [Locked subset](../../experiments/roissd_128_anomaly/data/subset.jsonl):
  ті самі 256 object-centered source samples.
  SHA256 `576e8a3eb9b27b05abe1f20eef3da2bb4b4c87bcd20b4bf5f192225606bf54d2`.
  Config snapshot SHA256
  `81ec5a55754462f2db9ad6630ee9a37fe2f79d00d43648cdbe0701e734c813bd`.
- Project FixedSizeTestTransform, RGB/ImageNet normalization, seed `20261023`,
  confidence `0.5`, NMS IoU `0.45`, cap `200`. Використано незмінений
  evaluator Stage 2b; AP — project area-integral AP@0.5, Recall — class-macro
  operating-point Recall@0.5, лише за GT-supported класами.
- У кожній парі depths використано **ті самі tensors і GT**. Visibility/GT
  між різними H/W може відрізнятися; це не повнокадровий resize benchmark.
  Збережено 35 840 per-sample tensor/GT hashes. Образи/labels перевірено
  за locked hashes, source/config/checkpoint hashes — до/після запуску.
- **16 controls** збіглися зі Stage 2c за input hashes, accuracy/confidence/
  anchor metrics, detection counts і структурою (metric tolerance `1e-6`).
  Старі CSV/metadata 2b/2c та модельні sources не змінено.
- Accuracy: NVIDIA GeForce RTX 3060 Ti, PyTorch `2.12.0+cu126`,
  deterministic algorithms, cuDNN benchmark off.
- CPU timing: **Intel Core i7-11700K @ 3.60 GHz**, Windows,
  4 intra-op / 1 inter-op thread. Це desktop CPU, **не Raspberry Pi**.
  CPU accuracy на всьому subset окремо не оцінювалася.

Повні версії, hashes, timestamps і весь plan:
[all_thresholds_metadata.json](all_thresholds_metadata.json).

## 2. Sweep і метод latency

Для кожного B=32/64/96/140/268 перевірено квадрати
`B + {-8,-4,-2,-1,0,1,2,4,8}`. Додано всі обов'язкові special points:
124/127/128/129/132/136/140/144 для L4→L5 та
240/248/252/256/260/264/268/272/280 для L5→L6, додатково 255/257.
Разом **56 квадратних і 84 прямокутні форми**.

Прямокутники: `M x round(1.25M)`, `round(1.25M) x M`,
`M x round(1.5M)`, `round(1.5M) x M`.
Округлення до найближчого pixel, .5 вгору, **не до stride**.

| Перехід | Прямокутні min_dim M |
|---|---|
| L1→L2 | 31,32,33 |
| L2→L3 | 63,64,65 |
| L3→L4 | 95,96,97 |
| L4→L5 | 127,128,129,139,140,141 |
| L5→L6 | 255,256,257,267,268,269 |

Latency виміряна для всіх **280 CUDA конфігурацій**, додатково для
**70 CPU конфігурацій**: квадрат і чотири прямокутники при
M=32/64/96/128/140/256/268, обидві depths.

- Batch=1; **6 warm-up + 30 timed forwards на depth/device/shape**.
- Циклічно використовуються source IDs 0/85/170; lower/upper порядок
  чергується щоповтору, стартова depth — між формами.
- Wall-clock `perf_counter_ns`; CUDA synchronize до/після forward.
  Tensors уже на потрібному device. Crop, H2D, model construction не входять;
  feature extraction, heads, decode, confidence filtering, NMS/cap входять.
- На всіх feasibility/accuracy/timing forwards перевірено slots.
  Для timing використано легкий shape-only guard без дублювання softmax/IoU.
  **Guard overhead входить у час**; це контрольований діагностичний timing,
  а не чистий production benchmark без hooks.
- Збережено всі repeats, median і p95. CPU не був ізольований/закріплений
  affinity; turbo, OS noise та shared-machine load не контролювалися.
  Малі різниці latency не трактуються як доведений speedup.

Описова practical-equivalence tolerance: **0.5 в.п. для обох AP і Recall**,
задекларована в metadata до оцінювання. Це не significance test.
Якщо обидві зміни в межах tolerance, перевага меншій depth заради CPU.
Матеріальний виграш: хоча б одна метрика >+0.5 в.п., інша не гірша ніж
-0.5 в.п.; матеріальна втрата — будь-яка метрика <-0.5 в.п.
У всьому sweep матеріальних втрат від більшої depth не було, але були
plateaus без корисного виграшу та з додатковою latency.

## 3. Підсумок переходів: старі та рекомендовані межі

CPU effect нижче — **медіана парних різниць median latency** за всіма CPU
формами цього переходу, не різниця pooled medians і не end-to-end FPS.
Confidence — якісна сила локальних свідчень, не статистичний інтервал.

| Transition | Old boundary | Status | Recommended boundary | Evidence / accuracy effect | CPU latency effect | Confidence |
|---|---:|---|---|---|---|---|
| L1→L2 | 32 | **INSUFFICIENT_EVIDENCE** | Не встановлено; 32 не сертифіковано | 21/21 пар: AP=Recall=0 при обох depths | +1.094 ms / +20.54% | Висока для нульового локального результату; недостатня для порогу |
| L2→L3 | 64 | **SHAPE_DEPENDENT** | Square onset 65; rectangles виграють вже при M=63 | 15 пар із матеріальним виграшем, 6 equivalent; AP до +16.61 в.п. | +0.983 ms / +13.43% | Висока для протестованих форм |
| L3→L4 | 96 | **LOWER** | Точна межа не встановлена; L4 краща вже при square 88 | 21/21 пар кращі; AP +11.41..+60.10 в.п. | +0.946 ms / +9.34% | Висока для напрямку; низька для точного onset <88 |
| L4→L5 | 140 | **SHAPE_DEPENDENT** | Square-кандидат 128; не універсальний scalar | 34 material-gain / 3 equivalent; rectangles M=127/128 теж потребують L5 | +1.358 ms / +10.50% | Висока локально; довільні форми не сертифіковані |
| L5→L6 | 268 | **SHAPE_DEPENDENT** | Square-кандидат 256; винятки за орієнтацією при M=255/256 | 32 material-gain / 8 equivalent; при M≥257 виграють усі перевірені форми | +0.624 ms / +2.59% | Висока локально; точні onset для довільних форм не встановлені |

Для L3→L4 статус LOWER означає **доведений напрямок зміни**, не схвалення
довільного нового числа 87/88. Найменша протестована square-точка 88 вже
виграє; точок, що обмежують onset знизу, у цьому локальному sweep немає.
Прямокутники для цього переходу тестувалися при M=95/96/97, не при 88.

Машинозчитуваний підсумок: [boundary_summary.csv](boundary_summary.csv).

## 4. L1→L2, стара межа 32

Квадрати **24,28,30,31,32,33,34,36,40** та всі 12 rectangles із M=31/32/33:
**mAP=0, Recall=0 при L1 і L2**.
При 32x32 mean raw max confidence `0.0022 -> 0.0022`, mean best-anchor IoU
`0.0313 -> 0.1180`, частка GT із IoU≥0.5 `0% -> 0%`, detections `0 -> 0`.
Структура коректна: anchors `16 -> 22`; features `2x2 -> 2x2,1x1`.

L2 при 32x32: CPU median `5.010 -> 5.946 ms` без виграшу accuracy.
Немає підстав витрачати більше CPU на цих перевірених формах, але немає
й підстав оголошувати L1 придатною detector-depth лише тому, що L2 теж не
детектує. **INSUFFICIENT_EVIDENCE**, а не KEEP чи довільне RAISE.
Нульовий AP не доводить фізичної неможливості detection або помилки anchors:
це спостереження для даного checkpoint/protocol/threshold.

## 5. L2→L3, стара межа 64

| Square | mAP L2 / L3, % | Recall L2 / L3, % |
|---:|---:|---:|
| 56 | 0.00 / 0.00 | 0.00 / 0.00 |
| 60 | 0.00 / 0.00 | 0.00 / 0.00 |
| 62 | 0.00 / 0.00 | 0.00 / 0.00 |
| 63 | 0.00 / 0.00 | 0.00 / 0.00 |
| 64 | 0.00 / 0.00 | 0.00 / 0.00 |
| 65 | 0.00 / 16.33 | 0.00 / 17.30 |
| 66 | 0.00 / 14.89 | 0.00 / 15.92 |
| 68 | 0.00 / 16.61 | 0.00 / 16.95 |
| 72 | 0.00 / 14.48 | 0.00 / 15.94 |

Межа 64 виглядає достатньою **лише для локального square-порівняння**.
Для 63x79 L3 вже підвищує AP `0 -> 9.37%`, Recall `0 -> 12.11%`;
для 79x63 — AP `0 -> 7.20%`. При M=64 у чотирьох rectangles AP
`0 -> 9.88/0.88/8.53/0.54%` (порядок: 64x80,64x96,80x64,96x64).
При 65x98 обидві depths знову мають нульовий AP, тоді як square 65x65
виграє на 16.33 в.п. Отже, min_dim не описує sensitivity до форми.

L3 не має матеріальної втрати в жодній із 21 пар, але еквівалентні squares
не виправдовують зайві CPU витрати: при 64x64 `6.633 -> 7.952 ms`.
Окремі rectangles із більшим виграшем виправдовують інший вибір.
**SHAPE_DEPENDENT**; абсолютна якість навіть L3 лишається низькою,
тому ці результати не сертифікують tiny-ROI detection для pilot.

## 6. L3→L4, стара межа 96

| Square | mAP L3 / L4, % | Recall L3 / L4, % |
|---:|---:|---:|
| 88 | 2.94 / 46.65 | 3.44 / 49.14 |
| 92 | 2.85 / 48.03 | 2.85 / 50.29 |
| 94 | 2.36 / 49.40 | 2.36 / 50.78 |
| 95 | 3.65 / 49.08 | 4.04 / 50.34 |
| 96 | 3.46 / 49.25 | 3.67 / 50.65 |
| 97 | 0.00 / 59.67 | 0.00 / 60.99 |
| 98 | 0.00 / 59.31 | 0.00 / 60.26 |
| 100 | 0.00 / 59.05 | 0.00 / 60.04 |
| 104 | 0.00 / 60.10 | 0.00 / 61.25 |

Усі 12 rectangles із M=95/96/97 теж виграють за AP і Recall.
Найменший AP gain серед усіх 21 пар — **+11.41 в.п.** для 143x95;
найбільший — **+60.10 в.п.** для square 104.
При 96x96 mean raw max confidence `0.3035 -> 0.7156`,
mean best-anchor IoU `0.2570 -> 0.5175`, coverage `3.63% -> 54.38%`.
Додана head має 1x1 feature і 6 anchors: total `222 -> 228`.

CPU 96x96 `8.951 -> 10.156 ms`; для перевірених значних gains додаткова
ціна не виглядає непропорційною. **LOWER**, але onset не локалізовано:
88 — найменший перевірений square, не доведений мінімально достатній M.
Нове загальне scalar-число тут поки не рекомендується.

## 7. L4→L5, стара межа 140

| Square | mAP L4 / L5, % | Recall L4 / L5, % |
|---:|---:|---:|
| 124 | 44.15 / 44.15 | 46.68 / 46.68 |
| 127 | 43.23 / 43.23 | 45.74 / 45.74 |
| 128 | 42.74 / 42.74 | 45.51 / 45.51 |
| 129 | 0.89 / 58.47 | 1.00 / 59.83 |
| 132 | 2.37 / 58.80 | 3.07 / 60.09 |
| 136 | 3.89 / 59.61 | 4.36 / 61.13 |
| 138 | 2.29 / 59.53 | 3.07 / 60.90 |
| 139 | 3.08 / 58.98 | 3.71 / 60.39 |
| 140 | 2.88 / 60.34 | 3.37 / 62.26 |
| 141 | 2.83 / 57.35 | 3.29 / 59.36 |
| 142 | 2.78 / 59.98 | 3.62 / 61.58 |
| 144 | 4.09 / 59.60 | 4.90 / 61.52 |
| 148 | 4.25 / 52.19 | 9.19 / 54.63 |

Для squares підтримано попередній кандидат межі **128**: L4 при 128
еквівалентна L5, а при 129 відбувається радикальний виграш L5.
Але всі **24 rectangles** від M=127 до M=141 теж виграють від L5,
включно з формами, які scalar-кандидат 128 залишає на L4.

| Rectangle | mAP L4 / L5, % | Recall L4 / L5, % |
|---|---:|---:|
| 127x159 | 21.66 / 32.86 | 27.94 / 37.66 |
| 127x191 | 18.15 / 29.18 | 28.43 / 40.02 |
| 159x127 | 11.82 / 21.80 | 18.97 / 30.25 |
| 191x127 | 10.05 / 13.59 | 20.26 / 26.02 |
| 128x160 | 20.82 / 32.90 | 27.08 / 37.46 |
| 128x192 | 19.15 / 30.31 | 30.52 / 41.62 |
| 160x128 | 11.68 / 21.96 | 20.47 / 32.23 |
| 192x128 | 9.52 / 13.39 | 20.47 / 25.18 |

При 140x140 coverage `23.71% -> 89.67%`, raw confidence
`0.2260 -> 0.9060`; total anchors `552 -> 556`, без mismatch.
При 128x128 coverage теж зростає `54.85% -> 79.39%`, але AP/Recall і
кількість returned detections не змінюються. Coverage сама по собі не є
доказом accuracy gain.

CPU square 128 `11.956 -> 13.492 ms` **без accuracy-виграшу**;
square 140 `12.154 -> 14.061 ms` із +57.47 в.п. AP.
Це прямий accuracy/CPU аргумент на користь **SHAPE_DEPENDENT** вибору,
а не «завжди вмикати максимальну фізично можливу depth».
Межа 127 також не доведена: square 127 не отримує виграшу, а нижчі
прямокутні min_dim для цього переходу не перевірені.

## 8. L5→L6, стара межа 268

| Square | mAP L5 / L6, % | Recall L5 / L6, % |
|---:|---:|---:|
| 240 | 63.37 / 63.34 | 63.84 / 63.84 |
| 248 | 62.02 / 61.84 | 63.00 / 63.00 |
| 252 | 60.64 / 60.57 | 61.72 / 61.72 |
| 255 | 60.39 / 60.73 | 61.93 / 62.67 |
| 256 | 60.33 / 60.51 | 61.83 / 62.20 |
| 257 | 44.09 / 49.62 | 47.60 / 52.75 |
| 260 | 44.79 / 53.14 | 48.19 / 55.52 |
| 264 | 46.14 / 56.52 | 49.68 / 58.75 |
| 266 | 50.25 / 58.01 | 52.66 / 59.82 |
| 267 | 48.53 / 58.70 | 51.01 / 60.63 |
| 268 | 49.45 / 59.26 | 52.17 / 61.42 |
| 269 | 46.83 / 57.56 | 49.66 / 59.87 |
| 270 | 48.96 / 59.77 | 52.12 / 62.27 |
| 272 | 53.67 / 64.60 | 55.82 / 66.47 |
| 276 | 59.72 / 62.89 | 61.36 / 64.28 |
| 280 | 61.92 / 64.95 | 63.83 / 66.49 |

На всіх перевірених формах із M≥257 L6 покращує AP і Recall; при square
268 AP +9.82 в.п., Recall +9.25 в.п., CPU `20.522 -> 22.411 ms`.
Отже, стара межа 268 запізнюється; для square-порівнянь **256** є кращим
локальним кандидатом. Проте L6 не повністю усуває провал square 257:
49.62% AP лишається нижчим за 60.33% при square 256 із L5.
Це різні crops, не парна оцінка єдиної причини залишкового падіння.

При M=255/256 потрібні форма **і орієнтація**:

| Shape | ΔmAP L6−L5, в.п. | ΔRecall, в.п. | Вибір за описовою tolerance |
|---|---:|---:|---|
| 255x255 | +0.34 | +0.74 | Recall gain; не equivalent, CPU тут окремо не виміряно |
| 256x256 | +0.18 | +0.37 | L5: equivalent, зайва CPU ціна +0.813 ms |
| 256x320 | +0.78 | +1.47 | L6: CPU +0.678 ms |
| 256x384 | +1.31 | +1.44 | L6: CPU +0.446 ms |
| 320x256 | +0.35 | +0.48 | L5: equivalent |
| 384x256 | -0.15 | +0.00 | L5: equivalent |

При M=255 landscape rectangles теж виграють, тоді як transposes практично
еквівалентні. Тому scalar onset не є оптимальним для всіх форм.
Необхідно враховувати **ordered H/W або signed W/H**: одного симетричного
`max_dim/min_dim` недостатньо, щоб розрізнити ці орієнтації.
**SHAPE_DEPENDENT**, із доведеним напрямком lowering для square-кандидата.
Точки 258/259 і всі можливі aspect ratios не перевірялися.

Узагальнення висновку Stage 2b «немає підстав для L6» на всі 144+ px
було б неправильним: тоді L6 тестувалася при 144x144, тепер її користь
експериментально показана для значно більших ROI.

## 9. CPU ціна та p95

| Square | Lower→upper | Median, ms | p95, ms | Accuracy effect |
|---:|---|---:|---:|---|
| 32 | L1→L2 | 5.010 → 5.946 | 5.839 → 7.366 | Обидві нульові |
| 64 | L2→L3 | 6.633 → 7.952 | 8.043 → 8.831 | Обидві нульові |
| 96 | L3→L4 | 8.951 → 10.156 | 10.257 → 11.601 | AP +45.79 в.п. |
| 128 | L4→L5 | 11.956 → 13.492 | 13.858 → 15.206 | Еквівалентні |
| 140 | L4→L5 | 12.154 → 14.061 | 14.509 → 15.890 | AP +57.47 в.п. |
| 256 | L5→L6 | 20.421 → 21.234 | 25.399 → 26.700 | Практично еквівалентні |
| 268 | L5→L6 | 20.522 → 22.411 | 23.683 → 26.828 | AP +9.82 в.п. |

Діапазони CPU overhead за всіма representative формами:
L1→L2 `+0.936..+1.388 ms`, L2→L3 `+0.274..+1.319`,
L3→L4 `+0.551..+1.941`, L4→L5 `+0.524..+1.907`,
L5→L6 `-0.115..+1.986`.
Остання від'ємна мала різниця — не доказ прискорення від додаткової head.
Великі gains на L3→L4/L4→L5 та M≥257 для L5→L6 мають помірну виміряну
CPU ціну. На accuracy plateaus зайву depth обирати не слід.
Без затвердженого latency budget та частот реальних ROI не можна обчислити
чесний єдиний прогноз FPS або середній overhead майбутньої pipeline.

## 10. Рекомендація для наступного CPU-pilot

**Нову реалізацію `get_max_feature_layer_by_roi_size(min_dim)` не наводимо
як затверджену:** дані не підтримують оптимальне scalar rule для всіх ROI.
Збереження старого коду зараз — лише відсутність автоматичного deployment,
**не** рекомендація його правильності чи перенесення старих accuracy-провалів
у фінальний pilot.

Проста shape-aware пропозиція для окремого схвалення — **закритий whitelist
перевірених H×W buckets із фіксованою depth**, а не інтерполяція нових порогів:

| Перевірений набір | Кандидат depth / дія |
|---|---|
| Tiny ROI біля 32 | Немає підтвердженої корисної detector-depth серед L1/L2; окремо перевірити fallback/full-frame, не сертифікувати нульову точність |
| Square 65/66/68/72; rectangles із матеріальним L3 gain біля 63–65 | L3; не поширювати висновок про корисність на всі tiny ROI |
| Squares 88/92/94/95/96/97/98/100/104 і перевірені rectangles M=95/96/97 | L4 |
| Squares 124/127/128 | L4 |
| Squares 129/132/136/138/139/140/141/142/144/148 і всі перевірені rectangles M=127/128/129/139/140/141 | L5 |
| Squares 240/248/252/256 та practically equivalent portrait rectangles M=255/256 | L5 |
| Landscape rectangles M=255/256 із матеріальним gain; усі перевірені форми M≥257 | L6 |
| Square 255 | Окреме рішення: AP +0.34, Recall +0.74 в.п.; CPU не виміряно саме тут, не оголошувати equivalent |

Whitelist має зберігати **обидві впорядковані сторони H/W**, не лише M чи
симетричний aspect ratio. Його можна сформувати з парних результатів за
задекларованим AP/Recall tolerance із tie-break на меншу depth; це offline
таблиця для затвердження, не нова складна runtime-евристика.
Для неперевірених H/W не робити автоматичних висновків: перед deployment
потрібні додаткові парні тести або окремо затверджений fallback.

Фактичний shape наступної feature map можна використати як наступну просту
гіпотезу: наприклад, при 128x128 L4 map `1x1`, при 128x160 — `1x2`;
при 256x256 L5 map `1x1`, при 256x320 — `1x2`. Проте правило «додай head,
якщо map не 1x1» **не доведене** для всіх переходів і не розрізняє всі
ефекти орієнтації; реалізовувати його без окремого порівняння не слід.

Перш ніж затверджувати універсальну policy для Stage 3:
локалізувати нижню межу L3→L4 нижче 88 із rectangles; дослідити корисні
глибини/fallback для tiny crops; перевірити непокриті розміри/aspect ratios,
CPU accuracy parity, реальний ROI distribution і end-to-end latency.
Незалежний test split, статистична значущість, Raspberry Pi latency та
tracker/full-frame schedule у цьому експерименті не оцінювалися.

## Файли та відтворення

- [all_thresholds_results.csv](all_thresholds_results.csv): усі 280
  конфігурацій, AP/Recall/raw confidence/anchor IoU/coverage, H/W/min_dim,
  forced/current depth, active heads, NCHW shapes, anchors/head, total,
  loc/cls counts, latency median/p95 та provenance.
- [latency_summary.csv](latency_summary.csv): 350 device/depth/shape timing
  записів, включно з 70 CPU, усі repeats і warm-up/count settings.
- [all_thresholds_metadata.json](all_thresholds_metadata.json),
  [boundary_summary.csv](boundary_summary.csv),
  [input_hashes.jsonl](input_hashes.jsonl).
- [execution.log](execution.log): виконання; [paired_evidence.log](paired_evidence.log):
  компактне порівняння всіх 140 пар.
- На кожному з **71 680 accuracy + 280 feasibility + 12 600 latency/warm-up
  = 84 560 forwards** slots збіглися. **18 focused tests PASS**.

```powershell
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.all_thresholds --device auto --threads 4 --cpu-name "Intel Core i7-11700K @ 3.60GHz"
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.threshold_evidence
.\.venv\Scripts\python.exe -m unittest discover -s experiments\roissd_128_anomaly\tests -v
```
