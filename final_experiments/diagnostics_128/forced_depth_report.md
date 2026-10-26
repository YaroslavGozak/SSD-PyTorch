# Етап 2b: примусова глибина ROI-SSD-MobileNet

**Статус: COMPLETE. Усі 15 конфігурацій фізично обчислюються.**
Production/inference policy, архітектуру та ваги не змінено; retraining не було.

## Протокол

- ROI-specialist із [manifest](../FINAL_EXPERIMENTS_MANIFEST.md):
  [20260928-phase2-bn-freeze/best_map.pt](../../trained_models/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt).
  SHA256: `7fa171ab669ac53f0d4f39586c08e61a60ca79fed98fd70874ef42bfd52356ce`.
- Ті самі 256 кадрів/GT з [locked subset](../../experiments/roissd_128_anomaly/data/subset.jsonl),
  SHA256: `576e8a3eb9b27b05abe1f20eef3da2bb4b4c87bcd20b4bf5f192225606bf54d2`.
  Усі tensor/GT hashes збіглися з [етапом 2](../02_diagnostics_128/preprocessing.csv).
  Для трьох режимів одного розміру повторно використано ті самі tensors.
- Project FixedSizeTestTransform, RGB/ImageNet normalization, seed `20261023`;
  confidence `0.5`, NMS IoU `0.45`, detection cap `200`.
  GT: 330 для 128 px, 329 для решти розмірів. Це object-centered crops,
  не повнокадровий чи tracker benchmark.
- CUDA, PyTorch `2.12.0+cu126`, deterministic algorithms.
  П'ять current-baselines відтворили mAP, Recall, confidence та anchor matching
  етапу 2 без числової різниці (допуск перевірки `1e-6`).
- Forced depth реалізовано лише scoped override методу конкретного instance;
  після кожного режиму перевірено відновлення початкової policy.

## Точність

У таблиці mAP/Recall наведено у відсотках. AP: project area-integral,
IoU `0.5`; Recall: macro-average за GT-supported класами при confidence `0.5`.

| Input | current L_R | current mAP / Recall | L_R-1 | mAP / Recall | L_R+1 | mAP / Recall | Зміна mAP для +1, в.п. |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128x128 | 4 | 42.74 / 45.51 | 3 | 0.28 / 0.83 | 5 | 42.74 / 45.51 | 0.00 |
| 132x132 | 4 | 2.37 / 3.07 | 3 | 1.03 / 1.42 | 5 | 58.80 / 60.09 | +56.43 |
| 136x136 | 4 | 3.89 / 4.36 | 3 | 2.59 / 2.81 | 5 | 59.61 / 61.13 | +55.72 |
| 140x140 | 4 | 2.88 / 3.37 | 3 | 1.78 / 2.13 | 5 | 60.34 / 62.26 | +57.47 |
| 144x144 | 5 | 59.60 / 61.52 | 4 | 4.09 / 4.90 | 6 | 59.38 / 61.52 | -0.23 |

**Forced L_R+1 усуває провал 132–140 px на цьому subset:** точність
відновлюється до рівня current 144 px (59.60% mAP). Вимкнення head 4 при
144 px повертає провал. Це парне причинне свідчення впливу depth gating
для цього checkpoint/protocol, а не лише кореляція між різними розмірами.

**128 px:** +1 не покращує й не погіршує mAP, Recall та post-NMS confidence;
кількість detections та їх score statistics незмінні. Зменшення до L_R=3
різко шкодить. **144 px:** +1 до L_R=6 трохи погіршує mAP (на 0.23 в.п.),
не змінює Recall; detections зростають із 254 до 294. Статистичну значущість
цієї малої різниці не оцінювали; підстав додавати шосту head немає.

## Структура та anchor coverage

Heads індексуються з нуля; L_R означає їх кількість. Для всіх 3840
оцінених прикладів (15 x 256), а також feasibility forwards перевірено
`N_loc_predictions == N_cls_predictions == N_anchors`. Невідповідностей,
невалідних boxes/scores чи фізично неможливих конфігурацій не було.

| Input | L_R | Активні heads | Просторові feature shapes | Anchors за heads | Разом |
|---|---:|---|---|---|---:|
| 128 | 3 | 0,1,2 | 8x8, 4x4, 2x2 | 256,96,24 | 376 |
| 128 | 4 | 0,1,2,3 | 8x8, 4x4, 2x2, 1x1 | 256,96,24,6 | 382 |
| 128 | 5 | 0,1,2,3,4 | 8x8, 4x4, 2x2, 1x1, 1x1 | 256,96,24,6,4 | 386 |
| 132/136/140 | 3 | 0,1,2 | 9x9, 5x5, 3x3 | 324,150,54 | 528 |
| 132/136/140/144 | 4 | 0,1,2,3 | 9x9, 5x5, 3x3, 2x2 | 324,150,54,24 | 552 |
| 132/136/140/144 | 5 | 0,1,2,3,4 | 9x9, 5x5, 3x3, 2x2, 1x1 | 324,150,54,24,4 | 556 |
| 144 | 6 | 0,1,2,3,4,5 | 9x9, 5x5, 3x3, 2x2, 1x1, 1x1 | 324,150,54,24,4,4 | 560 |

| Input | Mean best-anchor IoU, current -> +1 | Частка GT із IoU >= 0.5, current -> +1 | Mean raw max confidence, current -> +1 |
|---|---:|---:|---:|
| 128 | 0.5262 -> 0.6686 | 54.85% -> 79.39% | 0.6720 -> 0.6724 |
| 132 | 0.4213 -> 0.6882 | 23.10% -> 89.97% | 0.2196 -> 0.8950 |
| 136 | 0.4238 -> 0.6889 | 23.40% -> 89.67% | 0.2262 -> 0.8956 |
| 140 | 0.4263 -> 0.6891 | 23.71% -> 89.67% | 0.2260 -> 0.9060 |
| 144 | 0.6889 -> 0.7513 | 89.36% -> 89.36% | 0.9063 -> 0.9135 |

При 132–140 px додавання навченої head 4 (`conv10_2`, 1x1, чотири anchors)
одночасно відновлює coverage і впевненість. Простого збільшення anchor IoU
недостатньо для покращення AP: приклади 128 px і L_R=6 при 144 px це показують.
Це не помилка узгодження кількості anchors/predictions.

## Рекомендація для фінального CPU-pilot

1. **Локальна зміна правила без retraining обґрунтована для перевірених
   квадратних ROI цього MobileNet-checkpoint.** Не використовувати правило
   «завжди обирати найбільшу фізично можливу глибину».
2. Для pilot із перевіреними розмірами рекомендується **L_R=4 при 128x128;
   L_R=5 при 132x132, 136x136, 140x140 і 144x144**. Не зменшувати глибину;
   при 144 px не вмикати L_R=6. Окрема п'ята head при 128 px не дала
   виграшу точності, тому додаткова робота для CPU не виправдана цими даними.
3. Кандидат подальшого загального правила `get_max_feature_layer_by_roi_size()`:
   зберегти межі 32/64/96, перенести межу L_R=4 з 140 на 128, залишити
   верхню межу L_R=5 рівною 268. **Це не сертифікація всього діапазону:**
   129–131 px, проміжні неперевірені розміри, прямокутні ROI та інші checkpoints
   не оцінювалися; не переносити висновок на VGG ROI-SSD.
4. Для фінального CPU-pilot обмежити нову policy перевіреними квадратними
   buckets; для ширшої зміни спочатку перевірити відсутні межі/форми.
   Цей запуск перевіряв якість на CUDA, **не CPU latency, Raspberry Pi чи
   end-to-end tracking**. CPU backend parity і бюджет затримки мають бути
   підтверджені самим pilot. Subset уже використовувався в діагностиці:
   це не незалежний test split і не оцінка узагальнення.

**У межах етапу 2b production policy залишено незмінною.** Рекомендація
призначена для окремого наступного впровадження/перевірки, не є прихованим
перемиканням inference.

## Відтворення та повні логи

```powershell
.\.venv\Scripts\python.exe -m experiments.roissd_128_anomaly.forced_depth --device auto
.\.venv\Scripts\python.exe -m unittest discover -s experiments\roissd_128_anomaly\tests -v
```

[CSV: усі 15 запусків](forced_depth_probe.csv) містить точні метрики [0,1],
NCHW shapes, heads, anchor counts, перевірки та provenance hashes.
`mean_confidence`/`max_confidence` агрегують усі повернені detections після
confidence/NMS/cap; `mean_raw_max_confidence` є середнім за кадрами максимумом
foreground softmax до фільтрації, як в етапі 2. IoU/coverage агрегують усі GT
без confidence-фільтра. [Metadata](forced_depth_metadata.json) фіксує COMPLETE,
ідентичність входів/baselines, незмінність weights/policy та source hashes.
Усі 11 focused regression tests пройдено.
