# Етап 2. Діагностика аномалії ROI-SSD на вході 128×128

## Передумова

Цей етап запускається **лише після завершення етапу 1**.

Обов'язково прочитати:

`final_experiments/FINAL_EXPERIMENTS_MANIFEST.md`

Використовувати **саме вибрані на етапі 1 найкращі checkpoint-и**:

- ROI-SSD;
- ROI-SSD-MobileNet.

Не шукати нові `best.pt` / `.pth` самостійно і не підміняти checkpoint-и.

## Мета

Встановити, чому в попередньому експерименті ROI-SSD конфігурація 128×128 дала аномально низький результат порівняно з 96×96 і 160×160.

Перевірити, чи причина пов'язана з:

- перемиканням адаптивної глибини мережі;
- набором активних detection heads;
- формами feature maps;
- кількістю та формою сформованих anchor/default boxes;
- невідповідністю між predictions та anchors;
- anchor-to-GT matching;
- preprocessing;
- або старою/неактуальною експериментальною точкою.

Додатково визначити, чи успадковує ROI-SSD-MobileNet ту саму проблему.

## Головне правило

**Не виправляти модель до того, як проблема локалізована і задокументована.**

Спочатку отримати відтворюваний diagnostic report. Якщо знайдено implementation bug — зафіксувати до/після і лише тоді виправляти.

## 1. Відтворюваність старої аномалії

Для ROI-SSD і ROI-SSD-MobileNet окремо виконати контрольну оцінку для:

- 96×96;
- 128×128;
- 160×160.

Використовувати:

- один і той самий checkpoint;
- один і той самий набір прикладів;
- однаковий preprocessing;
- однакові confidence/NMS parameters;
- однакову metric implementation.

Зберегти:

- `mAP@0.5`;
- `mAP@0.5:0.95`, якщо доступний;
- `Recall@0.5`;
- кількість GT;
- кількість predictions до NMS;
- кількість predictions після NMS;
- середню кількість predictions на image;
- середню максимальну confidence;
- AP per class.

### Питання, на яке треба відповісти

Чи відтворюється аномалія 128×128 на **фінальному best ROI-SSD checkpoint**?

І окремо:

Чи відтворюється вона на **best ROI-SSD-MobileNet checkpoint**?

## 2. Щільний sweep розмірів

Для діагностики виконати sweep квадратних входів:

`96, 100, 104, ..., 156, 160`

тобто крок 4 пікселі.

Якщо модель технічно не приймає окремі розміри, **не округляти їх мовчки**. Записати:

- requested shape;
- actual tensor shape;
- факт padding/rounding;
- причину обмеження.

Для швидкості цей sweep можна виконати на репрезентативному фіксованому subset, наприклад 200–500 прикладів, але цей subset має бути **однаковим для всіх input sizes і для обох ROI-SSD моделей**.

Зберегти CSV:

`final_experiments/diagnostics_128/size_sweep_metrics.csv`

Мінімальні поля:

- `model_family`
- `checkpoint_id`
- `requested_h`
- `requested_w`
- `actual_h`
- `actual_w`
- `num_samples`
- `map50`
- `map5095`
- `recall50`
- `predictions_pre_nms_mean`
- `predictions_post_nms_mean`
- `max_confidence_mean`

## 3. Інструментувати внутрішню структуру мережі

Для кожного досліджуваного input size залогувати:

- actual input tensor shape;
- активні backbone blocks / layers;
- останній реально обчислений backbone level;
- список активних detection heads;
- shape feature map для кожної активної detection head;
- кількість spatial positions кожної head;
- anchors per location;
- кількість anchors/default boxes кожної head;
- загальну кількість anchors;
- shape localization output;
- shape classification output;
- загальну кількість decoded boxes.

Зберегти:

`final_experiments/diagnostics_128/internal_shapes.csv`

Бажані поля:

- `model_family`
- `input_h`
- `input_w`
- `active_backbone_depth`
- `head_index`
- `head_active`
- `feature_h`
- `feature_w`
- `anchors_per_location`
- `num_anchors_head`
- `loc_predictions_head`
- `cls_predictions_head`

## 4. Обов'язкові assertions

Для кожного forward pass перевірити:

```text
N_localization_predictions == N_classification_predictions
N_localization_predictions == N_generated_anchors
```

Якщо output організований по класах/координатах і потребує reshape — порівнювати кількість **box slots**, а не raw tensor elements.

При невідповідності:

- зупинити відповідний diagnostic run;
- записати input size;
- active heads;
- feature map shapes;
- кількість anchors;
- output shapes;
- stack trace / assertion message.

## 5. Перевірити геометрію anchors

Для кожного input size зберегти статистику:

- minimum anchor width/height;
- maximum anchor width/height;
- median width/height;
- кількість anchors;
- кількість anchors, що частково виходять за межі image;
- кількість anchors, що повністю виходять за межі image, якщо такі можливі;
- статистику aspect ratios.

Перевірити normalization/de-normalization:

- чи використовуються правильні `H` і `W`;
- чи не переплутані width/height;
- чи масштаб anchors дійсно відповідає actual input shape;
- чи не використовується випадково fixed 300×300 normalization для variable-size input.

## 6. Anchor-to-GT matching

Для кожного GT bbox обчислити:

`max_anchor_iou = max IoU(GT, anchor)`

Для кожного input size зберегти:

- mean `max_anchor_iou`;
- median;
- p5;
- p25;
- частку GT з `max_anchor_iou >= 0.3`;
- частку GT з `max_anchor_iou >= 0.5`;
- середню кількість positive matches;
- positive matches per GT.

Зберегти:

`final_experiments/diagnostics_128/anchor_matching.csv`

Це потрібно зробити **до NMS і незалежно від predicted confidence**.

## 7. Перевірити boundary transitions

Особливо уважно проаналізувати місця, де при збільшенні input size:

- активується новий backbone block;
- з'являється нова detection head;
- змінюється feature map resolution;
- різко змінюється кількість anchors.

У звіті побудувати таблицю:

| Input size | Active depth | Active heads | Feature maps | Total anchors | mAP50 |
|---:|---:|---|---|---:|---:|

Потрібно знайти, чи збігається падіння accuracy зі структурною межею.

## 8. Якщо аномалію знайдено

Розділити результат на один із типів.

### A. Output/anchor mismatch

Ознаки:

- кількість predictions не збігається з anchors;
- проблема виникає на конкретному size/depth/head transition.

Висновок: implementation bug.

### B. Anchor geometry/matching failure

Ознаки:

- counts узгоджені;
- `max_anchor_iou` або positive matches різко падають.

Висновок: проблема генерації/масштабування default boxes для конкретної конфігурації.

### C. Feature/head quality failure

Ознаки:

- counts і anchor matching нормальні;
- confidence/mAP різко падають;
- проблема корелює з певною active head/depth.

Висновок: проблема learned features / head specialization / training distribution.

### D. Аномалія не відтворюється

Якщо best checkpoint більше не показує падіння 128×128:

- не вигадувати причину старої точки;
- зафіксувати, що старий результат не відтворився на фінальній моделі;
- повторити таблицю розмірів на актуальному checkpoint;
- рекомендувати замінити стару точку в дисертації актуальним результатом.

## 9. Якщо виявлено implementation bug

Після того як bug задокументований:

1. створити мінімальне виправлення;
2. не змінювати іншу логіку моделі;
3. повторити diagnostic sweep;
4. повторити 96/128/160 accuracy test;
5. у звіті показати `before` і `after`.

Не перетреновувати модель автоматично. Якщо виправлення змінює семантику anchors настільки, що checkpoint більше не валідний, **зупинитися і повідомити**, що потрібне retraining/re-evaluation.

## 10. Обов'язкові outputs

Створити директорію:

`final_experiments/diagnostics_128/`

У ній:

- `diagnostic_report.md`
- `size_sweep_metrics.csv`
- `internal_shapes.csv`
- `anchor_geometry.csv`
- `anchor_matching.csv`
- `errors_or_assertions.log`
- plots, якщо вони реально допомагають:
  - `map50_vs_input_size.png`
  - `total_anchors_vs_input_size.png`
  - `max_anchor_iou_vs_input_size.png`

У `diagnostic_report.md` окремо дати:

### ROI-SSD
- чи відтворюється 128×128 anomaly;
- встановлена або найбільш обґрунтована причина;
- evidence;
- чи потрібне виправлення.

### ROI-SSD-MobileNet
- чи є та сама проблема;
- чи безпечно використовувати модель у фінальному end-to-end benchmark.

## STOP CONDITIONS

Не переходити до етапу 3, якщо:

- ROI-SSD-MobileNet має невиправлений structural/anchor bug на реальних segment sizes;
- predictions та anchors не узгоджуються;
- class compatibility з manifest порушена;
- variable-size inference працює некоректно;
- виправлення потребує retraining, але retraining ще не виконаний.

Якщо проблема існує лише у старому ROI-SSD, але ROI-SSD-MobileNet коректна, це **не блокує** фінальний pipeline з ROI-SSD-MobileNet. Зафіксувати це у звіті.

## Критерій завершення

Етап завершений, коли можна однозначно відповісти:

1. чи реальна аномалія 128×128;
2. де саме вона виникає;
3. чи впливає вона на best ROI-SSD;
4. чи впливає вона на best ROI-SSD-MobileNet;
5. чи безпечно запускати ROI-SSD-MobileNet у повному відеоконвеєрі.
