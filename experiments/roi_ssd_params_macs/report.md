# Порівняльний аналіз Params та MACs ROI-SSD VGG16 і ROI-SSD-MobileNet

## 1. Мета експерименту

Кількісно порівняти повну кількість параметрів та згорткову складність двох актуальних детекторів за однакових розмірів ROI й геометричної політики `until_1x1`. Навчання, датасети, mAP evaluation, latency та FPS не використовувалися.

## 2. Архітектури та конфігурації

`model/roissd.py:RoiSSD` використовує VGG16, learned L2 scale для conv4_3, dilated fc6/fc7, додаткові conv8–conv11 та звичайні Conv2d heads. `model/roissd_mobilenet.py:RoiSSDMobileNet` використовує частину MobileNetV3-Large, SE-блоки, нові channel adapters і depthwise/pointwise feature layers та heads. Початкові classifier/невикористані частини torchvision backbone не входять до детекторів.

Використано поточні YAML `config/imagenet-vid-roissd.yaml` та `config/imagenet-vid-roissd-mobilenet.yaml`, 31 вихідний клас (30 foreground + background). Архітектурні aspect ratios, channels та реально спожиті scales звірено зі збереженими final inference configs. На локацію припадає 4/6/6/6/4/4 anchors для шести можливих heads. Історичний VGG YAML уже містить terminal scale 1.0 від старого mutating constructor; порівнювалися сім реально спожитих scales, а не зайвий невикористаний хвіст списку.

**Checkpoint і pretrained weights не завантажувалися.** Factory torchvision тимчасово отримує `weights=None`; структура і detector forward не змінюються. Всі шари ініціалізовані до підрахунку; parameter identities не змінювалися при forward. Ваги випадкові, а MACs визначаються виконаними модулями й формами, не їх значеннями.

Структурний поділ компонентів: VGG backbone — усі збережені pretrained feature convolutions включно з conv5 усередині `conv5_3_fc`; нові dilated fc6/fc7 — additional feature layers. MobileNet backbone — `features_stage1` та збережений original block `features_stage2[0]`; додані 112→672 і 160→480 adapters та conv8–11 — additional layers. Усі depthwise/pointwise/BN всередині detection head належать відповідній head. VGG `scale_weight` із 512 параметрів виділено окремо. Належність встановлюється за реальними об’єктами модулів; shared parameters рахуються за унікальним identity.

## 3. Методика підрахунку Params/MACs

Params — сума `numel()` усіх унікальних параметрів повного детектора, незалежно від активної глибини. Окремо зафіксовані `requires_grad` та nontrainable parameters. BN running mean/variance — buffers, а не Params. Поточні BN flags залишають affine trainable; `eval()` не вимикає `requires_grad`.

**1 MAC — одна операція множення-накопичення. Основна метрика — лише щільні Conv2d MACs.** Для кожного реально виконаного Conv2d використано `B × Hout × Wout × Cout × (Cin/groups) × Kh × Kw`. Bias additions не включені. Depthwise та SE 1×1 convolutions враховані цією ж формулою. Це не повна кількість арифметичних операцій detector forward і не FLOPs; множник 2 не застосовувався.

Правило `dense_conv2d_MAC_v1`; THOP 2.0.19, PyTorch 2.12.0+cu126, torchvision 0.27.0+cu126, Python 3.13.5. CPU, float32, B=1, 4 threads, `model.eval()`, `torch.no_grad()`, synthetic tensors, seed=20261027. Git revision `f279f352c831f085684a9edbc78607d12a732b47`; hashes поточних dirty source/config files записані у `results/results.json`.

THOP counting hooks для non-convolution arithmetic занулено в основній метриці. Щоб не приховати підтримувані ним інші операції, окремо збережено штатний THOP підрахунок BN/pooling для 320×320 у `thop_supplementary_ops.csv`. Це змішаний operation count, який не додається до Conv2d MACs і не називається чистими MACs.

Штатний THOP для 320×320 (допоміжний підрахунок, **не чисті MACs**):

| Model | Conv2d MACs | Додаткові підтримувані non-conv operations | Штатний змішаний operation count |
|---|---:|---:|---:|
| VGG16 | 35,785,759,232 | 0 | 35,785,759,232 |
| MobileNet | 465,706,304 | 39,048,152 | 504,754,456 |

Нуль non-conv у VGG не означає відсутності такої арифметики: його функціональна L2-нормалізація/scale та інші перелічені операції цим профайлером не охоплені. Додаткова підтримувана арифметика MobileNet наведена для прозорості; жоден із двох підрахунків не оголошується точною кількістю всіх операцій detector.

## 4. Перевірка правильності профілювання

Чотири незалежні тести звичайної, grouped, depthwise та depthwise з multiplier=2 згорток підтвердили формулу MACs і THOP. Для кожного виконаного Conv2d в експерименті перевірено рівність аналітичного та THOP increment: 5216 перевірок, повний журнал `conv_layer_validation.csv`. Leaf-level сума збігається з підсумком; одна convolution не виконувалася повторно в одному forward.

Hooks перевіряють порядок feature block→classification head→localization head, відсутність пізніших блоків після зупинки та `N_loc == N_cls == N_anchors`. Для 320×320 згорткова частина штатного THOP також збігається з основною метрикою. Перевірено повторне профілювання після unsupported forward; залишкові THOP hooks і counters очищуються, тому не накопичуються між ROI.

THOP Params порівняно з прямими **active-path** Params. У VGG THOP не враховує standalone learned L2 scale (512 параметрів); ця різниця пояснена й перевірена. MobileNet active-path counts збігаються. Основна Params-таблиця завжди показує повну модель; inactive heads не вилучаються з неї. Значення active/full Params винесено в різні CSV columns.

## 5. Результати Params

| Component | ROI-SSD VGG16 | ROI-SSD-MobileNet |
|---|---:|---:|
| Backbone | 14 714 688 | 1 221 712 |
| Additional feature layers | 8 228 736 | 606 528 |
| Classification heads | 4 143 522 | 389 346 |
| Localization heads | 534 648 | 72 312 |
| Other trainable modules | 512 | 0 |
| Total Params | 27 622 106 | 2 289 898 |
| Trainable Params | 27 622 106 | 2 289 898 |
| Nontrainable Params | 0 | 0 |

`R_P = P_VGG16 / P_MobileNet = 12.062592`; `ΔP = (P_VGG16 − P_MobileNet) / P_VGG16 × 100% = 91.709908%`.

## 6. MACs для різних ROI

Усі запитані основні та rectangular ROI підтримуються обома моделями. Нижче наведені **Conv2d GMACs**, де 1 GMAC = 10⁹ MACs.

| ROI H×W | VGG16 GMACs | MobileNet GMACs | Ratio | Reduction, % | VGG/Mobile levels |
|---|---:|---:|---:|---:|---|
| 64×64 | 1.431139 | 0.019064 | 75.069 | 98.6679 | 4/3 |
| 96×96 | 3.221425 | 0.042955 | 74.996 | 98.6666 | 4/4 |
| 128×128 | 5.724555 | 0.074854 | 76.476 | 98.6924 | 4/4 |
| 140×140 | 7.048280 | 0.097250 | 72.476 | 98.6202 | 5/5 |
| 160×160 | 8.947672 | 0.117929 | 75.873 | 98.6820 | 5/5 |
| 192×192 | 12.881162 | 0.168057 | 76.647 | 98.6953 | 5/5 |
| 224×224 | 17.537365 | 0.229509 | 76.412 | 98.6913 | 5/5 |
| 256×256 | 22.901215 | 0.297866 | 76.884 | 98.6993 | 5/5 |
| 268×268 | 25.480135 | 0.340964 | 74.730 | 98.6618 | 6/6 |
| 300×300 | 31.871340 | 0.423180 | 75.314 | 98.6722 | 6/6 |
| 320×320 | 35.785759 | 0.465706 | 76.842 | 98.6986 | 6/6 |
| 64×128 | 2.862278 | 0.037766 | 75.790 | 98.6806 | 4/4 |
| 96×192 | 6.441293 | 0.084832 | 75.930 | 98.6830 | 4/5 |
| 128×256 | 11.449111 | 0.149263 | 76.704 | 98.6963 | 4/5 |
| 160×320 | 17.893526 | 0.234137 | 76.423 | 98.6915 | 5/6 |
| 256×128 | 11.449111 | 0.149263 | 76.704 | 98.6963 | 4/5 |
| 320×160 | 17.893526 | 0.234137 | 76.423 | 98.6915 | 5/6 |

Формули: `R_M = MAC_VGG16 / MAC_MobileNet`; `ΔM = (MAC_VGG16 − MAC_MobileNet) / MAC_VGG16 × 100%`. Дані для всіх додаткових та transition ROI збережено у CSV/JSON без екстраполяції за площею.

## 7. Вплив динамічної глибини

Перебрано кожний цілий square side від 1 до 320 через **реальний feature iterator** кожної моделі. Для виявлених переходів виконано повний detector forward у точках n−1/n/n+1, якщо n−1 існує. Розмір 640×640 не використовувався. VGG16 не підтримує 1–3 px через pooling; мінімальний перевірений валідний square side — 4. MobileNet підтримав side=1 геометрично; це не твердження про корисну точність на таких малих входах.

| Model | Side n | Попередня→нова глибина | GMACs n−1 | GMACs n | GMACs n+1 |
|---|---:|---|---:|---:|---:|
| VGG16 | 4 | —→1 | unsupported/— | 0.009520 | 0.009867 |
| VGG16 | 12 | 1→2 | 0.022641 | 0.067745 | 0.068710 |
| VGG16 | 28 | 2→3 | 0.192268 | 0.312873 | 0.315073 |
| VGG16 | 44 | 3→4 | 0.542256 | 0.740763 | 0.744198 |
| VGG16 | 140 | 4→5 | 6.395560 | 7.048280 | 7.059124 |
| VGG16 | 268 | 5→6 | 24.220584 | 25.480135 | 25.500859 |
| MobileNet | 1 | —→1 | unsupported/— | 0.000961 | 0.000961 |
| MobileNet | 17 | 1→2 | 0.001315 | 0.004109 | 0.004109 |
| MobileNet | 33 | 2→3 | 0.005020 | 0.010380 | 0.010380 |
| MobileNet | 65 | 3→4 | 0.019064 | 0.029034 | 0.029034 |
| MobileNet | 129 | 4→5 | 0.074854 | 0.093535 | 0.093535 |
| MobileNet | 257 | 5→6 | 0.297866 | 0.333781 | 0.333781 |

Політика однакова за принципом, але геометрія різна. VGG має unpadded 3×3 conv10/conv11 і pooling у conv5/fc. Якщо наступна карта має непозитивний H або W, forward зупиняється **до невалідної згортки**, навіть коли поточна карта не 1×1. Наприклад, VGG 128×128 завершується 2×2 з чотирма рівнями. У MobileNet extra depthwise convolutions мають stride=2/padding=1, і 128×128 завершується 1×1 також на четвертому рівні. Тому назву until_1x1 не можна трактувати як гарантію однакової фінальної карти чи однакової глибини для всіх H/W. Усі stop reasons та feature shapes записані у `feature_depth_by_shape.csv`.

На 64×64 VGG/MobileNet мають 4/3 рівні; на 140×140 — 5/5; на 268×268 — 6/6; на 160×320 — 5/6. Зміна глибини додає head/block, а округлення backbone feature maps також змінює MACs сусідніх розмірів; величина стрибка не є чистою ціною однієї нової head.

## 8. Складові архітектур і вплив форми

| ROI | Model | Backbone GMAC | Additional GMAC | Class GMAC | Loc GMAC | Heads share |
|---|---|---:|---:|---:|---:|---:|
| 128×128 | VGG16 | 5.011145 | 0.406979 | 0.271411 | 0.035021 | 5.35% |
| 128×128 | MobileNet | 0.056941 | 0.008841 | 0.007667 | 0.001405 | 12.12% |
| 256×256 | VGG16 | 20.044579 | 1.629618 | 1.086788 | 0.140231 | 5.36% |
| 256×256 | MobileNet | 0.226009 | 0.035530 | 0.030702 | 0.005625 | 12.20% |
| 320×320 | VGG16 | 31.319654 | 2.547679 | 1.699177 | 0.219249 | 5.36% |
| 320×320 | MobileNet | 0.352810 | 0.055857 | 0.048209 | 0.008831 | 12.25% |

Поділ backbone/adapters не є механічно однаковим за назвами: нові fc6/fc7 VGG та нові MobileNet adapters віднесено до additional за походженням, а не назвою контейнера. Інші trainable modules VGG мають 512 Params, але їхня функціональна арифметика L2/scale не належить до згорткових MACs.

За сталої площі 16 384 px VGG має однакові 5.724555 GMAC для 128×128 і 64×256; MobileNet має відповідно 0.074854 та 0.075087 GMAC, причому активує 4 та 5 рівнів. Для площі 25 600 px 160×160 і 100×256 також відрізняються через feature geometry та rounding. У перевірених транспонованих парах H×W/W×H MACs однакові. Отже, загальна залежність близька до площі, але точні значення не визначаються лише кількістю пікселів.

## 9. Обмеження експерименту

1. Основна оцінка охоплює всі виконані Conv2d, а не повну арифметику мережі. Не враховано bias adds, BN arithmetic, activations (ReLU/ReLU6/Hardswish/Hardsigmoid), pooling reductions/comparisons, residual adds, SE gating multiplication, VGG L2 normalization/scale multiplication, softmax, box decoding exp/div/add, anchor generation, clipping, comparisons/TopK, NMS, preprocessing, copies та memory access. Підтримувані штатним THOP non-conv operations подані окремо й не змішані з MACs.
2. MACs щільних згорток враховують padded kernel positions за стандартною формулою; sparsity, operator fusion, hardware utilization та datatype-specific acceleration не оцінено.
3. Випадкові ваги не використовуються для висновків про accuracy. Checkpoint/state та mAP не оцінювалися. MACs не доводять фактичного прискорення або економії енергії.
4. Повні Params та active-path Params — різні величини. BN buffers не входять до Params. Результат стосується 31 класу; зміна кількості класів змінить classification heads.

## 10. Висновки для конференційних тез

ROI-SSD-MobileNet має у 12.063 раза менше повних параметрів (91.710% reduction). Для 320×320 згорткові MACs знижуються з 35.785759 до 0.465706 GMAC (76.842 раза; 98.6986%). Для основних square ROI 64–320 ratio змінюється від 72.476 до 76.884; відносна економія не є точно сталою. Для ще менших transition probes відношення змінюється сильніше.

Це порівняння **двох цілісних архітектур**, включно з різними additional layers та heads, а не ізольована оцінка заміни VGG backbone на MobileNet.

## Готові результати для наукової публікації

| Показник | ROI-SSD VGG16 | ROI-SSD-MobileNet |
|---|---:|---:|
| Params, млн | 27.622106 | 2.289898 |
| Conv2d GMACs, 128×128 | 5.724555 | 0.074854 |
| Conv2d GMACs, 256×256 | 22.901215 | 0.297866 |
| Conv2d GMACs, 320×320 | 35.785759 | 0.465706 |
| Активні рівні, 320×320 | 6 | 6 |

Порівняльний аналіз актуальних реалізацій ROI-SSD на основі VGG16 та MobileNetV3 зафіксував зменшення кількості параметрів із 27.622 до 2.290 млн, що відповідає скороченню на 91.71%. Для входу 320×320 і 31 вихідного класу кількість множень-накопичень у фактично виконаних згортках зменшилася з 35.786 до 0.466 GMAC (76.84 раза). Обидві моделі використовували геометричний вибір глибини until_1x1 без відновлення legacy threshold policy.

Для сегментів різної форми зафіксовано відмінності кількості активних рівнів та згорткової складності, зумовлені геометрією feature maps і detection heads. Отриману економію слід відносити до цілісної архітектури ROI-SSD-MobileNet. Оцінка охоплює convolutional MACs і не включає іншу арифметику, preprocessing та NMS; вона не є вимірюванням фактичного прискорення інференсу.

Графіки: [MACs vs area PNG](figures/macs_vs_area.png), [SVG](figures/macs_vs_area.svg); [ratio PNG](figures/macs_ratio.png), [SVG](figures/macs_ratio.svg); [components PNG](figures/macs_components.png), [SVG](figures/macs_components.svg).

![MACs залежно від площі](figures/macs_vs_area.png)

![Відношення MACs](figures/macs_ratio.png)

![Декомпозиція MACs](figures/macs_components.png)

Відтворення з кореня репозиторію:

```powershell
.venv/Scripts/python.exe -m experiments.roi_ssd_params_macs.profile_models
.venv/Scripts/python.exe -m experiments.roi_ssd_params_macs.validate_profiler
.venv/Scripts/python.exe -m experiments.roi_ssd_params_macs.make_report
```
