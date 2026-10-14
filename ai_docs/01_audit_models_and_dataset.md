# Етап 1. Аудит моделей, датасету та вибір фінальних checkpoint-ів

## Мета

Перед фінальними експериментами потрібно однозначно встановити:

1. які саме навчені моделі YOLO26n, ROI-SSD та ROI-SSD-MobileNet є фінальними/найкращими;
2. на яких класах вони навчені;
3. яка спеціально підготовлена вибірка ImageNet-VID є спільною для всіх моделей;
4. чи збігаються семантика та порядок класів без додаткового class mapping;
5. які checkpoint-и, конфігурації та dataset manifests мають використовувати всі наступні етапи.

У репозиторії може бути багато старих checkpoint-ів і конфігурацій. Не робити вибір лише за назвою файлу, датою або припущенням.

## Головне правило

**Не змінювати та не перенавчати моделі на цьому етапі.**

Спочатку провести аудит наявного коду, конфігурацій, checkpoint-ів, логів і збережених метрик. Якщо неможливо надійно визначити найкращу модель за старими логами, виконати повторну коротку оцінку кандидатів на одному й тому самому контрольному наборі.

## 1. Знайти всі кандидати

Просканувати репозиторій і знайти всі релевантні:

- checkpoint-и YOLO / YOLO26n;
- checkpoint-и ROI-SSD;
- checkpoint-и ROI-SSD-MobileNet;
- конфігурації моделей;
- label maps / class lists;
- конфігурації датасетів;
- training logs;
- validation/test metrics;
- старі benchmark scripts і результати, з яких можна встановити, який checkpoint використовувався.

Для кожного кандидата зберегти:

- architecture/model family;
- checkpoint path;
- config path;
- training run directory;
- training dataset;
- validation dataset;
- кількість класів, яку очікує модель;
- чи враховується background як окремий class index;
- список класів у точному порядку індексів;
- найкращі доступні validation metrics;
- дату/ідентифікатор training run, якщо вона надійно доступна;
- git commit/hash, якщо training run це зберіг.

Не робити висновок про кількість класів лише з числа `num_classes`. Для SSD-подібної реалізації окремо перевірити, чи входить background у це число.

## 2. Встановити спільний ImageNet-VID dataset

Знайти спеціально сформовану вибірку ImageNet-VID, яка створювалася для того, щоб YOLO та ROI-SSD могли оцінюватися на одному наборі класів без ручного mapping.

Потрібно встановити:

- dataset root;
- train/validation/test manifests;
- кількість відео;
- кількість кадрів;
- точний список класів;
- точні numeric class IDs у dataset annotations;
- чи збігаються ці IDs з YOLO;
- чи збігаються ці IDs з ROI-SSD / ROI-SSD-MobileNet;
- чи існує окремий background index тільки всередині ROI-SSD;
- чи є GT-об'єкти, клас яких не підтримується хоча б однією з моделей.

Якщо у репозиторії є декілька схожих підмножин ImageNet-VID, не вибирати одну за назвою. Встановити зв'язок із training configs/checkpoints і старими benchmark results.

Особливо перевірити, чи набір із **5064 кадрів**, використаний у контрольованому дослідженні формування сегментів, є цією ж спільною вибіркою та чи може він використовуватися для фінального end-to-end benchmark.

## 3. Перевірити class compatibility

Побудувати таблицю:

| Dataset class id | Dataset class name | YOLO class id | ROI-SSD class id | ROI-SSD-MobileNet class id | Status |
|---:|---|---:|---:|---:|---|

Допустимі статуси:

- `IDENTICAL`
- `BACKGROUND_OFFSET_ONLY`
- `MAPPING_REQUIRED`
- `MISSING_IN_MODEL`
- `UNKNOWN`

### Бажаний результат

Для всіх foreground classes:

- YOLO і dataset мають однакову семантику;
- ROI-SSD / ROI-SSD-MobileNet або мають той самий foreground index, або відрізняються лише на систематичний background offset, який уже є частиною реалізації моделі;
- не потрібен спеціально вигаданий mapping для фінального benchmark.

Якщо mapping все ж потрібен, **не приховувати це**. Зафіксувати його явно і пояснити причину.

## 4. Вибрати найкращий checkpoint кожної сім'ї

Потрібно вибрати по одному фінальному checkpoint:

- `YOLO26n`
- `ROI-SSD`
- `ROI-SSD-MobileNet`

### 4.1. Спочатку відфільтрувати несумісні моделі

Кандидат не може бути фінальним, якщо:

- навчений на іншому наборі класів;
- не може бути оцінений на спільній ImageNet-VID вибірці;
- checkpoint не завантажується;
- конфігурація/label map не відновлюється;
- training run явно незавершений або checkpoint є проміжним, якщо існує коректний final/best.

### 4.2. Правило ранжування

Якщо старі validation metrics отримані **на одному й тому самому validation split і з однаковою процедурою**, ранжувати:

1. основний критерій — `mAP@0.5`;
2. якщо різниця `mAP@0.5 < 0.005`, використовувати `mAP@0.5:0.95`, якщо він доступний;
3. наступний критерій — `Recall@0.5`.

Не вибирати модель за FPS замість точності: на цьому етапі обирається найкраща навчена конфігурація всередині тієї самої архітектурної сім'ї.

### 4.3. Якщо старі метрики не можна чесно порівнювати

Якщо candidate checkpoints оцінювалися на різних split/configs:

1. відібрати до 3 найбільш перспективних checkpoint-ів кожної сім'ї;
2. виконати повторну оцінку на одному canonical validation subset;
3. використати однакові:
   - кадри;
   - preprocessing;
   - confidence threshold;
   - NMS;
   - class set;
   - metric implementation;
4. вибрати найкращий checkpoint за правилом вище.

Не змішувати результати різних validation protocols.

## 5. Перевірити, що вибрані моделі реально працюють

Для кожного фінального checkpoint виконати smoke test:

- модель завантажується;
- 1 повний кадр обробляється без помилки;
- predictions декодуються;
- class IDs валідні;
- координати bbox знаходяться в очікуваній системі координат;
- batch=1 працює;
- inference на CPU працює;
- якщо Raspberry Pi environment відрізняється — зафіксувати потрібні версії пакетів.

Для ROI-SSD і ROI-SSD-MobileNet додатково перевірити декілька variable-size inputs, наприклад:

- 96×96;
- 128×128;
- 160×160;
- 160×192;

але **не робити на цьому етапі висновків про аномалію 128×128** — це завдання етапу 2.

## 6. Створити службовий manifest для всіх наступних етапів

Обов'язково створити в репозиторії файл:

`final_experiments/FINAL_EXPERIMENTS_MANIFEST.md`

Якщо директорії немає — створити.

Цей файл є **єдиним джерелом істини** для етапів 2–4. Наступні скрипти не повинні самостійно шукати `best.pt` по репозиторію.

### Обов'язкова структура manifest

```md
# Final experiments manifest

## Common dataset
- Dataset name:
- Dataset root:
- Train manifest:
- Validation manifest:
- Final evaluation manifest:
- Number of videos:
- Number of frames:
- Foreground class count:
- Exact class list:
- Annotation format:
- Notes:

## YOLO26n
- Selected run:
- Checkpoint: [relative link]
- Config: [relative link]
- Training dataset:
- Validation dataset:
- Foreground class count:
- Exact class list:
- Validation mAP@0.5:
- Validation mAP@0.5:0.95:
- Validation Recall@0.5:
- Selection reason:

## ROI-SSD
- Selected run:
- Checkpoint: [relative link]
- Config: [relative link]
- Training dataset:
- Validation dataset:
- Foreground class count:
- Background handling:
- Exact foreground class list:
- Validation mAP@0.5:
- Validation mAP@0.5:0.95:
- Validation Recall@0.5:
- Selection reason:

## ROI-SSD-MobileNet
- Selected run:
- Checkpoint: [relative link]
- Config: [relative link]
- Training dataset:
- Validation dataset:
- Foreground class count:
- Background handling:
- Exact foreground class list:
- Validation mAP@0.5:
- Validation mAP@0.5:0.95:
- Validation Recall@0.5:
- Selection reason:

## Class compatibility
- Status:
- Mapping required: yes/no
- Background offset:
- Full compatibility table: [relative link if stored separately]

## Software
- Repository commit:
- Python:
- PyTorch:
- Ultralytics:
- torchvision:
- OpenCV:
- Other relevant versions:

## Unresolved issues
- ...
```

Markdown links мають бути **відносними посиланнями на реальні файли в репозиторії**, де це можливо.

## 7. Додаткові outputs

Окрім MD manifest, бажано зберегти:

- `final_experiments/audit/model_candidates.csv`
- `final_experiments/audit/class_compatibility.csv`
- `final_experiments/audit/audit_report.md`

Головний обов'язковий файл — `FINAL_EXPERIMENTS_MANIFEST.md`.

## 8. STOP CONDITIONS

Не переходити до етапу 2, якщо:

- не встановлено однозначний checkpoint ROI-SSD;
- не встановлено однозначний checkpoint ROI-SSD-MobileNet;
- не встановлено checkpoint YOLO26n для фінального pipeline;
- невідомий точний список класів;
- не зрозуміло, який ImageNet-VID subset є спільним;
- GT містить класи, яких немає в одній із моделей;
- кандидат `best` вибраний лише за назвою файлу без метрик або повторної оцінки.

У разі проблеми записати її в `Unresolved issues` та зупинити наступні довгі експерименти.

## Критерій завершення етапу

Етап завершений, якщо існує `final_experiments/FINAL_EXPERIMENTS_MANIFEST.md`, у якому однозначно зафіксовані:

1. спільний ImageNet-VID dataset;
2. точний список foreground classes;
3. class compatibility;
4. один фінальний YOLO26n checkpoint;
5. один фінальний ROI-SSD checkpoint;
6. один фінальний ROI-SSD-MobileNet checkpoint;
7. реальні relative paths/links до checkpoint/config files;
8. метрики та причина вибору кожного checkpoint.
