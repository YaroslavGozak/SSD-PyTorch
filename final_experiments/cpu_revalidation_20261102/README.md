# Повторна CPU-валідація — 2026-11-02

Стан: підготовлено; довгі експерименти не запускалися. Почніть із [RUNBOOK.md](RUNBOOK.md).

Одна команда `--stage all` послідовно виконує full-frame accuracy та обидва
accuracy-loop для трьох моделей, окрему ROI-SSD/VGG16 confidence-діагностику,
усі підетапи Stage 03 та агрегацію.
`--stage accuracy` об'єднує full-frame → pad-loop → fixed-size-loop;
`--stage stage03` об'єднує confidence selection/freeze → padding → calibration → pipeline.
Тільки повний кадр для всіх трьох: `--stage accuracy --accuracy-modes full-frame`.
Full-frame accuracy охоплює всі 5064 validation frames при configured input
320×320 без ROI crop; Stage03 FF на pilot subset залишається окремим.

ROI-SSD/VGG16 при `conf=0.01,0.05,0.1,0.25,0.5` однією командою:
`--stage confidence-diagnostic`. Outputs — `confidence_diagnostics/roi_ssd_vgg16/`;
спільна таблиця — `summary.csv`. Додати до звичайного accuracy-запуску можна
через `--with-confidence-diagnostic`; `--stage all` уже включає її.
Ця validation-діагностика не змінює quality floor=0.01 або pilot-only вибір
operational thresholds.

Runner: `experiments/cpu_revalidation.py`. Експериментальна логіка залишається
в `tools.infer` і `experiments.cpu_pilot.run_pilot`.

Checkpoint-и використовуються безпосередньо з manifest. Quality conf_floor=0.01
для всіх трьох моделей. Окремі operational thresholds обираються на full-frame
pilot за максимумом micro F1@IoU=0.5, за рівності — вищим recall, потім нижчим
threshold, і фіксуються до решти Stage03/final benchmark. Selection ще не запускалася.
За уточненням користувача YOLO pad-loop тепер
зберігає прямокутний crop і доповнює кожну сторону незалежно до кратності 32.

`experiment_manifest.md` та `session.json` містять checkpoint/config/source/
dataset-manifest SHA-256 і CPU-параметри. `validation_checks/` містить лише
короткі технічні перевірки; їх не можна включати до експериментальних результатів.
