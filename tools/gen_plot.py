import csv
import os
import matplotlib.pyplot as plt

# ── Plot 1: mAP vs ROI padding (fixed_padding_results.csv) ────────────────────
_csv_path = os.path.join(os.path.dirname(__file__),
                         '..', 'trained_models', 'voc-roissd-sc-tr-roi-decrease',
                         'fixed_padding_results.csv')

_crop_px = []
_map_vals = []
_no_crop_map = None

with open(_csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw = row['crop_px'].strip() if row['crop_px'] else ''
        val = row['mAP'].strip() if row['mAP'] else ''
        if not raw or val == 'nan' or val == '':
            continue
        if raw == 'no_crop':
            _no_crop_map = float(val)
        else:
            _crop_px.append(int(raw))
            _map_vals.append(float(val))

fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(_crop_px, _map_vals, marker='o', linewidth=2,
         label='Fixed-padding ROI crop')

if _no_crop_map is not None:
    ax1.axhline(_no_crop_map, color='red', linestyle='--', linewidth=1.5,
                label=f'No crop (mAP = {_no_crop_map:.4f})')

ax1.set_xlabel('Padding (px)')
ax1.set_ylabel('mAP@0.5')
ax1.set_title('mAP vs ROI Padding – VOC RoI-SSD (fixed padding evaluation)')
ax1.grid(True)
ax1.legend()
fig1.tight_layout()

# ── Plot 2: mAP vs ROI area ratio ─────────────────────────────────────────────

roi_ratio = [0.11, 0.28, 0.75]

classic = [0.0797, 0.2037, 0.4214]
progressive = [0.5144, 0.5441, 0.5784]

plt.figure(figsize=(7,5))

plt.plot(roi_ratio, classic, marker='o', linewidth=2,
         label='Класичне тренування')

plt.plot(roi_ratio, progressive, marker='o', linewidth=2,
         label='Прогресивне багаторівневе тренування')

plt.xlabel("Відношення площі ROI до площі зображення (A_ROI / A_image)")
plt.ylabel("mAP@0.5")
plt.title("Точність детекції залежно від відношення площі ROI до площі зображення")

plt.grid(True)
plt.legend()

plt.show()