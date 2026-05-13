import csv
import glob
import os
import matplotlib.pyplot as plt

BASE_DIR = 'H:\\Projects\\University\\SSD-PyTorch'

MODEL_LABELS = {
    # 'voc-roissd-full-frame': 'ROI-SSD (Standart training)',
    # 'voc-roissd-best': 'ROI-SSD',
    # 'voc-roissd-scheduled-training': 'Прогресивне навчання зі зменшеними регіонами',
    # 'voc-roissd-sc-tr-roi-decrease': 'ROI-SSD',
    # 'voc-roissd-0-pad-stage-2': 'Навчання з 0-піксельним відступом',
    # 'voc-roissd-0-pad': 'ROI-SSD (Progressive training)',
    # 'voc-roissd-small-pads': 'Навчання з малими відступами',
    # 'voc-roissd-smaller-pads': 'Навчання з ще меншими відступами',
    # 'voc-yolo11n': 'YOLOv11n',
    # 'voc-yolo26s': 'YOLOv26s',
    # 'voc-yolo26m': 'YOLOv26m',
    # 'imagenet-vid-roissd' : 'RoI-SSD (ImageNet-VID)',
    'imagenet-vid-yolo26n' : 'YOLO 26n (ImageNet-VID)',
}

def _load_voc_ssd_baseline():
    """Load Mean Average Precision from voc-ssd baseline"""
    baseline_path = os.path.join(
        BASE_DIR, 'trained_models', 
        'voc-ssd', 'ssd_results', 'mAp.txt'
    )
    
    if not os.path.exists(baseline_path):
        print(f"Warning: Baseline mAp file not found at {baseline_path}")
        return None
    
    with open(baseline_path, 'r') as f:
        for line in f:
            if 'Mean Average Precision' in line:
                try:
                    return float(line.split(':')[1].strip())
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse mAp value from line: {line}")
                    return None
    print(f"Warning: mAp value not found in {baseline_path}")
    return None

def _load_padding_curve(csv_path):
    model_name = os.path.basename(os.path.dirname(csv_path))
    crop_px = []
    map_vals = []
    no_crop_map = None

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row['crop_px'].strip() if row.get('crop_px') else ''
            val = row['mAP'].strip() if row.get('mAP') else ''
            if not raw or val in ('nan', ''):
                continue
            if raw == 'no_crop':
                no_crop_map = float(val)
            else:
                crop_px.append(int(raw))
                map_vals.append(float(val))

    pairs = sorted(zip(crop_px, map_vals), key=lambda p: p[0])
    if not pairs:
        return None

    xs, ys = zip(*pairs)
    return {
        'model': model_name,
        'name': MODEL_LABELS.get(model_name, model_name),
        'x': list(xs),
        'y': list(ys),
        'no_crop_map': no_crop_map,
    }

def _interp_piecewise(xs, ys, x):
    if x < xs[0] or x > xs[-1]:
        return None
    if x == xs[-1]:
        return ys[-1]

    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= x <= x1:
            y0, y1 = ys[i], ys[i + 1]
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None

def _find_crossings(ref_curve, other_curve):
    ref_x, ref_y = ref_curve['x'], ref_curve['y']
    oth_x, oth_y = other_curve['x'], other_curve['y']

    lo = max(ref_x[0], oth_x[0])
    hi = min(ref_x[-1], oth_x[-1])
    if lo > hi:
        return []

    knots = sorted({x for x in (ref_x + oth_x) if lo <= x <= hi})
    if len(knots) < 2:
        return []

    crossings = []
    for i in range(len(knots) - 1):
        x0, x1 = knots[i], knots[i + 1]
        r0 = _interp_piecewise(ref_x, ref_y, x0)
        r1 = _interp_piecewise(ref_x, ref_y, x1)
        o0 = _interp_piecewise(oth_x, oth_y, x0)
        o1 = _interp_piecewise(oth_x, oth_y, x1)
        if None in (r0, r1, o0, o1):
            continue

        d0 = r0 - o0
        d1 = r1 - o1

        if d0 == 0:
            crossings.append((x0, r0))
            continue
        if d0 * d1 < 0:
            x_cross = x0 + (0 - d0) * (x1 - x0) / (d1 - d0)
            y_cross = _interp_piecewise(ref_x, ref_y, x_cross)
            if y_cross is not None:
                crossings.append((x_cross, y_cross))

    return crossings


base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models')
pattern = os.path.join(base_dir, '*', 'fixed_padding_results.csv')
csv_paths = sorted(glob.glob(pattern))

if not csv_paths:
    raise FileNotFoundError(f'No files found for pattern: {pattern}')

csv_paths = [
    p for p in csv_paths
    if os.path.basename(os.path.dirname(p)) in MODEL_LABELS
]

curves = []
for csv_path in csv_paths:
    curve = _load_padding_curve(csv_path)
    if curve is not None:
        curves.append(curve)

if not curves:
    raise RuntimeError('No valid padding curves found in fixed_padding_results.csv files')

ref_name = 'voc-roissd-sc-tr-roi-decrease'
ref_curve = next((c for c in curves if c['model'] == ref_name), curves[0])

voc_ssd_baseline = _load_voc_ssd_baseline()

fig, ax = plt.subplots(figsize=(11, 6))

for curve in curves:
    ax.plot(curve['x'], curve['y'], marker='o', linewidth=2, label=curve['name'])

if voc_ssd_baseline is not None:
    ax.axhline(
        voc_ssd_baseline,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label=f"Classic SSD = {voc_ssd_baseline:.3f} mAp",
    )

if ref_curve['no_crop_map'] is not None:
    ax.axhline(
        ref_curve['no_crop_map'],
        color='black',
        linestyle='--',
        linewidth=1.2,
        label=f"{ref_curve['model']} no_crop = {ref_curve['no_crop_map']:.4f}",
    )

for curve in curves:
    if curve['model'] == ref_curve['model']:
        continue
    crossings = _find_crossings(ref_curve, curve)
    for x_cross, y_cross in crossings:
        ax.scatter([x_cross], [y_cross], color='black', s=35, zorder=5)
        ax.annotate(
            f"x={x_cross:.1f}",
            (x_cross, y_cross),
            textcoords='offset points',
            xytext=(6, 6),
            fontsize=8,
            color='black',
        )

ax.set_xlabel('Padding (pixels)')
ax.set_ylabel('mAP@0.5')
ax.set_title('Dependence of Accuracy on Object Padding - VOC RoI-SSD Models')
ax.grid(True, alpha=0.35)
ax.legend(loc='best', fontsize=9)
fig.tight_layout()

plt.show()