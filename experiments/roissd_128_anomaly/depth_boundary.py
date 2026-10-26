"""Stage 2c: validate the 128-pixel depth boundary without changing policy."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import torch
import yaml

from .forced_depth import BASELINE, evaluate, input_digest
from .run import DATA, HERE, ROOT, crop, csv_rows, load_model, prepare, sha, write_json


BOUNDARIES = ((129, 129), (130, 130), (131, 131))
RECTANGLES = (
    (128, 160), (160, 128), (132, 160), (160, 132), (136, 192),
    (192, 136), (140, 192), (192, 140), (144, 192), (192, 144),
)
CONTROLS = ((128, 128), (132, 132), (136, 136), (140, 140), (144, 144))
OUT = ROOT / 'final_experiments' / 'diagnostics_128'
FIELDS = (
    'map50', 'recall50', 'mean_raw_max_confidence', 'mean_best_anchor_iou',
    'fraction_gt_best_anchor_iou_ge_05', 'num_gt', 'num_detections',
    'N_loc_predictions', 'N_cls_predictions', 'N_anchors',
)


def check_control(result, reference):
    for key in FIELDS:
        assert abs(result[key] - float(reference[key])) <= 1e-6, (
            f'Stage-2b control mismatch: {key}: {result[key]} != {reference[key]}')
    for key in ('active_heads', 'feature_map_shapes', 'anchors_per_head'):
        assert result[key] == json.loads(reference[key]), f'Stage-2b structure changed: {key}'


def pair_deltas(results):
    low, high = (next(row for row in results if row['actual_L_R'] == depth) for depth in (4, 5))
    assert low['input_gt_sha256'] == high['input_gt_sha256']
    assert low['num_gt'] == high['num_gt']
    return {
        'delta_map50_L5_minus_L4': high['map50'] - low['map50'],
        'delta_recall50_L5_minus_L4': high['recall50'] - low['recall50'],
    }


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    settings = yaml.safe_load((HERE / 'config.yaml').read_text())
    lock = prepare(settings)
    entry, = [item for item in lock['models'] if item['model_family'] == 'ROI-SSD-MobileNet']
    previous = json.loads((OUT / 'forced_depth_metadata.json').read_text())
    stage2 = json.loads((BASELINE / 'run_metadata.json').read_text())
    assert previous['status'] == stage2['status'] == 'COMPLETE'
    for key in ('checkpoint', 'checkpoint_sha256', 'config_sha256'):
        assert entry[key] == previous[key], f'Stage-2b selection changed: {key}'
    assert lock['subset_sha256'] == previous['subset_sha256'] == stage2['subset_sha256']
    for key, old_key in (('seed', 'seed'), ('confidence', 'confidence_threshold'),
                         ('nms_iou', 'nms_iou'), ('max_detections', 'max_detections')):
        assert settings[key] == previous[old_key] == stage2[key], f'Settings changed: {key}'
    assert sha(HERE / 'forced_depth.py') == previous['probe_source_sha256']
    assert sha(BASELINE / 'run_metadata.json') == previous['stage2_metadata_sha256']
    source_hashes = {}
    for name, digest in stage2['source_sha256'].items():
        assert sha(ROOT / name) == digest, f'Stage-2 source changed: {name}'
        source_hashes[name] = digest
    rows = [json.loads(line) for line in (DATA / 'subset.jsonl').read_text().splitlines()]
    assert len(rows) == previous['num_samples'] == 256
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    images = []
    for row in rows:
        path = Path(lock['dataset_root']) / row['image']
        assert sha(path) == row['image_sha256'], f'Image changed: {path}'
        assert sha(Path(lock['dataset_root']) / row['label']) == row['label_sha256']
        image = cv2.imread(str(path))
        assert image is not None, f'Cannot read image: {path}'
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with (OUT / 'forced_depth_probe.csv').open(newline='', encoding='utf-8') as stream:
        controls = {(int(row['input_size']), int(row['actual_L_R'])): row
                    for row in csv.DictReader(stream)}
    assert len(controls) == 15 and all(row['status'] == 'PASS' for row in controls.values())
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        'checkpoint': entry['checkpoint'], 'checkpoint_sha256': entry['checkpoint_sha256'],
        'config_sha256': entry['config_sha256'], 'subset_sha256': lock['subset_sha256'],
        'seed': settings['seed'], 'confidence_threshold': settings['confidence'],
        'nms_iou': settings['nms_iou'], 'max_detections': settings['max_detections'],
        'device': str(device), 'torch': torch.__version__,
    }
    metadata = {
        **provenance, 'status': 'RUNNING', 'num_samples': len(rows),
        'shape_convention': 'height,width; no transpose, padding or size rounding',
        'boundary_shapes': BOUNDARIES, 'rectangular_shapes': RECTANGLES,
        'control_shapes': CONTROLS, 'production_policy_changed': False,
        'stage2b_metadata_sha256': sha(OUT / 'forced_depth_metadata.json'),
        'stage2b_csv_sha256': sha(OUT / 'forced_depth_probe.csv'),
        'source_sha256': {**source_hashes,
                          'experiments/roissd_128_anomaly/forced_depth.py': sha(HERE / 'forced_depth.py'),
                          'experiments/roissd_128_anomaly/depth_boundary.py': sha(Path(__file__))},
    }
    metadata_path = args.output / 'depth_boundary_metadata.json'
    write_json(metadata_path, metadata)
    print('Loading', entry['checkpoint'], 'on', device, flush=True)
    model = load_model(entry, settings, device)
    results = []
    checks = []
    groups = [('boundary', BOUNDARIES), ('rectangle', RECTANGLES), ('control', CONTROLS)]
    with (args.output / 'depth_boundary_inputs.jsonl').open('w', encoding='utf-8') as inputs:
        for group, shapes in groups:
            for height, width in shapes:
                samples = [crop(image, row, (height, width), settings['seed'])
                           for image, row in zip(images, rows)]
                digests = [input_digest(*sample) for sample in samples]
                shared = hashlib.sha256(''.join(digests).encode('ascii')).hexdigest()
                for row, digest in zip(rows, digests):
                    inputs.write(json.dumps({'input_h': height, 'input_w': width,
                                             'sample_id': row['sample_id'],
                                             'tensor_gt_sha256': digest}) + '\n')
                min_dim = min(height, width)
                current = model.get_max_feature_layer_by_roi_size(min_dim)
                candidate = 5 if 128 < min_dim <= 140 else current
                pair = []
                depths = (4, 5, 6) if (height, width) == (144, 144) else (4, 5)
                for depth in depths:
                    print(group, f'{height}x{width}', 'L_R', depth, flush=True)
                    result = {
                        'group': group, 'input_h': height, 'input_w': width,
                        'min_dim': min_dim, 'requested_L_R': depth,
                        'current_policy_L_R': current, 'candidate_policy_L_R': candidate,
                        'input_gt_sha256': shared, **provenance,
                        **evaluate(model, samples, lock['class_names'], device, depth),
                    }
                    if group == 'control':
                        reference = controls[height, depth]
                        assert shared == reference['input_gt_sha256'], 'Control preprocessing changed'
                        check_control(result, reference)
                        result['stage2b_control_matches'] = True
                    assert model.get_max_feature_layer_by_roi_size(min_dim) == current
                    assert 'get_max_feature_layer_by_roi_size' not in model.__dict__
                    pair.append(result)
                    print('  mAP50', result['map50'], 'Recall50', result['recall50'], flush=True)
                deltas = pair_deltas(pair)
                for result in pair:
                    result.update(deltas)
                checks.append({'group': group, 'input_h': height, 'input_w': width, **deltas})
                results.extend(pair)
                csv_rows(args.output / 'depth_boundary_validation.csv', results)
    assert len(results) == 37
    for name, digest in metadata['source_sha256'].items():
        assert sha(ROOT / name) == digest, f'Source changed during run: {name}'
    assert sha(ROOT / entry['checkpoint']) == entry['checkpoint_sha256']
    metadata.update(
        status='COMPLETE', num_configurations=len(results), paired_inputs_verified=True,
        stage2b_controls_verified=True, slot_checks_passed=True, weights_unchanged=True,
        all_boundary_L5_better=all(row['delta_map50_L5_minus_L4'] > 0
                                  and row['delta_recall50_L5_minus_L4'] > 0
                                  for row in checks if row['group'] == 'boundary'),
        rectangular_min_gt_128_L5_better=all(row['delta_map50_L5_minus_L4'] > 0
                                           and row['delta_recall50_L5_minus_L4'] > 0
                                           for row in checks if row['group'] == 'rectangle'
                                           and min(row['input_h'], row['input_w']) > 128),
        pair_comparisons=checks,
    )
    write_json(metadata_path, metadata)
    print('COMPLETE:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args())
