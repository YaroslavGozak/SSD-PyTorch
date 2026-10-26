"""Stage 2b: paired depth interventions on the locked stage-2 ROI subset."""
import argparse
import ast
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

from .instrumentation import ForwardObserver, assert_slots, geometry, matching
from .run import DATA, HERE, ROOT, crop, csv_rows, load_model, prepare, sha, write_json


SIZES = (128, 132, 136, 140, 144)
BASELINE = ROOT / 'final_experiments' / '02_diagnostics_128'


@contextmanager
def selected_depth(model, depth):
    if depth is None:
        yield
        return
    if not 1 <= depth <= len(model.cls_heads):
        raise ValueError(f'Depth {depth} is outside the existing detection pyramid')
    with patch.object(model, 'get_max_feature_layer_by_roi_size', return_value=depth):
        yield


def input_digest(tensor, gt, labels):
    return hashlib.sha256(
        tensor.numpy().tobytes() + gt.numpy().tobytes() + labels.numpy().tobytes()
    ).hexdigest()


def structure(observer, depth):
    heads = sorted(observer.heads)
    assert heads == list(range(depth)), (heads, depth)
    loc = sum(h['loc_slots'] for h in observer.heads.values())
    cls = sum(h['cls_slots'] for h in observer.heads.values())
    assert_slots(loc, cls, len(observer.anchors))
    return {
        'actual_L_R': len(heads),
        'active_heads': heads,
        'feature_map_shapes': [observer.heads[h]['cls_output_shape'][:1]
                               + [observer.model.cls_heads[h].depthwise.in_channels]
                               + observer.feature_shapes[h] for h in heads],
        'anchors_per_head': [observer.heads[h]['cls_slots'] for h in heads],
        'N_loc_predictions': loc,
        'N_cls_predictions': cls,
        'N_anchors': len(observer.anchors),
        'slot_counts_equal': True,
    }


def spatially_impossible(error):
    message = str(error).lower()
    return ('kernel size can\'t be greater than actual input size' in message
            or 'output size is too small' in message)


def evaluate(model, samples, names, device, depth):
    predictions, truths, difficult = [], [], []
    scores, raw_maxima, best_ious = [], [], []
    layout = None
    with selected_depth(model, depth), ForwardObserver(model) as observer, torch.inference_mode():
        for index, (tensor, gt, labels) in enumerate(samples):
            observer.reset()
            _, detections = model(tensor.to(device), None)
            assert len(detections) == 1
            det = detections[0]
            current_layout = structure(observer, model.get_max_feature_layer_by_roi_size(tensor.shape[-1]))
            if layout is None:
                layout = current_layout
                geometry(observer.anchors, observer.reference, *tensor.shape[-2:])
            assert layout == current_layout, 'Feature/anchor structure changed across samples'
            assert torch.isfinite(det['boxes']).all() and torch.isfinite(det['scores']).all()
            assert ((det['labels'] >= 1) & (det['labels'] <= len(names))).all()
            assert ((det['boxes'] >= 0) & (det['boxes'] <= 1)).all()
            assert (det['boxes'][:, 2:] >= det['boxes'][:, :2]).all()
            scores.extend(det['scores'].cpu().tolist())
            raw_maxima.append(max(h['foreground_max_confidence'] for h in observer.heads.values()))
            best, _, _ = matching(observer.anchors, gt.to(device), .5)
            best_ious.extend(best)
            pred = {name: [] for name in names}
            truth = {name: [] for name in names}
            for box, label, score in zip(det['boxes'].cpu().tolist(),
                                         det['labels'].cpu().tolist(), det['scores'].cpu().tolist()):
                pred[names[label - 1]].append([*box, score])
            for box, label in zip(gt.tolist(), labels.tolist()):
                truth[names[label - 1]].append(box)
            predictions.append(pred)
            truths.append(truth)
            difficult.append({name: [0] * len(boxes) for name, boxes in truth.items()})
            if (index + 1) % 64 == 0:
                print(f'  samples {index + 1}/{len(samples)}', flush=True)

    source = ast.parse((ROOT / 'tools' / 'infer.py').read_text())
    functions = [node for node in source.body
                 if isinstance(node, ast.FunctionDef) and node.name in ('get_iou', 'compute_map')]
    assert len(functions) == 2
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'tools/infer.py', 'exec'), namespace)
    map50, _, recall50, _ = namespace['compute_map'](
        predictions, truths, iou_threshold=.5, method='area', difficult=difficult)
    assert best_ious, 'No GT remained in the control subset'
    return {
        **layout, 'status': 'PASS', 'num_samples': len(samples), 'num_gt': len(best_ious),
        'map50': map50, 'recall50': recall50,
        'mean_confidence': float(np.mean(scores)) if scores else None,
        'max_confidence': max(scores) if scores else None,
        'mean_raw_max_confidence': float(np.mean(raw_maxima)),
        'num_detections': len(scores),
        'mean_best_anchor_iou': float(np.mean(best_ious)),
        'fraction_gt_best_anchor_iou_ge_05': float(np.mean(np.asarray(best_ious) >= .5)),
    }


def baseline_rows(filename):
    with (BASELINE / filename).open(newline='', encoding='utf-8') as stream:
        return {int(row['requested_h']): row for row in csv.DictReader(stream)
                if row['model_family'] == 'ROI-SSD-MobileNet'}


def check_baseline(result, metrics, anchors):
    for field, reference, key in (
        ('map50', metrics, 'map50'), ('recall50', metrics, 'recall50'),
        ('mean_raw_max_confidence', metrics, 'max_confidence_mean'),
        ('mean_best_anchor_iou', anchors, 'max_anchor_iou_mean'),
        ('fraction_gt_best_anchor_iou_ge_05', anchors, 'fraction_gt_iou_ge_05'),
        ('num_gt', metrics, 'num_gt'),
    ):
        assert abs(result[field] - float(reference[key])) <= 1e-6, (
            f'Stage-2 baseline mismatch: {field}: {result[field]} != {reference[key]}')


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    settings = yaml.safe_load((HERE / 'config.yaml').read_text())
    lock = prepare(settings)
    entry, = [item for item in lock['models'] if item['model_family'] == 'ROI-SSD-MobileNet']
    assert entry['checkpoint'] in (ROOT / settings['stage1_manifest']).read_text(encoding='utf-8')
    previous = json.loads((BASELINE / 'run_metadata.json').read_text())
    assert previous['status'] == 'COMPLETE'
    for key in ('seed', 'confidence', 'nms_iou', 'max_detections'):
        assert settings[key] == previous[key], f'Stage-2 setting changed: {key}'
    assert lock['subset_sha256'] == previous['subset_sha256']
    for filename in ('model/roissd_mobilenet.py', 'dataset/transforms/fixed_size_test_transform.py',
                     'experiments/roissd_128_anomaly/run.py',
                     'experiments/roissd_128_anomaly/instrumentation.py', 'tools/infer.py'):
        assert sha(ROOT / filename) == previous['source_sha256'][filename], f'Stage-2 source changed: {filename}'
    rows = [json.loads(line) for line in (DATA / 'subset.jsonl').read_text().splitlines()]
    assert len(rows) == lock['num_samples'] == previous['num_samples'] == 256
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
    with (BASELINE / 'preprocessing.csv').open(newline='', encoding='utf-8') as stream:
        hashes = {(int(row['requested_h']), int(row['sample_id'])): row['tensor_gt_sha256']
                  for row in csv.DictReader(stream) if row['model_family'] == 'ROI-SSD-MobileNet'}
    metrics = baseline_rows('size_sweep_metrics.csv')
    anchors = baseline_rows('anchor_matching.csv')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        'checkpoint': entry['checkpoint'], 'checkpoint_sha256': entry['checkpoint_sha256'],
        'subset_sha256': lock['subset_sha256'], 'config_sha256': entry['config_sha256'],
        'seed': settings['seed'], 'confidence_threshold': settings['confidence'],
        'nms_iou': settings['nms_iou'], 'max_detections': settings['max_detections'],
        'device': str(device), 'torch': torch.__version__,
    }
    metadata = {
        **provenance, 'status': 'RUNNING', 'num_samples': len(rows),
        'stage2_metadata_sha256': sha(BASELINE / 'run_metadata.json'),
        'probe_source_sha256': sha(Path(__file__)),
        'model_source_sha256': sha(ROOT / 'model' / 'roissd_mobilenet.py'),
        'baseline_tolerance': 1e-6, 'production_policy_changed': False,
    }
    write_json(out / 'forced_depth_metadata.json', metadata)
    print('Loading', entry['checkpoint'], 'on', device, flush=True)
    model = load_model(entry, settings, device)
    results = []
    for size in SIZES:
        samples = [crop(image, row, (size, size), settings['seed']) for image, row in zip(images, rows)]
        digests = [input_digest(*sample) for sample in samples]
        for row, digest in zip(rows, digests):
            assert digest == hashes[size, row['sample_id']], 'Preprocessing differs from stage 2'
        shared_digest = hashlib.sha256(''.join(digests).encode('ascii')).hexdigest()
        current = model.get_max_feature_layer_by_roi_size(size)
        baseline_result = None
        for mode, depth in (('current', None), ('forced_L_R_minus_1', current - 1),
                            ('forced_L_R_plus_1', current + 1)):
            requested = current if depth is None else depth
            identity = {
                'input_size': size, 'mode': mode, 'current_L_R': current,
                'requested_L_R': requested, **provenance,
                'input_gt_sha256': shared_digest, 'preprocessing_matches_stage2': True,
            }
            print(size, mode, 'L_R', requested, flush=True)
            # Only spatial convolution failures are feasibility results, never CUDA/OOM or logic errors.
            try:
                with selected_depth(model, depth), ForwardObserver(model) as observer, torch.inference_mode():
                    model(samples[0][0].to(device), None)
                    structure(observer, requested)
            except RuntimeError as error:
                if depth is None or not spatially_impossible(error):
                    raise
                result = {**identity, 'status': 'PHYSICALLY_IMPOSSIBLE', 'error': str(error),
                          'num_samples': 0}
                print(result['status'], result['error'], flush=True)
            else:
                result = {**identity, **evaluate(model, samples, lock['class_names'], device, depth)}
                if depth is None:
                    check_baseline(result, metrics[size], anchors[size])
                    result['stage2_baseline_matches'] = True
                    baseline_result = result
                assert baseline_result is not None
                result['delta_map50_vs_current'] = result['map50'] - baseline_result['map50']
                result['delta_recall50_vs_current'] = result['recall50'] - baseline_result['recall50']
                print('  mAP50', result['map50'], 'Recall50', result['recall50'], flush=True)
            assert model.get_max_feature_layer_by_roi_size(size) == current, 'Depth override leaked'
            assert 'get_max_feature_layer_by_roi_size' not in model.__dict__
            results.append(result)
            csv_rows(out / 'forced_depth_probe.csv', results)
    assert len(results) == 15
    assert sha(ROOT / 'model' / 'roissd_mobilenet.py') == metadata['model_source_sha256']
    assert sha(ROOT / entry['checkpoint']) == entry['checkpoint_sha256']
    metadata.update(status='COMPLETE', num_configurations=len(results),
                    paired_inputs_verified=True, stage2_baselines_verified=True,
                    slot_checks_passed=True, weights_unchanged=True)
    write_json(out / 'forced_depth_metadata.json', metadata)
    print('COMPLETE:', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'final_experiments' / 'diagnostics_128')
    run(parser.parse_args())
