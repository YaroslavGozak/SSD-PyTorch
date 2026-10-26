"""Stage 2e: actual feature geometry, immutable weights, per-GT paired evidence."""
import argparse
import ast
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import cv2
import numpy as np
import torch
import yaml

from .all_thresholds import OUT as STAGE2D, timed_pair
from .depth_boundary import check_control
from .forced_depth import input_digest, selected_depth, spatially_impossible, structure
from .instrumentation import ForwardObserver, geometry, matching
from .run import DATA, HERE, ROOT, crop, csv_rows, load_model, prepare, sha, write_json
from .threshold_evidence import paired_rows, read_csv


OUT = ROOT / 'final_experiments' / 'feature_map_depth_policy'
SQUARES = (32, 64, 88, 96, 104, 124, 127, 128, 129, 136, 140, 144,
           240, 252, 255, 256, 257, 264, 268, 280)
RECTANGLES = ((96, 144), (144, 96), (127, 159), (159, 127),
              (128, 160), (160, 128), (128, 192), (192, 128),
              (140, 175), (175, 140), (256, 320), (320, 256),
              (256, 384), (384, 256), (268, 335), (335, 268))
POLICIES = ('legacy', 'max_feasible_depth', 'until_1x1', 'best_adjacent_reference')


def shapes():
    return [(size, size) for size in SQUARES] + list(RECTANGLES)


def legacy_depth(minimum):
    for depth, boundary in enumerate((32, 64, 96, 140, 268), 1):
        if minimum <= boundary:
            return depth
    return 6


def reference_depth(lower, upper, tolerance=.005):
    da, dr = (float(upper[key]) - float(lower[key]) for key in ('map50', 'recall50'))
    if da >= -tolerance and dr >= -tolerance and (da > tolerance or dr > tolerance):
        return int(upper['forced_L_R'])
    return int(lower['forced_L_R'])


def verify_sources(metadata):
    """Allow only the user-approved policy-method drift, never architecture drift."""
    source_hashes = metadata['source_sha256']
    name = 'model/roissd_mobilenet.py'
    expected = source_hashes[name]
    for path, digest in source_hashes.items():
        if path != name:
            assert sha(ROOT / path) == digest, f'Unapproved source drift: {path}'
    original = subprocess.check_output(['git', 'show', 'HEAD:model/roissd_mobilenet.py'], cwd=ROOT)
    baseline_variants = (original, original.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    assert expected in {hashlib.sha256(data).hexdigest() for data in baseline_variants}, (
        'HEAD (LF or Windows CRLF checkout) is not the Stage-2d source')
    current = (ROOT / name).read_bytes()
    old_tree, new_tree = ast.parse(original), ast.parse(current)
    old_cls, = [node for node in old_tree.body if isinstance(node, ast.ClassDef)
                and node.name == 'RoiSSDMobileNet']
    new_cls, = [node for node in new_tree.body if isinstance(node, ast.ClassDef)
                and node.name == 'RoiSSDMobileNet']
    old_method, = [node for node in old_cls.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'get_max_feature_layer_by_roi_size']
    new_index, = [index for index, node in enumerate(new_cls.body)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == 'get_max_feature_layer_by_roi_size']
    new_cls.body[new_index] = old_method
    assert ast.dump(old_tree) == ast.dump(new_tree), 'Model changed outside the approved depth method'
    return {
        'stage2d_model_sha256': expected, 'current_model_sha256': sha(ROOT / name),
        'baseline_git_blob_sha256': hashlib.sha256(original).hexdigest(),
        'baseline_checkout_line_endings': 'CRLF' if b'\r\n' in current else 'LF',
        'current_model_git_diff': subprocess.check_output(
            ['git', '--no-pager', 'diff', '--', name], cwd=ROOT, text=True),
        'approved_drift': 'User-approved threshold-only source drift; all policies scoped, '
                          'legacy explicitly uses 32/64/96/140/268',
    }


def metric_functions():
    tree = ast.parse((ROOT / 'tools' / 'infer.py').read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('compute_map', 'get_iou')]
    assert len(nodes) == 2
    namespace = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'tools/infer.py', 'exec'), namespace)
    return namespace['compute_map'], namespace['get_iou']


def probe_geometry(model, tensor, device):
    layouts, failures = {}, []
    for depth in range(1, len(model.cls_heads) + 1):
        try:
            with selected_depth(model, depth), ForwardObserver(model) as observer, torch.inference_mode():
                model(tensor.to(device), None)
                layouts[depth] = structure(observer, depth)
        except RuntimeError as error:
            if not spatially_impossible(error):
                raise
            failures.append({'depth': depth, 'status': 'INFEASIBLE', 'error': str(error)})
            break
    assert layouts, 'No feasible detection head'
    maximum = max(layouts)
    first = next((depth for depth, layout in layouts.items()
                  if layout['feature_map_shapes'][-1][-2:] == [1, 1]), maximum)
    return layouts, failures, maximum, first


def object_geometry(gt, labels, sample_id, height, width):
    result = []
    for index, (box, label) in enumerate(zip(gt.tolist(), labels.tolist())):
        wr, hr = box[2] - box[0], box[3] - box[1]
        assert 0 < wr <= 1 and 0 < hr <= 1
        result.append({
            'input_h': height, 'input_w': width, 'sample_id': sample_id, 'gt_index': index,
            'class_id': label, 'crop_h': height, 'crop_w': width,
            'gt_w': wr * width, 'gt_h': hr * height, 'gt_area': wr * hr * width * height,
            'crop_area': width * height, 'gt_width_ratio': wr, 'gt_height_ratio': hr,
            'gt_area_ratio': wr * hr, 'gt_max_side_ratio': max(wr, hr),
            'gt_min_side_ratio': min(wr, hr),
        })
    return result


def localization(det, gt, labels, get_iou):
    """Per-GT same-class localization and project-compatible one-to-one TP."""
    records = [dict(success50=0, best_predicted_iou=0., max_class_confidence=0.,
                    best_localized_confidence=0.) for _ in labels]
    boxes, classes, scores = (det[key].cpu().tolist() for key in ('boxes', 'labels', 'scores'))
    targets = gt.tolist()
    gt_classes = labels.tolist()
    for index, (target, label) in enumerate(zip(targets, gt_classes)):
        candidates = [(get_iou(box, target), score)
                      for box, cid, score in zip(boxes, classes, scores) if cid == label]
        if candidates:
            iou, confidence = max(candidates, key=lambda pair: pair[0])
            records[index].update(best_predicted_iou=iou,
                                  max_class_confidence=max(score for _, score in candidates),
                                  best_localized_confidence=confidence)
    matched = set()
    for index in sorted(range(len(scores)), key=lambda index: -scores[index]):
        candidates = [(get_iou(boxes[index], target), target_index)
                      for target_index, (target, label) in enumerate(zip(targets, gt_classes))
                      if label == classes[index]]
        if candidates:
            # The project evaluator resolves equal IoU in original GT order.
            iou, target_index = max(candidates, key=lambda pair: pair[0])
            if iou >= .5 and target_index not in matched:
                matched.add(target_index)
                records[target_index]['success50'] = 1
    return records


def evaluate_objects(model, samples, names, device, depth, sample_rows, output_log):
    compute_map, get_iou = metric_functions()
    predictions, truths, difficult = [], [], []
    objects, raw_maxima, best_ious, scores = [], [], [], []
    layout = None
    with selected_depth(model, depth), ForwardObserver(model) as observer, torch.inference_mode():
        for index, ((tensor, gt, labels), source) in enumerate(zip(samples, sample_rows)):
            observer.reset()
            _, detections = model(tensor.to(device), None)
            assert len(detections) == 1
            det = detections[0]
            current = structure(observer, depth)
            if layout is None:
                layout = current
                geometry(observer.anchors, observer.reference, *tensor.shape[-2:])
            assert current == layout
            assert torch.isfinite(det['boxes']).all() and torch.isfinite(det['scores']).all()
            assert ((det['boxes'] >= 0) & (det['boxes'] <= 1)).all()
            assert (det['boxes'][:, 2:] >= det['boxes'][:, :2]).all()
            assert ((det['labels'] >= 1) & (det['labels'] <= len(names))).all()
            raw = max(head['foreground_max_confidence'] for head in observer.heads.values())
            raw_maxima.append(raw)
            best, _, _ = matching(observer.anchors, gt.to(device))
            best_ious.extend(best)
            scores.extend(det['scores'].cpu().tolist())
            per_gt = localization(det, gt, labels, get_iou)
            assert len(per_gt) == len(best)
            for gt_index, (record, iou) in enumerate(zip(per_gt, best)):
                objects.append({
                    'sample_id': source['sample_id'], 'gt_index': gt_index,
                    'class_id': int(labels[gt_index]),
                    'best_anchor_iou': iou, 'crop_raw_max_confidence': raw, **record,
                })
            pred, truth = ({name: [] for name in names} for _ in range(2))
            for box, label, score in zip(det['boxes'].cpu().tolist(),
                                         det['labels'].cpu().tolist(), det['scores'].cpu().tolist()):
                pred[names[label - 1]].append([*box, score])
            for box, label in zip(gt.tolist(), labels.tolist()):
                truth[names[label - 1]].append(box)
            predictions.append(pred)
            truths.append(truth)
            difficult.append({name: [0] * len(boxes) for name, boxes in truth.items()})
            h, w = tensor.shape[-2:]
            output_log.write(json.dumps({
                'input_h': h, 'input_w': w, 'forced_L_R': depth,
                'sample_id': source['sample_id'], 'gt': truth, 'predictions': pred,
                'input_gt_sha256': input_digest(tensor, gt, labels),
            }) + '\n')
            if (index + 1) % 64 == 0:
                print(' samples', index + 1, '/', len(samples), flush=True)
    map50, _, recall50, class_recalls = compute_map(
        predictions, truths, iou_threshold=.5, method='area', difficult=difficult)
    for label, name in enumerate(names, 1):
        selected = [record for record in objects if record['class_id'] == label]
        if selected:
            assert abs(np.mean([record['success50'] for record in selected]) - class_recalls[name]) < 1e-12
    return {
        **layout, 'status': 'PASS', 'num_samples': len(samples), 'num_gt': len(best_ious),
        'map50': map50, 'recall50': recall50, 'num_detections': len(scores),
        'mean_raw_max_confidence': float(np.mean(raw_maxima)),
        'mean_best_anchor_iou': float(np.mean(best_ious)),
        'fraction_gt_best_anchor_iou_ge_05': float(np.mean(np.asarray(best_ious) >= .5)),
        'micro_gt_recall50': float(np.mean([obj['success50'] for obj in objects])),
    }, objects


def object_pairs(kind, lower, upper, geometry_rows, by_depth, layouts, pair_id):
    low = {(row['sample_id'], row['gt_index']): row for row in by_depth[lower]}
    high = {(row['sample_id'], row['gt_index']): row for row in by_depth[upper]}
    assert low.keys() == high.keys()
    added = [shape[-2:] for shape in layouts[upper]['feature_map_shapes'][lower:]]
    for geo in geometry_rows:
        key = geo['sample_id'], geo['gt_index']
        lo, hi = low[key], high[key]
        record = {
            **geo, 'pair_kind': kind, 'pair_id': pair_id, 'lower_depth': lower, 'upper_depth': upper,
            'added_feature_shapes': added, 'upper_adds_1x1': [1, 1] in added,
            'all_added_heads_1x1': bool(added) and all(shape == [1, 1] for shape in added),
            'num_added_heads': upper - lower,
        }
        for name in ('success50', 'best_predicted_iou', 'max_class_confidence',
                     'best_localized_confidence', 'best_anchor_iou', 'crop_raw_max_confidence'):
            record.update({name + '_lower': lo[name], name + '_upper': hi[name],
                           'delta_' + name: hi[name] - lo[name]})
        yield record


def run(args):
    if args.warmup < 3 or args.repeats < 10:
        raise ValueError('Need >=3 warm-up and >=10 timed forwards')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    settings = yaml.safe_load((HERE / 'config.yaml').read_text())
    lock = prepare(settings)
    prior = json.loads((STAGE2D / 'all_thresholds_metadata.json').read_text())
    assert prior['status'] == 'COMPLETE'
    approved = verify_sources(prior)
    entry, = [item for item in lock['models'] if item['model_family'] == 'ROI-SSD-MobileNet']
    for key in ('checkpoint', 'checkpoint_sha256', 'config_sha256'):
        assert entry[key] == prior[key]
    assert lock['subset_sha256'] == prior['subset_sha256']
    for key, prior_key in (('seed', 'seed'), ('confidence', 'confidence_threshold'),
                           ('nms_iou', 'nms_iou'), ('max_detections', 'max_detections')):
        assert settings[key] == prior[prior_key]
    for filename, key in (('all_thresholds_results.csv', 'results_sha256'),
                           ('latency_summary.csv', 'latency_sha256'),
                           ('input_hashes.jsonl', 'input_hashes_sha256')):
        assert sha(STAGE2D / filename) == prior[key]
    previous_pairs = {(h, w): pair for (_, h, w), pair in paired_rows(
        read_csv(STAGE2D / 'all_thresholds_results.csv')).items()}
    assert set(shapes()) <= previous_pairs.keys()
    hashes = {}
    for line in (STAGE2D / 'input_hashes.jsonl').read_text().splitlines():
        record = json.loads(line)
        if (record['input_h'], record['input_w']) in shapes():
            hashes[record['input_h'], record['input_w'], record['sample_id']] = record['tensor_gt_sha256']
    sample_rows = [json.loads(line) for line in (DATA / 'subset.jsonl').read_text().splitlines()]
    assert len(sample_rows) == prior['num_samples'] == 256
    images = []
    for row in sample_rows:
        path = Path(lock['dataset_root']) / row['image']
        assert sha(path) == row['image_sha256']
        assert sha(Path(lock['dataset_root']) / row['label']) == row['label_sha256']
        image = cv2.imread(str(path))
        assert image is not None
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_names = set(prior['source_sha256']) | {
        'experiments/roissd_128_anomaly/feature_map_policy.py',
        'experiments/roissd_128_anomaly/threshold_evidence.py',
    }
    metadata = {
        'status': 'RUNNING', 'started_at': datetime.now(timezone.utc).isoformat(),
        'checkpoint': entry['checkpoint'], 'checkpoint_sha256': entry['checkpoint_sha256'],
        'config_sha256': entry['config_sha256'], 'subset_sha256': lock['subset_sha256'],
        'seed': settings['seed'], 'confidence_threshold': settings['confidence'],
        'nms_iou': settings['nms_iou'], 'max_detections': settings['max_detections'],
        'source_sha256': {name: sha(ROOT / name) for name in source_names}, **approved,
        'stage2d_metadata_sha256': sha(STAGE2D / 'all_thresholds_metadata.json'),
        'stage2d_results_sha256': prior['results_sha256'],
        'shapes': shapes(), 'policies': POLICIES, 'num_samples': 256,
        'device': str(device), 'torch': torch.__version__, 'opencv': cv2.__version__,
        'platform': platform.platform(), 'cpu_name': args.cpu_name,
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'threads': args.threads, 'interop_threads': 1,
        'warmup': args.warmup, 'repeats': args.repeats, 'batch_size': 1,
        'reference_rule': 'Stage-2d descriptive 0.5pp tolerance; upper if AP or recall '
                          'gain >0.5pp and neither loss >0.5pp; otherwise lower',
        'reference_is_oracle': True,
        'geometry_selection': 'Probe actual forced forwards 1..6, stop only on spatial '
                              'convolution failure; until_1x1 selects first active 1x1 map',
        'latency_scope': 'CPU forward incl NMS and slot guard; cached geometry selection; '
                         'excludes one-time shape probing, crop, transfer, construction',
        'geometry_cache_scope': 'diagnostic H/W->depth only, not a production implementation',
        'production_policy_changed': False, 'weights_changed': False,
    }
    write_json(out / 'metadata.json', metadata)
    model = load_model(entry, settings, device)
    cpu_model = model if device.type == 'cpu' else load_model(entry, settings, torch.device('cpu'))
    per_shape, all_geometry, comparisons, latency_rows, unique_results, probes = [], [], [], [], [], []
    controls = 0
    with ExitStack() as stack:
        predictions_log = stack.enter_context((out / 'predictions.jsonl').open('w', encoding='utf-8'))
        objects_log = stack.enter_context((out / 'per_depth_object_metrics.jsonl').open('w', encoding='utf-8'))
        for shape_index, (h, w) in enumerate(shapes()):
            print(f'Shape {shape_index + 1}/{len(shapes())}: {h}x{w}', flush=True)
            samples = [crop(image, row, (h, w), settings['seed']) for image, row in zip(images, sample_rows)]
            digests = [input_digest(*sample) for sample in samples]
            for row, digest in zip(sample_rows, digests):
                assert digest == hashes[h, w, row['sample_id']], 'Stage-2d input changed'
            digest = hashlib.sha256(''.join(digests).encode('ascii')).hexdigest()
            geo = [record for (_, gt, labels), row in zip(samples, sample_rows)
                   for record in object_geometry(gt, labels, row['sample_id'], h, w)]
            all_geometry.extend(geo)
            layouts, failures, maximum, until = probe_geometry(model, samples[0][0], device)
            probes.append({'input_h': h, 'input_w': w, 'layouts': layouts, 'failures': failures})
            old_low, old_high = previous_pairs[h, w]
            lower, upper = int(old_low['forced_L_R']), int(old_high['forced_L_R'])
            selections = dict(zip(POLICIES, (legacy_depth(min(h, w)), maximum, until,
                                            reference_depth(old_low, old_high))))
            depths = sorted(set(selections.values()) | {lower, upper})
            metrics, object_metrics = {}, {}
            for depth in depths:
                assert depth in layouts, f'Infeasible selected depth {depth}'
                print(' depth', depth, flush=True)
                result, objects = evaluate_objects(model, samples, lock['class_names'],
                                                  device, depth, sample_rows, predictions_log)
                assert result['num_gt'] == len(geo)
                for reference in (old_low, old_high):
                    if depth == int(reference['forced_L_R']):
                        assert reference['input_gt_sha256'] == digest
                        check_control(result, reference)
                        controls += 1
                result.update(input_h=h, input_w=w, min_dim=min(h, w), forced_L_R=depth,
                              input_gt_sha256=digest)
                metrics[depth], object_metrics[depth] = result, objects
                unique_results.append(result)
                for obj in objects:
                    objects_log.write(json.dumps({'input_h': h, 'input_w': w, 'forced_L_R': depth, **obj}) + '\n')
                print(' mAP/Recall', result['map50'], result['recall50'], flush=True)
            timing = timed_pair(cpu_model, samples, depths, torch.device('cpu'),
                                args.warmup, args.repeats, shape_index % 2)
            by_timing = {row['forced_L_R']: row for row in timing}
            for item in timing:
                latency_rows.append({'input_h': h, 'input_w': w, **item,
                                     'cpu_threads': args.threads, 'input_gt_sha256': digest})
            for policy, depth in selections.items():
                next_layout = layouts.get(depth + 1)
                record = {
                    'policy': policy, 'selected_L_R': depth, **metrics[depth],
                    'spatial_feature_shapes': [shape[-2:] for shape in layouts[depth]['feature_map_shapes']],
                    'last_active_spatial_shape': layouts[depth]['feature_map_shapes'][-1][-2:],
                    'next_potential_feature_shape': (next_layout['feature_map_shapes'][-1][-2:]
                                                      if next_layout else None),
                    'next_head_feasible': depth + 1 in layouts,
                    'next_head_status': ('FEASIBLE' if next_layout else
                                         'PYRAMID_EXHAUSTED' if depth == len(model.cls_heads) else 'INFEASIBLE'),
                    'cpu_median_latency_ms': by_timing[depth]['median_latency_ms'],
                    'cpu_p95_latency_ms': by_timing[depth]['p95_latency_ms'],
                    'first_1x1_depth': until, 'max_feasible_depth': maximum,
                    'reference_depth': selections['best_adjacent_reference'],
                }
                per_shape.append(record)
            candidates = [('stage2d_adjacent', lower, upper)]
            if until < maximum:
                candidates.append(('until_vs_max', until, maximum))
            if selections['legacy'] < until:
                candidates.append(('legacy_vs_until', selections['legacy'], until))
            for kind, low, high in candidates:
                comparisons.extend(object_pairs(kind, low, high, geo, object_metrics, layouts,
                                                f'{kind}:{h}x{w}:L{low}-L{high}'))
            for name, records in (
                ('per_shape_results.csv', per_shape), ('per_object_crop_geometry.csv', all_geometry),
                ('per_object_pair_comparisons.csv', comparisons), ('latency_summary.csv', latency_rows),
                ('per_depth_results.csv', unique_results),
            ):
                csv_rows(out / name, records)
            write_json(out / 'geometry_probes.json', probes)
            assert 'get_max_feature_layer_by_roi_size' not in model.__dict__
            for name, source_hash in metadata['source_sha256'].items():
                assert sha(ROOT / name) == source_hash, f'Source changed during run: {name}'
            predictions_log.flush()
            objects_log.flush()
    assert len(per_shape) == 4 * len(shapes())
    assert controls == 2 * len(shapes())
    assert sha(ROOT / entry['checkpoint']) == entry['checkpoint_sha256']
    assert sha(STAGE2D / 'all_thresholds_metadata.json') == metadata['stage2d_metadata_sha256']
    metadata.update(
        status='COMPLETE', completed_at=datetime.now(timezone.utc).isoformat(),
        num_policy_configurations=len(per_shape), num_unique_depth_configurations=len(unique_results),
        num_geometry_objects=len(all_geometry), num_object_pair_comparisons=len(comparisons),
        stage2d_controls_checked=controls, exact_inputs_verified=True, slot_checks_passed=True,
        source_integrity_verified=True, weights_unchanged=True,
        output_sha256={path.name: sha(path) for path in out.iterdir()
                       if path.suffix in ('.csv', '.json', '.jsonl') and path.name != 'metadata.json'},
    )
    write_json(out / 'metadata.json', metadata)
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--cpu-name', default='Intel Core i7-11700K @ 3.60GHz')
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args())
