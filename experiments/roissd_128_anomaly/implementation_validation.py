"""Stage 2f validation for runtime-selected ROI-SSD-MobileNet feature depth."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import cv2
import numpy as np
import torch
import yaml

from .feature_map_policy import (
    STAGE2D, localization, metric_functions,
)
from .forced_depth import input_digest, structure
from .instrumentation import ForwardObserver
from .run import DATA, HERE, ROOT, crop, load_model, prepare, sha, write_json

STAGE2E = ROOT / 'final_experiments' / 'feature_map_depth_policy'
OUTPUT = ROOT / 'final_experiments' / 'feature_map_depth_policy_implementation'
POLICIES = ('until_1x1',)
REPRESENTATIVE_SHAPES = (
    (88, 88), (96, 96), (104, 104), (124, 124), (128, 128),
    (128, 160), (160, 128), (129, 129), (140, 140), (144, 144),
    (140, 175), (175, 140), (240, 240), (256, 256), (256, 320),
    (320, 256), (257, 257), (268, 268), (280, 280),
)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, separators=(',', ':'))
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def state_dict_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def check_close(actual, expected, label, tolerance=1e-6):
    difference = abs(float(actual) - float(expected))
    assert difference <= tolerance, (
        f'{label} changed from Stage 2e by {difference:.9g}: '
        f'new={actual}, reference={expected}')


def assert_predictions_close(actual, expected, label, tolerance=1e-6):
    assert actual.keys() == expected.keys(), f'Prediction classes changed: {label}'
    for class_name in actual:
        actual_boxes = sorted(tuple(map(float, box)) for box in actual[class_name])
        expected_boxes = sorted(tuple(map(float, box)) for box in expected[class_name])
        assert len(actual_boxes) == len(expected_boxes), (
            f'Prediction count changed for {label}, class {class_name}')
        for actual_box, expected_box in zip(actual_boxes, expected_boxes):
            if len(actual_box) != len(expected_box):
                raise AssertionError(f'Prediction shape changed for {label}, class {class_name}')
            difference = max(
                (abs(a - b) for a, b in zip(actual_box, expected_box)), default=0.0)
            assert difference <= tolerance, (
                f'Prediction values changed for {label}, class {class_name}: '
                f'max_abs_delta={difference:.9g}')


def load_reference_predictions(required_keys):
    predictions = {}
    with (STAGE2E / 'predictions.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = (int(row['input_h']), int(row['input_w']),
                   int(row['forced_L_R']), int(row['sample_id']))
            if key in required_keys:
                predictions[key] = row['predictions']
    missing = required_keys - predictions.keys()
    assert not missing, f'Missing Stage 2e prediction references: {len(missing)}'
    return predictions


def aggregate_policy(model, samples, sample_rows, class_names, device, shape,
                     reference_predictions):
    compute_map, get_iou = metric_functions()
    predictions, truths, difficult = [], [], []
    raw_confidences, best_ious, object_records = [], [], []
    expected_layout = None
    with ForwardObserver(model) as observer, torch.inference_mode():
        for sample_index, ((tensor, gt, labels), source) in enumerate(zip(samples, sample_rows)):
            observer.reset()
            _, detections = model(tensor.to(device), None)
            assert len(detections) == 1
            det = detections[0]
            trace = model.get_runtime_log()
            assert trace is not None, 'Runtime trace was not populated'
            depth = trace['selected_L_R']
            layout = structure(observer, depth)
            if expected_layout is None:
                expected_layout = layout
                assert trace['active_heads'] == list(range(depth))
                assert trace['feature_shapes'] == [item[-2:] for item in layout['feature_map_shapes']]
            assert layout == expected_layout, 'Feature structure changed within one input shape'
            assert trace['N_loc_predictions'] == trace['N_cls_predictions'] == trace['N_anchors']
            assert trace['N_anchors'] == layout['N_anchors']
            assert trace['anchors_per_head'] == layout['anchors_per_head']
            assert torch.isfinite(det['boxes']).all() and torch.isfinite(det['scores']).all()
            assert ((det['labels'] >= 1) & (det['labels'] <= len(class_names))).all()
            assert ((det['boxes'] >= 0) & (det['boxes'] <= 1)).all()
            raw = max(head['foreground_max_confidence'] for head in observer.heads.values())
            raw_confidences.append(raw)
            anchor_ious, _, _ = __import__(
                'experiments.roissd_128_anomaly.instrumentation',
                fromlist=['matching']).matching(observer.anchors, gt.to(device), .5)
            best_ious.extend(anchor_ious)
            per_object = localization(det, gt, labels, get_iou)
            object_records.extend({
                'input_h': shape[0], 'input_w': shape[1],
                'sample_id': source['sample_id'], 'gt_index': index,
                'success50': int(obj['success50']),
            } for index, obj in enumerate(per_object))

            pred = {name: [] for name in class_names}
            truth = {name: [] for name in class_names}
            for box, label, score in zip(det['boxes'].cpu().tolist(),
                                         det['labels'].cpu().tolist(),
                                         det['scores'].cpu().tolist()):
                pred[class_names[label - 1]].append([*box, score])
            reference_key = (shape[0], shape[1], depth, source['sample_id'])
            assert_predictions_close(
                pred, reference_predictions[reference_key],
                f'until_1x1 {shape[0]}x{shape[1]} sample {source["sample_id"]}')
            for box, label in zip(gt.tolist(), labels.tolist()):
                truth[class_names[label - 1]].append(box)
            predictions.append(pred)
            truths.append(truth)
            difficult.append({name: [0] * len(boxes) for name, boxes in truth.items()})
            if (sample_index + 1) % 64 == 0:
                print(f'  {shape[0]}x{shape[1]}: {sample_index + 1}/256 samples', flush=True)

    map50, _, recall50, _ = compute_map(
        predictions, truths, iou_threshold=.5, method='area', difficult=difficult)
    result = {
        **expected_layout,
        'actual_L_R': expected_layout['actual_L_R'],
        'input_h': shape[0], 'input_w': shape[1], 'min_dim': min(shape),
        'num_samples': len(samples), 'num_gt': len(best_ious),
        'map50': map50, 'recall50': recall50,
        'mean_raw_max_confidence': float(np.mean(raw_confidences)),
        'mean_best_anchor_iou': float(np.mean(best_ious)),
        'fraction_gt_best_anchor_iou_ge_05': float(np.mean(np.asarray(best_ious) >= .5)),
        'micro_gt_recall50': float(np.mean([row['success50'] for row in object_records])),
        'prediction_rows_checked': len(samples),
    }
    return result, object_records


def add_execution_hooks(model):
    names = ('features_stage1', 'conv_to_672', 'features_stage2',
             'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2')
    called = []
    handles = []
    events = []
    for name in names:
        def record_block(module, inputs, output, key=name):
            called.append(key)
            events.append(key)
        handles.append(getattr(model, name).register_forward_hook(record_block))
    heads = {'cls': [], 'loc': []}
    for kind, modules in (('cls', model.cls_heads), ('loc', model.bbox_reg_heads)):
        for index, module in enumerate(modules):
            def record_head(module, inputs, output, head_kind=kind, head_index=index):
                heads[head_kind].append(head_index)
                events.append(f'{head_kind}_{head_index}')
            handles.append(module.register_forward_hook(record_head))
    return called, heads, events, handles


def expected_block_calls(depth):
    names = ['features_stage1', 'conv_to_672']
    pyramid = ('features_stage2', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2')
    names.extend(pyramid[:depth - 1])
    return names


def expected_execution_events(depth):
    events = ['features_stage1', 'conv_to_672', 'cls_0', 'loc_0']
    pyramid = ('features_stage2', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2')
    for index, block in enumerate(pyramid[:depth - 1], start=1):
        events.extend((block, f'cls_{index}', f'loc_{index}'))
    return events


def time_policy(model, tensor, policy, warmup, repeats):
    assert policy == 'until_1x1', f'Unsupported runtime policy: {policy}'
    called, heads, events, handles = add_execution_hooks(model)
    times = []
    try:
        with torch.inference_mode():
            for iteration in range(warmup + repeats):
                called.clear()
                events.clear()
                heads['cls'].clear()
                heads['loc'].clear()
                started = time.perf_counter_ns()
                _, detections = model(tensor, None)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                trace = model.get_runtime_log()
                assert trace is not None
                depth = trace['selected_L_R']
                expected_heads = list(range(depth))
                assert called == expected_block_calls(depth), (
                    f'Unexpected executed blocks at depth {depth}: {called}')
                assert heads['cls'] == expected_heads and heads['loc'] == expected_heads
                assert events == expected_execution_events(depth), (
                    f'Feature/head execution was not sequential: {events}')
                assert trace['N_loc_predictions'] == trace['N_cls_predictions'] == trace['N_anchors']
                assert len(detections) == 1
                if iteration >= warmup:
                    times.append(elapsed)
    finally:
        for handle in handles:
            handle.remove()
    trace = model.get_runtime_log()
    return {
        'policy': policy, 'selected_L_R': trace['selected_L_R'],
        'active_heads': trace['active_heads'], 'feature_shapes': trace['feature_shapes'],
        'anchors_per_head': trace['anchors_per_head'], 'total_anchors': trace['N_anchors'],
        'median_latency_ms': float(statistics.median(times)),
        'p95_latency_ms': float(np.percentile(times, 95)),
        'warmup_forwards': warmup, 'timed_forwards': repeats,
        'executed_blocks_per_forward': expected_block_calls(trace['selected_L_R']),
    }


def run(args):
    if args.warmup < 3 or args.repeats < 10:
        raise ValueError('Require at least 3 warm-up and 10 timed CPU forwards')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    settings = yaml.safe_load((HERE / 'config.yaml').read_text(encoding='utf-8'))
    lock = prepare(settings)
    stage2e_meta = json.loads((STAGE2E / 'metadata.json').read_text(encoding='utf-8'))
    assert stage2e_meta['status'] == 'COMPLETE'
    assert stage2e_meta['num_samples'] == 256
    assert stage2e_meta['checkpoint_sha256'] == next(
        entry['checkpoint_sha256'] for entry in lock['models']
        if entry['model_family'] == 'ROI-SSD-MobileNet')
    assert stage2e_meta['subset_sha256'] == lock['subset_sha256']
    for filename, digest in stage2e_meta['output_sha256'].items():
        assert sha(STAGE2E / filename) == digest, f'Stage 2e artifact changed: {filename}'
    stage2e_rows = read_csv(STAGE2E / 'per_shape_results.csv')
    references = {(int(row['input_h']), int(row['input_w']), row['policy']): row
                  for row in stage2e_rows}
    assert len(references) == 4 * 36
    subset_rows = [json.loads(line) for line in (DATA / 'subset.jsonl').read_text(
        encoding='utf-8').splitlines()]
    assert len(subset_rows) == 256
    required_prediction_keys = {
        (h, w, int(references[(h, w, policy)]['selected_L_R']), sample['sample_id'])
        for h, w in stage2e_meta['shapes']
        for policy in POLICIES
        for sample in subset_rows
    }
    reference_predictions = load_reference_predictions(required_prediction_keys)
    stage2d_hashes = {}
    for line in (STAGE2D / 'input_hashes.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        stage2d_hashes[(row['input_h'], row['input_w'], row['sample_id'])] = row['tensor_gt_sha256']
    images = []
    for row in subset_rows:
        image_path = Path(lock['dataset_root']) / row['image']
        assert sha(image_path) == row['image_sha256']
        image = cv2.imread(str(image_path))
        assert image is not None
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior_latency_path = output / 'legacy_vs_until1x1_latency.csv'
    prior_latency = read_csv(prior_latency_path) if prior_latency_path.exists() else []
    prior_legacy_latency = {
        (int(row['input_h']), int(row['input_w'])): row
        for row in prior_latency if row['policy'] == 'legacy'
    }
    entry, = [item for item in lock['models'] if item['model_family'] == 'ROI-SSD-MobileNet']
    metadata = {
        'status': 'RUNNING', 'started_at': datetime.now(timezone.utc).isoformat(),
        'checkpoint': entry['checkpoint'], 'checkpoint_sha256': entry['checkpoint_sha256'],
        'subset_sha256': lock['subset_sha256'], 'config_sha256': entry['config_sha256'],
        'stage2e_metadata_sha256': sha(STAGE2E / 'metadata.json'),
        'stage2d_metadata_sha256': sha(STAGE2D / 'all_thresholds_metadata.json'),
        'stage2d_input_hashes_sha256': sha(STAGE2D / 'input_hashes.jsonl'),
        'stage2e_output_sha256': stage2e_meta['output_sha256'],
        'num_samples': len(subset_rows), 'num_shapes': 36,
        'shapes': stage2e_meta['shapes'], 'runtime_policy': 'until_1x1',
        'legacy_comparison': 'Stage 2e saved legacy accuracy; prior Stage 2f legacy CPU latency',
        'device': str(device), 'torch': torch.__version__, 'opencv': cv2.__version__,
        'platform': platform.platform(), 'cpu_name': args.cpu_name,
        'cpu_threads': args.threads, 'interop_threads': 1,
        'cpu_latency_warmup': args.warmup, 'cpu_latency_repeats': args.repeats,
        'batch_size': 1, 'confidence_threshold': stage2e_meta['confidence_threshold'],
        'nms_iou': stage2e_meta['nms_iou'], 'max_detections': stage2e_meta['max_detections'],
        'accuracy_comparison_tolerance': 1e-6,
        'latency_scope': 'CPU model forward including detection postprocessing; excludes crop/transfer/model construction; lightweight block/head hooks validate early exit.',
        'legacy_runtime_supported': False, 'weights_changed': False,
    }
    model = load_model(entry, settings, device)
    model.record_execution_trace = True
    state_digest_before = state_dict_digest(model)
    cpu_model = (model if device.type == 'cpu'
                 else load_model(entry, settings, torch.device('cpu')))
    cpu_model.record_execution_trace = True
    accuracy_rows, selected_rows = [], []

    for shape_index, shape in enumerate(stage2e_meta['shapes']):
        h, w = shape
        print(f'Accuracy {shape_index + 1}/36: {h}x{w}', flush=True)
        samples = [crop(image, row, (h, w), settings['seed'])
                   for image, row in zip(images, subset_rows)]
        sample_digests = [
            input_digest(tensor, gt, labels) for tensor, gt, labels in samples
        ]
        for source, digest in zip(subset_rows, sample_digests):
            assert digest == stage2d_hashes[(h, w, source['sample_id'])], (
                f'Stage 2d exact input/GT identity changed for {h}x{w}, '
                f'sample {source["sample_id"]}')
        combined_digest = hashlib.sha256(''.join(sample_digests).encode('ascii')).hexdigest()
        reference_digest = references[(h, w, 'legacy')]['input_gt_sha256']
        assert combined_digest == reference_digest, f'Stage 2e tensor/GT digest changed at {h}x{w}'
        legacy_reference = references[(h, w, 'legacy')]
        accuracy_rows.append({
            **legacy_reference, 'policy': 'legacy',
            'accuracy_source': 'Stage 2e saved legacy measurement',
        })
        policy = 'until_1x1'
        result, _ = aggregate_policy(
            model, samples, subset_rows, lock['class_names'], device, (h, w),
            reference_predictions)
        baseline = references[(h, w, policy)]
        assert result['actual_L_R'] == int(baseline['selected_L_R']), (
            f'{policy} selected a different depth for {h}x{w}: '
            f'{result["actual_L_R"]} != {baseline["selected_L_R"]}')
        for field in ('map50', 'recall50', 'mean_raw_max_confidence',
                      'mean_best_anchor_iou', 'fraction_gt_best_anchor_iou_ge_05'):
            check_close(result[field], baseline[field], f'{policy} {h}x{w} {field}')
        for field in ('N_loc_predictions', 'N_cls_predictions', 'N_anchors'):
            assert result[field] == int(baseline[field])
        accuracy_rows.append({
            'policy': policy, **result, 'input_gt_sha256': combined_digest,
            'stage2e_map50': float(baseline['map50']),
            'stage2e_recall50': float(baseline['recall50']),
            'map50_delta_vs_stage2e': result['map50'] - float(baseline['map50']),
            'recall50_delta_vs_stage2e': result['recall50'] - float(baseline['recall50']),
            'stage2e_metrics_match_1e_6': True,
            'accuracy_source': 'Current sequential until_1x1 forward',
        })
        trace = model.get_runtime_log()
        selected_rows.append({
            'policy': policy, 'input_h': h, 'input_w': w, 'min_dim': min(h, w),
            'selected_L_R': result['actual_L_R'], 'active_heads': trace['active_heads'],
            'feature_shapes': trace['feature_shapes'],
            'final_feature_shape': trace['final_feature_shape'],
            'next_feature_shape': trace['next_feature_shape'],
            'next_head_feasible': trace['next_head_feasible'],
            'next_head_status': trace['next_head_status'],
            'anchors_per_head': trace['anchors_per_head'],
            'total_anchors': trace['N_anchors'],
            'N_loc_predictions': trace['N_loc_predictions'],
            'N_cls_predictions': trace['N_cls_predictions'],
            'slot_counts_equal': trace['N_loc_predictions'] == trace['N_cls_predictions'] == trace['N_anchors'],
            'num_samples': 256, 'num_gt': result['num_gt'],
            'input_gt_sha256': combined_digest,
        })
        write_csv(output / 'legacy_vs_until1x1_accuracy.csv', accuracy_rows)
        write_csv(output / 'selected_depths.csv', selected_rows)

    summaries = []
    for policy in ('legacy', 'until_1x1'):
        rows = [row for row in accuracy_rows if row['policy'] == policy]
        summaries.append({
            'policy': policy, 'num_shapes': len(rows),
            'macro_map50': float(np.mean([float(row['map50']) for row in rows])),
            'macro_recall50': float(np.mean([float(row['recall50']) for row in rows])),
            'macro_mean_raw_max_confidence': float(np.mean(
                [float(row['mean_raw_max_confidence']) for row in rows])),
            'macro_mean_best_anchor_iou': float(np.mean(
                [float(row['mean_best_anchor_iou']) for row in rows])),
            'macro_gt_coverage_iou_ge_05': float(np.mean(
                [float(row['fraction_gt_best_anchor_iou_ge_05']) for row in rows])),
            'selected_depth_counts': {
                str(depth): sum(int(row['actual_L_R']) == depth for row in rows)
                for depth in range(1, 7)
            },
            'stage2e_exact_shape_parity': all(
                row.get('stage2e_metrics_match_1e_6', policy == 'legacy')
                for row in rows),
            'measurement_source': (
                'Stage 2e frozen reference' if policy == 'legacy'
                else 'Current sequential until_1x1 forward'),
        })

    latency_rows = []
    for shape_index, (h, w) in enumerate(REPRESENTATIVE_SHAPES):
        tensor = crop(images[0], subset_rows[0], (h, w), settings['seed'])[0]
        print(f'CPU latency {shape_index + 1}/{len(REPRESENTATIVE_SHAPES)}: {h}x{w}', flush=True)
        timed = time_policy(cpu_model, tensor, 'until_1x1', args.warmup, args.repeats)
        latency_rows.append({
            'input_h': h, 'input_w': w, 'min_dim': min(h, w), **timed,
            'measurement_source': 'Current sequential until_1x1 implementation',
            'device': 'cpu', 'cpu_name': args.cpu_name, 'cpu_threads': args.threads,
            'batch_size': 1, 'latency_scope': metadata['latency_scope'],
        })
        legacy_latency = prior_legacy_latency.get((h, w))
        if legacy_latency:
            latency_rows.append({
                **legacy_latency,
                'measurement_source': 'Prior Stage 2f legacy implementation; historical reference',
            })
    assert state_dict_digest(model) == state_digest_before, 'Policy switching changed trained tensors'
    metadata.update(
        status='COMPLETE', completed_at=datetime.now(timezone.utc).isoformat(),
        device=device.type, state_dict_sha256_before=state_digest_before,
        state_dict_sha256_after=state_dict_digest(model),
        stage2e_shape_policy_pairs_checked=len(selected_rows),
        exact_stage2d_inputs_verified=True,
        stage2e_accuracy_parity_within_1e_6=True,
        stage2e_predictions_parity_within_1e_6=True,
        prediction_rows_checked=len(required_prediction_keys),
        slot_checks_passed=all(row['slot_counts_equal'] for row in selected_rows),
        early_exit_execution_checked=True,
        representative_cpu_shapes=len(REPRESENTATIVE_SHAPES),
        all_timed_forwards_block_head_and_order_checked=True,
        source_sha256={
            name: sha(ROOT / name) for name in (
                'model/roissd_mobilenet.py',
                'experiments/roissd_128_anomaly/implementation_validation.py',
                'experiments/roissd_128_anomaly/instrumentation.py',
                'experiments/roissd_128_anomaly/run.py',
                'experiments/roissd_128_anomaly/feature_map_policy.py',
                'tools/infer.py',
            )
        },
    )
    write_csv(output / 'policy_summary.csv', summaries)
    write_csv(output / 'legacy_vs_until1x1_latency.csv', latency_rows)
    summary_by_policy = {row['policy']: row for row in summaries}
    legacy_map = summary_by_policy['legacy']['macro_map50']
    until_map = summary_by_policy['until_1x1']['macro_map50']
    legacy_recall = summary_by_policy['legacy']['macro_recall50']
    until_recall = summary_by_policy['until_1x1']['macro_recall50']
    legacy_latency_values = [
        float(row['median_latency_ms']) for row in latency_rows
        if row['policy'] == 'legacy' and row.get('median_latency_ms')
    ]
    until_latency_values = [
        float(row['median_latency_ms']) for row in latency_rows
        if row['policy'] == 'until_1x1' and row.get('median_latency_ms')
    ]
    latency_comparison = (
        f"Prior legacy CPU median across {len(legacy_latency_values)} shapes: "
        f"{statistics.median(legacy_latency_values):.3f} ms; current until_1x1 "
        f"median across {len(until_latency_values)} shapes: "
        f"{statistics.median(until_latency_values):.3f} ms."
        if legacy_latency_values and until_latency_values
        else 'No complete historical legacy latency reference was available.')
    report = (
        '# Stage 2f: sequential until_1x1 implementation\n\n'
        '## Implementation\n\n'
        '- `RoiSSDMobileNet.forward` now computes a feature level, applies its '
        'classification and localization heads, then advances to the next block.\n'
        '- The first 1x1 feature map is included with its heads and is the final '
        'level. No deeper block or head executes afterward.\n'
        '- Runtime policy is fixed to `until_1x1`; legacy selection and the old '
        'threshold helper have been removed from the model.\n'
        '- No trainable layers, tensor shapes, or state-dict keys were changed. '
        'The selected checkpoint loaded with `strict=True`.\n\n'
        '## Validation\n\n'
        f'- Exact locked inputs and GT hashes verified: {metadata["exact_stage2d_inputs_verified"]}.\n'
        f'- Until-1x1 Stage 2e accuracy and prediction parity at 1e-6: '
        f'{metadata["stage2e_accuracy_parity_within_1e_6"]} / '
        f'{metadata["stage2e_predictions_parity_within_1e_6"]} '
        f'({metadata["prediction_rows_checked"]} per-sample outputs).\n'
        f'- Prediction/anchor slot checks passed: {metadata["slot_checks_passed"]}.\n'
        '- CPU timings verify sequential block/head execution using forward hooks; '
        f'{latency_comparison}\n\n'
        '## Accuracy comparison\n\n'
        '| Reference | Macro mAP@0.5 | Macro Recall@0.5 |\n'
        '|---|---:|---:|\n'
        f'| Stage 2e legacy reference | {legacy_map:.6f} | {legacy_recall:.6f} |\n'
        f'| Sequential until_1x1 | {until_map:.6f} | {until_recall:.6f} |\n\n'
        'Legacy figures are frozen Stage 2e measurements; legacy inference is '
        'intentionally no longer supported by the runtime model. The until_1x1 '
        'results reproduce Stage 2e on the same checkpoint, subset, preprocessing, '
        'evaluator and per-sample predictions. This supports using the policy in '
        'the Stage 3 CPU pilot, subject to pilot-level validation.\n'
    )
    (output / 'implementation_report.md').write_text(report, encoding='utf-8')
    metadata['output_sha256'] = {
        filename: sha(output / filename) for filename in (
            'implementation_report.md', 'legacy_vs_until1x1_accuracy.csv',
            'legacy_vs_until1x1_latency.csv', 'policy_summary.csv',
            'selected_depths.csv',
        )
    }
    write_json(output / 'metadata.json', metadata)
    print(f'COMPLETE {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--cpu-name', default='Intel Core i7-11700K @ 3.60GHz')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
