"""Focused checkpoint and Stage 2e regression check for until_1x1."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import traceback

import cv2
import torch
import yaml

from model.roissd_mobilenet import RoiSSDMobileNet
from tools.helpers.config_reader import load_config
from .implementation_validation import (
    STAGE2D,
    STAGE2E,
    aggregate_policy,
    expected_block_calls,
    expected_execution_events,
    load_reference_predictions,
    read_csv,
    state_dict_digest,
    time_policy,
)
from .run import DATA, HERE, ROOT, crop, prepare, sha, write_json

OUTPUT = ROOT / 'final_experiments' / 'feature_map_depth_policy_implementation_check'
SHAPES = (
    (96, 96), (128, 128), (128, 160), (129, 129), (140, 140),
    (144, 144), (256, 256), (256, 320), (257, 257), (268, 268),
)
CPU_SHAPES = ((128, 128), (140, 140), (256, 256), (268, 268))
TOLERANCE = 1e-6


def state_mapping_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Checkpoint state entry is not a tensor: {key}')
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, separators=(',', ':'))
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def load_stage2e_predictions(shapes, reference_rows, subset_rows):
    keys = {
        (h, w, int(reference_rows[(h, w)]['selected_L_R']), sample['sample_id'])
        for h, w in shapes
        for sample in subset_rows
    }
    return load_reference_predictions(keys), keys


def load_checkpoint(entry, settings, device):
    model_config = load_config(str(DATA / entry['config_snapshot']))
    params = dict(model_config['model_params'])
    params.update(
        low_score_threshold=settings['confidence'],
        nms_threshold=settings['nms_iou'],
        detections_per_img=settings['max_detections'],
    )
    model = RoiSSDMobileNet(
        config=params, num_classes=31, record_execution_trace=True)
    checkpoint = torch.load(ROOT / entry['checkpoint'], map_location='cpu', weights_only=True)
    state = checkpoint.get('model', checkpoint)
    model_state = model.state_dict()
    checkpoint_keys, model_keys = set(state), set(model_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatches = {
        key: {'checkpoint': list(state[key].shape), 'model': list(model_state[key].shape)}
        for key in sorted(model_keys & checkpoint_keys)
        if not isinstance(state[key], torch.Tensor) or state[key].shape != model_state[key].shape
    }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f'Checkpoint structure mismatch: missing={missing}, '
            f'unexpected={unexpected}, shape_mismatches={shape_mismatches}')

    checkpoint_digest = state_mapping_digest(state)
    incompatible = model.load_state_dict(state, strict=True)
    loaded_digest = state_dict_digest(model)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f'Strict loading returned incompatible keys: {incompatible}')
    if loaded_digest != checkpoint_digest:
        raise RuntimeError('Loaded model state digest differs from checkpoint tensors')
    return model.to(device).eval(), {
        'strict': True,
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
        'shape_mismatches': shape_mismatches,
        'checkpoint_tensor_sha256': checkpoint_digest,
        'loaded_state_sha256': loaded_digest,
        'weights_match_checkpoint': loaded_digest == checkpoint_digest,
    }


def install_order_guard(model):
    events = []
    handles = []
    blocks = (
        'features_stage1', 'conv_to_672', 'features_stage2', 'conv8_2',
        'conv9_2', 'conv10_2', 'conv11_2',
    )

    def reset_events(module, inputs):
        events.clear()

    def check_order(module, inputs, output):
        trace = module.get_runtime_log()
        if trace is None:
            raise AssertionError('Runtime trace missing during structural check')
        expected = expected_execution_events(trace['selected_L_R'])
        if events != expected:
            raise AssertionError(f'Unexpected feature/head execution: {events} != {expected}')
        if trace['final_feature_shape'] != [1, 1]:
            raise AssertionError(f'Final feature shape is not 1x1: {trace["final_feature_shape"]}')
        if not trace['stopped_at_1x1']:
            raise AssertionError('Forward did not report stopping at the first 1x1 map')
        if not (
                trace['N_loc_predictions'] == trace['N_cls_predictions']
                == trace['N_anchors']):
            raise AssertionError('Prediction and anchor counts differ')

    handles.append(model.register_forward_pre_hook(reset_events))
    for name in blocks:
        def record_block(module, inputs, output, block_name=name):
            events.append(block_name)
        handles.append(getattr(model, name).register_forward_hook(record_block))
    for kind, modules in (('cls', model.cls_heads), ('loc', model.bbox_reg_heads)):
        for index, module in enumerate(modules):
            def record_head(module, inputs, output, head_kind=kind, head_index=index):
                events.append(f'{head_kind}_{head_index}')
            handles.append(module.register_forward_hook(record_head))
    handles.append(model.register_forward_hook(check_order))
    return handles


def verify_reference_structure(result, trace, reference, shape):
    checks = {
        'selected_depth': int(reference['selected_L_R']) == result['actual_L_R'],
        'active_heads': json.loads(reference['active_heads']) == trace['active_heads'],
        'feature_shapes': json.loads(reference['spatial_feature_shapes']) == trace['feature_shapes'],
        'last_map_1x1': trace['final_feature_shape'] == [1, 1],
        'anchors': int(reference['N_anchors']) == trace['N_anchors'],
        'anchors_per_head': json.loads(reference['anchors_per_head']) == trace['anchors_per_head'],
        'prediction_anchor_slots': (
            trace['N_loc_predictions'] == trace['N_cls_predictions'] == trace['N_anchors']),
        'input_h': int(reference['input_h']) == shape[0],
        'input_w': int(reference['input_w']) == shape[1],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f'Stage 2e structure mismatch at {shape}: {failed}')
    return checks


def run(args):
    if args.warmup < 3 or args.repeats < 10:
        raise ValueError('CPU sanity timing requires at least 3 warm-ups and 10 repeats')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        'status': 'RUNNING',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'shapes': [list(shape) for shape in SHAPES],
        'cpu_latency_shapes': [list(shape) for shape in CPU_SHAPES],
        'num_samples_per_shape': 256,
        'tolerance': TOLERANCE,
        'device': args.device,
        'cpu_threads': args.threads,
        'cpu_warmup': args.warmup,
        'cpu_repeats': args.repeats,
        'batch_size': 1,
        'torch': torch.__version__,
        'opencv': cv2.__version__,
        'platform': platform.platform(),
    }
    checkpoint_lines = []
    report_path = OUTPUT / 'quick_regression_report.md'
    checkpoint_report_path = OUTPUT / 'checkpoint_load_report.txt'
    try:
        settings = yaml.safe_load((HERE / 'config.yaml').read_text(encoding='utf-8'))
        lock = prepare(settings)
        stage2e_meta = json.loads((STAGE2E / 'metadata.json').read_text(encoding='utf-8'))
        if stage2e_meta['status'] != 'COMPLETE' or stage2e_meta['num_samples'] != 256:
            raise RuntimeError('Stage 2e reference is not a complete 256-sample run')
        if stage2e_meta['subset_sha256'] != lock['subset_sha256']:
            raise RuntimeError('Locked Stage 2e subset differs from the current subset lock')
        for filename, digest in stage2e_meta['output_sha256'].items():
            if sha(STAGE2E / filename) != digest:
                raise RuntimeError(f'Stage 2e reference artifact hash mismatch: {filename}')
        entry, = [item for item in lock['models']
                  if item['model_family'] == 'ROI-SSD-MobileNet']
        if sha(ROOT / entry['checkpoint']) != entry['checkpoint_sha256']:
            raise RuntimeError('Checkpoint differs from the locked checkpoint hash')
        metadata.update({
            'checkpoint': entry['checkpoint'],
            'checkpoint_sha256': entry['checkpoint_sha256'],
            'subset_sha256': lock['subset_sha256'],
            'stage2e_metadata_sha256': sha(STAGE2E / 'metadata.json'),
            'stage2e_output_sha256': stage2e_meta['output_sha256'],
            'stage2d_input_hashes_sha256': sha(STAGE2D / 'input_hashes.jsonl'),
            'confidence_threshold': settings['confidence'],
            'nms_iou': settings['nms_iou'],
            'max_detections': settings['max_detections'],
        })

        device = torch.device(
            'cuda' if args.device == 'auto' and torch.cuda.is_available()
            else 'cpu' if args.device == 'auto' else args.device)
        model, checkpoint_info = load_checkpoint(entry, settings, device)
        checkpoint_lines.extend((
            f'checkpoint: {entry["checkpoint"]}',
            f'checkpoint_sha256: {entry["checkpoint_sha256"]}',
            'load_state_dict(strict=True): PASS',
            'missing_keys: []',
            'unexpected_keys: []',
            'shape_mismatches: {}',
            f'checkpoint_tensor_sha256: {checkpoint_info["checkpoint_tensor_sha256"]}',
            f'loaded_state_sha256: {checkpoint_info["loaded_state_sha256"]}',
            'checkpoint_weights_match_loaded_model: True',
        ))
        metadata.update({
            'device': str(device),
            'checkpoint_load': checkpoint_info,
            'state_dict_sha256_before_inference': state_dict_digest(model),
        })
        checkpoint_report_path.write_text('\n'.join(checkpoint_lines) + '\n', encoding='utf-8')

        reference_rows = {
            (int(row['input_h']), int(row['input_w'])): row
            for row in read_csv(STAGE2E / 'per_shape_results.csv')
            if row['policy'] == 'until_1x1'
        }
        if len(reference_rows) != 36:
            raise RuntimeError('Stage 2e until_1x1 per-shape reference is incomplete')
        for shape in SHAPES:
            if shape not in reference_rows:
                raise RuntimeError(f'Missing Stage 2e result for input shape {shape}')
        subset_rows = [
            json.loads(line) for line in (DATA / 'subset.jsonl').read_text(
                encoding='utf-8').splitlines()
        ]
        if len(subset_rows) != 256:
            raise RuntimeError(f'Expected locked 256-sample subset, got {len(subset_rows)}')
        reference_predictions, required_prediction_keys = load_stage2e_predictions(
            SHAPES, reference_rows, subset_rows)

        stage2d_hashes = {}
        for line in (STAGE2D / 'input_hashes.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            stage2d_hashes[(int(row['input_h']), int(row['input_w']), row['sample_id'])] = (
                row['tensor_gt_sha256'])
        images = []
        for row in subset_rows:
            image_path = Path(lock['dataset_root']) / row['image']
            if sha(image_path) != row['image_sha256']:
                raise RuntimeError(f'Locked source image hash changed: {row["image"]}')
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f'Unable to read locked image: {image_path}')
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        results = []
        cpu_model = copy.deepcopy(model).to('cpu').eval()
        cpu_model.record_execution_trace = True
        cpu_model_state_before = state_dict_digest(cpu_model)
        for shape_index, shape in enumerate(SHAPES, start=1):
            print(f'Quick accuracy {shape_index}/{len(SHAPES)}: {shape[0]}x{shape[1]}',
                  flush=True)
            samples = [
                crop(image, row, shape, settings['seed'])
                for image, row in zip(images, subset_rows)
            ]
            sample_digests = [
                __import__(
                    'experiments.roissd_128_anomaly.forced_depth',
                    fromlist=['input_digest']).input_digest(tensor, gt, labels)
                for tensor, gt, labels in samples
            ]
            for source, digest in zip(subset_rows, sample_digests):
                expected = stage2d_hashes[(shape[0], shape[1], source['sample_id'])]
                if digest != expected:
                    raise RuntimeError(
                        f'Locked crop/GT hash mismatch for {shape} sample {source["sample_id"]}')
            combined_digest = hashlib.sha256(
                ''.join(sample_digests).encode('ascii')).hexdigest()
            reference = reference_rows[shape]
            if combined_digest != reference['input_gt_sha256']:
                raise RuntimeError(f'Stage 2e tensor/GT digest mismatch at {shape}')

            order_handles = install_order_guard(model)
            try:
                result, _ = aggregate_policy(
                    model, samples, subset_rows, lock['class_names'], device,
                    shape, reference_predictions)
            finally:
                for handle in order_handles:
                    handle.remove()
            trace = model.get_runtime_log()
            structure_checks = verify_reference_structure(result, trace, reference, shape)
            for metric in ('map50', 'recall50'):
                actual, expected = float(result[metric]), float(reference[metric])
                delta = abs(actual - expected)
                if delta > TOLERANCE:
                    raise AssertionError(
                        f'{metric} mismatch at {shape}: actual={actual}, '
                        f'reference={expected}, delta={delta}')
            reference_detections = int(reference['num_detections'])
            if result['num_detections'] != reference_detections:
                raise AssertionError(
                    f'Detection count mismatch at {shape}: {result["num_detections"]} '
                    f'!= {reference_detections}')
            results.append({
                'input_h': shape[0], 'input_w': shape[1],
                'selected_L_R': result['actual_L_R'],
                'active_heads': trace['active_heads'],
                'feature_shapes': trace['feature_shapes'],
                'final_feature_shape': trace['final_feature_shape'],
                'anchors_per_head': trace['anchors_per_head'],
                'total_anchors': trace['N_anchors'],
                'N_loc_predictions': trace['N_loc_predictions'],
                'N_cls_predictions': trace['N_cls_predictions'],
                'num_detections': result['num_detections'],
                'stage2e_num_detections': reference_detections,
                'map50': result['map50'], 'stage2e_map50': float(reference['map50']),
                'map50_abs_delta': abs(result['map50'] - float(reference['map50'])),
                'recall50': result['recall50'],
                'stage2e_recall50': float(reference['recall50']),
                'recall50_abs_delta': abs(
                    result['recall50'] - float(reference['recall50'])),
                'prediction_rows_compared': result['prediction_rows_checked'],
                'prediction_output_parity': True,
                'structure_checks': structure_checks,
                'input_gt_sha256': combined_digest,
            })
            write_csv(OUTPUT / 'stage2e_reproduction.csv', results)

        latency_rows = []
        for shape in CPU_SHAPES:
            tensor = crop(images[0], subset_rows[0], shape, settings['seed'])[0]
            timed = time_policy(
                cpu_model, tensor, 'until_1x1', args.warmup, args.repeats)
            expected_blocks = expected_block_calls(timed['selected_L_R'])
            if timed['executed_blocks_per_forward'] != expected_blocks:
                raise AssertionError(f'Unexpected CPU block execution at {shape}')
            if not (0 < timed['median_latency_ms'] < 1000):
                raise AssertionError(f'Implausible CPU latency at {shape}: {timed}')
            latency_rows.append({
                'input_h': shape[0], 'input_w': shape[1], **timed,
                'early_exit_verified': True,
                'latency_sanity': 'PASS',
            })
        write_csv(OUTPUT / 'cpu_latency_sanity.csv', latency_rows)

        after_inference_digest = state_dict_digest(model)
        if after_inference_digest != metadata['state_dict_sha256_before_inference']:
            raise AssertionError('Inference modified checkpoint weights')
        if state_dict_digest(cpu_model) != cpu_model_state_before:
            raise AssertionError('CPU timing modified model weights')
        metadata.update({
            'status': 'PASS',
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'representative_shapes_checked': len(results),
            'per_sample_prediction_rows_compared': len(required_prediction_keys),
            'metric_tolerance_met': True,
            'structural_checks_passed': True,
            'early_exit_execution_checked': True,
            'inference_preserved_weights': True,
            'cpu_latency_sanity_shapes': len(latency_rows),
            'median_cpu_latency_ms': {
                f'{row["input_h"]}x{row["input_w"]}': row['median_latency_ms']
                for row in latency_rows
            },
            'stage2e_accuracy_mean': {
                'map50': statistics.mean(row['map50'] for row in results),
                'recall50': statistics.mean(row['recall50'] for row in results),
            },
            'source_sha256': {
                filename: sha(ROOT / filename) for filename in (
                    'model/roissd_mobilenet.py',
                    'experiments/roissd_128_anomaly/quick_regression_check.py',
                    'experiments/roissd_128_anomaly/implementation_validation.py',
                    'experiments/roissd_128_anomaly/run.py',
                )
            },
            'output_sha256': {
                'stage2e_reproduction.csv': sha(OUTPUT / 'stage2e_reproduction.csv'),
                'cpu_latency_sanity.csv': sha(OUTPUT / 'cpu_latency_sanity.csv'),
                'checkpoint_load_report.txt': sha(checkpoint_report_path),
            },
        })
        report = [
            '# ROI-SSD-MobileNet quick regression check',
            '',
            '**Result: PASS**',
            '',
            '- Existing checkpoint loaded with `strict=True`; key sets and tensor shapes match.',
            '- Locked Stage 2e inputs and GT were reproduced for the 10 requested shapes.',
            '- All per-sample predictions, detections, selected depths, feature maps, anchors,',
            '  mAP@0.5 and Recall@0.5 match Stage 2e; metric tolerance is 1e-6.',
            '- Each checked forward ends at its first 1x1 map; execution hooks verify no',
            '  later block/head runs and verify `N_loc == N_cls == N_anchors`.',
            '- CPU batch-1 timing and early-exit checks passed on four requested shapes.',
            '',
            '## Stage 2e comparison',
            '',
            '| Input HxW | L_R | mAP@0.5 | Recall@0.5 | Detections | Anchors | Final map |',
            '|---|---:|---:|---:|---:|---:|---|',
        ]
        report.extend(
            f'| {row["input_h"]}x{row["input_w"]} | {row["selected_L_R"]} | '
            f'{row["map50"]:.6f} | {row["recall50"]:.6f} | '
            f'{row["num_detections"]} | {row["total_anchors"]} | '
            f'{row["final_feature_shape"]} |'
            for row in results
        )
        report.extend(('', '## CPU latency sanity check', '',
                       '| Input HxW | L_R | Median (ms) | p95 (ms) | Executed blocks |',
                       '|---|---:|---:|---:|---|'))
        report.extend(
            f'| {row["input_h"]}x{row["input_w"]} | {row["selected_L_R"]} | '
            f'{row["median_latency_ms"]:.3f} | {row["p95_latency_ms"]:.3f} | '
            f'{row["executed_blocks_per_forward"]} |'
            for row in latency_rows
        )
        report.extend(('', 'Checkpoint compatibility, inference parity, structural checks,',
                       'and runtime sanity support proceeding to the Stage 3 CPU pilot.'))
        report_path.write_text('\n'.join(report) + '\n', encoding='utf-8')
        metadata['output_sha256']['quick_regression_report.md'] = sha(report_path)
    except Exception:
        metadata['status'] = 'FAIL'
        metadata['error'] = traceback.format_exc()
        metadata['completed_at'] = datetime.now(timezone.utc).isoformat()
        if checkpoint_lines and not checkpoint_report_path.exists():
            checkpoint_report_path.write_text(
                '\n'.join(checkpoint_lines) + '\n', encoding='utf-8')
        report_path.write_text(
            '# ROI-SSD-MobileNet quick regression check\n\n'
            '**Result: FAIL**\n\n```\n'
            f'{metadata["error"]}```\n', encoding='utf-8')
        write_json(OUTPUT / 'metadata.json', metadata)
        raise
    write_json(OUTPUT / 'metadata.json', metadata)
    print(f'PASS {OUTPUT}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
