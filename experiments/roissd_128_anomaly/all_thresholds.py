"""Stage 2d: immutable-checkpoint accuracy sweeps and paired inference timing."""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

from .depth_boundary import check_control
from .forced_depth import evaluate, input_digest, selected_depth, spatially_impossible, structure
from .instrumentation import ForwardObserver, assert_slots
from .run import DATA, HERE, ROOT, crop, csv_rows, load_model, prepare, sha, write_json


OUT = ROOT / 'final_experiments' / 'depth_policy_validation'
PRIOR = ROOT / 'final_experiments' / 'diagnostics_128'
SPECS = (
    (32, 1, (), (31, 32, 33)),
    (64, 2, (), (63, 64, 65)),
    (96, 3, (), (95, 96, 97)),
    (140, 4, (124, 127, 128, 129, 132, 136, 140, 144), (127, 128, 129, 139, 140, 141)),
    (268, 5, (240, 248, 252, 255, 256, 257, 260, 264, 268, 272, 280),
     (255, 256, 257, 267, 268, 269)),
)
OFFSETS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
CPU_MINIMA = {32, 64, 96, 128, 140, 256, 268}


def round_pixel(value):
    return int(np.floor(value + .5))


def plan():
    result = []
    for boundary, lower, extra, rectangle_minima in SPECS:
        squares = sorted({boundary + offset for offset in OFFSETS} | set(extra))
        shapes = {(size, size): 'square' for size in squares}
        for minimum in rectangle_minima:
            for ratio in (1.25, 1.5):
                longer = round_pixel(minimum * ratio)
                shapes[minimum, longer] = 'rectangle'
                shapes[longer, minimum] = 'rectangle'
        for (height, width), group in sorted(shapes.items(), key=lambda item: (
                min(item[0]), item[0][0] != item[0][1], item[0])):
            result.append({
                'transition': f'L{lower}->L{lower + 1}', 'old_boundary': boundary,
                'lower_depth': lower, 'upper_depth': lower + 1,
                'input_h': height, 'input_w': width, 'min_dim': min(height, width),
                'shape_group': group,
                'cpu_representative': min(height, width) in CPU_MINIMA,
            })
    return result


class SlotGuard:
    """Cheap shape-only guards; no duplicate softmax or device synchronization."""
    def __init__(self, model):
        self.model = model
        self.checked_forwards = 0
        self.counts = {}
        self.stack = ExitStack()

    def reset(self, module, args):
        self.counts = {'loc': 0, 'cls': 0}

    def head(self, kind, coordinates):
        def check(module, args, output):
            assert output.shape[0] == 1 and output.shape[1] % coordinates == 0
            self.counts[kind] += output.numel() // coordinates
        return check

    def __enter__(self):
        self.stack.callback(self.model.register_forward_pre_hook(self.reset).remove)
        for heads, kind, coordinates in ((self.model.cls_heads, 'cls', self.model.num_classes),
                                          (self.model.bbox_reg_heads, 'loc', 4)):
            for head in heads:
                self.stack.callback(head.register_forward_hook(self.head(kind, coordinates)).remove)
        module = importlib.import_module(self.model.__class__.__module__)
        original = module.generate_default_boxes

        def generate(*args, **kwargs):
            boxes = original(*args, **kwargs)
            assert len(boxes) == 1
            assert_slots(self.counts['loc'], self.counts['cls'], len(boxes[0]))
            self.checked_forwards += 1
            return boxes

        self.stack.enter_context(patch.object(module, 'generate_default_boxes', generate))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def timed_pair(model, samples, depths, device, warmup, repeats, first_upper):
    tensors = [samples[index][0].to(device) for index in (0, 85, 170)]
    values = {depth: [] for depth in depths}
    counters = {depth: 0 for depth in depths}

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    with SlotGuard(model) as guard, torch.inference_mode():
        for iteration in range(warmup + repeats):
            order = list(depths)
            if (iteration + int(first_upper)) % 2:
                order.reverse()
            tensor = tensors[iteration % len(tensors)]
            for depth in order:
                with selected_depth(model, depth):
                    sync()
                    start = time.perf_counter_ns()
                    model(tensor, None)
                    sync()
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                counters[depth] += 1
                if iteration >= warmup:
                    values[depth].append(elapsed)
        assert guard.checked_forwards == len(depths) * (warmup + repeats)
    return [{
        'forced_L_R': depth, 'latency_device': str(device),
        'median_latency_ms': float(np.median(values[depth])),
        'p95_latency_ms': float(np.percentile(values[depth], 95)),
        'warmup_forwards': warmup, 'timed_forwards': repeats,
        'slot_checked_forwards': counters[depth], 'latency_samples_ms': values[depth],
        'timing_sample_ids': [0, 85, 170], 'batch_size': 1,
        'slot_guard_included_in_timing': True,
    } for depth in depths]


def verify_sources(hashes):
    for name, digest in hashes.items():
        assert sha(ROOT / name) == digest, f'Source drift: {name}'


def prior_controls():
    with (PRIOR / 'depth_boundary_validation.csv').open(newline='', encoding='utf-8') as stream:
        return {(int(row['input_h']), int(row['input_w']), int(row['actual_L_R'])): row
                for row in csv.DictReader(stream)}


def run(args):
    if args.warmup < 3 or args.repeats < 10 or args.threads < 1:
        raise ValueError('Use at least 3 warm-up forwards, 10 repeats and 1 thread')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    settings = yaml.safe_load((HERE / 'config.yaml').read_text())
    lock = prepare(settings)
    entry, = [item for item in lock['models'] if item['model_family'] == 'ROI-SSD-MobileNet']
    previous = json.loads((PRIOR / 'depth_boundary_metadata.json').read_text())
    assert previous['status'] == 'COMPLETE'
    for key in ('checkpoint', 'checkpoint_sha256', 'config_sha256'):
        assert entry[key] == previous[key], f'Selection changed: {key}'
    assert lock['subset_sha256'] == previous['subset_sha256']
    for key, old_key in (('seed', 'seed'), ('confidence', 'confidence_threshold'),
                         ('nms_iou', 'nms_iou'), ('max_detections', 'max_detections')):
        assert settings[key] == previous[old_key], f'Setting changed: {key}'
    source_hashes = dict(previous['source_sha256'])
    verify_sources(source_hashes)
    source_hashes['experiments/roissd_128_anomaly/all_thresholds.py'] = sha(Path(__file__))
    controls = prior_controls()
    rows = [json.loads(line) for line in (DATA / 'subset.jsonl').read_text().splitlines()]
    assert len(rows) == lock['num_samples'] == previous['num_samples'] == 256
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    images = []
    for row in rows:
        path = Path(lock['dataset_root']) / row['image']
        assert sha(path) == row['image_sha256'], f'Image changed: {path}'
        assert sha(Path(lock['dataset_root']) / row['label']) == row['label_sha256']
        image = cv2.imread(str(path))
        assert image is not None, f'Cannot read image: {path}'
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    jobs = plan()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        'checkpoint': entry['checkpoint'], 'checkpoint_sha256': entry['checkpoint_sha256'],
        'config_sha256': entry['config_sha256'], 'subset_sha256': lock['subset_sha256'],
        'seed': settings['seed'], 'confidence_threshold': settings['confidence'],
        'nms_iou': settings['nms_iou'], 'max_detections': settings['max_detections'],
    }
    metadata = {
        **provenance, 'status': 'RUNNING', 'started_at': datetime.now(timezone.utc).isoformat(),
        'accuracy_device': str(device), 'torch': torch.__version__,
        'torchvision': importlib.import_module('torchvision').__version__,
        'opencv': cv2.__version__, 'numpy': np.__version__,
        'platform': platform.platform(), 'processor': platform.processor(),
        'cpu_name': args.cpu_name, 'cpu_intraop_threads': args.threads, 'cpu_interop_threads': 1,
        'cuda_device': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'latency_warmup': args.warmup, 'latency_repeats': args.repeats,
        'timing_scope': 'forward incl decode/filter/NMS/cap and shape-only slot guard; '
                        'excludes crop, H2D, construction; CUDA synchronized wall time',
        'timing_order': 'alternate lower/upper each iteration; starting depth alternates per shape',
        'timing_sample_ids': [0, 85, 170], 'batch_size': 1,
        'practical_equivalence_absolute': .005,
        'equivalence_scope': 'descriptive 0.5 percentage points AP and recall, not significance',
        'num_samples': 256, 'planned_configurations': 2 * len(jobs),
        'plan': jobs, 'source_sha256': source_hashes,
        'stage2c_metadata_sha256': sha(PRIOR / 'depth_boundary_metadata.json'),
        'stage2c_csv_sha256': sha(PRIOR / 'depth_boundary_validation.csv'),
        'production_policy_changed': False, 'retraining': False,
    }
    metadata_path = out / 'all_thresholds_metadata.json'
    write_json(metadata_path, metadata)
    print('Configurations:', metadata['planned_configurations'], 'on', device, flush=True)
    model = load_model(entry, settings, device)
    cpu_model = model if device.type == 'cpu' else load_model(entry, settings, torch.device('cpu'))
    results, latencies = [], []
    controls_checked = 0
    with (out / 'input_hashes.jsonl').open('w', encoding='utf-8') as input_log:
        for job_index, job in enumerate(jobs):
            h, w = job['input_h'], job['input_w']
            samples = [crop(image, row, (h, w), settings['seed']) for image, row in zip(images, rows)]
            digests = [input_digest(*sample) for sample in samples]
            shared = hashlib.sha256(''.join(digests).encode('ascii')).hexdigest()
            for row, digest in zip(rows, digests):
                input_log.write(json.dumps({'input_h': h, 'input_w': w, 'sample_id': row['sample_id'],
                                           'tensor_gt_sha256': digest}) + '\n')
            current = model.get_max_feature_layer_by_roi_size(min(h, w))
            pair, feasible = [], []
            for depth in (job['lower_depth'], job['upper_depth']):
                identity = {**job, **provenance, 'forced_L_R': depth,
                            'current_policy_L_R': current, 'accuracy_device': str(device),
                            'input_gt_sha256': shared}
                print(f'[{job_index + 1}/{len(jobs)}]', job['transition'], f'{h}x{w}', depth, flush=True)
                try:
                    with selected_depth(model, depth), ForwardObserver(model) as observer, torch.inference_mode():
                        model(samples[0][0].to(device), None)
                        structure(observer, depth)
                except RuntimeError as error:
                    if not spatially_impossible(error):
                        raise
                    result = {**identity, 'status': 'INFEASIBLE', 'num_samples': 0,
                              'error': str(error), 'partial_heads': sorted(observer.heads)}
                    print('INFEASIBLE:', str(error), flush=True)
                else:
                    result = {**identity, **evaluate(model, samples, lock['class_names'], device, depth)}
                    if (h, w, depth) in controls:
                        reference = controls[h, w, depth]
                        assert reference['input_gt_sha256'] == shared, 'Stage-2c crop changed'
                        check_control(result, reference)
                        result['stage2c_control_matches'] = True
                        controls_checked += 1
                    feasible.append(depth)
                    print(' mAP/Recall', result['map50'], result['recall50'], flush=True)
                pair.append(result)
                assert model.get_max_feature_layer_by_roi_size(min(h, w)) == current
                assert 'get_max_feature_layer_by_roi_size' not in model.__dict__
            if len(feasible) == 2:
                for result in pair:
                    result['delta_map50_upper_minus_lower'] = pair[1]['map50'] - pair[0]['map50']
                    result['delta_recall50_upper_minus_lower'] = pair[1]['recall50'] - pair[0]['recall50']
            for timing_device, timing_model in [(device, model)] + (
                    [(torch.device('cpu'), cpu_model)]
                    if job['cpu_representative'] and device.type != 'cpu' else []):
                timing = timed_pair(timing_model, samples, feasible, timing_device,
                                    args.warmup, args.repeats, job_index % 2)
                for item in timing:
                    latencies.append({**job, **provenance, **item,
                                      'cpu_threads': args.threads, 'input_gt_sha256': shared})
                    result = next(row for row in pair if row['forced_L_R'] == item['forced_L_R'])
                    prefix = 'cpu' if timing_device.type == 'cpu' else 'cuda'
                    result[prefix + '_median_latency_ms'] = item['median_latency_ms']
                    result[prefix + '_p95_latency_ms'] = item['p95_latency_ms']
            results.extend(pair)
            csv_rows(out / 'all_thresholds_results.csv', results)
            csv_rows(out / 'latency_summary.csv', latencies)
            input_log.flush()
    assert len(results) == 2 * len(jobs)
    verify_sources(source_hashes)
    assert sha(ROOT / entry['checkpoint']) == entry['checkpoint_sha256']
    assert sha(PRIOR / 'depth_boundary_validation.csv') == metadata['stage2c_csv_sha256']
    metadata.update(
        status='COMPLETE', completed_at=datetime.now(timezone.utc).isoformat(),
        num_configurations=len(results), num_latency_configurations=len(latencies),
        num_infeasible=sum(row['status'] == 'INFEASIBLE' for row in results),
        stage2c_controls_checked=controls_checked, paired_inputs_verified=True,
        slot_checks_passed=True, source_integrity_verified=True, weights_unchanged=True,
        results_sha256=sha(out / 'all_thresholds_results.csv'),
        latency_sha256=sha(out / 'latency_summary.csv'),
        input_hashes_sha256=sha(out / 'input_hashes.jsonl'),
    )
    write_json(metadata_path, metadata)
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--cpu-name', default=platform.processor())
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args())
