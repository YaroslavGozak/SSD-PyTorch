"""Paired VGG16 regression: historical source, locked weights and 256 inputs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from unittest.mock import patch

import cv2
import numpy as np
import torch
import torchvision
import yaml

from model.roissd import RoiSSD
from tools.helpers.config_reader import load_config
from .run import ROOT, HERE, DATA, crop, prepare, sha, csv_rows, write_json
from .instrumentation import ForwardObserver, geometry
from .feature_map_policy import metric_functions
from .quick_regression_check import state_mapping_digest
from .vgg16_updated_regression import SQUARES, RECTANGLES, REF, read_csv

OUT = ROOT / 'final_experiments/roissd_vgg16_fixed_regression'
REVISION = '06b0bf4ea24168a189318fa6d8b146f897cdda84'


def legacy_class():
    source = subprocess.check_output(['git', 'show', f'{REVISION}:model/roissd.py'], cwd=ROOT)
    module = types.ModuleType('vgg16_legacy_regression')
    sys.modules[module.__name__] = module
    exec(compile(source, f'{REVISION}:model/roissd.py', 'exec'), module.__dict__)
    return module.RoiSSD, hashlib.sha256(source).hexdigest()


def build(cls, params, state, device, **kwargs):
    # All backbone parameters are present in strict checkpoint loading, so no
    # ImageNet download or redundant pretrained file loading is needed here.
    vgg = torchvision.models.vgg16
    with patch('torchvision.models.vgg16', side_effect=lambda **_: vgg(weights=None)):
        model = cls(copy.deepcopy(params), num_classes=31, **kwargs)
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert state_mapping_digest(model.state_dict()) == state_mapping_digest(state)
    return model.eval().to(device)


def full_frame(image, row):
    tensor = torch.from_numpy(cv2.resize(image, (320, 320))).permute(2, 0, 1).float() / 255
    tensor = ((tensor - torch.tensor([.485, .456, .406])[:, None, None]) /
              torch.tensor([.229, .224, .225])[:, None, None]).unsqueeze(0)
    gt = torch.tensor([[(cx-w/2), (cy-h/2), (cx+w/2), (cy+h/2)]
                       for _, cx, cy, w, h in row['gt']], dtype=torch.float32).reshape(-1, 4)
    labels = torch.tensor([int(g[0])+1 for g in row['gt']])
    return tensor, gt, labels


def structure(model, tensor, mode):
    with ForwardObserver(model) as ob, torch.inference_mode():
        _, detections = model(tensor, None)
        geometry(ob.anchors, ob.reference, *tensor.shape[-2:])
        det = detections[0]
        assert torch.isfinite(det['scores']).all() and torch.isfinite(det['boxes']).all()
        assert ((det['boxes'] >= 0) & (det['boxes'] <= 1)).all()
        assert (det['boxes'][:, 2:] >= det['boxes'][:, :2]).all()
        depth = len(ob.heads)
        expected = ['features', 'conv5_3_fc', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2'][:depth]
        assert ob.blocks == expected, (ob.blocks, expected)
        trace = model.get_runtime_log()
        assert trace['N_loc_predictions'] == trace['N_cls_predictions'] == trace['N_anchors']
        assert trace['active_heads'] == list(range(depth))
        assert trace['feature_shapes'] == ob.feature_shapes
        assert trace['N_anchors'] == len(ob.anchors)
        return {'mode': mode, 'input_h': tensor.shape[-2], 'input_w': tensor.shape[-1],
                **trace, 'executed_blocks': ob.blocks, 'status': 'PASS'}


def evaluate(model, samples, names, device):
    compute_map, _ = metric_functions()
    preds, truths, difficult, raw, ious, outputs = [], [], [], [], [], []
    anchors = None
    with ForwardObserver(model) as ob, torch.inference_mode():
        for tensor, gt, labels in samples:
            ob.reset()
            _, detections = model(tensor.to(device), None)
            det = {k: v.cpu() for k, v in detections[0].items()}
            assert torch.isfinite(det['scores']).all() and torch.isfinite(det['boxes']).all()
            assert ((det['labels'] >= 1) & (det['labels'] <= 30)).all()
            assert ((det['boxes'] >= 0) & (det['boxes'] <= 1)).all()
            assert (det['boxes'][:, 2:] >= det['boxes'][:, :2]).all()
            if anchors is None:
                anchors = ob.anchors.cpu()
            if len(gt):
                ious.extend(torchvision.ops.box_iou(gt, anchors).max(1).values.tolist())
            raw.append(max(head['foreground_max_confidence'] for head in ob.heads.values()))
            pred, truth = {n: [] for n in names}, {n: [] for n in names}
            for box, label, score in zip(det['boxes'].tolist(), det['labels'].tolist(), det['scores'].tolist()):
                pred[names[label-1]].append([*box, score])
            for box, label in zip(gt.tolist(), labels.tolist()):
                truth[names[label-1]].append(box)
            preds.append(pred)
            truths.append(truth)
            difficult.append({n: [0]*len(boxes) for n, boxes in truth.items()})
            outputs.append(det)
    ap, _, recall, _ = compute_map(preds, truths, iou_threshold=.5, method='area', difficult=difficult)
    return {'map50': ap, 'recall50': recall, 'num_detections': sum(len(d['scores']) for d in outputs),
            'mean_raw_max_confidence': float(np.mean(raw)), 'mean_best_anchor_iou': float(np.mean(ious)),
            'fraction_gt_best_anchor_iou_ge_05': float(np.mean(np.asarray(ious) >= .5)),
            'depth': len(ob.heads), 'total_anchors': len(anchors)}, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    OUT.mkdir(exist_ok=True, parents=True)
    settings = yaml.safe_load((HERE / 'config.yaml').read_text())
    lock = prepare(settings)
    entry, = [e for e in lock['models'] if e['model_family'] == 'ROI-SSD']
    cfg = load_config(str(DATA / entry['config_snapshot']))
    params = dict(cfg['model_params'], low_score_threshold=settings['confidence'],
                  nms_threshold=settings['nms_iou'], detections_per_img=settings['max_detections'])
    checkpoint = torch.load(ROOT / entry['checkpoint'], map_location='cpu', weights_only=True)
    state = checkpoint.get('model', checkpoint)
    old_class, old_sha = legacy_class()
    old = build(old_class, params, state, args.device)
    new = build(RoiSSD, params, state, args.device, record_execution_trace=True)
    before = state_mapping_digest(new.state_dict())
    metadata = {'status': 'RUNNING', 'checkpoint': entry, 'config': cfg,
                'config_overrides': {k: params[k] for k in ('low_score_threshold', 'nms_threshold', 'detections_per_img')},
                'subset_sha256': lock['subset_sha256'], 'seed': settings['seed'],
                'device': args.device, 'cpu_threads': 4, 'torch': torch.__version__,
                'legacy_revision': REVISION, 'legacy_source_sha256': old_sha,
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'source_sha256': {p: sha(ROOT / p) for p in ('model/roissd.py',
                    'experiments/roissd_128_anomaly/vgg16_fixed_regression.py',
                    'experiments/roissd_128_anomaly/run.py', 'tools/infer.py')},
                'metric_tolerance': 1e-6, 'num_samples': 256}
    write_json(OUT / 'metadata.json', metadata)
    rows = [json.loads(line) for line in (DATA / 'subset.jsonl').read_text().splitlines()]
    assert len(rows) == 256
    images = []
    for row in rows:
        path = Path(lock['dataset_root']) / row['image']
        assert sha(path) == row['image_sha256']
        assert sha(Path(lock['dataset_root']) / row['label']) == row['label_sha256']
        images.append(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
    shapes = [(n, n) for n in SQUARES] + RECTANGLES
    structural = [structure(new, crop(images[0], rows[0], shape, settings['seed'])[0].to(args.device), 'crop')
                  for shape in shapes]
    structural.append(structure(new, full_frame(images[0], rows[0])[0].to(args.device), 'full_frame'))
    csv_rows(OUT / 'structural_checks.csv', structural)
    print('Structural checks PASS:', len(structural), flush=True)
    saved = {int(r['requested_h']): r for r in read_csv(REF / 'size_sweep_metrics.csv') if r['model_family'] == 'ROI-SSD'}
    saved_meta = json.loads((REF / 'run_metadata.json').read_text())
    assert saved_meta['subset_sha256'] == lock['subset_sha256']
    assert all(saved_meta[k] == settings[k] for k in ('seed', 'confidence', 'nms_iou', 'max_detections'))
    results = []
    # Full rectangular accuracy for the four optional probes as well.
    accuracy_shapes = [(n, n) for n in SQUARES] + [(128, 160), (160, 128), (140, 192), (192, 140)]
    for shape in accuracy_shapes + [(320, 320)]:
        mode = 'full_frame' if shape == (320, 320) else 'crop'
        if mode == 'full_frame':
            old.low_score_threshold = new.low_score_threshold = .01
            samples = [full_frame(im, row) for im, row in zip(images, rows)]
        else:
            samples = [crop(im, row, shape, settings['seed']) for im, row in zip(images, rows)]
        digest = hashlib.sha256(b''.join(t.numpy().tobytes()+g.numpy().tobytes()+l.numpy().tobytes()
                                          for t, g, l in samples)).hexdigest()
        print('Accuracy', mode, shape, flush=True)
        old_result, old_outputs = evaluate(old, samples, lock['class_names'], args.device)
        new_result, new_outputs = evaluate(new, samples, lock['class_names'], args.device)
        same_depth = old_result['depth'] == new_result['depth']
        parity = True
        if same_depth:
            for a, b in zip(old_outputs, new_outputs):
                for key in a:
                    if a[key].shape != b[key].shape or not torch.allclose(a[key], b[key], atol=1e-6, rtol=0):
                        parity = False
        historical = saved.get(shape[0]) if mode == 'crop' and shape[0] == shape[1] else None
        historical_match = None
        if historical:
            historical_match = all(abs(old_result[k] - float(historical[k])) <= 1e-6 for k in ('map50', 'recall50'))
            assert historical_match, ('Historical replay mismatch', shape, old_result, historical)
        result = {'mode': mode, 'input_h': shape[0], 'input_w': shape[1],
                  **{'old_'+k: v for k, v in old_result.items()},
                  **{'new_'+k: v for k, v in new_result.items()},
                  'delta_mAP50': new_result['map50']-old_result['map50'],
                  'delta_Recall50': new_result['recall50']-old_result['recall50'],
                  'same_depth_prediction_parity': parity if same_depth else None,
                  'historical_metrics_reproduced': historical_match, 'input_gt_sha256': digest,
                  'confidence_threshold': new.low_score_threshold}
        result['status'] = 'PASS' if result['delta_mAP50'] >= -1e-6 and result['delta_Recall50'] >= -1e-6 and parity else 'FAIL'
        results.append(result)
        csv_rows(OUT / 'accuracy_comparison.csv', results)
        print('  old/new mAP', old_result['map50'], new_result['map50'], result['status'], flush=True)
    # Timing without observer overhead. Separate hooks verify the blocks/heads once per shape.
    cpu_new, cpu_old = new.to('cpu'), old.to('cpu')
    cpu_new.low_score_threshold = cpu_old.low_score_threshold = settings['confidence']
    latency = []
    for n in (96, 128, 140, 256, 300):
        tensor = crop(images[0], rows[0], (n, n), settings['seed'])[0]
        trace = structure(cpu_new, tensor, 'crop')
        timed = {}
        for name, model in [('old', cpu_old), ('new', cpu_new)]:
            times = []
            with torch.inference_mode():
                for i in range(13):
                    start = time.perf_counter_ns()
                    model(tensor, None)
                    if i >= 3:
                        times.append((time.perf_counter_ns()-start)/1e6)
            timed[name+'_median_latency_ms'] = statistics.median(times)
            timed[name+'_p95_latency_ms'] = float(np.percentile(times, 95))
        latency.append({'input_h': n, 'input_w': n, 'active_depth': trace['selected_L_R'],
                        'feature_shapes': trace['feature_shapes'], 'executed_blocks': trace['executed_blocks'], **timed})
    csv_rows(OUT / 'latency_sanity.csv', latency)
    assert state_mapping_digest(new.state_dict()) == state_mapping_digest(old.state_dict()) == before
    compatibility = {'strict': True, 'missing_keys': [], 'unexpected_keys': [], 'shape_mismatches': {},
                     'checkpoint_tensor_sha256': before, 'weights_unchanged_after_inference': True,
                     'finite_inference': True}
    (OUT / 'checkpoint_compatibility.txt').write_text(json.dumps(compatibility, indent=2)+'\n')
    status = 'PASS' if all(r['status'] == 'PASS' for r in results) else 'FAIL'
    metadata.update(status=status, checkpoint_compatibility=compatibility,
                    structural_shapes=len(structural), accuracy_shapes=len(results),
                    input_pairs_evaluated=len(results)*len(rows),
                    reference_artifact_sha256={p: sha(REF / p) for p in ('size_sweep_metrics.csv', 'run_metadata.json')})
    report = ['# Fixed ROI-SSD VGG16 regression', '', f'**Result: {status}**', '',
              'Strict checkpoint compatibility, finite/bounded outputs, independent anchor geometry and '
              'prediction counts verified. Old and new run on the identical frozen 256-frame tensors/GT. '
              'Historical Stage 2 CSV metrics are replayed at tolerance 1e-6 where available. '
              'Full-frame comparison uses the same ROI checkpoint at confidence 0.01, resize 320x320, '
              'ImageNet normalization, project area-AP evaluator and NMS 0.45. Crop confidence remains 0.5.', '',
              '| Mode/input H×W | Old depth | New depth | Old mAP50 | New mAP50 | ΔmAP50 | ΔRecall50 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        report.append(f'| {r["mode"]} {r["input_h"]}×{r["input_w"]} | {r["old_depth"]} | {r["new_depth"]} | '
                      f'{r["old_map50"]:.9f} | {r["new_map50"]:.9f} | {r["delta_mAP50"]:+.9f} | {r["delta_Recall50"]:+.9f} |')
    report += ['', 'CPU latency: 4 threads, batch 1, 3 warmups + 10 repeats per shape/model; '
               'see latency_sanity.csv. Hook traces verify each active block runs once and no deeper block runs. '
               'This is a short sanity measurement, not a deployment benchmark. '
               'Results establish regression behavior on this locked validation subset, not all possible inputs.']
    (OUT / 'regression_report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    metadata['output_sha256'] = {p: sha(OUT / p) for p in ('checkpoint_compatibility.txt',
        'structural_checks.csv', 'accuracy_comparison.csv', 'latency_sanity.csv', 'regression_report.md')}
    write_json(OUT / 'metadata.json', metadata)
    print(status, OUT, flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
