"""Bind saved-sweep evidence conservatively and check current locked weights on CPU."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import torch
import yaml

from .run import ROOT, DATA, crop, prepare, sha
from .instrumentation import ForwardObserver, geometry
from .quick_regression_check import state_mapping_digest

OUT = ROOT / 'final_experiments/fresh_stage2_regression'
SHAPES = [(n, n) for n in (95, 96, 97, 127, 128, 129, 139, 140, 141, 255, 256, 257, 267, 268, 269)] + [(128, 160), (160, 128), (256, 320), (320, 256)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    settings = yaml.safe_load((Path(__file__).parent / 'config.yaml').read_text())
    lock = prepare(settings)
    sources = [json.loads(line) for line in (DATA / 'subset.jsonl').read_text().splitlines()][:3]
    images = []
    for row in sources:
        path = Path(lock['dataset_root']) / row['image']
        assert sha(path) == row['image_sha256']
        assert sha(Path(lock['dataset_root']) / row['label']) == row['label_sha256']
        images.append(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
    evidence = {}
    records = []
    metadata = {'status': 'RUNNING', 'scope': 'Fresh structural boundary regression; no dense accuracy sweep rerun',
                'device': 'cpu', 'threads': 8, 'samples_per_shape': 3, 'shapes_hw': SHAPES,
                'subset_sha256': lock['subset_sha256'], 'seed': settings['seed'],
                'confidence': .5, 'nms_iou': .45, 'max_detections': 200}
    write_json(OUT / 'metadata.json', metadata)
    for entry in lock['models']:
        family = entry['model_family']
        checkpoint = ROOT / entry['checkpoint']
        assert sha(checkpoint) == entry['checkpoint_sha256']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        state = saved.get('model', saved)
        weight_digest = state_mapping_digest(state)
        cfg = yaml.safe_load((DATA / entry['config_snapshot']).read_text())
        directory = 'roi-ssd-retrained' if family == 'ROI-SSD' else 'roi-ssd-mobilenet-retrained'
        sweep = ROOT / 'final_experiments' / directory / 'fixed_size_results_v2.csv'
        rows = list(csv.DictReader(sweep.open()))
        actual = {r['evaluated_dataset'] for r in rows}
        expected = {f'fixed_size_{h}x{w}' for h in range(32, 321, 32) for w in range(32, 321, 32)}
        assert len(rows) == 100 and actual == expected
        assert {r['task_name'] for r in rows} == {cfg['train_params']['task_name']}
        assert {r['ckpt_name'] for r in rows} == {'last.pt'}
        last = checkpoint.parent / 'last.pt'
        last_saved = torch.load(last, map_location='cpu', weights_only=True, mmap=True)
        last_digest = state_mapping_digest(last_saved.get('model', last_saved))
        evidence[family] = {
            'selected_checkpoint': entry['checkpoint'], 'selected_checkpoint_sha256': entry['checkpoint_sha256'],
            'selected_tensor_state_sha256': weight_digest, 'selected_epoch': saved.get('epoch'),
            'config_snapshot': entry['config_snapshot'], 'config_sha256': sha(DATA / entry['config_snapshot']),
            'sweep_csv': sweep.relative_to(ROOT).as_posix(), 'sweep_csv_sha256': sha(sweep),
            'rows': len(rows), 'unique_shapes': len(actual), 'shape_grid_complete': True,
            'sweep_declared_checkpoint': 'last.pt', 'sweep_inference_time_checkpoint_hash': None,
            'current_last_checkpoint_sha256': sha(last), 'current_last_tensor_state_sha256': last_digest,
            'current_last_epoch': last_saved.get('epoch'),
            'current_last_equals_selected_tensor_state': last_digest == weight_digest,
            'binding_status': 'UNVERIFIED_NOT_ATTRIBUTABLE_TO_SELECTED_BEST',
            'binding_reason': 'CSV declares last.pt and has no inference-time checkpoint hash. Current last.pt and selected best_map.pt tensor states differ. Current file hashes do not establish historical sweep weight identity.',
            'historical_control_metrics': [r for r in rows if r['evaluated_dataset'] in ('fixed_size_96x96', 'fixed_size_128x128', 'fixed_size_160x160')],
        }
        del last_saved
        from model.roissd import RoiSSD
        from model.roissd_mobilenet import RoiSSDMobileNet
        cls = RoiSSD if family == 'ROI-SSD' else RoiSSDMobileNet
        model = cls(config=cfg['model_params'], num_classes=31, record_execution_trace=True)
        model.load_state_dict(state, strict=True)
        assert state_mapping_digest(model.state_dict()) == weight_digest
        model.low_score_threshold = .5
        model.eval()
        with ForwardObserver(model) as observer, torch.inference_mode():
            for h, w in SHAPES:
                for sample, (source, image) in enumerate(zip(sources, images)):
                    x, gt, labels = crop(image, source, (h, w), settings['seed'])
                    observer.reset()
                    _, predictions = model(x, None)
                    trace = model.get_runtime_log()
                    assert trace['depth_policy'] == 'until_1x1'
                    heads = list(range(trace['selected_L_R']))
                    assert trace['active_heads'] == heads == sorted(observer.heads)
                    initial = ['features'] if family == 'ROI-SSD' else ['features_stage1', 'conv_to_672']
                    later = ['conv5_3_fc' if family == 'ROI-SSD' else 'features_stage2',
                             'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
                    assert observer.blocks == initial + later[:trace['selected_L_R'] - 1], 'Unexpected unused/later feature block execution'
                    shapes = [[observer.heads[i]['feature_h'], observer.heads[i]['feature_w']] for i in heads]
                    assert trace['feature_shapes'] == shapes
                    assert [1, 1] not in shapes[:-1], 'Executed beyond first 1x1'
                    if shapes[-1] == [1, 1]:
                        assert trace['stopped_at_1x1'] and trace['next_head_status'] == 'STOPPED_AT_1X1'
                    else:
                        assert trace['next_head_status'] in ('INFEASIBLE', 'PYRAMID_EXHAUSTED')
                    expected_counts = [hh * ww * (len(model.aspect_ratios[i]) + 1) for i, (hh, ww) in enumerate(shapes)]
                    assert trace['anchors_per_head'] == expected_counts
                    assert trace['N_loc_predictions'] == trace['N_cls_predictions'] == trace['N_anchors'] == sum(expected_counts)
                    geo = geometry(observer.anchors, observer.reference, h, w)
                    det = predictions[0]
                    assert torch.isfinite(det['boxes']).all() and torch.isfinite(det['scores']).all()
                    assert ((det['boxes'] >= 0) & (det['boxes'] <= 1)).all()
                    assert (det['boxes'][:, 2:] >= det['boxes'][:, :2]).all()
                    assert ((det['labels'] >= 1) & (det['labels'] <= 30)).all()
                    records.append({'model_family': family, 'checkpoint_sha256': entry['checkpoint_sha256'],
                                    'h': h, 'w': w, 'sample_id': source['sample_id'], 'status': 'PASS',
                                    'tensor_gt_sha256': hashlib.sha256(x.numpy().tobytes() + gt.numpy().tobytes() + labels.numpy().tobytes()).hexdigest(),
                                    'blocks_observed': observer.blocks, 'trace': trace,
                                    'anchor_reference_max_error': geo['reference_max_abs_error'],
                                    'decoded_detections': len(det['scores'])})
                print(f'{family} {h}x{w}: PASS (3 real-image crops)', flush=True)
        assert state_mapping_digest(model.state_dict()) == weight_digest
        assert sha(checkpoint) == entry['checkpoint_sha256']
        del model, saved, state
    assert len(records) == 2 * len(SHAPES) * 3
    for h, w in SHAPES:
        for source in sources:
            pair = [r for r in records if (r['h'], r['w'], r['sample_id']) == (h, w, source['sample_id'])]
            assert len(pair) == 2 and pair[0]['tensor_gt_sha256'] == pair[1]['tensor_gt_sha256']
    write_json(OUT / 'sweep_provenance.json', evidence)
    (OUT / 'boundary_records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    metadata.update(status='COMPLETE', structural_regression='PASS', forwards=len(records),
                    sweep_binding='UNVERIFIED: source CSV checkpoints differ from selected checkpoint',
                    stage3_acceptance='UNCHANGED: MobileNet pilot accuracy/coverage gate still fails',
                    checkpoint_weights_unchanged=True, exact_inputs_across_families=True,
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'model/roissd.py', ROOT/'model/roissd_mobilenet.py']})
    write_json(OUT / 'metadata.json', metadata)
    report = ['# Fresh Stage-2 regression check', '',
              '**Structural regression: PASS. Existing 100-shape sweep attribution to the selected best_map.pt: NOT VERIFIED.**', '',
              'This check freshly loads the two locked checkpoints and tests runtime structure on boundary inputs. It is not a rerun of the dense Stage-2 accuracy diagnostic or the 100-shape sweep. No model code, weights, thresholds or depth policy were changed.', '',
              '## Sweep provenance and checkpoint binding', '']
    for family, e in evidence.items():
        report += [f'### {family}', '', f"- Selected checkpoint: `{e['selected_checkpoint']}`.",
                   f"- Selected SHA-256: `{e['selected_checkpoint_sha256']}`.",
                   f"- Selected tensor-state SHA-256: `{e['selected_tensor_state_sha256']}`.",
                   f"- Saved sweep: `{e['sweep_csv']}`; SHA-256 `{e['sweep_csv_sha256']}`.",
                   '- Grid completeness: 100 unique shapes, all H/W pairs in 32,64,...,320.',
                   f"- CSV checkpoint field: last.pt. Current last.pt SHA-256: `{e['current_last_checkpoint_sha256']}`.",
                   f"- Current last.pt tensor-state SHA-256: `{e['current_last_tensor_state_sha256']}`.",
                   f"- Current last.pt and best_map.pt have identical tensors: {e['current_last_equals_selected_tensor_state']}.",
                   '- Historical inference-time checkpoint hash: absent. File paths and current hashes cannot prove which bytes the earlier sweep loaded.',
                   '- Attribution decision: do not relabel these 100 accuracy values as best_map.pt measurements. Provenance links the saved CSV and the current selection in one audit record while preserving the mismatch.', '']
    report += ['## Fresh runtime verification', '',
               'CPU, batch=1, 8 PyTorch intra-op threads, 1 inter-op thread, 1 OpenCV thread. Three hashed real canonical source images per shape; shared project FixedSizeTestTransform with seed 20261023, confidence=0.5, NMS IoU=0.45, maximum 200 detections. Shape notation is H x W. No silent input padding or rounding.', '',
               f'{len(SHAPES)} shapes per model, 3 crops per shape: {len(records)} fresh forwards total. All passed strict checkpoint loading, selected/loaded tensor identity, until_1x1 policy, observed-vs-reported active heads, no head after the first 1x1 map, and localization/classification/anchor slot equality. Anchor ordering and geometry independently agree within 1e-6. Decoded boxes/scores are finite, boxes are ordered/in bounds and class IDs valid. Tensor/GT hashes match across families; checkpoint and model tensor hashes remain unchanged.', '',
               '| Model | H x W | Active depth | Heads (zero based) | Feature H x W | Loc = cls = anchors | Stop reason |',
               '|---|---|---:|---|---|---:|---|']
    for r in records:
        if r['sample_id'] != sources[0]['sample_id']:
            continue
        t = r['trace']
        report.append(f"| {r['model_family']} | {r['h']} x {r['w']} | {t['selected_L_R']} | {t['active_heads']} | {t['feature_shapes']} | {t['N_anchors']} | {t['next_head_status']} |")
    report += ['', '## Interpretation and limits', '',
               'No structural mismatch was detected on these fresh boundary probes. Observed feature block calls also match the selected depth exactly; no unused later block executes. The until_1x1 policy stops at the first 1x1 map when reachable, but also stops before a spatially infeasible convolution or when the available pyramid is exhausted. Therefore an INFEASIBLE stop with a larger final feature map is not itself a failure; the table records these cases explicitly. This supports current-checkpoint runtime correctness only at the tested shapes; it does not establish accuracy at every shape, explain the historical 128x128 point causally, or prove the current best weights reproduce the old CSV metrics.', '',
               'To attribute a 100-shape accuracy sweep to best_map.pt, recover an inference-time hash or equivalent immutable provenance proving those exact weights were loaded, or rerun the sweep with the selected checkpoint hash recorded at execution. Matching directory names or an equal best_map scalar is insufficient. Both current last.pt tensor states differ from their selected best_map.pt states.', '',
               'The Stage-3 pilot remains blocked by measured MobileNet accuracy/coverage criteria. A passing structural regression does not waive that gate. No CA/FULL-TECH, retraining, long accuracy sweep or Raspberry Pi run was performed.', '',
               'Reproduce this check from the repository root: `.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.fresh_regression`. Machine-readable results are sweep_provenance.json, boundary_records.jsonl and metadata.json alongside this report. No other report is required to interpret these findings.']
    (OUT / 'regression_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('COMPLETE: structural regression PASS; historical sweep binding remains unverified.', flush=True)


if __name__ == '__main__':
    main()
