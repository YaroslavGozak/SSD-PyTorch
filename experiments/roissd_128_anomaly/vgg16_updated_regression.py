"""Reproduce the updated VGG16 forward failure without modifying the model.

Run: .venv/Scripts/python.exe -m experiments.roissd_128_anomaly.vgg16_updated_regression
This diagnostic fails closed: successful preflight alone never awards PASS.
"""
import copy
import csv
from datetime import datetime, timezone
import json
import platform
import subprocess
import traceback

import cv2
import torch
import yaml

from .run import ROOT, HERE, DATA, crop, prepare, sha, load_model, write_json, csv_rows
from .quick_regression_check import state_mapping_digest
from .instrumentation import ForwardObserver, geometry

OUT = ROOT / 'final_experiments/roissd_vgg16_updated_regression'
REF = ROOT / 'final_experiments/02_diagnostics_128'
SQUARES = sorted(set([64, 88, 96, 104, 120, 124, 128, 132, 136, 140,
                      144, 160, 256, 268, 300] + list(range(108, 145, 4))))
RECTANGLES = [(96, 144), (144, 96), (128, 160), (160, 128),
              (140, 192), (192, 140), (256, 320), (320, 256)]


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    settings = yaml.safe_load((HERE / 'config.yaml').read_text(encoding='utf-8'))
    lock = prepare(settings)
    entry, = [item for item in lock['models'] if item['model_family'] == 'ROI-SSD']
    assert entry['checkpoint'] in (ROOT / settings['stage1_manifest']).read_text(encoding='utf-8')
    rows = [json.loads(line) for line in (DATA / 'subset.jsonl').read_text().splitlines()]
    assert len(rows) == 256
    reference_meta = json.loads((REF / 'run_metadata.json').read_text())
    assert reference_meta['status'] == 'COMPLETE'
    assert reference_meta['subset_sha256'] == lock['subset_sha256']
    for name in ('seed', 'confidence', 'nms_iou', 'max_detections'):
        assert reference_meta[name] == settings[name], name
    # Check shared input/evaluation implementations against historical provenance.
    shared = ['tools/infer.py', 'dataset/transforms/fixed_size_test_transform.py',
              'experiments/roissd_128_anomaly/run.py',
              'experiments/roissd_128_anomaly/instrumentation.py']
    for name in shared:
        assert sha(ROOT / name) == reference_meta['source_sha256'][name], name
    source_hashes = {name: sha(ROOT / name) for name in [
        *shared, 'model/roissd.py',
        'experiments/roissd_128_anomaly/vgg16_updated_regression.py']}
    metadata = {
        'status': 'RUNNING', 'started_at': datetime.now(timezone.utc).isoformat(),
        'checkpoint': entry, 'config': yaml.safe_load((DATA / entry['config_snapshot']).read_text()),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_tracked_diff': subprocess.check_output(['git', 'diff', '--stat'], cwd=ROOT, text=True),
        'source_sha256': source_hashes, 'subset_sha256': lock['subset_sha256'],
        'seed': settings['seed'], 'confidence': settings['confidence'],
        'nms_iou': settings['nms_iou'], 'max_detections': settings['max_detections'],
        'metric_tolerance': 1e-6, 'device': 'cpu', 'cpu_threads': 4,
        'torch': torch.__version__, 'platform': platform.platform(),
        'reference_git_revision': reference_meta['git_commit'],
        'reference_artifact_sha256': {name: sha(REF / name) for name in [
            'run_metadata.json', 'size_sweep_metrics.csv', 'internal_shapes.csv', 'anchor_matching.csv']},
        'scope': 'Regression diagnosis only; no model/config/threshold/anchor changes',
    }
    # Verify the entire frozen subset even though a failed first forward stops accuracy evaluation.
    for row in rows:
        assert sha(__import__('pathlib').Path(lock['dataset_root']) / row['image']) == row['image_sha256']
        assert sha(__import__('pathlib').Path(lock['dataset_root']) / row['label']) == row['label_sha256']
    image = cv2.imread(str(__import__('pathlib').Path(lock['dataset_root']) / rows[0]['image']))
    assert image is not None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    state = torch.load(ROOT / entry['checkpoint'], map_location='cpu', weights_only=True)
    state = state.get('model', state)
    model = load_model(entry, settings, torch.device('cpu'))
    incompatible = model.load_state_dict(state, strict=True)
    before = state_mapping_digest(model.state_dict())
    assert before == state_mapping_digest(state)
    compatibility = {
        'strict': True, 'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
        'shape_mismatches': {k: [list(v.shape), list(model.state_dict()[k].shape)]
                             for k, v in state.items() if v.shape != model.state_dict()[k].shape},
        'checkpoint_tensor_sha256': state_mapping_digest(state),
        'loaded_state_sha256': before, 'weights_match': True,
    }
    structures = []
    first_traceback = None
    with torch.inference_mode():
        for shape in [(n, n) for n in SQUARES] + RECTANGLES + [(320, 320)]:
            mode = 'full_frame_resize' if shape == (320, 320) else 'object_centered'
            tensor = crop(image, rows[0], shape, settings['seed'])[0]
            if mode == 'full_frame_resize':
                pix = torch.from_numpy(cv2.resize(image, (320, 320))).permute(2, 0, 1).float() / 255
                tensor = ((pix - torch.tensor([.485, .456, .406])[:, None, None]) /
                          torch.tensor([.229, .224, .225])[:, None, None]).unsqueeze(0)
            result = {'mode': mode, 'input_h': shape[0], 'input_w': shape[1],
                      'status': 'FAIL', 'active_depth': 0, 'active_heads': [],
                      'feature_shapes': [], 'final_feature_shape': None,
                      'anchors_per_head': None, 'total_anchors': None,
                      'N_loc_predictions': None, 'N_cls_predictions': None,
                      'slot_counts_equal': None}
            with ForwardObserver(model) as observer:
                try:
                    model(tensor, None)
                    geometry(observer.anchors, observer.reference, *shape)
                    result.update(status='PREFLIGHT_ONLY', active_depth=len(observer.heads),
                                  active_heads=sorted(observer.heads), feature_shapes=observer.feature_shapes,
                                  final_feature_shape=observer.feature_shapes[-1],
                                  anchors_per_head=[h['cls_slots'] for h in observer.heads.values()],
                                  total_anchors=len(observer.anchors), N_loc_predictions=observer.loc_shape[1],
                                  N_cls_predictions=observer.cls_shape[1], slot_counts_equal=True)
                except Exception as error:
                    result['error'] = f'{type(error).__name__}: {error}'
                    if first_traceback is None:
                        first_traceback = traceback.format_exc()
                result['executed_blocks'] = observer.blocks
            structures.append(result)
            print(f'{shape}: {result["status"]} {result.get("error", "")}', flush=True)
    csv_rows(OUT / 'structural_checks.csv', structures)
    old_metrics = {int(r['requested_h']): r for r in read_csv(REF / 'size_sweep_metrics.csv')
                   if r['model_family'] == 'ROI-SSD'}
    old_heads = [r for r in read_csv(REF / 'internal_shapes.csv')
                 if r['model_family'] == 'ROI-SSD' and r['head_active'] == 'True']
    old_matching = {int(r['requested_h']): r for r in read_csv(REF / 'anchor_matching.csv')
                    if r['model_family'] == 'ROI-SSD'}
    comparison = []
    for n in SQUARES:
        old = old_metrics.get(n, {})
        heads = [r for r in old_heads if int(r['requested_h']) == n]
        match = old_matching.get(n, {})
        comparison.append({
            'mode': 'object_centered', 'input_h': n, 'input_w': n,
            'old_depth': heads[0]['active_backbone_depth'] if heads else None,
            'old_map50': old.get('map50'), 'old_recall50': old.get('recall50'),
            'old_mean_raw_max_foreground_confidence': old.get('max_confidence_mean'),
            'old_mean_best_anchor_iou': match.get('max_anchor_iou_mean'),
            'old_fraction_gt_best_anchor_iou_ge_05': match.get('fraction_gt_iou_ge_05'),
            'old_returned_detections': round(float(old['predictions_post_nms_mean']) * 256) if old else None,
            'new_depth': None, 'new_map50': None, 'new_recall50': None,
            'delta_mAP50': None, 'delta_Recall50': None,
            'new_mean_raw_max_foreground_confidence': None, 'new_mean_best_anchor_iou': None,
            'new_fraction_gt_best_anchor_iou_ge_05': None, 'new_returned_detections': None,
            'final_feature_shape': None, 'new_status': 'UNAVAILABLE_FORWARD_FAILURE',
            'reference': '02_diagnostics_128/size_sweep_metrics.csv' if old else 'NO_SAVED_REFERENCE',
        })
    comparison.append({'mode': 'full_frame_resize', 'input_h': 320, 'input_w': 320,
                       'new_status': 'UNAVAILABLE_FORWARD_FAILURE',
                       'reference': 'No saved same-checkpoint full-frame reference asserted'})
    csv_rows(OUT / 'accuracy_comparison.csv', comparison)
    csv_rows(OUT / 'latency_sanity.csv', [
        {'input_h': n, 'input_w': n, 'median_latency_ms': None, 'p95_latency_ms': None,
         'active_depth': None, 'feature_shapes': None, 'status': 'UNAVAILABLE_FORWARD_FAILURE',
         'reason': 'Failed forwards are not valid inference latency samples'}
        for n in [96, 128, 140, 256, 300]])
    compatibility['weights_unchanged_after_attempts'] = state_mapping_digest(model.state_dict()) == before
    compatibility['finite_inference'] = 'UNAVAILABLE_FORWARD_FAILURE'
    (OUT / 'checkpoint_compatibility.txt').write_text(
        json.dumps({'checkpoint': entry, **compatibility}, indent=2) + '\n', encoding='utf-8')
    failed = sum(r['status'] == 'FAIL' for r in structures)
    metadata.update(status='FAIL' if failed else 'INCOMPLETE',
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    checkpoint_compatibility=compatibility,
                    structural_attempts=len(structures), structural_failures=failed,
                    verified_subset_frames=len(rows), accuracy_samples_evaluated=0,
                    first_traceback=first_traceback,
                    accuracy_status='UNAVAILABLE_FORWARD_FAILURE',
                    latency_status='UNAVAILABLE_FORWARD_FAILURE')
    report = [
        '# Updated ROI-SSD VGG16 regression check', '', f'**Result: {metadata["status"]}**', '',
        f'Checkpoint: `{entry["checkpoint"]}`.', f'SHA256: `{entry["checkpoint_sha256"]}`.', '',
        'The existing checkpoint loads with `strict=True`, no missing/unexpected keys or shape mismatches. '
        'Every loaded tensor matches the checkpoint and remains unchanged after all attempted forwards.', '',
        f'All {failed}/{len(structures)} attempted forwards fail before any feature block or detection head runs. '
        '`RoiSSD._iter_feature_levels()` calls `self.features_stage1(x)`, but VGG16 defines `self.features`. '
        'The new iterator also references MobileNet-only `conv_to_672` and `features_stage2`. '
        'The constructor does not initialize `record_execution_trace` or `last_execution_trace` either. '
        'These are source findings; the observed first exception is shown below.', '',
        '```text', first_traceback or 'No forward exception', '```', '',
        'All requested squares and rectangles were attempted, plus 108/112/116 for the unstable range '
        'and a real full-frame image resized to 320x320. The frozen 256-frame image and label hashes, '
        'checkpoint/config/subset hashes, seed, thresholds and shared preprocessing/evaluator source '
        'were verified against the saved Stage 2 provenance.', '',
        '## Accuracy and unstable range', '',
        'Exact historical CSV values are used below. New metrics and deltas are unavailable, not zero: '
        'the first forward fails independently of image contents. The 256-frame accuracy sweep and CPU '
        'latency repetitions cannot produce meaningful results. No claim of preserved full-frame accuracy, '
        'anchor alignment, finite predictions, early exit, or reduced 108–136 collapse can be made.', '',
        '| Input | Old depth | New depth | Old mAP50 | New mAP50 | ΔmAP50 | Final feature shape |',
        '|---:|---:|---|---:|---|---|---|',
    ]
    for row in comparison:
        if row['mode'] == 'object_centered' and 108 <= row['input_h'] <= 144:
            report.append(f'| {row["input_h"]} | {row["old_depth"]} | unavailable | '
                          f'{float(row["old_map50"]):.9f} | unavailable | unavailable | unavailable |')
    report += ['', '## Decision', '',
               'Checkpoint compatibility: PASS. Structural/inference checks: FAIL. Accuracy, full-frame '
               'non-regression and latency: NOT EVALUABLE. Overall regression-safe decision: **FAIL**. '
               'The model was not modified, retrained or retuned. Correcting the forward is outside this task. '
               'Rerun complete accuracy and timing checks after a separately authorized fix.', '',
               'Reproduce: `.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.vgg16_updated_regression`. '
               'This failure-diagnostic runner does not award PASS if the forward is subsequently repaired.']
    (OUT / 'regression_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    assert all(sha(ROOT / name) == digest for name, digest in source_hashes.items())
    metadata['output_sha256'] = {name: sha(OUT / name) for name in [
        'regression_report.md', 'checkpoint_compatibility.txt', 'accuracy_comparison.csv',
        'structural_checks.csv', 'latency_sanity.csv']}
    write_json(OUT / 'metadata.json', metadata)
    print(f'{metadata["status"]}: {OUT}', flush=True)
    return 1 if failed else 2


if __name__ == '__main__':
    raise SystemExit(main())
