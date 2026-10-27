"""Check the manifest's historical full-frame VGG checkpoint as a second old checkpoint."""
import json
from pathlib import Path

import cv2
import torch
import yaml

from model.roissd import RoiSSD
from .run import ROOT, DATA, sha, write_json
from .vgg16_fixed_regression import OUT, legacy_class, build, full_frame, evaluate
from .quick_regression_check import state_mapping_digest


def finalize_report():
    from .vgg16_updated_regression import read_csv
    result = json.loads((OUT / 'fullframe_checkpoint_regression.json').read_text())
    assert result['status'] == 'PASS'
    meta_path = OUT / 'metadata.json'
    meta = json.loads(meta_path.read_text())
    assert meta['status'] == 'PASS'
    assert result['current_source_sha256'] == sha(ROOT / 'model/roissd.py')
    report_path = OUT / 'regression_report.md'
    marker = '\n## Additional checkpoint and runtime verification\n'
    report = report_path.read_text(encoding='utf-8').split(marker)[0]
    extra = [marker, '', f'Second checkpoint: `{result["checkpoint"]}`. Strict load and unchanged weights: '
             'PASS. All 256 full-frame prediction arrays match the old source within 1e-6. '
             f'mAP50 {result["new_metrics"]["map50"]:.9f}, Recall50 '
             f'{result["new_metrics"]["recall50"]:.9f}; both deltas are zero. '
             'These paired results use the same checkpoint/protocol; the manifest historical canonical '
             'score additionally used its audit postprocessing and is not asserted to match bit for bit.', '',
             'The known 108–136 px accuracy collapse is unchanged, not repaired by this forward fix. '
             'At 140 px the feasible fifth level improves mAP50 from 0.620585258 to 0.664639423. '
             'Heads retain their original parameters, scales, aspect ratios and class ordering.', '',
             'The initial sequential CPU timing is retained in `latency_sanity.csv`. '
             'Interleaved timing below alternates model order, disables optional tracing for production '
             'behavior, and uses 5 warmups + 30 repeats. It reduces ordering and thermal bias; '
             'p95 still has short-run noise.', '',
             '| Input | Old median ms | New median ms | New/old | Old p95 ms | New p95 ms |',
             '|---:|---:|---:|---:|---:|---:|']
    for row in read_csv(OUT / 'latency_interleaved.csv'):
        extra.append(f'| {row["input_h"]} | {float(row["old_median_latency_ms"]):.3f} | '
                     f'{float(row["new_median_latency_ms"]):.3f} | '
                     f'{float(row["median_ratio_new_old"]):.3f} | '
                     f'{float(row["old_p95_latency_ms"]):.3f} | '
                     f'{float(row["new_p95_latency_ms"]):.3f} |')
    report_path.write_text(report+'\n'.join(extra)+'\n', encoding='utf-8')
    meta['additional_checkpoint'] = result
    meta['interleaved_cpu_timing'] = {'warmups': 5, 'repeats': 30, 'threads': 4,
                                     'trace_enabled': False, 'file': 'latency_interleaved.csv'}
    for path in ('experiments/roissd_128_anomaly/vgg16_latency_regression.py',
                 'experiments/roissd_128_anomaly/vgg16_fullframe_checkpoint_regression.py',
                 'experiments/roissd_128_anomaly/tests/test_vgg16_runtime.py'):
        meta['source_sha256'][path] = sha(ROOT / path)
    for path in ('regression_report.md', 'fullframe_checkpoint_regression.json', 'latency_interleaved.csv'):
        meta['output_sha256'][path] = sha(OUT / path)
    write_json(meta_path, meta)


def main():
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = 'trained_models/imagenet-vid-roissd-20261002/best_map_stage2.pt'
    expected_sha = 'e6f15e01032ce2f2d509f72f96efff6b0ed447d125b858fb099e44e4e23f79f3'
    assert checkpoint.split('/')[-2] in (ROOT / 'final_experiments/FINAL_EXPERIMENTS_MANIFEST.md').read_text()
    assert sha(ROOT / checkpoint) == expected_sha
    cfg_path = ROOT / 'final_experiments/configs/roi_ssd.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    state = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=True)
    state = state.get('model', state)
    old_cls, old_sha = legacy_class()
    old = build(old_cls, cfg['model_params'], state, device)
    new = build(RoiSSD, cfg['model_params'], state, device, record_execution_trace=True)
    digest = state_mapping_digest(state)
    lock = json.loads((DATA / 'selection_lock.json').read_text())
    assert sha(DATA / 'subset.jsonl') == lock['subset_sha256']
    rows = [json.loads(line) for line in (DATA / 'subset.jsonl').read_text().splitlines()]
    assert len(rows) == 256
    samples = []
    for row in rows:
        path = Path(lock['dataset_root']) / row['image']
        assert sha(path) == row['image_sha256']
        samples.append(full_frame(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), row))
    old_metrics, old_outputs = evaluate(old, samples, lock['class_names'], device)
    new_metrics, new_outputs = evaluate(new, samples, lock['class_names'], device)
    for a, b in zip(old_outputs, new_outputs):
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=0)
    assert state_mapping_digest(old.state_dict()) == state_mapping_digest(new.state_dict()) == digest
    assert all(abs(old_metrics[k]-new_metrics[k]) <= 1e-6 for k in ('map50', 'recall50'))
    result = {'status': 'PASS', 'checkpoint': checkpoint, 'checkpoint_sha256': expected_sha,
              'config': str(cfg_path.relative_to(ROOT)), 'config_sha256': sha(cfg_path),
              'strict': True, 'missing_keys': [], 'unexpected_keys': [], 'shape_mismatches': {},
              'weights_unchanged': True, 'prediction_parity': True, 'num_samples': 256,
              'input_shape': [320, 320], 'confidence': cfg['model_params']['low_score_threshold'],
              'subset_sha256': lock['subset_sha256'], 'old_metrics': old_metrics, 'new_metrics': new_metrics,
              'delta_mAP50': new_metrics['map50']-old_metrics['map50'],
              'delta_Recall50': new_metrics['recall50']-old_metrics['recall50'],
              'legacy_source_sha256': old_sha, 'current_source_sha256': sha(ROOT / 'model/roissd.py')}
    write_json(OUT / 'fullframe_checkpoint_regression.json', result)
    if (OUT / 'latency_interleaved.csv').exists():
        finalize_report()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
