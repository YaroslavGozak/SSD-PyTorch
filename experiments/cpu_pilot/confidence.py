"""Pilot-only operational confidence selection; reuse the existing FF pipeline and matcher."""
import csv
import hashlib
import json
from pathlib import Path

QUALITY_CONFIDENCE_FLOOR = 0.01
CONFIDENCE_GRID = (0.01,) + tuple(value / 100 for value in range(5, 100, 5))
SELECTION_RULE = 'maximum micro F1@IoU=0.5 on full-frame pilot; ties: higher recall, then lower threshold'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_threshold(frame_rows):
    from .oracle_diagnosis import match_objects

    curves = []
    records = [(json.loads(row['prediction_records']), json.loads(row['ground_truth_records'])) for row in frame_rows]
    total_gt = sum(len(boxes) for _, gt in records for boxes in gt.values())
    if total_gt == 0:
        raise ValueError('Operational confidence cannot be selected without pilot GT')
    for threshold in CONFIDENCE_GRID:
        tp = fp = 0
        for predictions, gt in records:
            counts = match_objects([det for det in predictions if det['confidence'] >= threshold], gt)
            tp += counts['tp']
            fp += counts['fp']
        fn = total_gt - tp
        curves.append(dict(threshold=threshold, tp=tp, fp=fp, fn=fn,
                           precision=tp / (tp + fp) if tp + fp else 0., recall=tp / total_gt,
                           f1=2 * tp / (2 * tp + fp + fn)))
    selected = max(curves, key=lambda row: (row['f1'], row['recall'], -row['threshold']))
    if selected['tp'] == 0:
        raise ValueError('No operational confidence candidate detects any pilot GT')
    return selected, curves


def load_frozen_threshold(directory, model_name, config):
    path = directory / 'operational_confidence.json'
    receipt = directory / 'operational_confidence.sha256'
    if not path.is_file() or not receipt.is_file():
        raise FileNotFoundError('Select and freeze operational confidence on pilot first: --stage confidence')
    if digest(path) != receipt.read_text(encoding='utf-8').strip():
        raise ValueError('Frozen operational confidence was modified')
    decision = json.loads(path.read_text(encoding='utf-8'))
    expected = {'model': model_name, 'checkpoint_sha256': config['audit_protocol']['checkpoint_sha256'],
                'pilot_manifest_sha256': digest(directory / 'pilot_manifest.txt'),
                'config_sha256': digest(directory / 'configs' / f'{model_name}.yaml'),
                'selection_rule': SELECTION_RULE}
    if any(decision.get(key) != value for key, value in expected.items()):
        raise ValueError('Operational confidence identity/protocol mismatch')
    for filename, expected_hash in decision['evidence_sha256'].items():
        if filename not in ('confidence_candidates.csv', 'confidence_frame_metrics.csv'):
            raise ValueError('Unexpected confidence evidence file')
        if digest(directory / filename) != expected_hash:
            raise ValueError('Operational confidence pilot evidence changed')
    with (directory / 'confidence_candidates.csv').open(encoding='utf-8', newline='') as stream:
        candidates = list(csv.DictReader(stream))
    best = max(candidates, key=lambda row: (float(row['f1']), float(row['recall']), -float(row['threshold'])))
    threshold = float(decision['threshold'])
    if threshold != float(best['threshold']) or threshold not in CONFIDENCE_GRID:
        raise ValueError('Frozen threshold differs from the pilot selection')
    return threshold


def run_selection(split, manifest_rows, device):
    from . import run_pilot as pilot

    directory = pilot.PILOT_DIR
    for name in pilot.MODEL_NAMES:
        if (directory / 'operational_confidence.json').exists():
            raise FileExistsError('Operational confidence is already frozen; use a fresh session')
        model, dataset, config, floor = pilot._load_model(name, device, confidence_floor=QUALITY_CONFIDENCE_FLOOR)
        indices = pilot._dataset_indices(dataset, manifest_rows, pilot._dataset_root(config))
        summary, rows = pilot._run_pipeline(name, model, dataset, config, floor, manifest_rows, indices,
                                            'confidence_full_frame_floor', 'FF', 0)
        selected, curve = select_threshold(rows)
        summary['confidence_role'] = 'pilot_selection_floor'
        pilot._write_csv(directory / 'confidence_candidates.csv', curve)
        pilot._write_csv(directory / 'confidence_frame_metrics.csv', rows)
        pilot._write_csv(directory / 'confidence_ff_summary.csv', [summary])
        decision = dict(selected, model=name, selection_rule=SELECTION_RULE,
                        status='FROZEN_FROM_PILOT', inference_floor=QUALITY_CONFIDENCE_FLOOR,
                        candidate_grid=list(CONFIDENCE_GRID), pilot_frames=len(rows),
                        pilot_manifest_sha256=split['provenance']['pilot_manifest_sha256'],
                        checkpoint_sha256=config['audit_protocol']['checkpoint_sha256'],
                        config_sha256=digest(directory / 'configs' / f'{name}.yaml'),
                        evidence_sha256={filename: digest(directory / filename) for filename in
                                         ('confidence_candidates.csv', 'confidence_frame_metrics.csv')})
        path = directory / 'operational_confidence.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(decision, stream, indent=2)
        with (directory / 'operational_confidence.sha256').open('x', encoding='utf-8') as stream:
            stream.write(digest(path) + '\n')
        del model, dataset
