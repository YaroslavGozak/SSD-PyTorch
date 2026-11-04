"""Sequential CPU revalidation launcher; experiment logic stays in existing runners."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml
from experiments.cpu_pilot.confidence import QUALITY_CONFIDENCE_FLOOR, CONFIDENCE_GRID, SELECTION_RULE, load_frozen_threshold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'final_experiments/cpu_revalidation_20261102'
PILOT_SOURCE = ROOT / 'final_experiments/cpu_pilot'
MODELS = {
    'yolo26n': ('yolo26n', 'trained_models/best-imagenet-vid-yolo/best.pt',
                '5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027'),
    'roi_ssd_vgg16': ('roi_ssd', 'trained_models/imagenet-vid-roissd-20261028/stage3/best_map.pt',
                      'f8f49132e08a7c5bf3744ce32e46e1b0fed6595aabd9cf59cf68699569503f76'),
    'roi_ssd_mobilenet': ('roi_ssd_mobilenet', 'trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt',
                          '1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804'),
}
ENV = {'CUDA_VISIBLE_DEVICES': '', 'SSD_INFERENCE_DEVICE': 'cpu',
       'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '1',
       'NUMEXPR_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE', 'MKL_DYNAMIC': 'FALSE',
       'PYTHONHASHSEED': '0'}
ACCURACY_MODES = ('full-frame', 'pad-loop', 'fixed-size-loop')
ROI_SSD_CONFIDENCE_VALUES = (0.01, 0.05, 0.1, 0.25, 0.5)
DIAGNOSTIC_MODEL = 'roi_ssd_vgg16'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def confidence_config(config, role, floor=QUALITY_CONFIDENCE_FLOOR):
    cfg = copy.deepcopy(config)
    cfg['confidence_protocol'] = {'role': role, 'quality_confidence_floor': QUALITY_CONFIDENCE_FLOOR}
    if role == 'quality':
        cfg['train_params']['infer_conf_threshold'] = floor
        if 'model_params' in cfg:
            cfg['model_params']['low_score_threshold'] = floor
        cfg['audit_protocol']['confidence'] = floor
    else:
        cfg['confidence_protocol'].update(selection_rule=SELECTION_RULE, candidate_grid=list(CONFIDENCE_GRID),
                                          selection_required=True)
    return cfg


def confidence_tag(value):
    return 'conf_' + format(value, 'g').replace('.', 'p')


def prepare_confidence_diagnostic(output):
    directory = output / 'configs/confidence_diagnostic' / DIAGNOSTIC_MODEL
    directory.mkdir(parents=True, exist_ok=False)
    source = yaml.safe_load((output / 'configs' / f'{DIAGNOSTIC_MODEL}.yaml').read_text(encoding='utf-8'))
    for value in ROI_SSD_CONFIDENCE_VALUES:
        cfg = confidence_config(source, 'quality', floor=value)
        cfg['confidence_protocol'] = {'role': 'confidence_diagnostic', 'threshold': value,
                                      'reference_quality_confidence_floor': QUALITY_CONFIDENCE_FLOOR}
        (directory / f'{confidence_tag(value)}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')


def prepare(output):
    if (output / 'session.json').exists() or any((output / d).exists() for d in ('configs', 'accuracy_sweeps', 'confidence_diagnostics', 'pilot')):
        raise FileExistsError('Preparation requires a new root (documentation files may exist)')
    manifest = (ROOT / 'final_experiments/FINAL_EXPERIMENTS_MANIFEST.md').read_text(encoding='utf-8')
    configs = {}
    checkpoints = {}
    for name, (config_name, weights, expected) in MODELS.items():
        if weights not in manifest.replace('../', '') or expected not in manifest:
            raise ValueError(f'Selection differs from FINAL_EXPERIMENTS_MANIFEST.md: {name}')
        if sha(ROOT / weights) != expected:
            raise ValueError(f'Checkpoint SHA-256 mismatch: {weights}')
        source = ROOT / 'final_experiments/configs' / f'{config_name}.yaml'
        cfg = yaml.safe_load(source.read_text(encoding='utf-8'))
        train = cfg['train_params']
        if train['device'] != 'cpu' or cfg['audit_protocol']['checkpoint_sha256'] != expected:
            raise ValueError(f'Config identity/device mismatch: {source}')
        selected_path = train.get('yolo_weights') if name == 'yolo26n' else f"trained_models/{train['task_name']}/{train['ckpt_name']}"
        if selected_path != weights:
            raise ValueError(f'Config checkpoint path mismatch: {name}')
        configs[name] = (source, cfg)
        checkpoints[name] = {'path': weights, 'sha256': expected}
    split = read_json(PILOT_SOURCE / 'split_metadata.json')
    manifests = ('pilot_manifest.txt', 'final_eval_manifest_candidate.txt')
    for filename, key in zip(manifests, ('pilot_manifest_sha256', 'final_eval_manifest_candidate_sha256')):
        if sha(PILOT_SOURCE / filename) != split['provenance'][key]:
            raise ValueError(f'Frozen pilot manifest changed: {filename}')
    dataset_audit = read_json(ROOT / 'final_experiments/01_audit/dataset_audit.json')
    dataset_yaml = Path(next(iter(configs.values()))[1]['dataset_params']['yolo_dataset_yaml'])
    if sha(dataset_yaml) != dataset_audit['yaml_sha256']:
        raise ValueError('Dataset YAML differs from the canonical audit')
    for _source, cfg in configs.values():
        if Path(cfg['dataset_params']['yolo_dataset_yaml']) != dataset_yaml:
            raise ValueError('Models do not share the same dataset YAML')
    pilot_videos = {r['video_id'] for r in read_tsv(PILOT_SOURCE / manifests[0])}
    final_videos = {r['video_id'] for r in read_tsv(PILOT_SOURCE / manifests[1])}
    if pilot_videos & final_videos:
        raise ValueError('Pilot/final video overlap')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'configs').mkdir()
    (output / 'manifests').mkdir()
    config_info = {}
    for name, (source, cfg) in configs.items():
        snapshot = output / 'configs' / f'{name}.yaml'
        # Keep selected source configs intact; quality snapshots use the common score floor.
        cfg['audit_protocol']['canonical_manifest'] = 'final_experiments/01_audit/canonical_validation.jsonl'
        cfg['audit_protocol']['final_manifest'] = 'final_experiments/01_audit/val_manifest.jsonl'
        snapshot.write_text(yaml.safe_dump(confidence_config(cfg, 'quality'), sort_keys=False), encoding='utf-8')
        config_info[name] = {'source': source.relative_to(ROOT).as_posix(),
                             'snapshot': snapshot.relative_to(ROOT).as_posix(), 'sha256': sha(snapshot)}
    prepare_confidence_diagnostic(output)
    for filename in manifests:
        shutil.copyfile(PILOT_SOURCE / filename, output / 'manifests' / filename)
    for filename in ('val_manifest.jsonl', 'canonical_validation.jsonl'):
        shutil.copyfile(ROOT / 'final_experiments/01_audit' / filename, output / 'manifests' / filename)
    for name in MODELS:
        directory = output / 'pilot' / name
        (directory / 'configs').mkdir(parents=True)
        pilot_config_info = {}
        for model, (source, cfg) in configs.items():
            target = directory / 'configs' / f'{model}.yaml'
            target.write_text(yaml.safe_dump(confidence_config(cfg, 'operational'), sort_keys=False), encoding='utf-8')
            pilot_config_info[model] = {'source': source.relative_to(ROOT).as_posix(),
                                       'snapshot': target.relative_to(ROOT).as_posix(), 'sha256': sha(target)}
        for filename in manifests:
            shutil.copyfile(output / 'manifests' / filename, directory / filename)
        provenance = dict(split['provenance'], selected_checkpoints=checkpoints, model_configs=pilot_config_info)
        write_json(directory / 'split_metadata.json', dict(split, provenance=provenance,
                   diagnostic_on_padding_gate_failure=True, operational_confidence_selection_required=True))
        (directory / 'pilot_report.md').write_text(
            f'# CPU revalidation: {name}\n\n## Preparation status\n\nPREPARED_NOT_BENCHMARKED\n\n'
            '## Locked split\n\nFrozen manifests copied byte-for-byte from final_experiments/cpu_pilot.\n'
            'Operational confidence must be selected and frozen from the pilot before other stages.\n'
            'If no padding passes, PADDING_GATE_FAILED remains diagnostic throughout.\n', encoding='utf-8')
    source_paths = ['tools/infer.py', 'tools/helpers/pipeline.py', 'model/model_adapters.py',
                    'model/roissd.py', 'model/roissd_mobilenet.py',
                    'experiments/cpu_pilot/run_pilot.py', 'experiments/cpu_revalidation.py',
                    'experiments/cpu_pilot/confidence.py', 'experiments/cpu_pilot/oracle_diagnosis.py',
                    'dataset/transforms/fixed_size_test_transform.py',
                    'dataset/transforms/fixed_padding_roi_crop_test_transform.py',
                    'dataset/transforms/fixed_padding_roi_crop_yolo_test_transform.py',
                    'transformers/fixed_roi_crop.py', 'tools/benchmarks/video_artifacts.py',
                    'final_experiments/FINAL_EXPERIMENTS_MANIFEST.md',
                    'final_experiments/01_audit/dataset_audit.json']
    locked = {p.relative_to(ROOT).as_posix(): sha(p) for p in (output / 'manifests').iterdir()}
    locked.update({p: sha(ROOT / p) for p in source_paths})
    locked.update({info['snapshot']: info['sha256'] for info in config_info.values()})
    locked.update({p.relative_to(ROOT).as_posix(): sha(p) for p in (output / 'configs/confidence_diagnostic').rglob('*.yaml')})
    for p in (output / 'pilot').rglob('*'):
        if p.is_file() and p.name != 'pilot_report.md':
            locked[p.relative_to(ROOT).as_posix()] = sha(p)
    locked[str(dataset_yaml)] = sha(dataset_yaml)
    session = {'status': 'PREPARED_NOT_BENCHMARKED', 'identifier': output.name,
               'models': checkpoints, 'configs': config_info, 'locked_files': locked,
               'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'python': sys.executable, 'environment': ENV, 'pilot_source': str(PILOT_SOURCE),
               'torch_threads': 8, 'torch_interop_threads': 1, 'opencv_threads': 1,
               'dataset_root': dataset_audit['root'], 'class_names': dataset_audit['names'],
               'accuracy_modes': list(ACCURACY_MODES), 'quality_confidence_floor': QUALITY_CONFIDENCE_FLOOR,
               'operational_confidence_rule': SELECTION_RULE, 'operational_confidence_grid': list(CONFIDENCE_GRID),
               'roi_ssd_confidence_diagnostic': {'model': DIAGNOSTIC_MODEL, 'thresholds': list(ROI_SSD_CONFIDENCE_VALUES)}}
    write_json(output / 'session.json', session)
    lines = ['# CPU revalidation experiment manifest', '', 'PREPARED_NOT_BENCHMARKED; no measured results.', '',
             f"Code commit: `{session['git_commit']}`; working tree includes uncommitted changes, source hashes below.",
             f'Python: `{sys.executable}`. Device CPU; torch intra/inter-op=8/1; OpenCV=1.',
             f'Environment: `{json.dumps(ENV)}`.', '',
             'Dataset: canonical sampled ImageNet-VID. Accuracy loops use the full frozen 5,064-frame val_manifest.jsonl;',
             'canonical_validation.jsonl (256 selection frames) is retained as provenance, not substituted for the full split.',
             'Pilot/final candidate are copied unchanged from final_experiments/cpu_pilot; video overlap=0.',
             '30 foreground classes in the exact FINAL_EXPERIMENTS_MANIFEST.md order; 27 supported by GT.',
             'SSD: 31 outputs, background 0, foreground i -> i+1. YOLO: no background; adapter adds 1.', '',
             'Full-frame accuracy: all 5,064 validation frames, full scene resized to the configured 320x320 input; no ROI crop.',
             'All quality evaluations: conf_floor=0.01 for all models; existing AP confidence ranking is unchanged.',
             f'Separate ROI-SSD/VGG16 full-validation confidence diagnostic: {ROI_SSD_CONFIDENCE_VALUES}; not operational tuning.',
             f'Operational confidence: {SELECTION_RULE}; candidate grid {CONFIDENCE_GRID}.',
             'Pilot thresholds remain UNSELECTED until --stage confidence; subsequent stages require frozen pilot decisions.',
             'Accuracy padding: 0,10,20,...,200. Fixed H/W: 32,64,96,128,160,192,224,256,288,320 (100 pairs).',
             'Pilot padding: 32,64,96,112,128,144,160. Static, hold=0, interval=10.',
             'Cost shapes: 32x32,64x64,96x96,128x128,192x192,256x256,320x320,64x128,128x64,160x320,320x160;',
             'two warmups and eight repetitions/shape. Fit uses observed network tensor areas.', '',
             '| Model | Checkpoint | SHA-256 | Config | Confidence | Config NMS IoU (effective differences: RUNBOOK) |',
             '|---|---|---|---|---:|---:|']
    for name, (_, cfg) in configs.items():
        lines.append(f"| {name} | {checkpoints[name]['path']} | {checkpoints[name]['sha256']} | {config_info[name]['snapshot']} | {QUALITY_CONFIDENCE_FLOOR} (quality); pilot-selected (operational) | 0.45 |")
    lines += ['', 'Class order (zero-based foreground): ' + ', '.join(f'{i}: {n}' for i, n in enumerate(dataset_audit['names'])),
              '', '## Locked files (SHA-256)', ''] + [f'- `{path}`: `{value}`' for path, value in locked.items()]
    (output / 'experiment_manifest.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Prepared: {output}')


def read_tsv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def check(output):
    session = read_json(output / 'session.json')
    for path, expected in session['locked_files'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Locked file changed; prepare a fresh root: {path}')
    for value in session['models'].values():
        if sha(ROOT / value['path']) != value['sha256']:
            raise ValueError(f"Locked checkpoint changed: {value['path']}")
    if 'dataset_root' in session:
        dataset_root = Path(session['dataset_root'])
        rows = [json.loads(line) for line in (output / 'manifests/val_manifest.jsonl').read_text(encoding='utf-8').splitlines() if line]
        for row in rows:
            if not (dataset_root / row['image']).is_file() or sha(dataset_root / row['label']) != row['label_sha256']:
                raise ValueError(f"Dataset image missing or label changed: {row['image']}")
    print('PASS: checkpoint/config/source/manifest hashes; CPU configuration. No inference executed.')
    return session


def accuracy_job(output, config, directory, mode):
    command = [sys.executable, '-u', '-m', 'tools.infer', '--config', str(config),
               '--evaluate', 'true', '--infer-samples', 'false', '--eval-mode', mode,
               '--device', 'cpu', '--cpu-threads', '8', '--eval-batch-size', '1',
               '--num-workers', '0', '--frame-manifest', str(output / 'manifests/val_manifest.jsonl'),
               '--audit-details', '--summary-path', str(directory), '--results-path', str(directory)]
    return directory, 'accuracy', command


def jobs(args):
    output = args.output
    result = []
    if args.stage in ('accuracy', 'all'):
        for mode in args.accuracy_modes:
            for name in args.models:
                directory = output / 'accuracy_sweeps' / name / mode.replace('-', '_')
                result.append(accuracy_job(output, output / 'configs' / f'{name}.yaml', directory,
                                           'default' if mode == 'full-frame' else mode))
    diagnostic = args.stage in ('all', 'confidence-diagnostic') or (
        args.stage == 'accuracy' and getattr(args, 'with_confidence_diagnostic', False))
    if diagnostic and DIAGNOSTIC_MODEL in args.models:
        for value in ROI_SSD_CONFIDENCE_VALUES:
            tag = confidence_tag(value)
            config = output / 'configs/confidence_diagnostic' / DIAGNOSTIC_MODEL / f'{tag}.yaml'
            directory = output / 'confidence_diagnostics' / DIAGNOSTIC_MODEL / tag
            result.append(accuracy_job(output, config, directory, 'default'))
    stages = ('confidence', 'padding', 'cost', 'pipeline') if args.stage in ('stage03', 'all') else ((args.stage,) if args.stage in ('confidence', 'smoke', 'padding', 'cost', 'pipeline') else ())
    for stage in stages:
        for name in args.models:
            directory = output / 'pilot' / name
            command = [sys.executable, '-u', '-m', 'experiments.cpu_pilot.run_pilot',
                       '--output-dir', str(directory), '--models', name, '--stage', stage]
            if stage == 'smoke':
                command += ['--max-frames', '20']
            result.append((directory, stage, command))
    return result


def validate_destinations(planned):
    for directory, stage, _command in planned:
        if directory.parent.parent.name == 'confidence_diagnostics' and (directory.parent / 'summary.csv').exists():
            raise FileExistsError('Confidence diagnostic summary already exists')
        if stage == 'accuracy' and directory.exists():
            raise FileExistsError(f'Result directory exists; use a fresh root: {directory}')
        if (directory / f'job_{stage}.json').exists():
            raise FileExistsError(f'Stage already attempted; use a fresh root: {directory}/{stage}')
        outputs = {'confidence': 'operational_confidence.json', 'padding': 'padding_sweep.csv', 'cost': 'cost_profiles_cpu.json',
                   'pipeline': 'pipeline_summary.csv', 'smoke': 'smoke_metadata.json'}
        if stage in outputs and (directory / outputs[stage]).exists():
            raise FileExistsError(f'Existing stage output: {directory / outputs[stage]}')


def execute(planned, output, session):
    environment = dict(os.environ, **ENV)
    for directory, stage, command in planned:
        frozen_hash = None
        if stage in ('smoke', 'padding', 'cost', 'pipeline'):
            cfg = yaml.safe_load((directory / 'configs' / f'{directory.name}.yaml').read_text(encoding='utf-8'))
            load_frozen_threshold(directory, directory.name, cfg)
            frozen_hash = sha(directory / 'operational_confidence.json')
            for prior_job in directory.glob('job_*.json'):
                previous_hash = read_json(prior_job).get('operational_confidence_sha256')
                if previous_hash and previous_hash != frozen_hash:
                    raise ValueError('Operational confidence changed between pipeline stages')
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / f'job_{stage}.json'
        state = {'status': 'RUNNING', 'command': command, 'environment': ENV,
                 'checkpoint_identity': session['models'], 'session_sha256': sha(output / 'session.json')}
        if frozen_hash:
            state['operational_confidence_sha256'] = frozen_hash
        with state_path.open('x', encoding='utf-8') as stream:
            json.dump(state, stream, indent=2)
        print(subprocess.list2cmdline(command), flush=True)
        try:
            with (directory / f'{stage}_execution.log').open('x', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['status'] = 'COMPLETE'
            if stage == 'confidence':
                state['operational_confidence_sha256'] = sha(directory / 'operational_confidence.json')
        except BaseException as error:
            state.update(status='FAILED', error=str(error))
            raise
        finally:
            write_json(state_path, state)


def write_csv(path, rows):
    if rows:
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)


def confidence_diagnostic_rows(output):
    directory = output / 'confidence_diagnostics' / DIAGNOSTIC_MODEL
    if not directory.exists():
        return []
    checkpoint = read_json(output / 'session.json')['models'][DIAGNOSTIC_MODEL]
    rows = []
    for value in ROI_SSD_CONFIDENCE_VALUES:
        run = directory / confidence_tag(value)
        if read_json(run / 'job_accuracy.json')['status'] != 'COMPLETE':
            raise ValueError(f'Incomplete confidence diagnostic: {run}')
        row = read_json(run / 'ssd_audit.json')
        if row['confidence_floor'] != value or row['evaluated_frames'] != row['total_frames'] or row['total_frames'] <= 0:
            raise ValueError(f'Confidence/frame coverage mismatch: {run}')
        row.update(model=DIAGNOSTIC_MODEL, confidence_threshold=value, checkpoint=checkpoint['path'],
                   checkpoint_sha256=checkpoint['sha256'], purpose='confidence_diagnostic_only')
        config = output / 'configs/confidence_diagnostic' / DIAGNOSTIC_MODEL / f'{confidence_tag(value)}.yaml'
        row.update(config=str(config), config_sha256=sha(config))
        rows.append({key: json.dumps(v) if isinstance(v, (list, dict)) else v for key, v in row.items()})
    return rows


def summarize_confidence_diagnostic(output):
    directory = output / 'confidence_diagnostics' / DIAGNOSTIC_MODEL
    if (directory / 'summary.csv').exists():
        raise FileExistsError('Confidence diagnostic summary already exists')
    rows = confidence_diagnostic_rows(output)
    if len(rows) != len(ROI_SSD_CONFIDENCE_VALUES):
        raise ValueError('All five confidence diagnostic runs are required')
    write_csv(directory / 'summary.csv', rows)


def aggregate(output, models):
    destination = output / 'combined_reports'
    if destination.exists():
        raise FileExistsError(f'Aggregate output already exists: {destination}')
    accuracy = []
    pipeline = []
    comparisons = []
    paddings = {}
    operational = []
    completeness = {}
    diagnostic = confidence_diagnostic_rows(output) if DIAGNOSTIC_MODEL in models else []
    if DIAGNOSTIC_MODEL in models:
        completeness[f'{DIAGNOSTIC_MODEL}/confidence_diagnostic'] = bool(diagnostic)
    for name in models:
        checkpoint = read_json(output / 'session.json')['models'][name]
        for mode in ('full_frame', 'pad_loop', 'fixed_size_loop'):
            directory = output / 'accuracy_sweeps' / name / mode
            details = sorted(directory.glob('*_audit.json'))
            expected_count = {'full_frame': 1, 'pad_loop': 21, 'fixed_size_loop': 100}[mode]
            if details and len(details) != expected_count:
                raise ValueError(f'Incomplete {name}/{mode}: {len(details)}/{expected_count}')
            if details and read_json(directory / 'job_accuracy.json')['status'] != 'COMPLETE':
                raise ValueError(f'Accuracy job did not complete: {directory}')
            completeness[f'{name}/{mode}'] = len(details) == expected_count
            for path in details:
                row = read_json(path)
                row.update(model=name, mode=mode, checkpoint=checkpoint['path'], checkpoint_sha256=checkpoint['sha256'])
                if mode == 'fixed_size_loop':
                    row['requested_h'], row['requested_w'] = map(int, row['evaluated_dataset'].rsplit('_', 1)[1].split('x'))
                elif mode == 'pad_loop':
                    row['padding'] = int(row['evaluated_dataset'].rsplit('_', 1)[1])
                elif row['evaluated_frames'] != row['total_frames'] or row['total_frames'] <= 0:
                    raise ValueError(f'Full-frame accuracy must evaluate every manifest frame: {path}')
                actual = row['actual_input_shapes']
                row['actual_input_h'] = actual[0]['h'] if len(actual) == 1 else ('variable; see actual_input_shapes' if actual else None)
                row['actual_input_w'] = actual[0]['w'] if len(actual) == 1 else ('variable; see actual_input_shapes' if actual else None)
                accuracy.append({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        directory = output / 'pilot' / name
        threshold_path = directory / 'operational_confidence.json'
        completeness[f'{name}/operational_confidence'] = threshold_path.exists()
        if threshold_path.exists():
            cfg = yaml.safe_load((directory / 'configs' / f'{name}.yaml').read_text(encoding='utf-8'))
            load_frozen_threshold(directory, name, cfg)
            operational.append(read_json(threshold_path))
        path = directory / 'pipeline_summary.csv'
        completeness[f'{name}/pipeline'] = path.exists()
        if path.exists():
            with path.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 4:
                raise ValueError(f'Expected all four modes: {path}')
            pipeline.extend(rows)
            frame_rows = read_csv(directory / 'frame_level_metrics.csv')
            ca = {r['frame_id']: r for r in frame_rows if r['mode'] == 'CA'}
            full = {r['frame_id']: r for r in frame_rows if r['mode'] == 'FULL-TECH'}
            if ca.keys() != full.keys():
                raise ValueError('CA/FULL-TECH frame mismatch')
            for key, row in full.items():
                paired = ca[key]
                same_shapes = json.loads(row['actual_network_shapes_hw']) == json.loads(paired['actual_network_shapes_hw'])
                same_predictions = json.loads(row['prediction_records']) == json.loads(paired['prediction_records'])
                comparisons.append({'model': name, 'frame_id': key,
                    'predicted_segment_cost': row['predicted_segment_cost'],
                    'predicted_full_frame_cost': row['predicted_full_frame_cost'],
                    'selector_decision': row['selector_decision'], 'fallback_reason': row['fallback_reason'],
                    'CA_actual_tensor_shapes': paired['actual_network_shapes_hw'],
                    'FULL_TECH_actual_tensor_shapes': row['actual_network_shapes_hw'],
                    'same_actual_tensor_shapes': same_shapes, 'same_predictions': same_predictions,
                    'interpretation': ('selector was logically active, but did not change actual NN workload because CA input was already equivalent to full-frame input; '
                                       'do not attribute timing differences to selector' if row['fallback_reason'] == 'cost_prefers_full_frame' and same_shapes and same_predictions else '')})
        decision = directory / 'padding_decision.json'
        if decision.exists():
            paddings[name] = read_json(decision)[name]
    if not accuracy and not pipeline and not diagnostic:
        raise ValueError('No new-run results to aggregate')
    destination.mkdir()
    write_csv(destination / 'accuracy_sweeps.csv', accuracy)
    write_csv(destination / 'full_frame_accuracy.csv', [row for row in accuracy if row['mode'] == 'full_frame'])
    write_csv(destination / 'roi_ssd_confidence_diagnostic.csv', diagnostic)
    write_json(destination / 'operational_confidence.json', {row['model']: row for row in operational})
    write_csv(destination / 'pipeline_summary.csv', pipeline)
    write_csv(destination / 'ca_vs_full_tech.csv', comparisons)
    write_json(destination / 'completeness.json', completeness)
    labels = {'yolo26n': 'YOLO_PADDING_FINAL', 'roi_ssd_vgg16': 'ROISSD_VGG_PADDING_FINAL',
              'roi_ssd_mobilenet': 'ROISSD_MOBILENET_PADDING_FINAL'}
    report = ['# CPU revalidation results', '', 'Complete requested scope: ' + str(all(completeness.values())), '',
              'Quality confidence floor=0.01 for all models. Pipeline uses separate frozen pilot-selected operational thresholds.',
              'Visibility/postprocessing differences are documented in RUNBOOK; YOLO pad crops remain rectangular with stride-32 padding.',
              'Coverage differs by fixed shape; compare eligible subsets explicitly.', '']
    report += [f"{labels[name]} = {d['padding_final']}; {d['status']}" for name, d in paddings.items()]
    report += ['', 'CA/FULL-TECH workload and prediction equivalence: ca_vs_full_tech.csv.',
               'Logical FF selection with identical actual tensors/predictions does not establish a speedup from selector.']
    (destination / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--stage', choices=('prepare', 'check', 'accuracy', 'stage03', 'all', 'confidence-diagnostic', 'confidence', 'smoke', 'padding', 'cost', 'pipeline', 'aggregate'), required=True)
    parser.add_argument('--models', nargs='+', choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument('--accuracy-modes', nargs='+', choices=ACCURACY_MODES, default=list(ACCURACY_MODES))
    parser.add_argument('--with-confidence-diagnostic', action='store_true',
                        help='Add five ROI-SSD/VGG16 full-frame confidence runs to --stage accuracy; included in --stage all')
    parser.add_argument('--dry-run', action='store_true', help='Validate identities and print commands; do not run experiments')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / 'final_experiments'):
        parser.error('--output must be inside final_experiments')
    if len(set(args.models)) != len(args.models) or len(set(args.accuracy_modes)) != len(args.accuracy_modes):
        parser.error('Duplicate models/modes are not allowed')
    if args.stage == 'confidence-diagnostic' and DIAGNOSTIC_MODEL not in args.models:
        parser.error('Confidence diagnostic requires --models roi_ssd_vgg16')
    if args.with_confidence_diagnostic and args.stage not in ('accuracy', 'all'):
        parser.error('--with-confidence-diagnostic requires --stage accuracy or all')
    if args.stage == 'prepare':
        if args.dry_run:
            parser.error('Use --stage check after preparation')
        prepare(args.output)
        return
    session = check(args.output)
    if args.stage == 'check':
        return
    planned = jobs(args)
    validate_destinations(planned)
    should_aggregate = args.stage == 'aggregate' or (args.stage in ('all', 'stage03') and set(args.models) == set(MODELS))
    if args.dry_run:
        for _directory, _stage, command in planned:
            print(subprocess.list2cmdline(command))
        if should_aggregate:
            print('Aggregate new-run reports after requested stages.')
        return
    if should_aggregate and (args.output / 'combined_reports').exists():
        raise FileExistsError('combined_reports already exists; use a fresh root')
    execute(planned, args.output, session)
    if any(directory.is_relative_to(args.output / 'confidence_diagnostics') for directory, _, _ in planned):
        summarize_confidence_diagnostic(args.output)
    if should_aggregate:
        aggregate(args.output, args.models)


if __name__ == '__main__':
    main()
