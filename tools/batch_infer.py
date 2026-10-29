"""Recursively evaluate trained checkpoints with both ROI evaluation sweeps.

Run from repository root: python -m tools.batch_infer --dry-run
Then: python -m tools.batch_infer --resume
Optional --config-map YAML maps root-relative checkpoint or directory paths to
config YAML paths. The most specific matching entry wins. Auto-resolution is
limited to the repository's known model layouts; ambiguous models are reported.

Known VOC/VID datasets reuse weights and annotation metadata within each sweep;
--no-reuse restores per-setting loading. Data loading remains single-process
and batch size remains one unless explicitly changed. For opt-in throughput:
  python -m tools.batch_infer --num-workers 2 --pin-memory --fixed-size-batch-size 4
Larger batches report amortized detector time per image, not single-image latency.
Worker processes can change the random crop sequence. Use a separate --output
when comparing execution settings. Resume preserves completed results even when
worker count, pinning, batch size, or model reuse changes.
"""
import argparse
import copy
import hashlib
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

import torch
import yaml
from tools.helpers.config_reader import load_config
from tools.helpers.evaluation_results import read_completed_results

ROOT = Path(__file__).resolve().parents[1]
MODES = ('pad-loop', 'fixed-size-loop')


def mode_completed(marker, identity):
    """Match checkpoint/config while allowing execution settings to change."""
    try:
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if not isinstance(saved, dict):
            return False
        return ({key: value for key, value in saved.items() if key != 'execution'} ==
                {key: value for key, value in identity.items() if key != 'execution'})
    except (OSError, ValueError):
        # Missing or interrupted marker writes must cause a fresh evaluation.
        return False


def sweep_summary(mode, model):
    if mode == 'pad-loop':
        prefix = 'fixed_padding_roi_crop_yolo_' if model == 'yolo' else 'fixed_padding_roi_crop_'
        return 'results.csv', {prefix + str(pad) for pad in range(0, 201, 10)}
    prefix = 'fixed_size_yolo_' if model == 'yolo' else 'fixed_size_'
    return 'fixed_size_results_v2.csv', {f'{prefix}{h}x{w}' for h in range(32, 321, 32)
                                    for w in range(32, 321, 32)}


def recover_summary_rows(destination, mode, identity, config_text):
    """Adopt legacy CSVs only with matching saved config and checkpoint age."""
    config = yaml.safe_load(identity['config'])
    name, expected = sweep_summary(mode, config['train_params']['model'])
    summary = destination/mode/name
    if not summary.exists():
        return []
    markers = [destination/f'{mode}.{kind}.json' for kind in ('progress', 'complete')]
    marker = next((path for path in markers if path.exists()), None)
    if marker is not None:
        if not mode_completed(marker, identity):
            return []
    else:
        try:
            if yaml.safe_load(config_text or '') != config:
                return []
        except yaml.YAMLError:
            return []
        if summary.stat().st_mtime_ns < identity['mtime_ns']:
            return []
    train = config['train_params']
    return [row for row in read_completed_results(summary)
            if row['evaluated_dataset'] in expected and all(
                row.get(key) == str(train.get(key, ''))
                for key in ('task_name', 'model', 'dataset', 'ckpt_name'))]


def write_recovered_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def inspect_checkpoint(path):
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        from ultralytics import YOLO
        model = YOLO(str(path))
        return {'model': 'yolo', 'classes': len(model.names),
                'data_yaml': model.ckpt.get('train_args', {}).get('data')}
    state = obj.get('model', obj) if isinstance(obj, dict) else None
    if not isinstance(state, dict):
        raise ValueError('Not a model state dictionary (possibly an epoch counter)')
    heads = [(k, v) for k, v in state.items()
             if re.search(r'cls_heads\.0\.(weight|pointwise.weight)$', k)]
    if not heads:
        raise ValueError('Unknown architecture: supply support/config rather than guessing')
    key, head = heads[0]
    # VGG SSD and ROI-SSD share state layouts; use the known run naming for that
    # distinction, or an explicit configuration for custom run directories.
    family = 'roissd-mobilenet' if 'pointwise' in key else 'roissd'
    if family == 'roissd' and any(p.startswith(('voc-ssd', 'imagenet-vid-ssd')) for p in path.parts):
        family = 'ssd'
    return {'model': family, 'classes': head.shape[0] // 4,
            'penalized': key.startswith('model.')}


def resolve_config(checkpoint, models_root, overrides):
    relative = checkpoint.relative_to(models_root).as_posix()
    matches = [(key, value) for key, value in overrides.items()
               if relative == key or relative.startswith(key.rstrip('/') + '/')]
    if matches:
        source = max(matches, key=lambda pair: len(pair[0]))[1]
        return copy.deepcopy(load_config(source)), str(source)
    # A saved run snapshot is stronger evidence than a folder-name heuristic.
    for directory in [checkpoint.parent, *checkpoint.parent.parents]:
        if not directory.is_relative_to(models_root):
            break
        source = directory / 'mixed_sampling_eval.yaml'
        if source.exists():
            cfg = load_config(str(source))
            info = inspect_checkpoint(checkpoint)
            cfg['dataset_params']['num_classes'] = info['classes']
            cfg['train_params']['model_num_classes'] = info['classes']
            cfg['train_params']['penalized_roissd'] = info.get('penalized', False)
            return cfg, str(source)
    info = inspect_checkpoint(checkpoint)
    family, classes = info['model'], info['classes']
    if family == 'yolo':
        if classes == 20:
            cfg = load_config('config/voc.yaml')
        else:
            cfg = load_config('config/imagenet-vid-yolo.yaml')
            data_yaml = info.get('data_yaml')
            if not data_yaml or not Path(data_yaml).is_file():
                raise ValueError('YOLO training dataset YAML missing; provide --config-map')
            cfg['dataset_params']['yolo_dataset_yaml'] = data_yaml
        cfg['train_params']['model'] = 'yolo'
        return cfg, 'checkpoint metadata + repository dataset defaults'
    if classes == 21:
        source = 'config/voc-roissd-mobilenet.yaml' if family == 'roissd-mobilenet' else 'config/voc.yaml'
    elif classes == 31:
        source = 'config/imagenet-vid-roissd-mobilenet.yaml' if family == 'roissd-mobilenet' else 'config/imagenet-vid-roissd.yaml'
    elif classes == 34:
        source = 'config/imagenet-vid-roissd.yaml'
    else:
        raise ValueError(f'Unknown {classes}-class SSD label space; provide --config-map')
    cfg = load_config(source)
    if classes == 34:
        base = load_config('config/imagenet-vid-base.yaml')
        cfg['dataset_params'] = {**cfg['dataset_params'], **base['dataset_params']}
        cfg['train_params'].update(dataset='imagenet-vid', model_label_space='imagenet-vid')
    elif classes == 21:
        cfg['train_params']['model_label_space'] = 'voc'
    cfg['dataset_params']['num_classes'] = classes
    cfg['train_params'].update(model=family, model_num_classes=classes,
                               penalized_roissd=info['penalized'])
    return cfg, source + ' (inferred defaults; override with --config-map if needed)'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root', type=Path, default=ROOT/'trained_models')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/batch_infer')
    parser.add_argument('--config-map', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Discover and write configs without running inference')
    parser.add_argument('--resume', action='store_true',
                        help='Skip completed modes/settings even if execution options changed')
    parser.add_argument('--resume-from', type=Path, action='append', default=[],
                        help='Additional batch output folder to recover matching CSV results from (repeatable)')
    parser.add_argument('--reverse', action='store_true',
                        help='Evaluate checkpoints in reverse sorted path order')
    parser.add_argument('--match', default='*', help='Glob filter on checkpoint paths relative to models root')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Data-loading workers per evaluation (default: 0)')
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--fixed-size-batch-size', type=int, default=1,
                        help='Opt-in batching for fixed-size sweep only')
    parser.add_argument('--no-reuse', action='store_true',
                        help='Reload models and datasets for every sweep setting')
    args = parser.parse_args(argv)
    if args.num_workers < 0 or args.fixed_size_batch_size < 1:
        parser.error('workers must be nonnegative and batch size must be positive')
    if args.resume_from and not args.resume:
        parser.error('--resume-from requires --resume')
    models_root, output = args.models_root.resolve(), args.output.resolve()
    if not models_root.is_relative_to(ROOT/'trained_models'):
        parser.error('--models-root must be within this repository\'s trained_models directory')
    overrides = yaml.safe_load(args.config_map.read_text(encoding='utf-8')) if args.config_map else {}
    overrides = {str(k).replace('\\', '/'): v for k, v in (overrides or {}).items()}
    checkpoints = sorted(p for p in models_root.rglob('*')
                         if p.suffix.lower() in {'.pt', '.pth'} and p.name != 'epoch.pth'
                         and p.relative_to(models_root).match(args.match))
    if args.reverse:
        checkpoints.reverse()
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for index, checkpoint in enumerate(checkpoints, 1):
        relative = checkpoint.relative_to(models_root)
        destination = output / relative  # Keep extension to distinguish x.pt and x.pth.
        destination.mkdir(parents=True, exist_ok=True)
        entry = {'checkpoint': str(checkpoint), 'modes': {}}
        print(f'[{index}/{len(checkpoints)}] {relative}', flush=True)
        try:
            cfg, source = resolve_config(checkpoint, models_root, overrides)
            train = cfg['train_params']
            train['task_name'] = checkpoint.parent.relative_to(ROOT/'trained_models').as_posix()
            train['ckpt_name'] = checkpoint.name
            if train['model'] == 'yolo':
                train['yolo_weights'] = str(checkpoint)
            config_path = destination/'config.yaml'
            config_text = yaml.safe_dump(cfg, sort_keys=False)
            previous_config = config_path.read_text(encoding='utf-8') if config_path.exists() else None
            config_path.write_text(config_text, encoding='utf-8')
            entry.update(config=str(config_path), config_source=source)
            identity = {'path': str(checkpoint), 'size': checkpoint.stat().st_size,
                        'mtime_ns': checkpoint.stat().st_mtime_ns, 'config': config_text}
            if train['model'] == 'yolo':
                identity['adapter_sha256'] = hashlib.sha256(
                    (ROOT/'model/model_adapters.py').read_bytes()
                ).hexdigest()
            # Retain execution metadata for reference; it does not invalidate resume.
            if args.num_workers or args.pin_memory or args.fixed_size_batch_size != 1 or args.no_reuse:
                identity['execution'] = {'num_workers': args.num_workers,
                                         'pin_memory': args.pin_memory,
                                         'fixed_size_batch_size': args.fixed_size_batch_size,
                                         'reuse': not args.no_reuse}
            for mode in MODES:
                marker = destination/f'{mode}.complete.json'
                if args.resume and mode_completed(marker, identity):
                    entry['modes'][mode] = 'SKIPPED_COMPLETED'
                    print('  '+mode+': SKIPPED_COMPLETED', flush=True)
                    continue
                recovered = {}
                if args.resume:
                    # Prefer current output rows; additional sources fill gaps only.
                    sources = [(destination, previous_config)]
                    for source_root in args.resume_from:
                        source_dir = source_root.resolve()/relative
                        if source_dir == destination:
                            continue
                        saved_config = source_dir/'config.yaml'
                        sources.append((source_dir, saved_config.read_text(encoding='utf-8')
                                        if saved_config.exists() else None))
                    for source_dir, saved_config in sources:
                        for row in recover_summary_rows(source_dir, mode, identity, saved_config):
                            recovered.setdefault(row['evaluated_dataset'], row)
                summary_name, expected = sweep_summary(mode, train['model'])
                if recovered:
                    print(f'  {mode}: found {len(recovered)}/{len(expected)} completed settings', flush=True)
                    if not args.dry_run:
                        write_recovered_summary(destination/mode/summary_name, list(recovered.values()))
                    if recovered.keys() >= expected:
                        entry['modes'][mode] = 'SKIPPED_COMPLETED'
                        if not args.dry_run:
                            temporary_marker = marker.with_suffix('.json.tmp')
                            temporary_marker.write_text(json.dumps(identity, indent=2), encoding='utf-8')
                            temporary_marker.replace(marker)
                        continue
                command = [sys.executable, '-u', '-m', 'tools.infer', '--config', str(config_path),
                           '--evaluate', 'true', '--infer-samples', 'false', '--eval-mode', mode,
                           '--results-path', str(destination/mode), '--summary-path', str(destination/mode)]
                command += ['--num-workers', str(args.num_workers)]
                command.append('--resume-evaluation')
                if args.pin_memory:
                    command.append('--pin-memory')
                if args.no_reuse:
                    command.append('--no-reuse')
                if mode == 'fixed-size-loop':
                    command += ['--eval-batch-size', str(args.fixed_size_batch_size)]
                print('  '+subprocess.list2cmdline(command), flush=True)
                if args.dry_run:
                    entry['modes'][mode] = 'PLANNED'
                    continue
                marker.unlink(missing_ok=True)
                progress_marker = destination/f'{mode}.progress.json'
                if not (args.resume and mode_completed(progress_marker, identity)):
                    # Clear incompatible summaries before recording the new identity.
                    if not recovered:
                        for name in ('results.csv', 'fixed_size_results_v2.csv'):
                            (destination/mode/name).unlink(missing_ok=True)
                    temporary_progress = progress_marker.with_suffix('.json.tmp')
                    temporary_progress.write_text(json.dumps(identity, indent=2), encoding='utf-8')
                    temporary_progress.replace(progress_marker)
                with (destination/f'{mode}.log').open('w', encoding='utf-8') as log:
                    with subprocess.Popen(
                        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding='utf-8', errors='replace', bufsize=1,
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)
                            log.write(line)
                            log.flush()
                        returncode = process.wait()
                entry['modes'][mode] = 'PASS' if returncode == 0 else f'FAILED ({returncode})'
                if returncode == 0:
                    temporary_marker = marker.with_suffix('.json.tmp')
                    temporary_marker.write_text(json.dumps(identity, indent=2), encoding='utf-8')
                    temporary_marker.replace(marker)
                print('  '+mode+': '+entry['modes'][mode], flush=True)
        except Exception as exc:
            entry['error'] = str(exc)
            (destination/'error.log').write_text(traceback.format_exc(), encoding='utf-8')
            print('  ERROR: '+str(exc), flush=True)
        report.append(entry)
        (output/'batch_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    failures = sum('error' in e or any(s.startswith('FAILED') for s in e['modes'].values()) for e in report)
    print(f'Done: {len(report)} checkpoints; {failures} failed/unresolved. Report: {output / "batch_report.json"}')
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
