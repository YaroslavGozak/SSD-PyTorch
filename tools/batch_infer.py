"""Recursively evaluate trained checkpoints with both ROI evaluation sweeps.

Run from repository root: python -m tools.batch_infer --dry-run
Then: python -m tools.batch_infer --resume
Optional --config-map YAML maps root-relative checkpoint or directory paths to
config YAML paths. The most specific matching entry wins. Auto-resolution is
limited to the repository's known model layouts; ambiguous models are reported.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

import torch
import yaml
from tools.helpers.config_reader import load_config

ROOT = Path(__file__).resolve().parents[1]
MODES = ('pad-loop', 'fixed-size-loop')


def mode_completed(marker, identity):
    """Use the mode's own snapshot, independently of the shared config file."""
    try:
        return json.loads(marker.read_text(encoding='utf-8')) == identity
    except (OSError, ValueError):
        # Missing or interrupted marker writes must cause a fresh evaluation.
        return False


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
    parser.add_argument('--resume', action='store_true', help='Skip modes with a successful completion marker')
    parser.add_argument('--match', default='*', help='Glob filter on checkpoint paths relative to models root')
    args = parser.parse_args(argv)
    models_root, output = args.models_root.resolve(), args.output.resolve()
    if not models_root.is_relative_to(ROOT/'trained_models'):
        parser.error('--models-root must be within this repository\'s trained_models directory')
    overrides = yaml.safe_load(args.config_map.read_text(encoding='utf-8')) if args.config_map else {}
    overrides = {str(k).replace('\\', '/'): v for k, v in (overrides or {}).items()}
    checkpoints = sorted(p for p in models_root.rglob('*')
                         if p.suffix.lower() in {'.pt', '.pth'} and p.name != 'epoch.pth'
                         and p.relative_to(models_root).match(args.match))
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
            config_path.write_text(config_text, encoding='utf-8')
            entry.update(config=str(config_path), config_source=source)
            identity = {'path': str(checkpoint), 'size': checkpoint.stat().st_size,
                        'mtime_ns': checkpoint.stat().st_mtime_ns, 'config': config_text}
            for mode in MODES:
                marker = destination/f'{mode}.complete.json'
                if args.resume and mode_completed(marker, identity):
                    entry['modes'][mode] = 'SKIPPED_COMPLETED'
                    print('  '+mode+': SKIPPED_COMPLETED', flush=True)
                    continue
                command = [sys.executable, '-u', '-m', 'tools.infer', '--config', str(config_path),
                           '--evaluate', 'true', '--infer-samples', 'false', '--eval-mode', mode,
                           '--results-path', str(destination/mode), '--summary-path', str(destination/mode)]
                print('  '+subprocess.list2cmdline(command), flush=True)
                if args.dry_run:
                    entry['modes'][mode] = 'PLANNED'
                    continue
                marker.unlink(missing_ok=True)
                # A failed/interrupted mode may have appended partial results.
                # Restart its summaries rather than duplicating those rows.
                for name in ('results.csv', 'fixed_size_results.csv'):
                    (destination/mode/name).unlink(missing_ok=True)
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
