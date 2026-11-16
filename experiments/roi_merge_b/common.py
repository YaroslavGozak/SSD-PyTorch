"""Portable IO and fail-closed artifact identities; no model imports at startup."""
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS = ('yolo26n', 'roi_ssd_mobilenet')
POLICIES = ('no_merge', 'geometric', 'affine', 'piecewise', 'lookup', 'conservative_lookup', 'oracle')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(canonical(row) + '\n' for row in value) if lines else canonical(value) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = stream.name
    os.replace(temporary, path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def relative_path(value):
    path = PurePosixPath(value)
    if not value or '\\' in value or ':' in value or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Nonportable asset path: {value}')
    return path


def asset(root, relative):
    path = (Path(root) / relative_path(relative)).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Asset escapes root: {relative}')
    return path


def config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if cfg.get('schema_version') != 1 or set(cfg['models']) != set(MODELS):
        raise ValueError('Unsupported config schema/models')
    for name, spec in cfg['models'].items():
        relative_path(spec['checkpoint'])
        if spec['shape_grid'] != (32 if name == 'yolo26n' else 1):
            raise ValueError('Unsupported executed shape rule')
    if cfg['profiling']['device'] != 'cpu' or cfg['profiling']['timing_scope'] != 'prepared_detector_forward':
        raise ValueError('This protocol supports CPU prepared detector forward only')
    for key in ('cpu_threads', 'interop_threads', 'opencv_threads'):
        if int(cfg['runtime'][key]) < 1:
            raise ValueError('Thread counts must be positive')
    for key in ('warmup', 'repetitions', 'evaluation_repetitions', 'calibration_max_shapes', 'max_heldout_shapes'):
        if int(cfg['profiling'][key]) < (4 if key == 'evaluation_repetitions' else 1):
            raise ValueError(f'Invalid profiling {key}')
    if len(set(cfg['profiling']['grid_sides'])) < 3:
        raise ValueError('At least three calibration grid sides required')
    options=cfg['policies']
    for field in ('geometric_area_ratio_max','confidence_z'):
        if not math.isfinite(options[field]) or options[field] <= 0:
            raise ValueError(f'Invalid policy {field}')
    if not math.isfinite(options['minimum_gain_ms']) or options['minimum_gain_ms'] < 0:
        raise ValueError('minimum_gain_ms must be finite and nonnegative')
    if options['bootstrap_repetitions'] < 1:
        raise ValueError('Bootstrap repetitions must be positive')
    bins=options['boundary_bins_ms']
    if not bins or bins != sorted(set(bins)) or any(not math.isfinite(v) or v <= 0 for v in bins):
        raise ValueError('Boundary bins must be increasing positive milliseconds')
    canvas=cfg['geometry']['canvas_hw']
    if len(canvas)!=2 or canvas[0]!=canvas[1] or canvas[0]%32:
        raise ValueError('Shared SSD transform requires a square canvas divisible by YOLO stride')
    if any(not isinstance(v,int) or v<32 or v>min(canvas) for v in cfg['profiling']['grid_sides']):
        raise ValueError('Calibration grid must remain inside the supported canvas, with sides >=32')
    if not 0 <= cfg['policies']['exact_oracle_max_rois'] <= 10:
        raise ValueError('Exact oracle cap must be 0..10')
    return cfg


def dataset_root(cfg, override=None):
    value = override or os.environ.get(cfg['dataset']['root_env'])
    if not value or not Path(value).is_dir():
        raise FileNotFoundError(f"Set {cfg['dataset']['root_env']} or --dataset-root to the dataset directory")
    return Path(value).resolve()


def checkpoint(cfg, model, root=None):
    # Defaults are canonical repository-relative paths, with an explicit portable root override.
    spec = cfg['models'][model]
    path = asset(root or ROOT, spec['checkpoint'])
    if not path.is_file():
        raise FileNotFoundError(f'Missing selected checkpoint: {path}')
    actual = sha(path)
    if actual != spec['checkpoint_sha256']:
        raise ValueError(f'Checkpoint SHA mismatch for {model}: expected {spec["checkpoint_sha256"]}, found {actual}')
    return path


def code_identity():
    paths = list((ROOT / 'experiments/roi_merge_b').rglob('*.py'))
    paths += list((ROOT / 'experiments/cost_model_validation').glob('*.py'))
    paths += list((ROOT / 'model').glob('*.py'))
    paths += [ROOT / p for p in ('tools/helpers/pipeline.py', 'tools/benchmarks/measurek.py', 'dataset/transforms/ssd_transform.py', 'experiments/cpu_pilot/runtime_threads.py')]
    return digest({p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))})


def git_identity():
    def run(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    return {'commit': run('rev-parse', 'HEAD'), 'dirty': bool(run('status', '--porcelain'))}


def load_manifest(path, cfg):
    path = Path(path)
    meta = read(path.with_name('manifest.meta.json'))
    if sha(path) != meta['manifest_sha256']:
        raise ValueError('Manifest SHA mismatch')
    if meta.get('seed') != cfg['seed'] or meta.get('split_parameters') != cfg['splits']:
        raise ValueError('Manifest seed/split configuration mismatch')
    if meta['geometry'] != cfg['geometry'] or meta['model_shape_grids'] != {m: cfg['models'][m]['shape_grid'] for m in MODELS}:
        raise ValueError('Manifest geometry/shape rule mismatch')
    rows = jsonl(path)
    if len(rows) != meta['frames']:
        raise ValueError('Manifest frame count mismatch')
    sequence_split, keys = {}, set()
    for row in rows:
        relative_path(row['image']); relative_path(row['label'])
        key = (row['video_id'], row['frame_index'])
        if key in keys or sequence_split.setdefault(row['video_id'], row['split']) != row['split']:
            raise ValueError('Duplicate frame or sequence split leakage')
        keys.add(key)
        ids = [roi['roi_id'] for roi in row['rois']]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate ROI ID')
        for roi in row['rois']:
            x1, y1, x2, y2 = roi['box']
            if not (0 <= x1 < x2 <= row['width'] and 0 <= y1 < y2 <= row['height']):
                raise ValueError('ROI outside frame')
    if rows != sorted(rows, key=lambda r: (r['video_id'], r['frame_index'])):
        raise ValueError('Manifest order is not canonical')
    return rows, meta


def cohort(rows, cfg, subset='full', limit=None):
    rows = [r for r in rows if r['split'] == 'test']
    count = len(rows) if subset == 'full' else cfg['subsets'][subset + '_frames']
    # Prefix of a frozen seeded frame permutation; IDs persist in every run identity.
    import random
    order = list(range(len(rows)))
    random.Random(cfg['seed']).shuffle(order)
    selected = set(order[:count])
    rows = [row for i, row in enumerate(rows) if i in selected]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError('Empty test cohort')
    return rows
