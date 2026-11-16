"""Canonical annotation-derived workload; policy independent and portable."""
import csv
import itertools
import math
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .common import MODELS, ROOT, asset, atomic, canonical, digest, jsonl, sha


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def clip_context(cx, cy, w, h, width, height, cfg):
    canvas_h, canvas_w = cfg['canvas_hw']
    padding = cfg['padding_canvas_px']
    half_w = max(w * width / 2 + padding * width / canvas_w,
                 cfg['minimum_side_canvas_px'] * width / canvas_w / 2)
    half_h = max(h * height / 2 + padding * height / canvas_h,
                 cfg['minimum_side_canvas_px'] * height / canvas_h / 2)
    x1 = max(0, min(width - math.ceil(2 * half_w), math.floor(cx * width - half_w)))
    y1 = max(0, min(height - math.ceil(2 * half_h), math.floor(cy * height - half_h)))
    return [int(x1), int(y1), min(width, math.ceil(x1 + 2 * half_w)), min(height, math.ceil(y1 + 2 * half_h))]


@lru_cache(maxsize=32768)
def effective(box, original_hw, canvas_hw, grid):
    """The real crop helper is the single source of executed shape rounding."""
    import torch
    from tools.helpers.pipeline import convert_crop_to_input_tensor
    tensor = torch.empty((3, *canvas_hw))
    crop, snapped = convert_crop_to_input_tensor(tensor, list(box), original_hw, roi_grid=grid)
    return tuple(map(int, crop.shape[-2:])), tuple(snapped)


def shape_for(box, row, cfg, model):
    return effective(tuple(box), (row['height'], row['width']), tuple(cfg['geometry']['canvas_hw']), cfg['models'][model]['shape_grid'])[0]


def all_rectangles(rois, maximum=20000):
    """All distinct enclosing boxes of nonempty subsets, without enumerating partitions."""
    boxes = set()
    for roi in rois:
        box = tuple(roi['box'])
        boxes |= {tuple(union(old, box)) for old in list(boxes)} | {box}
        if len(boxes) > maximum:
            raise ValueError('Rectangle closure exceeds budget; use an explicitly frozen smaller cohort')
    return sorted(boxes)


def frame_shapes(row, cfg, model):
    boxes = all_rectangles(row['rois']) if row['rois'] else [(0, 0, row['width'], row['height'])]
    return {shape_for(box, row, cfg, model) for box in boxes}


def parse_labels(path, classes):
    values = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'Expected YOLO class/cx/cy/w/h: {path}')
        label = int(fields[0])
        coords = list(map(float, fields[1:]))
        if not 0 <= label < classes or any(not math.isfinite(v) or not 0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
            raise ValueError(f'Invalid label or coordinates: {path}')
        values.append([label, *coords])
    return sorted(values)


def source_rows(cfg, cohort_path=None):
    rows = jsonl(ROOT / cfg['dataset']['index'])
    source = cohort_path or cfg['dataset'].get('cohort_manifest')
    if source:
        with Path(source).open(encoding='utf-8', newline='') as stream:
            selected = {(r['video_id'], int(r['frame_index'])) for r in csv.DictReader(stream, delimiter='\t')}
        rows = [r for r in rows if (r['video_id'], r['frame_index']) in selected]
        if len(rows) != len(selected):
            raise ValueError('Cohort frame IDs absent/duplicated in source audit index')
    rows = sorted(rows, key=lambda r: (r['video_id'], r['frame_index']))
    if len({(r['video_id'], r['frame_index']) for r in rows}) != len(rows):
        raise ValueError('Duplicate source IDs')
    return rows


def build_rows(source, root, cfg):
    from PIL import Image
    sequences = sorted({r['video_id'] for r in source})
    random.Random(cfg['seed']).shuffle(sequences)
    ncal, nval = (cfg['splits'][k] for k in ('calibration_sequences', 'validation_sequences'))
    if ncal < 1 or nval < 1 or len(sequences) <= ncal + nval:
        raise ValueError('Need nonempty calibration, validation and test sequence splits')
    assignment = {video: 'calibration' if i < ncal else 'validation' if i < ncal+nval else 'test' for i, video in enumerate(sequences)}
    records = []
    for index, src in enumerate(source):
        image, label = asset(root, src['image']), asset(root, src['label'])
        if sha(label) != src['label_sha256']:
            raise ValueError(f'Label SHA mismatch: {label}')
        labels = parse_labels(label, cfg['dataset']['expected_classes'])
        if len(labels) != len(src['gt']) or any(abs(a-b) > 2e-6 for observed, expected in zip(labels, sorted(src['gt'])) for a,b in zip(observed,expected)):
            raise ValueError(f'Index/label GT mismatch: {label}')
        with Image.open(image) as decoded:
            decoded.load()
            width, height = decoded.size
        grouped = {}
        for object_index, (label_id, cx, cy, w, h) in enumerate(labels):
            box = tuple(clip_context(cx,cy,w,h,width,height,cfg['geometry']))
            key = box if cfg['geometry']['deduplicate'] else (*box, object_index)
            entry = grouped.setdefault(key, {'box': list(box), 'objects': []})
            entry['objects'].append({'class_id':label_id, 'annotation_index':object_index})
        row = {'video_id': src['video_id'], 'frame_index':src['frame_index'], 'image':src['image'], 'label':src['label'],
               'width':width, 'height':height, 'image_sha256':sha(image), 'label_sha256':sha(label),
               'split':assignment[src['video_id']], 'source':'gt-derived', 'annotation_count':len(labels), 'rois':[]}
        for i, entry in enumerate(sorted(grouped.values(), key=lambda r:r['box'])):
            roi = {'roi_id':f'r{i:04d}', **entry}
            roi['effective_hw'] = {m:list(shape_for(roi['box'],row,cfg,m)) for m in MODELS}
            row['rois'].append(roi)
        records.append(row)
        if (index+1)%250 == 0:
            print(f'Frozen geometry: {index+1}/{len(source)}', flush=True)
    return records


def write_workload(rows, cfg, output, class_names, source_digest, fixture=False):
    output = Path(output)
    if (output/'manifest.jsonl').exists() or (output/'manifest.meta.json').exists():
        raise FileExistsError('Use a fresh workload directory; manifests are immutable')
    atomic(output/'manifest.jsonl', rows, lines=True)
    counts = Counter(len(r['rois']) for r in rows)
    meta = {'schema_version':1,'experiment_id':cfg['experiment_id'], 'fixture':fixture,
            'manifest_sha256':sha(output/'manifest.jsonl'), 'frames':len(rows), 'sequences':len({r['video_id'] for r in rows}),
            'source_index_sha256':source_digest,'dataset_content_hash':digest([(r['image'],r['image_sha256'],r['label_sha256']) for r in rows]),
            'geometry':cfg['geometry'],'seed':cfg['seed'],'split_parameters':cfg['splits'],'model_shape_grids':{m:cfg['models'][m]['shape_grid'] for m in MODELS},
            'class_names':class_names,'roi_count_distribution':dict(sorted(counts.items())),
            'split_counts':dict(Counter(r['split'] for r in rows)),
            'raw_annotation_distribution':dict(Counter(r.get('annotation_count',len(r['rois'])) for r in rows)), 'coordinate_convention':'original pixel xyxy, exclusive x2/y2',
            'candidate_hash':digest([(r['video_id'],r['frame_index'],r['rois']) for r in rows])}
    atomic(output/'manifest.meta.json',meta)
    return meta


def geometry_strata(a,b):
    overlap = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    gx, gy = max(0,a[0]-b[2],b[0]-a[2]), max(0,a[1]-b[3],b[1]-a[3])
    ratio = area(union(a,b))/(area(a)+area(b))
    return {'layout':'overlap' if overlap else 'diagonal' if gx and gy else 'separated',
            'area_ratio':ratio,'empty_area_stratum':'high' if ratio>1.4 else 'low'}
