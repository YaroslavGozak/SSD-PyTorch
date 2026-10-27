"""Stage 3a: locked CPU oracle-crop diagnosis; no production changes."""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from unittest.mock import patch

from . import run_pilot as pilot
import cv2
import numpy as np
import torch
from tools.helpers import pipeline
from tools.trackers.static_padding_tracker import StaticPaddingTracker
from tools.benchmarks.video_artifacts import detection_metrics, write_json
from tools.infer import compute_map, get_iou

OUT = pilot.ROOT / 'final_experiments/pilot_oracle_diagnosis'
PADDINGS = (32, 64, 96, 128)
SOURCE_FILES = ('model/roissd_mobilenet.py', 'tools/helpers/pipeline.py',
                'tools/trackers/static_padding_tracker.py', 'tools/infer.py',
                'experiments/cpu_pilot/run_pilot.py',
                'experiments/cpu_pilot/oracle_diagnosis.py')


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def objects(gt):
    return [(cls, index, box) for cls, boxes in gt.items() for index, box in enumerate(boxes)]


def match_objects(detections, gt):
    """Same best-GT/score ordering as compute_map; count all classes for FP."""
    matched = set()
    tp, fp, duplicate = 0, 0, 0
    for det in sorted(detections, key=lambda d: -d['confidence']):
        candidates = gt.get(det['class'], [])
        ious = [get_iou(det['bbox'], box) for box in candidates]
        if ious and max(ious) >= .5:
            key = (det['class'], int(np.argmax(ious)))
            if key not in matched:
                matched.add(key)
                tp += 1
            else:
                fp += 1
                duplicate += 1
        else:
            fp += 1
    return {'tp': tp, 'fp': fp, 'duplicate_detections': duplicate, 'matched': matched}


def pair_counts(detections):
    pairs = [get_iou(a['bbox'], b['bbox']) for i, a in enumerate(detections)
             for b in detections[i+1:] if a['class'] == b['class']]
    return {'same_class_pairs_iou_gt_05': sum(i > .5 for i in pairs),
            'same_class_pairs_iou_gt_07': sum(i > .7 for i in pairs)}


def postprocess(detections, threshold, width, height, nms=True):
    selected = pipeline.merge_detections_nms(detections, .45) if nms else detections
    final = []
    for det in selected:
        box = pipeline.clip_bbox(det['bbox'], width, height)
        if det['confidence'] >= threshold and box is not None:
            final.append(dict(det, bbox=box))
    return final


def geometry(box, roi):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    w, h, cw, ch = x2-x1, y2-y1, rx2-rx1, ry2-ry1
    dx = abs((x1+x2-rx1-rx2)/2)/cw
    dy = abs((y1+y2-ry1-ry2)/2)/ch
    margins = [(x1-rx1)/cw, (rx2-x2)/cw, (y1-ry1)/ch, (ry2-y2)/ch]
    return {'crop_h': ch, 'crop_w': cw, 'gt_h_in_crop': h, 'gt_w_in_crop': w,
            'gt_area': w*h, 'crop_area': cw*ch, 'gt_area_ratio': w*h/(cw*ch),
            'gt_width_ratio': w/cw, 'gt_height_ratio': h/ch,
            'gt_max_side_ratio': max(w/cw, h/ch), 'gt_min_side_ratio': min(w/cw, h/ch),
            'center_dx_ratio': dx, 'center_dy_ratio': dy,
            'center_distance_ratio': math.hypot(dx, dy),
            'touches_crop_boundary': min(x1-rx1, rx2-x2, y1-ry1, ry2-y2) <= 1,
            'min_normalized_boundary_distance': min(margins)}


def oracle(model, tensor, gt, idx2label, frame_hw, padding, threshold, traces):
    h, w = frame_hw
    gt_objects = objects(gt)
    generator = StaticPaddingTracker(pad_x=padding, pad_y=padding, conf_det_min=threshold,
                                     model_input_size=(320, 320), hold_last_for_frames=0)
    proposals = generator.update([{'bbox': box, 'class': cls, 'confidence': 1.}
                                  for cls, _, box in gt_objects], frame_shape=frame_hw)['rois']
    assert len(proposals) == len(gt_objects)
    rois = [r['roi'] for r in proposals]
    # Match the normal pipeline's source -> model -> source round trip exactly.
    mh, mw = tensor.shape[-2:]
    model_rois = pipeline._convert_rois_between_spaces(rois, (w, h), (mw, mh))
    rois = pipeline._convert_rois_between_spaces(model_rois, (mw, mh), (w, h))
    all_dets, own, used = [], [], []
    for (cls, index, box), roi in zip(gt_objects, rois):
        roi = pipeline.clip_bbox(roi, w, h)
        assert roi is not None
        crop, snapped = pipeline.convert_crop_to_input_tensor(tensor, roi, frame_hw, roi_grid=1)
        rx1, ry1, rx2, ry2 = snapped
        _, output = pipeline.run_model_inference(model, crop.unsqueeze(0))
        dets = pipeline.tensor_to_detection_list(output[0], idx2label, rx2-rx1, ry2-ry1, (rx1, ry1))
        all_dets.extend(dets)
        used.append(snapped)
        same_class = [d for d in dets if d['class'] == cls]
        own.append({'class': cls, 'gt_index_in_class': index, 'gt_box': box, 'crop_box': snapped,
                    **geometry(box, snapped), 'network_crop_h': crop.shape[-2],
                    'network_crop_w': crop.shape[-1],
                    'own_crop_best_prediction_iou': max((get_iou(d['bbox'], box) for d in same_class), default=0),
                    'own_crop_max_confidence': max((d['confidence'] for d in same_class), default=0),
                    'selected_L_R': traces[-1]['selected_L_R'],
                    'feature_shapes': traces[-1]['feature_shapes']})
    return all_dets, used, own


def frame_record(mode, dets, pre, gt, diff, rois, trace, full, frame_meta):
    match = match_objects(dets, gt)
    count = len(objects(gt))
    covered, _ = pilot._coverage(gt, rois, frame_meta['width'], frame_meta['height'], full)
    pred = pilot._map_prediction(dets)
    ap, _, recall, _ = compute_map([pred], [gt], difficult=[diff])
    return {**frame_meta, 'mode': mode, 'detections': dets, 'pre_nms_detections': pre,
            'gt': gt, 'difficult': diff, 'rois': rois, 'traces': copy.deepcopy(trace),
            'full_frame': full, 'gt_count': count, 'gt_covered': covered,
            'frame_map50': ap, 'frame_recall50': recall,
            'tp': match['tp'], 'fp': match['fp'],
            'duplicates': match['duplicate_detections'],
            'before_nms_count': len(pre), 'after_nms_count': len(dets),
            **pair_counts(dets),
            **{'pre_'+k: v for k, v in pair_counts(pre).items()}}


def summarize(mode, records):
    n = len(records)
    metrics = detection_metrics([pilot._map_prediction(r['detections']) for r in records],
                                [r['gt'] for r in records], [r['difficult'] for r in records])
    tp, fp = sum(r['tp'] for r in records), sum(r['fp'] for r in records)
    return {'mode': mode, 'num_frames': n, **metrics,
            'gt_coverage': sum(r['gt_covered'] for r in records)/sum(r['gt_count'] for r in records),
            'precision50_micro': tp/max(1, tp+fp),
            'mean_detections_per_frame': sum(len(r['detections']) for r in records)/n,
            'false_positives_per_frame': fp/n, 'duplicates_per_frame': sum(r['duplicates'] for r in records)/n,
            'same_class_pairs_iou_gt_05_per_frame': sum(r['same_class_pairs_iou_gt_05'] for r in records)/n,
            'same_class_pairs_iou_gt_07_per_frame': sum(r['same_class_pairs_iou_gt_07'] for r in records)/n}


def object_bins(rows):
    output = []
    for feature, edges in [('gt_area_ratio', [0, .25, .5, .7, .8, .9, .95, float('inf')]),
                           ('gt_max_side_ratio', [0, .25, .5, .7, .8, .9, .95, float('inf')]),
                           ('center_distance_ratio', [0, .05, .1, .2, .3, .4, .5, float('inf')])]:
        for lo, hi in zip(edges, edges[1:]):
            selected = [r for r in rows if lo <= r[feature] < hi]
            recall = np.mean([r['matched50_after_global_nms'] for r in selected]) if selected else None
            output.append({'feature': feature, 'bin': f'[{lo},{hi})', 'N_GT': len(selected),
                           'Recall50_micro': recall, 'miss_rate': 1-recall if recall is not None else None,
                           'mean_best_prediction_iou': np.mean([r['best_prediction_iou'] for r in selected]) if selected else None,
                           'mean_max_confidence': np.mean([r['own_crop_max_confidence'] for r in selected]) if selected else None})
    return output


def overlay_panel(record, title):
    image = cv2.imread(record['image'])
    scale = 640/image.shape[1]
    image = cv2.resize(image, (640, round(image.shape[0]*scale)))
    def box(rect, color, label):
        x1, y1, x2, y2 = [round(v*scale) for v in rect]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(image, label, (max(0,x1), max(12,y1)), cv2.FONT_HERSHEY_SIMPLEX, .36, color, 1, cv2.LINE_AA)
    for cls, _, rect in objects(record['gt']):
        box(rect, (0, 220, 0), 'GT '+cls)
    for index, rect in enumerate(record['rois']):
        box(rect, (0, 160, 255), f'C{index}')
    for det in record['detections']:
        box(det['bbox'], (20, 20, 255), f'{det["class"]} {det["confidence"]:.2f}')
    lines = [f'{title}  AP50={record["frame_map50"]:.3f} TP={record["tp"]} FP={record["fp"]}',
             f'video={record["video_id"]} frame={record["frame_index"]}; green=GT orange=crop red=pred']
    for index, trace in enumerate(record['traces']):
        shapes = '/'.join(f'{h}x{w}' for h,w in trace['feature_shapes'])
        lines.append(f'C{index}: L_R={trace["selected_L_R"]} maps={shapes}')
    canvas = np.full((image.shape[0]+24*len(lines)+12, 640, 3), 245, dtype=np.uint8)
    canvas[:image.shape[0]] = image
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (6, image.shape[0]+22+24*i), cv2.FONT_HERSHEY_SIMPLEX, .43, (20,20,20), 1, cv2.LINE_AA)
    return canvas


def export_overlays(records):
    ranking = []
    for first, second, directory in [('FF', 'ORACLE_GT_SEG', 'ff_vs_oracle'),
                                      ('ORACLE_GT_SEG', 'PRED_SEG', 'oracle_vs_pred')]:
        ranked = sorted(range(320), key=lambda i: (
            records[first][i]['frame_map50']-records[second][i]['frame_map50'],
            records[first][i]['tp']-records[second][i]['tp']), reverse=True)[:20]
        dest = OUT / 'diagnostics' / directory
        dest.mkdir(parents=True, exist_ok=True)
        for rank, index in enumerate(ranked, 1):
            a, b = records[first][index], records[second][index]
            panels = [overlay_panel(a, first), overlay_panel(b, second)]
            canvas = np.full((max(p.shape[0] for p in panels), 1280, 3), 245, dtype=np.uint8)
            for j, panel in enumerate(panels):
                canvas[:panel.shape[0], j*640:(j+1)*640] = panel
            name = f'{rank:02d}_{a["video_id"]}_{a["frame_index"]}.jpg'
            assert cv2.imwrite(str(dest / name), canvas)
            ranking.append({'comparison': directory, 'rank': rank, 'video_id': a['video_id'],
                            'frame_index': a['frame_index'],
                            'frame_map50_drop': a['frame_map50']-b['frame_map50'],
                            'overlay': str((dest/name).relative_to(OUT))})
    write_csv('diagnostic_frame_ranking.csv', ranking)


def make_report(summaries, bins, meta):
    by = {r['mode']: r for r in summaries}
    ff, oracle_row, pred = [by[m] for m in ('FF', 'ORACLE_GT_SEG', 'PRED_SEG')]
    decomposition = {}
    for metric in ('mAP50', 'recall50'):
        total = ff[metric]-pred[metric]
        crop_loss = ff[metric]-oracle_row[metric]
        formation_loss = oracle_row[metric]-pred[metric]
        decomposition[metric] = {'total_loss': total, 'crop_domain_loss': crop_loss,
                                 'segment_formation_loss': formation_loss,
                                 'crop_domain_share': crop_loss/total if total else None,
                                 'segment_formation_share': formation_loss/total if total else None}
    d = decomposition['mAP50']
    retention = oracle_row['mAP50']/ff['mAP50']
    if retention >= .90 and d['segment_formation_share'] >= .67:
        case = 'A — segment formation / temporal propagation dominates'
        recommendation = 'Next: tracker/full-frame refresh/hold analysis on the same locked protocol.'
    elif d['crop_domain_share'] >= .67:
        case = 'B — crop-domain robustness dominates'
        recommendation = 'Next: inspect training crop distribution, context and anchor/head specialization; do not start with tracker tuning.'
    else:
        case = 'C — both crop-domain and segment formation contribute materially'
        recommendation = 'Next: investigate crop context/head behavior and temporal proposal quality separately; retain the locked acceptance criteria.'
    report = ['# Stage 3a: ROI-SSD-MobileNet oracle segment diagnosis', '', f'**{case}**', '',
              '## Locked protocol', '',
              '32 videos / 320 frames, original pilot manifest, checkpoint, CPU (8 threads), confidence 0.5, '
              'NMS IoU 0.45, existing area-AP evaluator. FF and PRED_SEG were rerun and every saved prediction '
              'reproduced within 1e-6. PRED_SEG uses previous localization, hold=0, interval=10, no merge. '
              'ORACLE uses current-frame GT on every frame, with no keyframe/fallback/tracker memory.', '',
              '**Padding coordinates:** pilot p=96 means 96 pixels in the base 320×320 model tensor, '
              'scaled independently to source-frame X/Y. ORACLE uses the same scaling and source→model→source '
              'round trip as PRED_SEG, rather than silently substituting 96 native-image pixels. '
              'Inference uses slices of the already normalized/resized full-frame tensor, exactly as the pilot.', '',
              '| Mode | mAP50 | mAP50:95 | Recall50 | Coverage | Det/frame | FP/frame | Duplicates/frame |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        report.append(f'| {r["mode"]} | {r["mAP50"]:.6f} | {r["mAP50_95"]:.6f} | {r["recall50"]:.6f} | '
                      f'{r["gt_coverage"]:.6f} | {r["mean_detections_per_frame"]:.3f} | '
                      f'{r["false_positives_per_frame"]:.3f} | {r["duplicates_per_frame"]:.3f} |')
    report += ['', '## Practical loss decomposition', '',
               '| Metric | FF−PRED | FF−ORACLE (crop) | ORACLE−PRED (formation) | Crop share | Formation share |',
               '|---|---:|---:|---:|---:|---:|']
    for metric, values in decomposition.items():
        report.append(f'| {metric} | {values["total_loss"]:.6f} | {values["crop_domain_loss"]:.6f} | '
                      f'{values["segment_formation_loss"]:.6f} | {values["crop_domain_share"]:.1%} | '
                      f'{values["segment_formation_share"]:.1%} |')
    report += ['', f'ORACLE mAP50 retention versus FF: **{retention:.2%}**. '
               'These are practical contrasts, not independent causal effects. ORACLE also changes proposal count, '
               'geometry and refresh behavior. Negative components/shares, if present, are retained.', '',
               '## Object geometry and misses', '',
               'Each GT contributes exactly once at p=96, linked to its own GT-generated crop. Recall bins use '
               'unique one-to-one, class-correct matches after global NMS; IoU uses the best class-correct final '
               'prediction. Confidence is the maximum class-correct returned confidence in that object’s own crop '
               '(zero when absent). These are thresholded detector outputs, not raw logits. '
               'Crop dimensions/areas are in source pixels; actual network H/W are logged separately.', '',
               '| Feature | Bin | N GT | Recall50 micro | Mean best IoU | Mean own-crop max confidence | Miss rate |',
               '|---|---|---:|---:|---:|---:|---:|']
    for b in bins:
        if b['N_GT']:
            report.append(f'| {b["feature"]} | {b["bin"]} | {b["N_GT"]} | {b["Recall50_micro"]:.4f} | '
                          f'{b["mean_best_prediction_iou"]:.4f} | {b["mean_max_confidence"]:.4f} | {b["miss_rate"]:.4f} |')
    report += ['', 'Empty bins remain in CSV and support no conclusion. Bin associations are descriptive and '
               'confounded by class, source size, clipping and context; no monotonic or causal size effect is assumed.', '',
               '## Duplicates and global NMS', '',
               'The normal pipeline already performs global class-wise NMS at IoU 0.45 after remapping. '
               'The no-global-NMS variant retains detector-internal per-crop NMS and the same confidence/clipping. '
               f'At p=96, global NMS changes oracle mAP50 from '
               f'{by["ORACLE_GT_SEG_NO_GLOBAL_NMS"]["mAP50"]:.6f} to {oracle_row["mAP50"]:.6f} '
               f'(Δ={oracle_row["mAP50"]-by["ORACLE_GT_SEG_NO_GLOBAL_NMS"]["mAP50"]:+.6f}). '
               'Duplicate counts mean repeated predictions matching an already matched GT; pair counts at '
               'IoU>0.5/>0.7 before/after NMS are separately provided in duplicate_nms_analysis.csv. '
               'FP counts include predictions from classes absent in a frame.', '',
               '## Frame-level evidence', '',
               'Twenty largest FF→ORACLE drops and twenty largest ORACLE→PRED drops are ranked by per-frame '
               'mAP50, then TP loss, with stable manifest order for ties. Each comparison overlay shows GT, '
               'actual snapped crop boundaries, predictions/confidence, selected L_R and feature-map shapes. '
               'Raw detections, crops and traces are retained in frame_records.jsonl.', '',
               '## Next action', '', recommendation,
               'Production weights, thresholds, pipeline and padding decision are unchanged. '
               'CA/FULL-TECH and Raspberry Pi remain on hold; this diagnostic does not waive pilot acceptance.']
    meta['decomposition'] = decomposition
    meta['case'] = case
    meta['recommendation'] = recommendation
    (OUT / 'oracle_diagnosis_report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')


def main():
    pilot._configure_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    split, manifests = pilot._load_session()
    pilot_meta = json.loads((pilot.PILOT_DIR/'metadata.json').read_text())
    for key in ('config', 'checkpoint'):
        item = (pilot_meta['selected_models'] if key == 'config' else pilot_meta['selected_checkpoints'])['roi_ssd_mobilenet']
        assert pilot._sha256(pilot.ROOT/item[key if key == 'config' else 'path']) == item[key+'_sha256' if key == 'config' else 'sha256']
    protected = {str(p.relative_to(pilot.ROOT)): pilot._sha256(p) for p in pilot.PILOT_DIR.rglob('*') if p.is_file()}
    sources = {p: pilot._sha256(pilot.ROOT/p) for p in SOURCE_FILES}
    meta = {'status': 'RUNNING', 'started_at': datetime.now(timezone.utc).isoformat(),
            'pilot_manifest_sha256': pilot._sha256(pilot.PILOT_DIR/'pilot_manifest.txt'),
            'pilot_metadata_sha256': pilot._sha256(pilot.PILOT_DIR/'metadata.json'),
            'checkpoint': pilot_meta['selected_checkpoints']['roi_ssd_mobilenet'],
            'config': pilot_meta['selected_models']['roi_ssd_mobilenet'],
            'device': 'cpu', 'torch_threads': 8, 'torch_interop_threads': 1, 'opencv_threads': 1,
            'padding_model_tensor_pixels': 96, 'context_probe_paddings': PADDINGS,
            'hold': 0, 'full_frame_interval': 10, 'merge': False, 'confidence': .5, 'nms_iou': .45,
            'git_revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'source_sha256': sources, 'production_and_pilot_unchanged': False}
    write_json(OUT/'metadata.json', meta)
    model, dataset, config, threshold = pilot._load_model('roi_ssd_mobilenet', torch.device('cpu'))
    assert threshold == .5
    model.record_execution_trace = True
    weight_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    names = dataset.idx2label
    indices = pilot._dataset_indices(dataset, manifests['pilot'], pilot._dataset_root(config))
    assert len(indices) == 320 and len({r['video_id'] for r in manifests['pilot']}) == 32
    saved = {}
    with (pilot.PILOT_DIR/'frame_level_metrics.csv').open(newline='', encoding='utf-8') as stream:
        for r in csv.DictReader(stream):
            if r['model'] == 'roi_ssd_mobilenet' and r['run'] in ('padding_full_frame', 'padding_96'):
                saved[(r['run'], r['video_id'], int(r['frame_index']))] = r
    assert len(saved) == 640
    modes = ['FF', 'ORACLE_GT_SEG', 'PRED_SEG', 'ORACLE_GT_SEG_NO_GLOBAL_NMS',
             'ORACLE_GT_SEG_P32', 'ORACLE_GT_SEG_P64', 'ORACLE_GT_SEG_P128']
    records = {m: [] for m in modes}
    geometries, input_hashes = [], []
    traces = []
    def observe(module, args, output):
        trace = module.get_runtime_log()
        assert trace['depth_policy'] == 'until_1x1'
        assert trace['N_loc_predictions'] == trace['N_cls_predictions'] == trace['N_anchors']
        traces.append(copy.deepcopy(trace))
    hook = model.register_forward_hook(observe)
    tracker, video, next_rois = None, None, []
    with torch.inference_mode():
        for number, (batch, manifest) in enumerate(zip(pilot._dataloader(dataset, indices), manifests['pilot']), 1):
            tensor, target, fname = batch
            image_path = Path(fname[0])
            frame = cv2.imread(str(image_path))
            h, w = frame.shape[:2]
            gt, diff = pipeline.extract_gt_for_map(target, names, w, h)
            assert not any(v for values in diff.values() for v in values)
            vid = str(pilot._first(target.get('video_id', manifest['video_id'])))
            index = int(pilot._first(target.get('frame_idx', manifest['frame_index'])))
            info = {'video_id': vid, 'frame_index': index, 'image': str(image_path), 'width': w, 'height': h}
            input_hashes.append({'image': manifest['image'], 'image_sha256': pilot._sha256(image_path),
                                 'tensor_sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest()})
            if tracker is None or video != vid or bool(pilot._first(target.get('is_first_frame', False))):
                tracker = StaticPaddingTracker(96, 96, threshold, (320,320), hold_last_for_frames=0)
                next_rois, video = [], vid
            for mode in ('FF', 'PRED_SEG'):
                traces.clear()
                pre = []
                nms = pipeline.merge_detections_nms
                def capture(detections, iou_threshold):
                    pre.extend(copy.deepcopy(detections))
                    return nms(detections, iou_threshold)
                ff_tracker = StaticPaddingTracker(96,96,threshold,(320,320),hold_last_for_frames=0)
                with patch.object(pipeline, 'merge_detections_nms', side_effect=capture):
                    result = pipeline.process_frame(model=model, idx2label=names, frame_bgr=frame,
                        im_tensor=tensor.float(), tracker=ff_tracker if mode == 'FF' else tracker,
                        next_frame_rois=[] if mode == 'FF' else next_rois,
                        frame_idx=index, key_frame_interval=10, im_size_hw=(320,320),
                        conf_threshold=threshold, nms_iou=.45, merge_fn=lambda rois, **_: list(rois),
                        model_device=torch.device('cpu'), roi_grid=1)
                if mode == 'PRED_SEG':
                    next_rois = result.next_frame_rois
                r = frame_record(mode, result.final_detections, pre, gt, diff, result.rois_used,
                                 traces, result.use_full_frame, info)
                reference = saved[('padding_full_frame' if mode == 'FF' else 'padding_96', vid, index)]
                assert json.loads(reference['ground_truth_records']) == gt
                old = json.loads(reference['prediction_records'])
                assert len(old) == len(r['detections'])
                for a,b in zip(old,r['detections']):
                    assert a['bbox'] == b['bbox'] and a['class'] == b['class'] and abs(a['confidence']-b['confidence']) <= 1e-6
                records[mode].append(r)
            for padding in PADDINGS:
                traces.clear()
                pre, rois, own = oracle(model, tensor, gt, names, (h,w), padding, threshold, traces)
                dets = postprocess(pre, threshold, w, h)
                mode = 'ORACLE_GT_SEG' if padding == 96 else f'ORACLE_GT_SEG_P{padding}'
                r = frame_record(mode, dets, pre, gt, diff, rois, traces, False, info)
                records[mode].append(r)
                if padding == 96:
                    matched = match_objects(dets, gt)['matched']
                    for own_row in own:
                        cls, j = own_row['class'], own_row['gt_index_in_class']
                        same = [d for d in dets if d['class'] == cls]
                        geometries.append({**info, **own_row,
                            'matched50_after_global_nms': (cls,j) in matched,
                            'best_prediction_iou': max((get_iou(d['bbox'], own_row['gt_box']) for d in same), default=0)})
                    no_nms = postprocess(pre, threshold, w, h, nms=False)
                    records['ORACLE_GT_SEG_NO_GLOBAL_NMS'].append(frame_record(
                        'ORACLE_GT_SEG_NO_GLOBAL_NMS', no_nms, pre, gt, diff, rois, traces, False, info))
            if number % 20 == 0:
                print(f'Stage 3a {number}/320: FF/PRED replay verified, oracle p=32/64/96/128 complete', flush=True)
    hook.remove()
    summaries = [summarize(m, records[m]) for m in modes]
    for mode, run in [('FF','padding_full_frame'), ('PRED_SEG','padding_96')]:
        with (pilot.PILOT_DIR/'padding_sweep.csv').open(newline='',encoding='utf-8') as stream:
            old, = [r for r in csv.DictReader(stream) if r['model']=='roi_ssd_mobilenet' and r['run']==run]
        fresh = next(r for r in summaries if r['mode']==mode)
        assert all(abs(fresh[k]-float(old[k]))<=1e-6 for k in ('mAP50','mAP50_95','recall50'))
    write_csv('mode_summary.csv', summaries)
    write_csv('per_object_crop_geometry.csv', geometries)
    bins = object_bins(geometries)
    write_csv('object_ratio_bins.csv', bins)
    comparison, duplicate_rows = [], []
    for i in range(320):
        base = {k: records['FF'][i][k] for k in ('video_id','frame_index','image')}
        row = dict(base)
        for mode in ('FF','ORACLE_GT_SEG','PRED_SEG'):
            r = records[mode][i]
            row.update({mode+'_'+k: r[k] for k in ('frame_map50','frame_recall50','tp','fp','gt_count','gt_covered','full_frame')})
        row['FF_to_ORACLE_map50_drop'] = row['FF_frame_map50']-row['ORACLE_GT_SEG_frame_map50']
        row['ORACLE_to_PRED_map50_drop'] = row['ORACLE_GT_SEG_frame_map50']-row['PRED_SEG_frame_map50']
        comparison.append(row)
        for mode in modes:
            r = records[mode][i]
            duplicate_rows.append({**base, 'mode': mode, **{k:r[k] for k in (
                'before_nms_count','after_nms_count','duplicates','same_class_pairs_iou_gt_05',
                'same_class_pairs_iou_gt_07','pre_same_class_pairs_iou_gt_05','pre_same_class_pairs_iou_gt_07')}})
    write_csv('frame_level_comparison.csv', comparison)
    write_csv('duplicate_nms_analysis.csv', duplicate_rows)
    write_csv('input_hashes.csv', input_hashes)
    with (OUT/'frame_records.jsonl').open('w',encoding='utf-8') as stream:
        for mode in modes:
            for record in records[mode]:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
    export_overlays(records)
    make_report(summaries, bins, meta)
    assert hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest() == weight_hash
    assert all(pilot._sha256(pilot.ROOT/p)==digest for p,digest in sources.items())
    assert all(pilot._sha256(pilot.ROOT/p)==digest for p,digest in protected.items())
    meta.update(status='COMPLETE', completed_at=datetime.now(timezone.utc).isoformat(),
                production_and_pilot_unchanged=True, weights_unchanged=True,
                num_videos=32, num_frames=320, num_gt=len(geometries),
                saved_ff_pred_predictions_replayed=640, overlays=40,
                pilot_acceptance_unchanged=True, ca_fulltech_pi_allowed=False,
                output_sha256={str(p.relative_to(OUT)):pilot._sha256(p) for p in OUT.rglob('*')
                               if p.is_file() and p.name!='metadata.json'})
    write_json(OUT/'metadata.json', meta)
    print('COMPLETE', meta['case'], OUT, flush=True)


if __name__ == '__main__':
    main()
