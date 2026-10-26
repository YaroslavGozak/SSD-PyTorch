"""Refresh checkpoint-role recommendations from saved results, without inference."""
import csv
import json
import math
import os
from pathlib import Path
import statistics
import re

import yaml

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT/'final_experiments/audit'
BATCH = ROOT/'outputs/batch_infer'
START = '<!-- checkpoint-role-recommendations:start -->'
END = '<!-- checkpoint-role-recommendations:end -->'


def collect():
    summaries = []
    for config in sorted(BATCH.rglob('config.yaml')):
        cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
        train, data = cfg['train_params'], cfg['dataset_params']
        row = {'checkpoint': 'trained_models/'+config.parent.relative_to(BATCH).as_posix(),
               'family': train['model'], 'dataset': train['dataset'],
               'dataset_yaml': data.get('yolo_dataset_yaml'),
               'classes': train.get('model_num_classes', data.get('num_classes')),
               'confidence': train.get('infer_conf_threshold'),
               'penalized': train.get('penalized_roissd', False),
               'config': config.relative_to(ROOT).as_posix()}
        for mode, name in [('pad', 'pad-loop/results.csv'), ('size', 'fixed-size-loop/fixed_size_results.csv')]:
            path = config.parent/name
            if not path.exists():
                continue
            records = list(csv.DictReader(path.open(encoding='utf-8-sig')))
            values = {}
            conflicts = []
            for record in records:
                try:
                    value = float(record['mAP'])
                    key = record['evaluated_dataset']
                    if not math.isfinite(value):
                        continue
                except (KeyError, TypeError, ValueError):
                    continue  # An active run can have a partially written last row.
                if key in values and values[key] != value:
                    conflicts.append(key)
                values[key] = value
            expected = ({f'fixed_padding_roi_crop_{p}' for p in range(0, 201, 10)} if mode == 'pad'
                        else {f'fixed_size_{h}x{w}' for h in range(32,321,32) for w in range(32,321,32)})
            if train['model'] == 'yolo':
                expected = {key.replace('fixed_padding_roi_crop_', 'fixed_padding_roi_crop_yolo_')
                            .replace('fixed_size_', 'fixed_size_yolo_') for key in expected}
            row[mode] = {'count': len(values), 'complete': set(values) == expected and not conflicts,
                         'conflicting_duplicates': conflicts, 'source': path.relative_to(ROOT).as_posix()}
            if values:
                row[mode]['mean_map50'] = statistics.mean(values.values())
            if mode == 'size':
                sizes = {tuple(map(int,re.search(r'(\d+)x(\d+)$',key).groups())):value for key,value in values.items()}
                small = [value for (h,w),value in sizes.items() if max(h,w)<=160]
                row[mode]['small_mean_map50'] = statistics.mean(small) if small else None
                row[mode]['squares'] = {str(h): value for (h,w),value in sizes.items() if h==w}
        if 'pad' in row or 'size' in row:
            summaries.append(row)
    return summaries


def update():
    summaries = collect()
    by_path = {r['checkpoint']:r for r in summaries}
    previous = json.loads((AUDIT/'selected_models.json').read_text(encoding='utf-8'))
    roles = {}
    for family, roi, alternative in [
        ('ROI-SSD', 'trained_models/best-imagenet-vid-roissd/imagenet26022026stage3.pt', None),
        ('ROI-SSD-MobileNet', 'trained_models/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt',
         'trained_models/best-imagenet-vid-roissd-mobilenet/20260928-best-phase3/best_map.pt'),
    ]:
        old = previous['models'][family]
        row = by_path[roi]
        assert row['pad']['complete'] and row['size']['complete'] and not row['penalized']
        assert row['classes']==31 and row['dataset']=='yolo-imagenet-vid'
        roles[family] = {'full_frame_checkpoint':old['checkpoint'],
                         'full_frame_map50':old['map50'], 'full_frame_map50_95':old['map50_95'],
                         'roi_checkpoint':roi, 'roi_evidence':row,
                         'small_roi_alternative':alternative,
                         'pair_status':'RECOMMENDED_FOR_END_TO_END_VALIDATION',
                         'full_frame_config':old.get('final_config'), 'roi_config':row['config']}
    assert all((ROOT/r['full_frame_checkpoint']).is_file() and (ROOT/r['roi_checkpoint']).is_file() for r in roles.values())
    recommendation = {'updated':'2026-10-23', 'penalized_candidates':'EXCLUDED_BY_USER',
                      'selection_scope':'individual role specialists; no measured pair performance',
                      'families':roles, 'batch_snapshot':summaries,
                      'note':'selected_models.json retains the original canonical full-frame audit; use these explicit roles for paired experiments.'}
    (AUDIT/'model_role_recommendations.json').write_text(json.dumps(recommendation,indent=2)+'\n',encoding='utf-8')

    for target in [AUDIT/'audit_report.md', ROOT/'final_experiments/FINAL_EXPERIMENTS_MANIFEST.md']:
        def link(path, label=None):
            return f'[{label or Path(path).name}]({Path(os.path.relpath(ROOT/path,target.parent)).as_posix()})'
        def checkpoint(path):
            return link(path, path.removeprefix('trained_models/'))
        complete = sum(r.get('pad',{}).get('complete',False) and r.get('size',{}).get('complete',False) for r in summaries)
        lines = [START, '## Updated individual models and same-family pairs (2026-10-23)', '',
                 '**This section supersedes the original single-checkpoint recommendation for ROI processing.** The canonical full-frame results below remain historical evidence for that role. Penalized ROI-SSD candidates are excluded at the user\'s request.', '',
                 f'Saved batch results currently cover **{len(summaries)} checkpoints; {complete} have both complete sweeps**. Partial or conflicting sweeps, penalized wrappers, legacy class spaces, and other datasets are excluded from the ROI shortlist. Only non-penalized 31-output models on the sampled 30-foreground-class VID schema support the recommendations below.', '',
                 '### Individual best models by role', '',
                 'There is no established universal winner for both roles. Full-frame specialists come from the original 256-frame canonical evaluation at confidence 0.01. ROI specialists come from the saved 5,064-frame crop sweeps at configured confidence 0.5. These scores measure different protocols and must not be compared directly.', '',
                 '| Family | Role | Checkpoint | Evidence |', '|---|---|---|---|']
        for family,r in roles.items():
            e=r['roi_evidence']
            lines += [f'| {family} | Best full-frame candidate in canonical shortlist | {checkpoint(r["full_frame_checkpoint"])} | mAP50 {r["full_frame_map50"]:.2%}; mAP50:95 {r["full_frame_map50_95"]:.2%}. {link("final_experiments/audit/canonical_results.json","Canonical results")} |',
                      f'| {family} | Recommended individual ROI checkpoint | {checkpoint(r["roi_checkpoint"])} | Mean padding mAP50 {e["pad"]["mean_map50"]:.2%}; mean fixed-size mAP50 {e["size"]["mean_map50"]:.2%}. {link(e["pad"]["source"],"Padding CSV")}; {link(e["size"]["source"],"size CSV")} |']
        lines += ['', 'The ROI averages are descriptive: padding averages all 21 settings from 0 to 200 px; fixed-size averages all 100 height/width combinations from 32 to 320 px in 32-pixel steps. They are not a pooled mAP and do not represent deployment-frequency weights.', '',
                  '### What changed relative to the original selection', '',
                  '| Family | Previous full-frame selection: mean padding / size mAP50 | Recommended ROI specialist: mean padding / size mAP50 |', '|---|---:|---:|']
        for family,r in roles.items():
            old=by_path[r['full_frame_checkpoint']]; new=r['roi_evidence']
            lines.append(f'| {family} | {old["pad"]["mean_map50"]:.2%} / {old["size"]["mean_map50"]:.2%} | {new["pad"]["mean_map50"]:.2%} / {new["size"]["mean_map50"]:.2%} |')
        small=by_path[roles['ROI-SSD-MobileNet']['small_roi_alternative']]
        phase2=roles['ROI-SSD-MobileNet']['roi_evidence']
        lines += ['', '### MobileNet alternatives', '',
                  f'- For smaller crops, retain {checkpoint(small["checkpoint"])}: at 128x128 its mAP50 is {small["size"]["squares"]["128"]:.2%}, versus {phase2["size"]["squares"]["128"]:.2%} for the recommended phase-2 checkpoint. Across the 25 shapes with both sides <=160 px, its mean is {small["size"]["small_mean_map50"]:.2%}, versus {phase2["size"]["small_mean_map50"]:.2%}. {link(small["size"]["source"],"Size CSV")}.',
                  '- The phase2-bn-freeze `last.pt` has a slightly higher fixed-size mean (24.85%) than `best_map.pt` (24.72%), but lower padding mean (49.88% versus 51.53%). This does not establish a universal winner or statistical significance; the recommended checkpoint favors padding robustness, with the phase-3 alternative retained for small crops.', '',
                  '### Recommended pairs', '', '| Family | Full-frame / discovery / fallback | ROI processing between full-frame calls |', '|---|---|---|']
        for family,r in roles.items():
            lines.append(f'| {family} | {checkpoint(r["full_frame_checkpoint"])} | {checkpoint(r["roi_checkpoint"])} |')
        lines += ['', 'Each pair uses two checkpoints from the same architecture family, the same foreground class order, and SSD background index 0. Both outputs must be converted into the same source-frame coordinates and update the same tracker. Full-frame calls initialize detection, discover objects, and handle fallback; ROI calls process intervening frames. This is role switching, not prediction averaging.', '',
                  '**Pair status: recommended for validation, not yet demonstrated superior end to end.** Compare full-frame/full-frame, ROI/ROI, full-frame/ROI, and the reversed pair on identical videos, tracker settings, ROI generation, full-frame schedule and fallback rules. Measure mAP, recall, end-to-end latency, full-frame invocation rate, and peak memory with both checkpoints resident. No pair accuracy or Pi memory/latency is inferred from separate crop tests.', '',
                  f'Explicit role paths and evidence: {link("final_experiments/audit/model_role_recommendations.json","machine-readable role recommendations")}. Original full-frame configs remain unchanged. ROI configs linked there are the saved batch configs (confidence 0.5), not a claim that a mixed pipeline has been implemented.', '',
                  '### Remaining limitations', '',
                  '- Full-frame winners were selected among three canonical candidates per family, not every checkpoint on full frames. The later ROI sweeps do not expand that full-frame comparison.',
                  '- Near ties can reflect crop jitter because fixed-size crops use randomness without a frozen per-image geometry manifest. Use shared deterministic crops before resolving small differences.',
                  '- The updated YOLO adapter fixes preprocessing and prediction-setting propagation, but corrected sweeps must finish before a new YOLO ranking is published. YOLO is outside the two-family paired recommendation here.',
                  '- Penalized checkpoints are excluded; no rerun or ranking of those intermediate experiments is required.',
                  '- Existing dataset limitations still apply: 30 schema classes, only 27 with supplied GT, and validation reused for selection rather than an untouched test split.', '', END]
        text=target.read_text(encoding='utf-8')
        if START in text:
            a=text.index(START);b=text.index(END,a)+len(END)
            text=text[:a]+text[b:]
        title,body=text.split('\n',1)
        target.write_text(title+'\n\n'+'\n'.join(lines)+'\n'+body.lstrip('\n'),encoding='utf-8')
    print(f'Updated report and manifest: {len(summaries)} checkpoints, {complete} complete; two explicit same-family pairs.')


if __name__=='__main__':
    update()
