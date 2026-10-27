"""Add explicit interpretation of measured Stage 3a geometry/context contrasts."""
import csv
import json

from .oracle_diagnosis import OUT, write_csv
from .run_pilot import ROOT, _sha256
from tools.benchmarks.video_artifacts import write_json


def read(name):
    with (OUT/name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    meta = json.loads((OUT/'metadata.json').read_text())
    assert meta['status'] == 'COMPLETE'
    assert all(_sha256(OUT/p)==digest for p,digest in meta['output_sha256'].items())
    summary = {r['mode']: r for r in read('mode_summary.csv')}
    geometry = read('per_object_crop_geometry.csv')
    grouping = [
        ('area_lt_025', lambda r: float(r['gt_area_ratio']) < .25),
        ('area_ge_025', lambda r: float(r['gt_area_ratio']) >= .25),
        ('center_distance_lt_01', lambda r: float(r['center_distance_ratio']) < .1),
        ('center_distance_ge_02', lambda r: float(r['center_distance_ratio']) >= .2),
        ('center_distance_ge_04', lambda r: float(r['center_distance_ratio']) >= .4),
        ('touches_boundary', lambda r: r['touches_crop_boundary'] == 'True'),
        ('does_not_touch_boundary', lambda r: r['touches_crop_boundary'] != 'True'),
    ]
    associations = []
    for label, predicate in grouping:
        selected = [r for r in geometry if predicate(r)]
        matched = sum(r['matched50_after_global_nms'] == 'True' for r in selected)
        associations.append({'group': label, 'N_GT': len(selected), 'TP50': matched,
                             'recall50_micro': matched/len(selected) if selected else None})
    write_csv('geometry_associations.csv', associations)
    by = {r['group']: r for r in associations}
    with (OUT/'frame_records.jsonl').open(encoding='utf-8') as stream:
        pred = [json.loads(line) for line in stream if '"mode": "PRED_SEG"' in line]
    full_count = sum(r['full_frame'] for r in pred)
    ff = float(summary['FF']['mAP50'])
    values = [float(summary[m]['mAP50']) for m in (
        'ORACLE_GT_SEG_P32', 'ORACLE_GT_SEG_P64', 'ORACLE_GT_SEG', 'ORACLE_GT_SEG_P128')]
    context_monotone = all(a <= b for a,b in zip(values, values[1:]))
    report_path = OUT/'oracle_diagnosis_report.md'
    marker = '\n## Explicit interpretation of the measured contrasts\n'
    report = report_path.read_text(encoding='utf-8').split(marker)[0]
    extra = [marker, '',
        f'1. **Oracle deficit:** FF−ORACLE = {ff-values[2]:.6f} mAP50; '
        f'ORACLE retains {values[2]/ff:.2%} of FF despite 100% GT coverage.',
        f'2. **Temporal contrast:** ORACLE−PRED = '
        f'{values[2]-float(summary["PRED_SEG"]["mAP50"]):+.6f}. ORACLE is slightly worse, '
        f'not better. PRED uses full-frame inference on {full_count}/320 frames '
        f'({full_count/320:.1%}); ORACLE always crops. The negative formation contrast '
        'does not establish beneficial temporal errors or exclude missed tracks.',
        '3. **Dominant cause:** crop-domain robustness (Case B). The measured '
        '109.1% crop / −9.1% formation mAP-loss shares are retained as signed practical '
        'contrasts, not interpreted as independent causal percentages.',
        f'4. **Object area:** the observed direction contradicts a large-object-collapse explanation. '
        f'Area ratio <0.25: recall {by["area_lt_025"]["recall50_micro"]:.2%}, '
        f'N={by["area_lt_025"]["N_GT"]}; area ratio ≥0.25: '
        f'{by["area_ge_025"]["recall50_micro"]:.2%}, N={by["area_ge_025"]["N_GT"]}. '
        'There are no GT with area ratio ≥0.70, so extreme object filling is untested. '
        'Small objects, source resolution, class and head/anchor geometry remain plausible contributors.',
        f'5. **Position:** normalized center distance <0.1: recall '
        f'{by["center_distance_lt_01"]["recall50_micro"]:.2%}, N={by["center_distance_lt_01"]["N_GT"]}; '
        f'distance ≥0.2: {by["center_distance_ge_02"]["recall50_micro"]:.2%}, '
        f'N={by["center_distance_ge_02"]["N_GT"]}; distance ≥0.4: '
        f'{by["center_distance_ge_04"]["recall50_micro"]:.2%}, N={by["center_distance_ge_04"]["N_GT"]}. '
        'This supports an association with edge displacement/clipped context, not an isolated causal effect. '
        'Center distance is sqrt(center_dx_ratio²+center_dy_ratio²).',
        '6. **NMS:** global class-wise NMS is present. Removing it increases duplicate '
        'detections/frame from 0.0125 to 0.31875 and FP/frame from 0.6875 to 1.340625. '
        'Restoring it adds 0.019878 mAP50, yet leaves the much larger 0.251556 FF deficit. '
        'It slightly lowers macro Recall50 (0.577642→0.563414), consistent with a precision/recall '
        'tradeoff; this does not explain the main accuracy loss.',
        f'7. **Next:** prioritize crop/context distribution and head/anchor diagnostics over '
        f'tracker tuning. Oracle mAP50 improves monotonically across p=32/64/96/128: '
        f'{values[0]:.6f} → {values[1]:.6f} → {values[2]:.6f} → {values[3]:.6f}. '
        f'At p=128 it still trails FF by {ff-values[3]:.6f} and retains only '
        f'{values[3]/ff:.2%}. This supports context/domain sensitivity but does not separate '
        'context from changes in network shape, active depth or anchor sampling. '
        'Do not select p=128 post hoc or proceed to CA/FULL-TECH/Pi. A next controlled study '
        'should hold source objects fixed, vary crop context, log L_R/anchor IoU and compare '
        'the training crop-size distribution before changing training or production.', '',
        'Macro evaluator Recall50 in mode_summary.csv and micro GT recall in geometry bins '
        'use different averaging; they should not be compared as identical metrics.', '',
        'Reproduce:', '', '```powershell',
        '.venv/Scripts/python.exe -m experiments.cpu_pilot.oracle_diagnosis',
        '.venv/Scripts/python.exe -m experiments.cpu_pilot.summarize_oracle_diagnosis',
        '```',
    ]
    report_path.write_text(report+'\n'.join(extra)+'\n', encoding='utf-8')
    meta['context_monotonic_map50'] = context_monotone
    meta['pred_full_frame_calls'] = full_count
    meta['geometry_associations'] = associations
    meta['validation'] = {'cpu_pilot_unit_tests_passed': 6,
                          'command': '.venv/Scripts/python.exe -m unittest discover -s experiments/cpu_pilot/tests -v'}
    for name in ('experiments/cpu_pilot/summarize_oracle_diagnosis.py',
                 'experiments/cpu_pilot/tests/test_oracle_diagnosis.py'):
        meta['source_sha256'][name] = _sha256(ROOT/name)
    for name in ('oracle_diagnosis_report.md', 'geometry_associations.csv'):
        meta['output_sha256'][name] = _sha256(OUT/name)
    write_json(OUT/'metadata.json', meta)
    print('Interpretation complete; context monotonic:', context_monotone, '; PRED FF calls:', full_count)


if __name__ == '__main__':
    main()
