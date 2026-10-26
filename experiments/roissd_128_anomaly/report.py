"""Generate a diagnostic report from measured data, never from assumed causes."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(out,name):
    with (out/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))


def build_report(out,lock):
    metadata=json.loads((out/'run_metadata.json').read_text())
    metrics=read(out,'size_sweep_metrics.csv')
    shapes=read(out,'internal_shapes.csv')
    matching=read(out,'anchor_matching.csv')
    complete=(metadata['status']=='COMPLETE' and metadata['num_samples']>=200
              and metadata['sizes']==list(range(96,161,4)))
    text=['# ROI-SSD 128x128 diagnostic report', '',
          f'Run status: **{metadata["status"]}**. '+('Complete stage-2 dense diagnostic sweep.' if complete else '**SMOKE/PARTIAL evidence only; stage 2 is not complete.**'), '',
          '## Checkpoints and protocol', '',
          'Checkpoints are the ROI-role specialists explicitly selected in the updated stage-1 manifest; full-frame specialists are not substituted. Penalized candidates are excluded.', '']
    for m in lock['models']:
        text += [f'- {m["model_family"]}: `{m["checkpoint"]}`; SHA256 `{m["checkpoint_sha256"]}`.']
    text += ['',f'{metadata["num_samples"]} fixed canonical source examples, shared across both models and all sizes. '+
             f'Confidence {metadata["confidence"]}, NMS IoU {metadata["nms_iou"]}, max detections {metadata["max_detections"]}. '+
             'RGB project FixedSizeTestTransform, ImageNet normalization, per-image seed reused across sizes and families. Requested dimensions are fed exactly: no stride rounding or silent padding. Foreground IDs 1..30 correspond to dataset IDs 0..29; background is index 0.', '',
             'The transform crops around annotated GT and can retain a different number of visible GT objects at different sizes. This is an object-centered ROI diagnostic, not whole-frame downsampling or a tracker benchmark. Tensor/GT hashes verify the same inputs across families.', '',
             'mAP50:95 is the mean of the existing project area-integral AP at ten IoU thresholds, not COCO 101-point interpolation. Absent-class AP is undefined. Pre-NMS counts are class-wise candidates after confidence/top-k filtering; decoded box slots are recorded separately. Max-confidence mean uses unthresholded foreground softmax maxima; zero post-NMS scores are recorded separately.', '',
             '## Structural boundaries', '', '| Family | Input size | Active depth | Heads (zero-based) | Feature maps | Anchors | mAP50 |', '|---|---:|---:|---|---|---:|---:|']
    for r in metrics:
        if r['status']!='PASS':continue
        ss=[s for s in shapes if s['model_family']==r['model_family'] and s['requested_h']==r['requested_h'] and s['head_active']=='True']
        if not ss:continue
        features=[f'{s["feature_h"]}x{s["feature_w"]}' for s in ss]
        text += [f'| {r["model_family"]} | {r["requested_h"]} | {ss[0]["active_backbone_depth"]} | {", ".join(s["head_index"] for s in ss)} | {", ".join(features)} | {ss[0]["total_anchors"]} | {float(r["map50"]):.4f} |']
    conclusions={}
    neighboring_hazards={}
    for family in ['ROI-SSD','ROI-SSD-MobileNet']:
        text += ['',f'## {family}', '']
        ms={int(r['requested_h']):r for r in metrics if r['model_family']==family and r['status']=='PASS'}
        am={int(r['requested_h']):r for r in matching if r['model_family']==family}
        if not all(size in ms for size in [96,128,160]):
            text += ['Insufficient 96/128/160 evidence. No anomaly classification or stage-3 clearance.']
            conclusions[family]='INCOMPLETE';continue
        text += ['| Input | mAP50 | mAP50:95 | Recall50 | GT | Candidates pre-NMS | Detections/image | Mean max confidence | Mean best anchor IoU |',
                 '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for size in [96,128,160]:
            r=ms[size];a=am[size]
            text += [f'| {size} | {float(r["map50"]):.4f} | {float(r["map5095"]):.4f} | {float(r["recall50"]):.4f} | {r["num_gt"]} | {float(r["predictions_pre_nms_mean"]):.2f} | {float(r["predictions_post_nms_mean"]):.2f} | {float(r["max_confidence_mean"]):.4f} | {float(a["max_anchor_iou_mean"]):.4f} |']
        text += ['', 'Per-head evidence (each entry is an average over examples or GT, not a confidence-selected matching statistic):', '',
                 '| Input | Head | Feature map | Mean max foreground confidence | Mean best anchor IoU | Mean best decoded IoU before confidence/NMS |',
                 '|---:|---:|---|---:|---:|---:|']
        for s in shapes:
            if s['model_family']==family and s['head_active']=='True' and int(s['requested_h']) in [96,128,160]:
                text += [f'| {s["requested_h"]} | {s["head_index"]} | {s["feature_h"]}x{s["feature_w"]} | {float(s["foreground_max_confidence_mean"]):.4f} | {float(s["head_max_anchor_iou_mean"]):.4f} | {float(s["head_max_decoded_iou_mean"]):.4f} |']
        native128=[s for s in shapes if s['model_family']==family and s['requested_h']=='128' and s['head_active']=='True']
        same_depth=[int(s['requested_h']) for s in shapes if s['model_family']==family and s['head_index']=='0'
                    and s['active_backbone_depth']==native128[0]['active_backbone_depth']]
        text += ['', f'128 uses depth {native128[0]["active_backbone_depth"]}; that same depth is used at tested sizes {", ".join(map(str,sorted(same_depth)))}. '
                 'A 128 accuracy valley is therefore not evidence of a depth switch occurring exactly at 128. Feature-grid resolution and head confidence must be checked separately.']
        # This classifies a valley, not statistical significance. Preserve the
        # numeric differences and distinguish suspected mechanisms from proof.
        valley=float(ms[128]['map50'])<min(float(ms[96]['map50']),float(ms[160]['map50']))
        matching_valley=float(am[128]['max_anchor_iou_mean'])<min(float(am[96]['max_anchor_iou_mean']),float(am[160]['max_anchor_iou_mean']))-.05
        if valley:
            result='B_SUSPECTED' if matching_valley else 'C_CONSISTENT_NOT_CAUSALLY_PROVEN'
            text += ['', '**128x128 valley reproduced on this checkpoint.**',
                     'Classification: **B, suspected anchor/matching sensitivity**.' if matching_valley else
                     'Classification: **C, consistent with feature/head or training-distribution sensitivity; a causal learned-feature explanation is not proven**.',
                     'All observed localization/classification/anchor box-slot counts agree. '+
                     ('Mean best-anchor IoU also drops by over 0.05 relative to both controls; inspect per-GT distributions before attributing causality.' if matching_valley else
                      'The mean best-anchor IoU does not show a corresponding >0.05 valley relative to both controls. This rules out a gross slot-count mismatch; it does not by itself prove anchor matching is optimal.'),
                     'Compare 124/128/132 as well as the 96->100 and 140->144 boundaries in the table above. Depth selection is size-dependent; correlation with a feature-map/head boundary is evidence of sensitivity, not proof that the adaptive-depth code is incorrect.',
                     'The per-head table separates two mechanisms: which head has confidence high enough to survive filtering, and whether that head has anchors/regressed boxes that cover GT well. A shift toward a coarser confident head while a better-localizing head loses confidence supports joint head-specialization/feature-grid sensitivity. Anchor coverage can still constrain quality even when counts and the anchor-reference implementation agree.',
                     '**No implementation fix is justified solely by this accuracy valley.** No production model or checkpoint was changed. A causal head/depth intervention or training-distribution study is a follow-up if needed.']
        else:
            result='D_NO_128_VALLEY'
            text += ['', '**D: the historical 128x128 valley did not reproduce on this selected checkpoint.**',
                     'Do not invent a cause for the old point. Use these checkpoint/protocol-specific measurements when replacing historical figures. This does not establish good accuracy at every tested size.']
        low_sizes=[size for size,r in ms.items() if 96<size<160
                   and float(r['map50'])<.5*min(float(ms[128]['map50']),float(ms[160]['map50']))]
        neighboring_hazards[family]=sorted(low_sizes)
        if low_sizes:
            text += ['', '**Additional neighboring-size sensitivity:** '+', '.join(map(str,sorted(low_sizes)))+
                     ' px have mAP50 below half of both the 128 and 160 control values. '+
                     'This is a descriptive screening rule, not a significance test or an acceptable-accuracy threshold.',
                     'Do not interpret a non-reproduced 128 valley as proof of robust arbitrary-size inference. '+
                     'Inspect the same-depth feature-grid transition, per-head confidence and anchor matching at these sizes. '+
                     'Counts and reference agreement can pass while the learned detector is ineffective on real segment sizes. '+
                     'An explicit padding/size policy would need a separate accuracy check; no silent rounding or model fix was applied.']
        conclusions[family]=result
    errors=(out/'errors_or_assertions.log').read_text()
    structurally_safe=complete and not errors and metadata['structural_checks_passed']
    mobile_quality_hazard=bool(neighboring_hazards.get('ROI-SSD-MobileNet'))
    text += ['', '## Anchor geometry and normalization', '',
             'The independent center/size reference checks anchor order and normalized xyxy geometry against actual generated anchors. Statistics include both the returned clipped anchors and the reconstructed pre-clipping boxes; returned anchors may all be in bounds because the model clamps them. Normalized widths/heights are also converted using actual W/H into pixel dimensions. Rectangular probes distinguish width/height bookkeeping from square-only evidence.',
             'Positive matches per GT at IoU >=0.5 are confidence-independent. A separate assigned-positive count reproduces one-GT-per-anchor matching and forced best-anchor assignment, which need not equal the simple pairwise threshold count.', '',
             '## Stage-3 gate', '',
             ('**Structurally eligible on the tested MobileNet sizes and rectangular probes.** No observed slot-count, anchor-reference, finite-output, class-ID or coordinate-bound failure. This is not a guarantee of every possible segment shape or acceptable accuracy below the useful ROI-size range.' if structurally_safe else
              '**BLOCKED:** full diagnostic scope or structural checks are incomplete/failed. Inspect errors_or_assertions.log. Do not infer clearance from a smoke run.'),
             'The two-checkpoint video pipeline itself has not been benchmarked here. Retraining was not performed or assumed necessary.', '',
             '## Evidence and reproduction', '',
             '- [Size sweep](size_sweep_metrics.csv)', '- [Internal shapes and per-head confidence/IoU](internal_shapes.csv)',
             '- [Anchor geometry](anchor_geometry.csv)', '- [Anchor matching](anchor_matching.csv)',
             '- [Per-GT matching](anchor_matching_per_gt.jsonl)', '- [Per-class AP](per_class_ap.csv)',
             '- [Per-example predictions and GT](per_example.jsonl)', '- [Preprocessing hashes](preprocessing.csv)',
             '- [Rectangular probes](rectangular_probes.csv)', '- [Errors/assertions](errors_or_assertions.log)',
             '- [Runtime metadata](run_metadata.json)', '',
             '```powershell', '.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.run --device auto', '```', '']
    plot_specs=[('map50_vs_input_size.png','mAP@0.5',lambda f,s:float(next(r['map50'] for r in metrics if r['model_family']==f and int(r['requested_h'])==s))),
                ('total_anchors_vs_input_size.png','Total anchors',lambda f,s:float(next(r['total_anchors'] for r in shapes if r['model_family']==f and int(r['requested_h'])==s))),
                ('max_anchor_iou_vs_input_size.png','Mean best anchor IoU',lambda f,s:float(next(r['max_anchor_iou_mean'] for r in matching if r['model_family']==f and int(r['requested_h'])==s)))]
    for filename,ylabel,value in plot_specs:
        fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
        for family in ['ROI-SSD','ROI-SSD-MobileNet']:
            sizes=sorted(int(r['requested_h']) for r in metrics if r['model_family']==family and r['status']=='PASS')
            if sizes:ax.plot(sizes,[value(family,s) for s in sizes],marker='o',label=family)
        ax.axvline(128,color='gray',linestyle='--',linewidth=1);ax.set(xlabel='Requested square input side (px)',ylabel=ylabel)
        ax.legend();ax.grid(alpha=.2)
        fig.savefig(out/filename,dpi=160);plt.close(fig)
        text += [f'![{ylabel}]({filename})', '']
    if structurally_safe and mobile_quality_hazard:
        position=text.index('## Evidence and reproduction')
        text[position:position] = ['**Unrestricted MobileNet stage-3 runs remain on hold for size-policy validation:** '+
                                  'the dense sweep found severe neighboring-size accuracy sensitivity even though structural checks passed. '+
                                  'This is an accuracy limitation, not an identified structural/anchor implementation bug.', '']
    root=Path(__file__).resolve().parents[2]
    source_files=['model/roissd.py','model/roissd_mobilenet.py','tools/infer.py',
                  'dataset/transforms/fixed_size_test_transform.py',
                  'experiments/roissd_128_anomaly/run.py','experiments/roissd_128_anomaly/instrumentation.py',
                  'experiments/roissd_128_anomaly/report.py']
    metadata.update(stage3_allowed=structurally_safe and not mobile_quality_hazard,diagnosis=conclusions,
                    neighboring_low_accuracy_sizes=neighboring_hazards,
                    source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_files})
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'diagnostic_report.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Rebuild report/plots from measured diagnostic CSVs; no inference')
    parser.add_argument('--output',type=Path,default=Path('final_experiments/diagnostics_128'))
    args=parser.parse_args()
    lock=json.loads((Path(__file__).parent/'data/selection_lock.json').read_text())
    build_report(args.output,lock)
