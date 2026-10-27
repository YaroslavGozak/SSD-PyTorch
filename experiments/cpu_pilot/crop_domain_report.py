"""Aggregate Stage 3b paired measurements and historical training proxy."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .crop_domain import OUT, PADDINGS, write_csv
from .run_pilot import ROOT, _sha256
from tools.benchmarks.video_artifacts import write_json


def read(name):
    with (OUT/name).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def yes(row,key='TP50_own_crop'):
    return row[key]=='True'


def avg(rows,key):return float(np.mean([float(r[key]) for r in rows])) if rows else None


def aggregate(rows):
    return {'N_GT':len(rows),'Recall50_own_crop_micro':np.mean([yes(r) for r in rows]) if rows else None,
            **{'mean_'+key:avg(rows,key) for key in ('best_prediction_iou','max_class_correct_confidence',
                                                   'best_anchor_iou','center_distance_ratio')},
            'depth_counts':dict(Counter(r['selected_L_R'] for r in rows))}


def mechanism(a,b):
    old_heads,new_heads=set(json.loads(a['active_heads'])),set(json.loads(b['active_heads']))
    new_matched=b['matched_prediction_head']
    new_matched=int(new_matched) if new_matched else None
    da=float(b['best_anchor_iou'])-float(a['best_anchor_iou'])
    dc=float(b['raw_max_class_correct_confidence'])-float(a['raw_max_class_correct_confidence'])
    dl=float(b['best_decoded_iou'])-float(a['best_decoded_iou'])
    added=bool(new_heads-old_heads)
    if added and new_matched in new_heads-old_heads:
        label='New active head appeared'
    elif old_heads==new_heads and da>=.05:
        label='Same heads, anchor IoU improved'
    elif old_heads==new_heads and abs(da)<.05 and dc>=.10 and abs(dl)<.10:
        label='Same anchor regime, confidence improved'
    elif old_heads==new_heads and abs(da)<.05 and abs(dc)<=.05 and dl>=.10:
        label='Same confidence, localization improved'
    else:label='Mixed/unclear'
    return {'mechanism':label,'new_head_present':added,'new_head_gave_match':added and new_matched in new_heads-old_heads,
            'delta_best_anchor_iou':da,'delta_raw_confidence':dc,'delta_best_decoded_iou':dl,
            'depth_changed':a['selected_L_R']!=b['selected_L_R'],
            'final_feature_shape_changed':a['final_feature_shape']!=b['final_feature_shape']}


def ks_distance(a,b):
    a,b=np.sort(a),np.sort(b)
    grid=np.unique(np.r_[a,b])
    return float(np.max(np.abs(np.searchsorted(a,grid,side='right')/len(a)-np.searchsorted(b,grid,side='right')/len(b))))


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT/name,dpi=170,bbox_inches='tight')
    plt.close()


def main():
    meta=json.loads((OUT/'metadata.json').read_text())
    assert meta['status'] in ('INFERENCE_COMPLETE','COMPLETE')
    assert all(_sha256(ROOT/p)==digest for p,digest in meta['source_sha256'].items())
    audit=json.loads((OUT/'training_preprocessing_audit.json').read_text())
    assert audit['exact_training_revision_known'] is False
    rows=read('per_object_context_trajectories.csv')
    contexts=defaultdict(dict)
    for r in rows:contexts[r['object_id']][int(r['padding'])]=r
    assert len(contexts)==560 and all(set(v)==set(PADDINGS) for v in contexts.values())
    wide,rescues,adjacent=[],[],[]
    for uid,trajectory in contexts.items():
        values=[trajectory[p] for p in PADDINGS]
        first=next((p for p,r in zip(PADDINGS,values) if yes(r)),None)
        first_global=next((p for p,r in zip(PADDINGS,values) if yes(r,'TP50_after_global_nms')),None)
        rescue=not yes(trajectory[96]) and (yes(trajectory[128]) or yes(trajectory[160]))
        rescue_global=not yes(trajectory[96],'TP50_after_global_nms') and (
            yes(trajectory[128],'TP50_after_global_nms') or yes(trajectory[160],'TP50_after_global_nms'))
        item={'object_id':uid,'class':values[0]['class'],'first_TP_padding_own_crop':first,
              'first_TP_padding_global':first_global,'MISS_all_paddings':first is None,
              'MISS_at_p160':not yes(trajectory[160]),'p96_miss_rescued_128_or_160':rescue,
              'p96_miss_rescued_128_or_160_global':rescue_global}
        for key in ('TP50_own_crop','TP50_after_global_nms','max_class_correct_confidence',
                    'raw_max_class_correct_confidence','best_prediction_iou','best_anchor_iou',
                    'selected_L_R','final_feature_shape','matched_prediction_head'):
            item[key+'_trajectory']=[r[key] for r in values]
        wide.append(item)
        if rescue:
            destination=128 if yes(trajectory[128]) else 160
            rescues.append({'object_id':uid,'from_padding':96,'to_padding':destination,
                            **mechanism(trajectory[96],trajectory[destination])})
        for a,b in zip(PADDINGS,PADDINGS[1:]):
            if not yes(trajectory[a]) and yes(trajectory[b]):
                adjacent.append({'object_id':uid,'from_padding':a,'to_padding':b,
                                 **mechanism(trajectory[a],trajectory[b])})
    write_csv('paired_object_trajectories.csv',wide)
    write_csv('p96_rescue_head_transitions.csv',rescues)
    write_csv('adjacent_head_transitions.csv',adjacent)
    transition_summary=[]
    for a,b in zip(PADDINGS,PADDINGS[1:]):
        rescued_ab=sum(not yes(v[a]) and yes(v[b]) for v in contexts.values())
        lost_ab=sum(yes(v[a]) and not yes(v[b]) for v in contexts.values())
        misses_a=sum(not yes(v[a]) for v in contexts.values())
        transition_summary.append({'from_padding':a,'to_padding':b,'N_GT':560,
                                   'MISS_to_TP':rescued_ab,'TP_to_MISS':lost_ab,
                                   'MISS_to_TP_fraction_all_GT':rescued_ab/560,
                                   'MISS_to_TP_fraction_prior_misses':rescued_ab/misses_a})
    write_csv('context_transition_summary.csv',transition_summary)
    counts=Counter(r['mechanism'] for r in rescues)
    mechanisms=[{'mechanism':label,'N_MISS_to_TP':counts[label],'fraction':counts[label]/len(rescues) if rescues else None}
                for label in ('New active head appeared','Same heads, anchor IoU improved',
                              'Same anchor regime, confidence improved','Same confidence, localization improved','Mixed/unclear')]
    write_csv('head_transition_summary.csv',mechanisms)
    border=read('clipped_vs_padded.csv')
    pairs=defaultdict(dict)
    for r in border:pairs[r['object_id']][r['variant']]=r
    assert len(pairs)==504 and all(len(v)==2 for v in pairs.values())
    b_summary=[]
    for lo,hi in [(0,float('inf')),(0,.1),(.1,.2),(.2,.3),(.3,.4),(.4,float('inf'))]:
        selected=[v for v in pairs.values() if lo<=float(v['CLIPPED']['center_distance_ratio'])<hi]
        for variant in ('CLIPPED','PADDED_OUTSIDE_FRAME'):
            b_summary.append({'clipped_center_bin':f'[{lo},{hi})','variant':variant,
                              **aggregate([v[variant] for v in selected])})
    write_csv('clipped_vs_padded_summary.csv',b_summary)
    pre=read('preprocessing_comparison.csv')
    pre_by=defaultdict(dict)
    for r in pre:pre_by[r['object_id']][r['variant']]=r
    assert len(pre_by)==560 and all(len(v)==4 for v in pre_by.values())
    pre_pair=[]
    metrics={r['variant']:r for r in read('context_sweep_summary.csv')}
    for variant in ('CROP_FIRST_MATCHED_HW','CROP_FIRST_224','CROP_FIRST_300'):
        for outcome in ('TP_both','TP_only_CURRENT','TP_only_CROP_FIRST','MISS_both'):
            for scope,key in [('own_crop','TP50_own_crop'),('after_global_nms','TP50_after_global_nms')]:
                count=0
                for v in pre_by.values():
                    a,b=yes(v['CURRENT_PIPELINE'],key),yes(v[variant],key)
                    name='TP_both' if a and b else 'TP_only_CURRENT' if a else 'TP_only_CROP_FIRST' if b else 'MISS_both'
                    count+=name==outcome
                pre_pair.append({'variant':variant,'scope':scope,'outcome':outcome,'N_GT':count,'fraction':count/560})
    write_csv('preprocessing_paired_outcomes.csv',pre_pair)
    pre_summary=[]
    for variant in ('CURRENT_PIPELINE','CROP_FIRST_MATCHED_HW','CROP_FIRST_224','CROP_FIRST_300'):
        selected=[r for r in pre if r['variant']==variant]
        metric=metrics['P96' if variant=='CURRENT_PIPELINE' else variant]
        pre_summary.append({'variant':variant,**{k:float(metric[k]) for k in ('mAP50','mAP50_95','recall50')},
                            **aggregate(selected),'tensor_hw_counts':dict(Counter(f'{r["network_h"]}x{r["network_w"]}' for r in selected))})
    write_csv('preprocessing_summary.csv',pre_summary)
    train=read('training_crop_sample.csv')
    pilot=[r for r in rows if int(r['padding'])==96]
    domains={'historical_training_proxy_all':train,
             'historical_training_proxy_ROI_only':[r for r in train if r['mode']!='full'],
             'pilot_oracle_p96':pilot}
    features={'gt_area_ratio':np.linspace(0,1,21),'center_distance_ratio':np.linspace(0,.75,16),
              'network_aspect_ratio':np.array([0,.25,.5,.75,.9,1.01,1.25,1.5,2,3,5,10]),
              'min_normalized_boundary_distance':np.linspace(-.01,.5,18)}
    for r in pilot:r['network_aspect_ratio']=str(float(r['network_w'])/float(r['network_h']))
    distributions,dist_stats=[],[]
    for feature,edges in features.items():
        for domain,values in domains.items():
            data=np.array([float(r[feature]) for r in values])
            hist,_=np.histogram(data,bins=edges)
            assert hist.sum()==len(data),(feature,domain,data.min(),data.max())
            for i,count in enumerate(hist):
                distributions.append({'feature':feature,'domain':domain,'provenance':'RECONSTRUCTED_PROXY' if 'proxy' in domain else 'MEASURED',
                                      'bin_left':edges[i],'bin_right':edges[i+1],
                                      'count':int(count),'fraction':count/len(data),'N':len(data)})
            reference=np.array([float(r[feature]) for r in pilot])
            dist_stats.append({'feature':feature,'domain':domain,'N':len(data),
                               'mean':float(data.mean()),'median':float(np.median(data)),
                               'p05':float(np.quantile(data,.05)),'p95':float(np.quantile(data,.95)),
                               'KS_distance_to_pilot_descriptive':ks_distance(data,reference)})
    write_csv('training_vs_pilot_distribution.csv',distributions)
    write_csv('training_vs_pilot_statistics.csv',dist_stats)
    x=list(PADDINGS)
    plt.rcParams.update({'figure.figsize':(7,4.5),'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    plt.figure()
    plt.plot(x,[float(metrics[f'P{p}']['recall50']) for p in x],'o-',label='Frame evaluator macro Recall50')
    plt.plot(x,[float(metrics[f'P{p}']['Recall50_own_crop_micro']) for p in x],'s--',label='Own-crop paired GT recall')
    plt.axhline(.7549146931499874,color='gray',linestyle=':',label='FF macro Recall50')
    plt.ylim(0,1);plt.xticks(x);plt.xlabel('Padding in base 320×320 tensor pixels');plt.ylabel('Recall@0.5')
    plt.legend();savefig('recall_vs_padding.png')
    for filename,key,ylabel in [('confidence_vs_padding.png','max_class_correct_confidence','Mean returned class-correct confidence'),
                                ('pred_iou_vs_padding.png','best_prediction_iou','Mean best class-correct prediction IoU')]:
        plt.figure()
        plt.plot(x,[avg([r for r in rows if int(r['padding'])==p],key) for p in x],'o-')
        plt.ylim(0,1);plt.xticks(x);plt.xlabel('Padding in base 320×320 tensor pixels');plt.ylabel(ylabel)
        savefig(filename)
    plt.figure()
    labels=['<0.1','0.1–0.2','0.2–0.3','0.3–0.4','≥0.4']
    for variant in ('CLIPPED','PADDED_OUTSIDE_FRAME'):
        points=[r for r in b_summary if r['variant']==variant][1:]
        plt.plot(range(5),[r['Recall50_own_crop_micro'] for r in points],'o-',label=variant)
    plt.xticks(range(5),labels);plt.ylim(0,1);plt.xlabel('Normalized center distance in clipped crop')
    plt.ylabel('Paired own-crop GT Recall50');plt.legend();savefig('clipped_vs_padded_by_center_offset.png')
    names={'gt_area_ratio':'train_vs_pilot_area_ratio.png','center_distance_ratio':'train_vs_pilot_center_offset.png',
           'network_aspect_ratio':'train_vs_pilot_aspect_ratio.png',
           'min_normalized_boundary_distance':'train_vs_pilot_context_to_edge.png'}
    for feature,filename in names.items():
        plt.figure()
        for domain,values in domains.items():
            plt.hist([float(r[feature]) for r in values],bins=features[feature],weights=np.ones(len(values))/len(values),
                     histtype='step',linewidth=1.6,label=domain.replace('historical_training_proxy','Training proxy'))
        plt.xlabel(feature);plt.ylabel('Fraction of reference objects');plt.legend(fontsize=8)
        plt.title('Training is a historical reconstruction, not observed crops',fontsize=10)
        savefig(filename)
    rescued=sum(r['p96_miss_rescued_128_or_160'] for r in wide)
    missed160=sum(r['MISS_at_p160'] for r in wide)
    never=sum(r['MISS_all_paddings'] for r in wide)
    old96miss=sum(not yes(v[96]) for v in contexts.values())
    bs=[r for r in b_summary if r['clipped_center_bin']=='[0,inf)']
    ff=.7391474798825811
    report=['# Stage 3b: controlled crop-domain diagnosis', '',
            '**Conclusion: context, network input scale/head regime and historical preprocessing distribution '
            'interact. Crop-first alone is not the explanation. No production change or retraining was performed.**', '',
            '## Protocol and evidence', '',
            'Locked 32 videos / 320 frames / 560 GT, unchanged MobileNet checkpoint, CPU 8 threads, '
            'until_1x1, confidence 0.5, internal/global class-wise NMS 0.45 and the original evaluator. '
            'No tracker, merge or refresh. p=32/64/96/128 exactly replay Stage 3a predictions. '
            'Padding remains in base 320×320 tensor units. Each GT has five paired context observations.', '',
            'Frame metrics combine all current-frame GT crops with global NMS. Object trajectories, '
            'border comparisons and head attribution primarily use detection in that GT’s own crop: '
            'this isolates the controlled object from help/suppression by another object’s crop. '
            'Both own-crop and after-global-NMS outcomes are saved. These recalls differ from the '
            'class-macro evaluator recall and must not be equated. Target-only FP counts in long object '
            'rows are not frame FP counts; valid neighboring GT may be present in a crop.', '',
            '## A. Paired context trajectories', '',
            '| p | Frame mAP50 | Macro Recall50 | Own-crop Recall50 | Mean GT area ratio | Best anchor IoU | Confidence | Prediction IoU |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in x:
        r=metrics[f'P{p}']
        report.append(f'| {p} | '+ ' | '.join(f'{float(r[k]):.6f}' for k in ('mAP50','recall50','Recall50_own_crop_micro',
            'mean_gt_area_ratio','mean_best_anchor_iou','mean_confidence','mean_pred_iou'))+' |')
    report += ['',f'At p=160, mAP50 retention versus FF is {float(metrics["P160"]["mAP50"])/ff:.2%}; '
               f'the remaining mAP50 gap is {ff-float(metrics["P160"]["mAP50"]):.6f}. '
               f'{rescued}/560 GT ({rescued/560:.1%}) are MISS at p96 and TP at p128/p160 '
               f'({rescued/old96miss:.1%} of p96 own-crop misses). '
               f'{missed160}/560 ({missed160/560:.1%}) are still MISS at p160; '
               f'{never}/560 ({never/560:.1%}) are MISS at every padding. '
               'The wide trajectories include first TP, confidence, prediction/anchor IoU, depth, final shape, '
               'and nonmonotonic outcomes; larger context is not assumed to improve every object.', '',
               '| Padding transition | MISS→TP | Fraction of all GT | Fraction of prior misses | TP→MISS |',
               '|---|---:|---:|---:|---:|']
    for r in transition_summary:
        report.append(f'| {r["from_padding"]}→{r["to_padding"]} | {r["MISS_to_TP"]} | '
                      f'{r["MISS_to_TP_fraction_all_GT"]:.1%} | {r["MISS_to_TP_fraction_prior_misses"]:.1%} | {r["TP_to_MISS"]} |')
    report += ['', '## B. Clipped versus outside-frame padding at p96', '',
               '504/560 nominal symmetric ROIs cross a source boundary. The padded arm retains the nominal '
               'tensor-grid ROI and fills outside with exactly zero after ImageNet normalization '
               '(equivalent to ImageNet mean RGB before normalization). No reflection/replication is used. '
               'Remapped outputs are clipped to the real source frame before scoring.', '',
               '| Variant | N GT | Own-crop Recall50 | Mean prediction IoU | Mean confidence | Mean anchor IoU |',
               '|---|---:|---:|---:|---:|---:|']
    for r in bs:
        report.append(f'| {r["variant"]} | {r["N_GT"]} | {r["Recall50_own_crop_micro"]:.6f} | '
                      f'{r["mean_best_prediction_iou"]:.6f} | {r["mean_max_class_correct_confidence"]:.6f} | {r["mean_best_anchor_iou"]:.6f} |')
    report += ['', '| Clipped center distance | N GT | Clipped recall | Padded recall |', '|---|---:|---:|---:|']
    for a,b in zip(b_summary[2::2],b_summary[3::2]):
        report.append(f'| {a["clipped_center_bin"]} | {a["N_GT"]} | {a["Recall50_own_crop_micro"]:.4f} | {b["Recall50_own_crop_micro"]:.4f} |')
    report += ['',f'For the full set (unchanged crops for non-border GT), PADDED mAP50 is '
               f'{float(metrics["PADDED_P96"]["mAP50"]):.6f}, macro Recall50 '
               f'{float(metrics["PADDED_P96"]["recall50"]):.6f}; clipped p96 is '
               f'{float(metrics["P96"]["mAP50"]):.6f}/{float(metrics["P96"]["recall50"]):.6f}. '
               'Thus own-crop recovery must not be mistaken for an overall AP/recall improvement. '
               'The border intervention changes position, valid context fraction, H/W and head/anchor geometry '
               'together. Edge position is associated with failure but is not established as an independent cause.', '',
               '## C. Training evidence and preprocessing comparison', '',
               '**Exact training revision is unavailable.** The checkpoint contains stage=2, epoch=18 '
               '(zero based), best_map=0.693501514842189, matching training_metrics.csv epoch 19. '
               'It has no source hash, launch command or crop manifest. The saved YAML is an evaluation '
               'snapshot. September Git source is used only as a contemporaneous reconstruction; an '
               'uncommitted training working tree cannot be ruled out.', '',
               f'Historical source `{audit["historical_revision"]}` uses native RGB CHW float input, GT-reference '
               'center/scale jitter and random source-pixel padding, clips a source ROI, resizes each crop '
               'independently with bilinear interpolation (align_corners=False), then applies ImageNet '
               'normalization. No photometric transform or RandomIoUCrop is passed by that mixed-training '
               'entry point. Stage-2 output sides are 300/224/160/96/64, with probabilities 0.5/0.2/0.1/0.1/0.1; '
               'they are not automatically scaled to YAML im_size=320 in that revision. The historical '
               'processor builds a map from stages 2 then 3: repeated mode names are overwritten by stage-3 '
               'jitter/context settings. The reconstruction executes that behavior and records the effective '
               'mode map. October’s rewritten sampler is not substituted as September-training evidence.', '',
               'CURRENT uses the exact Stage 3a tensor slice. CROP_FIRST_MATCHED_HW crops the same snapped '
               'source ROI and resizes it to exactly the current tensor H/W, isolating order/interpolation '
               'at fixed geometry. CROP_FIRST_224 was declared as the primary historical large-ROI mode; '
               'CROP_FIRST_300 is a declared historical full-output-size control, not a post-hoc production '
               'choice. Both resize to a square and therefore additionally change sampling density, aspect '
               'distortion, active depth and head geometry. Every arm uses the exact same source ROI/GT.', '',
               '| Variant | Frame mAP50 | Macro Recall50 | Own-crop Recall50 | Mean confidence | Pred IoU | Anchor IoU |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for r in pre_summary:
        report.append(f'| {r["variant"]} | {r["mAP50"]:.6f} | {r["recall50"]:.6f} | '
                      f'{r["Recall50_own_crop_micro"]:.6f} | {r["mean_max_class_correct_confidence"]:.6f} | '
                      f'{r["mean_best_prediction_iou"]:.6f} | {r["mean_best_anchor_iou"]:.6f} |')
    report += ['', '| CROP_FIRST variant (own crop) | TP both | TP only CURRENT | TP only CROP_FIRST | MISS both |',
               '|---|---:|---:|---:|---:|']
    for v in ('CROP_FIRST_MATCHED_HW','CROP_FIRST_224','CROP_FIRST_300'):
        vals=[next(r['N_GT'] for r in pre_pair if r['variant']==v and r['scope']=='own_crop' and r['outcome']==o)
              for o in ('TP_both','TP_only_CURRENT','TP_only_CROP_FIRST','MISS_both')]
        report.append(f'| {v} | '+' | '.join(map(str,vals))+' |')
    report += ['', 'Matched-H/W crop-first preprocessing does not recover accuracy. This control also '
               'changes interpolation/pixel alignment (historical bilinear without antialiasing versus '
               'the pilot full-frame torchvision resize); it is not a normalization-only intervention. Resizing to 224 also '
               'underperforms CURRENT; resizing to 300 improves mAP50/Recall50. Therefore these measurements '
               'implicate input resolution/head regime and preprocessing distribution together, not a simple '
               'normalization-order bug. The best 300 result still has much lower mAP50:95 than FF, so '
               'its mAP50 advantage is not equivalent to restoring full localization quality.', '',
               '## D. Head/anchor transitions', '',
               'Per-head rows include spatial shape, anchor count, target anchor IoU, raw class-correct '
               'foreground confidence, best decoded IoU and own-crop matched head. Scores are softmax '
               'probabilities; decoded IoU scans all anchors before score filtering. Matched-head provenance '
               'is recovered from returned box+class+score against decoded anchors at tolerance 1e-6 '
               '(first matching source if a tie). This is descriptive attribution, not a head ablation.', '',
               'The table uses unique GT rescued from p96 to their first successful p128/p160 own crop. '
               'Classification is exclusive in table order: a newly active head must supply the match; '
               'otherwise unchanged head set and anchor IoU gain ≥0.05; otherwise |anchor delta|<0.05 '
               'and raw confidence gain ≥0.10 with |decoded-IoU delta|<0.10; otherwise unchanged head set, '
               '|anchor delta|<0.05, |raw confidence delta|≤0.05 and decoded-IoU gain ≥0.10; remaining '
               'cases are mixed/unclear. “Same anchor regime” here is a coarse diagnostic label, not '
               'identical anchor coordinates. Thresholds are diagnostic bins, not inference thresholds.', '',
               '| Mechanism | N MISS→TP | Fraction |', '|---|---:|---:|']
    for r in mechanisms:report.append(f'| {r["mechanism"]} | {r["N_MISS_to_TP"]} | {r["fraction"]:.1%} |')
    report += ['',f'{sum(r["depth_changed"] for r in rescues)}/{len(rescues)} rescued GT change selected depth; '
               f'{sum(r["final_feature_shape_changed"] for r in rescues)}/{len(rescues)} change final feature shape. '
               'Although 57 rescues change depth, none is directly supplied by a newly activated head under '
               'the recovered matched-head attribution. Existing-head feature/score/regression changes '
               'therefore deserve attention alongside the geometry changes. '
               'Mean best-anchor IoU does not rise monotonically with context while confidence/prediction '
               'quality generally recover. Anchor overlap alone is not a sufficient explanation.', '',
               '## E. Reconstructed training distribution versus pilot', '',
               'A fixed-seed sample of 1,000 distinct training-manifest frames was processed by the historical '
               'mixed sampler, without training the model. One reference GT is measured per sampled crop; '
               'full-mode crops use one random GT. This is a representative reconstruction under available '
               'code/config, not the actual crops seen by the checkpoint. Both all-mode and ROI-only '
               'distributions are shown to avoid hiding the 50% full-frame mixture. Classes/reference-object '
               'sampling and source-frame distributions differ, so this is descriptive domain-shift evidence.', '',
               '| Feature | Domain | N | Median | p05 | p95 | Descriptive KS vs pilot |',
               '|---|---|---:|---:|---:|---:|---:|']
    for r in dist_stats:
        report.append(f'| {r["feature"]} | {r["domain"]} | {r["N"]} | {r["median"]:.4f} | '
                      f'{r["p05"]:.4f} | {r["p95"]:.4f} | {r["KS_distance_to_pilot_descriptive"]:.4f} |')
    report += ['', 'Historical network crops are square at fixed scheduled sizes, whereas pilot inputs are '
               'variable rectangles cut from a full-frame 320×320 resize. Source-pixel training padding '
               'also differs from pilot padding scaled from tensor units. Area/center/context histograms '
               'quantify the additional shifts; they do not prove the checkpoint saw this exact distribution. '
               'In this reconstruction, ROI-only training has median GT area ratio 0.7551 versus pilot 0.0724, '
               'and median center distance 0.0379 versus pilot 0.1093: pilot objects are typically much '
               'smaller in the crop and farther from its center.', '',
               '## Answers and next decision', '',
               '1. More context substantially restores accuracy for the same GT, but changes input H/W and '
               'heads at the same time. p160 reaches 94.86% of FF mAP50 on this oracle study; it is not '
               'a new production padding selection.',
               '2. Edge position is not isolated as an independent factor: clipping, centering, context and '
               'network geometry change jointly. Paired border bins provide the measured effect.',
               '3. Outside-frame padding improves own-crop recall in aggregate but barely changes full-set '
               'mAP50 and lowers macro recall. It is not a demonstrated standalone fix.',
               '4. Available historical training code differs from CURRENT; exact run provenance remains '
               'unverified. No assumption is made that current training source describes the old checkpoint.',
               '5. CROP_FIRST is not uniformly better. Fixed-H/W and 224 comparisons are worse, while 300 '
               'is better at mAP50. This points to a resolution/head/preprocessing interaction.',
               '6. The rescue table measures new-head attribution and mixed changes; it does not establish '
               'causality without head/geometry-controlled ablations.',
               '7. Reconstructed training and pilot geometry distributions differ, especially fixed square '
               'versus variable rectangular network inputs. Historical identity cannot be established.',
               '8. Most likely: a combination of context loss and input-scale/head specialization under '
               'a shifted preprocessing/crop distribution; edge sensitivity contributes but zero padding '
               'alone does not remove the frame-level loss. Anchor IoU alone is insufficient.',
               '9. A no-retraining inference candidate exists (crop-first 300), but this oracle result '
               'does not demonstrate a production fix: normal PRED_SEG, localization quality and CPU '
               'cost have not been validated for that candidate. No automatic rollout is justified.',
               '10. If a separate fixed-protocol validation cannot meet the original acceptance/cost criteria, '
               'test training on the deployment path: full-frame resize then variable rectangular slices, '
               'matched context/offset/border distributions, coverage across head transition sizes, and '
               'a controlled mixture retaining full-frame behavior. Compare this against crop-first '
               'training with a deliberately matched inference path. Recover exact source/config/seed '
               'and save crop manifests in that future run; no retraining is performed here.', '',
               'Production weights, anchors, loss, thresholds, NMS, tracker, merge, padding and acceptance '
               'criteria are unchanged. CA/FULL-TECH/Pi remain on hold. All claims are limited to this '
               'validation pilot; multiple frames per video are correlated. No statistical independence '
               'or untouched-test generalization is claimed.', '',
               '## Figures', '',
               '![Recall versus padding](recall_vs_padding.png)', '',
               '![Confidence versus padding](confidence_vs_padding.png)', '',
               '![Prediction IoU versus padding](pred_iou_vs_padding.png)', '',
               '![Border intervention by center offset](clipped_vs_padded_by_center_offset.png)', '',
               '![Reconstructed training versus pilot area ratio](train_vs_pilot_area_ratio.png)', '',
               '![Reconstructed training versus pilot center offset](train_vs_pilot_center_offset.png)', '',
               '![Network aspect ratio](train_vs_pilot_aspect_ratio.png)', '',
               '![Minimum normalized context](train_vs_pilot_context_to_edge.png)', '',
               'Reproduce:', '', '```powershell',
               '.venv/Scripts/python.exe -m experiments.cpu_pilot.crop_domain',
               '.venv/Scripts/python.exe -m experiments.cpu_pilot.crop_training_proxy',
               '.venv/Scripts/python.exe -m experiments.cpu_pilot.crop_domain_validate',
               '.venv/Scripts/python.exe -m experiments.cpu_pilot.crop_domain_report', '```']
    (OUT/'crop_domain_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    meta.update(status='COMPLETE',completed_at=datetime.now(timezone.utc).isoformat(),
                training_provenance='RECONSTRUCTED_HISTORICAL_PROXY_NOT_EXACT_RUN',training_sample_crops=1000,
                rescued_p96_own_crop=rescued,miss_at_p160=missed160,miss_all_paddings=never,
                training_preprocessing_audit=audit,head_transition_mechanisms=mechanisms,
                inference_candidates_not_production_validated=['CROP_FIRST_300'],
                production_change_authorized=False,ca_fulltech_pi_allowed=False)
    for path in ('experiments/cpu_pilot/crop_training_proxy.py','experiments/cpu_pilot/crop_domain_report.py',
                 'experiments/cpu_pilot/tests/test_crop_domain.py'):
        meta['source_sha256'][path]=_sha256(ROOT/path)
    meta['output_sha256']={str(p.relative_to(OUT)):_sha256(p) for p in OUT.rglob('*') if p.is_file() and p.name!='metadata.json'}
    write_json(OUT/'metadata.json',meta)
    print('COMPLETE:',rescued,'p96 rescues;',missed160,'miss at p160;',never,'miss all paddings',flush=True)


if __name__=='__main__':main()
