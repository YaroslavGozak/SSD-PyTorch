"""Paired metrics, sequence bootstrap and fit diagnostics, in milliseconds."""
import math
from collections import defaultdict

import numpy as np


def safe_div(a,b):
    return float(a/b) if b else None


def pair_summary(rows):
    if not rows:
        return {'n':0}
    counts={name:0 for name in ('TP','TN','FP','FN')}
    for row in rows:
        counts[('TP' if row['oracle_merge'] else 'FP') if row['merge'] else ('FN' if row['oracle_merge'] else 'TN')]+=1
    tp,tn,fp,fn=(counts[k] for k in ('TP','TN','FP','FN'))
    chosen=np.asarray([r['chosen_ms'] for r in rows])
    baseline=sum(r['separate_ms'] for r in rows)
    oracle=sum(r['oracle_ms'] for r in rows)
    regrets=[r['regret_ms'] for r in rows]
    return {'n':len(rows),**counts,'accuracy':(tp+tn)/len(rows),'precision':safe_div(tp,tp+fp),
            'recall':safe_div(tp,tp+fn),'f1':safe_div(2*tp,2*tp+fp+fn),
            'false_merge_rate':safe_div(fp,fp+tn),'false_separate_rate':safe_div(fn,fn+tp),
            'mean_cost_ms':float(chosen.mean()),'median_cost_ms':float(np.median(chosen)),
            'baseline_total_ms':baseline,'oracle_total_ms':oracle,'chosen_total_ms':float(chosen.sum()),
            'saved_ms':baseline-float(chosen.sum()),'speedup':safe_div(baseline,chosen.sum()),
            'realized_oracle_gain':safe_div(baseline-chosen.sum(),baseline-oracle) if baseline-oracle>0 else None,
            'mean_regret_ms':float(np.mean(regrets)),'sum_regret_ms':sum(regrets),
            'normalized_regret':safe_div(sum(regrets),oracle),
            'noise_disagreement_count':sum(r['reference_scoring_disagreement'] for r in rows),
            'negative_reference_regret_count':sum(r['regret_ms']<0 for r in rows),
            'mean_scoring_min_regret_ms':float(np.mean([r['scoring_min_regret_ms'] for r in rows]))}


def bootstrap_pairs(rows,repetitions,seed):
    if not rows:
        return None
    videos=defaultdict(list)
    for row in rows:
        videos[row['video_id']].append(row)
    keys=sorted(videos)
    rng=np.random.default_rng(seed)
    fields=('accuracy','false_merge_rate','mean_regret_ms','realized_oracle_gain','speedup')
    samples={k:[] for k in fields}
    for _ in range(repetitions):
        selected=[row for index in rng.integers(0,len(keys),len(keys)) for row in videos[keys[index]]]
        summary=pair_summary(selected)
        for field in fields:
            if summary[field] is not None:
                samples[field].append(summary[field])
    return {'method':'paired sequence-cluster percentile bootstrap','sequences':len(keys),
            'intervals_95':{k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in samples.items()}}


def frame_summary(rows):
    if not rows:
        return {'n':0}
    # Average repeats per frame before reporting across-frame percentiles.
    groups=defaultdict(list)
    for row in rows:
        groups[(row['video_id'],row['frame_index'])].append(row)
    latencies=[np.mean([r['serial_inference_ms'] for r in group]) for group in groups.values()]
    return {'n':len(groups),'mean_serial_inference_ms':float(np.mean(latencies)),
            'median_ms':float(np.median(latencies)),'p90_ms':float(np.quantile(latencies,.90)),
            'p95_ms':float(np.quantile(latencies,.95)), 'fps_equivalent':1000/float(np.mean(latencies)),
            'mean_calls':float(np.mean([r['calls'] for r in rows])),
            'mean_processed_pixels':float(np.mean([r['effective_pixels'] for r in rows])),
            'mean_planning_ms':float(np.mean([r['planning_ms'] for r in rows])),
            'mean_pipeline_overhead_ms':float(np.mean([r['pipeline_overhead_ms'] for r in rows]))}


def fit_diagnostics(estimator,table):
    actual=[];predicted=[];exact=0
    for key,values in table.items():
        shape=tuple(map(int,key.split('x')))
        value,_,covered=estimator.estimate(shape)
        if value is None:
            continue
        actual.append(np.mean(values));predicted.append(value);exact+=covered
    if not actual:
        return {'n':0}
    y,p=np.asarray(actual),np.asarray(predicted)
    return {'n':len(y),'exact_shape_count':exact,'mae_ms':float(np.abs(y-p).mean()),
            'rmse_ms':float(np.sqrt(np.mean((y-p)**2))),
            'r2':float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)) if np.var(y)>0 else None}


def bootstrap_frames(rows,repetitions,seed):
    """Cluster paired frame rows by sequence; report serial replay speedup CI."""
    videos=defaultdict(list)
    for row in rows:
        videos[row['video_id']].append(row)
    if not videos:
        return None
    keys=sorted(videos);rng=np.random.default_rng(seed)
    values=[]
    for _ in range(repetitions):
        picked=[r for i in rng.integers(0,len(keys),len(keys)) for r in videos[keys[i]]]
        values.append(float(np.mean([r['serial_inference_ms'] for r in picked])))
    return {'method':'sequence-cluster bootstrap of frame/repetition blocks','sequences':len(keys),
            'mean_serial_ms_ci95':np.quantile(values,[.025,.975]).tolist()}


def paired_frame_comparison(all_rows,policy,repetitions,seed):
    by_key={(r['video_id'],r['frame_index'],r['repetition'],r['policy']):r['serial_inference_ms'] for r in all_rows}
    videos=defaultdict(list)
    for row in all_rows:
        if row['policy']!=policy:
            continue
        prefix=(row['video_id'],row['frame_index'],row['repetition'])
        videos[row['video_id']].append((by_key[(*prefix,'no_merge')],row['serial_inference_ms'],by_key[(*prefix,'oracle')]))
    def stats(values):
        baseline,chosen,oracle=map(sum,zip(*values))
        denominator=baseline-oracle
        return {'speedup':baseline/chosen,'saved_ms':baseline-chosen,
                'realized_oracle_gain':(baseline-chosen)/denominator if denominator>0 else None,
                'oracle_saving_positive':denominator>0}
    if not videos:
        return None
    result=stats([v for group in videos.values() for v in group])
    keys=sorted(videos);rng=np.random.default_rng(seed);samples=[]
    for _ in range(repetitions):
        draw=[v for index in rng.integers(0,len(keys),len(keys)) for v in videos[keys[index]]]
        samples.append(stats(draw)['speedup'])
    result['speedup_sequence_bootstrap_ci95']=np.quantile(samples,[.025,.975]).tolist()
    return result
