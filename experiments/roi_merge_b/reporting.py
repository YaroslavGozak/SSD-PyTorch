"""Strict cohort aggregation and standalone publication-oriented figures."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import POLICIES,atomic,canonical,jsonl,read


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def aggregate_results(root,output,cfg):
    root,output=Path(root),Path(output)
    summaries=[(p,read(p)) for p in sorted(root.rglob('summary.json'))]
    summaries=[(p,s) for p,s in summaries if s.get('status')=='COMPLETE' and 'policies' in s]
    if not summaries:
        raise ValueError('No complete Experiment B evaluations found')
    identities={(s['identity']['manifest_sha256'],s['identity']['cohort_sha256'],s['fixture']) for _,s in summaries}
    if len(identities)!=1:
        raise ValueError('Different manifests/cohorts/fixture status cannot be pooled; aggregate common subsets separately')
    output.mkdir(parents=True,exist_ok=True)
    table=[];fits=[]
    for path,summary in summaries:
        identity=summary['identity'];runner=identity['runner']
        for policy in POLICIES:
            data=summary['policies'][policy];pair=data['pairwise'];frame=data['frame_all']
            runtime=runner['runtime']
            table.append({'hardware':runtime['host'],'detector':runner['detector'],'policy':policy,
                'fixture':summary['fixture'],'cohort_sha256':identity['cohort_sha256'],
                'calibration_id':identity['calibration_id'],'frames':frame['n'],'pairs':pair['n'],
                **{k:pair.get(k) for k in ('accuracy','false_merge_rate','false_separate_rate','mean_regret_ms','mean_cost_ms','speedup','realized_oracle_gain')},
                'frame_mean_serial_ms':frame.get('mean_serial_inference_ms'),
                'frame_p95_ms':frame.get('p95_ms'),'fps_equivalent':frame.get('fps_equivalent'),
                'frame_speedup':data['frame_paired_comparison']['speedup'],
                'frame_realized_oracle_gain':data['frame_paired_comparison']['realized_oracle_gain'],
                'planning_ms':frame.get('mean_planning_ms'),'pipeline_overhead_ms':frame.get('mean_pipeline_overhead_ms'),
                'pairwise_ci':canonical(data['pairwise_ci']),'frame_ci':canonical(data.get('frame_ci'))})
        for policy,data in summary['test_fit_diagnostics'].items():
            fits.append({'hardware':runner['runtime']['host'],'detector':runner['detector'],'policy':policy,**data})
    write_csv(output/'main_results.csv',table)
    write_csv(output/'heldout_fit.csv',fits)
    atomic(output/'aggregate.json',{'evaluations':[str(p.relative_to(root)) for p,_ in summaries],
        'manifest_sha256':summaries[0][1]['identity']['manifest_sha256'],'table':table,'fits':fits})
    fixture=any(s['fixture'] for _,s in summaries)
    lines=['# Experiment B results','',
        '**SYNTHETIC FIXTURE: software validation only; no scientific findings.**' if fixture else 'Measured fixed GT-derived ROI workload; separate from previously published video results.',
        '', '| Hardware | Detector | Policy | Pair accuracy | Mean regret ms | Serial replay ms | Oracle gain |',
        '|---|---|---|---:|---:|---:|---:|']
    def fmt(x):
        return 'N/A' if x is None else f'{x:.6g}'
    for row in table:
        lines.append('| '+' | '.join(str(row[k]) if k in ('hardware','detector','policy') else fmt(row[k]) for k in ('hardware','detector','policy','accuracy','mean_regret_ms','frame_mean_serial_ms','realized_oracle_gain'))+' |')
    lines += ['', '## Scope and interpretation','',
        'Pair decisions use independent synthetic-input shape timing: first-half reference trials select the retrospective oracle; disjoint second-half trials score every policy. Negative reference regret and oracle gains outside [0,1] can occur from noise and are retained. Scoring-minimum regret is reported separately.',
        '', 'Frame results execute actual fixed-content crops in randomized serial policy blocks. They do not measure online tracking, scheduling, frame drops, or detector accuracy. No mAP was computed. Empty-ROI frames use a full-frame fallback; all-frame and merge-eligible results remain separate.',
        '', 'The small-n frame oracle is the exact optimum only for additive held-out shape costs. Larger-n references are explicitly greedy. Neither is a global optimum of actual content-dependent wall time. YOLO raw forward and SSD forward including its internal postprocessing have detector-specific timing boundaries; compare policies within a detector.',
        '', 'Sequence bootstrap intervals, boundary/geometry/coverage strata, raw merge traces, and exclusions are in each evaluation summary and JSONL traces. Errors stop a run and are never silently dropped. No hypothesis of a preferred ranking is imposed.',
        '', 'Tables: `main_results.csv`, `heldout_fit.csv`. Figures: `figures/`. Each calibration directory retains table size, elapsed session time and independent validation fit diagnostics. Cross-platform claims require matching cohort hashes and verified real weights on all hosts.']
    (output/'paper_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    figures(summaries,output,root)
    print(f'Report: {output}/paper_results.md')


def figures(summaries,output,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=output/'figures';directory.mkdir(exist_ok=True)
    calibrations={read(p)['calibration_id']:read(p) for p in root.rglob('cost_model.json')}
    for index,(path,summary) in enumerate(summaries):
        prefix=f'{index:02d}_{summary["identity"]["runner"]["detector"]}'
        def save(fig,path):
            if summary['fixture']:
                fig.suptitle('SYNTHETIC FIXTURE: not scientific results',fontsize=10,color='darkred')
            fig.tight_layout(rect=(0,0,1,.94))
            fig.savefig(path,dpi=180,bbox_inches='tight')
            plt.close(fig)
        pairs=jsonl(path.parent/'pair_decisions.jsonl')
        lookup=[r for r in pairs if r['policy']=='lookup' and r.get('gain_ms') is not None]
        if lookup:
            fig,ax=plt.subplots(figsize=(5,4))
            ax.scatter([r['gain_ms'] for r in lookup],[r['separate_ms']-r['merged_ms'] for r in lookup],s=9,alpha=.4)
            ax.set(xlabel='Calibration-predicted merge gain (ms)',ylabel='Held-out scoring merge gain (ms)')
            save(fig,directory/f'{prefix}_gain.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        axes[0].bar(POLICIES,[summary['policies'][p]['frame_all']['mean_serial_inference_ms'] for p in POLICIES])
        axes[0].set_ylabel('Mean serial replay (ms)')
        axes[1].bar(POLICIES,[summary['policies'][p]['pairwise'].get('realized_oracle_gain') or 0 for p in POLICIES])
        axes[1].set_ylabel('Realized pairwise oracle gain\n(N/A shown as 0)')
        for ax in axes:
            ax.tick_params(axis='x',rotation=65)
        save(fig,directory/f'{prefix}_policies.png')
        bins=sorted({r['boundary'] for r in pairs})
        fig,ax=plt.subplots(figsize=(8,4))
        for policy in ('geometric','affine','piecewise','lookup','conservative_lookup'):
            ax.plot(bins,[np.mean([r['regret_ms'] for r in pairs if r['policy']==policy and r['boundary']==b]) for b in bins],marker='o',label=policy)
        ax.set(xlabel='Reference absolute merge gain bin',ylabel='Mean regret (ms)');ax.legend(fontsize=8)
        save(fig,directory/f'{prefix}_boundary.png')
        artifact=calibrations.get(summary['identity']['calibration_id'])
        if artifact:
            shapes=[tuple(map(int,k.split('x'))) for k in artifact['lookup']]
            means=[artifact['lookup'][f'{h}x{w}']['mean_ms'] for h,w in shapes]
            fig,ax=plt.subplots(figsize=(5,4));ax.scatter([h*w for h,w in shapes],means,s=12)
            ax.set(xlabel='Effective input pixels',ylabel='Calibration mean latency (ms)')
            save(fig,directory/f'{prefix}_area.png')
            hs,ws=sorted({s[0] for s in shapes}),sorted({s[1] for s in shapes})
            heat=np.full((len(hs),len(ws)),np.nan)
            for (h,w),value in zip(shapes,means):
                heat[hs.index(h),ws.index(w)]=value
            fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(heat,origin='lower',aspect='auto')
            ax.set_xticks(range(len(ws)),ws,rotation=90);ax.set_yticks(range(len(hs)),hs)
            ax.set(xlabel='Effective W',ylabel='Effective H');fig.colorbar(im,ax=ax,label='Calibration mean ms')
            save(fig,directory/f'{prefix}_heatmap.png')
