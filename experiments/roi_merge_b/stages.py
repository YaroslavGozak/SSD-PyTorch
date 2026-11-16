"""Phase orchestration for the fixed ROI merging experiment."""
import argparse
import csv
import itertools
import json
import os
import random
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from .common import (MODELS,POLICIES,ROOT,asset,atomic,canonical,checkpoint,cohort,config,dataset_root,
                     digest,git_identity,jsonl,load_manifest,read,sha)
from .workload import (all_rectangles,area,build_rows,frame_shapes,geometry_strata,shape_for,
                       source_rows,union,write_workload)
from .policies import Estimator,OracleCosts,fit_models,key,plan,summarize_trials
from .metrics import bootstrap_pairs,bootstrap_frames,fit_diagnostics,frame_summary,pair_summary,paired_frame_comparison


def names_from_yaml(cfg,root):
    path=Path(os.environ.get(cfg['dataset']['yaml_env'],str(root/'imagenet_vid.yaml')))
    source=yaml.safe_load(path.read_text(encoding='utf-8'))
    names=source['names']
    if isinstance(names,dict):
        if set(map(int,names))!=set(range(len(names))):
            raise ValueError('Dataset class indices must be contiguous zero-based')
        names=[names[i] if i in names else names[str(i)] for i in range(len(names))]
    if len(names)!=cfg['dataset']['expected_classes'] or len(set(names))!=len(names):
        raise ValueError('Unexpected dataset class mapping')
    return names


def verify_assets(rows,root,classes,cfg):
    from PIL import Image
    from .workload import parse_labels
    for row in rows:
        image,label=asset(root,row['image']),asset(root,row['label'])
        if sha(image)!=row['image_sha256'] or sha(label)!=row['label_sha256']:
            raise ValueError(f'Frozen frame/annotation content changed: {row["image"]}')
        parse_labels(label,classes)
        for roi in row['rois']:
            for model in MODELS:
                if roi['effective_hw'][model] != list(shape_for(roi['box'],row,cfg,model)):
                    raise ValueError('Frozen effective shape mismatch')
        with Image.open(image) as decoded:
            decoded.load()
            if decoded.size!=(row['width'],row['height']):
                raise ValueError('Frozen frame dimension mismatch')


def runner_for(args,cfg):
    from .execution import Runner
    if args.synthetic_smoke:
        _,meta=load_manifest(args.manifest,cfg)
        if not meta['fixture']:
            raise ValueError('Synthetic runner is prohibited for real manifests')
        from .fixture import SyntheticRunner
        return SyntheticRunner(cfg,args.model)
    if args.manifest:
        _,meta=load_manifest(args.manifest,cfg)
        if meta['fixture']:
            raise ValueError('Synthetic fixture requires --synthetic-smoke')
    return Runner(cfg,args.model,args.checkpoint_root,
                  {k:getattr(args,k) for k in ('cpu_threads','interop_threads','opencv_threads')},args.affinity)


def estimate_work(rows,cfg,model):
    shapes=set()
    for row in rows:
        shapes.update(frame_shapes(row,cfg,model))
    repeats=cfg['profiling']['evaluation_repetitions']
    warm=cfg['profiling']['warmup']
    maximum_calls=sum(max(1,len(r['rois'])) for r in rows)*len(POLICIES)*repeats*(warm+1)
    return {'frames':len(rows),'roi_distribution':dict(Counter(len(r['rois']) for r in rows)),
            'merge_eligible_frames':sum(len(r['rois'])>=2 for r in rows),'unique_heldout_shapes':len(shapes),
            'shape_network_calls_including_warmup':len(shapes)*repeats*(warm+1),
            'frame_policy_runs':len(rows)*len(POLICIES)*repeats,
            'frame_network_calls_upper_bound_including_warmup':maximum_calls,
            'runtime_estimate_note':'Multiply shape calls by locally measured pilot milliseconds; no hardware timings assumed',
            'all_union_shapes':sorted(shapes)}


def prepare(args,cfg):
    root=dataset_root(cfg,args.dataset_root)
    source=source_rows(cfg,args.cohort_manifest)
    records=build_rows(source,root,cfg)
    meta=write_workload(records,cfg,args.output,names_from_yaml(cfg,root),sha(ROOT/cfg['dataset']['index']))
    atomic(Path(args.output)/'config.snapshot.json',cfg)
    print(json.dumps(meta,indent=2))


def inspect_run(args,cfg):
    from .execution import versions
    report={'git':git_identity(),'config_sha256':digest(cfg),'software':versions(),'blockers':[],'models':{}}
    rows=[]
    try:
        root=dataset_root(cfg,args.dataset_root)
        names=names_from_yaml(cfg,root)
        if args.manifest:
            rows,meta=load_manifest(args.manifest,cfg)
            verify_assets(rows,root,len(names),cfg)
            if meta['class_names']!=names:
                raise ValueError('Manifest/dataset class mapping mismatch')
            report['manifest']=meta
        else:
            rows=build_rows(source_rows(cfg,args.cohort_manifest),root,cfg)
        report['actual_frames']=len(rows)
        report['actual_sequences']=len({r['video_id'] for r in rows})
    except (OSError,ValueError,KeyError,ImportError) as error:
        report['blockers'].append(str(error))
    for model in MODELS:
        try:
            path=checkpoint(cfg,model,args.checkpoint_root)
            report['models'][model]={'checkpoint_sha256':sha(path),'checkpoint_relative':cfg['models'][model]['checkpoint']}
            if rows:
                report['models'][model]['work']=estimate_work(cohort(rows,cfg,args.subset,args.limit_frames),cfg,model)
            if args.skip_inference:
                report['models'][model]['smoke']='NOT_EXECUTED'
            else:
                if not rows:
                    raise ValueError('Dataset preflight failed; model inference smoke not executed')
                # Each detector smoke in a separate process avoids inter-op/library state reuse.
                import subprocess,sys
                command=[sys.executable,'-m','experiments.roi_merge_b','model-smoke','--config',str(args.config),'--model',model,'--dataset-root',str(root)]
                for flag in ('checkpoint_root','cpu_threads','interop_threads','opencv_threads'):
                    value=getattr(args,flag)
                    if value is not None:
                        command += ['--'+flag.replace('_','-'),str(value)]
                if args.affinity is not None:
                    command+=['--affinity',*map(str,args.affinity)]
                completed=subprocess.run(command,capture_output=True,text=True)
                report['models'][model]['smoke_output']=completed.stdout+completed.stderr
                if completed.returncode:
                    raise RuntimeError(f'{model} smoke failed: {completed.stderr}')
        except (OSError,ValueError,KeyError,ImportError,RuntimeError) as error:
            report['blockers'].append(str(error))
    report['status']='BLOCKED' if report['blockers'] else 'ASSETS_ONLY_PASS' if args.skip_inference else 'PREFLIGHT_PASS'
    atomic(Path(args.output)/'preflight.json',report)
    print(json.dumps(report,indent=2))
    return 1 if report['blockers'] else 0


def model_smoke(args,cfg):
    root=dataset_root(cfg,args.dataset_root)
    runner=runner_for(args,cfg)
    print(json.dumps({'identity':runner.identity,'smoke':runner.smoke(names_from_yaml(cfg,root))},indent=2))


def calibration_shapes(rows,cfg,model):
    grid=cfg['profiling']['grid_sides']
    maximum=cfg['profiling']['calibration_max_shapes']
    grid_shapes={(h,w) for h in grid for w in grid}
    # Grid is specified as executed dimensions; enforce model-specific admissibility.
    stride=cfg['models'][model]['shape_grid']
    if any(h%stride or w%stride for h,w in grid_shapes):
        raise ValueError('Calibration grid includes nonexecuted shape dimensions')
    observed=set()
    validation=set()
    for row in rows:
        if row['split']=='calibration':
            observed.update(frame_shapes(row,cfg,model))
        elif row['split']=='validation':
            validation.update(frame_shapes(row,cfg,model))
    # Test geometries are never used to choose calibration entries.
    selected=sorted(grid_shapes)
    random.Random(cfg['seed']).shuffle(selected)
    selected=selected[:maximum]
    extra=sorted(observed-set(selected));random.Random(cfg['seed']+1).shuffle(extra)
    selected+=extra[:max(0,maximum-len(selected))]
    validation=sorted(validation)
    random.Random(cfg['seed']+2).shuffle(validation)
    validation=validation[:maximum]
    if not validation:
        raise ValueError('No held-out validation shapes')
    return sorted(selected),sorted(validation)


def calibrate(args,cfg):
    from .execution import Store,sample_shapes
    rows,meta=load_manifest(args.manifest,cfg)
    shapes,validation=calibration_shapes(rows,cfg,args.model)
    if args.dry_run:
        print(json.dumps({'calibration_shapes':shapes,'validation_shapes':validation,
            'calls':(len(shapes)+len(validation))*cfg['profiling']['repetitions']*(cfg['profiling']['warmup']+1)},indent=2));return
    runner=runner_for(args,cfg)
    runner.smoke(meta['class_names'])
    identity={'phase':'calibration','runner':runner.identity,'manifest_sha256':meta['manifest_sha256']}
    identity['run_id']=digest(identity)
    store=Store(args.output,identity)
    if (Path(args.output)/'cost_model.json').exists():
        saved=load_calibration(args.output)
        if saved['identity'] != identity:
            raise ValueError('Frozen calibration identity mismatch')
        print(f'Validated existing frozen calibration: {args.output}',flush=True)
        return
    started=time.monotonic()
    train=sample_shapes(runner,shapes,store,'calibration_trials',cfg['profiling']['repetitions'],cfg['seed'])
    artifact=fit_models(train)
    artifact.update({'identity':identity,'role':'calibration','calibration_trials':train,'policy_options':cfg['policies']})
    # Measure independent validation replicates after fitting; never refit from these values.
    valid=sample_shapes(runner,validation,store,'validation_trials',cfg['profiling']['repetitions'],cfg['seed']+100)
    artifact['validation_diagnostics']={p:fit_diagnostics(Estimator(artifact,p,cfg['policies']),valid) for p in ('affine','piecewise','lookup','conservative_lookup')}
    artifact['validation_trials']=valid
    artifact['lookup_table_bytes']=len(canonical(artifact['lookup']).encode())
    artifact['calibration_elapsed_this_session_s']=time.monotonic()-started
    artifact['calibration_id']=digest({'identity':identity,'table':train})
    atomic(Path(args.output)/'cost_model.json',artifact)
    atomic(Path(args.output)/'cost_model.receipt.json',{'sha256':sha(Path(args.output)/'cost_model.json')})
    print(f'Calibration complete: {args.output}/cost_model.json',flush=True)


def load_calibration(directory):
    directory=Path(directory)
    path=directory/'cost_model.json'
    if sha(path)!=read(directory/'cost_model.receipt.json')['sha256']:
        raise ValueError('Calibration artifact receipt mismatch')
    artifact=read(path)
    if artifact['role']!='calibration':
        raise ValueError('Not a deployable calibration artifact')
    return artifact


def pair_decisions(row,cfg,model,artifact,reference,scoring):
    shape=lambda box:shape_for(box,row,cfg,model)
    oracle=OracleCosts(reference)
    results=[]
    for a,b in itertools.combinations(row['rois'],2):
        shapes=[shape(box) for box in (a['box'],b['box'],union(a['box'],b['box']))]
        reference_separate=oracle.cost(shapes[0])+oracle.cost(shapes[1])
        reference_merged=oracle.cost(shapes[2])
        reference_merge=reference_merged<reference_separate
        # Scoring replicates are disjoint from the replicates selecting oracle decisions.
        separate=[x+y for x,y in zip(scoring[key(shapes[0])],scoring[key(shapes[1])])]
        merged=scoring[key(shapes[2])]
        sep,mer=float(np.mean(separate)),float(np.mean(merged))
        oracle_ms=mer if reference_merge else sep
        values=[np.asarray(separate)-np.asarray(merged)]
        gap=abs(reference_separate-reference_merged)
        bins=cfg['policies']['boundary_bins_ms']
        boundary=next((f'<= {v} ms' for v in bins if gap<=v),f'> {bins[-1]} ms')
        noise=float(np.std(values[0],ddof=1)/np.sqrt(len(merged))) if len(merged)>1 else 0
        for policy in POLICIES:
            if policy=='no_merge':
                decision={'merge':False,'fallback':None}
            elif policy=='geometric':
                decision={'merge':area(union(a['box'],b['box']))/(area(a['box'])+area(b['box']))<=cfg['policies']['geometric_area_ratio_max'],'fallback':None}
            elif policy=='oracle':
                decision={'merge':reference_merge,'fallback':None}
            else:
                decision=Estimator(artifact,policy,cfg['policies']).decision(*shapes)
            chosen=mer if decision['merge'] else sep
            results.append({'video_id':row['video_id'],'frame_index':row['frame_index'],'pair':[a['roi_id'],b['roi_id']],
                'policy':policy,**decision,'shapes':shapes,'oracle_merge':reference_merge,
                'separate_ms':sep,'merged_ms':mer,'chosen_ms':chosen,'oracle_ms':oracle_ms,
                'regret_ms':chosen-oracle_ms,'scoring_min_regret_ms':chosen-min(sep,mer),
                'reference_scoring_disagreement':reference_merge!=(mer<sep),
                'boundary':boundary,'reference_abs_gain_ms':gap,'noise_se_ms':noise,
                'near_noise_boundary':gap<=cfg['policies']['confidence_z']*noise,
                'lookup_exact_domain':all(key(s) in artifact['lookup'] for s in shapes),
                **geometry_strata(a['box'],b['box'])})
    return results


def evaluate(args,cfg):
    from .execution import Store,sample_shapes,sensor
    rows,meta=load_manifest(args.manifest,cfg)
    selected=cohort(rows,cfg,args.subset,args.limit_frames)
    work=estimate_work(selected,cfg,args.model)
    if args.dry_run:
        if args.models_dir and (args.models_dir/'cost_model.json').exists():
            prior=load_calibration(args.models_dir)
            average=np.mean([v['mean_ms'] for v in prior['lookup'].values()])
            calls=work['shape_network_calls_including_warmup']+work['frame_network_calls_upper_bound_including_warmup']
            work['rough_seconds_at_calibration_mean_upper_call_budget']=float(calls*average/1000)
            work['runtime_estimate_note']='Rough calibration-mean estimate; excludes IO/planning/thermal drift and shape/content variation'
        print(json.dumps(work,indent=2));return
    if work['unique_heldout_shapes']>cfg['profiling']['max_heldout_shapes']:
        raise ValueError(f'Held-out shape budget exceeded ({work["unique_heldout_shapes"]}); freeze a common smaller subset for all hosts')
    root=dataset_root(cfg,args.dataset_root)
    if names_from_yaml(cfg,root)!=meta['class_names']:
        raise ValueError('Dataset class mapping changed')
    verify_assets(selected,root,len(meta['class_names']),cfg)
    runner=runner_for(args,cfg)
    runner.smoke(meta['class_names'])
    artifact=load_calibration(args.models_dir)
    if artifact['identity']['runner']!=runner.identity or artifact['identity']['manifest_sha256']!=meta['manifest_sha256']:
        raise ValueError('Calibration model/hardware/software/thread/source/manifest identity mismatch')
    identity={'phase':'test','runner':runner.identity,'manifest_sha256':meta['manifest_sha256'],
              'calibration_id':artifact['calibration_id'],'calibration_sha256':sha(Path(args.models_dir)/'cost_model.json'),
              'cohort_sha256':digest([(r['video_id'],r['frame_index']) for r in selected]),'subset':args.subset,
              'fixture':meta['fixture'],'limit_frames':args.limit_frames}
    identity['run_id']=digest(identity)
    store=Store(args.output,identity)
    plans=[]
    # Freeze every deployable plan before any held-out cost table is read or collected.
    for row in selected:
        shape=lambda box:shape_for(box,row,cfg,args.model)
        frame={'video_id':row['video_id'],'frame_index':row['frame_index'],'policies':{}}
        for policy in POLICIES[:-1]:
            estimator=Estimator(artifact,policy,cfg['policies'])
            started=time.perf_counter_ns()
            groups,trace,kind=plan(row['rois'],policy,shape,cfg['policies'],estimator)
            elapsed=(time.perf_counter_ns()-started)/1e6
            frame['policies'][policy]={'groups':groups,'trace':trace,'planner_kind':kind,'planning_ms':elapsed}
        plans.append(frame)
    plan_path=Path(args.output)/'deployable_plans.json'
    if plan_path.exists():
        saved=read(plan_path)
        # On resume preserve original planner measurements and require identical decisions.
        def decisions(value):
            return [{**frame,'policies':{p:{k:v for k,v in data.items() if k!='planning_ms'} for p,data in frame['policies'].items()}} for frame in value]
        if canonical(decisions(saved))!=canonical(decisions(plans)):
            raise ValueError('Policy plan changed on resume')
        plans=saved
    else:
        atomic(plan_path,plans)
    table=sample_shapes(runner,work['all_union_shapes'],store,'heldout_trials',cfg['profiling']['evaluation_repetitions'],cfg['seed']+200)
    # Independent blocks: first half defines retrospective oracle; second half scores it.
    half=cfg['profiling']['evaluation_repetitions']//2
    reference={'role':'heldout_reference','calibration_id':artifact['calibration_id'],
               'table':{k:v[:half] for k,v in table.items()},'manifest_sha256':meta['manifest_sha256']}
    scoring={k:v[half:] for k,v in table.items()}
    atomic(Path(args.output)/'heldout_reference.json',reference)
    atomic(Path(args.output)/'heldout_scoring.json',{'role':'heldout_scoring','table':scoring})
    oracle=OracleCosts(reference)
    pairs=[];frame_results=[]
    rng=random.Random(cfg['seed']+300)
    for row,frame in zip(selected,plans):
        pairs.extend(pair_decisions(row,cfg,args.model,artifact,reference,scoring))
        shape=lambda box:shape_for(box,row,cfg,args.model)
        oracle_started=time.perf_counter_ns()
        oracle_groups,trace,kind=plan(row['rois'],'oracle',shape,cfg['policies'],oracle=oracle)
        oracle_planning_ms=(time.perf_counter_ns()-oracle_started)/1e6
        frame['policies']['oracle']={'groups':oracle_groups,'trace':trace,'planner_kind':kind,'planning_ms':oracle_planning_ms}
        tensor=runner.base_tensor(asset(root,row['image']))
        for repetition in range(cfg['profiling']['evaluation_repetitions']):
            order=list(POLICIES);rng.shuffle(order)
            for position,policy in enumerate(order):
                item_key=[row['video_id'],row['frame_index'],repetition,policy]
                existing=store.get('frame_trials',item_key)
                if existing:
                    frame_results.append(existing);continue
                proposal=frame['policies'][policy]
                groups=proposal['groups']
                fallback=not groups
                boxes=[g['box'] for g in groups] if groups else [[0,0,row['width'],row['height']]]
                # Prepare and warm outside timed serial block; crop overhead is measured separately.
                t0=time.perf_counter_ns()
                prepared=[runner.crop(tensor,box,row) for box in boxes]
                crop_ms=(time.perf_counter_ns()-t0)/1e6
                for value,_ in prepared:
                    for _ in range(cfg['profiling']['warmup']):
                        runner.infer(value)
                before=sensor()
                started=time.perf_counter_ns()
                call_ms=[runner.elapsed(value) for value,_ in prepared]
                serial_ms=(time.perf_counter_ns()-started)/1e6
                effective_shapes=[value.tensor_hw for value,_ in prepared]
                predicted=[]
                for s in effective_shapes:
                    if policy=='oracle':
                        predicted.append(oracle.cost(s))
                    else:
                        predictor=Estimator(artifact,policy if policy not in ('geometric','no_merge') else 'lookup',cfg['policies'])
                        predicted.append(predictor.estimate(s)[0])
                result={'video_id':row['video_id'],'frame_index':row['frame_index'],'policy':policy,
                    'repetition':repetition,'policy_order':position,'candidate_rois':len(row['rois']),
                    'final_segments':len(boxes),'groups':groups,'boxes':boxes,'snapped_boxes':[b for _,b in prepared],
                    'effective_shapes':effective_shapes,'effective_pixels':sum(h*w for h,w in effective_shapes),
                    'calls':len(boxes),'predicted_total_ms':sum(predicted) if all(v is not None for v in predicted) else None,
                    'prediction_source':'retrospective_reference' if policy=='oracle' else 'lookup_diagnostic' if policy in ('geometric','no_merge') else policy,
                    'serial_inference_ms':serial_ms,'call_latencies_ms':call_ms,'crop_preprocess_ms':crop_ms,
                    'planning_ms':proposal['planning_ms'],'pipeline_overhead_ms':crop_ms+max(0,serial_ms-sum(call_ms)),
                    'full_frame_fallback':fallback,'planner_kind':proposal['planner_kind'],'trace':proposal['trace'],
                    'before':before,'after':sensor(),'last_execution_trace':runner.execution_trace()}
                store.put('frame_trials',item_key,result)
                frame_results.append(result)
        print(f'Frame replay {row["video_id"]}/{row["frame_index"]}',flush=True)
    atomic(Path(args.output)/'pair_decisions.jsonl',pairs,lines=True)
    atomic(Path(args.output)/'frame_results.jsonl',frame_results,lines=True)
    summaries={}
    for policy in POLICIES:
        pr=[r for r in pairs if r['policy']==policy]
        fr=[r for r in frame_results if r['policy']==policy]
        strata={}
        for field in ('boundary','layout','empty_area_stratum','near_noise_boundary','lookup_exact_domain'):
            strata[field]={str(v):pair_summary([r for r in pr if r[field]==v]) for v in sorted({r[field] for r in pr},key=str)}
        summaries[policy]={'pairwise':pair_summary(pr),'pairwise_ci':bootstrap_pairs(pr,cfg['policies']['bootstrap_repetitions'],cfg['seed']),
            'minimum_gain_sensitivity':{str(m):pair_summary([r for r in pr if r['reference_abs_gain_ms']>m]) for m in [0,*cfg['policies']['boundary_bins_ms']]},
            'frame_paired_comparison':paired_frame_comparison(frame_results,policy,cfg['policies']['bootstrap_repetitions'],cfg['seed']),
            'frame_all':frame_summary(fr),'frame_ci':bootstrap_frames(fr,cfg['policies']['bootstrap_repetitions'],cfg['seed']),'frame_merge_eligible':frame_summary([r for r in fr if r['candidate_rois']>=2]),'strata':strata}
    summary={'identity':identity,'status':'COMPLETE','fixture':meta['fixture'], 'work':work,'policies':summaries,
        'test_fit_diagnostics':{p:fit_diagnostics(Estimator(artifact,p,cfg['policies']),scoring) for p in ('affine','piecewise','lookup','conservative_lookup')},
        'candidates':dict(Counter(len(r['rois']) for r in selected)),
        'scope':'Fixed GT-derived proposals, synthetic shape microbenchmark pair decisions plus actual content serial ROI replay. No detection mAP or closed-loop tracking.'}
    write_evaluation_report(args.output,summary)
    atomic(Path(args.output)/'summary.json',summary)
    print(f'Evaluation complete: {args.output}/summary.json',flush=True)


def write_evaluation_report(output,summary):
    lines=['# Experiment B evaluation','',
           '**SYNTHETIC FIXTURE: not scientific evidence.**' if summary['fixture'] else 'Measured fixed-workload evaluation.',
           '',summary['scope'],'',f"Cohort SHA-256: `{summary['identity']['cohort_sha256']}`",'',
           '| Policy | Pair accuracy | False merge rate | Mean regret ms | Mean serial frame ms |',
           '|---|---:|---:|---:|---:|']
    for policy in POLICIES:
        data=summary['policies'][policy];pair=data['pairwise'];frame=data['frame_all']
        lines.append('| '+' | '.join(map(str,[policy,pair.get('accuracy'),pair.get('false_merge_rate'),pair.get('mean_regret_ms'),frame.get('mean_serial_inference_ms')]))+' |')
    lines += ['', 'Pairwise oracle selection and scoring use disjoint held-out trial halves. Negative reference regret is retained as measurement noise; scoring-minimum regret is separate.',
        'Frame oracle is exact only for the small-n additive shape-cost model; larger-n references are greedy. No real-time scheduling, closed-loop recovery or mAP is measured.',
        'Independent validation/test fit errors, sequence bootstrap intervals, merge-eligible frames, domain coverage, boundary sensitivity and geometry strata are recorded in summary.json.',
        'No frames are silently excluded; an error stops the run. Atomic checkpoints support resuming with an identical source/config/runtime identity.']
    (Path(output)/'evaluation_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def aggregate(args,cfg):
    from .reporting import aggregate_results
    aggregate_results(args.results_root,args.output,cfg)


def bundle(args,cfg):
    rows,meta=load_manifest(args.manifest,cfg)
    source=Path(args.manifest).parent
    members={
        'manifest.jsonl':Path(args.manifest),
        'manifest.meta.json':source/'manifest.meta.json',
        'config.example.yaml':Path(args.config),
    }
    checksums={name:sha(path) for name,path in members.items()}
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in members.items():
            archive.write(path,name)
        archive.writestr('checksums.json',canonical(checksums)+'\n')
    print(f'Transfer bundle: {target}; manifest {meta["manifest_sha256"]}')


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('command',choices=('inspect','prepare','calibrate','evaluate','aggregate','bundle','model-smoke','fixture'))
    result.add_argument('--config',type=Path,default=Path(__file__).with_name('config.example.yaml'))
    result.add_argument('--manifest',type=Path)
    result.add_argument('--output',type=Path,default=ROOT/'outputs/roi_merge_b')
    result.add_argument('--dataset-root',type=Path)
    result.add_argument('--checkpoint-root',type=Path,help='Root containing canonical trained_models/... tree; defaults to repository root')
    result.add_argument('--cohort-manifest',type=Path,help='Optional frozen Stage 3 final TSV; exact IDs must occur in audit index')
    result.add_argument('--model',choices=MODELS)
    result.add_argument('--models-dir',type=Path,help='Calibration directory, never held-out timings')
    result.add_argument('--results-root',type=Path)
    result.add_argument('--subset',choices=('smoke','pilot','full'),default='full')
    result.add_argument('--limit-frames',type=int,help='Explicit non-headline prefix of selected frozen cohort')
    result.add_argument('--dry-run',action='store_true')
    result.add_argument('--skip-inference',action='store_true',help='Inspect assets only; not a complete preflight')
    result.add_argument('--cpu-threads',type=int)
    result.add_argument('--interop-threads',type=int)
    result.add_argument('--opencv-threads',type=int)
    result.add_argument('--affinity',type=int,nargs='+')
    result.add_argument('--synthetic-smoke',action='store_true',help='TEST ONLY: permitted only with explicitly synthetic fixture manifest; never publication data')
    return result


def main(argv=None):
    cli=parser();args=cli.parse_args(argv);cfg=config(args.config)
    for k in ('cpu_threads','interop_threads','opencv_threads','limit_frames'):
        if getattr(args,k) is not None and getattr(args,k)<1:
            cli.error(f'{k} must be positive')
    if args.command in ('calibrate','evaluate','bundle') and args.manifest is None:
        cli.error('--manifest is required')
    if args.command in ('calibrate','evaluate','model-smoke') and args.model is None:
        cli.error('--model is required')
    if args.command=='evaluate' and args.models_dir is None:
        cli.error('--models-dir is required')
    if args.command=='aggregate' and args.results_root is None:
        cli.error('--results-root is required')
    if args.command=='fixture':
        from .fixture import create_fixture
        create_fixture(args.output,cfg);return 0
    try:
        return globals()[{'inspect':'inspect_run','model-smoke':'model_smoke'}.get(args.command,args.command)](args,cfg) or 0
    except Exception as error:
        import traceback
        failure=args.output/'failure.json' if args.command!='bundle' else args.output.with_suffix('.failure.json')
        atomic(failure,{'command':args.command,'config_sha256':digest(cfg),'error_type':type(error).__name__,
                        'message':str(error),'traceback':traceback.format_exc()})
        raise
