"""Stage-2 runner. Run from repository root with ``python -m ...run``."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import traceback

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.ops import box_iou
import yaml

from dataset.transforms.fixed_size_test_transform import FixedSizeTestTransform
from model.roissd import RoiSSD
from model.roissd_mobilenet import RoiSSDMobileNet
from tools.helpers.config_reader import load_config
from .instrumentation import ForwardObserver, geometry, matching

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DATA=HERE/'data'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):digest.update(part)
    return digest.hexdigest()


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_rows(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def prepare(settings):
    """Freeze manifest-selected ROI checkpoints and the existing canonical subset."""
    DATA.mkdir(exist_ok=True)
    lock_path=DATA/'selection_lock.json'
    if lock_path.exists():
        lock=json.loads(lock_path.read_text())
        for entry in lock['models']:
            assert sha(ROOT/entry['checkpoint'])==entry['checkpoint_sha256'], 'Locked checkpoint changed'
            assert sha(DATA/entry['config_snapshot'])==entry['config_sha256'], 'Locked config changed'
        assert sha(DATA/'subset.jsonl')==lock['subset_sha256'], 'Locked subset changed'
        return lock
    manifest=ROOT/settings['stage1_manifest']
    roles_path=ROOT/settings['role_manifest']
    roles=json.loads(roles_path.read_text())['families']
    audit=json.loads((ROOT/settings['dataset_audit']).read_text())
    names=audit['names']
    assert len(names)==30 and len(set(names))==30
    models=[]
    (DATA/'configs').mkdir(exist_ok=True)
    for family in ['ROI-SSD','ROI-SSD-MobileNet']:
        selected=roles[family]
        checkpoint=selected['roi_checkpoint']
        # Do not discover or substitute a checkpoint from the filesystem.
        assert checkpoint in manifest.read_text(encoding='utf-8'), 'Checkpoint absent from stage-1 MD manifest'
        cfg=load_config(str(ROOT/selected['roi_config']))
        assert not cfg['train_params'].get('penalized_roissd',False)
        assert cfg['train_params']['dataset']=='yolo-imagenet-vid'
        assert cfg['dataset_params']['num_classes']==31
        expected='roissd' if family=='ROI-SSD' else 'roissd-mobilenet'
        assert cfg['train_params']['model']==expected
        source_yaml=Path(cfg['dataset_params']['yolo_dataset_yaml'])
        actual=yaml.safe_load(source_yaml.read_text(encoding='utf-8'))['names']
        actual=[actual[i] for i in range(len(actual))] if isinstance(actual,dict) else actual
        assert actual==names, 'Dataset class compatibility broken'
        filename='configs/'+expected+'.yaml'
        (DATA/filename).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        models.append({'model_family':family,'checkpoint':checkpoint,'checkpoint_sha256':sha(ROOT/checkpoint),
                       'role':'roi','config_snapshot':filename,'config_sha256':sha(DATA/filename),
                       'source_config':selected['roi_config']})
    source=ROOT/settings['source_subset']
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    for i,row in enumerate(rows):
        image=Path(audit['root'])/row['image']
        label=Path(audit['root'])/row['label']
        assert sha(label)==row['label_sha256'], 'Dataset labels changed since stage 1'
        row.update(sample_id=i,image_sha256=sha(image))
    (DATA/'subset.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows),encoding='utf-8')
    lock={'stage1_manifest':settings['stage1_manifest'],'stage1_manifest_sha256':sha(manifest),
          'role_manifest':settings['role_manifest'],'role_manifest_sha256':sha(roles_path),
          'dataset_root':audit['root'],'class_names':names,'background_index':0,'foreground_offset':1,
          'models':models,'num_samples':len(rows),'source_subset':settings['source_subset'],
          'source_subset_sha256':sha(source),'subset_sha256':sha(DATA/'subset.jsonl')}
    write_json(lock_path,lock)
    return lock


def crop(image,row,size,seed):
    """Use the project transform with reproducible per-image jitter for every size/model."""
    random.seed(seed+row['sample_id']);torch.manual_seed(seed+row['sample_id'])
    h,w=image.shape[:2]
    boxes=[];labels=[]
    for cid,cx,cy,bw,bh in row['gt']:
        boxes.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]);labels.append(cid+1)
    target={'boxes':tv_tensors.BoundingBoxes(torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
                                          format='XYXY',canvas_size=(h,w)),
            'labels':torch.tensor(labels,dtype=torch.long),'difficult':torch.zeros(len(labels),dtype=torch.long)}
    transform=FixedSizeTestTransform(size[0],size[1],[.485,.456,.406],[.229,.224,.225],normalize=True)
    tensor,target=transform.transforms['test'](tv_tensors.Image(torch.from_numpy(image.copy()).permute(2,0,1)),target)
    ah,aw=tensor.shape[-2:]
    assert (ah,aw)==tuple(size), f'Requested {size}, got {(ah,aw)}; no silent padding/rounding permitted'
    gt=torch.as_tensor(target['boxes']).float()/torch.tensor([aw,ah,aw,ah])
    assert torch.isfinite(tensor).all() and torch.isfinite(gt).all()
    assert ((target['labels']>=1)&(target['labels']<=30)).all()
    return tensor.unsqueeze(0),gt,target['labels']


def qstats(values,prefix):
    if not values:return {prefix+k:None for k in ['mean','median','p5','p25']}
    a=np.asarray(values)
    return {prefix+'mean':float(a.mean()),prefix+'median':float(np.median(a)),
            prefix+'p5':float(np.percentile(a,5)),prefix+'p25':float(np.percentile(a,25))}


def load_model(entry,settings,device):
    cfg=load_config(str(DATA/entry['config_snapshot']))
    params=dict(cfg['model_params'])
    params.update(low_score_threshold=settings['confidence'],nms_threshold=settings['nms_iou'],
                  detections_per_img=settings['max_detections'])
    cls=RoiSSD if entry['model_family']=='ROI-SSD' else RoiSSDMobileNet
    model=cls(config=params,num_classes=31)
    checkpoint=torch.load(ROOT/entry['checkpoint'],map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint.get('model',checkpoint),strict=True)
    return model.to(device).eval()


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    settings=yaml.safe_load((HERE/'config.yaml').read_text())
    lock=prepare(settings)
    if args.prepare_only:
        print('Prepared locked data:',DATA);return
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    rows=[json.loads(line) for line in (DATA/'subset.jsonl').read_text().splitlines()]
    if args.max_samples:rows=rows[:args.max_samples]
    sizes=args.sizes or settings['sizes']
    out=(ROOT/settings['output']) if args.output is None else args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    (out/'plots').mkdir(exist_ok=True)
    (out/'samples').mkdir(exist_ok=True)
    images=[]
    for row in rows:
        path=Path(lock['dataset_root'])/row['image']
        assert sha(path)==row['image_sha256'], 'Image bytes changed'
        assert sha(Path(lock['dataset_root'])/row['label'])==row['label_sha256'], 'GT labels changed'
        image=cv2.imread(str(path));assert image is not None,str(path)
        images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    runtime={'device':str(device),'torch':torch.__version__,'opencv':cv2.__version__,
             'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'seed':settings['seed'],'confidence':settings['confidence'],'nms_iou':settings['nms_iou'],
             'max_detections':settings['max_detections'],'num_samples':len(rows),'sizes':sizes,
             'subset_sha256':lock['subset_sha256'],'role_manifest_sha256':lock['role_manifest_sha256'],
             'settings_sha256':sha(HERE/'config.yaml'),'command':__import__('sys').argv,
             'scope':'diagnosis only; no model edits, training or size rounding',
             'stage3_allowed':False,'status':'RUNNING'}
    write_json(out/'run_metadata.json',runtime)
    (out/'errors_or_assertions.log').write_text('',encoding='utf-8')
    metrics=[];internals=[];geometries=[];matches=[];per_class=[];preprocessing=[];rectangles=[]
    # Share the exact project AP implementation, extracted without importing the
    # inference CLI's global device resolution or sample-output machinery.
    import ast
    source=ast.parse((ROOT/'tools/infer.py').read_text())
    functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ['get_iou','compute_map']]
    namespace={'np':np}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'tools/infer.py','exec'),namespace)
    compute_map=namespace['compute_map']
    names=lock['class_names']
    with (out/'per_example.jsonl').open('w',encoding='utf-8') as examples, (out/'anchor_matching_per_gt.jsonl').open('w',encoding='utf-8') as gtlog:
        for entry in lock['models']:
            print('Loading',entry['model_family'],entry['checkpoint'],flush=True)
            model=load_model(entry,settings,device)
            with ForwardObserver(model) as observer,torch.inference_mode():
                for size in sizes:
                    identity={'model_family':entry['model_family'],'checkpoint_id':entry['checkpoint'],
                              'requested_h':size,'requested_w':size,'actual_h':size,'actual_w':size,
                              'padding_or_rounding':False}
                    print(entry['model_family'],f'{size}x{size}',flush=True)
                    predictions=[];gts=[];difficult=[];pre=[];post=[];maxconf=[];postconf=[];uncapped=[]
                    ious_all=[];positive_all=[];forced_all=[];head_stats={};head_iou={};decoded_iou={}
                    failed=False
                    for index,(image,row) in enumerate(zip(images,rows)):
                        try:
                            tensor,gt,labels=crop(image,row,(size,size),settings['seed'])
                            digest=hashlib.sha256(tensor.numpy().tobytes()+gt.numpy().tobytes()+labels.numpy().tobytes()).hexdigest()
                            preprocessing.append({**identity,'sample_id':row['sample_id'],'tensor_gt_sha256':digest,
                                                  'image':row['image'],'gt_count':len(gt),'input_min':float(tensor.min()),
                                                  'input_max':float(tensor.max()),'seed':settings['seed']+row['sample_id']})
                            if index<2 and size in [96,128,160]:
                                pix=(tensor[0]*torch.tensor([.229,.224,.225]).view(3,1,1)+torch.tensor([.485,.456,.406]).view(3,1,1)).clamp(0,1)
                                cv2.imwrite(str(out/'samples'/f'sample{index}_{size}.png'),cv2.cvtColor((pix.permute(1,2,0).numpy()*255).round().astype('uint8'),cv2.COLOR_RGB2BGR))
                            observer.reset()
                            _,result=model(tensor.to(device),None)
                            assert len(result)==1
                            det=result[0]
                            assert torch.isfinite(det['boxes']).all() and torch.isfinite(det['scores']).all()
                            assert ((det['labels']>=1)&(det['labels']<=30)).all()
                            assert ((det['boxes']>=0)&(det['boxes']<=1)).all()
                            assert (det['boxes'][:,2:]>=det['boxes'][:,:2]).all()
                            assert len(observer.nms_calls)==1
                            call=observer.nms_calls[0]
                            pre.append(call['pre']);post.append(len(det['scores']));uncapped.append(call['after_nms_before_cap'])
                            maxconf.append(max(h['foreground_max_confidence'] for h in observer.heads.values()))
                            postconf.append(float(det['scores'].max()) if len(det['scores']) else 0.)
                            if index==0:
                                geo=geometry(observer.anchors,observer.reference,size,size)
                                geometries.append({**identity,**geo})
                                for h in range(len(model.cls_heads)):
                                    head=observer.heads.get(h)
                                    internals.append({**identity,'input_h':size,'input_w':size,
                                                      'active_backbone_depth':len(observer.heads),'active_backbone_blocks':observer.blocks,
                                                      'last_computed_block':observer.blocks[-1],
                                                      'active_heads':sorted(observer.heads),'head_index':h,'head_active':head is not None,
                                                      'feature_h':head['feature_h'] if head else 0,'feature_w':head['feature_w'] if head else 0,
                                                      'spatial_positions':head['feature_h']*head['feature_w'] if head else 0,
                                                      'anchors_per_location':head['anchors_per_location'] if head else 0,
                                                      'num_anchors_head':head['cls_slots'] if head else 0,
                                                      'loc_predictions_head':head['loc_slots'] if head else 0,
                                                      'cls_predictions_head':head['cls_slots'] if head else 0,
                                                      'loc_output_shape':head['loc_output_shape'] if head else [],
                                                      'cls_output_shape':head['cls_output_shape'] if head else [],
                                                      'total_anchors':len(observer.anchors),'localization_shape':observer.loc_shape,
                                                      'classification_shape':observer.cls_shape,'decoded_box_slots':len(observer.decoded)})
                            gt_device=gt.to(device)
                            best,positive,forced=matching(observer.anchors,gt_device,model.iou_threshold)
                            ious_all.extend(best);positive_all.extend(positive);forced_all.extend(forced)
                            for j,(a,b,c) in enumerate(zip(best,positive,forced)):
                                gtlog.write(json.dumps({**identity,'sample_id':row['sample_id'],'gt_index':j,
                                                       'class_id':int(labels[j])-1,'max_anchor_iou':a,
                                                       'positive_matches':b,'assigned_positive_including_forced':c})+'\n')
                            offset=0
                            for h,head in observer.heads.items():
                                head_stats.setdefault(h,[]).append(head['foreground_max_confidence'])
                                end=offset+head['cls_slots']
                                if len(gt):
                                    head_iou.setdefault(h,[]).extend(box_iou(gt_device,observer.anchors[offset:end]).max(1).values.tolist())
                                    decoded_iou.setdefault(h,[]).extend(box_iou(gt_device,observer.decoded[offset:end].clamp(0,1)).max(1).values.tolist())
                                offset=end
                            pred={n:[] for n in names};truth={n:[] for n in names}
                            for box,label,score in zip(det['boxes'].cpu().tolist(),det['labels'].cpu().tolist(),det['scores'].cpu().tolist()):
                                pred[names[label-1]].append([*box,score])
                            for box,label in zip(gt.tolist(),labels.tolist()):truth[names[label-1]].append(box)
                            predictions.append(pred);gts.append(truth);difficult.append({n:[0]*len(b) for n,b in truth.items()})
                            examples.write(json.dumps({**identity,'sample_id':row['sample_id'],'input_sha256':digest,
                                                       'predictions_pre_nms':pre[-1],'predictions_post_nms':post[-1],
                                                       'max_confidence':maxconf[-1],'gt':truth,'predictions':pred})+'\n')
                            if (index+1)%64==0:print('  samples',index+1,'/',len(rows),flush=True)
                        except Exception:
                            failed=True
                            error={'identity':identity,'sample_id':row['sample_id'],'heads':observer.heads,
                                   'blocks':observer.blocks,'traceback':traceback.format_exc()}
                            with (out/'errors_or_assertions.log').open('a',encoding='utf-8') as log:log.write(json.dumps(error,indent=2)+'\n')
                            metrics.append({**identity,'status':'ERROR','num_samples':index})
                            print('ERROR',traceback.format_exc(),flush=True)
                            break  # Stop this model run: do not hide a structural failure.
                    if failed:break
                    ap=[compute_map(predictions,gts,iou_threshold=.5+.05*i,method='area',difficult=difficult) for i in range(10)]
                    metrics.append({**identity,'status':'PASS','num_samples':len(rows),'map50':ap[0][0],
                                    'map5095':float(np.mean([a[0] for a in ap])),'recall50':ap[0][2],
                                    'num_gt':len(ious_all),'predictions_pre_nms_mean':float(np.mean(pre)),
                                    'predictions_post_nms_mean':float(np.mean(post)),
                                    'predictions_after_nms_before_cap_mean':float(np.mean(uncapped)),
                                    'predictions_per_image_mean':float(np.mean(post)),
                                    'max_confidence_mean':float(np.mean(maxconf)),
                                    'post_nms_max_confidence_mean':float(np.mean(postconf))})
                    matches.append({**identity,**qstats(ious_all,'max_anchor_iou_'),'num_gt':len(ious_all),
                                    'fraction_gt_iou_ge_03':float(np.mean(np.array(ious_all)>=.3)) if ious_all else None,
                                    'fraction_gt_iou_ge_05':float(np.mean(np.array(ious_all)>=.5)) if ious_all else None,
                                    'positive_matches_per_gt_mean':float(np.mean(positive_all)) if positive_all else None,
                                    'assigned_positive_per_gt_mean':float(np.mean(forced_all)) if forced_all else None,
                                    'positive_matches_per_image_mean':sum(positive_all)/len(rows)})
                    for record in internals:
                        if record['model_family']==entry['model_family'] and record['requested_h']==size and record['head_active']:
                            h=record['head_index']
                            record.update(foreground_max_confidence_mean=float(np.mean(head_stats[h])),
                                          head_max_anchor_iou_mean=float(np.mean(head_iou[h])) if head_iou.get(h) else None,
                                          head_max_decoded_iou_mean=float(np.mean(decoded_iou[h])) if decoded_iou.get(h) else None)
                    for n in names:
                        value=ap[0][1].get(n,float('nan'))
                        per_class.append({**identity,'class_name':n,'gt_count':sum(len(g[n]) for g in gts),
                                          'ap50':value if np.isfinite(value) else None,
                                          'ap5095':float(np.mean([a[1][n] for a in ap])) if np.isfinite(value) else None})
                    for name,data in [('size_sweep_metrics',metrics),('internal_shapes',internals),('anchor_geometry',geometries),
                                      ('anchor_matching',matches),('per_class_ap',per_class),('preprocessing',preprocessing)]:
                        csv_rows(out/(name+'.csv'),data)
                    print('  mAP50',metrics[-1]['map50'],'GT',len(ious_all),flush=True)
                if not failed:
                    for h,w in settings['rectangular_probes']:
                        try:
                            x,_,_=crop(images[0],rows[0],(h,w),settings['seed']);observer.reset()
                            model(x.to(device),None)
                            rectangles.append({'model_family':entry['model_family'],'requested_h':h,'requested_w':w,
                                               'actual_h':x.shape[-2],'actual_w':x.shape[-1],
                                               **geometry(observer.anchors,observer.reference,h,w),'status':'PASS'})
                        except Exception:
                            failed=True
                            with (out/'errors_or_assertions.log').open('a') as log:log.write(traceback.format_exc())
                            break
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
    # Persist failure rows too, even if an assertion occurred before any completed size.
    for name,data in [('size_sweep_metrics',metrics),('internal_shapes',internals),('anchor_geometry',geometries),
                      ('anchor_matching',matches),('per_class_ap',per_class),('preprocessing',preprocessing),('rectangular_probes',rectangles)]:
        csv_rows(out/(name+'.csv'),data)
    # Both models must have exactly the same tensors/GT for every sample and shape.
    comparisons={}
    for r in preprocessing:
        key=(r['sample_id'],r['requested_h'],r['requested_w'])
        if key in comparisons:assert comparisons[key]==r['tensor_gt_sha256'],'Preprocessing differs across model families'
        comparisons[key]=r['tensor_gt_sha256']
    expected=2*len(sizes)
    error=(out/'errors_or_assertions.log').stat().st_size>0
    runtime.update(status='ERROR' if error or len(metrics)!=expected else 'COMPLETE',
                   exact_preprocessing_across_families=True,
                   structural_checks_passed=not error and len(metrics)==expected,
                   stage3_allowed=False)
    write_json(out/'run_metadata.json',runtime)
    from .report import build_report
    build_report(out,lock)
    if runtime['status']!='COMPLETE':raise RuntimeError('Diagnostic failed; inspect errors_or_assertions.log. Stage 3 blocked.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--max-samples',type=int,default=0,help='0 means all 256 locked samples; smaller runs are smoke only')
    parser.add_argument('--sizes',type=int,nargs='+')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    assert args.max_samples>=0 and args.threads>0
    run(args)


if __name__=='__main__':main()
