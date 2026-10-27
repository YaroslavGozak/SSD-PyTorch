"""Stage 3b paired context, border, preprocessing and head diagnostics."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

from . import run_pilot as pilot
from . import oracle_diagnosis as stage3a
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from tools.helpers import pipeline
from tools.trackers.static_padding_tracker import StaticPaddingTracker
from tools.benchmarks.video_artifacts import write_json, detection_metrics
from experiments.roissd_128_anomaly.instrumentation import ForwardObserver

OUT = pilot.ROOT/'final_experiments/crop_domain_diagnosis'
PADDINGS = (32,64,96,128,160)
MEAN = torch.tensor([.485,.456,.406])[:,None,None]
STD = torch.tensor([.229,.224,.225])[:,None,None]


def write_csv(name, rows):
    import csv
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict,tuple)) else v for k,v in row.items()})


class HeadObserver(ForwardObserver):
    def _head_hook(self,index,kind):
        original=super()._head_hook(index,kind)
        def hook(module,args,output):
            original(module,args,output)
            if kind=='cls':
                n,_,h,w=output.shape
                scores=output.view(n,-1,self.model.num_classes,h,w).permute(0,3,4,1,2)
                self.heads[index]['scores']=scores.reshape(-1,self.model.num_classes).softmax(-1)
        return hook


def crop_rois(gt, width, height, padding):
    generator=StaticPaddingTracker(padding,padding,.5,(320,320),hold_last_for_frames=0)
    rois=generator.update([{'class':c,'bbox':box,'confidence':1.} for c,_,box in stage3a.objects(gt)],
                           (height,width))['rois']
    source=[r['roi'] for r in rois]
    model=pipeline._convert_rois_between_spaces(source,(width,height),(320,320))
    return pipeline._convert_rois_between_spaces(model,(320,320),(width,height))


def padded_tensor(tensor, box, padding, frame_hw):
    h,w=frame_hw
    tx1=math.floor(box[0]*320/w-padding)
    ty1=math.floor(box[1]*320/h-padding)
    tx2=math.ceil(box[2]*320/w+padding)
    ty2=math.ceil(box[3]*320/h+padding)
    source=tensor[0] if tensor.ndim==4 else tensor
    result=torch.zeros(3,ty2-ty1,tx2-tx1,dtype=source.dtype)
    ix1,iy1,ix2,iy2=max(0,tx1),max(0,ty1),min(320,tx2),min(320,ty2)
    result[:,iy1-ty1:iy2-ty1,ix1-tx1:ix2-tx1]=source[:,iy1:iy2,ix1:ix2]
    return result,[tx1*w/320,ty1*h/320,tx2*w/320,ty2*h/320]


def crop_first(image, roi, shape):
    x1,y1,x2,y2=map(int,roi)
    raw=torch.from_numpy(image[y1:y2,x1:x2].copy()).permute(2,0,1).float()/255
    resized=F.interpolate(raw[None],size=shape,mode='bilinear',align_corners=False)[0]
    return (resized-MEAN)/STD


def unique_matches(detections, gt):
    matches={}
    for det in sorted(detections,key=lambda d:-d['confidence']):
        candidates=gt.get(det['class'],[])
        overlaps=[stage3a.get_iou(det['bbox'],box) for box in candidates]
        if overlaps and max(overlaps)>=.5:
            key=(det['class'],int(np.argmax(overlaps)))
            if key not in matches:matches[key]=det
    return matches


def infer_target(model, observer, tensor, roi, box, cls, label, frame_hw, identity):
    observer.reset()
    _,out=model(tensor[None],None)
    det=out[0]
    trace=model.get_runtime_log()
    assert trace['depth_policy']=='until_1x1'
    assert trace['N_anchors']==trace['N_loc_predictions']==trace['N_cls_predictions']
    x1,y1,x2,y2=roi
    cw,ch=x2-x1,y2-y1
    target=torch.tensor([[(box[0]-x1)/cw,(box[1]-y1)/ch,(box[2]-x1)/cw,(box[3]-y1)/ch]])
    anchor_iou=box_iou(target,observer.anchors)[0]
    decoded=observer.decoded.clamp(0,1)
    decoded_iou=box_iou(target,decoded)[0]
    scores=torch.cat([observer.heads[i]['scores'] for i in sorted(observer.heads)])
    counts=trace['anchors_per_head']
    ends=np.cumsum(counts)
    head_rows=[]
    start=0
    for head,count in enumerate(counts):
        head_rows.append({**identity,'head':head,'spatial_shape':trace['feature_shapes'][head],
                          'anchor_count':count,'max_anchor_iou':float(anchor_iou[start:start+count].max()),
                          'max_class_correct_foreground_confidence':float(scores[start:start+count,label].max()),
                          'best_decoded_iou':float(decoded_iou[start:start+count].max())})
        start+=count
    mapped=pipeline.tensor_to_detection_list(det,{i: str(i) for i in range(model.num_classes)},cw,ch,(x1,y1))
    provenance=[]
    for index,d in enumerate(mapped):
        candidates=torch.nonzero((decoded-det['boxes'][index]).abs().amax(1)<=1e-6).flatten()
        candidates=candidates[(scores[candidates,int(det['labels'][index])]-det['scores'][index]).abs()<=1e-6]
        assert len(candidates), 'No source anchor for a returned prediction'
        source=int(candidates[0])
        d['head']=int(np.searchsorted(ends,source,side='right'))
        d['class_id']=int(det['labels'][index])
        d['class']=cls if d['class_id']==label else str(d['class_id'])
        d['origin_object_id']=identity['object_id']
        provenance.append(d)
    valid=stage3a.postprocess(provenance,.5,frame_hw[1],frame_hw[0],nms=False)
    correct=[d for d in valid if d['class_id']==label]
    ranked=sorted(correct,key=lambda d:-d['confidence'])
    matched=next((d for d in ranked if stage3a.get_iou(d['bbox'],box)>=.5),None)
    predicted_iou=max((stage3a.get_iou(d['bbox'],box) for d in correct),default=0.)
    maxconf=max((d['confidence'] for d in correct),default=0.)
    # Attribution uses a returned, thresholded, class-correct matched prediction.
    matched_head=matched['head'] if matched else None
    geo=stage3a.geometry(box,roi)
    row={**identity, 'class':cls,'roi':roi,'gt_box':box,
         'gt_xyxy_in_crop':[box[0]-x1,box[1]-y1,box[2]-x1,box[3]-y1],
         'gt_xyxy_normalized':target[0].tolist(),**geo,
         'network_h':tensor.shape[-2],'network_w':tensor.shape[-1],
         'selected_L_R':trace['selected_L_R'],'active_heads':trace['active_heads'],
         'feature_shapes':trace['feature_shapes'],'final_feature_shape':trace['final_feature_shape'],
         'anchors_per_head':counts,'total_anchors':trace['N_anchors'],
         'best_anchor_iou':float(anchor_iou.max()),
         'raw_max_class_correct_confidence':float(scores[:,label].max()),
         'max_class_correct_confidence':maxconf,'best_prediction_iou':predicted_iou,
         'best_decoded_iou':float(decoded_iou.max()), 'TP50_own_crop':matched is not None,
         'FN50_own_crop':matched is None,
         'FP_relative_to_target_only':len(correct)-(1 if matched else 0),
         'matched_prediction_head':matched_head,
         'matched_prediction_confidence':matched['confidence'] if matched else None,
         'matched_prediction_iou':stage3a.get_iou(matched['bbox'],box) if matched else None}
    for hr in head_rows:
        hr['gave_own_crop_matched_prediction']=hr['head']==matched_head
    return row,head_rows,provenance


def main():
    pilot._configure_runtime()
    OUT.mkdir(parents=True,exist_ok=True)
    old_meta=json.loads((stage3a.OUT/'metadata.json').read_text())
    assert old_meta['status']=='COMPLETE'
    assert all(pilot._sha256(stage3a.OUT/p)==digest for p,digest in old_meta['output_sha256'].items())
    split,manifests=pilot._load_session()
    sources=['model/roissd_mobilenet.py','tools/helpers/pipeline.py','tools/train_samplers/roi_mixed_sampling.py',
             'experiments/cpu_pilot/crop_domain.py','experiments/cpu_pilot/oracle_diagnosis.py',
             'dataset/transforms/ssd_transform.py','tools/infer.py']
    protected={str(p.relative_to(pilot.ROOT)):pilot._sha256(p) for directory in (pilot.PILOT_DIR,stage3a.OUT)
               for p in directory.rglob('*') if p.is_file()}
    meta={'status':'RUNNING','started_at':datetime.now(timezone.utc).isoformat(),
          'checkpoint':old_meta['checkpoint'],'config':old_meta['config'],
          'pilot_manifest_sha256':old_meta['pilot_manifest_sha256'],
          'stage3a_metadata_sha256':pilot._sha256(stage3a.OUT/'metadata.json'),
          'device':'cpu','threads':8,'paddings':PADDINGS,'confidence':.5,'nms_iou':.45,
          'seed':20261027,'predeclared_crop_first_primary_side':224,
          'preprocessing_variants':['CURRENT_PIPELINE','CROP_FIRST_MATCHED_HW','CROP_FIRST_224','CROP_FIRST_300'],
          'padding_fill':'0 in normalized tensor; ImageNet channel means before normalization',
          'source_sha256':{p:pilot._sha256(pilot.ROOT/p) for p in sources},
          'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write_json(OUT/'metadata.json',meta)
    assert pilot._sha256(pilot.ROOT/old_meta['checkpoint']['path'])==old_meta['checkpoint']['sha256']
    assert pilot._sha256(pilot.ROOT/old_meta['config']['config'])==old_meta['config']['config_sha256']
    model,dataset,config,threshold=pilot._load_model('roi_ssd_mobilenet',torch.device('cpu'))
    assert threshold==.5 and next(model.parameters()).device.type=='cpu'
    model.record_execution_trace=True
    state_digest=lambda:hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    before=state_digest()
    indices=pilot._dataset_indices(dataset,manifests['pilot'],pilot._dataset_root(config))
    names=dataset.idx2label
    name2id={v:k for k,v in names.items()}
    mode_preds={f'P{p}':[] for p in PADDINGS}
    mode_preds.update({m:[] for m in ('PADDED_P96','CROP_FIRST_MATCHED_HW','CROP_FIRST_224','CROP_FIRST_300')})
    truths,difficult=[],[]
    context,preprocessing,border,heads=[],[],[],[]
    expected={}
    with (stage3a.OUT/'frame_records.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['mode'].startswith('ORACLE_GT_SEG') and r['mode']!='ORACLE_GT_SEG_NO_GLOBAL_NMS':
                expected[(r['mode'],r['video_id'],r['frame_index'])]=r
    with HeadObserver(model) as observer,torch.inference_mode():
        for number,(batch,manifest) in enumerate(zip(pilot._dataloader(dataset,indices),manifests['pilot']),1):
            tensor,target,fname=batch
            image_path=Path(fname[0])
            image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
            h,w=image.shape[:2]
            gt,diff=pipeline.extract_gt_for_map(target,names,w,h)
            truths.append(gt);difficult.append(diff)
            frame_id=(manifest['video_id'],int(manifest['frame_index']))
            frame_dets={m:[] for m in mode_preds}
            per_frame_rows={m:[] for m in mode_preds}
            for padding in PADDINGS:
                rois=crop_rois(gt,w,h,padding)
                for (cls,j,box),roi in zip(stage3a.objects(gt),rois):
                    roi=pipeline.clip_bbox(roi,w,h)
                    crop,roi=pipeline.convert_crop_to_input_tensor(tensor,roi,(h,w),roi_grid=1)
                    identity={'video_id':frame_id[0],'frame_index':frame_id[1],
                              'object_id':f'{frame_id[0]}:{frame_id[1]}:{cls}:{j}',
                              'class_gt_index':j,'padding':padding,'variant':'CURRENT_PIPELINE'}
                    row,hr,dets=infer_target(model,observer,crop,roi,box,cls,name2id[cls],(h,w),identity)
                    for d in dets:d['class']=names[d['class_id']]
                    context.append(row);heads.extend(hr)
                    frame_dets[f'P{padding}'].extend(dets)
                    per_frame_rows[f'P{padding}'].append(row)
                    if padding!=96:continue
                    preprocessing.append(row)
                    for variant,shape in [('CROP_FIRST_MATCHED_HW',tuple(crop.shape[-2:])),
                                          ('CROP_FIRST_224',(224,224)),('CROP_FIRST_300',(300,300))]:
                        other=crop_first(image,roi,shape)
                        r,_,ds=infer_target(model,observer,other,roi,box,cls,name2id[cls],(h,w),dict(identity,variant=variant))
                        for d in ds:d['class']=names[d['class_id']]
                        r['tensor_mean_abs_delta_vs_current']=float((other-crop).abs().mean()) if other.shape==crop.shape else None
                        preprocessing.append(r);frame_dets[variant].extend(ds);per_frame_rows[variant].append(r)
                    outside=box[0]-96*w/320<0 or box[1]-96*h/320<0 or box[2]+96*w/320>w or box[3]+96*h/320>h
                    if outside:
                        padded,nominal=padded_tensor(tensor,box,96,(h,w))
                        br,_,bd=infer_target(model,observer,padded,nominal,box,cls,name2id[cls],(h,w),dict(identity,variant='PADDED_OUTSIDE_FRAME'))
                        for d in bd:d['class']=names[d['class_id']]
                        border.extend([dict(row,variant='CLIPPED'),br])
                        frame_dets['PADDED_P96'].extend(bd);per_frame_rows['PADDED_P96'].append(br)
                    else:
                        frame_dets['PADDED_P96'].extend(dets);per_frame_rows['PADDED_P96'].append(dict(row))
            for mode,dets in frame_dets.items():
                final=stage3a.postprocess(dets,.5,w,h)
                mode_preds[mode].append(pilot._map_prediction(final))
                matched=unique_matches(final,gt)
                for row in per_frame_rows[mode]:
                    row['TP50_after_global_nms']=(row['class'],row['class_gt_index']) in matched
                    own_global=matched.get((row['class'],row['class_gt_index']))
                    row['frame_matched_origin_object_id']=own_global['origin_object_id'] if own_global else None
                    row['frame_matched_prediction_head']=own_global['head'] if own_global else None
                if mode.startswith('P') and mode!='PADDED_P96' and mode!='P160':
                    p=int(mode[1:]);key='ORACLE_GT_SEG' if p==96 else f'ORACLE_GT_SEG_P{p}'
                    reference=expected[(key,*frame_id)]
                    stripped=[{k:d[k] for k in ('bbox','class','confidence')} for d in final]
                    assert stripped==reference['detections'],('Stage3a replay differs',frame_id,p)
            if number%20==0:
                print(f'Stage 3b {number}/320 frames; {len(context)} object/context pairs',flush=True)
    write_csv('per_object_context_trajectories.csv',context)
    write_csv('clipped_vs_padded.csv',border)
    write_csv('preprocessing_comparison.csv',preprocessing)
    write_csv('head_anchor_analysis.csv',heads)
    metrics=[]
    for mode,preds in mode_preds.items():
        result={'variant':mode,**detection_metrics(preds,truths,difficult)}
        selected=[r for r in context if mode==f'P{r["padding"]}']
        if selected:
            result.update(padding=selected[0]['padding'],N_GT=len(selected),
                          Recall50_own_crop_micro=float(np.mean([r['TP50_own_crop'] for r in selected])))
            for source,dest in [('gt_area_ratio','mean_gt_area_ratio'),('best_anchor_iou','mean_best_anchor_iou'),
                                ('max_class_correct_confidence','mean_confidence'),('best_prediction_iou','mean_pred_iou')]:
                result[dest]=float(np.mean([r[source] for r in selected]))
        metrics.append(result)
    write_csv('context_sweep_summary.csv',metrics)
    write_json(OUT/'predictions.json',{'predictions':mode_preds,'ground_truth':truths,'difficult':difficult})
    assert state_digest()==before
    assert all(pilot._sha256(pilot.ROOT/p)==digest for p,digest in protected.items())
    assert all(pilot._sha256(pilot.ROOT/p)==digest for p,digest in meta['source_sha256'].items())
    meta.update(status='INFERENCE_COMPLETE',inference_completed_at=datetime.now(timezone.utc).isoformat(),
                num_frames=320,num_videos=32,num_objects=560,num_context_pairs=len(context),
                num_border_objects=len(border)//2,num_preprocessing_pairs=len(preprocessing),
                stage3a_paddings_replayed=[32,64,96,128],weights_unchanged=True,
                production_and_prior_artifacts_unchanged=True)
    write_json(OUT/'metadata.json',meta)
    print('INFERENCE_COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
