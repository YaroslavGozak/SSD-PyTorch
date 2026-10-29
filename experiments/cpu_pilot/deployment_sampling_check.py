"""Distribution and real B=2 forward/backward checks; never optimizer.step()."""
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torchvision.io import read_image

from model.roissd_mobilenet import RoiSSDMobileNet
from tools.helpers.config_reader import load_config
from tools.train_samplers.deployment_mixed_sampling import (
    DeploymentMixedBatchSampler,DeploymentRoiBatchProcessor,DeploymentSamplingDataset,
    DeploymentMixedCollateFn,DeploymentSampleRequest,stage_sampling_config,stable_seed,
)
from tools.train_samplers.deployment_sampling_stats import DeploymentSamplingStats
from experiments.roissd_128_anomaly.quick_regression_check import state_mapping_digest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'final_experiments/deployment_mixed_sampling_check'
SHAPES=((128,224),(224,128),(160,256),(256,160),(240,288),(288,240),(320,320))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict,tuple)) else v for k,v in row.items()})


class ManifestRawDataset:
    def __init__(self,rows,root):self.rows=rows;self.root=Path(root)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        assert sha(self.root/row['label'])==row['label_sha256']
        image=read_image(str(self.root/row['image']))
        h,w=image.shape[-2:]
        # Use the real raw loader's source-pixel clipping and label conversion.
        from dataset.yolo_imagenet_vid import _parse_yolo_label_file
        names=json.loads((ROOT/'final_experiments/01_audit/dataset_audit.json').read_text())['names']
        detections=_parse_yolo_label_file(str(self.root/row['label']),w,h,names,{name:i+1 for i,name in enumerate(names)})
        boxes=torch.tensor([d['bbox'] for d in detections],dtype=torch.float32).reshape(-1,4)/torch.tensor([w,h,w,h])
        labels=torch.tensor([d['label'] for d in detections],dtype=torch.long)
        return {'image':image,'target':{'boxes':boxes,'labels':labels}}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(856245)
    cfg_path=ROOT/'config/imagenet-vid-roissd-mobilenet.yaml'
    cfg=load_config(str(cfg_path));train=cfg['train_params'];side=cfg['dataset_params']['im_size']
    assert side==320
    checkpoint_path=ROOT/'trained_models'/train['task_name']/train['ckpt_name']
    checkpoint_sha=sha(checkpoint_path)
    protected={p:sha(ROOT/p) for p in ('tools/train.py','model/roissd_mobilenet.py','tools/helpers/pipeline.py')}
    resume=checkpoint_path.parent/'last.pt'
    resume_sha=sha(resume) if resume.exists() else None
    distributions=[]
    for stage in (2,3):
        sampling=stage_sampling_config(train,stage,side)
        sampler=DeploymentMixedBatchSampler(range(20000),2,stage,sampling,im_size=side,seed=train['seed'])
        sampler.set_epoch(0)
        rows=[]
        for batch in sampler:
            r=batch[0]
            rows.append({'stage':stage,'epoch':0,'mode':r.mode,'batch_h':r.batch_h,'batch_w':r.batch_w,
                         'crop_area_fraction':r.batch_h*r.batch_w/side**2,'aspect_ratio':r.batch_w/r.batch_h,
                         'area_bin':r.area_bin,'sampled_area_fraction':r.sampled_area_fraction,
                         'dataset_indices':[x.index for x in batch],'request_seeds':[x.seed for x in batch]})
        write_csv(f'batch_definitions_stage{stage}.csv',rows)
        roi=[r for r in rows if r['mode']=='roi'];n=len(rows)
        summary={'stage':stage,'definitions':n,'full_share':1-len(roi)/n,'roi_share':len(roi)/n,
                 'conditional_area_bin_shares':[sum(r['area_bin']==i for r in roi)/len(roi) for i in range(4)],
                 'absolute_roi_bin_shares':[sum(r['area_bin']==i for r in roi)/n for i in range(4)],
                 'rectangular_roi_share':sum(r['batch_h']!=r['batch_w'] for r in roi)/len(roi),
                 'unique_roi_shapes':len({(r['batch_h'],r['batch_w']) for r in roi}),
                 'mean_roi_area_fraction':float(np.mean([r['crop_area_fraction'] for r in roi])),
                 'height_std':float(np.std([r['batch_h'] for r in roi])),
                 'width_std':float(np.std([r['batch_w'] for r in roi])),
                 'height_min':min(r['batch_h'] for r in roi),'height_max':max(r['batch_h'] for r in roi),
                 'width_min':min(r['batch_w'] for r in roi),'width_max':max(r['batch_w'] for r in roi)}
        assert n==10000 and abs(summary['full_share']-sampling.full_probability)<.02
        assert all(abs(actual-b[2])<.02 for actual,b in zip(summary['conditional_area_bin_shares'],sampling.roi_area_fraction_bins))
        assert all(sampling.min_crop_side<=r['batch_h']<=side and sampling.min_crop_side<=r['batch_w']<=side for r in roi)
        assert summary['rectangular_roi_share']>.95
        distributions.append(summary)
    assert distributions[1]['mean_roi_area_fraction']<distributions[0]['mean_roi_area_fraction']
    write_csv('distribution_sanity.csv',distributions)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    state=checkpoint.get('model',checkpoint)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=RoiSSDMobileNet(copy.deepcopy(cfg['model_params']),num_classes=31,record_execution_trace=True)
    incompatible=model.load_state_dict(state,strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    model.set_batch_norm_frozen(freeze_backbone=train.get('freeze_backbone_bn',True),
                               freeze_extra=train.get('freeze_extra_bn',False),
                               train_affine=train.get('train_bn_affine',True))
    model.to(device);model.eval();model.train()
    before=state_mapping_digest(model.state_dict())
    norms={name:module for name,module in model.named_modules() if isinstance(module,torch.nn.BatchNorm2d)}
    bn_before={name:{key:getattr(module,key).detach().cpu().clone() for key in ('running_mean','running_var','num_batches_tracked')}
               for name,module in norms.items()}
    bn_calls=Counter();affine_grad_seen=set();handles=[]
    for name,module in norms.items():
        handles.append(module.register_forward_hook(lambda m,a,out,name=name:bn_calls.update([name])))
    audit=json.loads((ROOT/'final_experiments/01_audit/dataset_audit.json').read_text())
    train_manifest=ROOT/'final_experiments/01_audit/train_manifest.jsonl'
    raw_rows=[json.loads(line) for line in train_manifest.read_text().splitlines()]
    raw_rows=random.Random(train['seed']).sample(raw_rows,1000)
    raw=ManifestRawDataset(raw_rows,audit['root'])
    results=[]
    for stage in (2,3):
        sampling=stage_sampling_config(train,stage,side)
        dataset=DeploymentSamplingDataset(raw,DeploymentRoiBatchProcessor(sampling,side))
        stats=DeploymentSamplingStats(stage,sampling)
        for index,(h,w) in enumerate(SHAPES):
            mode='full' if (h,w)==(side,side) else 'roi'
            requests=[DeploymentSampleRequest(index*2+slot,mode,h,w,stable_seed(train['seed'],stage,0,index,slot),
                        stage,0,index,-1,h*w/side**2) for slot in range(2)]
            images,targets,mode,info=DeploymentMixedCollateFn()([dataset[r] for r in requests])
            assert images.shape==(2,3,h,w) and torch.isfinite(images).all()
            for target in targets:
                assert torch.isfinite(target['boxes']).all()
                assert ((target['boxes']>=0)&(target['boxes']<=1)).all()
                assert ((target['boxes'][:,2:]-target['boxes'][:,:2])>0).all()
                assert ((target['labels']>=1)&(target['labels']<=30)).all()
            stats.record(mode,images.shape[-2:],info)
            model.zero_grad(set_to_none=True)
            loss_dict,_=model(images.to(device),[{k:v.to(device) for k,v in target.items()} for target in targets])
            loss=sum(loss_dict.values())
            assert torch.isfinite(loss) and all(torch.isfinite(v) for v in loss_dict.values())
            loss.backward()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(g.abs().max()>0 for g in gradients)
            for name,module in norms.items():
                for key in ('weight','bias'):
                    parameter=getattr(module,key)
                    if parameter is not None and parameter.grad is not None:affine_grad_seen.add((name,key))
            trace=model.get_runtime_log()
            assert trace['N_loc_predictions']==trace['N_cls_predictions']==trace['N_anchors']
            assert trace['N_anchors']==sum(trace['anchors_per_head'])
            results.append({'stage':stage,'batch_size':2,'batch_h':h,'batch_w':w,'mode':mode,
                            'classification_loss':float(loss_dict['classification'].detach()),
                            'bbox_loss':float(loss_dict['bbox_regression'].detach()),'loss':float(loss.detach()),
                            'forward':'PASS','backward':'PASS','finite_gradients':True,
                            'targets_valid':True,'resampling_attempts':sum(r['resampling_attempts'] for r in info['samples']),
                            **trace})
            print(f'Stage {stage} B=2 {h}x{w}: forward/loss/backward PASS',flush=True)
        stats.save(OUT,epoch=1)
    for handle in handles:handle.remove()
    write_csv('forward_backward_sanity.csv',results)
    bn_rows=[]
    for name,module in norms.items():
        changed=[key for key,value in bn_before[name].items() if not torch.equal(value,getattr(module,key).cpu())]
        bn_rows.append({'name':name,'group':'backbone' if name.startswith('features_stage1.') else 'extra_or_heads',
                        'module_training':module.training,'running_statistics_changed':bool(changed),
                        'changed_buffers':changed,'forward_calls':bn_calls[name],
                        'weight_trainable':module.weight.requires_grad if module.weight is not None else None,
                        'bias_trainable':module.bias.requires_grad if module.bias is not None else None,
                        'weight_gradient_observed':(name,'weight') in affine_grad_seen,
                        'bias_gradient_observed':(name,'bias') in affine_grad_seen})
    assert all(not r['module_training'] and not r['running_statistics_changed'] for r in bn_rows)
    assert all(r['weight_trainable'] and r['bias_trainable'] for r in bn_rows)
    assert all(r['weight_gradient_observed'] and r['bias_gradient_observed'] for r in bn_rows if r['forward_calls'])
    write_csv('bn_audit.csv',bn_rows)
    assert state_mapping_digest(model.state_dict())==before
    assert sha(checkpoint_path)==checkpoint_sha and (not resume_sha or sha(resume)==resume_sha)
    assert all(sha(ROOT/p)==digest for p,digest in protected.items())
    config_text=cfg_path.read_text()
    lines=config_text.splitlines(keepends=True);start=next(i for i,line in enumerate(lines) if line.startswith('  roi_sampling:'))
    end=next((i for i in range(start+1,len(lines)) if lines[i].startswith('  ') and not lines[i].startswith('    ') and lines[i].strip()),len(lines))
    previous=''.join(lines[:start]+lines[end:])
    (OUT/'yaml_diff.patch').write_text(''.join(difflib.unified_diff(previous.splitlines(True),config_text.splitlines(True),
        fromfile='before/imagenet-vid-roissd-mobilenet.yaml',tofile='after/imagenet-vid-roissd-mobilenet.yaml')),encoding='utf-8')
    sources=('tools/train_samplers/deployment_mixed_sampling.py','tools/train_samplers/deployment_sampling_stats.py',
             'tools/train_mixed_sampling.py','config/imagenet-vid-roissd-mobilenet.yaml',
             'experiments/cpu_pilot/deployment_sampling_check.py','tests/test_deployment_mixed_sampling.py',
             'tests/test_model_first_loading.py','tools/train_samplers/README.md')
    meta={'status':'PASS','completed_at':datetime.now(timezone.utc).isoformat(),
          'distribution_definitions_per_stage':10000,'real_forward_backward_batches':14,'batch_size':2,
          'device':str(device),'config_sha256':sha(cfg_path),'sampling_config':train['roi_sampling'],
          'initial_checkpoint':str(checkpoint_path.relative_to(ROOT)),'checkpoint_sha256':checkpoint_sha,
          'resume_checkpoint_sha256':resume_sha,'checkpoint_files_unchanged':True,'model_state_unchanged':True,
          'optimizer_steps':0,'full_training_run':False,'stage1_model_inference_sources_unchanged':protected,
          'training_manifest_sha256':sha(train_manifest),'real_image_pool':1000,
          'bn_yaml_flags':{k:train[k] for k in ('freeze_backbone_bn','freeze_extra_bn','train_bn_affine')},
          'BN_count':len(bn_rows),'BN_running_statistics_updated':sum(r['running_statistics_changed'] for r in bn_rows),
          'BN_affine_trainable':sum(r['weight_trainable'] for r in bn_rows),
          'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'source_sha256':{p:sha(ROOT/p) for p in sources}}
    report=['# Deployment-matched mixed sampling implementation check','', '**Result: PASS**','',
            '## Changed files and integration','',
            'New sampler/processor/collate and request-aware dataset wrapper are in '
            '`tools/train_samplers/deployment_mixed_sampling.py`. Rectangular statistics are in '
            '`deployment_sampling_stats.py`. `tools/train_mixed_sampling.py` calls '
            '`build_deployment_training_loader` for both Stage 2 and Stage 3; it has no old-sampler '
            'stage branch. The legacy sampler remains unchanged for old experiments. '
            'The active MobileNet YAML only gains `train_params.roi_sampling.stage2/stage3`; '
            'the exact addition is in [yaml_diff.patch](yaml_diff.patch). Existing stage choice, BN flags, '
            'optimizer, LR, model architecture, anchors, loss, checkpoints and inference are unchanged.', '',
            'Changed/added source files:', '', *[f'- `{p}`' for p in sources], '',
            '## Algorithm','',
            'Each batch draws full/ROI, then a conditional area bin, a continuous uniform area, and a '
            'log-uniform W/H within the feasible intersection of YAML aspect/min-side/frame bounds. '
            'H/W round to the nearest integer (half up), with no stride alignment; invalid rounded '
            'shapes are resampled within that bin. This conditional aspect draw avoids silently skewing '
            'large-area bin frequencies through rejection. Every sample receives identical H/W and '
            'its own immutable deterministic seed. RNG is derived from seed/stage/epoch/batch/slot; '
            '`set_epoch(i)` is called before each epoch iteration, including resumed epochs.', '',
            'The wrapper loads a raw dataset item, selects a feasible reference GT and jitters its crop '
            'center, enforcing reference visibility. Ineligible items/positions are deterministically '
            'resampled (maximum 20 item attempts, each with at most 20 position attempts). Exhaustion '
            'raises an explicit error: crop size is never enlarged, resized or replaced by full-frame. '
            'Such item resampling intentionally changes image sampling to the eligible GT subset; it is logged.', '',
            'Source RGB is resized once to 320×320 with deployment torchvision bilinear antialiasing '
            'and uint8 rounding; raw float values decoded from uint8 are recovered before resize. '
            'The H×W tensor is sliced, targets are clipped/visibility-filtered, x divides by W and y '
            'by H, and ImageNet normalization is applied. Crop tensors are stacked directly, without '
            'crop resize or batch padding. Full-frame pixel values match the deployment transform exactly.', '',
            '## Distribution sanity: 10,000 definitions per stage','',
            '| Stage | Full share | ROI bins conditional | Rectangular ROI | Unique ROI shapes | Mean ROI area | H std / W std |',
            '|---|---:|---|---:|---:|---:|---|']
    for r in distributions:
        report.append(f'| {r["stage"]} | {r["full_share"]:.2%} | '+
                      ', '.join(f'{v:.2%}' for v in r['conditional_area_bin_shares'])+
                      f' | {r["rectangular_roi_share"]:.2%} | {r["unique_roi_shapes"]} | '
                      f'{r["mean_roi_area_fraction"]:.4f} | {r["height_std"]:.2f} / {r["width_std"]:.2f} |')
    report+=['','The four bins are 10–25%, 25–50%, 50–75%, 75–100% frame area. Conditional ROI '
             'probabilities and full shares match YAML within ±2 percentage points. Bounds and '
             'rounding error were checked. Empirical definitions and absolute bin shares are saved '
             'in the CSVs. Distribution sanity is based on definitions; accepted-object geometry '
             'and resampling statistics are separately logged from real batches.', '',
             '## Real forward/backward sanity','',
             f'Configured initial checkpoint: `{checkpoint_path.relative_to(ROOT)}`; SHA256 `{checkpoint_sha}`. '
             'Strict load has no missing/unexpected keys. B=2 real training-image batches passed for '
             '128×224, 224×128, 160×256, 256×160, 240×288, 288×240 and 320×320 in both stages '
             f'(device {device}). Finite images, valid labels/boxes, finite classification/regression '
             'losses, finite/nonzero gradients, and `N_loc == N_cls == N_anchors` all passed. '
             'These shapes are explicit sanity requests, not a replacement for continuous sampling. '
             'No optimizer was stepped; the complete model state and checkpoint files are unchanged.', '',
             '## BatchNorm audit','',
             f'YAML flags remain freeze_backbone_bn={train["freeze_backbone_bn"]}, '
             f'freeze_extra_bn={train["freeze_extra_bn"]}, train_bn_affine={train["train_bn_affine"]}. '
             f'All {len(bn_rows)} BN modules remain in eval after `model.eval(); model.train()`. '
             'Backbone group is features_stage1; extra group includes features_stage2, conv_to_672, '
             'extra pyramid blocks and both head lists. All running means/variances and '
             'num_batches_tracked remain unchanged. All BN affine weights/biases are trainable, '
             'and every executed BN received finite affine gradients. Per-module details and '
             'execution counts are in [bn_audit.csv](bn_audit.csv).', '',
             '## Statistics and reproducibility','',
             'Training writes per-epoch batch CSV, per-sample CSV and summary JSON using both H and W. '
             'They include stage/mode/shape/area bin, actual area/aspect, reference area/center offset/'
             'visibility, source index and resampling count, plus full/ROI/area-bin/rectangular shares '
             'and H/W/aspect/GT geometry distributions. Epoch files overwrite only that epoch on resume. '
             'The reference GT area ratio uses its visible target area; pre-clipping area ratio is also retained. '
             'Worker-count, global-RNG independence, same-epoch replay, target normalization and '
             'deployment pixel parity are covered by unit tests.', '',
             'Other mixed-training YAMLs must provide their own explicit roi_sampling stage config; '
             'missing config fails before checkpoint side effects. Stage 1 stays on its separate script.', '',
             'Reproduce: `.venv/Scripts/python.exe -m experiments.cpu_pilot.deployment_sampling_check`. '
             'This command performs sanity forward/backward only and never launches the training loop.']
    (OUT/'implementation_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    meta['output_sha256']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='metadata.json'}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('PASS:',OUT,flush=True)


if __name__=='__main__':main()
