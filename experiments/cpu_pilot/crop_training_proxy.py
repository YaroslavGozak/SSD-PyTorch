"""Sample historical mixed-training geometry; explicitly not exact training replay."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import torch
from torchvision.io import read_image
from .crop_domain import OUT, write_csv, MEAN, STD
from .oracle_diagnosis import geometry
from .run_pilot import ROOT, _sha256
from tools.benchmarks.video_artifacts import write_json

REV='03042cc32ad3daefdbc958a9283e78dbaa8ca797'
SEED=20261027


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','show',f'{REV}:tools/train_samplers/roi_mixed_sampling.py'])
    module=types.ModuleType('historical_roi_sampling_stage3b')
    sys.modules[module.__name__]=module
    exec(compile(source,f'{REV}:roi_mixed_sampling.py','exec'),module.__dict__)
    processor=module.RoiBatchProcessor(normalize_transform=lambda image:(image-MEAN)/STD)
    modes=module.build_stage_mode_configs(2)
    audit=json.loads((ROOT/'final_experiments/01_audit/dataset_audit.json').read_text())
    manifest=ROOT/'final_experiments/01_audit/train_manifest.jsonl'
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    rng=random.Random(SEED)
    selected=rng.sample(rows,1000)
    captures={}
    original_choose=module.choose_reference_box
    original_crop=module.make_roi_crop_box
    def choose(boxes):
        result=original_choose(boxes)
        captures['ref']=int(result)
        return result
    def make_crop(*args,**kwargs):
        result=original_crop(*args,**kwargs)
        captures['roi']=list(result)
        return result
    module.choose_reference_box=choose
    module.make_roi_crop_box=make_crop
    sample_rows=[]
    for i,row in enumerate(selected):
        mode=rng.choices(modes,weights=[m.prob for m in modes],k=1)[0]
        random.seed(SEED+i);torch.manual_seed(SEED+i)
        image_path=Path(audit['root'])/row['image']
        assert _sha256(Path(audit['root'])/row['label'])==row['label_sha256']
        image=read_image(str(image_path)).float()/255
        _,h,w=image.shape
        boxes=torch.tensor([[cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2] for _,cx,cy,bw,bh in row['gt']],dtype=torch.float32)
        labels=torch.tensor([int(g[0])+1 for g in row['gt']],dtype=torch.long)
        target={'boxes':boxes,'labels':labels}
        captures.clear()
        output,output_target=processor.process_sample(image,target,mode.name,mode.out_size)
        assert torch.isfinite(output).all()
        ref=captures.get('ref',rng.randrange(len(boxes)))
        roi=captures.get('roi',[0,0,w,h])
        box=(boxes[ref]*torch.tensor([w,h,w,h])).tolist()
        # Geometry seen in training targets after visibility clipping.
        box=[max(roi[0],box[0]),max(roi[1],box[1]),min(roi[2],box[2]),min(roi[3],box[3])]
        geo=geometry(box,roi)
        sample_rows.append({'sample_id':i,'image':row['image'],'image_sha256':_sha256(image_path),
                            'label_sha256':row['label_sha256'],'mode':mode.name,
                            'reference_index':ref,'source_h':h,'source_w':w,'roi':roi,
                            'network_h':output.shape[-2],'network_w':output.shape[-1],
                            'network_aspect_ratio':output.shape[-1]/output.shape[-2],
                            'source_crop_aspect_ratio':geo['crop_w']/geo['crop_h'],
                            'N_retained_GT':len(output_target['boxes']),**geo,
                            'provenance':'RECONSTRUCTED_HISTORICAL_PROXY_NOT_OBSERVED_TRAINING_CROPS'})
        if (i+1)%100==0:print(f'Training proxy {i+1}/1000',flush=True)
    write_csv('training_crop_sample.csv',sample_rows)
    config_path=ROOT/'trained_models/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/mixed_sampling_eval.yaml'
    checkpoint_path=ROOT/'trained_models/best-imagenet-vid-roissd-mobilenet/20260928-phase2-bn-freeze/best_map.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    evidence={
        'status':'RECONSTRUCTED_PROXY','exact_training_revision_known':False,
        'limitation':'Checkpoint has no training source hash, launch command or saved crop manifest. '
                     'Historical source is contemporaneous evidence, not proof of the exact training working tree.',
        'checkpoint_stage':checkpoint.get('stage'),'checkpoint_epoch_zero_based':checkpoint.get('epoch'),
        'checkpoint_best_map':checkpoint.get('best_map'),
        'checkpoint_sha256':_sha256(checkpoint_path),'saved_eval_config_sha256':_sha256(config_path),
        'historical_revision':REV,'historical_sampler_sha256':hashlib.sha256(source).hexdigest(),
        'historical_files_sha256':{p:hashlib.sha256(subprocess.check_output(['git','show',f'{REV}:{p}'])).hexdigest()
            for p in ('tools/train_mixed_sampling.py','dataset/yolo_imagenet_vid.py')},
        'training_manifest_sha256':_sha256(manifest),'sample_seed':SEED,'N_sampled_crops':1000,
        'sampling':'1000 uniform distinct manifest frames; stage-2 mode weights; one reference GT/crop; no model training',
        'scheduled_modes':[asdict(m) for m in modes],
        'effective_processor_mode_map':{name:asdict(cfg) for name,cfg in processor.mode_cfg_map.items()},
        'historical_mode_collision':'Processor loops over stages 2 then 3 into one mode map; shared names use stage-3 parameters even for stage-2 sampled output sizes.',
        'historical_path':['read native RGB float CHW','random GT reference; bbox center/scale jitter',
                           'random source-pixel context; source ROI clipping','crop native image',
                           'bilinear resize independently to scheduled square (align_corners=False)',
                           'ImageNet normalize'],
        'historical_random_photometric_or_IoU_crops':False,
        'historical_object_centered':True,
        'historical_unscaled_slices_of_full_resized_tensor':False,
        'current_training_code_warning':'Current RoiBatchProcessor was changed in October; it is not used as September-training evidence.',
    }
    write_json(OUT/'training_preprocessing_audit.json',evidence)
    print('Historical proxy COMPLETE; exact training revision remains unverified',flush=True)


if __name__=='__main__':main()
