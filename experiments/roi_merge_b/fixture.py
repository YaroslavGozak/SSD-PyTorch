"""Explicit nonpublication fixtures. No real detector is substituted silently."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

from .common import MODELS,ROOT,asset,atomic,code_identity,digest,git_identity,sha
from .workload import build_rows,write_workload
from .execution import Runner


def create_fixture(output,cfg):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Fixture requires an empty directory')
    root=output/'data'
    source=[]
    patterns=[[],[(.15,.2,.15,.2)],[(.15,.2,.15,.2),(.8,.75,.15,.2)],
              [(.3,.3,.3,.3),(.4,.4,.25,.3)],[(.2,.5,.1,.1),(.5,.5,.1,.1),(.8,.5,.1,.1)],
              [(.2,.2,.15,.15),(.8,.8,.15,.15),(.2,.8,.15,.15)]]
    for video in range(8):
        for frame in range(2):
            image=f'images/val/fixture_{video:02d}_{frame:03d}.png'
            label=f'labels/val/fixture_{video:02d}_{frame:03d}.txt'
            img=asset(root,image);lab=asset(root,label)
            img.parent.mkdir(parents=True,exist_ok=True);lab.parent.mkdir(parents=True,exist_ok=True)
            pixels=np.random.default_rng(video*10+frame).integers(0,256,(176,288,3),dtype=np.uint8)
            Image.fromarray(pixels).save(img)
            gt=[[i%30,*box] for i,box in enumerate(patterns[(video+frame)%len(patterns)])]
            lab.write_text(''.join(' '.join(map(str,box))+'\n' for box in gt),encoding='utf-8')
            source.append({'video_id':f'fixture_{video:02d}','frame_index':frame,'image':image,'label':label,'gt':gt,'label_sha256':sha(lab)})
    test_cfg=copy.deepcopy(cfg)
    test_cfg['experiment_id']='SYNTHETIC-NONPUBLICATION-merge-b'
    test_cfg['splits']={'calibration_sequences':2,'validation_sequences':2}
    test_cfg['profiling'].update(warmup=1,repetitions=3,evaluation_repetitions=4,grid_sides=[32,96,192,320],calibration_max_shapes=20)
    test_cfg['policies']['bootstrap_repetitions']=30
    names=[f'fixture_class_{i:02d}' for i in range(30)]
    (root/'imagenet_vid.yaml').write_text(yaml.safe_dump({'names':names,'path':'.','val':'images/val'}))
    rows=build_rows(source,root,test_cfg)
    atomic(output/'source_index.jsonl',source,lines=True)
    write_workload(rows,test_cfg,output/'workload',names,sha(output/'source_index.jsonl'),fixture=True)
    (output/'config.yaml').write_text(yaml.safe_dump(test_cfg,sort_keys=False),encoding='utf-8')
    print(f'SYNTHETIC ONLY: config={output}/config.yaml; manifest={output}/workload/manifest.jsonl; dataset-root={root}')


class SyntheticRunner(Runner):
    def __init__(self,cfg,model):
        self.cfg,self.model=cfg,model
        self.grid=cfg['models'][model]['shape_grid']
        self.input_shape=None
        from experiments.cost_model_validation.adapters import PreparedInput
        self.PreparedInput=PreparedInput
        self.runtime={'host':'SYNTHETIC','versions':{'fixture':'1'}}
        self.identity={'experiment_id':cfg['experiment_id'],'seed':cfg['seed'],'hardware_id':'SYNTHETIC', 'detector':model,'checkpoint_sha256':'SYNTHETIC-NO-WEIGHTS',
            'config_sha256':digest(cfg),'code_sha256':code_identity(),'runtime':self.runtime,
            'shape_rule':f'pipeline_crop_grid_{self.grid}','timing_scope':'SYNTHETIC-NONPUBLICATION','git':git_identity()}

    def smoke(self,names):
        return {'fixture':True,'class_names':names}

    def synthetic(self,shape,seed):
        import torch
        return self.prepared(torch.zeros((1,3,*shape)))

    def infer(self,prepared):
        self.input_shape=prepared.tensor_hw
        return None

    def execution_trace(self):
        return {"fixture":True}

    def elapsed(self,prepared):
        self.infer(prepared)
        h,w=prepared.tensor_hw
        return .5+h*w*.00001+max(0,h*w-25000)*.000004
