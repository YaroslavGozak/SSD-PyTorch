"""Real detector execution, fixed timing boundary and resumable timed trials."""
import importlib.metadata
import os
import platform
import random
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .common import MODELS, ROOT, asset, atomic, checkpoint, code_identity, digest, git_identity, read, sha
from .workload import effective


class Store:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        path = self.directory/'identity.json'
        if path.exists() and read(path) != identity:
            raise ValueError(f'Resume identity mismatch: {path}; use a new run directory')
        if not path.exists():
            atomic(path,identity)

    def path(self, namespace, key):
        return self.directory/namespace/(digest(key)+'.json')

    def get(self, namespace, key):
        path = self.path(namespace,key)
        return read(path) if path.exists() else None

    def put(self, namespace, key, value):
        atomic(self.path(namespace,key),value)


def versions():
    result = {}
    for name in ('torch','torchvision','ultralytics','numpy','Pillow','opencv-python','threadpoolctl'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class Runner:
    def __init__(self,cfg,model,checkpoint_root=None,threads=None,affinity=None):
        import cv2
        import psutil
        import torch
        from experiments.cpu_pilot.portable_paths import cpu_model_fallback
        from experiments.cpu_pilot.runtime_threads import install_thread_policy
        from experiments.cost_model_validation.common import read_power_mode
        from experiments.cost_model_validation.adapters import PreparedInput

        self.cfg,self.model = cfg,model
        self.weight_path = checkpoint(cfg,model,checkpoint_root)
        runtime = dict(cfg['runtime'])
        for key,env in (('cpu_threads','SSD_CPU_THREADS'),('interop_threads','SSD_INTEROP_THREADS'),('opencv_threads','SSD_OPENCV_THREADS')):
            runtime[key] = int(os.environ.get(env,runtime[key]))
        runtime.update({k:v for k,v in (threads or {}).items() if v is not None})
        affinity = affinity if affinity is not None else runtime['affinity']
        if affinity is not None:
            psutil.Process().cpu_affinity(affinity)
        self.facade = SimpleNamespace(run_model_inference=lambda prepared:self.adapter.infer(prepared),
            _runtime_environment=lambda:{'cpu_model':cpu_model_fallback(), 'host':socket.gethostname(),
                'os':platform.platform(),'python':platform.python_version(),'architecture':platform.machine(),
                'affinity':psutil.Process().cpu_affinity(), 'physical_cores':psutil.cpu_count(logical=False),
                'intra_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
                'opencv_threads':cv2.getNumThreads(),'versions':versions(),'power':read_power_mode()})
        install_thread_policy(self.facade,SimpleNamespace(**runtime))
        self.facade._configure_runtime()
        self.grid = cfg['models'][model]['shape_grid']
        if model == 'yolo26n':
            if versions()['ultralytics'] != '8.4.60':
                raise ValueError('Frozen YOLO implementation requires ultralytics==8.4.60')
            from experiments.cost_model_validation.ultralytics_adapter import UltralyticsAdapter
            self.adapter = UltralyticsAdapter(str(self.weight_path),'cpu',32)
        else:
            from experiments.cost_model_validation.roissd_adapter import RoiSSDAdapter
            import tempfile
            import yaml
            architecture = cfg['models'][model]['architecture']
            with tempfile.TemporaryDirectory(prefix='merge-b-model-') as temp:
                path = Path(temp)/'model.yaml'
                path.write_text(yaml.safe_dump(architecture))
                self.adapter = RoiSSDAdapter(str(path),str(self.weight_path),'cpu',1,pretrained_backbone=False)
            self.adapter.model.record_execution_trace = True
        self.facade._configure_runtime()
        self.input_shape = None
        self.forward_calls = 0
        network = self.adapter.model.model if model == 'yolo26n' else self.adapter.model
        self.hook = network.register_forward_pre_hook(self.observe)
        self.PreparedInput = PreparedInput
        self.runtime = self.facade._runtime_environment()
        self.identity = {'experiment_id':cfg['experiment_id'],'seed':cfg['seed'],
            'hardware_id':self.runtime['host']+'-'+digest(self.runtime)[:12], 'detector':model,'checkpoint_sha256':sha(self.weight_path),'config_sha256':digest(cfg),
            'code_sha256':code_identity(),'runtime':self.runtime,'shape_rule':f'pipeline_crop_grid_{self.grid}',
            'timing_scope':cfg['profiling']['timing_scope'], 'git':git_identity()}

    def observe(self,module,args):
        self.forward_calls += 1
        self.input_shape = tuple(map(int,args[0].shape[-2:]))

    def prepared(self,tensor):
        h,w = map(int,tensor.shape[-2:])
        return self.PreparedInput(tensor,h,w,h,w,h,w)

    def synthetic(self,shape,seed):
        rng = np.random.default_rng(seed)
        image = rng.integers(0,256,(*shape,3),dtype=np.uint8)
        return self.adapter.prepare(image,shape)

    def base_tensor(self,image):
        import torch
        from torchvision import tv_tensors
        from PIL import Image
        from dataset.transforms.ssd_transform import SsdTransform
        with Image.open(image) as img:
            array = np.array(img.convert('RGB'),copy=True)
        tensor = tv_tensors.Image(torch.from_numpy(array).permute(2,0,1))
        hw = self.cfg['geometry']['canvas_hw']
        if hw[0] != hw[1]:
            raise ValueError('The reused SSD transform requires a square base canvas')
        transform = SsdTransform(hw[0],(123,117,104),(0.485,0.456,0.406),(0.229,0.224,0.225)).transforms['test']
        tensor = transform(tensor)
        if self.model == 'yolo26n':
            mean=tensor.new_tensor([0.485,0.456,0.406])[:,None,None]
            std=tensor.new_tensor([0.229,0.224,0.225])[:,None,None]
            tensor=(tensor*std+mean).clamp(0,1)
        return tensor

    def crop(self,tensor,box,row):
        from tools.helpers.pipeline import convert_crop_to_input_tensor
        crop,snapped = convert_crop_to_input_tensor(tensor,box,(row['height'],row['width']),roi_grid=self.grid)
        expected,_ = effective(tuple(box),(row['height'],row['width']),tuple(self.cfg['geometry']['canvas_hw']),self.grid)
        if tuple(crop.shape[-2:]) != expected:
            raise ValueError('Runtime crop differs from frozen geometry')
        return self.prepared(crop.unsqueeze(0)),snapped

    def infer(self,prepared):
        self.input_shape = None
        self.forward_calls = 0
        result = self.facade.run_model_inference(prepared)
        if self.forward_calls != 1 or self.input_shape != prepared.tensor_hw:
            raise ValueError(f'Observed network shape {self.input_shape} != effective {prepared.tensor_hw}')
        return result

    def execution_trace(self):
        return None if self.model == "yolo26n" else self.adapter.model.last_execution_trace

    def elapsed(self,prepared):
        started = time.perf_counter_ns()
        result = self.infer(prepared)
        value = (time.perf_counter_ns()-started)/1e6
        del result
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Invalid measured inference time')
        return value

    def smoke(self,class_names):
        if self.model == 'yolo26n':
            names=self.adapter.model.names
            if isinstance(names,dict):
                names=[names[i] for i in sorted(names)]
            if names != class_names:
                raise ValueError('Checkpoint/dataset category mapping mismatch')
        elif len(class_names)+1 != self.cfg['models'][self.model]['architecture']['dataset_params']['num_classes']:
            raise ValueError('SSD foreground/background class count mismatch')
        self.infer(self.synthetic(tuple(self.cfg['geometry']['canvas_hw']),self.cfg['seed']))
        trace = None if self.model == 'yolo26n' else self.adapter.model.last_execution_trace
        if trace is not None and trace.get('depth_policy') != 'until_1x1':
            raise ValueError('Unexpected SSD active-depth policy')
        return {'architecture':type(self.adapter.model).__name__,'class_names':class_names,
                'effective_hw':self.input_shape,'execution_trace':trace,'preprocessing':self.adapter.preprocessing_metadata()}


def sensor():
    from experiments.cost_model_validation.common import read_cpu_temp,read_cpu_freq
    return {'temperature_c':read_cpu_temp(),'cpu_frequency_mhz':read_cpu_freq(),
            'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'process_id':os.getpid()}


def sample_shapes(runner,shapes,store,namespace,repetitions,seed):
    """Balanced randomized repetition blocks; resume atomically per shape/trial."""
    shapes = sorted(set(tuple(s) for s in shapes))
    rng=random.Random(seed)
    # Bound RAM: prepare/warm only the current shape; never cache a dense tensor grid.
    records={shape:store.get(namespace,list(shape)) or {'shape':list(shape),'trials':[]} for shape in shapes}
    for block in range(repetitions):
        order=list(shapes); rng.shuffle(order)
        for i,shape in enumerate(order):
            record=records[shape]
            if len(record['trials'])>block:
                continue
            value=runner.synthetic(shape,seed)
            for _ in range(runner.cfg['profiling']['warmup']):
                runner.infer(value)
            before=sensor()
            elapsed=runner.elapsed(value)
            record['trials'].append({'block':block,'ms':elapsed,'before':before,'after':sensor(),
                                     'observed_hw':list(runner.input_shape),'execution_trace':runner.execution_trace()})
            store.put(namespace,list(shape),record)
            if i%25==0:
                print(f'{namespace}: block {block+1}/{repetitions}, shape {i+1}/{len(shapes)}',flush=True)
    return {f'{h}x{w}':[trial['ms'] for trial in records[(h,w)]['trials']] for h,w in shapes}
