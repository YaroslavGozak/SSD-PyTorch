"""Actual-forward convolution MAC profiling; no data, training or timing."""
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
from unittest.mock import patch

import torch
from torch import nn
import torchvision
import thop
from thop.vision.basic_hooks import count_convNd
from thop.profile import register_hooks

from model.roissd import RoiSSD
from model.roissd_mobilenet import RoiSSDMobileNet
from tools.helpers.config_reader import load_config

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
COMPONENTS=('Backbone','Additional feature layers','Classification heads','Localization heads','Other trainable modules')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (RESULTS/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(list,tuple,dict)) else v for k,v in r.items()})


def build_models(settings):
    models={};configs={}
    vgg=torchvision.models.vgg16;mobile=torchvision.models.mobilenet_v3_large
    for name,path in settings['models'].items():
        cfg=load_config(str(ROOT/path))
        assert cfg['dataset_params']['num_classes']==31
        # Check the architecture-defining values against the saved final configs.
        final_path=ROOT/'final_experiments/configs'/('roi_ssd.yaml' if name=='VGG16' else 'roi_ssd_mobilenet.yaml')
        final=load_config(str(final_path))
        for key in ('aspect_ratios','im_channels'):
            assert cfg['model_params'][key]==final['model_params'][key],(name,key)
        levels=len(cfg['model_params']['aspect_ratios'])
        effective=lambda c:(list(c['model_params']['scales'])+[1.0])[:levels+1]
        # Historical VGG audit YAML already contains the terminal scale added
        # by its old mutating constructor. Only the seven consumed scales matter.
        assert effective(cfg)==effective(final),(name,'effective scales')
        with patch('torchvision.models.vgg16',side_effect=lambda **_:vgg(weights=None)), \
             patch('torchvision.models.mobilenet_v3_large',side_effect=lambda **_:mobile(weights=None)):
            cls=RoiSSD if name=='VGG16' else RoiSSDMobileNet
            model=cls(copy.deepcopy(cfg['model_params']),num_classes=31,record_execution_trace=True)
        if name=='MobileNet':
            t=cfg['train_params']
            model.set_batch_norm_frozen(t.get('freeze_backbone_bn',True),t.get('freeze_extra_bn',False),t.get('train_bn_affine',True))
        model.eval().float().cpu()
        assert not any(isinstance(m,nn.Linear) for m in model.modules())
        models[name]=model
        configs[name]={'source':path,'source_sha256':sha(ROOT/path),'resolved':cfg,
                       'final_architecture_reference':str(final_path.relative_to(ROOT)),
                       'final_config_sha256':sha(final_path),'class':cls.__name__}
    return models,configs


def component_map(model,name):
    """Assign actual original-backbone objects separately from added adapters."""
    groups={id(module):'Other trainable modules' for module in model.modules()}
    def assign(root,component):
        for module in root.modules():
            previous=groups[id(module)]
            assert previous in ('Other trainable modules',component), 'Overlapping component ownership'
            groups[id(module)]=component
    if name=='VGG16':
        assign(model.features,'Backbone')
        children=list(model.conv5_3_fc.children())
        assert isinstance(children[-1],nn.Sequential)
        # Retained VGG conv5 block versus newly introduced dilated fc6/fc7.
        for child in children[:-1]:assign(child,'Backbone')
        assign(children[-1],'Additional feature layers')
    else:
        assign(model.features_stage1,'Backbone')
        assign(model.features_stage2[0],'Backbone')
        for child in list(model.features_stage2.children())[1:]:assign(child,'Additional feature layers')
        assign(model.conv_to_672,'Additional feature layers')
    for block in (model.conv8_2,model.conv9_2,model.conv10_2,model.conv11_2):assign(block,'Additional feature layers')
    assign(model.cls_heads,'Classification heads');assign(model.bbox_reg_heads,'Localization heads')
    return groups


def parameter_rows(model,name,groups):
    records={}
    for module in model.modules():
        for p in module.parameters(recurse=False):
            if id(p) in records:assert records[id(p)][0]==groups[id(module)]
            records[id(p)]=(groups[id(module)],p)
    assert set(records)=={id(p) for p in model.parameters()}
    rows=[]
    for component in COMPONENTS:
        params=[p for group,p in records.values() if group==component]
        rows.append({'model':name,'component':component,'params':sum(p.numel() for p in params),
                     'trainable_params':sum(p.numel() for p in params if p.requires_grad),
                     'nontrainable_params':sum(p.numel() for p in params if not p.requires_grad)})
    total=sum(p.numel() for p in model.parameters())
    assert total==sum(r['params'] for r in rows)
    return rows,{'model':name,'total_params':total,
                 'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),
                 'nontrainable_params':sum(p.numel() for p in model.parameters() if not p.requires_grad)}


def zero_ops(module,args,output):
    # THOP still measures leaf parameter counts; no non-conv arithmetic is
    # silently mixed into the primary MAC number.
    return None


def profiled_forward(model,name,shape,groups):
    module_names={id(m):n for n,m in model.named_modules()}
    old_buffers={id(m):set(m._buffers) for m in model.modules()}
    old_hooks={id(m):set(m._forward_hooks) for m in model.modules()}
    all_param_ids={id(p) for p in model.parameters()}
    active_params={};events=[];layers=[];handles=[]
    def activity(module,args,output):
        for p in module.parameters(recurse=False):active_params[id(p)]=p
    for module in model.modules():handles.append(module.register_forward_hook(activity))
    block_names=(['features','conv5_3_fc','conv8_2','conv9_2','conv10_2','conv11_2'] if name=='VGG16' else
                 ['features_stage1','conv_to_672','features_stage2','conv8_2','conv9_2','conv10_2','conv11_2'])
    for block in block_names:
        handles.append(getattr(model,block).register_forward_hook(lambda m,a,o,key=block:events.append(key)))
    for kind,heads in [('cls',model.cls_heads),('loc',model.bbox_reg_heads)]:
        for i,head in enumerate(heads):
            handles.append(head.register_forward_hook(lambda m,a,o,k=kind,i=i:events.append(f'{k}_{i}')))
    def convolution(module,args,output):
        before=float(module.total_ops)
        count_convNd(module,args,output)
        thop_delta=int(float(module.total_ops)-before)
        analytic=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        assert thop_delta==analytic,(module_names[id(module)],thop_delta,analytic)
        layers.append({'model':name,'input_h':shape[0],'input_w':shape[1],
                       'module':module_names[id(module)],'component':groups[id(module)],
                       'input_shape':list(args[0].shape),'output_shape':list(output.shape),
                       'in_channels':module.in_channels,'out_channels':module.out_channels,
                       'groups':module.groups,'kernel':list(module.kernel_size),
                       'depthwise':module.groups==module.in_channels and module.groups>1,
                       'thop_macs':thop_delta,'analytic_macs':analytic})
    custom={kind:zero_ops for kind in register_hooks}
    # Unregistered parameterless activations (e.g. Hardswish) are deliberately excluded.
    for module in model.modules():
        if not list(module.children()):custom[type(module)]=zero_ops
    custom[nn.Conv2d]=convolution
    x=torch.randn(1,3,*shape,dtype=torch.float32)
    try:
        with torch.no_grad():macs,thop_params=thop.profile(model,inputs=(x,),custom_ops=custom,verbose=False)
        trace=model.get_runtime_log()
        depth=trace['selected_L_R']
        assert trace['N_loc_predictions']==trace['N_cls_predictions']==trace['N_anchors']
        if name=='VGG16':
            expected=[]
            for i,block in enumerate(block_names[:depth]):expected.extend([block,f'cls_{i}',f'loc_{i}'])
        else:
            expected=['features_stage1','conv_to_672','cls_0','loc_0']
            for i,block in enumerate(block_names[2:depth+1],1):expected.extend([block,f'cls_{i}',f'loc_{i}'])
        assert events==expected,(shape,name,events,expected)
        assert len({r['module'] for r in layers})==len(layers), 'Convolution executed more than once'
        assert int(macs)==sum(r['analytic_macs'] for r in layers)
        active=sum(p.numel() for p in active_params.values())
        # THOP's module hooks miss VGG's standalone learned L2 scale only.
        missing=model.scale_weight.numel() if name=='VGG16' else 0
        assert int(thop_params)==active-missing,(name,thop_params,active,missing)
        components=[{'model':name,'input_h':shape[0],'input_w':shape[1],'component':component,
                     'macs':sum(r['analytic_macs'] for r in layers if r['component']==component)} for component in COMPONENTS]
        result={'model':name,'input_h':shape[0],'input_w':shape[1],'roi_area':shape[0]*shape[1],
                'status':'PASS','macs':int(macs),'gmacs':macs/1e9,
                'thop_active_path_params':int(thop_params),'direct_active_path_params':active,
                'thop_missing_standalone_params':missing,'full_model_params':sum(p.numel() for p in model.parameters()),
                'conv_calls':len(layers),'depthwise_calls':sum(r['depthwise'] for r in layers),
                'execution_events':events,**trace}
        return result,components,layers
    finally:
        for handle in handles:handle.remove()
        # THOP 2.0.19 leaves counters on unregistered composite modules.
        for module in model.modules():
            # THOP does not unwind hooks when an unsupported shape raises.
            for key in set(module._forward_hooks)-old_hooks[id(module)]:module._forward_hooks.pop(key,None)
            for key in ('total_ops','total_params'):
                if key not in old_buffers[id(module)]:module._buffers.pop(key,None)
        assert all_param_ids=={id(p) for p in model.parameters()}


def discover_depths(model,name,settings):
    rows=[];transitions=[];previous=None
    with torch.no_grad():
        for side in range(settings['transition_scan_min'],settings['transition_scan_max']+1):
            try:
                # Real model's feature iterator, not a hand-coded threshold table.
                features,_=model._build_feature_levels(torch.zeros(1,3,side,side))
                depth=len(features)
                row={'model':name,'side':side,'status':'PASS','depth':depth,
                     'feature_shapes':[list(f.shape[-2:]) for f in features],
                     'stop_reason':model.get_runtime_log()['next_head_status']}
                if previous is None or depth!=previous:
                    transitions.append({'model':name,'side':side,'previous_depth':previous,'new_depth':depth})
                previous=depth
            except RuntimeError as error:
                row={'model':name,'side':side,'status':'UNSUPPORTED','error':str(error).splitlines()[0]}
            rows.append(row)
    return rows,transitions


def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    settings=json.loads((HERE/'config.json').read_text())
    torch.set_num_threads(settings['threads']);torch.set_num_interop_threads(1)
    torch.manual_seed(settings['seed'])
    models,configs=build_models(settings)
    params=[];param_components=[];scans=[];transitions=[]
    for name,model in models.items():
        groups=component_map(model,name)
        rows,summary=parameter_rows(model,name,groups);param_components+=rows;params.append(summary)
        print('Discover actual square depth transitions:',name,flush=True)
        rows,edges=discover_depths(model,name,settings);scans+=rows;transitions+=edges
    write_csv('params_summary.csv',params);write_csv('params_by_component.csv',param_components)
    write_csv('depth_scan.csv',scans);write_csv('depth_transitions.csv',transitions)
    shapes={(s,s) for s in settings['squares']}
    shapes.update(map(tuple,settings['rectangles']));shapes.update(map(tuple,settings['equal_area_probes']))
    for edge in transitions:
        shapes.update((s,s) for s in (edge['side']-1,edge['side'],edge['side']+1) if 1<=s<=320)
    shapes=sorted(shapes,key=lambda s:(s[0]*s[1],s))
    measurements=[];components=[];layers=[]
    for name,model in models.items():
        groups=component_map(model,name)
        for shape in shapes:
            try:
                row,parts,details=profiled_forward(model,name,shape,groups)
                measurements.append(row);components+=parts;layers+=details
                print(name,f'{shape[0]}x{shape[1]}',f'{row["gmacs"]:.6f} GMAC',f'L={row["selected_L_R"]}',flush=True)
            except RuntimeError as error:
                measurements.append({'model':name,'input_h':shape[0],'input_w':shape[1],
                                     'roi_area':shape[0]*shape[1],'status':'UNSUPPORTED','error':str(error)})
    write_csv('macs_by_shape.csv',measurements);write_csv('macs_by_component.csv',components)
    write_csv('feature_depth_by_shape.csv',[{k:v for k,v in r.items() if k not in ('macs','gmacs')} for r in measurements])
    write_csv('conv_layer_validation.csv',layers)
    unsupported={name:sorted({type(m).__name__ for m in model.modules() if not list(m.children()) and not isinstance(m,nn.Conv2d)})
                 for name,model in models.items()}
    sources=('model/roissd.py','model/roissd_mobilenet.py','tools/helpers/config_reader.py',
             'experiments/roi_ssd_params_macs/profile_models.py','experiments/roi_ssd_params_macs/config.json')
    result={'status':'PROFILE_COMPLETE','created_at':datetime.now(timezone.utc).isoformat(),
            'settings':settings,'configs':configs,'checkpoints_used':False,'pretrained_weights_used':False,
            'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
            'thop':thop.__version__,'profiler_rule_version':'dense_conv2d_MAC_v1',
            'thop_counter_source_sha256':hashlib.sha256(inspect.getsource(count_convNd).encode()).hexdigest(),
            'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'git_tracked_diff':subprocess.check_output(['git','diff','--stat'],text=True).strip(),
            'source_sha256':{p:sha(ROOT/p) for p in sources},'nonconv_leaf_types_excluded':unsupported,
            'params':params,'params_by_component':param_components,'transitions':transitions,
            'measurements':measurements,'macs_by_component':components,
            'total_profiled_shapes_per_model':len(shapes),
            'conv_hook_analytic_checks_passed':len(layers)}
    write_json(RESULTS/'results.json',result)
    print('PROFILE_COMPLETE',HERE,flush=True)


if __name__=='__main__':main()
