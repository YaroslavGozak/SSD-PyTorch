"""Independent ordinary/grouped/depthwise Conv2d THOP validation."""
import json
import csv
import torch
from torch import nn
from thop import profile
from thop.vision.basic_hooks import count_convNd

from .profile_models import HERE,RESULTS,write_json,write_csv,build_models,profiled_forward,component_map


def main():
    RESULTS.mkdir(exist_ok=True,parents=True)
    cases=[]
    for name,cin,cout,groups in [('standard',3,7,1),('depthwise',8,8,8),('depthwise_multiplier_2',8,16,8),('grouped',8,12,4)]:
        model=nn.Conv2d(cin,cout,3,stride=2,padding=1,groups=groups,bias=True).eval()
        x=torch.zeros(1,cin,13,17)
        with torch.no_grad():y=model(x)
        expected=y.shape[2]*y.shape[3]*cout*(cin//groups)*9
        macs,params=profile(model,inputs=(x,),custom_ops={nn.Conv2d:count_convNd},verbose=False)
        assert int(macs)==expected
        assert int(params)==sum(p.numel() for p in model.parameters())
        cases.append({'case':name,'input':[1,cin,13,17],'output':[1,cout,7,9],
                      'groups':groups,'analytic_macs':expected,'thop_macs':int(macs),'status':'PASS',
                      'direct_params':sum(p.numel() for p in model.parameters()),'thop_params':int(params)})
    verification={'status':'PASS','cases':cases,
                  'note':'Direct parameter counts agree for standard registered Conv2d; full detectors also contain standalone parameters and inactive paths.'}
    if (RESULTS/'results.json').exists():
        settings=json.loads((HERE/'config.json').read_text())
        torch.set_num_threads(settings['threads']);torch.manual_seed(settings['seed'])
        models,_=build_models(settings)
        references=[]
        for name,model in models.items():
            groups=component_map(model,name)
            if name=='VGG16':
                try:profiled_forward(model,name,(3,3),groups)
                except RuntimeError:pass
                else:raise AssertionError('VGG16 3x3 should be unsupported')
            a,_,_=profiled_forward(model,name,(320,320),groups)
            b,_,_=profiled_forward(model,name,(320,320),groups)
            assert a['macs']==b['macs']
            assert not any(m._forward_hooks for m in model.modules())
            names=dict(model.named_modules())
            buffers={id(m):set(m._buffers) for m in model.modules()}
            try:
                ops,parameters,tree=profile(model,inputs=(torch.zeros(1,3,320,320),),verbose=False,ret_layer_info=True)
                leaves=[]
                def visit(nodes,prefix=''):
                    for key,(count,params,children) in nodes.items():
                        path=f'{prefix}.{key}' if prefix else key
                        if children:visit(children,path)
                        else:leaves.append((path,int(count)))
                visit(tree)
                conv_ops=sum(count for path,count in leaves if isinstance(names[path],nn.Conv2d))
                assert conv_ops==a['macs']
                assert int(ops)==sum(count for _,count in leaves)
                references.append({'model':name,'input_h':320,'input_w':320,
                                   'primary_conv_macs':a['macs'],'thop_default_mixed_operation_count':int(ops),
                                   'additional_nonconv_thop_operations':int(ops)-conv_ops,
                                   'default_count_is_not_a_pure_MAC_metric':True,
                                   'thop_active_path_params':int(parameters),
                                   'direct_full_params':sum(p.numel() for p in model.parameters())})
                for path,count in leaves:
                    if not isinstance(names[path],nn.Conv2d) and count:
                        references.append({'model':name,'module':path,'operation_type':type(names[path]).__name__,
                                           'nonconv_thop_operations':count,'default_count_is_not_a_pure_MAC_metric':True})
            finally:
                for m in model.modules():
                    for key in ('total_ops','total_params'):
                        if key not in buffers[id(m)]:m._buffers.pop(key,None)
        write_csv('thop_supplementary_ops.csv',references)
        verification.update(production_conv_totals_match_default_THOP=True,
                            repeated_profile_and_failed_forward_hook_cleanup=True,
                            verified_shapes=[[320,320]],models=list(models))
    write_json(RESULTS/'profiler_validation.json',verification)
    print('PASS: standard, depthwise, multiplier-2 and grouped convolution formula checks')


if __name__=='__main__':main()
