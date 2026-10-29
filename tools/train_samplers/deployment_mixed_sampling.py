"""Deterministic, batch-shaped crops of deployment-resized source frames.

The raw dataset remains unchanged. Requests carry all randomness into workers;
neither worker scheduling nor global Python/Torch RNG affects crop selection.
"""
from dataclasses import dataclass, replace
import hashlib
import math
import logging
from numbers import Integral
import random

import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as VF


@dataclass(frozen=True)
class DeploymentSamplingConfig:
    full_probability: float
    min_crop_side: int
    aspect_ratio_range: tuple
    center_jitter_ratio: float
    reference_min_visibility: float
    other_box_min_visibility: float
    max_resample_attempts: int
    roi_area_fraction_bins: tuple

    @classmethod
    def from_mapping(cls, values, im_size):
        required=set(cls.__dataclass_fields__)
        if not isinstance(values,dict) or set(values)!=required:
            raise ValueError(f'roi_sampling requires exactly {sorted(required)}')
        result=cls(**dict(values,aspect_ratio_range=tuple(values['aspect_ratio_range']),
                          roi_area_fraction_bins=tuple(tuple(b) for b in values['roi_area_fraction_bins'])))
        result.validate(im_size)
        return result

    def validate(self, im_size):
        if isinstance(im_size,bool) or not isinstance(im_size,Integral) or im_size<1:
            raise ValueError('im_size must be a positive integer')
        if not 0<=self.full_probability<=1:
            raise ValueError('full_probability must be in [0,1]')
        if isinstance(self.min_crop_side,bool) or not isinstance(self.min_crop_side,Integral) or not 1<=self.min_crop_side<=im_size:
            raise ValueError('min_crop_side must be an integer in [1, im_size]')
        if len(self.aspect_ratio_range)!=2 or not 0<self.aspect_ratio_range[0]<=self.aspect_ratio_range[1]<float('inf'):
            raise ValueError('aspect_ratio_range must be finite positive ordered W/H bounds')
        if not 0<=self.center_jitter_ratio<=1:
            raise ValueError('center_jitter_ratio must be in [0,1]')
        if not 0<self.reference_min_visibility<=1 or not 0<self.other_box_min_visibility<=1:
            raise ValueError('visibility thresholds must be in (0,1]')
        if isinstance(self.max_resample_attempts,bool) or not isinstance(self.max_resample_attempts,Integral) or self.max_resample_attempts<1:
            raise ValueError('max_resample_attempts must be a positive integer')
        if not self.roi_area_fraction_bins:
            raise ValueError('roi_area_fraction_bins must not be empty')
        previous=0
        for entry in self.roi_area_fraction_bins:
            if len(entry)!=3:
                raise ValueError('Area bins must be [lower, upper, probability]')
            lo,hi,prob=entry
            if not 0<lo<hi<=1 or lo<previous or not 0<=prob<=1:
                raise ValueError('Area bins must be ordered, nonoverlapping and inside (0,1]')
            previous=hi
            # Reject bins whose continuous support cannot contain a valid shape.
            amax=hi*im_size**2
            rlo,rhi=self.aspect_ratio_range
            max_area=im_size**2*min(rhi,1/rlo,1.)
            min_area=self.min_crop_side**2*max(rlo,1/rhi,1.)
            if prob and (amax<min_area or lo*im_size**2>=max_area):
                raise ValueError(f'Infeasible area/aspect/min-side bin: {entry}')
        if not math.isclose(sum(b[2] for b in self.roi_area_fraction_bins),1.,abs_tol=1e-9):
            raise ValueError('Conditional ROI bin probabilities must sum to 1')


def stage_sampling_config(train_config, stage, im_size):
    if stage not in (2,3):
        raise ValueError('Deployment mixed training supports Stage 2/3; Stage 1 uses the separate full-frame script')
    try: values=train_config['roi_sampling'][f'stage{stage}']
    except KeyError as error:
        raise ValueError(f'Missing train_params.roi_sampling.stage{stage}; legacy square sampling is not a fallback') from error
    return DeploymentSamplingConfig.from_mapping(values,im_size)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:8],'big')


@dataclass(frozen=True)
class DeploymentSampleRequest:
    index: int
    mode: str
    batch_h: int
    batch_w: int
    seed: int
    stage: int
    epoch: int
    batch_index: int
    area_bin: int = -1
    sampled_area_fraction: float = 1.


class DeploymentMixedBatchSampler(Sampler):
    def __init__(self,dataset,batch_size,stage,sampling_config,im_size=320,
                 drop_last=True,shuffle=True,seed=42):
        if stage not in (2,3):raise ValueError('Only stages 2 and 3 use deployment mixed sampling')
        if isinstance(batch_size,bool) or not isinstance(batch_size,Integral) or batch_size<1:
            raise ValueError('batch_size must be a positive integer')
        if len(dataset)<1:raise ValueError('Dataset must not be empty')
        self.config=(sampling_config if isinstance(sampling_config,DeploymentSamplingConfig)
                     else DeploymentSamplingConfig.from_mapping(sampling_config,im_size))
        self.config.validate(im_size)
        self.dataset=dataset;self.batch_size=batch_size;self.stage=stage;self.im_size=im_size
        self.drop_last=drop_last;self.shuffle=shuffle;self.seed=int(seed);self.epoch=0

    def set_epoch(self,epoch):
        if isinstance(epoch,bool) or not isinstance(epoch,Integral) or epoch<0:
            raise ValueError('epoch must be a nonnegative integer')
        self.epoch=int(epoch)

    def sample_shape(self,rng):
        cfg=self.config
        if rng.random()<cfg.full_probability:return 'full',self.im_size,self.im_size,-1,1.
        index=rng.choices(range(len(cfg.roi_area_fraction_bins)),weights=[b[2] for b in cfg.roi_area_fraction_bins],k=1)[0]
        lo,hi,_=cfg.roi_area_fraction_bins[index]
        for _ in range(cfg.max_resample_attempts):
            fraction=rng.uniform(lo,hi)
            area=fraction*self.im_size**2
            # Log-uniform W/H on the feasible intersection. This preserves the
            # chosen bin/area distribution even for nearly full-frame ROIs.
            rlo=max(cfg.aspect_ratio_range[0],area/self.im_size**2,cfg.min_crop_side**2/area)
            rhi=min(cfg.aspect_ratio_range[1],self.im_size**2/area,area/cfg.min_crop_side**2)
            if rlo>rhi:continue
            ratio=math.exp(rng.uniform(math.log(rlo),math.log(rhi)))
            h=int(math.floor(math.sqrt(area/ratio)+.5));w=int(math.floor(math.sqrt(area*ratio)+.5))
            actual=h*w/self.im_size**2
            if (cfg.min_crop_side<=h<=self.im_size and cfg.min_crop_side<=w<=self.im_size
                    and cfg.aspect_ratio_range[0]<=w/h<=cfg.aspect_ratio_range[1]
                    and lo<=actual<=hi):
                return 'roi',h,w,index,fraction
        raise RuntimeError(f'Unable to sample an integer H/W in area bin {index}; check roi_sampling constraints')

    def __len__(self):
        return len(self.dataset)//self.batch_size if self.drop_last else math.ceil(len(self.dataset)/self.batch_size)

    def __iter__(self):
        rng=random.Random(stable_seed(self.seed,self.stage,self.epoch))
        indices=list(range(len(self.dataset)))
        if self.shuffle:rng.shuffle(indices)
        for batch_index,start in enumerate(range(0,len(indices),self.batch_size)):
            current=indices[start:start+self.batch_size]
            if len(current)<self.batch_size and self.drop_last:break
            mode,h,w,area_bin,fraction=self.sample_shape(rng)
            yield [DeploymentSampleRequest(index,mode,h,w,stable_seed(self.seed,self.stage,self.epoch,batch_index,slot),
                                           self.stage,self.epoch,batch_index,area_bin,fraction)
                   for slot,index in enumerate(current)]


def rectangle_targets(boxes, labels, roi, other_visibility, reference_index):
    left,top,width,height=roi
    clipped=boxes.clone()
    clipped[:,[0,2]]=clipped[:,[0,2]].clamp(left,left+width)
    clipped[:,[1,3]]=clipped[:,[1,3]].clamp(top,top+height)
    areas=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(1)
    visible=(clipped[:,2:]-clipped[:,:2]).clamp_min(0).prod(1)
    visibility=visible/areas.clamp_min(1e-12)
    keep=(visibility>=other_visibility)&(visible>0)
    if reference_index is not None:keep[reference_index]=True
    result=clipped[keep]-torch.tensor([left,top,left,top],dtype=boxes.dtype)
    result=result/torch.tensor([width,height,width,height],dtype=boxes.dtype)
    return {'boxes':result.clamp(0,1),'labels':labels[keep]},visibility


class DeploymentRoiBatchProcessor:
    def __init__(self,sampling_config,im_size=320):
        self.config=(sampling_config if isinstance(sampling_config,DeploymentSamplingConfig)
                     else DeploymentSamplingConfig.from_mapping(sampling_config,im_size))
        self.config.validate(im_size)
        self.im_size=im_size

    def resize_frame(self,image):
        # Raw datasets decode 8-bit RGB then divide by 255. Recover uint8 so
        # torchvision's resize/rounding exactly matches deployment SsdTransform.
        if image.dtype!=torch.uint8:
            image=image.float()
            if not torch.isfinite(image).all():raise ValueError('Non-finite source image')
            scale=255. if image.numel() and image.max()<=1. else 1.
            image=(image*scale).round().clamp(0,255).to(torch.uint8)
        resized=VF.resize(image,[self.im_size,self.im_size],interpolation=InterpolationMode.BILINEAR,antialias=True)
        return VF.to_dtype(resized,torch.float32,scale=True)

    def process_sample(self,sample,request,rng):
        if request.mode not in ('full','roi'):raise ValueError('mode must be full or roi')
        h,w=request.batch_h,request.batch_w
        if not (1<=h<=self.im_size and 1<=w<=self.im_size):raise ValueError('Invalid requested crop H/W')
        if request.mode=='roi' and min(h,w)<self.config.min_crop_side:raise ValueError('ROI is below min_crop_side')
        if request.mode=='full' and (h,w)!=(self.im_size,self.im_size):raise ValueError('Full mode must use im_size square')
        boxes=sample['target']['boxes'].detach().clone().float().reshape(-1,4)
        labels=sample['target']['labels'].detach().clone().long()
        if len(boxes)!=len(labels) or not torch.isfinite(boxes).all():raise ValueError('Invalid target boxes/labels')
        if ((boxes<0)|(boxes>1)).any() or ((boxes[:,2:]-boxes[:,:2])<=0).any() or (labels<=0).any():
            raise ValueError('Targets must be positive-area normalized boxes and foreground labels')
        boxes*=self.im_size
        ref=None;position_attempts=0
        if request.mode=='full':left=top=0
        else:
            sizes=boxes[:,2:]-boxes[:,:2]
            maximum_visibility=sizes.clamp(max=torch.tensor([w,h])).prod(1)/sizes.prod(1).clamp_min(1e-12)
            candidates=torch.where(maximum_visibility>=self.config.reference_min_visibility)[0].tolist()
            if not candidates:return None
            found=False
            for position_attempts in range(1,self.config.max_resample_attempts+1):
                ref=rng.choice(candidates)
                box=boxes[ref]
                cx,cy=((box[:2]+box[2:])/2).tolist()
                left=max(0,min(self.im_size-w,int(math.floor(cx+rng.uniform(-1,1)*self.config.center_jitter_ratio*w-w/2+.5))))
                top=max(0,min(self.im_size-h,int(math.floor(cy+rng.uniform(-1,1)*self.config.center_jitter_ratio*h-h/2+.5))))
                _,visibility=rectangle_targets(boxes,labels,(left,top,w,h),self.config.other_box_min_visibility,ref)
                if float(visibility[ref])+1e-7>=self.config.reference_min_visibility:
                    found=True;break
            if not found:return None
        target,visibility=rectangle_targets(boxes,labels,(left,top,w,h),self.config.other_box_min_visibility,ref)
        frame=self.resize_frame(sample['image'])
        crop=frame[:,top:top+h,left:left+w]
        assert crop.shape[-2:]==(h,w)
        mean=crop.new_tensor([.485,.456,.406])[:,None,None]
        std=crop.new_tensor([.229,.224,.225])[:,None,None]
        crop=(crop-mean)/std
        info={'reference_gt_area_ratio':None,'reference_center_offset':None,'reference_visibility':None,
              'reference_index':ref,'crop_left':left,'crop_top':top,'crop_position_attempts':position_attempts}
        if ref is not None:
            box=boxes[ref]
            center=((box[:2]+box[2:])/2).tolist()
            dx=(center[0]-left-w/2)/w;dy=(center[1]-top-h/2)/h
            original_area_ratio=float((box[2:]-box[:2]).prod())/(w*h)
            info.update(reference_gt_area_ratio=original_area_ratio*float(visibility[ref]),
                        reference_original_gt_area_ratio=original_area_ratio,
                        reference_center_offset=math.hypot(dx,dy),reference_center_dx=dx,reference_center_dy=dy,
                        reference_visibility=float(visibility[ref]))
        return crop,target,info


class DeploymentSamplingDataset(Dataset):
    """Deterministic item resampling happens inside each immutable request."""
    def __init__(self,raw_dataset,processor):self.raw_dataset=raw_dataset;self.processor=processor
    def __len__(self):return len(self.raw_dataset)
    def __getitem__(self,request):
        if not isinstance(request,DeploymentSampleRequest):raise TypeError('Expected DeploymentSampleRequest')
        rng=random.Random(request.seed)
        tried=[]
        for attempt in range(self.processor.config.max_resample_attempts):
            index=request.index if attempt==0 else rng.randrange(len(self.raw_dataset))
            tried.append(index)
            sample=self.raw_dataset[index]
            result=self.processor.process_sample(sample,request,rng)
            if result is not None:
                image,target,info=result
                info.update(stage=request.stage,epoch=request.epoch,mode=request.mode,
                            batch_index=request.batch_index,batch_h=request.batch_h,batch_w=request.batch_w,
                            crop_area_fraction=request.batch_h*request.batch_w/self.processor.im_size**2,
                            aspect_ratio=request.batch_w/request.batch_h,area_bin=request.area_bin,
                            sampled_area_fraction=request.sampled_area_fraction,resampling_attempts=attempt,
                            original_dataset_index=request.index,dataset_index=index,request_seed=request.seed)
                return {'image':image,'target':target,'sampling_info':info,
                        '_dataset':self,'_request':request}
        return {'_failed':True,'_dataset':self,'_request':request,'_tried':tried}



class DeploymentMixedCollateFn:
    def __call__(self,batch):
        if not batch:raise ValueError('Empty batch')
        requests=[sample['_request'] for sample in batch]
        key=lambda r:(r.stage,r.epoch,r.batch_index,r.mode,r.batch_h,r.batch_w)
        if any(key(r)!=key(requests[0]) for r in requests):
            raise ValueError('Batch definitions differ; padding/resizing to stack is prohibited')
        if any(sample.get('_failed',False) for sample in batch):
            enlarged=[]
            for sample,request in zip(batch,requests):
                dataset=sample['_dataset']
                retry=replace(request,batch_h=min(request.batch_h+10,dataset.processor.im_size),
                              batch_w=min(request.batch_w+10,dataset.processor.im_size))
                result=dataset[retry]
                if result.get('_failed',False):
                    logging.getLogger(__name__).error(
                        'Skipping batch: no eligible reference after %s enlarged item attempts; '
                        'stage=%s, epoch=%s, batch=%s, H/W=%sx%s, seed=%s, tried=%s',
                        len(result['_tried']),retry.stage,retry.epoch,retry.batch_index,
                        retry.batch_h,retry.batch_w,retry.seed,result['_tried'])
                    return None
                result['sampling_info'].update(shape_enlarged=True,
                    requested_batch_h=request.batch_h,requested_batch_w=request.batch_w,
                    original_shape_resampling_attempts=len(sample.get('_tried',[])))
                enlarged.append(result)
            batch=enlarged
        info=[sample['sampling_info'] for sample in batch]
        shape=(3,info[0]['batch_h'],info[0]['batch_w'])
        if any(tuple(s['image'].shape)!=shape for s in batch):raise ValueError('Crop shape differs from batch definition')
        return torch.stack([s['image'] for s in batch]),[s['target'] for s in batch],info[0]['mode'],{'samples':info}
