"""Fixed crops from a single deployment-sized frame, without object rescaling."""
import math

import torch
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F


class FixedSizeTestTransform:
    """Resize the frame once, then deterministically select a visible reference.

    Ineligible images carry ``fixed_crop_eligible=False`` and must be excluded
    from accuracy evaluation. The returned tensor always has the requested size.
    Visibility thresholds apply to box area remaining inside the crop.
    """
    def __init__(self, height, width, imagenet_mean, imagenet_std, normalize=True,
                 frame_size=320, reference_min_visibility=.9, other_box_min_visibility=.3):
        self.height=int(height)
        self.width=int(width)
        self.frame_size=int(frame_size)
        if not 1<=self.height<=self.frame_size or not 1<=self.width<=self.frame_size:
            raise ValueError('Fixed crop dimensions must be inside the deployment frame')
        if not 0<reference_min_visibility<=1 or not 0<other_box_min_visibility<=1:
            raise ValueError('Visibility thresholds must be in (0,1]')
        self.reference_min_visibility=reference_min_visibility
        self.other_box_min_visibility=other_box_min_visibility
        self.normalize=normalize
        self.mean=imagenet_mean
        self.std=imagenet_std
        self.transforms={'test':self._core}

    def _core(self,image,target):
        image,target=F.resize(image,[self.frame_size,self.frame_size],
                              interpolation=InterpolationMode.BILINEAR,antialias=True),dict(target)
        key='boxes' if 'boxes' in target else 'bboxes'
        original=target[key]
        # Resize boxes from their original canvas, independently of image pixels.
        source_h,source_w=original.canvas_size
        boxes=torch.as_tensor(original).clone().float().reshape(-1,4)
        boxes*=boxes.new_tensor([self.frame_size/source_w,self.frame_size/source_h]*2)
        areas=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(1)
        reference=None
        left=top=0
        for index,box in enumerate(boxes):
            cx,cy=((box[:2]+box[2:])/2).tolist()
            x=max(0,min(self.frame_size-self.width,int(math.floor(cx-self.width/2+.5))))
            y=max(0,min(self.frame_size-self.height,int(math.floor(cy-self.height/2+.5))))
            visible=(torch.minimum(box[2:],box.new_tensor([x+self.width,y+self.height]))-
                     torch.maximum(box[:2],box.new_tensor([x,y]))).clamp_min(0).prod()
            if areas[index]>0 and float(visible/areas[index])+1e-7>=self.reference_min_visibility:
                reference=index
                left,top=x,y
                break
        clipped=boxes.clone()
        clipped[:,0::2]=clipped[:,0::2].clamp(left,left+self.width)-left
        clipped[:,1::2]=clipped[:,1::2].clamp(top,top+self.height)-top
        visible=(clipped[:,2:]-clipped[:,:2]).clamp_min(0).prod(1)
        keep=(visible>0)&(visible/areas.clamp_min(1e-12)>=self.other_box_min_visibility)
        if reference is not None:
            keep[reference]=True
        else:
            keep[:]=False
        target[key]=tv_tensors.BoundingBoxes(clipped[keep],format='XYXY',canvas_size=(self.height,self.width))
        for field in ('labels','difficult'):
            if field in target:
                target[field]=target[field][keep]
        target.update(fixed_crop_eligible=reference is not None,
                      fixed_crop_reference_index=reference,
                      fixed_crop_left=left,fixed_crop_top=top)
        image=F.crop(image,top,left,self.height,self.width)
        image=F.to_dtype(image.as_subclass(torch.Tensor),torch.float32,scale=True)
        if self.normalize:
            image=F.normalize(image,self.mean,self.std)
        return image,target
