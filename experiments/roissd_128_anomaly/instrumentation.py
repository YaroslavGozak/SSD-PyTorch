"""Scoped observation of the real forward; no model source or weights changed."""
import importlib
import math
from unittest.mock import patch

import torch
from torchvision.ops import box_iou


def assert_slots(loc, cls, anchors):
    if not (loc == cls == anchors):
        raise AssertionError(f'Box-slot mismatch: localization={loc}, classification={cls}, anchors={anchors}')


def reconstruct_anchors(features, aspects, scales, device):
    """Independent center/size reference before model clipping, normalized xyxy."""
    result = []
    for i, (h, w) in enumerate(features):
        wh = [(math.sqrt(scales[i]*scales[i+1]),)*2]
        wh += [(scales[i]*math.sqrt(ar), scales[i]/math.sqrt(ar)) for ar in aspects[i]]
        yy, xx = torch.meshgrid((torch.arange(h, device=device)+.5)/h,
                                (torch.arange(w, device=device)+.5)/w, indexing='ij')
        centers = torch.stack([xx, yy], -1).reshape(-1, 1, 2)
        dims = torch.tensor(wh, device=device)
        result.append(torch.cat([centers-dims/2, centers+dims/2], -1).reshape(-1, 4))
    return torch.cat(result)


def geometry(anchors, unclipped, height, width):
    wh = anchors[:, 2:]-anchors[:, :2]
    assert torch.isfinite(anchors).all() and (wh > 0).all(), 'Non-finite/degenerate anchors'
    row = {'num_anchors': len(anchors), 'coordinate_system': 'normalized_xyxy',
           'reference_max_abs_error': float((anchors-unclipped.clamp(0,1)).abs().max())}
    assert row['reference_max_abs_error'] < 1e-6, 'Anchor order/geometry differs from center-size reference'
    for name, values in [('width',wh[:,0]), ('height',wh[:,1]), ('aspect_ratio',wh[:,0]/wh[:,1]),
                         ('pixel_width',wh[:,0]*width), ('pixel_height',wh[:,1]*height)]:
        row.update({name+'_min':float(values.min()), name+'_median':float(values.median()),name+'_max':float(values.max())})
    for prefix, boxes in [('returned',anchors), ('before_clipping_reference',unclipped)]:
        outside = ((boxes[:,:2]<0).any(1) | (boxes[:,2:]>1).any(1))
        fully = ((boxes[:,2]<=0)|(boxes[:,3]<=0)|(boxes[:,0]>=1)|(boxes[:,1]>=1))
        row[prefix+'_partially_outside'] = int((outside & ~fully).sum())
        row[prefix+'_fully_outside'] = int(fully.sum())
    return row


def matching(anchors, gt, threshold=.5):
    if not len(gt):
        return [], [], []
    ious = box_iou(gt, anchors)
    best = ious.max(1).values
    positives_per_gt = (ious>=threshold).sum(1)
    # Reproduce actual training assignment semantics: one GT per anchor,
    # threshold filtering followed by forced best-anchor matching.
    vals, assignments = ious.max(0)
    assignments[vals<threshold] = -1
    assignments[ious.argmax(1)] = torch.arange(len(gt), device=anchors.device)
    forced = [(assignments==i).sum().item() for i in range(len(gt))]
    return best.tolist(), positives_per_gt.tolist(), forced


class ForwardObserver:
    def __init__(self, model):
        self.model = model
        self.module = importlib.import_module(model.__class__.__module__)
        self.handles = []
        self.heads = {}
        self.blocks = []
        self.nms_calls = []

    def __enter__(self):
        for i, head in enumerate(self.model.cls_heads):
            self.handles.append(head.register_forward_hook(self._head_hook(i,'cls')))
        for i, head in enumerate(self.model.bbox_reg_heads):
            self.handles.append(head.register_forward_hook(self._head_hook(i,'loc')))
        for name, block in self.model.named_children():
            if name not in ('cls_heads','bbox_reg_heads'):
                self.handles.append(block.register_forward_hook(self._block_hook(name)))
        original_generate = self.module.generate_default_boxes
        original_decode = self.module.apply_regression_pred_to_default_boxes
        original_nms = self.module.torchvision.ops.batched_nms

        def generate(features, aspects, scales, image_size):
            boxes = original_generate(features, aspects, scales, image_size)
            self.anchors = boxes[0]
            self.feature_shapes = [list(x.shape[-2:]) for x in features]
            self.image_size = list(image_size)
            self.reference = reconstruct_anchors(self.feature_shapes,aspects,scales,self.anchors.device)
            loc = sum(h['loc_slots'] for h in self.heads.values())
            cls = sum(h['cls_slots'] for h in self.heads.values())
            assert_slots(loc,cls,len(self.anchors))
            self.loc_shape = [1,loc,4]
            self.cls_shape = [1,cls,self.model.num_classes]
            return boxes

        def decode(deltas, anchors, *args, **kwargs):
            assert_slots(len(deltas),sum(h['cls_slots'] for h in self.heads.values()),len(anchors))
            self.decoded = original_decode(deltas,anchors,*args,**kwargs)
            assert torch.isfinite(self.decoded).all(), 'Non-finite decoded boxes'
            return self.decoded

        def nms(boxes,scores,labels,iou):
            keep = original_nms(boxes,scores,labels,iou)
            self.nms_calls.append({'pre':len(boxes),'after_nms_before_cap':len(keep),
                                   'scores_max':float(scores.max()) if len(scores) else 0.})
            return keep

        self.patches = [patch.object(self.module,'generate_default_boxes',generate),
                        patch.object(self.module,'apply_regression_pred_to_default_boxes',decode),
                        patch.object(self.module.torchvision.ops,'batched_nms',nms)]
        for p in self.patches:p.start()
        return self

    def _head_hook(self,index,kind):
        def hook(module,args,output):
            row = self.heads.setdefault(index,{})
            h,w = args[0].shape[-2:]
            coords = self.model.num_classes if kind=='cls' else 4
            assert output.shape[1] % coords == 0
            anchors_per_location = output.shape[1]//coords
            row.update(feature_h=h,feature_w=w,anchors_per_location=anchors_per_location)
            row[kind+'_slots'] = h*w*anchors_per_location
            row[kind+'_output_shape'] = list(output.shape)
            if kind=='cls':
                logits=output.view(1,anchors_per_location,self.model.num_classes,h,w)
                logits=logits.permute(0,3,4,1,2).reshape(-1,self.model.num_classes)
                fg=logits.softmax(-1)[:,1:]
                row['foreground_max_confidence']=float(fg.max())
                row['foreground_over_threshold']=int((fg>self.model.low_score_threshold).sum())
        return hook

    def _block_hook(self,name):
        def hook(module,args,output):self.blocks.append(name)
        return hook

    def reset(self):
        self.heads={};self.blocks=[];self.nms_calls=[]
        self.anchors=None;self.decoded=None

    def __exit__(self,*args):
        for p in reversed(self.patches):p.stop()
        for h in self.handles:h.remove()
