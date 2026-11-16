"""Test MobileNet executed shapes without network downloads or real weights."""
import inspect
import unittest
from unittest.mock import patch

from experiments.roi_merge_b.common import ROOT,config


class BackboneConstructionTests(unittest.TestCase):
    def test_full_checkpoint_constructor_needs_no_pretrained_download(self):
        import torch
        import torchvision
        from model.roissd_mobilenet import RoiSSDMobileNet
        cfg=config(ROOT/'experiments/roi_merge_b/config.example.yaml')
        params=cfg['models']['roi_ssd_mobilenet']['architecture']['model_params']
        original=torchvision.models.mobilenet_v3_large
        previous=torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with patch('torchvision.models.mobilenet_v3_large',wraps=original) as constructor:
                model=RoiSSDMobileNet(params,num_classes=31,record_execution_trace=True,pretrained_backbone=False).eval()
                self.assertIsNone(constructor.call_args.kwargs['weights'])
            with torch.inference_mode():
                for h,w in ((32,32),(32,320),(96,128),(320,320)):
                    model(torch.zeros(1,3,h,w),None)
                    trace=model.last_execution_trace
                    self.assertEqual(trace['depth_policy'],'until_1x1')
                    self.assertEqual(len(trace['active_heads']),trace['selected_L_R'])
                    self.assertTrue(all(min(s)>0 for s in trace['feature_shapes']))
            self.assertTrue(inspect.signature(RoiSSDMobileNet.__init__).parameters['pretrained_backbone'].default)
        finally:
            torch.set_num_threads(previous)


if __name__=='__main__':
    unittest.main()
