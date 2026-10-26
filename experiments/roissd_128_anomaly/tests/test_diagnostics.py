import json
from pathlib import Path
import random
import unittest

import numpy as np
import torch

from experiments.roissd_128_anomaly.instrumentation import ForwardObserver, assert_slots, geometry, matching, reconstruct_anchors
from experiments.roissd_128_anomaly.run import crop


class DiagnosticTests(unittest.TestCase):
    def test_slots_are_box_counts_not_tensor_elements(self):
        assert_slots(120,120,120)
        with self.assertRaisesRegex(AssertionError,'localization=120'):
            assert_slots(120,120,119)

    def test_observer_stops_mismatched_forward_and_restores_hooks(self):
        import model.roissd as architecture
        class MismatchedDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_classes=31
                self.low_score_threshold=.5
                self.cls_heads=torch.nn.ModuleList([torch.nn.Conv2d(3,62,1)])
                self.bbox_reg_heads=torch.nn.ModuleList([torch.nn.Conv2d(3,12,1)])
            def forward(self,x):
                self.cls_heads[0](x);self.bbox_reg_heads[0](x)
                return architecture.generate_default_boxes([x],[[1.]],[.1,.2],x.shape[-2:])
        MismatchedDetector.__module__='model.roissd'
        model=MismatchedDetector()
        original=architecture.generate_default_boxes
        with self.assertRaisesRegex(AssertionError,'Box-slot mismatch'):
            with ForwardObserver(model),torch.inference_mode():model(torch.zeros(1,3,4,4))
        self.assertIs(architecture.generate_default_boxes,original)
        self.assertFalse(model.cls_heads[0]._forward_hooks)
        self.assertFalse(model.bbox_reg_heads[0]._forward_hooks)

    def test_geometry_counts_clipping_without_losing_reference(self):
        raw=torch.tensor([[-.1,.2,.3,.6],[.4,.4,.6,.6]])
        row=geometry(raw.clamp(0,1),raw,100,200)
        self.assertEqual(row['returned_partially_outside'],0)
        self.assertEqual(row['before_clipping_reference_partially_outside'],1)
        self.assertAlmostEqual(row['pixel_width_max'],60,places=4)
        self.assertAlmostEqual(row['pixel_height_max'],40,places=4)
        with self.assertRaisesRegex(AssertionError,'geometry differs'):
            geometry(raw.clamp(0,1)+.01,raw,100,200)

    def test_reference_centers_use_both_rectangular_dimensions(self):
        boxes=reconstruct_anchors([(1,2)],[[1.]],[.2,.4],torch.device('cpu'))
        centers=(boxes[:,:2]+boxes[:,2:])/2
        torch.testing.assert_close(centers,torch.tensor([[.25,.5],[.25,.5],[.75,.5],[.75,.5]]))

    def test_matching_is_independent_of_confidence_and_tracks_forced_matches(self):
        anchors=torch.tensor([[0.,0.,.2,.2],[.7,.7,.9,.9]])
        gt=torch.tensor([[.01,.01,.19,.19],[.4,.4,.5,.5]])
        ious,positive,forced=matching(anchors,gt)
        self.assertGreater(ious[0],.5)
        self.assertEqual(positive,[1,0])
        self.assertGreaterEqual(forced[1],1)
        self.assertEqual(matching(anchors,torch.empty(0,4)),([],[],[]))

    def test_crops_are_reproducible_despite_intervening_rng_calls(self):
        row=json.loads((Path(__file__).parents[1]/'data/synthetic_example.json').read_text())
        image=(np.arange(240*320*3).reshape(240,320,3)%256).astype('uint8')
        first=crop(image,row,(128,160),123)
        random.random();torch.rand(5)
        second=crop(image,row,(128,160),123)
        for a,b in zip(first,second):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(first[0].shape,(1,3,128,160))
        self.assertTrue(((first[1]>=0)&(first[1]<=1)).all())
        self.assertEqual(first[2].tolist(),[1])


if __name__=='__main__':unittest.main()
