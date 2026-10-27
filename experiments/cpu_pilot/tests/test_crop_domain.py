import unittest

import numpy as np
import torch

from experiments.cpu_pilot.crop_domain import padded_tensor, crop_first, crop_rois, unique_matches, MEAN, STD
from experiments.cpu_pilot.oracle_diagnosis import geometry


class CropDomainTests(unittest.TestCase):
    def test_padding_preserves_center_and_uses_normalized_zero(self):
        tensor=torch.ones(1,3,320,320)
        box=[0,20,50,70]
        output,roi=padded_tensor(tensor,box,32,(320,320))
        self.assertEqual(roi,[-32,-12,82,102])
        self.assertEqual(tuple(output.shape),(3,114,114))
        self.assertTrue(torch.equal(output[:,:,:32],torch.zeros(3,114,32)))
        self.assertTrue(torch.equal(output[:,:12,:],torch.zeros(3,12,114)))
        self.assertTrue(torch.equal(output[:,12:,32:],torch.ones(3,102,82)))
        self.assertEqual(geometry(box,roi)['center_distance_ratio'],0)

    def test_padding_scales_anisotropically_with_source_resolution(self):
        _,roi=padded_tensor(torch.zeros(1,3,320,320),[0,40,100,140],32,(640,1280))
        self.assertEqual(roi,[-128,-24,228,204])

    def test_crop_first_keeps_explicit_roi_and_requested_shape(self):
        image=np.zeros((100,200,3),dtype=np.uint8)
        image[20:60,30:80]=255
        actual=crop_first(image,[30,20,80,60],(96,144))
        self.assertEqual(tuple(actual.shape),(3,96,144))
        torch.testing.assert_close(actual,((torch.ones(3,96,144)-MEAN)/STD))

    def test_oracle_rois_reuse_pilot_model_padding_units(self):
        self.assertEqual(crop_rois({'car':[[200,200,300,300]]},640,640,32),[[136,136,364,364]])

    def test_unique_head_provenance_uses_evaluator_best_gt_assignment(self):
        gt={'car':[[0,0,20,20],[30,0,50,20]]}
        detections=[{'class':'car','bbox':[0,0,20,20],'confidence':.9,'head':2},
                    {'class':'car','bbox':[0,0,20,20],'confidence':.8,'head':3},
                    {'class':'car','bbox':[30,0,50,20],'confidence':.7,'head':1}]
        result=unique_matches(detections,gt)
        self.assertEqual(result[('car',0)]['head'],2)
        self.assertEqual(result[('car',1)]['head'],1)
        self.assertEqual(len(result),2)


if __name__=='__main__':unittest.main()
