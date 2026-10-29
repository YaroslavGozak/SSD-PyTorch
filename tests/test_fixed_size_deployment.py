import random
import unittest
from unittest.mock import patch, MagicMock
import argparse
import csv
from test_batch_infer import test_directory

import torch
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

from dataset.transforms.fixed_size_test_transform import FixedSizeTestTransform
from tools import infer


class FixedSizeDeploymentTests(unittest.TestCase):
    def transform(self,h=64,w=96,**kwargs):
        return FixedSizeTestTransform(h,w,[.485,.456,.406],[.229,.224,.225],**kwargs).transforms['test']

    def target(self,boxes,canvas=(320,320)):
        return {'bboxes':tv_tensors.BoundingBoxes(torch.tensor(boxes).float().reshape(-1,4),
                    format='XYXY',canvas_size=canvas),
                'labels':torch.arange(len(boxes))+1,'difficult':torch.zeros(len(boxes),dtype=torch.long)}

    def test_exact_pixels_and_geometry_after_single_frame_resize(self):
        image=(torch.arange(3*40*60).reshape(3,40,60)%256).to(torch.uint8)
        target=self.target([[20,15,25,20]],canvas=(40,60))
        actual,out=self.transform()(image,target)
        self.assertTrue(out['fixed_crop_eligible'])
        frame=F.resize(image,[320,320],interpolation=InterpolationMode.BILINEAR,antialias=True)
        expected=frame[:,out['fixed_crop_top']:out['fixed_crop_top']+64,
                         out['fixed_crop_left']:out['fixed_crop_left']+96]
        expected=F.normalize(F.to_dtype(expected,torch.float32,scale=True),[.485,.456,.406],[.229,.224,.225])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        # Width and height stay at their deployment-frame pixel dimensions.
        torch.testing.assert_close(out['bboxes'][0,2:]-out['bboxes'][0,:2],torch.tensor([5*320/60,40.]))

    def test_selects_next_eligible_reference_without_rescaling_large_object(self):
        image=torch.zeros(3,320,320,dtype=torch.uint8)
        target=self.target([[0,0,320,320],[150,150,170,170]])
        a,out=self.transform()(image,target)
        self.assertEqual(out['fixed_crop_reference_index'],1)
        self.assertEqual(out['labels'].tolist(),[2])
        torch.testing.assert_close(out['bboxes'][0,2:]-out['bboxes'][0,:2],torch.tensor([20.,20.]))
        random.seed(135);torch.manual_seed(17)
        b,again=self.transform()(image,target)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(out['fixed_crop_left'],again['fixed_crop_left'])

    def test_ineligible_and_empty_images_have_no_evaluation_targets(self):
        for boxes in ([[0,0,320,320]],[]):
            image,out=self.transform()(torch.zeros(3,320,320,dtype=torch.uint8),self.target(boxes))
            self.assertFalse(out['fixed_crop_eligible'])
            self.assertEqual(tuple(image.shape),(3,64,96))
            self.assertEqual(len(out['bboxes']),0)

    def test_boundary_crop_is_clamped_and_visibility_threshold_is_applied(self):
        image=torch.zeros(3,320,320,dtype=torch.uint8)
        _,out=self.transform()(image,self.target([[300,300,320,320]]))
        self.assertEqual((out['fixed_crop_left'],out['fixed_crop_top']),(224,256))
        _,out=self.transform(reference_min_visibility=.5)(image,self.target([[0,0,96,100]]))
        self.assertTrue(out['fixed_crop_eligible'])
        _,out=self.transform(reference_min_visibility=.9)(image,self.target([[0,0,96,100]]))
        self.assertFalse(out['fixed_crop_eligible'])

    def test_oversize_requests_fail_instead_of_resizing(self):
        with self.assertRaises(ValueError):
            self.transform(h=321)

    def test_evaluator_excludes_ineligible_images_and_reports_coverage(self):
        samples=[]
        for eligible in (False,True,False):
            samples.append((torch.zeros(3,64,96),{
                'bboxes':torch.tensor([[.1,.1,.4,.4]]),
                'labels':torch.tensor([0]),'difficult':torch.tensor([0]),
                'fixed_crop_eligible':eligible},'frame'))
        dataset=MagicMock()
        dataset.label2idx={'object':0}
        dataset.idx2label={0:'object'}
        loader=torch.utils.data.DataLoader(samples,batch_size=2,collate_fn=infer.evaluation_collate)
        with test_directory() as directory:
            args=argparse.Namespace(summary_path=str(directory),results_path=None,pin_memory=False)
            def detect(model,images,*args,**kwargs):
                self.assertEqual(len(images),1)
                return [{'boxes':torch.tensor([[.1,.1,.4,.4]]),'labels':torch.tensor([0]),
                         'scores':torch.tensor([.9])}]
            with patch.object(infer,'evaluation_resources',return_value=(object(),dataset,loader,{'train_params':{}})), \
                 patch.object(infer,'device',torch.device('cpu')), \
                 patch.object(infer,'run_detector',side_effect=detect) as detector:
                infer._evaluate_single_transform(args,transform_name='fixed_size_64x96',
                    evaluated_dataset='fixed_size_64x96',model_name='roissd')
                self.assertEqual(detector.call_count,1)
            with (directory/'fixed_size_coverage_v2.csv').open() as stream:
                row=next(csv.DictReader(stream))
            self.assertEqual(row['eligible_images'],'1')
            self.assertEqual(row['ineligible_images'],'2')


if __name__=='__main__':
    unittest.main()
