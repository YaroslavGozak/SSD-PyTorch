import copy
import random
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tools.helpers.config_reader import load_config
from tools.train_samplers.deployment_mixed_sampling import (
    DeploymentSamplingConfig, DeploymentSampleRequest, DeploymentMixedBatchSampler,
    DeploymentRoiBatchProcessor, DeploymentSamplingDataset, DeploymentMixedCollateFn,
    rectangle_targets, stage_sampling_config,
)
from tools.train_samplers.deployment_sampling_stats import DeploymentSamplingStats
from dataset.transforms.ssd_transform import SsdTransform

CONFIG=Path(__file__).resolve().parents[1]/'config/imagenet-vid-roissd-mobilenet.yaml'


class RawFixture(torch.utils.data.Dataset):
    def __init__(self,length=16,impossible=False):self.length=length;self.impossible=impossible
    def __len__(self):return self.length
    def __getitem__(self,index):
        image=(torch.arange(3*40*60).reshape(3,40,60)%256).to(torch.uint8)
        box=[0.,0.,1.,1.] if self.impossible or index==0 else [.35,.35,.45,.45]
        return {'image':image,'target':{'boxes':torch.tensor([box]),'labels':torch.tensor([1])}}


class DeploymentSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=load_config(str(CONFIG))['train_params']
        torch.set_num_threads(2)

    def cfg(self,stage=2):return stage_sampling_config(self.train,stage,320)

    def request(self,index=0,stage=2,mode='roi',h=128,w=224):
        return DeploymentSampleRequest(index,mode,h,w,33,stage,4,0)

    def test_distribution_and_continuous_shapes_for_both_stages(self):
        summaries=[]
        for stage,full_expected,bins in [(2,.5,[.05,.15,.35,.45]),(3,.3,[.25,.35,.25,.15])]:
            sampler=DeploymentMixedBatchSampler(range(20000),2,stage,self.cfg(stage),seed=856245)
            definitions=[batch[0] for batch in sampler]
            self.assertEqual(len(definitions),10000)
            roi=[r for r in definitions if r.mode=='roi']
            self.assertAlmostEqual(1-len(roi)/10000,full_expected,delta=.02)
            shapes={(r.batch_h,r.batch_w) for r in roi}
            self.assertGreater(len(shapes),1000)
            self.assertGreater(sum(h!=w for h,w in [(r.batch_h,r.batch_w) for r in roi])/len(roi),.95)
            self.assertTrue(any(h%32 and w%32 for h,w in shapes))
            for r in roi:
                self.assertTrue(self.cfg(stage).min_crop_side<=r.batch_h<=320)
                self.assertTrue(self.cfg(stage).min_crop_side<=r.batch_w<=320)
                self.assertTrue(.5<=r.batch_w/r.batch_h<=2.)
                self.assertLess(abs(r.batch_h*r.batch_w/320**2-r.sampled_area_fraction),.004)
            for index,expected in enumerate(bins):
                self.assertAlmostEqual(sum(r.area_bin==index for r in roi)/len(roi),expected,delta=.02)
            summaries.append(sum(r.batch_h*r.batch_w/320**2 for r in roi)/len(roi))
        self.assertLess(summaries[1],summaries[0]-.15)

    def test_epoch_resume_and_global_rng_independence(self):
        sampler=DeploymentMixedBatchSampler(range(40),2,3,self.cfg(3),seed=9)
        sampler.set_epoch(7);a=list(sampler)
        random.seed(999);[random.randint(0,10**6) for _ in range(100)]
        torch.manual_seed(999)
        self.assertEqual(a,list(sampler))
        resumed=DeploymentMixedBatchSampler(range(40),2,3,self.cfg(3),seed=9)
        resumed.set_epoch(7);self.assertEqual(a,list(resumed))
        sampler.set_epoch(8);self.assertNotEqual(a,list(sampler))

    def test_deterministic_resampling_never_enlarges_or_resizes_crop(self):
        processor=DeploymentRoiBatchProcessor(self.cfg())
        dataset=DeploymentSamplingDataset(RawFixture(),processor)
        request=self.request()
        a=dataset[request];random.seed(888);b=dataset[request]
        self.assertEqual(tuple(a['image'].shape),(3,128,224))
        self.assertGreater(a['sampling_info']['resampling_attempts'],0)
        self.assertNotEqual(a['sampling_info']['dataset_index'],0)
        self.assertEqual(a['sampling_info'],b['sampling_info'])
        torch.testing.assert_close(a['image'],b['image'],rtol=0,atol=0)
        self.assertGreaterEqual(a['sampling_info']['reference_visibility'],.90-1e-6)
        info=a['sampling_info'];raw=RawFixture()[info['dataset_index']]
        frame=processor.resize_frame(raw['image'])
        expected=frame[:,info['crop_top']:info['crop_top']+128,info['crop_left']:info['crop_left']+224]
        expected=(expected-expected.new_tensor([.485,.456,.406])[:,None,None])/expected.new_tensor([.229,.224,.225])[:,None,None]
        torch.testing.assert_close(a['image'],expected,rtol=0,atol=0)
        fallback=DeploymentSamplingDataset(RawFixture(impossible=True),processor)[request]
        self.assertTrue(fallback['_failed'])
        self.assertEqual(len(fallback['_tried']),30)
        with self.assertLogs('tools.train_samplers.deployment_mixed_sampling',level='ERROR') as logs:
            self.assertIsNone(DeploymentMixedCollateFn()([a,fallback]))
        self.assertIn('H/W=138x234',logs.output[0])

    def test_enlargement_retries_entire_batch_at_ten_pixels_larger(self):
        processor=DeploymentRoiBatchProcessor(self.cfg())
        class BorderFixture(RawFixture):
            def __getitem__(self,index):
                sample=super().__getitem__(index)
                sample['target']['boxes']=torch.tensor([[.2,.2,.63,.63]])
                return sample
        ds=DeploymentSamplingDataset(BorderFixture(),processor)
        request=self.request(h=128,w=128)
        failed=ds[request]
        self.assertTrue(failed['_failed'])
        images,targets,mode,metadata=DeploymentMixedCollateFn()([failed,ds[request]])
        self.assertEqual(tuple(images.shape),(2,3,138,138))
        self.assertEqual(mode,'roi')
        self.assertTrue(all(r['shape_enlarged'] for r in metadata['samples']))
        self.assertTrue(all(r['reference_visibility']>=.9-1e-6 for r in metadata['samples']))


    def test_full_frame_pixel_parity_with_deployment_transform(self):
        processor=DeploymentRoiBatchProcessor(self.cfg())
        raw=RawFixture()[1]
        actual=processor.process_sample(raw,self.request(1,mode='full',h=320,w=320),random.Random(3))[0]
        transform=SsdTransform(320,[123,117,104],[.485,.456,.406],[.229,.224,.225]).transforms['test']
        expected=transform(raw['image'])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        # Float raw-dataset decoding recovers the same uint8 resize path.
        converted=dict(raw,image=raw['image'].float()/255)
        other=processor.process_sample(converted,self.request(1,mode='full',h=320,w=320),random.Random(3))[0]
        torch.testing.assert_close(actual,other,rtol=0,atol=0)

    def test_rectangular_target_xy_normalization_and_visibility(self):
        boxes=torch.tensor([[30.,40.,110.,100.],[0.,0.,50.,50.]])
        targets,visibility=rectangle_targets(boxes,torch.tensor([1,2]),(10,20,200,100),.3,0)
        torch.testing.assert_close(targets['boxes'][0],torch.tensor([.1,.2,.5,.8]))
        self.assertEqual(targets['labels'].tolist(),[1,2])
        self.assertAlmostEqual(float(visibility[1]),.48,places=5)

    def test_reference_statistics_distinguish_visible_and_original_area(self):
        processor=DeploymentRoiBatchProcessor(self.cfg(3))
        sample=RawFixture()[1]
        sample['target']['boxes']=torch.tensor([[.15,.2,.85,.8]])
        result=processor.process_sample(sample,self.request(1,stage=3,h=160,w=224),random.Random(3))
        self.assertIsNotNone(result)
        info=result[2]
        self.assertLessEqual(info['reference_gt_area_ratio'],1.+1e-6)
        self.assertGreater(info['reference_original_gt_area_ratio'],info['reference_gt_area_ratio'])
        self.assertGreaterEqual(info['reference_visibility'],.8-1e-6)

    def test_collate_rejects_different_shapes_instead_of_padding(self):
        ds=DeploymentSamplingDataset(RawFixture(),DeploymentRoiBatchProcessor(self.cfg()))
        a=ds[self.request(1)];b=ds[self.request(2,h=224,w=128)]
        with self.assertRaisesRegex(ValueError,'Batch definitions differ'):
            DeploymentMixedCollateFn()([a,b])

    def test_worker_count_does_not_change_pixels_targets_or_sampling(self):
        def loader(workers):
            dataset=DeploymentSamplingDataset(RawFixture(8),DeploymentRoiBatchProcessor(self.cfg(3)))
            sampler=DeploymentMixedBatchSampler(dataset,2,3,self.cfg(3),seed=999)
            sampler.set_epoch(2)
            return DataLoader(dataset,batch_sampler=sampler,collate_fn=DeploymentMixedCollateFn(),num_workers=workers)
        a=list(loader(0));b=list(loader(2))
        self.assertEqual(len(a),len(b))
        for x,y in zip(a,b):
            torch.testing.assert_close(x[0],y[0],atol=0,rtol=0)
            self.assertEqual(x[2:],y[2:])
            for left,right in zip(x[1],y[1]):
                for key in left:torch.testing.assert_close(left[key],right[key],atol=0,rtol=0)

    def test_shared_training_loader_selects_both_yaml_stages(self):
        from tools.train_mixed_sampling import build_deployment_training_loader
        for stage in (2,3):
            params=copy.deepcopy(self.train);params.update(roi_mixed_stage=stage,batch_size=2)
            loader,sampler=build_deployment_training_loader(RawFixture(),{'im_size':320},params,num_workers=0)
            self.assertIsInstance(sampler,DeploymentMixedBatchSampler)
            self.assertIsInstance(loader.collate_fn,DeploymentMixedCollateFn)
            self.assertEqual(sampler.config,self.cfg(stage))
            batch=next(iter(loader));self.assertEqual(batch[0].shape[0],2)
        with self.assertRaisesRegex(ValueError,'Stage 1'):
            stage_sampling_config(self.train,1,320)

    def test_config_validation_and_rectangular_statistics(self):
        invalid=copy.deepcopy(self.train['roi_sampling']['stage2'])
        invalid['roi_area_fraction_bins'][0][2]=.9
        with self.assertRaisesRegex(ValueError,'sum to 1'):
            DeploymentSamplingConfig.from_mapping(invalid,320)
        ds=DeploymentSamplingDataset(RawFixture(),DeploymentRoiBatchProcessor(self.cfg()))
        a,b=ds[self.request(1)],ds[self.request(2)]
        images,_,mode,info=DeploymentMixedCollateFn()([a,b])
        stats=DeploymentSamplingStats(2,self.cfg());stats.record(mode,images.shape[-2:],info)
        summary=stats.summary()
        self.assertEqual(summary['HW_counts'],{'128x224':1})
        self.assertEqual(summary['rectangular_roi_share'],1)
        directory=CONFIG.parents[1]/'final_experiments/deployment_mixed_sampling_check/test_stats'
        stats.save(directory,epoch=5)
        self.assertTrue((directory/'deployment_sampling_stage2_epoch0005_batches.csv').is_file())
        self.assertTrue((directory/'deployment_sampling_stage2_epoch0005_samples.csv').is_file())


if __name__=='__main__':unittest.main()
