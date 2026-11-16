"""Protocol invariants; real shape checks reuse the production crop path."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.roi_merge_b.common import (MODELS,ROOT,atomic,canonical,checkpoint,config,digest,load_manifest,sha)
from experiments.roi_merge_b.metrics import pair_summary
from experiments.roi_merge_b.policies import Estimator,OracleCosts,plan
from experiments.roi_merge_b.workload import (all_rectangles,area,clip_context,effective,shape_for,union,write_workload)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg=config(ROOT/'experiments/roi_merge_b/config.example.yaml')
        self.options=self.cfg['policies']
        self.artifact={'models':{'affine':{'b0':.01,'b1':.000001},'piecewise':{'b0':.01,'b1':.000001,'b2':.000001,'breakpoint':10000}},
                       'lookup':{'32x32':{'mean_ms':1.,'se_ms':.1,'n':8},'64x64':{'mean_ms':1.8,'se_ms':.1,'n':8}}}
        self.rois=[{'roi_id':'a','box':[0,0,32,32]},{'roi_id':'b','box':[64,64,96,96]},{'roi_id':'c','box':[96,64,128,96]}]
        self.shape=lambda box:(box[3]-box[1],box[2]-box[0])

    def test_affine_tau_and_strict_ties(self):
        estimator=Estimator(self.artifact,'affine',self.options)
        # tau = .01/.000001 = 10000 pixels. Strict inequality, equality separates.
        for merged in ((50,50),(100,100),(100,120)):
            expected=merged[0]*merged[1]-2*32*32<10000
            self.assertEqual(estimator.decision((32,32),(32,32),merged)['merge'],expected)
        simple={'models':{'affine':{'b0':1.,'b1':1.}}}
        self.assertFalse(Estimator(simple,'affine',self.options).decision((1,1),(1,1),(1,3))['merge'])

    def test_conservative_abstains_missing_and_uncertain(self):
        estimator=Estimator(self.artifact,'conservative_lookup',self.options)
        self.assertFalse(estimator.decision((32,32),(32,32),(64,64))['merge'])
        self.assertEqual(estimator.decision((32,32),(32,32),(96,96))['fallback'],'unmeasured_shape')
        del self.artifact['lookup']['32x32']['se_ms']
        # Missing uncertainty should abstain rather than infer a confidence bound.
        self.artifact['lookup']['32x32']['se_ms']=None
        self.assertFalse(estimator.decision((32,32),(32,32),(32,32))['merge'])

    def test_lookup_is_ordered_nearest_neighbor(self):
        estimator=Estimator(self.artifact,'lookup',self.options)
        self.assertEqual(estimator.estimate((33,32))[0],1.)
        self.assertFalse(estimator.estimate((33,32))[2])

    def test_oracle_role_separation(self):
        with self.assertRaises(ValueError):
            Estimator(self.artifact,'oracle',self.options)
        with self.assertRaises(ValueError):
            OracleCosts({'role':'calibration','table':{},'calibration_id':'x'})

    def test_geometric_matches_repository_rule(self):
        from tools.mergers.simple2 import simple_roi_merge_v2
        for rois in (self.rois,self.rois[::-1],[],self.rois[:1],
                     [{'roi_id':'a','box':[0,0,100,100]},{'roi_id':'b','box':[10,10,20,20]}]):
            groups,_,_=plan(rois,'geometric',self.shape,self.options)
            self.assertEqual([g['box'] for g in groups],[list(b) for b in simple_roi_merge_v2([r['box'] for r in rois],1.4)])

    def test_determinism_partition_and_enclosure(self):
        estimator=Estimator(self.artifact,'affine',self.options)
        first=plan(self.rois,'affine',self.shape,self.options,estimator)
        self.assertEqual(first,plan(self.rois,'affine',self.shape,self.options,estimator))
        groups=first[0]
        self.assertEqual(sorted(v for g in groups for v in g['ids']),['a','b','c'])
        for group in groups:
            for roi in self.rois:
                if roi['roi_id'] in group['ids']:
                    self.assertEqual(union(group['box'],roi['box']),group['box'])

    def test_exact_oracle_and_greedy_label(self):
        table={f'{h}x{w}':[.5+h*w*.00001]*3 for h,w in {self.shape(b) for b in all_rectangles(self.rois)}}
        oracle=OracleCosts({'role':'heldout_reference','calibration_id':'x','table':table})
        groups,_,kind=plan(self.rois,'oracle',self.shape,self.options,oracle=oracle)
        self.assertEqual(kind,'exact_additive_partition')
        self.assertEqual(len(groups),1)
        options={**self.options,'exact_oracle_max_rois':1}
        self.assertEqual(plan(self.rois,'oracle',self.shape,options,oracle=oracle)[2],'greedy_measured_reference')

    def test_serialized_cost_model_preserves_cross_platform_groups(self):
        estimator=Estimator(self.artifact,'affine',self.options)
        restored=Estimator(json.loads(canonical(self.artifact)),'affine',self.options)
        self.assertEqual(plan(self.rois,'affine',self.shape,self.options,estimator),
                         plan(self.rois,'affine',self.shape,self.options,restored))

    def test_empty_no_merge(self):
        self.assertEqual(plan([],'no_merge',self.shape,self.options)[0],[])
        self.assertEqual(len(plan(self.rois,'no_merge',self.shape,self.options)[0]),3)

    def test_context_clipping_and_minimum_shape(self):
        for cx,cy in ((0.,0.),(1.,1.),(.5,.5)):
            b=clip_context(cx,cy,.001,.001,1920,1080,self.cfg['geometry'])
            self.assertTrue(0<=b[0]<b[2]<=1920 and 0<=b[1]<b[3]<=1080)
            hw,_=effective(tuple(b),(1080,1920),(320,320),1)
            self.assertGreaterEqual(min(hw),32)

    def test_actual_crop_shape_and_yolo_grid(self):
        import torch
        from tools.helpers.pipeline import convert_crop_to_input_tensor
        for box in ((0,0,1920,1080),(7,11,801,302),(1800,1000,1920,1080),(0,0,1,1080)):
            for grid in (1,32):
                expected,snapped=effective(box,(1080,1920),(320,320),grid)
                tensor,actual=convert_crop_to_input_tensor(torch.zeros(3,320,320),list(box),(1080,1920),roi_grid=grid)
                self.assertEqual(expected,tuple(tensor.shape[-2:]));self.assertEqual(snapped,tuple(actual))
                if grid==32:
                    self.assertTrue(all(v%32==0 for v in expected))

    def manifest_row(self):
        return {'video_id':'a','frame_index':0,'image':'images/a.png','label':'labels/a.txt',
                'width':320,'height':320,'image_sha256':'x','label_sha256':'y','split':'test','rois':self.rois}

    def test_canonical_manifest_and_hash_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            for directory in (a,b):
                write_workload([self.manifest_row()],self.cfg,directory,['x'],'index',True)
            self.assertEqual((a/'manifest.jsonl').read_bytes(),(b/'manifest.jsonl').read_bytes())
            load_manifest(a/'manifest.jsonl',self.cfg)
            (a/'manifest.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                load_manifest(a/'manifest.jsonl',self.cfg)

    def test_sequence_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            row=self.manifest_row();other={**row,'frame_index':1,'split':'calibration'}
            write_workload([row,other],self.cfg,tmp,['x'],'index',True)
            with self.assertRaisesRegex(ValueError,'split leakage'):
                load_manifest(Path(tmp)/'manifest.jsonl',self.cfg)

    def test_checkpoint_missing_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                checkpoint(self.cfg,'yolo26n',tmp)
            path=Path(tmp)/self.cfg['models']['yolo26n']['checkpoint'];path.parent.mkdir(parents=True)
            path.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                checkpoint(self.cfg,'yolo26n',tmp)

    def test_metrics_denominators_and_regret(self):
        rows=[]
        for predicted,truth in ((True,True),(False,False),(True,False),(False,True)):
            sep,merged=(2.,1.) if truth else (1.,2.)
            chosen=merged if predicted else sep;oracle=min(sep,merged)
            rows.append({'merge':predicted,'oracle_merge':truth,'chosen_ms':chosen,'separate_ms':sep,
                         'oracle_ms':oracle,'regret_ms':chosen-oracle,'scoring_min_regret_ms':chosen-oracle,'reference_scoring_disagreement':False})
        result=pair_summary(rows)
        self.assertEqual([result[k] for k in ('TP','TN','FP','FN')],[1,1,1,1])
        self.assertEqual(result['false_merge_rate'],.5)
        self.assertEqual(result['false_separate_rate'],.5)
        self.assertGreaterEqual(result['mean_regret_ms'],0)


if __name__=='__main__':
    unittest.main()
