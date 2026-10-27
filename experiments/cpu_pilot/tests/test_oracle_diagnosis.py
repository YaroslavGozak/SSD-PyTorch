import unittest

import torch

from experiments.cpu_pilot.oracle_diagnosis import (
    geometry, match_objects, object_bins, oracle, pair_counts, postprocess,
)


class OracleDiagnosisTests(unittest.TestCase):
    def test_duplicate_matching_and_classwise_nms(self):
        dets = [{'bbox': [10, 10, 40, 40], 'class': cls, 'confidence': conf}
                for cls, conf in [('car', .9), ('car', .8), ('bird', .7)]]
        gt = {'car': [[10, 10, 40, 40]]}
        result = match_objects(dets, gt)
        self.assertEqual((result['tp'], result['fp'], result['duplicate_detections']), (1, 2, 1))
        self.assertEqual(pair_counts(dets)['same_class_pairs_iou_gt_05'], 1)
        self.assertEqual(len(postprocess(dets, .5, 100, 100)), 2)
        self.assertEqual(len(postprocess(dets, .5, 100, 100, nms=False)), 3)

    def test_geometry_reports_offcenter_and_boundary(self):
        result = geometry([0, 10, 50, 90], [0, 0, 100, 100])
        self.assertEqual(result['gt_area_ratio'], .4)
        self.assertEqual(result['center_dx_ratio'], .25)
        self.assertEqual(result['center_dy_ratio'], 0)
        self.assertTrue(result['touches_crop_boundary'])
        self.assertEqual(result['min_normalized_boundary_distance'], 0)

    def test_oracle_padding_matches_pilot_tensor_coordinates_and_remap(self):
        traces = []
        class Dummy(torch.nn.Module):
            def forward(self, tensor, target=None):
                traces.append({'selected_L_R': 1, 'feature_shapes': [list(tensor.shape[-2:])]})
                return None, [{'boxes': torch.tensor([[0., 0., 1., 1.]]),
                               'scores': torch.tensor([.9]), 'labels': torch.tensor([1])}]
        tensor = torch.zeros(1, 3, 320, 320)
        dets, rois, own = oracle(Dummy(), tensor, {'car': [[200, 200, 300, 300]]},
                                 {1: 'car'}, (640, 640), 32, .5, traces)
        self.assertEqual(rois, [[136, 136, 364, 364]])
        self.assertEqual(dets[0]['bbox'], rois[0])
        self.assertEqual(own[0]['network_crop_h'], 114)
        self.assertEqual(own[0]['center_distance_ratio'], 0)
        # A second current-frame GT never inherits any previous proposal.
        _, moved, _ = oracle(Dummy(), tensor, {'car': [[300, 300, 400, 400]]},
                             {1: 'car'}, (640, 640), 32, .5, traces)
        self.assertEqual(moved, [[236, 236, 464, 464]])

    def test_ratio_bins_are_disjoint_and_keep_empty_bins(self):
        row = {'gt_area_ratio': .95, 'gt_max_side_ratio': .95, 'center_distance_ratio': .1,
               'matched50_after_global_nms': False, 'best_prediction_iou': .4,
               'own_crop_max_confidence': .6}
        bins = object_bins([row])
        for feature in ('gt_area_ratio', 'gt_max_side_ratio', 'center_distance_ratio'):
            selected = [b for b in bins if b['feature'] == feature]
            self.assertEqual(sum(b['N_GT'] for b in selected), 1)
            self.assertTrue(any(b['N_GT'] == 0 and b['Recall50_micro'] is None for b in selected))


if __name__ == '__main__':
    unittest.main()
