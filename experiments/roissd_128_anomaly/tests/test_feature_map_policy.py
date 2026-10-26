import unittest

import torch
import numpy as np

from experiments.roissd_128_anomaly.feature_map_analysis import (
    correlation, describe_group, in_subset, ranks, ratio_bins,
)
from experiments.roissd_128_anomaly.feature_map_policy import (
    legacy_depth, localization, metric_functions, object_geometry, object_pairs,
    reference_depth, shapes,
)


class FeaturePolicyTests(unittest.TestCase):
    def test_subsets_and_undefined_statistics_do_not_silently_impute(self):
        record = dict(upper_adds_1x1='True', all_added_heads_1x1='True', num_added_heads='2')
        self.assertTrue(in_subset(record, 'all_pairs'))
        self.assertFalse(in_subset(record, 'single_added_1x1'))
        self.assertTrue(in_subset(record, 'all_added_heads_1x1'))
        with self.assertRaises(ValueError):
            in_subset(record, 'unknown')
        self.assertEqual(describe_group([])['status'], 'EMPTY')
        self.assertIsNone(correlation(np.array([1., 1., 1.]), np.array([1., 2., 3.])))
        np.testing.assert_equal(ranks(np.array([5, 5, 2, 9])), np.array([2.5, 2.5, 1., 4.]))

    def test_ratio_bins_are_half_open_and_recall_uses_all_gt(self):
        records = []
        for index, ratio in enumerate((.1, .25, .8, .95, 1.)):
            record = dict(pair_kind='stage2d_adjacent', input_h='128', input_w='128',
                          sample_id=str(index), upper_adds_1x1='True',
                          all_added_heads_1x1='True', num_added_heads='1',
                          gt_area_ratio=str(ratio), gt_width_ratio=str(ratio),
                          gt_height_ratio=str(ratio), gt_max_side_ratio=str(ratio))
            for name in ('success50', 'max_class_confidence', 'best_predicted_iou',
                         'best_anchor_iou', 'crop_raw_max_confidence'):
                record.update({name + '_lower': '0', name + '_upper': '1', 'delta_' + name: '1'})
            records.append(record)
        bins = ratio_bins(records)
        selected = [r for r in bins if r['pair_subset'] == 'single_added_1x1'
                    and r['ratio'] == 'gt_area_ratio']
        self.assertEqual(sum(r['N_GT'] for r in selected if r['bin'] not in ('<0.8', '>=0.8')), 5)
        self.assertEqual(next(r for r in selected if r['bin'] == '>=0.8')['N_GT'], 3)
        self.assertEqual(next(r for r in selected if r['bin'] == '>=0.95')['N_GT'], 2)
        self.assertEqual(next(r for r in selected if r['bin'] == '0.50-0.70')['status'], 'EMPTY')

    def test_legacy_is_fixed_independent_of_current_source_thresholds(self):
        self.assertEqual([legacy_depth(n) for n in (32, 33, 64, 65, 96, 97, 140, 141, 268, 269)],
                         [1, 2, 2, 3, 3, 4, 4, 5, 5, 6])

    def test_all_requested_shapes_are_unique(self):
        self.assertEqual(len(shapes()), 36)
        self.assertEqual(len(set(shapes())), 36)
        for height, width in shapes():
            if height != width:
                self.assertIn((width, height), shapes())

    def test_reference_uses_declared_tolerance_and_recall_gain(self):
        lower = dict(forced_L_R=5, map50=.60, recall50=.61)
        self.assertEqual(reference_depth(lower, dict(forced_L_R=6, map50=.602, recall50=.614)), 5)
        self.assertEqual(reference_depth(lower, dict(forced_L_R=6, map50=.603, recall50=.618)), 6)
        self.assertEqual(reference_depth(lower, dict(forced_L_R=6, map50=.58, recall50=.63)), 5)

    def test_per_gt_geometry_uses_actual_rectangular_dimensions(self):
        gt = torch.tensor([[.1, .2, .9, .8], [.2, .3, .4, .5]])
        records = object_geometry(gt, torch.tensor([1, 2]), 7, 100, 200)
        self.assertEqual(len(records), 2)
        first = records[0]
        self.assertAlmostEqual(first['gt_w'], 160, places=4)
        self.assertAlmostEqual(first['gt_h'], 60, places=4)
        self.assertAlmostEqual(first['gt_area_ratio'], .48, places=5)
        self.assertAlmostEqual(first['gt_max_side_ratio'], .8, places=5)
        self.assertEqual(first['crop_area'], 20000)

    def test_localization_is_class_correct_and_one_to_one_like_evaluator(self):
        _, get_iou = metric_functions()
        gt = torch.tensor([[.1, .1, .9, .9], [.1, .1, .9, .9], [.0, .0, .2, .2]])
        labels = torch.tensor([1, 1, 2])
        det = dict(boxes=torch.tensor([[.1, .1, .9, .9], [.1, .1, .9, .9], [.0, .0, .2, .2]]),
                   labels=torch.tensor([1, 1, 3]), scores=torch.tensor([.9, .8, .99]))
        result = localization(det, gt, labels, get_iou)
        self.assertEqual([r['success50'] for r in result], [1, 0, 0])
        self.assertGreater(result[1]['best_predicted_iou'], .99)
        self.assertEqual(result[2]['best_predicted_iou'], 0)
        self.assertAlmostEqual(result[0]['max_class_confidence'], .9, places=6)

    def test_pair_distinguishes_single_and_repeated_coarse_heads(self):
        geo = [dict(sample_id=0, gt_index=0)]
        low = dict(sample_id=0, gt_index=0, success50=0, best_predicted_iou=.2,
                   max_class_confidence=.1, best_localized_confidence=.1,
                   best_anchor_iou=.3, crop_raw_max_confidence=.4)
        high = {**low, 'success50': 1}
        layouts = {3: dict(feature_map_shapes=[[1, 2, 2, 2]] * 3),
                   4: dict(feature_map_shapes=[[1, 2, 2, 2]] * 3 + [[1, 2, 1, 1]]),
                   6: dict(feature_map_shapes=[[1, 2, 2, 2]] * 3 + [[1, 2, 1, 1]] * 3)}
        pair = list(object_pairs('test', 3, 4, geo, {3: [low], 4: [high]}, layouts, 'test'))[0]
        self.assertTrue(pair['upper_adds_1x1'])
        self.assertEqual(pair['num_added_heads'], 1)
        self.assertEqual(pair['delta_success50'], 1)
        repeated = list(object_pairs('test', 4, 6, geo, {4: [low], 6: [high]}, layouts, 'test'))[0]
        self.assertEqual(repeated['num_added_heads'], 2)
        self.assertTrue(repeated['all_added_heads_1x1'])


if __name__ == '__main__':
    unittest.main()
