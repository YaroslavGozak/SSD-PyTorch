import unittest

import torch

from experiments.roissd_128_anomaly.forced_depth import (
    check_baseline,
    selected_depth,
    spatially_impossible,
)


class Detector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_heads = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])

    def get_max_feature_layer_by_roi_size(self, size):
        return 4 if size <= 140 else 5


class ForcedDepthTests(unittest.TestCase):
    def test_current_does_not_override_policy(self):
        model = Detector()
        with selected_depth(model, None):
            self.assertNotIn('get_max_feature_layer_by_roi_size', model.__dict__)
            self.assertEqual(model.get_max_feature_layer_by_roi_size(128), 4)
            self.assertEqual(model.get_max_feature_layer_by_roi_size(144), 5)

    def test_forced_depth_is_restored_after_success_and_exception(self):
        model = Detector()
        for depth in (3, 5, 6):
            with selected_depth(model, depth):
                self.assertEqual(model.get_max_feature_layer_by_roi_size(128), depth)
            self.assertEqual(model.get_max_feature_layer_by_roi_size(128), 4)
            self.assertNotIn('get_max_feature_layer_by_roi_size', model.__dict__)
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with selected_depth(model, 5):
                raise RuntimeError('test failure')
        self.assertEqual(model.get_max_feature_layer_by_roi_size(144), 5)
        self.assertNotIn('get_max_feature_layer_by_roi_size', model.__dict__)

    def test_invalid_depth_does_not_change_architecture(self):
        model = Detector()
        for depth in (0, 7):
            with self.assertRaisesRegex(ValueError, 'outside'):
                with selected_depth(model, depth):
                    self.fail('Invalid depth was accepted')
        self.assertEqual(len(model.cls_heads), 6)

    def test_only_spatial_failures_are_feasibility_results(self):
        self.assertTrue(spatially_impossible(RuntimeError(
            "Kernel size can't be greater than actual input size")))
        self.assertTrue(spatially_impossible(RuntimeError('Output size is too small')))
        self.assertFalse(spatially_impossible(RuntimeError('CUDA out of memory')))
        self.assertFalse(spatially_impossible(RuntimeError('Box-slot mismatch')))

    def test_baseline_check_rejects_metric_and_gt_drift(self):
        result = dict(map50=.4, recall50=.5, mean_raw_max_confidence=.8,
                      mean_best_anchor_iou=.6, fraction_gt_best_anchor_iou_ge_05=.7,
                      num_gt=330)
        metrics = dict(map50='.4', recall50='.5', max_confidence_mean='.8', num_gt='330')
        anchors = dict(max_anchor_iou_mean='.6', fraction_gt_iou_ge_05='.7')
        check_baseline(result, metrics, anchors)
        for changed in (dict(map50=.41), dict(num_gt=329)):
            with self.assertRaisesRegex(AssertionError, 'Stage-2 baseline mismatch'):
                check_baseline({**result, **changed}, metrics, anchors)


if __name__ == '__main__':
    unittest.main()
