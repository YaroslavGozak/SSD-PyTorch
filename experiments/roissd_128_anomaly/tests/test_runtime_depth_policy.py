import unittest

import torch
import torch.nn as nn

from model.roissd_mobilenet import RoiSSDMobileNet


def stride_two(in_channels=3, out_channels=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
    )


def geometry_model(trace=True):
    model = RoiSSDMobileNet.__new__(RoiSSDMobileNet)
    nn.Module.__init__(model)
    model.record_execution_trace = trace
    model.last_execution_trace = None
    model.features_stage1 = nn.Sequential(*[stride_two() for _ in range(4)])
    model.conv_to_672 = nn.Identity()
    model.features_stage2 = stride_two()
    model.conv8_2 = stride_two()
    model.conv9_2 = stride_two()
    model.conv10_2 = stride_two()
    model.conv11_2 = stride_two()
    return model


class RuntimeDepthPolicyTests(unittest.TestCase):
    def test_runtime_depth_selection_does_not_use_legacy_threshold_helper(self):
        model = geometry_model()
        if hasattr(model, 'get_max_feature_layer_by_roi_size'):
            model.get_max_feature_layer_by_roi_size = lambda _: self.fail(
                'Forward depth selection consulted the legacy threshold helper')
        with torch.inference_mode():
            features, depth = model._build_feature_levels(torch.zeros(1, 3, 129, 129))
        self.assertEqual(depth, 5)
        self.assertEqual(tuple(features[-1].shape[-2:]), (1, 1))

    def test_dynamic_policy_tracks_actual_square_and_rectangle_shapes(self):
        model = geometry_model()
        expected = {
            (88, 88): (4, [[6, 6], [3, 3], [2, 2], [1, 1]]),
            (96, 96): (4, [[6, 6], [3, 3], [2, 2], [1, 1]]),
            (104, 104): (4, [[7, 7], [4, 4], [2, 2], [1, 1]]),
            (124, 124): (4, [[8, 8], [4, 4], [2, 2], [1, 1]]),
            (128, 128): (4, [[8, 8], [4, 4], [2, 2], [1, 1]]),
            (129, 129): (5, [[9, 9], [5, 5], [3, 3], [2, 2], [1, 1]]),
            (140, 140): (5, [[9, 9], [5, 5], [3, 3], [2, 2], [1, 1]]),
            (144, 144): (5, [[9, 9], [5, 5], [3, 3], [2, 2], [1, 1]]),
            (128, 160): (5, [[8, 10], [4, 5], [2, 3], [1, 2], [1, 1]]),
            (256, 256): (5, [[16, 16], [8, 8], [4, 4], [2, 2], [1, 1]]),
            (256, 320): (6, [[16, 20], [8, 10], [4, 5], [2, 3], [1, 2], [1, 1]]),
        }
        with torch.inference_mode():
            for shape, (depth, feature_shapes) in expected.items():
                model.last_execution_trace = None
                outputs, selected = model._build_feature_levels(torch.zeros(1, 3, *shape))
                log = model.get_runtime_log()
                self.assertEqual(selected, depth)
                self.assertEqual([list(feature.shape[-2:]) for feature in outputs], feature_shapes)
                self.assertEqual(log['depth_policy'], 'until_1x1')
                self.assertEqual(log['selected_L_R'], depth)
                self.assertEqual(log['active_heads'], list(range(depth)))
                self.assertEqual(log['final_feature_shape'], [1, 1])
                self.assertTrue(log['stopped_at_1x1'])

    def test_blocks_beyond_first_1x1_are_not_executed(self):
        model = geometry_model()
        counts = {name: 0 for name in ('features_stage2', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2')}
        handles = [getattr(model, name).register_forward_hook(
            lambda module, inputs, output, key=name: counts.__setitem__(key, counts[key] + 1))
                   for name in counts]
        try:
            with torch.inference_mode():
                model._build_feature_levels(torch.zeros(1, 3, 128, 128))
                self.assertEqual(counts, dict(features_stage2=1, conv8_2=1, conv9_2=1,
                                              conv10_2=0, conv11_2=0))
                counts.update({name: 0 for name in counts})
                model._build_feature_levels(torch.zeros(1, 3, 128, 160))
                self.assertEqual(counts, dict(features_stage2=1, conv8_2=1, conv9_2=1,
                                              conv10_2=1, conv11_2=0))
        finally:
            for handle in handles:
                handle.remove()

    def test_actual_next_block_geometry_and_invalid_size(self):
        block = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
        self.assertEqual(RoiSSDMobileNet._next_feature_spatial_shape(block, (2, 3)), (1, 2))
        self.assertEqual(RoiSSDMobileNet._next_feature_spatial_shape(block, (1, 1)), (1, 1))
        invalid = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0)
        self.assertEqual(RoiSSDMobileNet._next_feature_spatial_shape(invalid, (2, 2)), (0, 0))

    def test_feature_iterator_advances_one_block_after_each_level(self):
        model = geometry_model()
        events = []
        for name in ('features_stage1', 'conv_to_672', 'features_stage2',
                     'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2'):
            getattr(model, name).register_forward_hook(
                lambda module, inputs, output, key=name: events.append(key))
        with torch.inference_mode():
            for index, _ in model._iter_feature_levels(torch.zeros(1, 3, 128, 128)):
                events.append(f'head_{index}')
        self.assertEqual(events, [
            'features_stage1', 'conv_to_672', 'head_0',
            'features_stage2', 'head_1',
            'conv8_2', 'head_2',
            'conv9_2', 'head_3',
        ])

    def test_runtime_trace_is_optional_and_contains_policy_fields(self):
        model = geometry_model(trace=False)
        with torch.inference_mode():
            model._build_feature_levels(torch.zeros(1, 3, 128, 128))
        self.assertIsNone(model.get_runtime_log())
        model.record_execution_trace = True
        with torch.inference_mode():
            model._build_feature_levels(torch.zeros(1, 3, 128, 160))
        log = model.get_runtime_log()
        for key in ('depth_policy', 'selected_L_R', 'active_heads', 'feature_shapes',
                    'final_feature_shape', 'next_feature_shape', 'next_head_feasible'):
            self.assertIn(key, log)
        self.assertEqual(log['selected_L_R'], 5)

    def test_runtime_policy_is_fixed_to_until_1x1(self):
        model = geometry_model()
        self.assertFalse(hasattr(model, 'set_depth_policy'))
        with torch.inference_mode():
            model._build_feature_levels(torch.zeros(1, 3, 128, 128))
        self.assertEqual(model.get_runtime_log()['depth_policy'], 'until_1x1')


if __name__ == '__main__':
    unittest.main()
