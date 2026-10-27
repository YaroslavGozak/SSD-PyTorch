"""VGG-specific activation routing and pool-aware early-exit checks."""
import copy
import unittest
from unittest.mock import patch

import torch
import torchvision

from model.roissd import RoiSSD
from tools.helpers.config_reader import load_config
from experiments.roissd_128_anomaly.run import DATA


class VGG16RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        params = load_config(str(DATA / 'configs/roissd.yaml'))['model_params']
        cls.params = copy.deepcopy(params)
        vgg = torchvision.models.vgg16
        with patch('torchvision.models.vgg16', side_effect=lambda **_: vgg(weights=None)):
            cls.model = RoiSSD(params, num_classes=31, record_execution_trace=True).eval()
        cls.after = params

    def test_constructor_preserves_config_and_initializes_trace(self):
        self.assertEqual(self.params, self.after)
        # Fresh models have no state-dict entries for diagnostic attributes.
        self.assertNotIn('record_execution_trace', self.model.state_dict())
        self.model.last_execution_trace = None
        self.assertIsNone(self.model.get_runtime_log())

    def test_pool_geometry_matches_pytorch(self):
        for ceil in (False, True):
            block = torch.nn.Sequential(torch.nn.MaxPool2d(3, 2, 1, ceil_mode=ceil),
                                        torch.nn.Conv2d(3, 3, 3, padding=1))
            for shape in ((8, 11), (9, 12), (16, 20)):
                actual = block(torch.zeros(1, 3, *shape)).shape[-2:]
                self.assertEqual(RoiSSD._next_feature_spatial_shape(block, shape), tuple(actual))

    def test_optional_trace_does_not_change_predictions(self):
        model = self.model
        tensor = torch.randn(1, 3, 64, 64)
        try:
            with torch.inference_mode():
                model.record_execution_trace = False
                _, plain = model(tensor)
                self.assertIsNone(model.get_runtime_log())
                model.record_execution_trace = True
                _, traced = model(tensor)
            self.assertIsNotNone(model.get_runtime_log())
            for key in plain[0]:
                torch.testing.assert_close(plain[0][key], traced[0][key], rtol=0, atol=0)
        finally:
            model.record_execution_trace = True

    def test_activation_routing_head_order_and_no_late_execution(self):
        model = self.model
        events, raw, first_head, branch_input = [], [], [], []
        handles = [model.features.register_forward_hook(lambda m, a, out: raw.append(out.clone())),
                   model.cls_heads[0].register_forward_pre_hook(lambda m, a: first_head.append(a[0].clone())),
                   model.conv5_3_fc.register_forward_pre_hook(lambda m, a: branch_input.append(a[0].clone()))]
        names = ['features', 'conv5_3_fc', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
        for name in names:
            handles.append(getattr(model, name).register_forward_hook(
                lambda m, a, out, name=name: events.append(name)))
        for kind, heads in [('cls', model.cls_heads), ('loc', model.bbox_reg_heads)]:
            for index, head in enumerate(heads):
                handles.append(head.register_forward_hook(
                    lambda m, a, out, k=kind, i=index: events.append(f'{k}_{i}')))
        try:
            with torch.inference_mode():
                model(torch.randn(1, 3, 128, 160))
            trace = model.get_runtime_log()
            expected = []
            for index, name in enumerate(names[:trace['selected_L_R']]):
                expected.extend([name, f'cls_{index}', f'loc_{index}'])
            self.assertEqual(events, expected)
            torch.testing.assert_close(branch_input[0], raw[0], rtol=0, atol=0)
            expected_first = model.scale_weight.view(1, -1, 1, 1) * torch.nn.functional.normalize(
                raw[0], p=2, dim=1, eps=1e-12)
            torch.testing.assert_close(first_head[0], expected_first, rtol=0, atol=0)
            self.assertEqual(trace['N_loc_predictions'], trace['N_cls_predictions'])
            self.assertEqual(trace['N_loc_predictions'], trace['N_anchors'])
        finally:
            for handle in handles:
                handle.remove()


if __name__ == '__main__':
    unittest.main()
