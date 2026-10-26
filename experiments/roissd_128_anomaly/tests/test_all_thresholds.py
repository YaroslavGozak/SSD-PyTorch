import unittest

import torch

from experiments.roissd_128_anomaly.all_thresholds import (
    OFFSETS, SPECS, SlotGuard, plan, round_pixel,
)
from experiments.roissd_128_anomaly.threshold_evidence import paired_rows


class AllThresholdTests(unittest.TestCase):
    def test_plan_covers_every_square_and_rectangular_neighbor(self):
        jobs = plan()
        keys = [(job['transition'], job['input_h'], job['input_w']) for job in jobs]
        self.assertEqual(len(keys), len(set(keys)))
        for boundary, lower, extra, minima in SPECS:
            selected = [job for job in jobs if job['old_boundary'] == boundary]
            shapes = {(job['input_h'], job['input_w']) for job in selected}
            for size in {boundary + offset for offset in OFFSETS} | set(extra):
                self.assertIn((size, size), shapes)
            for minimum in minima:
                for ratio in (1.25, 1.5):
                    longer = round_pixel(minimum * ratio)
                    self.assertIn((minimum, longer), shapes)
                    self.assertIn((longer, minimum), shapes)
            self.assertTrue(all(job['lower_depth'] == lower
                                and job['upper_depth'] == lower + 1 for job in selected))

    def test_rounding_is_nearest_pixel_not_stride_or_bankers(self):
        self.assertEqual(round_pixel(32 * 1.25), 40)
        self.assertEqual(round_pixel(31 * 1.5), 47)
        self.assertEqual(round_pixel(33 * 1.5), 50)

    def test_persisted_pairs_require_same_inputs_and_adjacent_depths(self):
        low = dict(transition='L1->L2', input_h='32', input_w='40',
                   forced_L_R='1', input_gt_sha256='same')
        high = {**low, 'forced_L_R': '2'}
        pairs = paired_rows([high, low])
        self.assertEqual(pairs['L1->L2', 32, 40], [low, high])
        for bad in ({**high, 'forced_L_R': '3'}, {**high, 'input_gt_sha256': 'other'}):
            with self.assertRaises(AssertionError):
                paired_rows([low, bad])
        with self.assertRaisesRegex(AssertionError, 'Missing adjacent-depth pair'):
            paired_rows([low])

    def test_slot_guard_resets_each_forward_and_restores_on_mismatch(self):
        import model.roissd as architecture

        class Detector(torch.nn.Module):
            def __init__(self, wrong=False):
                super().__init__()
                self.num_classes = 31
                self.cls_heads = torch.nn.ModuleList([torch.nn.Conv2d(3, 62, 1)])
                self.bbox_reg_heads = torch.nn.ModuleList([torch.nn.Conv2d(3, 12 if wrong else 8, 1)])

            def forward(self, x):
                self.cls_heads[0](x)
                self.bbox_reg_heads[0](x)
                return architecture.generate_default_boxes([x], [[1.]], [.1, .2], x.shape[-2:])

        Detector.__module__ = 'model.roissd'
        original = architecture.generate_default_boxes
        good = Detector()
        with SlotGuard(good) as guard, torch.inference_mode():
            for _ in range(3):
                good(torch.zeros(1, 3, 4, 4))
            self.assertEqual(guard.checked_forwards, 3)
        self.assertIs(architecture.generate_default_boxes, original)
        self.assertFalse(good._forward_pre_hooks)
        bad = Detector(wrong=True)
        with self.assertRaisesRegex(AssertionError, 'Box-slot mismatch'):
            with SlotGuard(bad), torch.inference_mode():
                bad(torch.zeros(1, 3, 4, 4))
        self.assertIs(architecture.generate_default_boxes, original)
        self.assertFalse(bad.cls_heads[0]._forward_hooks)
        self.assertFalse(bad.bbox_reg_heads[0]._forward_hooks)
        self.assertFalse(bad._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()
