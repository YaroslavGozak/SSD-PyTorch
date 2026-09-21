import unittest

import numpy as np
import torch

from experiments.cost_model_validation.adapters import FakeAdapter
from experiments.cost_model_validation.geometry import (
    Rectangle, effective_area, stride_rounded_shape, union_rectangle,
)
from experiments.cost_model_validation.timing import measure
from experiments.cost_model_validation.models import (
    CalibrationEnvelope, PolynomialLatencyModel, decide_merge, fit_linear_model, merge_decision,
)
from experiments.cost_model_validation.collect_experiment_b import balanced_orders, generate_pairs
from experiments.cost_model_validation.common import bootstrap_ci


class CoreTests(unittest.TestCase):
    def test_roissd_adapter_uses_training_normalization(self):
        from experiments.cost_model_validation.roissd_adapter import RoiSSDAdapter
        adapter = RoiSSDAdapter.__new__(RoiSSDAdapter)
        adapter.device = torch.device("cpu")
        adapter.stride = 32
        prepared = adapter.prepare(np.zeros((32,32,3),dtype=np.uint8),(32,32))
        expected = -torch.tensor(RoiSSDAdapter.IMAGENET_MEAN) / torch.tensor(RoiSSDAdapter.IMAGENET_STD)
        self.assertTrue(torch.allclose(prepared.value[0,:,0,0],expected))
        self.assertEqual(adapter.preprocessing_metadata()["normalization"],"divide_by_255_then_imagenet")

    def test_union_and_areas(self):
        first = Rectangle(2, 3, 12, 13)
        second = Rectangle(8, 5, 20, 17)
        self.assertEqual(union_rectangle(first, second), Rectangle(2, 3, 20, 17))
        self.assertEqual(first.area, 100)
        self.assertEqual(effective_area((32, 64)), 2048)

    def test_stride_rounding_and_fake_adapter_read_actual_shape(self):
        self.assertEqual(stride_rounded_shape(33, 65, 32), (64, 96))
        adapter = FakeAdapter(stride=32)
        prepared, result = measure(adapter, None, (33, 65), "inference_only")
        self.assertEqual(prepared.tensor_hw, (64, 96))
        self.assertEqual(prepared.value.shape[-2:], (64, 96))
        self.assertGreaterEqual(result.inference_ms, 0.0)

    def test_timing_mode_is_explicit(self):
        with self.assertRaises(ValueError):
            measure(FakeAdapter(), None, (32, 32), "mixed")

    def test_linear_fit_and_merge_decision(self):
        fit = fit_linear_model([1, 2, 3, 4], [0.5, 0.7, 0.9, 1.1])
        self.assertAlmostEqual(fit.coefficients["b0"], 0.3, places=6)
        self.assertAlmostEqual(fit.coefficients["b1"], 0.2, places=6)
        self.assertTrue(merge_decision(9, 10))
        self.assertFalse(merge_decision(10, 10))

    def test_pair_generator_has_boundary_examples(self):
        pairs = generate_pairs(20, (640, 640), 100.0, 3)
        self.assertEqual(len(pairs), 20)
        self.assertGreaterEqual(sum(p["boundary_bin"] in {"near_low", "near_high"} for p in pairs), 8)

    def test_pair_generator_handles_large_tau_on_320_canvas(self):
        pairs = generate_pairs(20, (320, 320), 62825.29269413853, 20260913)
        near = [pair for pair in pairs if pair["boundary_bin"] in {"near_low", "near_high"}]
        self.assertGreaterEqual(len(near), 8)
        self.assertTrue(all(0 <= pair["first"].x1 < pair["first"].x2 <= 320 for pair in near))

    def test_experiment_b_orders_are_exactly_balanced(self):
        orders = balanced_orders(20, 20260913)
        self.assertEqual(orders.count("separate_first"), 10)
        self.assertEqual(orders.count("merged_first"), 10)

    def test_bootstrap_ci_uses_replicate_quantiles(self):
        ci = bootstrap_ci([0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(ci, [0.1, 3.9])

    def test_linear_direct_cost_matches_tau_and_strict_tie(self):
        envelope = CalibrationEnvelope(1, 1_000_000, 1, 1000, 1, 1000, .1, 10.0)
        model = PolynomialLatencyModel("linear", {"b0": 10.0, "b1": 2.0}, envelope)
        self.assertEqual(decide_merge(model, (2, 2), (2, 2), (2, 3))["predicted_merge"], True)
        self.assertEqual(decide_merge(model, (2, 2), (2, 2), (1, 13))["predicted_merge"], False)

    def test_piecewise_formula_is_continuous_at_breakpoint(self):
        envelope = CalibrationEnvelope(1, 1_000_000, 1, 1000, 1, 1000, .1, 10.0)
        model = PolynomialLatencyModel("piecewise", {"b0": 1.0, "b1": 2.0, "b2": 3.0}, envelope, 10)
        self.assertAlmostEqual(model.predict_seconds(10), 21.0)
        self.assertAlmostEqual(model.predict_seconds(11), 26.0)


if __name__ == "__main__":
    unittest.main()
