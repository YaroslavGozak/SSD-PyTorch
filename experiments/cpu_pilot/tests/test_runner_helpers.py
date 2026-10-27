import unittest

import torch

from experiments.cpu_pilot.run_pilot import (
    AffineCostMerger,
    _effective_shape_factory,
)


class PilotHelperTests(unittest.TestCase):
    def test_effective_shape_uses_source_to_model_mapping_and_roi_grid(self):
        image_tensor = torch.zeros((1, 3, 320, 320))
        source_shape = (480, 640)

        unaligned = _effective_shape_factory(image_tensor, source_shape, 1)
        aligned = _effective_shape_factory(image_tensor, source_shape, 32)

        self.assertEqual(unaligned([80, 80, 160, 160]), (80, 80))
        self.assertEqual(aligned([80, 80, 160, 160]), (96, 96))

    def test_affine_cost_merger_merges_when_union_cost_is_lower(self):
        merger = AffineCostMerger({"K_ms": 1.0, "c_ms_per_pixel": 0.01})

        clusters = merger(
            [[0, 0, 32, 32], [16, 0, 48, 32]],
            effective_shape=lambda roi: (roi[3] - roi[1], roi[2] - roi[0]),
        )

        self.assertEqual(clusters, [[0, 0, 48, 32]])
        self.assertTrue(any(decision["merged"] for decision in merger.decisions))


if __name__ == "__main__":
    unittest.main()
