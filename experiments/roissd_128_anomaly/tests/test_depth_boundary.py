import unittest

from experiments.roissd_128_anomaly.depth_boundary import (
    BOUNDARIES, CONTROLS, RECTANGLES, FIELDS, check_control, pair_deltas,
)


class BoundaryTests(unittest.TestCase):
    def test_exact_requested_shapes_include_both_orientations(self):
        self.assertEqual(BOUNDARIES, ((129, 129), (130, 130), (131, 131)))
        self.assertEqual(len(RECTANGLES), 10)
        self.assertEqual(len(set(RECTANGLES)), 10)
        for height, width in RECTANGLES:
            self.assertIn((width, height), RECTANGLES)
        self.assertEqual(sum(3 if shape == (144, 144) else 2
                             for shape in BOUNDARIES + RECTANGLES + CONTROLS), 37)

    def test_pair_deltas_use_depth_not_row_order(self):
        low = dict(actual_L_R=4, map50=.2, recall50=.3, num_gt=329, input_gt_sha256='same')
        high = dict(actual_L_R=5, map50=.8, recall50=.7, num_gt=329, input_gt_sha256='same')
        delta = pair_deltas([high, low, dict(actual_L_R=6)])
        self.assertAlmostEqual(delta['delta_map50_L5_minus_L4'], .6)
        self.assertAlmostEqual(delta['delta_recall50_L5_minus_L4'], .4)
        with self.assertRaises(AssertionError):
            pair_deltas([low, {**high, 'input_gt_sha256': 'different'}])
        with self.assertRaises(AssertionError):
            pair_deltas([low, {**high, 'num_gt': 330}])

    def test_controls_reject_metric_and_structure_changes(self):
        result = {key: .5 for key in FIELDS}
        result.update(active_heads=[0], feature_map_shapes=[[1, 672, 8, 8]],
                      anchors_per_head=[256])
        reference = {key: str(value) for key, value in result.items()}
        check_control(result, reference)
        with self.assertRaisesRegex(AssertionError, 'map50'):
            check_control({**result, 'map50': .6}, reference)
        with self.assertRaisesRegex(AssertionError, 'structure changed'):
            check_control({**result, 'feature_map_shapes': [[1, 672, 9, 8]]}, reference)


if __name__ == '__main__':
    unittest.main()
