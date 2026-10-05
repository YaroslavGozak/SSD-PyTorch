import unittest
from unittest.mock import patch

from experiments.cost_model_validation.geometry import Rectangle
from experiments.cost_aware_merging.core import (DEFAULT_SHAPES, decide, generate_pairs,
                                                 nearest_shape_cost, pair_geometry,
                                                 policy_decisions, summarize)
from experiments.cost_aware_merging.run import _progress


class GeometryTests(unittest.TestCase):
    def test_progress_reports_counts_and_time_estimate(self):
        with self.assertLogs("experiments.cost_aware_merging.run", level="INFO") as captured, \
             patch("experiments.cost_aware_merging.run.time.monotonic", return_value=12):
            reported = _progress("Calibration", 2, 10, 2, 2, 5)
        self.assertEqual(reported, 12)
        self.assertIn("Calibration: 2/10", captured.output[0])
        self.assertIn("estimated remaining 40.0s", captured.output[0])

    def test_union_area_extra_and_overlap(self):
        values = pair_geometry(Rectangle(0,0,32,32),Rectangle(16,0,48,32))
        self.assertEqual(values["merged_area"],48*32)
        self.assertEqual(values["area_extra"],-16*32)
        self.assertAlmostEqual(values["iou"],1/3)

    def test_generation_is_reproducible_and_bounded(self):
        args = ((640,640),48,7)
        first,_ = generate_pairs(*args,tau=10000)
        second,_ = generate_pairs(*args,tau=10000)
        self.assertEqual(first,second)
        self.assertTrue(any(item[0][0].width==32 or item[0][0].height==32 for item in first))
        for (r1,r2),_,_ in first:
            for rect in (r1,r2):
                self.assertGreaterEqual(rect.x1,0)
                self.assertGreaterEqual(rect.y1,0)
                self.assertLessEqual(rect.x2,640)
                self.assertLessEqual(rect.y2,640)

    def test_impossible_shape_is_recorded(self):
        pairs,skips = generate_pairs((64,64),1,9,shapes=[(32,32),(320,320)],sides=[32],tau=100)
        self.assertEqual(len(pairs),1)
        self.assertGreater(skips["cannot_fit"],0)

    def test_cost_decisions_and_summary(self):
        self.assertTrue(decide(5,3,3))
        self.assertFalse(decide(6,3,3))
        cost,key = nearest_shape_cost({(32,256):10,(64,128):20},(32,256))
        self.assertEqual((cost,key),(10,(32,256)))
        decisions = policy_decisions(Rectangle(0,0,32,32),Rectangle(32,0,64,32),
                                     actual=(5,3,3),affine=(7,3,3),lookup=(5,3,3))
        self.assertTrue(decisions["oracle"])
        self.assertFalse(decisions["cost_affine"])
        self.assertTrue(decisions["cost_shape_lookup"])
        row = {"actual_separate_ms":6,"actual_merged_ms":5,"oracle_decision":True,
               **{f"{name}_decision":name != "no_merge" for name in
                  ("no_merge","existing_simple_merge","geometric_area","cost_affine","cost_shape_lookup","oracle")}}
        report = summarize([row])
        self.assertEqual(report[-1]["decision_accuracy"],1)
        self.assertEqual(report[0]["FN"],1)


if __name__ == "__main__":
    unittest.main()
