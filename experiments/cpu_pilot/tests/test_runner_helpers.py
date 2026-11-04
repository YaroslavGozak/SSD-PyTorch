import unittest
import json
from unittest.mock import patch

import torch
import numpy as np
from unittest.mock import Mock
from tools.helpers.pipeline import process_frame

from experiments.cpu_pilot.run_pilot import (
    AffineCostMerger,
    _effective_shape_factory,
    run_pipeline_modes,
)


class PilotHelperTests(unittest.TestCase):
    def test_process_frame_supplies_aligned_shapes_to_affine_merger(self):
        merger = AffineCostMerger({"K_ms": 1.0, "c_ms_per_pixel": 0.01})
        tracker = Mock()
        tracker.update.return_value = {"rois": []}
        with patch("tools.helpers.pipeline.run_model_inference", return_value=(None, [torch.empty((0, 6))])), patch(
            "tools.helpers.pipeline.tensor_to_detection_list", return_value=[]
        ):
            result = process_frame(
                model=Mock(), idx2label={},
                frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
                im_tensor=torch.zeros((1, 3, 320, 320)), tracker=tracker,
                next_frame_rois=[[160, 120, 240, 180], [192, 120, 272, 180]],
                frame_idx=1, key_frame_interval=10, im_size_hw=(320, 320),
                conf_threshold=0.5, nms_iou=0.45, merge_fn=merger,
                model_device=torch.device("cpu"), roi_grid=32,
            )
        self.assertFalse(result.use_full_frame)
        self.assertTrue(result.merge_decisions)
        self.assertEqual(result.merge_decisions[0]["first_shape_hw"], (64, 64))
        self.assertEqual(result.merge_count, 1)

    def test_pipeline_stops_when_either_model_fails_coverage_despite_passing_accuracy(self):
        for rejected in ("yolo26n", "roi_ssd_mobilenet"):
            with self.subTest(model=rejected):
                decisions = {
                    name: {"criterion_met": name != rejected, "accuracy_gate_passed": True}
                    for name in ("yolo26n", "roi_ssd_mobilenet")
                }
                with patch("pathlib.Path.is_file", return_value=True), patch(
                    "pathlib.Path.read_text", side_effect=[json.dumps(decisions), "{}"]
                ), patch("experiments.cpu_pilot.run_pilot._load_model") as load:
                    with self.assertRaisesRegex(RuntimeError, "STOP: " + rejected):
                        run_pipeline_modes({}, [], torch.device("cpu"))
                    load.assert_not_called()

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
