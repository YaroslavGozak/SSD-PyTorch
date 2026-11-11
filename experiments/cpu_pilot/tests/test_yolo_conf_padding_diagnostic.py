import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_yolo_conf_padding_diagnostic import (
    OPERATIONAL_CONFS,
    PADDINGS,
    _coverage_for_gt,
    _gate_matrix,
    _output_paths,
    _recommend,
    _temporal_ranges,
    _write_analysis,
    _write_csv,
)


class YoloConfidencePaddingDiagnosticTests(unittest.TestCase):
    def test_uncovered_gt_preserves_class_and_bbox_and_uses_half_area_rule(self):
        gt = {"car": [[0, 0, 10, 10], [20, 20, 30, 30]]}
        uncovered = _coverage_for_gt(gt, [[0, 0, 5, 10]], False)
        self.assertEqual(uncovered, [{"class": "car", "bbox": [20, 20, 30, 30]}])
        self.assertEqual(_coverage_for_gt(gt, [], True), [])

    def test_temporal_ranges_include_consecutive_miss_frames_by_video(self):
        rows = [
            {"video_id": "00023010", "frame_id": 83, "uncovered_gt_count": 1},
            {"video_id": "00023010", "frame_id": 84, "uncovered_gt_count": 2},
            {"video_id": "00023010", "frame_id": 86, "uncovered_gt_count": 1},
            {"video_id": "00007008", "frame_id": 2, "uncovered_gt_count": 1},
        ]
        self.assertEqual(
            _temporal_ranges(rows),
            [
                "video 7008 frames 2-2: 1 consecutive frame(s), 1 missed GT",
                "video 23010 frames 83-84: 2 consecutive frame(s), 3 missed GT",
                "video 23010 frames 86-86: 1 consecutive frame(s), 1 missed GT",
            ],
        )

    def test_gate_matrix_has_all_30_cells_and_recommendation_filters_failed_gates(self):
        rows = []
        for confidence in OPERATIONAL_CONFS:
            for padding in PADDINGS:
                passed = confidence == 0.05 and padding == 96
                rows.append({
                    "operational_conf": confidence,
                    "padding": padding,
                    "FINAL_GATE_PASS": passed,
                    "tensor_area_ratio_mean": 0.20 if padding == 96 else 0.1,
                    "candidate_segments_per_frame": 2.0,
                })
        matrix = _gate_matrix(rows)
        self.assertEqual(len(matrix), 5)
        self.assertTrue(all(len(row) == 7 for row in matrix))
        self.assertEqual(_recommend(rows)["operational_conf"], 0.05)
        self.assertEqual(_recommend(rows)["padding"], 96)

    def test_analysis_writes_summary_matrix_and_report_from_csv_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = _output_paths(Path(directory))
            paths["results"].mkdir()
            paths["analysis"].mkdir()
            paths["baseline"].write_text(json.dumps({"mAP50": 0.8}), encoding="utf-8")
            sweep = []
            frames = []
            for confidence in OPERATIONAL_CONFS:
                for padding in PADDINGS:
                    sweep.append({
                        "operational_conf": confidence,
                        "padding": padding,
                        "mAP50": 0.8,
                        "mAP50_95": 0.5,
                        "recall50": 0.9,
                        "mAP50_retention": 1.0,
                        "gt_roi_coverage_mean": 1.0,
                        "gt_roi_coverage_p5": 1.0,
                        "evaluation_detections_per_frame": 2.0,
                        "operational_detections_per_frame": 1.0,
                        "candidate_segments_per_frame": 1.0,
                        "next_candidate_segments_per_frame": 1.0,
                        "nn_calls_per_frame": 1.0,
                        "processed_area_ratio_mean": 0.2,
                        "tensor_area_ratio_mean": 0.1,
                        "full_frame_fraction": 0.1,
                        "MAP_GATE_PASS": True,
                        "COVERAGE_GATE_PASS": True,
                        "FINAL_GATE_PASS": True,
                    })
                    frames.extend({
                        "video_id": "00007008",
                        "frame_id": index,
                        "operational_conf": confidence,
                        "padding": padding,
                        "gt_count": 1,
                        "evaluation_detection_count": 2,
                        "operational_detection_count": 1,
                        "candidate_segment_count": 1,
                        "next_candidate_segment_count": 1,
                        "covered_gt_count": 1,
                        "uncovered_gt_count": 0,
                        "uncovered_gt": "[]",
                        "full_frame_fallback": False,
                        "fallback_reason": "",
                    } for index in range(320))
            _write_csv(paths["sweep"], sweep)
            _write_csv(paths["frames"], frames)

            _write_analysis(paths, sweep, frames)

            with paths["gate_matrix"].open(encoding="utf-8", newline="") as stream:
                matrix = list(csv.DictReader(stream))
            self.assertEqual(len(matrix), 5)
            self.assertEqual(len(matrix[0]), 7)
            self.assertTrue(paths["summary"].is_file())
            self.assertIn("Gate matrix", paths["report"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
