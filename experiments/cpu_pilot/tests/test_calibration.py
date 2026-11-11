import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.cpu_pilot.calibration import (
    COVERAGE_GATE,
    MAP_RETENTION_GATE,
    calibration_grid,
    select_configuration,
    validate_selected_config,
)


class JointCalibrationTests(unittest.TestCase):
    def test_primary_model_grids_match_stage3_protocol(self):
        yolo_conf, yolo_padding = calibration_grid(
            {
                "calibration": {
                    "operational_conf_values": [0.01, 0.05, 0.1, 0.2, 0.35],
                    "padding_values": [64, 96, 128, 160, 192, 224],
                }
            },
            "yolo26n",
            0.01,
        )
        self.assertEqual(yolo_conf, [0.01, 0.05, 0.1, 0.2, 0.35])
        self.assertEqual(yolo_padding, [64, 96, 128, 160, 192, 224])

        mobile_conf, mobile_padding = calibration_grid(
            {
                "calibration": {
                    "operational_conf_values": [0.01, 0.05, 0.1, 0.2, 0.25, 0.35, 0.5],
                    "padding_values": [64, 96, 112, 128, 144, 160, 192, 224],
                }
            },
            "roi_ssd_mobilenet",
            0.01,
        )
        self.assertEqual(mobile_conf, [0.01, 0.05, 0.1, 0.2, 0.25, 0.35, 0.5])
        self.assertEqual(mobile_padding, [64, 96, 112, 128, 144, 160, 192, 224])

    def test_grid_is_validated_and_does_not_allow_threshold_below_evaluation_floor(self):
        with self.assertRaisesRegex(ValueError, "operational confidence grid"):
            calibration_grid(
                {"calibration": {
                    "operational_conf_values": [0.005, 0.1],
                    "padding_values": [64],
                }},
                "yolo26n",
                0.01,
            )
        with self.assertRaisesRegex(ValueError, "padding grid"):
            calibration_grid(
                {"calibration": {
                    "operational_conf_values": [0.01],
                    "padding_values": [0, 64],
                }},
                "yolo26n",
                0.01,
            )

    def test_selection_filters_both_gates_then_uses_area_and_tie_breaks(self):
        def candidate(confidence, padding, area, segments, retention=0.96, coverage=0.99):
            return {
                "operational_conf": confidence,
                "padding": padding,
                "tensor_area_ratio_mean": area,
                "candidate_segments_per_frame": segments,
                "mAP50_retention": retention,
                "gt_roi_coverage_mean": coverage,
                "MAP_GATE_PASS": retention >= MAP_RETENTION_GATE,
                "COVERAGE_GATE_PASS": coverage >= COVERAGE_GATE,
                "FINAL_GATE_PASS": retention >= MAP_RETENTION_GATE and coverage >= COVERAGE_GATE,
            }

        rows = [
            candidate(0.35, 224, 0.01, 1, coverage=0.97),
            candidate(0.2, 128, 0.2, 1),
            candidate(0.1, 96, 0.201, 1),
            candidate(0.35, 96, 0.201, 1),
            candidate(0.35, 128, 0.201, 1),
            candidate(0.01, 64, 0.15, 1, retention=0.94),
        ]
        selected = select_configuration(rows)
        self.assertEqual(selected["padding"], 96)
        self.assertEqual(selected["operational_conf"], 0.35)

    def test_no_passing_configuration_is_not_selected(self):
        row = {
            "tensor_area_ratio_mean": 0.1,
            "candidate_segments_per_frame": 1.0,
            "FINAL_GATE_PASS": False,
        }
        self.assertIsNone(select_configuration([row]))

    def test_selected_configuration_reuse_checks_receipt_and_full_provenance(self):
        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "calibration" / "yolo26n"
            directory.mkdir(parents=True)
            config_path = root / "configs" / "yolo26n.yaml"
            config_path.parent.mkdir()
            config_path.write_text("config snapshot\n", encoding="utf-8")
            manifest = root / "pilot_manifest.txt"
            manifest.write_text("manifest\n", encoding="utf-8")
            paths = {
                name: directory / filename
                for name, filename in (
                    ("sweep", "calibration_sweep.csv"),
                    ("matrix", "calibration_gate_matrix.csv"),
                    ("baseline", "full_frame_baseline.json"),
                    ("report", "calibration_report.md"),
                )
            }
            for name, path in paths.items():
                if name == "sweep":
                    continue
                path.write_text(name, encoding="utf-8")
            config = {
                "audit_protocol": {"checkpoint_sha256": "checkpoint"},
                "calibration": {
                    "operational_conf_values": [0.01, 0.1],
                    "padding_values": [64, 96],
                },
            }
            implementations = {"run_pilot.py": "impl-hash"}
            sweep_rows = []
            for confidence, padding in ((0.01, 64), (0.01, 96), (0.1, 64), (0.1, 96)):
                passed = confidence == 0.1 and padding == 64
                sweep_rows.append({
                    "model": "yolo26n",
                    "checkpoint_sha256": "checkpoint",
                    "pilot_manifest_sha256": digest(manifest),
                    "config_sha256": digest(config_path),
                    "implementation_sha256": implementations,
                    "evaluation_conf": 0.01,
                    "operational_conf": confidence,
                    "padding": padding,
                    "mAP50": 0.96 if passed else 0.8,
                    "mAP50_95": 0.5,
                    "Recall": 0.9,
                    "mAP50_retention": 0.96 if passed else 0.8,
                    "gt_roi_coverage_mean": 0.99 if passed else 0.9,
                    "candidate_segments_per_frame": 1.0,
                    "tensor_area_ratio_mean": 0.2 if passed else 0.5,
                    "MAP_GATE_PASS": passed,
                    "COVERAGE_GATE_PASS": passed,
                    "FINAL_GATE_PASS": passed,
                })
            with paths["sweep"].open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sweep_rows[0].keys())
                writer.writeheader()
                writer.writerows(sweep_rows)
            document = {
                "status": "CALIBRATION_PASS",
                "model": "yolo26n",
                "checkpoint_sha256": "checkpoint",
                "pilot_manifest_sha256": digest(manifest),
                "config_sha256": digest(config_path),
                "evaluation_conf": 0.01,
                "operational_conf": 0.1,
                "padding": 64,
                "mAP50_seg": 0.96,
                "mAP50_retention": 0.96,
                "gt_roi_coverage_mean": 0.99,
                "candidate_segments_per_frame": 1.0,
                "tensor_area_ratio_mean": 0.2,
                "calibration_sweep_sha256": digest(paths["sweep"]),
                "calibration_gate_matrix_sha256": digest(paths["matrix"]),
                "full_frame_baseline_sha256": digest(paths["baseline"]),
            }
            selected_path = directory / "selected_operational_config.json"
            selected_path.write_text(json.dumps(document), encoding="utf-8")
            (directory / "selected_operational_config.sha256").write_text(
                digest(selected_path), encoding="utf-8"
            )
            pilot = SimpleNamespace(
                PILOT_DIR=root,
                _sha256=digest,
                _implementation_hashes=lambda: implementations,
            )

            selected = validate_selected_config("yolo26n", 0.01, config, pilot)

            self.assertEqual(selected["padding"], 64)
            pilot._implementation_hashes = lambda: {"run_pilot.py": "changed-impl-hash"}
            with self.assertRaisesRegex(ValueError, "execution implementation changed"):
                validate_selected_config("yolo26n", 0.01, config, pilot)
            pilot._implementation_hashes = lambda: implementations
            config_path.write_text("changed config\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance/status mismatch"):
                validate_selected_config("yolo26n", 0.01, config, pilot)


if __name__ == "__main__":
    unittest.main()
