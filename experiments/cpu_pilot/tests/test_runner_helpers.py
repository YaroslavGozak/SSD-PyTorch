import csv
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
from unittest.mock import Mock
from tools.helpers.pipeline import process_frame

from experiments.cpu_pilot.run_pilot import (
    AffineCostMerger,
    DEFAULT_EVALUATION_CONF,
    MODEL_NAMES,
    _effective_shape_factory,
    _merge_frame_rows,
    _runtime_environment,
    _validate_confidence,
    _write_csv,
    run_pipeline_modes,
)
from experiments.cpu_pilot.prepare import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_MODELS,
    MODEL_CHECKPOINTS,
    MODEL_CONFIGS,
)
from experiments.cpu_pilot.confidence import (
    SELECTION_RULE,
    digest,
    load_frozen_threshold,
    run_selection,
    select_threshold,
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

    def test_pipeline_stops_when_joint_calibration_gate_failed(self):
        for rejected in MODEL_NAMES:
            with self.subTest(model=rejected), tempfile.TemporaryDirectory() as directory:
                pilot_dir = Path(directory)
                (pilot_dir / "configs").mkdir()
                for name in MODEL_NAMES:
                    (pilot_dir / "configs" / f"{name}.yaml").write_text(
                        json.dumps({"train_params": {}}), encoding="utf-8"
                    )
                (pilot_dir / "cost_profiles_cpu.json").write_text("{}", encoding="utf-8")
                decisions = {
                    name: {
                        "status": "CALIBRATION_GATE_FAILED"
                    }
                    for name in (rejected,)
                }
                with patch("pathlib.Path.is_file", return_value=True), patch(
                    "pathlib.Path.read_text", return_value="{}"
                ), patch(
                    "experiments.cpu_pilot.calibration.load_selected_for_models",
                    return_value=decisions,
                ), patch("experiments.cpu_pilot.run_pilot.PILOT_DIR", pilot_dir), patch(
                    "experiments.cpu_pilot.run_pilot.MODEL_NAMES", (rejected,)
                ), patch("experiments.cpu_pilot.run_pilot._load_model") as load:
                    with self.assertRaisesRegex(RuntimeError, "STOP: " + rejected):
                        run_pipeline_modes({}, [], torch.device("cpu"))
                    load.assert_not_called()

    def test_pipeline_uses_provenance_locked_joint_calibration(self):
        operational_conf = 0.35
        with tempfile.TemporaryDirectory() as directory:
            pilot_dir = Path(directory)
            (pilot_dir / "configs").mkdir()
            (pilot_dir / "configs" / "yolo26n.yaml").write_text(
                json.dumps({"train_params": {"infer_conf_threshold": CONFIDENCE_THRESHOLD}}),
                encoding="utf-8",
            )
            runtime = {"cpu_model": "test cpu"}
            selected = {
                "status": "CALIBRATION_PASS",
                "operational_conf": operational_conf,
                "padding": 64,
                "evaluation_conf": CONFIDENCE_THRESHOLD,
                "checkpoint_sha256": "checkpoint",
                "config_sha256": "config",
                "pilot_manifest_sha256": "manifest",
                "implementation_sha256": {"run_pilot.py": "impl"},
            }
            (pilot_dir / "cost_profiles_cpu.json").write_text(
                json.dumps({"yolo26n": {
                    "K_ms": 1.0, "c_ms_per_pixel": 0.01,
                    "evaluation_conf": CONFIDENCE_THRESHOLD,
                    "operational_conf": operational_conf,
                    "padding": 64,
                    "checkpoint_sha256": "checkpoint",
                    "config_sha256": "config",
                    "pilot_manifest_sha256": "manifest",
                    "implementation_sha256": {"run_pilot.py": "impl"},
                    "timing_mode": "inference_only_at_evaluation_conf",
                    "runtime_environment": runtime,
                }}),
                encoding="utf-8",
            )
            model = Mock()
            dataset = Mock()
            config = {"train_params": {"infer_conf_threshold": CONFIDENCE_THRESHOLD}}
            with (
                patch("experiments.cpu_pilot.run_pilot.PILOT_DIR", pilot_dir),
                patch("experiments.cpu_pilot.run_pilot.MODEL_NAMES", ("yolo26n",)),
            patch("experiments.cpu_pilot.run_pilot._runtime_environment", return_value=runtime),
            patch(
                "experiments.cpu_pilot.calibration.load_selected_for_models",
                return_value={"yolo26n": selected},
            ),
            patch("experiments.cpu_pilot.run_pilot._load_model", return_value=(
                model, dataset, config, CONFIDENCE_THRESHOLD
            )) as load_model,
                patch("experiments.cpu_pilot.run_pilot._dataset_indices", return_value=[]),
                patch("experiments.cpu_pilot.run_pilot._dataset_root", return_value=Path(".")),
                patch("experiments.cpu_pilot.run_pilot._run_pipeline", return_value=(
                    {"model": "yolo26n"}, []
                )) as run_pipeline,
                patch("experiments.cpu_pilot.run_pilot._metadata", return_value={}),
            ):
                run_pipeline_modes(
                    {}, [{"frame_index": 0}], torch.device("cpu"),
                    debug=False, partial=True,
                    operational_conf=operational_conf,
                )

            load_model.assert_called_once_with(
                "yolo26n",
                torch.device("cpu"),
                CONFIDENCE_THRESHOLD,
            )
            self.assertEqual(run_pipeline.call_count, 4)
            self.assertTrue(all(call.args[4] == CONFIDENCE_THRESHOLD for call in run_pipeline.call_args_list))
            self.assertTrue(all(call.args[5] == operational_conf for call in run_pipeline.call_args_list))

    def test_confidence_defaults_and_validation(self):
        self.assertEqual(DEFAULT_EVALUATION_CONF, 0.01)
        self.assertEqual(_validate_confidence("evaluation_conf", 0.02), 0.02)
        for invalid in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                _validate_confidence("operational_conf", invalid)

    def test_confidence_selection_uses_low_floor_predictions(self):
        frame_rows = [{
            "prediction_records": json.dumps([
                {"class": "object", "bbox": [0, 0, 10, 10], "confidence": 0.02},
                {"class": "false_positive", "bbox": [20, 20, 30, 30], "confidence": 0.8},
            ]),
            "ground_truth_records": json.dumps({"object": [[0, 0, 10, 10]]}),
        }]
        selected, curve = select_threshold(frame_rows)
        self.assertEqual(selected["threshold"], 0.01)
        self.assertEqual(selected["tp"], 1)
        self.assertGreater(selected["f1"], max(row["f1"] for row in curve[1:]))

    def test_frozen_operational_confidence_is_model_scoped_and_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "pilot_manifest.txt").write_text("frame\n", encoding="utf-8")
            config_path = root / "configs" / "yolo26n.yaml"
            config_path.write_text("config\n", encoding="utf-8")
            candidates = root / "confidence_yolo26n_candidates.csv"
            candidates.write_text(
                "threshold,f1,recall\n0.25,0.8,0.9\n",
                encoding="utf-8",
            )
            frame_metrics = root / "confidence_yolo26n_frame_metrics.csv"
            frame_metrics.write_text("evidence\n", encoding="utf-8")
            summary = root / "confidence_yolo26n_ff_summary.csv"
            summary.write_text("summary\n", encoding="utf-8")
            evidence_files = {
                "candidates": candidates.name,
                "frame_metrics": frame_metrics.name,
                "summary": summary.name,
            }
            evidence_hashes = {
                path.name: digest(path)
                for path in (candidates, frame_metrics, summary)
            }
            document = {
                "selection_rule": SELECTION_RULE,
                "models": {
                    "yolo26n": {
                        "operational_conf": 0.25,
                        "checkpoint_sha256": "checkpoint-hash",
                        "pilot_manifest_sha256": digest(root / "pilot_manifest.txt"),
                        "config_sha256": digest(config_path),
                        "evidence_files": evidence_files,
                        "evidence_sha256": evidence_hashes,
                    }
                },
            }
            decision_path = root / "operational_confidence.json"
            decision_path.write_text(json.dumps(document), encoding="utf-8")
            (root / "operational_confidence.sha256").write_text(
                digest(decision_path), encoding="utf-8"
            )
            threshold = load_frozen_threshold(
                root, "yolo26n", {"audit_protocol": {"checkpoint_sha256": "checkpoint-hash"}}
            )
            self.assertEqual(threshold, 0.25)

    def test_confidence_stage_uses_active_runner_module_and_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "configs").mkdir()
            (output_dir / "configs" / "yolo26n.yaml").write_text(
                "config\n", encoding="utf-8"
            )
            (output_dir / "pilot_manifest.txt").write_text("manifest\n", encoding="utf-8")
            model = Mock()
            dataset = Mock()
            dataset.idx2label = {0: "background", 1: "object"}
            config = {"audit_protocol": {"checkpoint_sha256": "checkpoint-hash"}}
            active_runner = SimpleNamespace(
                PILOT_DIR=output_dir,
                MODEL_NAMES=("yolo26n",),
                _load_model=Mock(return_value=(model, dataset, config, 0.01)),
                _dataset_indices=Mock(return_value=[0]),
                _dataset_root=Mock(return_value=Path(".")),
                _run_pipeline=Mock(return_value=(
                    {"mAP50": 0.5},
                    [{
                        "prediction_records": "[]",
                        "ground_truth_records": "{}",
                    }],
                )),
                _write_csv=_write_csv,
            )
            with patch(
                "experiments.cpu_pilot.confidence.select_threshold",
                return_value=(
                    {"threshold": 0.25, "tp": 1, "fp": 0, "fn": 0,
                     "precision": 1.0, "recall": 1.0, "f1": 1.0},
                    [{"threshold": 0.25, "tp": 1, "fp": 0, "fn": 0,
                      "precision": 1.0, "recall": 1.0, "f1": 1.0}],
                ),
            ):
                run_selection(
                    {"provenance": {"pilot_manifest_sha256": "manifest-hash"}},
                    [{"video_id": "v", "frame_index": 0}],
                    torch.device("cpu"),
                    pilot_module=active_runner,
                )

            active_runner._load_model.assert_called_once()
            decision = json.loads(
                (output_dir / "operational_confidence.json").read_text(encoding="utf-8")
            )
            self.assertEqual(decision["models"]["yolo26n"]["operational_conf"], 0.25)

    def test_process_frame_splits_evaluation_and_operational_detections_after_one_forward(self):
        tracker = Mock()
        tracker.update.return_value = {"rois": []}
        detections = {
            "boxes": torch.tensor([
                [0.05, 0.05, 0.15, 0.15],
                [0.25, 0.25, 0.35, 0.35],
                [0.45, 0.45, 0.55, 0.55],
            ]),
            "labels": torch.tensor([1, 1, 1]),
            "scores": torch.tensor([0.02, 0.35, 0.90]),
        }
        with patch(
            "tools.helpers.pipeline.run_model_inference",
            return_value=(None, [detections]),
        ) as inference:
            result = process_frame(
                model=Mock(), idx2label={1: "object"},
                frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
                im_tensor=torch.zeros((1, 3, 100, 100)), tracker=tracker,
                next_frame_rois=[], frame_idx=1, key_frame_interval=10,
                im_size_hw=(100, 100), conf_threshold=0.5,
                evaluation_conf_threshold=0.01, nms_iou=0.45,
                model_device=torch.device("cpu"), roi_grid=1,
            )

        inference.assert_called_once()
        self.assertEqual(len(result.evaluation_detections), 3)
        self.assertEqual(len(result.final_detections), 1)
        self.assertAlmostEqual(result.final_detections[0]["confidence"], 0.9)
        self.assertAlmostEqual(
            tracker.update.call_args.args[0][0]["confidence"], 0.9
        )

    def test_default_stage3_models_and_optional_vgg16_model_are_locked(self):
        self.assertEqual(DEFAULT_MODELS, ("yolo26n", "roi_ssd_mobilenet"))
        self.assertEqual(MODEL_NAMES, DEFAULT_MODELS)
        self.assertIn("roi_ssd", MODEL_CONFIGS)
        self.assertIn("roi_ssd", MODEL_CHECKPOINTS)
        for config_path in MODEL_CONFIGS.values():
            with self.subTest(config=config_path.name):
                import yaml

                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    config["train_params"]["infer_conf_threshold"],
                    CONFIDENCE_THRESHOLD,
                )
        self.assertEqual(
            MODEL_CHECKPOINTS["roi_ssd"][1],
            "f8f49132e08a7c5bf3744ce32e46e1b0fed6595aabd9cf59cf68699569503f76",
        )

    def test_runtime_environment_logs_requested_hardware_and_software(self):
        runtime = _runtime_environment()
        self.assertTrue(runtime["cpu_model"])
        self.assertGreater(runtime["physical_cores"], 0)
        self.assertGreaterEqual(runtime["logical_cores"], runtime["physical_cores"])
        self.assertEqual(runtime["pytorch_intra_op_threads"], torch.get_num_threads())
        self.assertEqual(runtime["pytorch_inter_op_threads"], torch.get_num_interop_threads())
        self.assertGreater(runtime["opencv_threads"], 0)
        self.assertTrue(runtime["os"])
        self.assertTrue(runtime["python"])
        self.assertEqual(runtime["pytorch"], str(torch.__version__))

    def test_merge_frame_rows_accepts_large_serialized_detection_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            pilot_dir = Path(directory)
            existing_path = pilot_dir / "frame_level_metrics.csv"
            original = {
                "run": "final_seg",
                "model": "roi_ssd_mobilenet",
                "video_id": "video-1",
                "frame_index": "1",
                "prediction_records": "x" * 150_000,
            }
            with existing_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=original)
                writer.writeheader()
                writer.writerow(original)

            replacement = {**original, "prediction_records": "updated"}
            with patch("experiments.cpu_pilot.run_pilot.PILOT_DIR", pilot_dir):
                _merge_frame_rows([replacement])

            with existing_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prediction_records"], "updated")

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
