import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.cost_model_validation.adapters import FakeAdapter
from experiments.cost_model_validation.analyze_experiment_a import _load_provenance, analyze as analyze_a
from experiments.cost_model_validation.analyze_experiment_b import analyze as analyze_b
from experiments.cost_model_validation.common import collection_provenance, file_sha256, write_json
from experiments.cost_model_validation.models import control_predictions, load_latency_models


def calibration():
    envelope = dict(min_effective_area=1, max_effective_area=10000, min_tensor_h=1,
                    max_tensor_h=100, min_tensor_w=1, max_tensor_w=100,
                    min_aspect_ratio=.01, max_aspect_ratio=100)
    models = {
        "linear": {"coefficients": {"b0": .01, "b1": .000001}},
        "quadratic": {"coefficients": {"b0": .02, "b1": .000001, "b2": .000000001}},
        "piecewise": {"coefficients": {"b0": .03, "b1": .000001, "b2": .000002}, "breakpoint_area": 100},
    }
    for model in models.values():
        model["calibration_envelope"] = envelope
    return {"latency_models": models, "calibration_envelope": envelope}


def write_pairs(directory):
    rows = []
    for pair, shapes in enumerate(((5, 5, 10), (10, 10, 20), (20, 20, 30))):
        for repetition in range(2):
            a, b, u = shapes
            rows.append(dict(pair_id=pair, repetition=repetition, order="merged_first" if repetition else "separate_first",
                             timing_mode="inference_only", difference_ms=1, separate_ms=11, merged_ms=10,
                             a1_effective=a*a, a2_effective=b*b, au_effective=u*u,
                             delta_effective_area=u*u-a*a-b*b, boundary_bin="merge", geometry_type="horizontal",
                             r1_tensor_h=a, r1_tensor_w=a, r2_tensor_h=b, r2_tensor_w=b,
                             union_tensor_h=u, union_tensor_w=u, linear_tau_predicted_merge=True))
    path = directory / "experiment_b_raw.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class ReportingTests(unittest.TestCase):
    def test_snapshot_restores_area_regimes_and_legacy_controls_without_fit_file(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            raw = write_pairs(directory)
            artifact = calibration()
            write_json(directory / "experiment_b_metadata.json", {
                "latency_models": artifact["latency_models"],
                "calibration_reference": {"path": "missing/linear_fit.json"},
                "control_records": [{"tensor_h": 10, "tensor_w": 10, "measured_ms": 40,
                                     "predicted_ms": 1, "relative_error": 39}],
            })
            report = analyze_b(str(raw), str(directory))
            self.assertTrue(report["area_regime_available"])
            self.assertEqual(set(report["metrics_by_area_regime"]), {"below_breakpoint", "spans_breakpoint", "above_breakpoint"})
            self.assertEqual(report["piecewise_breakpoint_area"], 100)
            control = report["control_records"][0]
            for name, expected in (("linear", 10.1), ("quadratic", 20.11), ("piecewise", 30.1)):
                self.assertAlmostEqual(control["predictions"][name]["predicted_ms"], expected)
                self.assertAlmostEqual(control["predictions"][name]["relative_error"], (40-expected)/expected)
            self.assertEqual(json.loads((directory / "decision_metrics.json").read_text())["area_regime_available"], True)

    def test_adjacent_fit_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            raw = write_pairs(directory)
            write_json(directory / "linear_fit.json", calibration())
            self.assertTrue(analyze_b(str(raw))["area_regime_available"])
            other = calibration()
            other["latency_models"]["piecewise"]["breakpoint_area"] = 1000
            write_json(directory / "other.json", other)
            self.assertEqual(analyze_b(str(raw), calibration_path=str(directory / "other.json"))["piecewise_breakpoint_area"], 1000)

    def test_relocated_reference_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            raw = write_pairs(directory)
            fit = directory / "linear_fit.json"
            write_json(fit, calibration())
            metadata = {"calibration_reference": {"path": "old/run/linear_fit.json", "content_hash": file_sha256(fit)}}
            write_json(directory / "experiment_b_metadata.json", metadata)
            self.assertTrue(analyze_b(str(raw))["area_regime_available"])
            metadata["calibration_reference"]["content_hash"] = "wrong"
            write_json(directory / "experiment_b_metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                analyze_b(str(raw))

    def test_missing_breakpoint_is_explained(self):
        with tempfile.TemporaryDirectory() as folder:
            report = analyze_b(str(write_pairs(Path(folder))))
            self.assertFalse(report["area_regime_available"])
            self.assertIn("--fit", report["area_regime_unavailable_reason"])

    def test_unavailable_control_model_is_explicit(self):
        artifact = calibration()
        del artifact["latency_models"]["quadratic"]
        predictions = control_predictions(load_latency_models(artifact), 100, 10)
        self.assertFalse(predictions["quadratic"]["available"])
        self.assertIsNone(predictions["quadratic"]["relative_error"])

    def test_collection_provenance_survives_calibration_export(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            weights = directory / "weights.bin"
            weights.write_bytes(b"test weights")
            config = {"seed": 42, "model": {"backend": "fake", "weights": str(weights)}}
            with patch("experiments.cost_model_validation.common.system_metadata", return_value={"git_commit": "collection-commit", "git_dirty": True}):
                provenance = collection_provenance(config, FakeAdapter(), "inference_only")
            write_json(directory / "metadata.json", {"config": config, "provenance": provenance})
            raw = directory / "experiment_a_raw.csv"
            rows = [dict(effective_area=side*side, tensor_h=side, tensor_w=side, run_id=run,
                         timing_mode="inference_only", inference_ms=10 + .01*side*side + run*.01)
                    for run in range(3) for side in (10, 20, 30, 40, 50)]
            with raw.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            analyze_a(str(raw), str(directory), bootstrap_count=4)
            exported = json.loads((directory / "linear_fit.json").read_text())["provenance"]
            self.assertTrue(exported["complete"])
            self.assertEqual(exported["weights_sha256"], file_sha256(weights))
            self.assertEqual(exported["git_commit"], "collection-commit")
            self.assertEqual(exported["seed"], 42)
            self.assertEqual(exported["device"], "cpu")
            self.assertTrue(exported["backend_version"])
            self.assertEqual(exported["preprocessing"]["stride"], 32)

    def test_legacy_provenance_does_not_invent_versions_or_preprocessing(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_json(directory / "metadata.json", {"config": {"seed": 0, "model": {"backend": "ultralytics", "device": "cpu"}},
                                                      "git_commit": "historical", "model_weights_sha256": "recorded"})
            provenance = _load_provenance(str(directory / "experiment_a_raw.csv"), None)
            self.assertFalse(provenance["complete"])
            self.assertIn("backend_version", provenance["missing_fields"])
            self.assertIn("preprocessing", provenance["missing_fields"])
            self.assertEqual(provenance["git_commit"], "historical")


if __name__ == "__main__":
    unittest.main()
