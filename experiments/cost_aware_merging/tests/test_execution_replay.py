import copy
import csv
import io
import json
import shutil
import uuid
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torchvision

from experiments.cost_aware_merging.compare_runs import compare
from experiments.cost_aware_merging.replay import frame_key, resolve_frames, validate_pairs
from experiments.cost_aware_merging.run import _measure_pair, run
from experiments.cost_model_validation.adapters import FakeAdapter
from experiments.cost_model_validation.geometry import Rectangle
from experiments.cost_model_validation.roissd_adapter import RoiSSDAdapter
from model.roissd import RoiSSD
from model.roissd_mobilenet import RoiSSDMobileNet
from tools.helpers.config_reader import load_config


@contextmanager
def artifact_directory():
    # Windows sandbox cannot traverse tempfile's mode=0o700 directories.
    workspace = Path.cwd().resolve()
    path = workspace / "outputs" / f"test-cost-aware-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        assert path.resolve().is_relative_to(workspace / "outputs")
        shutil.rmtree(path)


class ExecutionTests(unittest.TestCase):
    def test_real_forward_trace_matches_executed_heads_for_both_backbones(self):
        # Construct real networks offline, without a checkpoint or pretrained downloads.
        config = load_config("config/imagenet-vid-roissd.yaml")["model_params"]
        config["low_score_threshold"] = 1.0
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for cls in (RoiSSD, RoiSSDMobileNet):
                with self.subTest(backbone=cls.__name__):
                    if cls is RoiSSD:
                        factory = patch("torchvision.models.vgg16", side_effect=lambda **kw:
                            SimpleNamespace(features=torchvision.models.vgg.make_layers(
                                torchvision.models.vgg.cfgs["D"])))
                    else:
                        constructor = torchvision.models.mobilenet_v3_large
                        factory = patch("torchvision.models.mobilenet_v3_large", side_effect=lambda **kw:
                            constructor(weights=None))
                    with factory:
                        model = cls(copy.deepcopy(config), num_classes=2).eval()
                    adapter = RoiSSDAdapter.__new__(RoiSSDAdapter)
                    adapter.model, adapter.device, adapter.stride = model, torch.device("cpu"), 1
                    adapter.enable_execution_logging()
                    seen = []
                    handles = [head.register_forward_pre_hook(
                        lambda module, args, i=i: seen.append((i, list(args[0].shape))))
                               for i, head in enumerate(model.cls_heads)]
                    try:
                        for depth, side in enumerate((32, 64, 96, 128, 192, 320), start=1):
                            seen.clear()
                            prepared = adapter.prepare(np.zeros((side, side, 3), dtype=np.uint8), (side, side))
                            adapter.infer(prepared)
                            trace = adapter.execution_metadata()
                            self.assertEqual(trace["active_depth"], depth)
                            self.assertEqual(trace["active_head_count"], depth)
                            self.assertEqual([(f["head_index"], f["nchw"]) for f in trace["feature_maps"]], seen)
                        seen.clear()
                        frame = np.zeros((64, 96, 3), dtype=np.uint8)
                        rectangles = (Rectangle(0, 0, 32, 32), Rectangle(32, 0, 96, 64), Rectangle(0, 0, 96, 64))
                        *_, observations = _measure_pair(adapter, frame, rectangles,
                            dict(repetitions=2, timing_mode="inference_only"), 7)
                        for row in observations:
                            self.assertEqual(list(row), list(observations[0]))
                            self.assertEqual(row["r1_active_depth"], 1)
                            self.assertEqual(row["r2_active_depth"], 2)
                            self.assertEqual(row["merged_active_depth"], 2)
                            self.assertEqual(len(json.loads(row["merged_feature_maps"])), 2)
                    finally:
                        for handle in handles:
                            handle.remove()
        finally:
            torch.set_num_threads(threads)

    def test_non_roissd_invocations_have_explicit_empty_trace(self):
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        rectangles = (Rectangle(0, 0, 32, 32), Rectangle(32, 0, 64, 32), Rectangle(0, 0, 64, 32))
        *_, rows = _measure_pair(FakeAdapter(), frame, rectangles,
                                dict(repetitions=2, timing_mode="detector_call"), 7)
        for row in rows:
            for prefix in ("r1", "r2", "merged"):
                self.assertEqual(row[f"{prefix}_execution_trace_status"], "not_applicable")
                self.assertEqual(row[f"{prefix}_active_depth"], "")
            self.assertEqual(set(json.loads(row["separate_call_order"])), {"r1", "r2"})


class ReplayTests(unittest.TestCase):
    def test_full_runner_replays_identical_workload_with_new_seed_and_dataset_order(self):
        settings = dict(frame_count=2, calibration_frames=1, pair_count=8, warmup=0,
                        calibration_repetitions=2, repetitions=2, canvas_hw=[64, 64],
                        shapes=[(32, 32)], sides=[32], seed=7)
        config = dict(train_params=dict(model="roissd", dataset="test"), cost_aware_merging=settings)
        class Dataset:
            images_info = [dict(filename=f"/desktop/data/video/{i}.JPEG") for i in range(5)]
            def __len__(self):
                return len(self.images_info)
        dataset = Dataset()
        def frame(dataset, index, canvas):
            filename = dataset.images_info[index]["filename"]
            value = int(Path(filename).stem)
            return np.full((*canvas, 3), value, dtype=np.uint8), tuple(canvas), filename
        adapter = FakeAdapter(stride=1)
        adapter.device = torch.device("cpu")
        fit = SimpleNamespace(coefficients=dict(b0=.001, b1=.000001), r2=1, mae=0, rmse=0)
        with artifact_directory() as root, \
             patch("experiments.cost_aware_merging.run.load_config", return_value=config), \
             patch("experiments.cost_aware_merging.run.apply_runtime_controls", return_value={}), \
             patch("experiments.cost_aware_merging.run._model_settings", return_value=dict(weights="test.pt", backend="fake", stride=1)), \
             patch("experiments.cost_aware_merging.run.file_sha256", return_value="source-content"), \
             patch("experiments.cost_aware_merging.run.build_adapter", return_value=adapter), \
             patch("experiments.cost_aware_merging.run.load_dataset", return_value=dataset), \
             patch("experiments.cost_aware_merging.run._frame", side_effect=frame), \
             patch("experiments.cost_aware_merging.run._profile", return_value=({(32, 32): 2., (32, 64): 3., (64, 64): 5.}, fit, [])), \
             patch("experiments.cost_aware_merging.run._plots"):
            run("fake.yaml", root / "first")
            config["cost_aware_merging"].update(seed=99, pair_count=1, frame_count=4, calibration_frames=2)
            dataset.images_info = [dict(filename=f"/pi/different-root/video/{i}.JPEG") for i in reversed(range(5))]
            run("fake.yaml", root / "second", replay_pairs=root / "first" / "pairs.json")
            first = json.loads((root / "first" / "metadata.json").read_text())
            second = json.loads((root / "second" / "metadata.json").read_text())
            self.assertEqual(first["workload_sha256"], second["workload_sha256"])
            self.assertEqual(second["pair_count"], 8)
            self.assertEqual(first["frame_manifest"], second["frame_manifest"])
            with redirect_stdout(io.StringIO()):
                result = compare(root / "first", root / "second", root / "comparison")
            self.assertEqual(result["pair_count"], 8)
            with patch("experiments.cost_aware_merging.run._frame", side_effect=lambda d, i, c:
                       (np.full((*c, 3), 255, dtype=np.uint8), tuple(c), d.images_info[i]["filename"])):
                run("fake.yaml", root / "different_canvas", replay_pairs=root / "first" / "pairs.json")
                changed = json.loads((root / "different_canvas" / "metadata.json").read_text())
                self.assertNotEqual(first["frame_manifest"], changed["frame_manifest"])
            with patch("experiments.cost_aware_merging.run.file_sha256", return_value="changed-source"):
                with self.assertRaisesRegex(ValueError, "source_sha256"):
                    run("fake.yaml", root / "bad", replay_pairs=root / "first" / "pairs.json")

    def test_frame_selection_survives_root_platform_and_dataset_order_changes(self):
        self.assertEqual(frame_key(r"H:\data\video_a\00001.JPEG"), "video_a/00001.JPEG")
        infos = [dict(filename="/pi/data/video_b/00002.JPEG"), dict(filename="/pi/data/video_a/00001.JPEG")]
        manifest = dict(calibration=[dict(frame_key="video_a/00001.JPEG")],
                        evaluation=[dict(frame_key="video_b/00002.JPEG")])
        self.assertEqual(resolve_frames(infos, manifest), [1, 0])
        with self.assertRaisesRegex(ValueError, "missing"):
            resolve_frames(infos[:1], manifest)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            resolve_frames(infos + infos, manifest)

    def test_replay_rejects_invalid_geometry_and_duplicate_ids(self):
        payload = dict(canvas_hw=[64, 64], pairs=[dict(pair_id=0, r1=[0, 0, 32, 32], r2=[32, 0, 64, 32])])
        payload["pairs_geometry_sha256"] = validate_pairs(payload, [64, 64])
        for mutation, message in ((lambda p: p["pairs"][0]["r1"].__setitem__(0, 1), "hash"),
                                  (lambda p: p["pairs"][0]["r2"].__setitem__(2, 65), "invalid"),
                                  (lambda p: p["pairs"].append(p["pairs"][0]), "unique")):
            changed = copy.deepcopy(payload)
            mutation(changed)
            with self.assertRaisesRegex(ValueError, message):
                validate_pairs(changed, [64, 64])
        with self.assertRaisesRegex(ValueError, "canvas"):
            validate_pairs(payload, [32, 32])

    def test_flip_comparison_requires_same_frame_content(self):
        payload = dict(canvas_hw=[64, 64], pairs=[dict(pair_id=0, r1=[0, 0, 32, 32], r2=[32, 0, 64, 32])])
        with artifact_directory() as root:
            row = dict(pair_id=0, frame_key="video/frame.JPEG", frame_canvas_sha256="content",
                       actual_separate_ms=6, actual_merged_ms=5,
                       r1_x1=0, r1_y1=0, r1_x2=32, r1_y2=32, r2_x1=32, r2_y1=0, r2_x2=64, r2_y2=32)
            for name, merged in (("cpu", 5), ("pi", 7)):
                run = root / name
                run.mkdir()
                (run / "pairs.json").write_text(json.dumps(payload))
                (run / "metadata.json").write_text(json.dumps(dict(timing_boundary="inference_only", checkpoint_sha256="weights")))
                with (run / "pairs_raw.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow({**row, "actual_merged_ms": merged})
            with redirect_stdout(io.StringIO()):
                summary = compare(root / "cpu", root / "pi", root / "comparison")
            self.assertEqual(summary["oracle_flip_count"], 1)
            path = root / "pi" / "pairs_raw.csv"
            path.write_text(path.read_text().replace("content", "different"))
            with self.assertRaisesRegex(ValueError, "different frame"):
                compare(root / "cpu", root / "pi", root / "rejected")
