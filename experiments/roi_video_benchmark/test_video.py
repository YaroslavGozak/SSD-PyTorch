import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.cost_model_validation.tests.test_shape_validation import fixture
from experiments.cost_model_validation.shape_model import MergePolicy, policy_declaration
from experiments.roi_video_benchmark.run import policy_config, POLICIES
from tools.benchmarks.benchmark_framework_vid import VideoSequenceBenchmark
from tools.helpers.config_reader import load_config
from tools.helpers.pipeline import (
    process_frame, convert_crop_to_input_tensor, _convert_rois_between_spaces,
    tensor_to_detection_list, build_tracker,
)
from tools.mergers.calibrated import CalibratedRoiMerger


class Detector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.shapes = []

    def forward(self, images, target=None):
        self.shapes.append(tuple(images.shape[-2:]))
        return None, [dict(boxes=torch.tensor([[.25,.25,.75,.75]]),
                           scores=torch.tensor([.9]), labels=torch.tensor([1]))]


class VideoTests(unittest.TestCase):
    def test_linear_direct_equals_tau_on_effective_areas(self):
        policy = MergePolicy(fixture())
        model = policy.area_models["linear"]
        k, c = model.coefficients["b0"], model.coefficients["b1"]
        for a in (32,64,96,128):
            for b in (32,64,96,128):
                shapes = [(32,a), (32,b), (128,128)]
                result = policy.predict(shapes, "linear_tau", exact_shapes=True)
                self.assertEqual(result["predicted_merge"], 128*128-32*a-32*b < k/c)

    def test_exact_shape_does_not_silently_round(self):
        decl = policy_declaration({"policies":dict(primary="shape_lookup_conservative",fallback="separate")})
        policy = MergePolicy(fixture(), decl)
        result = policy.predict([(33,64)]*3, strict=False, exact_shapes=True)
        self.assertEqual(result["fallback_used"], "separate")
        self.assertEqual(policy.fallback_counts["separate"], 1)

    def test_missing_uncertainty_is_reported_as_fallback(self):
        artifact = fixture()
        for estimate in artifact["latency_models"]["shape_lookup"]["table"].values():
            estimate["standard_error_s"] = None
        with patch("tools.mergers.calibrated.load_calibration", return_value=artifact), \
             patch("tools.mergers.calibrated.file_sha256", return_value="fixture"):
            merger = CalibratedRoiMerger("fixture.json", dict(primary="shape_lookup_conservative",fallback="separate"))
            result = merger([[0,0,32,32],[32,0,64,32]], effective_shape=lambda r:(32,r[2]-r[0]))
            self.assertEqual(len(result),2)
            self.assertEqual(merger.decisions[0]["reason"],"missing_uncertainty")
            self.assertEqual(merger.policy.fallback_counts["separate"],1)

    def test_pipeline_shapes_match_actual_crop_and_fallback_counts(self):
        with patch("tools.mergers.calibrated.load_calibration", return_value=fixture()), \
             patch("tools.mergers.calibrated.file_sha256", return_value="fixture"):
            merger = CalibratedRoiMerger("fixture.json", dict(primary="shape_lookup_conservative",fallback="separate"))
            model = Detector()
            tracker = build_tracker(dict(type="static_padding",static_padding=dict(pad_x=0,pad_y=0)))
            image = torch.zeros(1,3,300,300)
            proposals = [[0,0,160,100],[200,110,499,199]]
            result = process_frame(model=model,idx2label={1:"car"},frame_bgr=np.zeros((200,500,3),np.uint8),
                im_tensor=image,tracker=tracker,next_frame_rois=proposals,frame_idx=1,key_frame_interval=10,
                im_size_hw=(300,300),conf_threshold=.5,nms_iou=.5,merge_fn=merger,model_device=torch.device("cpu"))
            self.assertEqual(result.merge_decisions[0]["shapes"][:2], model.shapes)
            self.assertEqual(len(model.shapes),2)
            self.assertEqual(merger.policy.fallback_counts["separate"],1)
            self.assertEqual(result.merge_decisions[0]["reason"],"outside_envelope")

    def test_crop_remapping_at_edge(self):
        image = torch.zeros(1,3,320,320)
        crop, rect = convert_crop_to_input_tensor(image,[610,400,640,480],(480,640),32)
        self.assertEqual(tuple(crop.shape[-2:]), (64,32))
        self.assertLessEqual(rect[2],640)
        self.assertLessEqual(rect[3],480)
        detections = tensor_to_detection_list(dict(boxes=torch.tensor([[0.,0.,1.,1.]]),
             labels=torch.tensor([1]),scores=torch.tensor([.9])),{1:"car"},rect[2]-rect[0],rect[3]-rect[1],rect[:2])
        self.assertEqual(detections[0]["bbox"],rect)

    def test_policy_configs_lock_everything_except_merging(self):
        base = load_config("config/benchmark-vid-roi-policies.yaml")
        configs = [policy_config(base,p,"unused") for p in POLICIES]
        for cfg in configs[1:]:
            self.assertEqual(cfg["benchmark_vid_params"]["tracker"],configs[0]["benchmark_vid_params"]["tracker"])
            self.assertEqual(cfg["train_overrides"],configs[0]["train_overrides"])
            self.assertEqual(cfg["benchmark_vid_params"]["inference"]["key_frame_interval"],10)
        self.assertEqual(configs[0]["benchmark_vid_params"]["inference"]["key_frame_interval"],1)

    def test_tracker_resets_on_video_boundary(self):
        with patch("tools.benchmarks.benchmark_framework_vid.Path.mkdir"):
            cfg = load_config("config/benchmark-vid-roi-policies.yaml")
            cfg.pop("video_experiment")
            cfg["benchmark_vid_params"]["output"].update(results_dir="unused",verbose=False)
            bench = VideoSequenceBenchmark.from_config_dict(cfg)
            model = Detector()
            dataset = type("Dataset",(),{"idx2label":{1:"car"}})()
            loader = [(torch.zeros(1,3,320,320),{},[f"{v}/frame.jpg"]) for v in ("a","a","b")]
            train = dict(train_params=dict(infer_conf_threshold=.5),dataset_params=dict(im_size=320))
            module = "tools.benchmarks.benchmark_framework_vid."
            seen = []
            def observe(**kwargs):
                seen.append(copy.deepcopy(kwargs["next_frame_rois"]))
                return process_frame(**kwargs)
            with patch(module+"cv2.imread",return_value=np.zeros((320,320,3),np.uint8)), \
                 patch(module+"extract_gt_for_map",return_value=({},{})), \
                 patch(module+"process_frame",side_effect=observe), \
                 patch.object(bench,"_compute",return_value={}), \
                 patch.object(bench,"_print"), patch.object(bench,"_save"):
                bench.run(prepared=(model,dataset,loader,train))
            self.assertEqual(seen[0],[])
            self.assertTrue(seen[1])
            self.assertEqual(seen[2],[])


if __name__ == "__main__":
    unittest.main()
