from .geometry import stride_rounded_shape
"""Adapter for the repository's RoiSSD and RoiSSDMobileNet models."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.roissd import RoiSSD
from model.roissd_mobilenet import RoiSSDMobileNet
from tools.helpers.config_reader import load_config

from .adapters import PreparedInput


class RoiSSDAdapter:
    """Load and invoke a project SSD model without loading a dataset."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_config: str, weights: str, device: str = "cpu", stride: int = 1):
        config = load_config(model_config)
        dataset_config = config["dataset_params"]
        train_config = config["train_params"]
        model_name = str(train_config["model"]).lower()
        print(f"Loading model: {model_name}")
        num_classes = int(dataset_config["num_classes"])
        if model_name == "roissd":
            self.model = RoiSSD(config=config["model_params"], num_classes=num_classes)
        elif model_name in {"roissd-mobilenet", "roissd_mobilenet"}:
            self.model = RoiSSDMobileNet(config=config["model_params"], num_classes=num_classes)
        else:
            raise ValueError(f"model config must select roissd or roissd-mobilenet, got {model_name!r}")

        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"ROI-SSD weights not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.stride = int(stride)

    def prepare(self, image: np.ndarray, requested_hw):
        height, width = (int(v) for v in requested_hw)
        height, width = stride_rounded_shape(height, width, self.stride)
        tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
        tensor = F.interpolate(tensor[None], size=(height, width), mode="bilinear", align_corners=False)
        mean = tensor.new_tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        std = tensor.new_tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.to(self.device)
        return PreparedInput(tensor, int(requested_hw[0]), int(requested_hw[1]), int(requested_hw[0]), int(requested_hw[1]), int(tensor.shape[-2]), int(tensor.shape[-1]))

    def infer(self, prepared):
        with torch.inference_mode():
            return self.model(prepared.value, None)

    def postprocess(self, raw_output):
        return raw_output

    def preprocessing_metadata(self):
        return {"input": "RGB uint8", "layout": "NCHW", "dtype": "float32", "batch_size": 1,
                "normalization": "divide_by_255_then_imagenet", "normalization_mean": list(self.IMAGENET_MEAN),
                "normalization_std": list(self.IMAGENET_STD), "resize": "bilinear", "align_corners": False,
                "shape_policy": "ceil_to_stride" if self.stride > 1 else "requested_hw", "stride": self.stride,
                "letterbox": False, "postprocess": "passthrough; model forward includes detection processing"}
