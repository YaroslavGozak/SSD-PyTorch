"""Optional Ultralytics implementation of the experiment adapter."""

import numpy as np
import torch

from .adapters import PreparedInput


class UltralyticsAdapter:
    def __init__(self, weights: str, device: str = "cpu", stride: int = 32):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.model.model.eval()
        self.device = torch.device(device)
        self.model.model.to(self.device)
        self.stride = int(stride)

    def prepare(self, image: np.ndarray, requested_hw):
        height, width = map(int, requested_hw)
        rounded = ((height + self.stride - 1) // self.stride * self.stride,
                   (width + self.stride - 1) // self.stride * self.stride)
        tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
        tensor = torch.nn.functional.interpolate(tensor[None], size=rounded, mode="bilinear", align_corners=False)
        tensor = tensor.to(self.device)
        return PreparedInput(tensor, height, width, height, width, int(tensor.shape[-2]), int(tensor.shape[-1]))

    def infer(self, prepared):
        with torch.inference_mode():
            # Keep the invocation compatible with the repository benchmark's
            # YOLO path (measurek.run_forward uses this same convention).
            from tools.benchmarks.measurek import run_forward
            return run_forward(self.model.model, prepared.value, is_yolo=True)

    def postprocess(self, raw_output):
        return raw_output

    def preprocessing_metadata(self):
        return {"input": "RGB uint8", "layout": "NCHW", "dtype": "float32", "batch_size": 1,
                "normalization": "divide_by_255", "resize": "bilinear", "align_corners": False,
                "shape_policy": "ceil_to_stride", "stride": self.stride,
                "letterbox": False, "postprocess": "passthrough"}
