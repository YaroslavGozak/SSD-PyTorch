"""Small adapter boundary used by collection code.

The package deliberately does not depend on an Ultralytics import at module load time.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class PreparedInput:
    tensor: Any
    requested_h: int
    requested_w: int
    crop_h: int
    crop_w: int
    tensor_h: int
    tensor_w: int
    preprocess_ms: float = 0.0
    prediction_count_before_nms: Optional[int] = None
    prediction_count_after_nms: Optional[int] = None

    @property
    def value(self) -> Any:
        return self.tensor

    @property
    def tensor_hw(self) -> Tuple[int, int]:
        return self.tensor_h, self.tensor_w

    @property
    def effective_area(self) -> int:
        return self.tensor_h * self.tensor_w


class DetectorAdapter(Protocol):
    stride: int

    def preprocessing_metadata(self) -> dict:
        ...

    def prepare(self, image: np.ndarray, requested_hw: Tuple[int, int]) -> PreparedInput:
        ...

    def infer(self, prepared: PreparedInput) -> Any:
        ...

    def postprocess(self, raw_output: Any) -> Any:
        ...


class FakeAdapter:
    """Deterministic adapter for tests and CPU smoke runs."""

    def __init__(self, stride: int = 32, scale: float = 1.0, fixed_ms: float = 0.05, pixel_ms: float = 0.00001):
        self.stride = int(stride)
        self.scale = float(scale)
        self.fixed_ms = float(fixed_ms)
        self.pixel_ms = float(pixel_ms)

    def prepare(self, image: np.ndarray, requested_hw: Tuple[int, int]) -> PreparedInput:
        height, width = (int(v) for v in requested_hw)
        rounded = ((height + self.stride - 1) // self.stride * self.stride,
                   (width + self.stride - 1) // self.stride * self.stride)
        tensor = np.zeros((1, 3, *rounded), dtype=np.float32)
        return PreparedInput(tensor, height, width, height, width, int(tensor.shape[-2]), int(tensor.shape[-1]))

    def infer(self, prepared: PreparedInput) -> float:
        # Keep smoke-test timing deterministic and intentionally linear in area.
        import time
        height, width = prepared.tensor_hw
        time.sleep(max(0.0, (self.fixed_ms + self.pixel_ms * height * width) / 1000.0))
        return self.scale * float(height * width)

    def postprocess(self, raw_output: Any) -> Any:
        return raw_output

    def preprocessing_metadata(self):
        return {"input": "ignored", "content": "zeros", "layout": "NCHW", "dtype": "float32",
                "batch_size": 1, "shape_policy": "ceil_to_stride", "stride": self.stride}
