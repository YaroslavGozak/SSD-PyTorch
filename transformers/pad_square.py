import torch
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors


class PadToSquare(torch.nn.Module):
    """
    Letterbox-style resize:
    - Resize image so that the short side equals S (preserve aspect ratio)
    - Pad to exactly S x S
    Compatible with torchvision v2 detection targets.
    """

    def __init__(self, size: int, fill=0):
        super().__init__()
        self.size = size
        self.fill = fill

    def forward(self, image, target=None):
        # Resize by short side (preserve aspect ratio)
        # image = F.resize(image, self.size, antialias=True)

        _, h, w = F.get_dimensions(image)

        pad_h = self.size - h
        pad_w = self.size - w

        # Pad equally left/right and top/bottom
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        image = F.pad(
            image,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill
        )

        # Update bounding boxes if present
        if target is not None and "bboxes" in target:
            boxes = target["bboxes"].clone()
            boxes[:, [0, 2]] += pad_left
            boxes[:, [1, 3]] += pad_top
            target = dict(target)
            target["bboxes"] = tv_tensors.BoundingBoxes(
                boxes,
                format=target["bboxes"].format,
                canvas_size=(self.size, self.size)  # Updated canvas size after padding
            )

        return image, target
