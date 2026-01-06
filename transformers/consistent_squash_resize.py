import math
import torch
import torchvision
from torchvision.transforms.v2 import functional as F

class ConsistentSquashResize(torch.nn.Module):
    """
    Resize a crop using the same anisotropic scaling as full-frame Resize((S,S)).
    Given original image size (W0,H0) and target square size S:
      sx = S/W0, sy = S/H0
      output crop size becomes (round(w*sx), round(h*sy)).
    This preserves consistency in the normalized full-frame square space.
    """

    def __init__(self, size: int, min_size: int = 1):
        super().__init__()
        self.size = int(size)
        self.min_size = int(min_size)

    def forward(self, image, target=None):
        # Original size must be known for consistent mapping.
        # Expect target["orig_size"] = (H0, W0) or infer from target if you store it elsewhere.
        if target is None or "orig_size" not in target:
            raise ValueError('target["orig_size"]=(H0,W0) is required for ConsistentSquashResize')

        H0, W0 = target["orig_size"]
        _, H, W = F.get_dimensions(image)

        sx = self.size / float(W0)
        sy = self.size / float(H0)

        out_w = max(self.min_size, int(round(W * sx)))
        out_h = max(self.min_size, int(round(H * sy)))

        # Resize image
        image = F.resize(image, size=[out_h, out_w], interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)

        # Resize boxes if present
        if target is not None and "boxes" in target:
            boxes = target["boxes"].clone()
            # boxes are in (x1,y1,x2,y2) in the crop coordinate system
            boxes[:, [0, 2]] *= (out_w / float(W))
            boxes[:, [1, 3]] *= (out_h / float(H))
            target = dict(target)
            target["boxes"] = boxes

        return image, target
