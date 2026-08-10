import math
import random

import torch
import torchvision.transforms.v2
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class FixedSizeTestTransform:
    """
    Crop an exact (height, width) region around the first annotated object.

    Pipeline
    --------
    1. If the first object's bounding box exceeds (height - margin) x (width - margin),
       scale the whole image down so the object fits within that budget.
    2. Crop an exact (height, width) patch centred on the (scaled) first object,
       clamped to image boundaries.
    3. ToPureTensor → ToDtype(float32, scale=True) → [optional] ImageNet Normalize.

    If there are no annotations the image is simply resized to (height, width).

    Parameters
    ----------
    height : int
        Target crop height in pixels.
    width : int
        Target crop width in pixels.
    imagenet_mean : list[float]
        Per-channel mean for ImageNet normalisation.
    imagenet_std : list[float]
        Per-channel std for ImageNet normalisation.
    normalize : bool
        If True (default), apply ImageNet Normalize — use for SSD / RoiSSD.
        If False, skip normalisation — use for YOLO.
    margin : int
        The object must be smaller than the crop by at least this many pixels
        in each dimension.  Default is 3.
    min_box_area : float
        Minimum bounding-box area (pixels²) to keep after cropping.
    """

    def __init__(
        self,
        height: int,
        width: int,
        imagenet_mean,
        imagenet_std,
        normalize: bool = True,
        margin: int = 3,
        min_box_area: float = 4.0,
    ):
        self.height = int(height)
        self.width = int(width)
        self.margin = int(margin)
        self.min_box_area = float(min_box_area)
        self._normalize = normalize
        self._imagenet_mean = imagenet_mean
        self._imagenet_std = imagenet_std
        self.transforms = self._build_transforms()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rescale_to_fit(self, image, target):
        """
        Down-scale image (and boxes) so the first object fits within
        (height - margin) x (width - margin).  No-op if already small enough.
        """
        boxes_ref = target.get("boxes", target.get("bboxes", None))
        if boxes_ref is None or boxes_ref.numel() == 0:
            return image, target

        boxes_t = torch.as_tensor(boxes_ref, dtype=torch.float32)
        x1, y1, x2, y2 = boxes_t[0].tolist()
        obj_w = x2 - x1
        obj_h = y2 - y1

        max_obj_w = float(self.width - self.margin)
        max_obj_h = float(self.height - self.margin)

        if obj_w <= max_obj_w and obj_h <= max_obj_h:
            return image, target

        scale = min(max_obj_w / obj_w, max_obj_h / obj_h)
        _, h, w = F.get_dimensions(image)
        new_h = max(1, int(math.floor(h * scale)))
        new_w = max(1, int(math.floor(w * scale)))

        resize = torchvision.transforms.v2.Resize(size=(new_h, new_w), antialias=True)
        image, target = resize(image, target)
        return image, target

    def _crop_around_first(self, image, target):
        """
        Crop a (self.height, self.width) patch centred on the first object.
        Adjusts all bounding boxes accordingly and drops boxes below min_box_area.
        Falls back to a plain resize when there are no annotations.
        """
        boxes_ref = target.get("boxes", target.get("bboxes", None))
        _, im_h, im_w = F.get_dimensions(image)

        if boxes_ref is None or boxes_ref.numel() == 0:
            # No annotations — just resize to target
            image = F.resize(image, size=(self.height, self.width), antialias=True)
            return image, target

        boxes_t = torch.as_tensor(boxes_ref, dtype=torch.float32).clone()
        x1, y1, x2, y2 = boxes_t[0].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Ideal top-left corner so the crop is centred on the object,
        # then jitter by up to ±margin pixels in each axis.
        #
        # If the jitter would push the object closer than the available room
        # on one side, the excess is transferred to the opposite side so that
        # the total margin (left+right or top+bottom) stays at 2×margin while
        # the object never falls outside the crop.  When the image itself is
        # narrower than 2×margin we just centre as best we can.
        jitter_x = random.uniform(-self.margin, self.margin)
        jitter_y = random.uniform(-self.margin, self.margin)

        def _apply_jitter(center, size, total, jitter):
            """Return clamped top-left so crop [tl, tl+size] fits inside [0, total],
            applying jitter but moving excess to the opposite side."""
            tl = center - size / 2.0 + jitter
            # How far the object's near edge is from each crop border
            # (after jitter, before clamping to image)
            # Just clamp to [0, total-size]; the excess jitter naturally shifts
            # the margin from one side to the other.
            tl = max(0.0, min(tl, float(total - size)))
            return tl

        crop_left = _apply_jitter(cx, self.width, im_w, jitter_x)
        crop_top = _apply_jitter(cy, self.height, im_h, jitter_y)

        cl = int(math.floor(crop_left))
        ct = int(math.floor(crop_top))
        cw = min(self.width, im_w - cl)
        ch = min(self.height, im_h - ct)

        image = F.crop(image, top=ct, left=cl, height=ch, width=cw)

        # Shift boxes by crop offset
        boxes_t[:, 0::2] -= cl
        boxes_t[:, 1::2] -= ct
        boxes_t[:, 0::2] = boxes_t[:, 0::2].clamp(0, cw)
        boxes_t[:, 1::2] = boxes_t[:, 1::2].clamp(0, ch)

        # If crop is smaller than target (edge: object near image boundary after
        # clamping), resize the patch to the exact target size.
        if cw != self.width or ch != self.height:
            scale_x = self.width / cw
            scale_y = self.height / ch
            boxes_t[:, 0::2] *= scale_x
            boxes_t[:, 1::2] *= scale_y
            boxes_t[:, 0::2] = boxes_t[:, 0::2].clamp(0, self.width)
            boxes_t[:, 1::2] = boxes_t[:, 1::2].clamp(0, self.height)
            image = F.resize(image, size=(self.height, self.width), antialias=True)
            ch, cw = self.height, self.width

        bw = boxes_t[:, 2] - boxes_t[:, 0]
        bh = boxes_t[:, 3] - boxes_t[:, 1]
        keep = (bw * bh) >= self.min_box_area
        boxes_t = boxes_t[keep]

        new_target = dict(target)
        new_boxes = tv_tensors.BoundingBoxes(
            boxes_t,
            format=boxes_ref.format,
            canvas_size=(ch, cw),
        )
        if "boxes" in target:
            new_target["boxes"] = new_boxes
        if "bboxes" in target:
            new_target["bboxes"] = new_boxes
        if "labels" in target:
            new_target["labels"] = target["labels"][keep]
        if "difficult" in target:
            new_target["difficult"] = target["difficult"][keep]

        return image, new_target

    def _core(self, image, target):
        image, target = self._rescale_to_fit(image, target)
        image, target = self._crop_around_first(image, target)
        return image, target

    # ------------------------------------------------------------------
    # Transform construction
    # ------------------------------------------------------------------

    def _build_transforms(self):
        steps = [
            self._core,
            torchvision.transforms.v2.ToPureTensor(),
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
        ]
        if self._normalize:
            steps.append(
                torchvision.transforms.v2.Normalize(
                    mean=self._imagenet_mean, std=self._imagenet_std
                )
            )
        return {
            'test': torchvision.transforms.v2.Compose(steps),
        }
