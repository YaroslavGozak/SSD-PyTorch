import os
import math

import torch
import torchvision.transforms.v2
from torchvision.io import write_png
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from transformers.resize_longer_edge import ResizeLongerEdge


class FixedPaddingRoiCropYOLOTestTransform:
    """
    YOLO-oriented fixed-padding test transform.

    Pipeline:
    1) Resize full image longer edge to 300
    2) Crop around first object with fixed padding
    3) Resize crop longer edge to nearest stride multiple (default 32)
    4) Make square (pad shorter side)
    """

    def __init__(
        self,
        im_size,
        imagenet_mean,
        imagenet_std,
        pad_x,
        pad_y,
        stride=32,
        min_box_area=4.0,
        letterbox_fill=114,
    ):
        self.base_resize = ResizeLongerEdge(size=im_size)
        self.pad_x = float(pad_x)
        self.pad_y = float(pad_y)
        self.stride = int(stride)
        self.min_box_area = float(min_box_area)
        self.letterbox_fill = int(letterbox_fill)
        self.debug_dir = os.environ.get('YOLO_ROI_DEBUG_DIR', '').strip()
        self.debug_max = int(os.environ.get('YOLO_ROI_DEBUG_MAX', '0') or 0)
        self._debug_count = 0
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
        self.transforms = self._get_transforms(imagenet_mean, imagenet_std)

    def _labels_getter(self, transform_input):
        return (transform_input[1]["labels"], transform_input[1]["difficult"])

    def _nearest_stride(self, value):
        v = max(1, int(round(float(value))))
        return max(self.stride, int(round(v / self.stride) * self.stride))

    def _expand_to_square_within_image(self, x1, y1, x2, y2, image_w, image_h):
        """Expand ROI to a square by using available image margins first."""
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0 or abs(w - h) < 1e-6:
            return x1, y1, x2, y2

        if w > h:
            need = w - h
            add_top = min(y1, need / 2.0)
            add_bottom = min(float(image_h) - y2, need - add_top)
            rem = need - add_top - add_bottom
            if rem > 0:
                extra_top = min(y1 - add_top, rem)
                add_top += max(0.0, extra_top)
                rem -= max(0.0, extra_top)
            if rem > 0:
                extra_bottom = min(float(image_h) - y2 - add_bottom, rem)
                add_bottom += max(0.0, extra_bottom)
            y1 -= add_top
            y2 += add_bottom
        else:
            need = h - w
            add_left = min(x1, need / 2.0)
            add_right = min(float(image_w) - x2, need - add_left)
            rem = need - add_left - add_right
            if rem > 0:
                extra_left = min(x1 - add_left, rem)
                add_left += max(0.0, extra_left)
                rem -= max(0.0, extra_left)
            if rem > 0:
                extra_right = min(float(image_w) - x2 - add_right, rem)
                add_right += max(0.0, extra_right)
            x1 -= add_left
            x2 += add_right

        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(image_w), x2)
        y2 = min(float(image_h), y2)
        return x1, y1, x2, y2

    def _crop_first_with_padding(self, image, target):
        boxes = target.get("boxes", target.get("bboxes", None))
        if boxes is None or boxes.numel() == 0:
            return image, target

        _, h, w = F.get_dimensions(image)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).clone()

        # Deterministic object choice: first object in annotation order.
        x1, y1, x2, y2 = boxes_t[0].tolist()
        rx1 = max(0.0, x1 - self.pad_x)
        ry1 = max(0.0, y1 - self.pad_y)
        rx2 = min(float(w), x2 + self.pad_x)
        ry2 = min(float(h), y2 + self.pad_y)

        # Step 4 part A: extend crop toward square using available margins.
        rx1, ry1, rx2, ry2 = self._expand_to_square_within_image(rx1, ry1, rx2, ry2, w, h)

        if rx2 <= rx1 or ry2 <= ry1:
            return image, target

        rx1_i = int(math.floor(rx1))
        ry1_i = int(math.floor(ry1))
        rx2_i = int(math.ceil(rx2))
        ry2_i = int(math.ceil(ry2))

        cw = rx2_i - rx1_i
        ch = ry2_i - ry1_i
        if cw <= 0 or ch <= 0:
            return image, target

        image = F.crop(image, top=ry1_i, left=rx1_i, height=ch, width=cw)

        boxes_t[:, 0::2] -= rx1_i
        boxes_t[:, 1::2] -= ry1_i
        boxes_t[:, 0::2] = boxes_t[:, 0::2].clamp(min=0, max=cw)
        boxes_t[:, 1::2] = boxes_t[:, 1::2].clamp(min=0, max=ch)

        bw = boxes_t[:, 2] - boxes_t[:, 0]
        bh = boxes_t[:, 3] - boxes_t[:, 1]
        keep = (bw * bh) >= self.min_box_area
        if keep.sum() == 0:
            return image, target

        boxes_t = boxes_t[keep]
        new_target = dict(target)
        new_boxes = tv_tensors.BoundingBoxes(
            boxes_t,
            format=boxes.format,
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

    def _resize_longer_to_stride(self, image, target):
        _, h, w = F.get_dimensions(image)
        longer = max(h, w)
        snapped = self._nearest_stride(longer)

        if longer == snapped:
            return image, target

        if h >= w:
            new_h = snapped
            new_w = max(1, int(round(w * snapped / float(h))))
        else:
            new_w = snapped
            new_h = max(1, int(round(h * snapped / float(w))))

        resize = torchvision.transforms.v2.Resize(size=(new_h, new_w), antialias=True)
        return resize(image, target)

    def _pad_to_square(self, image, target):
        _, h, w = F.get_dimensions(image)
        side = max(h, w)
        if h == w:
            return image, target

        pad_h = side - h
        pad_w = side - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        image = F.pad(
            image,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=self.letterbox_fill,
        )

        boxes = target.get("boxes", target.get("bboxes", None))
        if boxes is None:
            return image, target

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).clone()
        boxes_t[:, [0, 2]] += pad_left
        boxes_t[:, [1, 3]] += pad_top
        new_target = dict(target)
        new_boxes = tv_tensors.BoundingBoxes(
            boxes_t,
            format=boxes.format,
            canvas_size=(side, side),
        )
        if "boxes" in target:
            new_target["boxes"] = new_boxes
        if "bboxes" in target:
            new_target["bboxes"] = new_boxes
        return image, new_target

    def _save_debug_image(self, stage_name, image, target):
        if not self.debug_dir or self.debug_max <= 0:
            return
        if self._debug_count >= self.debug_max:
            return

        img = image
        if not isinstance(img, torch.Tensor):
            return

        if img.dtype != torch.uint8:
            img = img.detach().cpu()
            if img.is_floating_point():
                img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            else:
                img = img.to(torch.uint8)
        else:
            img = img.detach().cpu()

        out_name = f"{self._debug_count:04d}_{stage_name}.png"
        out_path = os.path.join(self.debug_dir, out_name)
        write_png(img, out_path)

        boxes = target.get("boxes", target.get("bboxes", None))
        if boxes is not None:
            boxes_t = torch.as_tensor(boxes).detach().cpu().tolist()
        else:
            boxes_t = []
        meta_path = os.path.join(self.debug_dir, f"{self._debug_count:04d}_{stage_name}.txt")
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(f"shape_chw={tuple(img.shape)}\n")
            f.write(f"boxes_xyxy={boxes_t}\n")

        if stage_name == '04_square':
            self._debug_count += 1

    def _core(self, image, target):
        image, target = self.base_resize(image, target)
        self._save_debug_image('01_resize300', image, target)
        image, target = self._crop_first_with_padding(image, target)
        self._save_debug_image('02_crop_pad', image, target)
        image, target = self._resize_longer_to_stride(image, target)
        self._save_debug_image('03_stride32', image, target)
        image, target = self._pad_to_square(image, target)
        self._save_debug_image('04_square', image, target)
        return image, target

    def _get_transforms(self, imagenet_mean, imagenet_std):
        # For YOLO-style tensor input, keep [0,1] scaling and avoid ImageNet normalization.
        transforms = {
            'test': torchvision.transforms.v2.Compose([
                self._core,
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            ]),
        }
        return transforms
