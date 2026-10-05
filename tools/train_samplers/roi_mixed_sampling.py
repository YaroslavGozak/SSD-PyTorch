from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import random
import math
from numbers import Integral

import torch
import torch.nn.functional as F


# =========================================================
# Batch mode configuration
# =========================================================

@dataclass
class BatchModeConfig:
    name: str
    out_size: int
    prob: float
    padding_px_range: Tuple[int, int]
    center_jitter_ratio: float = 0.0
    scale_jitter_ratio: float = 0.0
    min_box_visibility: float = 0.3


def build_stage_mode_configs(stage: int, im_size: int = 300):
    """
    Scale output sizes relative to the original 300-pixel full-frame schedule.

    Round ROI sides to the nearest pixel (half up), with a minimum of one.
    Padding is measured on the full image after its square resize; no stride
    alignment is imposed.
    """
    if isinstance(im_size, bool) or not isinstance(im_size, Integral) or im_size <= 0:
        raise ValueError("Mixed sampling requires a positive integer dataset_params.im_size")

    def scaled(size):
        return max(1, (size * int(im_size) + 150) // 300)

    if stage == 2:
        return [
            BatchModeConfig(
                name="full",
                out_size=scaled(300),
                prob=0.50,
                padding_px_range=(0, 0),
                center_jitter_ratio=0.0,
                scale_jitter_ratio=0.0,
            ),
            BatchModeConfig(
                name="large_roi",
                out_size=scaled(224),
                prob=0.20,
                padding_px_range=(70, 120),
                center_jitter_ratio=0.05,
                scale_jitter_ratio=0.05,
            ),
            BatchModeConfig(
                name="medium_roi",
                out_size=scaled(160),
                prob=0.10,
                padding_px_range=(10, 50),
                center_jitter_ratio=0.10,
                scale_jitter_ratio=0.10,
            ),
            BatchModeConfig(
                name="tight_roi",
                out_size=scaled(96),
                prob=0.10,
                padding_px_range=(0, 30),
                center_jitter_ratio=0.15,
                scale_jitter_ratio=0.15,
                min_box_visibility=0.5,
            ),
            BatchModeConfig(
                name="gt_roi",
                out_size=scaled(64),
                prob=0.10,
                padding_px_range=(0, 5),
                center_jitter_ratio=0.05,
                scale_jitter_ratio=0.05,
                min_box_visibility=0.5,
            ),
        ]
    elif stage == 3:
        return [
            BatchModeConfig(
                name="full",
                out_size=scaled(300),
                prob=0.30,
                padding_px_range=(0, 0),
                center_jitter_ratio=0.0,
                scale_jitter_ratio=0.0,
            ),
            BatchModeConfig(
                name="xlarge_roi",
                out_size=scaled(224),
                prob=0.10,
                padding_px_range=(80, 150),
                center_jitter_ratio=0.05,
                scale_jitter_ratio=0.05,
            ),
            BatchModeConfig(
                name="large_roi",
                out_size=scaled(160),
                prob=0.10,
                padding_px_range=(30, 80),
                center_jitter_ratio=0.10,
                scale_jitter_ratio=0.10,
            ),
            BatchModeConfig(
                name="medium_roi",
                out_size=scaled(96),
                prob=0.20,
                padding_px_range=(0, 30),
                center_jitter_ratio=0.15,
                scale_jitter_ratio=0.15,
                min_box_visibility=0.5,
            ),
            BatchModeConfig(
                name="tight_roi",
                out_size=scaled(64),
                prob=0.30,
                padding_px_range=(1, 10),
                center_jitter_ratio=0.05,
                scale_jitter_ratio=0.05,
                min_box_visibility=0.5,
            ),
            BatchModeConfig(
                name="gt_roi",
                out_size=scaled(32),
                prob=0.30,
                padding_px_range=(1, 10),
                center_jitter_ratio=0.05,
                scale_jitter_ratio=0.05,
                min_box_visibility=0.5,
            ),
        ]
    else:
        raise ValueError(f"Unsupported stage: {stage}")


# =========================================================
# Batch sampler
# =========================================================

class MixedBatchSampler(torch.utils.data.Sampler):
    """
    Returns batches of tuples:
        (dataset_index, mode_name, out_size)

    DataLoader will call dataset[item] for each tuple item in the batch.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        stage: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        im_size: int = 300,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.stage = stage
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self.mode_configs = build_stage_mode_configs(stage, im_size=im_size)
        self.indices = list(range(len(dataset)))

        probs = [m.prob for m in self.mode_configs]
        s = sum(probs)
        self.mode_probs = [p / s for p in probs]

    def __iter__(self):
        rng = random.Random(self.seed + random.randint(0, 10_000_000))

        indices = self.indices.copy()
        if self.shuffle:
            rng.shuffle(indices)

        n_full_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size != 0:
            n_full_batches += 1

        ptr = 0
        for _ in range(n_full_batches):
            batch_indices = indices[ptr: ptr + self.batch_size]
            ptr += self.batch_size

            if len(batch_indices) < self.batch_size and self.drop_last:
                break

            mode_cfg = rng.choices(self.mode_configs, weights=self.mode_probs, k=1)[0]

            batch = [
                (idx, mode_cfg.name, mode_cfg.out_size)
                for idx in batch_indices
            ]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return math.ceil(len(self.indices) / self.batch_size)
    

# =========================================================
# Geometry helpers
# =========================================================

def boxes_norm_to_abs(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    return boxes


def boxes_abs_to_norm(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, [0, 2]] /= max(w, 1)
    boxes[:, [1, 3]] /= max(h, 1)
    return boxes


def clip_boxes_xyxy(boxes: torch.Tensor, x1: float, y1: float, x2: float, y2: float) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(min=x1, max=x2)
    boxes[:, 1] = boxes[:, 1].clamp(min=y1, max=y2)
    boxes[:, 2] = boxes[:, 2].clamp(min=x1, max=x2)
    boxes[:, 3] = boxes[:, 3].clamp(min=y1, max=y2)
    return boxes


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def filter_boxes_by_visibility(
    boxes_before_clip: torch.Tensor,
    boxes_after_clip: torch.Tensor,
    min_visibility: float,
    min_size_px: float = 2.0,
) -> torch.Tensor:
    area_before = box_area_xyxy(boxes_before_clip)
    area_after = box_area_xyxy(boxes_after_clip)

    visibility = torch.zeros_like(area_after)
    valid_before = area_before > 0
    visibility[valid_before] = area_after[valid_before] / area_before[valid_before]

    widths = (boxes_after_clip[:, 2] - boxes_after_clip[:, 0]).clamp(min=0)
    heights = (boxes_after_clip[:, 3] - boxes_after_clip[:, 1]).clamp(min=0)

    keep = (
        (visibility >= min_visibility) &
        (widths >= min_size_px) &
        (heights >= min_size_px)
    )
    return keep


def choose_reference_box(boxes_abs: torch.Tensor) -> int:
    """
    Picks one GT box for ROI generation.
    You can later replace this with area-weighted sampling if needed.
    """
    num_boxes = boxes_abs.shape[0]
    return random.randrange(num_boxes)


def perturb_box_xyxy(
    box: torch.Tensor,
    img_w: int,
    img_h: int,
    center_jitter_ratio: float,
    scale_jitter_ratio: float,
) -> torch.Tensor:
    """
    Applies mild jitter to simulate imperfect previous-frame detection.
    """
    x1, y1, x2, y2 = box.tolist()
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    dx = random.uniform(-center_jitter_ratio, center_jitter_ratio) * bw
    dy = random.uniform(-center_jitter_ratio, center_jitter_ratio) * bh

    sx = 1.0 + random.uniform(-scale_jitter_ratio, scale_jitter_ratio)
    sy = 1.0 + random.uniform(-scale_jitter_ratio, scale_jitter_ratio)

    new_bw = max(bw * sx, 2.0)
    new_bh = max(bh * sy, 2.0)
    new_cx = cx + dx
    new_cy = cy + dy

    nx1 = max(0.0, new_cx - 0.5 * new_bw)
    ny1 = max(0.0, new_cy - 0.5 * new_bh)
    nx2 = min(float(img_w), new_cx + 0.5 * new_bw)
    ny2 = min(float(img_h), new_cy + 0.5 * new_bh)

    if nx2 <= nx1:
        nx2 = min(float(img_w), nx1 + 2.0)
    if ny2 <= ny1:
        ny2 = min(float(img_h), ny1 + 2.0)

    return torch.tensor([nx1, ny1, nx2, ny2], dtype=torch.float32)


def make_roi_crop_box(
    ref_box: torch.Tensor,
    img_w: int,
    img_h: int,
    padding_px_range: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Expands the reference box by random padding and clips to image boundaries.
    """
    pad = random.randint(padding_px_range[0], padding_px_range[1])

    x1, y1, x2, y2 = ref_box.tolist()

    crop_x1 = max(0, int(math.floor(x1 - pad)))
    crop_y1 = max(0, int(math.floor(y1 - pad)))
    crop_x2 = min(img_w, int(math.ceil(x2 + pad)))
    crop_y2 = min(img_h, int(math.ceil(y2 + pad)))

    # Safety fallback
    if crop_x2 <= crop_x1:
        crop_x2 = min(img_w, crop_x1 + 2)
    if crop_y2 <= crop_y1:
        crop_y2 = min(img_h, crop_y1 + 2)

    return crop_x1, crop_y1, crop_x2, crop_y2


# =========================================================
# Batch processor
# =========================================================

class RoiBatchProcessor:
    """
    Resize each full image to im_size square, then take unscaled square ROIs.
    All images in a batch use the same ROI side length.
    """

    def __init__(
        self,
        image_only_transform=None,
        normalize_transform=None,
        mode_configs=None,
        im_size=None,
    ):
        self.image_only_transform = image_only_transform
        self.normalize_transform = normalize_transform
        self.mode_cfg_map = {}

        modes = mode_configs if mode_configs is not None else build_stage_mode_configs(2)
        for cfg in modes:
            if cfg.name in self.mode_cfg_map:
                raise ValueError(f"Duplicate sampling mode: {cfg.name}")
            self.mode_cfg_map[cfg.name] = cfg
        self.im_size = int(im_size if im_size is not None else self.mode_cfg_map['full'].out_size)
        if self.im_size <= 0:
            raise ValueError('im_size must be positive')

    def _resize_frame(self, image):
        image = image.float()
        if image.numel() and image.max() > 1.0:
            image = image / 255.0
        return F.interpolate(
            image.unsqueeze(0), size=(self.im_size, self.im_size),
            mode='bilinear', align_corners=False,
        ).squeeze(0)

    def _finish(self, image, boxes_abs, labels, side):
        target = {
            'boxes': boxes_abs_to_norm(boxes_abs, side, side).clamp(0.0, 1.0),
            'labels': labels,
        }
        if self.image_only_transform is not None:
            image = self.image_only_transform(image)
        if self.normalize_transform is not None:
            image = self.normalize_transform(image)
        return image, target

    def _crop_origin(self, bounds, preferred_center, side):
        # Keep all boxes already included in bounds while staying inside the frame.
        x1, y1, x2, y2 = bounds
        cx, cy = preferred_center
        def axis_start(low, high, center):
            minimum = max(0, int(math.ceil(high - side)))
            maximum = min(self.im_size - side, int(math.floor(low)))
            desired = int(round(center - side / 2))
            return max(minimum, min(desired, maximum))
        return axis_start(x1, x2, cx), axis_start(y1, y2, cy)

    def _prepare_roi(self, image, target, mode_name):
        boxes = boxes_norm_to_abs(target['boxes'], self.im_size, self.im_size)
        if boxes.numel() == 0:
            return {'image': image, 'target': target, 'boxes': boxes,
                    'bounds': None, 'center': (self.im_size / 2, self.im_size / 2)}

        cfg = self.mode_cfg_map[mode_name]
        reference_idx = choose_reference_box(boxes)
        reference = boxes[reference_idx]
        perturbed = perturb_box_xyxy(
            reference, self.im_size, self.im_size,
            cfg.center_jitter_ratio, cfg.scale_jitter_ratio,
        )
        pad = random.randint(*cfg.padding_px_range)
        bounds = [
            max(0.0, min(reference[0].item(), perturbed[0].item()) - pad),
            max(0.0, min(reference[1].item(), perturbed[1].item()) - pad),
            min(float(self.im_size), max(reference[2].item(), perturbed[2].item()) + pad),
            min(float(self.im_size), max(reference[3].item(), perturbed[3].item()) + pad),
        ]
        center = ((perturbed[0].item() + perturbed[2].item()) / 2,
                  (perturbed[1].item() + perturbed[3].item()) / 2)
        return {'image': image, 'target': target, 'boxes': boxes,
                'bounds': bounds, 'center': center, 'reference_idx': reference_idx,
                'reference_box': reference}

    def process_batch(self, batch, mode_name, out_size, return_sampling_info=False):
        prepared = []
        for sample in batch:
            image = self._resize_frame(sample['image'])
            if mode_name == 'full':
                prepared.append({'image': image, 'target': sample['target']})
            else:
                prepared.append(self._prepare_roi(image, sample['target'], mode_name))

        if mode_name == 'full':
            processed = [self._finish(item['image'],
                                      boxes_norm_to_abs(item['target']['boxes'], self.im_size, self.im_size),
                                      item['target']['labels'], self.im_size)
                         for item in prepared]
            info = {'object_mismatch': False}
            return (processed, info) if return_sampling_info else processed

        # The scheduled side is a minimum. A single larger side is used by
        # every sample so tensors remain stackable without rescaling crops.
        side = min(self.im_size, int(out_size))
        for item in prepared:
            if item['bounds'] is not None:
                x1, y1, x2, y2 = item['bounds']
                # Integer crop origins need one extra pixel in some cases even
                # when the floating-point box width rounds up to the target.
                side = max(side, math.ceil(x2) - math.floor(x1),
                           math.ceil(y2) - math.floor(y1))
        side = min(side, self.im_size)
        object_mismatch = False
        for item in prepared:
            reference = item.get('reference_box')
            if reference is not None:
                ref_x1, ref_y1, ref_x2, ref_y2 = reference.tolist()
                required_for_object = max(math.ceil(ref_x2) - math.floor(ref_x1),
                                          math.ceil(ref_y2) - math.floor(ref_y1))
                object_mismatch |= required_for_object > out_size

        processed = []
        for item in prepared:
            if item['bounds'] is None:
                left = top = (self.im_size - side) // 2
            else:
                left, top = self._crop_origin(item['bounds'], item['center'], side)
            crop = item['image'][:, top:top + side, left:left + side]
            boxes = item['boxes']
            clipped = clip_boxes_xyxy(boxes, left, top, left + side, top + side)
            cfg = self.mode_cfg_map[mode_name]
            keep = filter_boxes_by_visibility(boxes, clipped, cfg.min_box_visibility)
            if item['bounds'] is not None:
                keep[item['reference_idx']] = True
            boxes = clipped[keep].clone()
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top
            processed.append(self._finish(crop, boxes, item['target']['labels'][keep], side))
        info = {'object_mismatch': object_mismatch}
        return (processed, info) if return_sampling_info else processed

    def process_sample(
        self,
        image: torch.Tensor,
        target: Dict[str, torch.Tensor],
        mode_name: str,
        out_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.process_batch(
            [{'image': image, 'target': target}], mode_name, out_size,
        )[0]


# =========================================================
# Collate function
# =========================================================

class MixedCollateFn:
    def __init__(self, processor, return_mode=False, return_sampling_info=False):
        self.processor = processor
        self.return_mode = return_mode
        self.return_sampling_info = return_sampling_info

    def __call__(self, batch):
        assert len(batch) > 0

        mode = batch[0]["mode"]
        out_size = batch[0]["out_size"]

        for sample in batch:
            assert sample["mode"] == mode
            assert sample["out_size"] == out_size

        result = self.processor.process_batch(
            batch, mode, out_size, return_sampling_info=self.return_sampling_info,
        )
        processed, sampling_info = result if self.return_sampling_info else (result, None)
        processed_images, processed_targets = zip(*processed)

        images_tensor = torch.stack(processed_images, dim=0)

        if self.return_mode:
            if self.return_sampling_info:
                return images_tensor, processed_targets, mode, sampling_info
            return images_tensor, processed_targets, mode

        return images_tensor, processed_targets
