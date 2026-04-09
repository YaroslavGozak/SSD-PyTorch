from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import random
import math

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


def build_stage_mode_configs(stage: int):
    """
    Returns a list of BatchModeConfig for the requested stage.
    Comments are in English on purpose.
    """
    if stage == 2:
        return [
            BatchModeConfig(
                name="full",
                out_size=300,
                prob=0.50,
                padding_px_range=(0, 0),
                center_jitter_ratio=0.0,
                scale_jitter_ratio=0.0,
            ),
            BatchModeConfig(
                name="large_roi",
                out_size=224,   # Start simple: keep all 300
                prob=0.20,
                padding_px_range=(70, 120),
                center_jitter_ratio=0.05,
                scale_jitter_ratio=0.05,
            ),
            BatchModeConfig(
                name="medium_roi",
                out_size=160,
                prob=0.10,
                padding_px_range=(10, 50),
                center_jitter_ratio=0.10,
                scale_jitter_ratio=0.10,
            ),
            BatchModeConfig(
                name="tight_roi",
                out_size=96,
                prob=0.10,
                padding_px_range=(0, 30),
                center_jitter_ratio=0.15,
                scale_jitter_ratio=0.15,
                min_box_visibility=0.5,
            ),
            BatchModeConfig(
                name="gt_roi",
                out_size=64,
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
                out_size=300,
                prob=0.30,
                padding_px_range=(0, 0),
                center_jitter_ratio=0.0,
                scale_jitter_ratio=0.0,
            ),
            BatchModeConfig(
                name="large_roi",
                out_size=224,
                prob=0.20,
                padding_px_range=(80, 150),
                center_jitter_ratio=0.05,
                scale_jitter_ratio=0.05,
            ),
            BatchModeConfig(
                name="medium_roi",
                out_size=160,
                prob=0.20,
                padding_px_range=(30, 80),
                center_jitter_ratio=0.10,
                scale_jitter_ratio=0.10,
            ),
            BatchModeConfig(
                name="tight_roi",
                out_size=96,
                prob=0.20,
                padding_px_range=(0, 30),
                center_jitter_ratio=0.15,
                scale_jitter_ratio=0.15,
                min_box_visibility=0.5,
            ),
            BatchModeConfig(
                name="gt_roi",
                out_size=64,
                prob=0.10,
                padding_px_range=(0, 5),
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
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.stage = stage
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self.mode_configs = build_stage_mode_configs(stage)
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
    Expects image tensors in CHW format.
    Expects target['boxes'] in normalized [0,1] xyxy format.
    Expects target['labels'] as int64.
    """

    def __init__(
        self,
        image_only_transform=None,
        normalize_transform=None,
    ):
        self.image_only_transform = image_only_transform
        self.normalize_transform = normalize_transform
        self.mode_cfg_map = {}

        for stage in [2, 3]:
            for cfg in build_stage_mode_configs(stage):
                self.mode_cfg_map[cfg.name] = cfg

    def process_sample(
        self,
        image: torch.Tensor,
        target: Dict[str, torch.Tensor],
        mode_name: str,
        out_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if image.dtype != torch.float32:
            image = image.float()

        if image.max() > 1.0:
            image = image / 255.0

        _, img_h, img_w = image.shape

        boxes_norm = target["boxes"]
        labels = target["labels"]

        if boxes_norm.numel() == 0:
            # No GT: fallback to full-frame resize
            return self._process_full(image, target, out_size)

        if mode_name == "full":
            return self._process_full(image, target, out_size)

        cfg = self.mode_cfg_map[mode_name]
        boxes_abs = boxes_norm_to_abs(boxes_norm, img_w, img_h)

        ref_idx = choose_reference_box(boxes_abs)
        ref_box = boxes_abs[ref_idx]
        ref_box = perturb_box_xyxy(
            ref_box,
            img_w=img_w,
            img_h=img_h,
            center_jitter_ratio=cfg.center_jitter_ratio,
            scale_jitter_ratio=cfg.scale_jitter_ratio,
        )

        crop_x1, crop_y1, crop_x2, crop_y2 = make_roi_crop_box(
            ref_box,
            img_w=img_w,
            img_h=img_h,
            padding_px_range=cfg.padding_px_range,
        )

        crop = image[:, crop_y1:crop_y2, crop_x1:crop_x2]

        boxes_before_clip = boxes_abs.clone()
        boxes_after_clip = clip_boxes_xyxy(
            boxes_abs,
            x1=float(crop_x1),
            y1=float(crop_y1),
            x2=float(crop_x2),
            y2=float(crop_y2),
        )

        keep = filter_boxes_by_visibility(
            boxes_before_clip=boxes_before_clip,
            boxes_after_clip=boxes_after_clip,
            min_visibility=cfg.min_box_visibility,
            min_size_px=2.0,
        )

        # Always keep reference object if it still has valid area after clipping
        ref_area = box_area_xyxy(boxes_after_clip[ref_idx:ref_idx + 1])[0]
        if ref_area > 0:
            keep[ref_idx] = True

        boxes_after_clip = boxes_after_clip[keep]
        labels_after_clip = labels[keep]

        # Fallback if crop became invalid after filtering
        if boxes_after_clip.numel() == 0:
            return self._process_full(image, target, out_size)

        # Remap to crop-local coordinates
        boxes_after_clip[:, [0, 2]] -= crop_x1
        boxes_after_clip[:, [1, 3]] -= crop_y1

        crop_h = crop_y2 - crop_y1
        crop_w = crop_x2 - crop_x1

        # Resize crop
        crop = F.interpolate(
            crop.unsqueeze(0),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        sx = out_size / max(crop_w, 1)
        sy = out_size / max(crop_h, 1)

        boxes_after_clip[:, [0, 2]] *= sx
        boxes_after_clip[:, [1, 3]] *= sy

        boxes_out = boxes_abs_to_norm(boxes_after_clip, out_size, out_size).clamp(0.0, 1.0)

        new_target = {
            "boxes": boxes_out,
            "labels": labels_after_clip,
        }

        if self.image_only_transform is not None:
            crop = self.image_only_transform(crop)

        if self.normalize_transform is not None:
            crop = self.normalize_transform(crop)

        return crop, new_target

    def _process_full(
        self,
        image: torch.Tensor,
        target: Dict[str, torch.Tensor],
        out_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, img_h, img_w = image.shape

        image_resized = F.interpolate(
            image.unsqueeze(0),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        boxes_abs = boxes_norm_to_abs(target["boxes"], img_w, img_h)
        boxes_abs[:, [0, 2]] *= out_size / max(img_w, 1)
        boxes_abs[:, [1, 3]] *= out_size / max(img_h, 1)
        boxes_out = boxes_abs_to_norm(boxes_abs, out_size, out_size).clamp(0.0, 1.0)

        new_target = {
            "boxes": boxes_out,
            "labels": target["labels"],
        }

        if self.image_only_transform is not None:
            image_resized = self.image_only_transform(image_resized)

        if self.normalize_transform is not None:
            image_resized = self.normalize_transform(image_resized)

        return image_resized, new_target


# =========================================================
# Collate function
# =========================================================

class MixedCollateFn:
    def __init__(self, processor, return_mode=False):
        self.processor = processor
        self.return_mode = return_mode

    def __call__(self, batch):
        assert len(batch) > 0

        mode = batch[0]["mode"]
        out_size = batch[0]["out_size"]

        processed_images = []
        processed_targets = []

        for sample in batch:
            assert sample["mode"] == mode
            assert sample["out_size"] == out_size

            img, tgt = self.processor.process_sample(
                image=sample["image"],
                target=sample["target"],
                mode_name=sample["mode"],
                out_size=sample["out_size"],
            )
            processed_images.append(img)
            processed_targets.append(tgt)

        images_tensor = torch.stack(processed_images, dim=0)

        if self.return_mode:
            return images_tensor, processed_targets, mode

        return images_tensor, processed_targets