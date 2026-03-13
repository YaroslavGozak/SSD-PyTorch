import multiprocessing

import torch
import math
import random
from collections import Counter
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors


MULTI_SCALE_SIZES = [
    (300, 300),
    (268, 300),
    (268, 268),
    (140, 300),
    (140, 140),
    (96, 96),
    (300, 96),
    (64, 300),
    (64, 64),
    (32, 32),
    (300, 32),
]


def _roi_crop_single(
    image: torch.Tensor,
    target: dict,
    size: tuple,
    delta_x: float = 16.0,
    delta_y: float = 16.0,
    delta_random: float = 0.5,
    min_box_area: float = 4.0,
    fill: tuple = (0.5, 0.5, 0.5),
):
    """
    Crop a single image around a randomly selected target box.

    Args:
        image:         (C, H, W) float tensor, normalized (imagenet stats)
        target:        dict with 'bboxes' normalized [0,1] (XYXY), 'labels', 'difficult'
        size:          (crop_h, crop_w) desired output size
        delta_x:       minimum horizontal padding around target (pixels)
        delta_y:       minimum vertical padding around target (pixels)
        delta_random:  random variation factor applied to deltas  [0..1]
        min_box_area:  minimum box area (px^2) to keep after crop
        fill:          padding fill value (should match imagenet normalised background)

    Returns:
        (image_cropped, new_target) — image is (C, crop_h, crop_w),
        bboxes are still normalized [0,1]
    """
    crop_h, crop_w = size
    _, H, W = image.shape

    bboxes_norm = target.get("bboxes", None)
    if bboxes_norm is None or bboxes_norm.numel() == 0:
        # No boxes — just resize to target size
        image_resized = F.resize(image, size=[crop_h, crop_w], antialias=True)
        return image_resized, target

    # ------------------------------------------------------------------ #
    # 1) Denormalize boxes to pixel coords for this image
    # ------------------------------------------------------------------ #
    wh = torch.tensor([[W, H, W, H]], dtype=torch.float32)
    boxes = bboxes_norm.float() * wh          # (N, 4) pixel coords

    N = boxes.shape[0]
    seed_idx = torch.randint(0, N, (1,)).item()

    x1, y1, x2, y2 = (boxes[seed_idx, i].item() for i in range(4))
    box_w = x2 - x1
    box_h = y2 - y1

    # ------------------------------------------------------------------ #
    # 2) Compute padded ROI size with random variation
    # ------------------------------------------------------------------ #
    rand_x = 1.0 + torch.rand(1).item() * delta_random
    rand_y = 1.0 + torch.rand(1).item() * delta_random
    pad_x = delta_x * rand_x
    pad_y = delta_y * rand_y

    required_w = box_w + 2 * pad_x
    required_h = box_h + 2 * pad_y

    # ------------------------------------------------------------------ #
    # 3) Downscale image if the target+padding doesn't fit in the crop
    # ------------------------------------------------------------------ #
    scale = 1.0
    if required_w > crop_w or required_h > crop_h:
        scale_w = crop_w / required_w if required_w > crop_w else 1.0
        scale_h = crop_h / required_h if required_h > crop_h else 1.0
        scale = min(scale_w, scale_h) * 0.95  # small safety margin

        new_h = max(1, int(H * scale))
        new_w = max(1, int(W * scale))
        image = F.resize(image, size=[new_h, new_w], antialias=True)
        boxes = boxes * scale
        H, W = new_h, new_w

        # Refresh seed box after scaling
        x1, y1, x2, y2 = (boxes[seed_idx, i].item() for i in range(4))

    # ------------------------------------------------------------------ #
    # 4) Pad image if it is smaller than the desired crop size
    # ------------------------------------------------------------------ #
    if W < crop_w or H < crop_h:
        pad_left   = (crop_w - W) // 2 if W < crop_w else 0
        pad_top    = (crop_h - H) // 2 if H < crop_h else 0
        pad_right  = max(0, crop_w - W - pad_left)
        pad_bottom = max(0, crop_h - H - pad_top)

        image = F.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=fill)
        boxes[:, 0::2] += pad_left
        boxes[:, 1::2] += pad_top
        H, W = image.shape[-2], image.shape[-1]

        # Refresh seed box after padding shift
        x1, y1, x2, y2 = (boxes[seed_idx, i].item() for i in range(4))

    # ------------------------------------------------------------------ #
    # 5) Decide crop origin so the seed box (+ padding) sits inside
    # ------------------------------------------------------------------ #
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    # How far we can randomly shift the crop center while still fully
    # containing the padded target
    max_off_x = max(0.0, (crop_w - (x2 - x1)) / 2.0 - pad_x)
    max_off_y = max(0.0, (crop_h - (y2 - y1)) / 2.0 - pad_y)

    off_x = (torch.rand(1).item() - 0.5) * 2.0 * max_off_x
    off_y = (torch.rand(1).item() - 0.5) * 2.0 * max_off_y

    crop_x1 = center_x - crop_w / 2.0 + off_x
    crop_y1 = center_y - crop_h / 2.0 + off_y

    # Clamp so the crop window stays inside the (possibly padded) image
    crop_x1 = max(0.0, min(crop_x1, W - crop_w))
    crop_y1 = max(0.0, min(crop_y1, H - crop_h))

    left = int(math.floor(crop_x1))
    top  = int(math.floor(crop_y1))

    # Actual crop dims (may be smaller than desired at image edges)
    actual_w = min(crop_w, W - left)
    actual_h = min(crop_h, H - top)

    image_cropped = F.crop(image, top=top, left=left, height=actual_h, width=actual_w)

    # Pad to exact size if needed (edge case)
    if actual_w < crop_w or actual_h < crop_h:
        image_cropped = F.pad(
            image_cropped,
            [0, 0, crop_w - actual_w, crop_h - actual_h],
            fill=fill,
        )

    # ------------------------------------------------------------------ #
    # 6) Update boxes → shift + clamp + filter
    # ------------------------------------------------------------------ #
    new_boxes = boxes.clone()
    new_boxes[:, 0::2] -= left
    new_boxes[:, 1::2] -= top
    new_boxes[:, 0::2] = new_boxes[:, 0::2].clamp(0, crop_w)
    new_boxes[:, 1::2] = new_boxes[:, 1::2].clamp(0, crop_h)

    bw = new_boxes[:, 2] - new_boxes[:, 0]
    bh = new_boxes[:, 3] - new_boxes[:, 1]
    keep = (bw * bh) >= min_box_area

    if keep.sum() == 0:
        # Fallback: just resize the original image, keep all boxes
        image_resized = F.resize(image, size=[crop_h, crop_w], antialias=True)
        return image_resized, target

    new_boxes = new_boxes[keep]

    # ------------------------------------------------------------------ #
    # 7) Renormalize boxes to [0, 1] with the new canvas size
    # ------------------------------------------------------------------ #
    wh_new = torch.tensor([[crop_w, crop_h, crop_w, crop_h]], dtype=torch.float32)
    new_boxes_norm = new_boxes / wh_new

    if hasattr(bboxes_norm, "format"):
        new_boxes_norm = tv_tensors.BoundingBoxes(
            new_boxes_norm,
            format=bboxes_norm.format,
            canvas_size=(crop_h, crop_w),
        )

    new_target = dict(target)
    new_target["bboxes"] = new_boxes_norm
    if "labels" in target:
        new_target["labels"] = target["labels"][keep]
    if "difficult" in target:
        new_target["difficult"] = target["difficult"][keep]

    return image_cropped, new_target


# ------------------------------------------------------------------ #
#  Public collate function
# ------------------------------------------------------------------ #

def multi_scale_collate_fn(
    batch,
    sizes=None,
    delta_x: float = 8.0,
    delta_y: float = 8.0,
    delta_random: float = 0.5,
    min_box_area: float = 4.0,
    fill: tuple = (0.5, 0.5, 0.5),
):
    """
    Collate function that:
      1. Picks a random (H, W) size for the whole batch.
      2. For each image, crops around a random target box with padding.
      3. Stacks into a (B, C, H, W) tensor.

    Bboxes come in normalized [0,1] from VOCDataset.__getitem__ and are
    returned normalized [0,1] relative to the new crop canvas.

    Args:
        batch:        list of (image_tensor, target_dict, filename)
        sizes:        list of (H, W) tuples to sample from
        delta_x:      min horizontal padding around target (px, pre-scale)
        delta_y:      min vertical padding around target (px, pre-scale)
        delta_random: random variation factor for padding  [0..1]
        min_box_area: minimum box area to keep after crop (px^2)
        fill:         fill colour for padding (normalised float per channel)
    """
    if sizes is None:
        sizes = MULTI_SCALE_SIZES

    # One size for the entire batch
    target_size = random.choice(sizes)   # (H, W)

    images, targets, filenames = [], [], []
    for im_tensor, target, filename in batch:
        img_c, new_target = _roi_crop_single(
            im_tensor, target, target_size,
            delta_x=delta_x,
            delta_y=delta_y,
            delta_random=delta_random,
            min_box_area=min_box_area,
            fill=fill,
        )
        images.append(img_c)
        targets.append(new_target)
        filenames.append(filename)

    return torch.stack(images, dim=0), targets, filenames


class EpochAwareCollateFn:
    """Collate function that progressively unlocks smaller crop sizes as training
    progresses (curriculum learning).

    At epoch 0 only the largest size(s) are used; by the final epoch the full
    ``all_sizes`` pool is available.  Set ``collate_fn.epoch = i`` at the start
    of every epoch so the loader picks up the updated schedule automatically —
    no need to recreate the DataLoader.

    Args:
        num_epochs:   total number of training epochs (used to compute the
                      schedule; does not need to be exact)
        all_sizes:    ordered list of (H, W) tuples, largest first.
                      Defaults to ``MULTI_SCALE_SIZES``.
        fill:         padding fill colour (normalised float per channel)
        delta_x:      min horizontal padding around the seed box (px)
        delta_y:      min vertical padding around the seed box (px)
        delta_random: random variation factor for padding  [0..1]
        min_box_area: minimum box area to keep after crop (px²)

    Example::

        collate_fn = EpochAwareCollateFn(num_epochs=120, fill=(...))
        loader = DataLoader(dataset, collate_fn=collate_fn)

        for epoch in range(num_epochs):
            collate_fn.epoch = epoch          # <-- unlock sizes for this epoch
            for batch in loader:
                ...
    """

    def __init__(
        self,
        num_epochs: int,
        all_sizes=None,
        fill: tuple = (0.5, 0.5, 0.5),
        delta_x: float = 8.0,
        delta_y: float = 8.0,
        delta_random: float = 0.5,
        min_box_area: float = 4.0,
    ):
        self.num_epochs = num_epochs
        # Sizes should be ordered large → small (MULTI_SCALE_SIZES already is)
        self.all_sizes = all_sizes if all_sizes is not None else MULTI_SCALE_SIZES
        self.fill = fill
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_random = delta_random
        self.min_box_area = min_box_area
        self.epoch = 0  # updated externally before each epoch
        # Shared across worker processes
        # self._manager = multiprocessing.Manager()
        # self._size_counts = self._manager.dict()  # {(H, W): int}

    def active_sizes(self):
        """Return the crop-size pool that is unlocked for the current epoch.

        Sizes are unlocked linearly from large to small:
        * epoch 0              → only ``all_sizes[0]`` (largest / easiest)
        * epoch num_epochs - 1 → all sizes
        """
        if self.epoch == 0:
            raise ValueError("EpochAwareCollateFn.epoch is not set. Please set collate_fn.epoch = i at the start of each epoch.")
        n = len(self.all_sizes)
        unlocked = max(1, round(1 + (n - 1) * self.epoch / max(1, self.num_epochs - 1)))
        return self.all_sizes[:unlocked]

    # def print_and_reset_stats(self, epoch: int) -> None:
    #     """Print a one-line crop-size summary for *epoch* then reset counters."""
    #     counts = dict(self._size_counts)  # snapshot from shared dict
    #     if not counts:
    #         print(f"Epoch {epoch + 1} crop-size stats: no data collected")
    #         return
    #     total = sum(counts.values())
    #     # Sort by the order they appear in all_sizes so output is consistent
    #     size_order = {s: idx for idx, s in enumerate(self.all_sizes)}
    #     sorted_counts = sorted(counts.items(),
    #                            key=lambda kv: size_order.get(kv[0], 999))
    #     parts = [f"{h}x{w}: {n} ({100*n/total:.1f}%)"
    #              for (h, w), n in sorted_counts]
    #     print(f"Epoch {epoch + 1} crop-size stats ({total} images total) | "
    #           + " | ".join(parts))
    #     self._size_counts.clear()

    def __call__(self, batch):
        sizes = self.active_sizes()
        chosen_size = random.choice(sizes)
        # Manager dict doesn't support += directly, must read-modify-write
        # self._size_counts[chosen_size] = self._size_counts.get(chosen_size, 0) + len(batch)
        # print(f"Collate epoch {self.epoch + 1} | self._size_counts: {self._size_counts}")
        return multi_scale_collate_fn(
            batch,
            sizes=[chosen_size],
            delta_x=self.delta_x,
            delta_y=self.delta_y,
            delta_random=self.delta_random,
            min_box_area=self.min_box_area,
            fill=self.fill,
        )