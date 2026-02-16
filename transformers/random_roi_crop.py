import torch
import math
from torchvision.transforms.v2 import functional as F

class RandomROICrop(torch.nn.Module):
    """
    Random ROI-based crop for training ROI-SSD.

    - Makes crop around random object with probability p.
    - Can join neighboring objects if the combined ROI
      doesn't become too "fat" (based on relative threshold area_ratio_max).
    - Works with target["boxes"] (x1, y1, x2, y2).
    """

    def __init__(
        self,
        p: float = 0.5,
        alpha_w: float = 0.3,
        alpha_h: float = 0.3,
        delta_x: float = 8.0,
        delta_y: float = 8.0,
        area_ratio_max: float = 1.4,  # A_union / (A_i + A_j) <= 1.4 => merge ok
        min_box_area: float = 4.0,    # min area (pixels^2) to keep box after crop
    ):
        super().__init__()
        self.p = p
        self.alpha_w = alpha_w
        self.alpha_h = alpha_h
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.area_ratio_max = area_ratio_max
        self.min_box_area = min_box_area

    def forward(self, image, target):
        # target expected: dict with "boxes" (N,4) and optionally "labels"
        boxes = target.get("bboxes", None)
        if boxes is None or boxes.numel() == 0:
            print("RandomROICrop: No boxes found in target, skipping crop.")
            return image, target

        if torch.rand(1).item() > self.p:
            return image, target

        if isinstance(image, torch.Tensor):
            # C x H x W
            _, H, W = image.shape
        else:
            # PIL Image
            W, H = image.size

        boxes = boxes.clone()  # Do not change the original boxes
        N = boxes.shape[0]

        # 1) Calculate individual ROIs with padding
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = x2 - x1
        h = y2 - y1

        px = self.alpha_w * w + self.delta_x
        py = self.alpha_h * h + self.delta_y

        roi_x1 = (x1 - px).clamp(min=0)
        roi_y1 = (y1 - py).clamp(min=0)
        roi_x2 = (x2 + px).clamp(max=W)
        roi_y2 = (y2 + py).clamp(max=H)

        # 2) Randomly select seed object
        # seed_idx = 0
        seed_idx = torch.randint(0, N, (1,)).item()
        cur_x1 = roi_x1[seed_idx].item()
        cur_y1 = roi_y1[seed_idx].item()
        cur_x2 = roi_x2[seed_idx].item()
        cur_y2 = roi_y2[seed_idx].item()

        def area(x1, y1, x2, y2):
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        cur_area = area(cur_x1, cur_y1, cur_x2, cur_y2)

        # 3) Try to add other objects if A_union / (A_i + A_cur) <= area_ratio_max
        for j in range(N):
            if j == seed_idx:
                continue
            cand_x1 = roi_x1[j].item()
            cand_y1 = roi_y1[j].item()
            cand_x2 = roi_x2[j].item()
            cand_y2 = roi_y2[j].item()

            cand_area = area(cand_x1, cand_y1, cand_x2, cand_y2)
            if cand_area <= 0:
                continue

            # new union
            u_x1 = min(cur_x1, cand_x1)
            u_y1 = min(cur_y1, cand_y1)
            u_x2 = max(cur_x2, cand_x2)
            u_y2 = max(cur_y2, cand_y2)
            u_area = area(u_x1, u_y1, u_x2, u_y2)

            #  coefficient "how much the ROI has been inflated"
            denom = (cur_area + cand_area)
            if denom <= 0:
                continue
            ratio = u_area / denom

            if ratio <= self.area_ratio_max:
                # beneficial to merge
                cur_x1, cur_y1, cur_x2, cur_y2 = u_x1, u_y1, u_x2, u_y2
                cur_area = u_area

        # Ensure the ROI is not degenerate
        if cur_x2 <= cur_x1 or cur_y2 <= cur_y1:
            return image, target

        # 4) Perform the crop
        roi_x1_int = int(math.floor(cur_x1))
        roi_y1_int = int(math.floor(cur_y1))
        roi_x2_int = int(math.ceil(cur_x2))
        roi_y2_int = int(math.ceil(cur_y2))

        crop_w = roi_x2_int - roi_x1_int
        crop_h = roi_y2_int - roi_y1_int
        if crop_w <= 0 or crop_h <= 0:
            return image, target

        # torchvision v2 functional crop: top, left, height, width
        image_cropped = F.crop(
            image,
            top=roi_y1_int,
            left=roi_x1_int,
            height=crop_h,
            width=crop_w,
        )

        # 5) Update boxes for the new crop
        new_boxes = boxes.clone()
        new_boxes[:, 0::2] -= roi_x1_int  # x1, x2
        new_boxes[:, 1::2] -= roi_y1_int  # y1, y2

        # clip to crop boundaries
        new_boxes[:, 0::2] = new_boxes[:, 0::2].clamp(min=0, max=crop_w)
        new_boxes[:, 1::2] = new_boxes[:, 1::2].clamp(min=0, max=crop_h)

        # filter boxes with very small area
        bw = new_boxes[:, 2] - new_boxes[:, 0]
        bh = new_boxes[:, 3] - new_boxes[:, 1]
        areas = bw * bh
        keep = areas >= self.min_box_area

        if keep.sum() == 0:
            # if everything is lost after the crop, better to keep the original
            return image, target

        new_boxes = new_boxes[keep]

        # Update canvas_size for the cropped image
        from torchvision import tv_tensors
        
        # Recreate BoundingBoxes with updated canvas_size
        if hasattr(boxes, 'format'):
            # If original boxes were tv_tensors.BoundingBoxes
            new_boxes = tv_tensors.BoundingBoxes(
                new_boxes,
                format=boxes.format,
                canvas_size=(crop_h, crop_w)  # Updated canvas size
            )
        
        # Create new target dict
        new_target = dict(target)
        
        # Handle both "boxes" and "bboxes" keys
        if "boxes" in target:
            new_target["boxes"] = new_boxes
        if "bboxes" in target:
            new_target["bboxes"] = new_boxes
        
        if "labels" in target:
            new_target["labels"] = target["labels"][keep]
        if "difficult" in target:
            new_target["difficult"] = target["difficult"][keep]

        return image_cropped, new_target
