import torch
import math
from torchvision.transforms.v2 import functional as F

class RoiCropResize(torch.nn.Module):
    """
    ROI-based crop with fixed output resolution for training.

    - Crops image to a fixed size (height, width) ensuring at least 1 target is in the crop.
    - If a target with padding is larger than the crop size, resizes the image first.
    - Adds random variation to padding around the selected target.
    - Works with target["bboxes"] (x1, y1, x2, y2).
    """

    def __init__(
        self,
        delta_x: float = 8.0,  # minimum horizontal padding around target
        delta_y: float = 8.0,  # minimum vertical padding around target
        delta_random: float = 0.5,  # random variation factor for deltas (0.0 to 1.0)
        min_box_area: float = 4.0,  # min area (pixels^2) to keep box after crop
        fill =(0.5, 0.5, 0.5),
        p: float = 1.0,  # probability of applying the transform
    ):
        super().__init__()
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_random = delta_random
        self.min_box_area = min_box_area
        self.fill = fill
        self.p = p

    def forward(self, 
                image, 
                target,
                size: tuple,  # (height, width) in pixels
                ):
        
        self.crop_h, self.crop_w = size
        # target expected: dict with "bboxes" (N,4) and optionally "labels"
        boxes = target.get("bboxes", None)
        if boxes is None or boxes.numel() == 0:
            print("RoiCropResize: No boxes found in target, skipping crop.")
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

        # 1) Randomly select a target object
        seed_idx = torch.randint(0, N, (1,)).item()
        
        # 2) Calculate padded ROI around the selected target
        x1, y1, x2, y2 = boxes[seed_idx, 0].item(), boxes[seed_idx, 1].item(), \
                         boxes[seed_idx, 2].item(), boxes[seed_idx, 3].item()
        
        box_w = x2 - x1
        box_h = y2 - y1
        
        # Add random variation to padding
        random_factor_x = 1.0 + torch.rand(1).item() * self.delta_random
        random_factor_y = 1.0 + torch.rand(1).item() * self.delta_random
        
        pad_x = self.delta_x * random_factor_x
        pad_y = self.delta_y * random_factor_y
        
        # Calculate required size with padding
        required_w = box_w + 2 * pad_x
        required_h = box_h + 2 * pad_y
        
        # 3) Check if we need to resize the image
        scale_factor = 1.0
        if required_w > self.crop_w or required_h > self.crop_h:
            # Need to resize image so target fits in crop
            scale_w = self.crop_w / required_w if required_w > self.crop_w else 1.0
            scale_h = self.crop_h / required_h if required_h > self.crop_h else 1.0
            scale_factor = min(scale_w, scale_h) * 0.95  # 0.95 for safety margin
            
            # Resize image
            new_h = int(H * scale_factor)
            new_w = int(W * scale_factor)
            
            if isinstance(image, torch.Tensor):
                image = F.resize(image, size=[new_h, new_w], antialias=True)
            else:
                image = F.resize(image, size=[new_h, new_w])
            
            # Scale boxes
            boxes = boxes * scale_factor
            
            # Update dimensions
            H, W = new_h, new_w
            
            # Recalculate target position
            x1, y1, x2, y2 = boxes[seed_idx, 0].item(), boxes[seed_idx, 1].item(), \
                             boxes[seed_idx, 2].item(), boxes[seed_idx, 3].item()
        
        # 4) Calculate crop coordinates
        # Center of the target box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Add random offset to center
        max_offset_x = max(0, (self.crop_w - (x2 - x1)) / 2 - pad_x)
        max_offset_y = max(0, (self.crop_h - (y2 - y1)) / 2 - pad_y)
        
        offset_x = (torch.rand(1).item() - 0.5) * 2 * max_offset_x
        offset_y = (torch.rand(1).item() - 0.5) * 2 * max_offset_y
        
        # Crop top-left corner
        crop_x1 = center_x - self.crop_w / 2 + offset_x
        crop_y1 = center_y - self.crop_h / 2 + offset_y
        
        # Ensure crop is within image bounds
        crop_x1 = max(0, min(crop_x1, W - self.crop_w))
        crop_y1 = max(0, min(crop_y1, H - self.crop_h))
        
        # Handle case where image is smaller than crop size
        if W < self.crop_w or H < self.crop_h:
            # Pad image to crop size
            pad_left = max(0, (self.crop_w - W) // 2)
            pad_top = max(0, (self.crop_h - H) // 2)
            pad_right = max(0, self.crop_w - W - pad_left)
            pad_bottom = max(0, self.crop_h - H - pad_top)
            
            if isinstance(image, torch.Tensor):
                image = F.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
            else:
                image = F.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
            
            # Adjust boxes for padding
            boxes[:, 0::2] += pad_left  # x coordinates
            boxes[:, 1::2] += pad_top   # y coordinates
            
            crop_x1 = 0
            crop_y1 = 0
        
        # Convert to integers
        crop_x1_int = int(math.floor(crop_x1))
        crop_y1_int = int(math.floor(crop_y1))
        
        # Ensure we don't exceed image boundaries
        if isinstance(image, torch.Tensor):
            _, H_current, W_current = image.shape
        else:
            W_current, H_current = image.size
            
        crop_w = min(self.crop_w, W_current - crop_x1_int)
        crop_h = min(self.crop_h, H_current - crop_y1_int)
        
        # 5) Perform the crop
        image_cropped = F.crop(
            image,
            top=crop_y1_int,
            left=crop_x1_int,
            height=crop_h,
            width=crop_w,
        )
        
        # Pad if crop is smaller than desired size
        if crop_w < self.crop_w or crop_h < self.crop_h:
            pad_right = self.crop_w - crop_w
            pad_bottom = self.crop_h - crop_h
            if isinstance(image_cropped, torch.Tensor):
                image_cropped = F.pad(image_cropped, [0, 0, pad_right, pad_bottom], fill=self.fill)
            else:
                image_cropped = F.pad(image_cropped, [0, 0, pad_right, pad_bottom], fill=self.fill)
        
        # 6) Update boxes for the new crop
        new_boxes = boxes.clone()
        new_boxes[:, 0::2] -= crop_x1_int  # x1, x2
        new_boxes[:, 1::2] -= crop_y1_int  # y1, y2

        # Clip to crop boundaries
        new_boxes[:, 0::2] = new_boxes[:, 0::2].clamp(min=0, max=self.crop_w)
        new_boxes[:, 1::2] = new_boxes[:, 1::2].clamp(min=0, max=self.crop_h)

        # Filter boxes with very small area
        bw = new_boxes[:, 2] - new_boxes[:, 0]
        bh = new_boxes[:, 3] - new_boxes[:, 1]
        areas = bw * bh
        keep = areas >= self.min_box_area

        if keep.sum() == 0:
            # If everything is lost after the crop, better to keep the original
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
                canvas_size=(self.crop_h, self.crop_w)  # Fixed output size
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
