import torch
import math
from torchvision.transforms.v2 import functional as F


class FixedROICrop(torch.nn.Module):
    """
    Crop around the first target bounding box with fixed padding.
    
    Args:
        pad_x: Fixed horizontal padding (in pixels)
        pad_y: Fixed vertical padding (in pixels)
        min_box_area: Minimum area (pixels^2) to keep a box after crop
    """

    def __init__(
        self,
        pad_x: float = 50.0,
        pad_y: float = 50.0,
        min_box_area: float = 4.0,
    ):
        super().__init__()
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.min_box_area = min_box_area

    def forward(self, image, target):
        # target expected: dict with "boxes" or "bboxes" (N,4) and optionally "labels"
        boxes = target.get("boxes", target.get("bboxes", None))
        if boxes is None or boxes.numel() == 0:
            print("FixedROICrop: No boxes found in target, skipping crop.")
            return image, target

        if isinstance(image, torch.Tensor):
            # C x H x W
            _, H, W = image.shape
        else:
            # PIL Image
            W, H = image.size

        boxes = boxes.clone()
        
        # Take the first bounding box
        first_box = boxes[0]
        x1, y1, x2, y2 = first_box[0].item(), first_box[1].item(), first_box[2].item(), first_box[3].item()
        
        # Add fixed padding
        roi_x1 = max(0, x1 - self.pad_x)
        roi_y1 = max(0, y1 - self.pad_y)
        roi_x2 = min(W, x2 + self.pad_x)
        roi_y2 = min(H, y2 + self.pad_y)
        
        # Ensure the ROI is not degenerate
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return image, target
        
        # Convert to integers for cropping
        roi_x1_int = int(math.floor(roi_x1))
        roi_y1_int = int(math.floor(roi_y1))
        roi_x2_int = int(math.ceil(roi_x2))
        roi_y2_int = int(math.ceil(roi_y2))
        
        crop_w = roi_x2_int - roi_x1_int
        crop_h = roi_y2_int - roi_y1_int
        if crop_w <= 0 or crop_h <= 0:
            return image, target
        
        # Perform the crop (torchvision v2 functional crop: top, left, height, width)
        image_cropped = F.crop(
            image,
            top=roi_y1_int,
            left=roi_x1_int,
            height=crop_h,
            width=crop_w,
        )
        
        # Update boxes for the new crop
        new_boxes = boxes.clone()
        new_boxes[:, 0::2] -= roi_x1_int  # x1, x2
        new_boxes[:, 1::2] -= roi_y1_int  # y1, y2
        
        # Clip to crop boundaries
        new_boxes[:, 0::2] = new_boxes[:, 0::2].clamp(min=0, max=crop_w)
        new_boxes[:, 1::2] = new_boxes[:, 1::2].clamp(min=0, max=crop_h)
        
        # Filter boxes with very small area
        bw = new_boxes[:, 2] - new_boxes[:, 0]
        bh = new_boxes[:, 3] - new_boxes[:, 1]
        areas = bw * bh
        keep = areas >= self.min_box_area
        
        if keep.sum() == 0:
            # If everything is lost after the crop, keep the original
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
