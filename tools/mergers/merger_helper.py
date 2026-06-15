import math
from itertools import combinations


# ---------------------------------------------
# Convenient helper functions
# ---------------------------------------------
@staticmethod
def area(r):
    x1, y1, x2, y2 = r
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

@staticmethod
def bbox_union(a, b):
    x11, y11, x12, y12 = a
    x21, y21, x22, y22 = b
    return (
        min(x11, x21),
        min(y11, y21),
        max(x12, x22),
        max(y12, y22),
    )

@staticmethod
def IoU(a, b):
    """Compute IoU between two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    
    inter_area = area((inter_x1, inter_y1, inter_x2, inter_y2))
    
    a_area = area((ax1, ay1, ax2, ay2))
    b_area = area((bx1, by1, bx2, by2))
    
    union = a_area + b_area - inter_area
    if union <= 0:
        return 0.0
    
    return inter_area / union

def specific_size_roi_merge(rois, target_size: float = 224.0):
    return [(0, 0, target_size, target_size)]
