import math
from tools.mergers.merger_helper import IoU, bbox_union


def simple_roi_merge(rois, iou_thresh=0.0, dist_thresh=40.0):
    """
    Simple sequential ROI merging.

    rois        : list of (x1,y1,x2,y2)
    iou_thresh  : minimum IoU to merge
    dist_thresh : maximum center distance to merge

    Returns a list of merged ROI (clusters), sequential greedy.
    """
    used = [False] * len(rois)
    merged = []

    def center(r):
        x1, y1, x2, y2 = r
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    for i in range(len(rois)):
        if used[i]:
            continue

        # Start new cluster with ROI_i
        cluster_box = rois[i]
        cx_i, cy_i = center(cluster_box)
        used[i] = True

        # Try to merge other ROI into this cluster
        merged_any = True
        while merged_any:
            merged_any = False
            for j in range(len(rois)):
                if used[j]:
                    continue

                rj = rois[j]
                cx_j, cy_j = center(rj)

                # Check spatial closeness via center distance
                dist = math.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)

                # Check overlap via IoU
                iou_val = IoU(cluster_box, rj)

                if dist <= dist_thresh or iou_val >= iou_thresh:
                    # Merge boxes into cluster
                    cluster_box = bbox_union(cluster_box, rj)
                    used[j] = True
                    merged_any = True
                    
                    # Update cluster center
                    cx_i, cy_i = center(cluster_box)

        merged.append(cluster_box)

    return merged