import math
from itertools import combinations


# ---------------------------------------------
# 1) FLOPs params via β(L)
# ---------------------------------------------
beta = {
    1: 0.35,
    2: 0.55,
    3: 0.75,
    4: 0.88,
    5: 0.95,
    6: 1.00
}

# Thresholds for determining L_R from s_R = sqrt(w*h)
@staticmethod
def __compute_L(s_R):
    if s_R <= 64:
        return 1
    elif s_R <= 120:
        return 2
    elif s_R <= 180:
        return 3
    elif s_R <= 230:
        return 4
    elif s_R <= 270:
        return 5
    else:
        return 6


# ---------------------------------------------
# Convenient helper functions
# ---------------------------------------------
@staticmethod
def area(r):
    x1, y1, x2, y2 = r
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

@staticmethod
def __bbox_union(a, b):
    x11, y11, x12, y12 = a
    x21, y21, x22, y22 = b
    return (
        min(x11, x21),
        min(y11, y21),
        max(x12, x22),
        max(y12, y22),
    )

@staticmethod
def __compute_cost_params(r):
    """Повертає (A_R, L_R, beta(L_R))"""
    x1, y1, x2, y2 = r
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    A = w * h
    s_R = math.sqrt(max(1.0, A))  # avoid sqrt(0)
    L_R = __compute_L(s_R)
    return A, L_R, beta[L_R]

@staticmethod
def __IoU(a, b):
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
# ---------------------------------------------
# 2) Main algorithms
# ---------------------------------------------

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
                iou_val = __IoU(cluster_box, rj)

                if dist <= dist_thresh or iou_val >= iou_thresh:
                    # Merge boxes into cluster
                    cluster_box = __bbox_union(cluster_box, rj)
                    used[j] = True
                    merged_any = True
                    
                    # Update cluster center
                    cx_i, cy_i = center(cluster_box)

        merged.append(cluster_box)

    return merged

def simple_roi_merge_v2(rois, area_ratio_max: float = 1.4):
    boxes = list(rois)
    N = len(boxes)
    used = [False] * len(boxes)
    merged = []

    # 3) Try to add other objects if A_union / (A_i + A_cur) <= area_ratio_max
    for i in range(N):
        if used[i]:
            continue

        # Start new cluster with ROI_i
        cluster_box = boxes[i]
        used[i] = True
        cur_area = area(boxes[i])

        merged_any = True
        while merged_any:
            merged_any = False
            for j in range(len(boxes)):
                if used[j]:
                    continue

                cand_area = area(boxes[j])
                if cand_area <= 0:
                    continue

                # new union
                u_box = __bbox_union(cluster_box, boxes[j])
                u_area = area(u_box)

                #  coefficient "how much the ROI has been inflated"
                denom = (cur_area + cand_area)
                if denom <= 0:
                    continue
                ratio = u_area / denom

                if ratio <= area_ratio_max:
                    # beneficial to merge
                    cluster_box = __bbox_union(cluster_box, boxes[j])
                    used[j] = True
                    merged_any = True
            merged.append(cluster_box)
    return merged

def greedy_roi_merge(rois, area_ratio_max: float = 100, tau = 5000.0):  # A_union / (A_i + A_j) <= 1.4 => merge ok)
    """
    rois: list of ROIs [(x1,y1,x2,y2), ...]
    tau: threshold (K/c_full), additional launch costs
    Returns a list of optimal merged ROIs.
    """

    # Clusters represented as a list
    clusters = list(rois)
    
    # Precompute cost parameters
    params = {id(r): __compute_cost_params(r) for r in clusters}

    # Gain matrix Δ_ij cached as a dict with keys (i,j)
    def compute_delta(i_r, j_r):
        # i_r, j_r — two ROIs
        A_i, _, b_i = params[id(i_r)]
        A_j, _, b_j = params[id(j_r)]

        # Union ROI
        r_u = __bbox_union(i_r, j_r)
        A_u, _, b_u = __compute_cost_params(r_u)

        # Δ = cost_i + cost_j - cost_u
        # cost_i = b_i * A_i, but +K already accounted for in τ
        delta = (b_i * A_i + b_j * A_j) - (b_u * A_u)
        return delta, r_u

    # Prepare all pairs
    # Represent ROIs by their indices in clusters
    while True:
        best_delta = -1e18
        best_pair = None
        best_union = None

        if len(clusters) <= 1:
            break

        # Iterate over all pairs
        for i, j in combinations(range(len(clusters)), 2):
            ri = clusters[i]
            rj = clusters[j]

            delta, r_u = compute_delta(ri, rj)
            u_area = area(r_u)
            cand_area = area(ri) + area(rj)
            #  coefficient "how much the ROI has been inflated"
            if cand_area <= 0:
                continue
            ratio = u_area / cand_area

            # Check if this is good: Δ > τ
            if delta > tau and delta > best_delta and ratio <= area_ratio_max:
                # beneficial to merge
                best_delta = delta
                best_pair = (i, j)
                best_union = r_u

        # If no pair is good — stop
        if best_pair is None:
            break

        i, j = best_pair

        # Merge ROI[i] + ROI[j] → union
        r_u = best_union

        # Replace two clusters with one
        new_clusters = []
        for k, r in enumerate(clusters):
            if k not in best_pair:
                new_clusters.append(r)
        new_clusters.append(r_u)

        clusters = new_clusters

        # Update parameters for the new ROI
        params = {id(r): __compute_cost_params(r) for r in clusters}

    return clusters