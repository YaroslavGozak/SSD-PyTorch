from itertools import combinations
import math

from tools.mergers.merger_helper import bbox_union, area

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

def greedy_roi_merge(rois, tau, area_ratio_max: float = 100):  # A_union / (A_i + A_j) <= 1.4 => merge ok)
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
        r_u = bbox_union(i_r, j_r)
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