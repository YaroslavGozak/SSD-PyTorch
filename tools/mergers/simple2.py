from tools.mergers.merger_helper import bbox_union, area


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
                u_box = bbox_union(cluster_box, boxes[j])
                u_area = area(u_box)

                #  coefficient "how much the ROI has been inflated"
                denom = (cur_area + cand_area)
                if denom <= 0:
                    continue
                ratio = u_area / denom

                if ratio <= area_ratio_max:
                    # beneficial to merge
                    cluster_box = bbox_union(cluster_box, boxes[j])
                    used[j] = True
                    merged_any = True
            merged.append(cluster_box)
    return merged