"""Pure geometry, pair generation, and policy accounting for the experiment."""

import random
from dataclasses import asdict

import numpy as np

from experiments.cost_model_validation.geometry import Rectangle, union_rectangle
from tools.mergers.simple import simple_roi_merge
from tools.mergers.simple2 import simple_roi_merge_v2


DEFAULT_SHAPES = [(32, 32), (32, 64), (64, 32), (32, 96), (96, 32),
                  (32, 128), (128, 32), (32, 256), (256, 32),
                  (64, 64), (64, 128), (128, 64), (96, 96), (96, 192),
                  (192, 96), (128, 128), (128, 256), (256, 128),
                  (160, 160), (192, 192), (256, 256), (320, 320)]
DEFAULT_SIDES = [32, 64, 96, 128, 160, 192, 256, 320]
GEOMETRIES = ("horizontal", "vertical", "diagonal", "overlap", "touching", "small_gap", "medium_gap", "large_gap")


def rect_values(prefix, rect):
    result = {f"{prefix}_{key}": value for key, value in asdict(rect).items()}
    result.update({f"{prefix}_width": rect.width, f"{prefix}_height": rect.height,
                   f"{prefix}_area": rect.area, f"{prefix}_aspect": rect.width / rect.height})
    return result


def pair_geometry(first, second):
    merged = union_rectangle(first, second)
    overlap_w = max(0, min(first.x2, second.x2) - max(first.x1, second.x1))
    overlap_h = max(0, min(first.y2, second.y2) - max(first.y1, second.y1))
    overlap = overlap_w * overlap_h
    return {**rect_values("r1", first), **rect_values("r2", second),
            "geometric_area_ratio": merged.area / (first.area + second.area),
            **rect_values("merged", merged), "area_extra": merged.area - first.area - second.area,
            "gap_x": max(0, second.x1-first.x2, first.x1-second.x2),
            "gap_y": max(0, second.y1-first.y2, first.y1-second.y2),
            "displacement_x": second.x1-first.x1, "displacement_y": second.y1-first.y1,
            "iou": overlap / (first.area + second.area - overlap)}


def make_pair(canvas_hw, shape1, shape2, geometry, rng):
    """Place a pair on one canvas; return None when the requested layout cannot fit."""
    height, width = map(int, canvas_hw)
    w1, h1 = map(int, shape1)
    w2, h2 = map(int, shape2)
    if min(w1, h1, w2, h2) <= 0 or max(w1, w2) > width or max(h1, h2) > height:
        return None
    if geometry in {"horizontal", "vertical", "diagonal", "small_gap", "medium_gap", "large_gap"}:
        gap_scale = {"small_gap": .1, "medium_gap": .5, "large_gap": 1.0}.get(geometry, .25)
        if geometry == "vertical":
            dx, dy = 0, h1 + round(gap_scale * max(h1, h2))
        elif geometry == "diagonal":
            dx, dy = w1 + round(gap_scale*w1), h1 + round(gap_scale*h1)
        else:
            dx, dy = w1 + round(gap_scale*max(w1,w2)), 0
    elif geometry == "touching":
        dx, dy = w1, 0
    elif geometry == "overlap":
        fraction = rng.choice((.1, .25, .5, .75))
        dx, dy = round(w1*fraction), round(h1*fraction)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    # Translate both rectangles together; preserve the requested displacement.
    min_x, max_x = min(0, dx), max(w1, dx+w2)
    min_y, max_y = min(0, dy), max(h1, dy+h2)
    if max_x-min_x > width or max_y-min_y > height:
        return None
    base_x = rng.randrange(width-(max_x-min_x)+1)-min_x
    base_y = rng.randrange(height-(max_y-min_y)+1)-min_y
    first = Rectangle(base_x, base_y, base_x+w1, base_y+h1)
    second = Rectangle(base_x+dx, base_y+dy, base_x+dx+w2, base_y+dy+h2)
    return first, second


def generate_pairs(canvas_hw, count, seed, shapes=DEFAULT_SHAPES, sides=DEFAULT_SIDES, tau=None):
    if count < 1:
        raise ValueError("pair_count must be positive")
    rng = random.Random(seed)
    shapes = [tuple(map(int, s)) for s in shapes]
    sides = list(map(int, sides))
    if not shapes or not sides:
        raise ValueError("shape and side grids must be nonempty")
    candidates, skipped = [], {"cannot_fit": 0}
    attempts = max(500, count*100)
    for i in range(attempts):
        if i < len(shapes):
            shape1, shape2 = shapes[i], rng.choice(shapes)
        else:
            shape1 = rng.choice(shapes) if i % 2 else (rng.choice(sides), rng.choice(sides))
            shape2 = rng.choice(shapes) if i % 3 else (rng.choice(sides), rng.choice(sides))
        geometry = GEOMETRIES[i % len(GEOMETRIES)]
        pair = make_pair(canvas_hw, shape1, shape2, geometry, rng)
        if pair is None:
            skipped["cannot_fit"] += 1
            continue
        extra = pair_geometry(*pair)["area_extra"]
        ratio = extra/tau if tau is not None and tau > 0 else None
        boundary = "unclassified" if ratio is None else (
            "near" if .8 <= ratio <= 1.2 else "below" if ratio < .8 else "above")
        candidates.append((pair, geometry, boundary))
    if not candidates:
        raise ValueError("No valid pairs fit the canvas; increase canvas size or reduce shapes")
    # Guarantee representative first-segment shapes where they can fit, then
    # distribute remaining slots around the fitted boundary where feasible.
    selected = []
    for shape in shapes:
        match = next((c for c in candidates if (c[0][0].width,c[0][0].height) == shape), None)
        if match and match not in selected and len(selected) < count:
            selected.append(match)
    for region in ("below", "near", "above") if tau is not None else ("unclassified",):
        pool = [c for c in candidates if c[2] == region]
        rng.shuffle(pool)
        for item in pool:
            if len(selected) >= count or sum(c[2] == region for c in selected) >= max(1,count//3):
                break
            if item not in selected:
                selected.append(item)
    rng.shuffle(candidates)
    seen = {(c[0][0],c[0][1]) for c in selected}
    for item in candidates:
        if len(selected) >= count:
            break
        key = (item[0][0],item[0][1])
        if key not in seen:
            selected.append(item)
            seen.add(key)
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} unique pairs could be generated for {count} requested")
    rng.shuffle(selected)
    return selected[:count], skipped


def decide(predicted_merged, predicted_first, predicted_second):
    return predicted_merged < predicted_first + predicted_second


def nearest_shape_cost(table, shape):
    """Nearest calibrated tensor shape in log width/height space."""
    h, w = shape
    key = min(table, key=lambda other: (np.log(other[0]/h)**2 + np.log(other[1]/w)**2, other))
    return table[key], key


def policy_decisions(first, second, actual, affine, lookup, gamma=1.4,
                     simple_iou=.01, simple_distance=40):
    merged = union_rectangle(first, second)
    boxes = [tuple(asdict(r).values()) for r in (first, second)]
    return {
        "no_merge": False,
        "existing_simple_merge": len(simple_roi_merge(boxes, iou_thresh=simple_iou, dist_thresh=simple_distance)) == 1,
        "geometric_area": len(simple_roi_merge_v2(boxes, area_ratio_max=gamma)) == 1,
        "cost_affine": decide(*affine),
        "cost_shape_lookup": decide(*lookup),
        "oracle": decide(*actual),
    }


def summarize(rows):
    if not rows:
        raise ValueError("No pair measurements to summarize")
    no_merge_total = sum(row["actual_separate_ms"] for row in rows)
    oracle_total = sum(min(row["actual_separate_ms"], row["actual_merged_ms"]) for row in rows)
    result = []
    for method in ("no_merge", "existing_simple_merge", "geometric_area", "cost_affine", "cost_shape_lookup", "oracle"):
        labels = [bool(row[f"{method}_decision"]) for row in rows]
        truths = [bool(row["oracle_decision"]) for row in rows]
        costs = np.array([row["actual_merged_ms"] if label else row["actual_separate_ms"]
                          for label,row in zip(labels, rows)])
        total = float(costs.sum())
        result.append(dict(method=method, decision_accuracy=sum(a == b for a,b in zip(labels,truths))/len(rows),
                           TP=sum(a and b for a,b in zip(labels,truths)),
                           FP=sum(a and not b for a,b in zip(labels,truths)),
                           FN=sum(not a and b for a,b in zip(labels,truths)),
                           TN=sum(not a and not b for a,b in zip(labels,truths)),
                           mean_effective_latency_ms=float(costs.mean()),
                           median_effective_latency_ms=float(np.median(costs)),
                           total_effective_latency_ms=total,
                           relative_to_no_merge_percent=100*(total/no_merge_total-1),
                           relative_to_oracle_percent=100*(total/oracle_total-1)))
    return result
