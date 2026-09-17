"""Generate frozen validation geometry independently of timing observations."""
import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from .common import file_sha256, load_config, write_json
from .geometry import Rectangle, union_rectangle, stride_rounded_shape
from .models import load_latency_models, decide_merge, PolynomialLatencyModel
from .reproducibility import canonical_hash

PRIORITY = ["model_disagreement", "piecewise_boundary", "quadratic_boundary", "linear_boundary", "broad_random"]


def stability(artifact, areas, config):
    result = {}
    settings = config.get("decision_stability", {})
    for name, model in load_latency_models(artifact).items():
        gains = []
        for replicate in artifact.get("bootstrap", {}).get("replicates", []):
            fit = replicate[name]
            if not fit["valid"]:
                continue
            sampled = PolynomialLatencyModel(name, fit["coefficients"], model.envelope, fit.get("breakpoint_area"))
            try:
                gains.append(1000*(sampled.predict_seconds(areas[0])+sampled.predict_seconds(areas[1])-sampled.predict_seconds(areas[2])))
            except ValueError:
                continue
        probability = float(np.mean(np.array(gains)>0)) if gains else None
        result[name] = dict(merge_probability=probability,
                            predicted_gain_ms_ci=np.quantile(gains, [.025,.975]).tolist() if gains else None,
                            decision_stable=probability is not None and (probability <= settings.get("stable_separate_max_probability", .1) or probability >= settings.get("stable_merge_min_probability", .9)),
                            valid_model_count=len(gains))
    return result


def classify(shapes, models, options):
    gains = {name: decide_merge(model, *shapes)["predicted_gain_s"]*1000 for name,model in models.items()}
    tau = models["linear"].coefficients["b0"]/models["linear"].coefficients["b1"]
    areas = [h*w for h,w in shapes]
    distance = areas[2]-areas[0]-areas[1]-tau
    tags = ["broad_random"]
    if abs(distance) <= float(options.get("linear_boundary_width_pixels", abs(tau)*.2)):
        tags.append("linear_boundary")
    for name in ("piecewise", "quadratic"):
        if name in gains and abs(gains[name]) <= float(options.get("boundary_width_ms", .5)):
            tags.append(name+"_boundary")
    if len({v>0 for v in gains.values()}) > 1:
        tags.append("model_disagreement")
    return next(t for t in PRIORITY if t in tags), tags, gains, distance


def generate(config, calibration, output, regenerate=False):
    target = Path(output)
    if target.exists() and not regenerate:
        raise ValueError("Pair specs already exist; use --regenerate-pairs explicitly")
    artifact = json.loads(Path(calibration).read_text(encoding="utf-8"))
    models = load_latency_models(artifact)
    options = config.get("experiment_b", {})
    seed = int(options.get("generator_seed", config.get("seed", 0)))
    rng = random.Random(seed)
    stride = int(config.get("model", {}).get("stride", 32))
    canvas = list(map(int, options.get("canvas_hw", [640,640])))
    count = int(options.get("pair_count", 500))
    if count <= 0:
        raise ValueError("pair_count must be positive")
    fractions = options.get("strata_quotas", {name:.2 for name in PRIORITY})
    if set(fractions)-set(PRIORITY) or any(v<0 for v in fractions.values()) or abs(sum(fractions.values())-1)>1e-8:
        raise ValueError("strata_quotas must be nonnegative fractions summing to one")
    quotas = {name:int(count*fraction) for name,fraction in fractions.items()}
    quotas[list(quotas)[-1]] += count-sum(quotas.values())
    selected = {name:[] for name in quotas}
    keys, pairs = set(), []
    digest = file_sha256(calibration)
    for attempt in range(int(options.get("max_generation_attempts", count*10000))):
        rectangles = []
        for _ in range(2):
            h,w = rng.randint(1,canvas[0]),rng.randint(1,canvas[1])
            x,y = rng.randint(0,canvas[1]-w),rng.randint(0,canvas[0]-h)
            rectangles.append(Rectangle(x,y,x+w,y+h))
        first, second = rectangles
        union = union_rectangle(first,second)
        shapes = [stride_rounded_shape(r.height,r.width,stride) for r in (first,second,union)]
        if not all(models["linear"].is_in_domain(*hw) for hw in shapes):
            continue
        key = tuple(v for hw in shapes for v in hw)
        if key in keys:
            continue
        primary,tags,gains,distance = classify(shapes,models,options)
        if primary not in quotas or len(selected[primary]) >= quotas[primary]:
            continue
        # Reserve half of each boundary quota for each sign.
        if primary.endswith("_boundary"):
            name = primary.removesuffix("_boundary")
            side = gains[name]>0
            limit = (quotas[primary]+1)//2 if side else quotas[primary]//2
            if sum((p["predicted_gains_ms"][name]>0) == side for p in selected[primary]) >= limit:
                continue
        contained = ((first.x1<=second.x1 and first.y1<=second.y1 and first.x2>=second.x2 and first.y2>=second.y2) or
                     (second.x1<=first.x1 and second.y1<=first.y1 and second.x2>=first.x2 and second.y2>=first.y2))
        horizontal = first.x2<=second.x1 or second.x2<=first.x1
        vertical = first.y2<=second.y1 or second.y2<=first.y1
        geometry = "containment" if contained else "diagonal_separation" if horizontal and vertical else "horizontal_separation" if horizontal else "vertical_separation" if vertical else "partial_overlap"
        areas = [h*w for h,w in shapes]
        pair = dict(pair_id=len(pairs), coordinates=[[r.x1,r.y1,r.x2,r.y2] for r in (first,second,union)],
                    geometry_type=geometry, requested_shapes=[[r.height,r.width] for r in (first,second,union)],
                    tensor_shapes=shapes, effective_areas=areas, delta_area=areas[2]-areas[0]-areas[1],
                    domain_result=True, computational_key=key, primary_stratum=primary, stratum_tags=tags,
                    predicted_gains_ms=gains, linear_signed_distance=distance,
                    threshold_side="merge" if distance<0 else "separate", calibration_hash=digest,
                    generator_seed=seed, schema_version=1)
        selected[primary].append(pair)
        pairs.append(pair)
        keys.add(key)
        if len(pairs) == count:
            break
    counts = {name:len(items) for name,items in selected.items()}
    if counts != quotas:
        raise ValueError(f"Unattainable strata quotas/boundary sides: obtained={counts}, required={quotas}. Adjust explicitly; no substitution performed.")
    for name,items in selected.items():
        if name.endswith("_boundary") and items and len({p["predicted_gains_ms"][name.removesuffix("_boundary")]>0 for p in items}) != 2:
            raise ValueError(f"Both sides of {name} are required")
    for pair in pairs:
        pair["bootstrap_stability"] = stability(artifact,pair["effective_areas"],config)
    payload = dict(schema_version=1, canvas_hw=canvas, stride=stride, calibration_hash=digest,
                   generator_seed=seed, priority=PRIORITY, quotas=quotas, config=options, decision_stability=config.get("decision_stability", {"stable_separate_max_probability":.1,"stable_merge_min_probability":.9}), pairs=pairs)
    write_json(target, dict(payload=payload, sha256=canonical_hash(payload)))
    return payload


def load(path, config, calibration, adapter, image):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = document["payload"]
    if document["sha256"] != canonical_hash(payload):
        raise ValueError("Pair specs hash mismatch")
    if payload["schema_version"] != 1 or payload["stride"] != adapter.stride or payload["canvas_hw"] != config.get("experiment_b", {}).get("canvas_hw", [640,640]):
        raise ValueError("Pair specs schema/stride/canvas mismatch")
    if payload["calibration_hash"] != file_sha256(calibration):
        raise ValueError("Pair specs calibration hash mismatch")
    models = load_latency_models(json.loads(Path(calibration).read_text(encoding="utf-8")))
    pairs, keys, ids = [],set(),set()
    for specification in payload["pairs"]:
        rectangles = [Rectangle(*coords) for coords in specification["coordinates"]]
        if rectangles[2] != union_rectangle(*rectangles[:2]):
            raise ValueError("Invalid union rectangle")
        if any(not (0<=r.x1<r.x2<=payload["canvas_hw"][1] and 0<=r.y1<r.y2<=payload["canvas_hw"][0]) for r in rectangles):
            raise ValueError("Geometry outside canvas")
        prepared = [adapter.prepare(image,(r.height,r.width)) for r in rectangles]
        shapes = [list(p.tensor_hw) for p in prepared]
        key = tuple(v for hw in shapes for v in hw)
        if key in keys or specification["pair_id"] in ids:
            raise ValueError("Duplicate computational key or pair_id")
        if shapes != specification["tensor_shapes"] or not all(models["linear"].is_in_domain(*hw) for hw in shapes):
            raise ValueError("Actual tensor shape/domain mismatch")
        areas = [h*w for h,w in shapes]
        primary,tags,gains,distance = classify(shapes,models,payload["config"])
        if (list(key) != specification["computational_key"] or areas != specification["effective_areas"] or
            areas[2]-areas[0]-areas[1] != specification["delta_area"] or not specification["domain_result"] or
            primary != specification["primary_stratum"] or tags != specification["stratum_tags"] or
            [[r.height,r.width] for r in rectangles] != specification["requested_shapes"] or
            specification["calibration_hash"] != payload["calibration_hash"]):
            raise ValueError("Pair specification derived fields mismatch")
        keys.add(key); ids.add(specification["pair_id"])
        pairs.append(dict(first=rectangles[0],second=rectangles[1],union=rectangles[2], prepared=[replace(p,tensor=None) for p in prepared],
                          delta=specification["delta_area"], geometry_type=specification["geometry_type"],
                          boundary_bin=specification["primary_stratum"], specification=specification))
    counts = {name:sum(p["boundary_bin"] == name for p in pairs) for name in payload["quotas"]}
    if counts != payload["quotas"] or sum(counts.values()) != len(pairs):
        raise ValueError("Frozen strata quota mismatch")
    for name,count in counts.items():
        if count and name.endswith("_boundary"):
            model_name = name.removesuffix("_boundary")
            sides = {p["specification"]["predicted_gains_ms"][model_name]>0 for p in pairs if p["boundary_bin"] == name}
            if sides != {True,False}:
                raise ValueError(f"Both boundary sides required for {name}")
    return pairs,document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regenerate-pairs", action="store_true")
    args = parser.parse_args()
    generate(load_config(args.config),args.calibration,args.output,args.regenerate_pairs)


if __name__ == "__main__":
    main()
