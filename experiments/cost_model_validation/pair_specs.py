"""Generate frozen validation geometry independently of timing observations."""
import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .common import file_sha256, load_config, write_json
from .geometry import Rectangle, union_rectangle, stride_rounded_shape
from .models import load_latency_models, decide_merge, PolynomialLatencyModel
from .reproducibility import canonical_hash
from .artifacts import load_calibration
from .shape_model import ShapeLookupLatencyModel, policy_declaration
from .validation_design import validation_declaration

PRIORITY = ["shape_lookup_conservative_boundary", "shape_lookup_boundary",
            "linear_tau_shape_lookup_conservative_disagreement", "model_disagreement",
            "piecewise_boundary", "quadratic_boundary", "linear_boundary", "broad_random"]
DEFAULT_STRATA = ["model_disagreement", "piecewise_boundary", "quadratic_boundary", "linear_boundary", "broad_random"]


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


def _lookup_gains(shapes, lookup, declaration):
    if lookup is None or not all(lookup.is_in_domain(*hw) for hw in shapes):
        return {}
    estimates = [lookup.estimate(*hw) for hw in shapes]
    raw = estimates[0].point_s + estimates[1].point_s - estimates[2].point_s
    ses = [estimate.standard_error_s for estimate in estimates]
    if not all(value is not None for value in ses):
        return {"shape_lookup": raw * 1000}
    conservative = raw - NormalDist().inv_cdf(declaration["confidence_level"]) * sum(ses)
    return {"shape_lookup": raw * 1000, "shape_lookup_conservative": conservative * 1000}


def _boundary_score(name, gains, declaration):
    margin_ms = 1000 * float(declaration.get("decision_margin_s", 0))
    return gains[name] - margin_ms if name.startswith("shape_lookup") else gains[name]


def classify(shapes, models, options, lookup=None, declaration=None):
    declaration = declaration or policy_declaration({})
    gains = {name: decide_merge(model, *shapes)["predicted_gain_s"]*1000 for name,model in models.items()}
    gains.update(_lookup_gains(shapes, lookup, declaration))
    tau = models["linear"].coefficients["b0"]/models["linear"].coefficients["b1"]
    areas = [h*w for h,w in shapes]
    distance = areas[2]-areas[0]-areas[1]-tau
    tags = ["broad_random"]
    if abs(distance) <= float(options.get("linear_boundary_width_pixels", abs(tau)*.2)):
        tags.append("linear_boundary")
    for name in ("piecewise", "quadratic"):
        if name in gains and abs(gains[name]) <= float(options.get("boundary_width_ms", .5)):
            tags.append(name+"_boundary")
    enabled = set(options.get("strata_quotas", {}))
    for name in ("shape_lookup", "shape_lookup_conservative"):
        tag = name + "_boundary"
        if tag in enabled and name in gains and abs(_boundary_score(name, gains, declaration)) <= float(options.get("boundary_width_ms", .5)):
            tags.append(tag)
    if len({gains[name]>0 for name in models}) > 1:
        tags.append("model_disagreement")
    disagreement = "linear_tau_shape_lookup_conservative_disagreement"
    if (disagreement in enabled and "shape_lookup_conservative" in gains
            and (gains["linear"] > 0) != (_boundary_score("shape_lookup_conservative", gains, declaration) > 0)):
        tags.append(disagreement)
    scores = {name:_boundary_score(name,gains,declaration) for name in gains}
    return next(t for t in PRIORITY if t in tags), tags, gains, distance, scores


def grid_candidates(models, lookup, declaration, canvas, stride, options, quotas, selected, seed):
    """Search realizable tensor geometry directly instead of hoping random gaps hit boundaries.

    For any three shapes with union dimensions >= both ROI dimensions, putting
    ROI 1 at the origin and ROI 2 at the union's bottom-right realizes that union.
    On a stride-aligned canvas this covers every possible computational key.
    """
    shapes = [(h,w) for h in range(stride,canvas[0]+1,stride)
              for w in range(stride,canvas[1]+1,stride) if models["linear"].is_in_domain(h,w)]
    random.Random(seed).shuffle(shapes)
    if not shapes:
        return
    array = np.asarray(shapes)
    areas = array[:,0]*array[:,1]
    costs = {name:np.array([model.predict_seconds(int(a)) for a in areas]) for name,model in models.items()}
    lookup_points = lookup_ses = None
    if lookup is not None:
        estimates = [lookup.estimate(*shape) if lookup.is_in_domain(*shape) else None for shape in shapes]
        lookup_points = np.array([estimate.point_s if estimate else np.nan for estimate in estimates])
        lookup_ses = np.array([estimate.standard_error_s if estimate and estimate.standard_error_s is not None else np.nan for estimate in estimates])
    margin_ms = 1000 * float(declaration.get("decision_margin_s", 0))
    z = NormalDist().inv_cdf(declaration["confidence_level"])
    tau = models["linear"].coefficients["b0"]/models["linear"].coefficients["b1"]
    for i,(h1,w1) in enumerate(shapes):
        for j,(h2,w2) in enumerate(shapes):
            if all(len(selected[name]) == quota for name,quota in quotas.items()):
                return
            possible = (array[:,0]>=max(h1,h2)) & (array[:,1]>=max(w1,w2))
            gains = {name:1000*(v[i]+v[j]-v) for name,v in costs.items()}
            if lookup_points is not None:
                gains["shape_lookup"] = 1000*(lookup_points[i]+lookup_points[j]-lookup_points)
                gains["shape_lookup_conservative"] = gains["shape_lookup"]-1000*z*(lookup_ses[i]+lookup_ses[j]+lookup_ses)
            decisions = np.array([gains[name]>0 for name in models])
            tags = {"model_disagreement":np.any(decisions != decisions[0],axis=0),
                    "linear_boundary":np.abs(areas-areas[i]-areas[j]-tau)<=float(options.get("linear_boundary_width_pixels",abs(tau)*.2)),
                    "broad_random":np.ones(len(shapes),dtype=bool)}
            for name in ("piecewise","quadratic"):
                tags[name+"_boundary"] = np.abs(gains[name])<=float(options.get("boundary_width_ms",.5)) if name in gains else np.zeros(len(shapes),dtype=bool)
            for name in ("shape_lookup", "shape_lookup_conservative"):
                score = gains.get(name, np.full(len(shapes),np.nan))-margin_ms
                tag = name+"_boundary"
                tags[tag] = np.abs(score)<=float(options.get("boundary_width_ms",.5)) if tag in quotas else np.zeros(len(shapes),dtype=bool)
            conservative_score = gains.get("shape_lookup_conservative",np.full(len(shapes),np.nan))-margin_ms
            disagreement = "linear_tau_shape_lookup_conservative_disagreement"
            tags[disagreement] = ((gains["linear"]>0) != (conservative_score>0)) if disagreement in quotas else np.zeros(len(shapes),dtype=bool)
            claimed = np.zeros(len(shapes),dtype=bool)
            wanted = np.zeros(len(shapes),dtype=bool)
            for name in PRIORITY:
                primary = tags[name] & ~claimed
                claimed |= tags[name]
                if name not in quotas or len(selected[name]) >= quotas[name]:
                    continue
                if name.endswith("_boundary"):
                    model_name = name.removesuffix("_boundary")
                    score = gains[model_name] - (margin_ms if model_name.startswith("shape_lookup") else 0)
                    for side in (False,True):
                        limit = (quotas[name]+int(side))//2
                        present = sum((p["boundary_scores_ms"][model_name]>0)==side for p in selected[name])
                        if present < limit:
                            wanted |= primary & ((score>0)==side)
                elif name in {"model_disagreement", "linear_tau_shape_lookup_conservative_disagreement"}:
                    wanted |= primary
            # Broad/random coverage is exclusively supplied by random candidates.
            for index in np.flatnonzero(possible & wanted):
                hu,wu = shapes[index]
                yield Rectangle(0,0,w1,h1), Rectangle(wu-w2,hu-h2,wu,hu)


def candidates(models, lookup, declaration, canvas, stride, options, quotas, selected, rng, seed):
    # Preserve broad random coverage, then search the rare boundary keys directly.
    maximum = int(options.get("max_generation_attempts",5000000))
    attempts = min(maximum,
                   int(options.get("random_generation_attempts",10000)))

    def random_pair():
        rectangles = []
        for _ in range(2):
            h,w = rng.randint(1,canvas[0]),rng.randint(1,canvas[1])
            x,y = rng.randint(0,canvas[1]-w),rng.randint(0,canvas[0]-h)
            rectangles.append(Rectangle(x,y,x+w,y+h))
        return rectangles

    for _ in range(attempts):
        yield random_pair()
    if options.get("tensor_grid_search", True):
        yield from grid_candidates(models,lookup,declaration,canvas,stride,options,quotas,selected,seed)
    # Grid candidates never fill broad_random. Preserve the remaining random
    # budget when that quota is still short (or grid search was disabled).
    if (len(selected.get("broad_random",[])) < quotas.get("broad_random",0)
            or not options.get("tensor_grid_search",True)):
        for _ in range(maximum-attempts):
            yield random_pair()


def generate(config, calibration, output, regenerate=False):
    target = Path(output)
    if target.exists() and not regenerate:
        raise ValueError("Pair specs already exist; use --regenerate-pairs explicitly")
    artifact = load_calibration(calibration,bootstrap=True)
    models = load_latency_models(artifact)
    options = config.get("experiment_b", {})
    seed = int(options.get("generator_seed", config.get("seed", 0)))
    rng = random.Random(seed)
    stride = int(config.get("model", {}).get("stride", 32))
    canvas = list(map(int, options.get("canvas_hw", [640,640])))
    count = int(options.get("pair_count", 500))
    if count <= 0:
        raise ValueError("pair_count must be positive")
    evaluation_design = options.get("evaluation_design","challenge")
    if evaluation_design not in {"representative","challenge"}:
        raise ValueError("Unknown evaluation design")
    representative = evaluation_design == "representative"
    fractions = {"reference":1.} if representative else options.get("strata_quotas", {name:.2 for name in DEFAULT_STRATA})
    if set(fractions)-set(PRIORITY+["reference"]) or any(v<0 for v in fractions.values()) or abs(sum(fractions.values())-1)>1e-8:
        raise ValueError("strata_quotas must be nonnegative fractions summing to one")
    quotas = {name:int(count*fraction) for name,fraction in fractions.items()}
    quotas[list(quotas)[-1]] += count-sum(quotas.values())
    selected = {name:[] for name in quotas}
    keys, pairs = set(), []
    digest = file_sha256(calibration)
    strict_lookup = options.get("strict_lookup",False) or config.get("publication_run",False)
    lookup_data = artifact.get("latency_models",{}).get("shape_lookup")
    lookup = ShapeLookupLatencyModel(lookup_data,artifact["calibration_envelope"],stride) if lookup_data else None
    declaration = policy_declaration(config)
    lookup_strata = {"shape_lookup_boundary", "shape_lookup_conservative_boundary",
                     "linear_tau_shape_lookup_conservative_disagreement"}
    if lookup_strata.intersection(quotas) and lookup is None:
        raise ValueError("Lookup-based strata require a shape_lookup calibration model")
    if strict_lookup and (lookup is None or not artifact.get("lookup_coverage",{}).get("complete")):
        raise ValueError("Strict validation requires complete shape lookup calibration")
    excluded_keys, prior_hashes = set(),[]
    for prior in options.get("prior_pair_specs",[]):
        document = json.loads(Path(prior).read_text(encoding="utf-8"))
        if canonical_hash(document["payload"]) != document["sha256"]:
            raise ValueError("Prior pair specs hash mismatch")
        if seed == document["payload"]["generator_seed"]:
            raise ValueError("Fresh validation requires a different generator seed")
        prior_hashes.append(document["sha256"])
        excluded_keys.update(tuple(p["computational_key"]) for p in document["payload"]["pairs"])
    diagnostics = dict(generation_attempts=0,rejection_reasons={},by_stratum={name:dict(requested_pairs=q,generated_pairs=0,generation_attempts=0,rejection_reasons={}) for name,q in quotas.items()})
    def reject(reason, stratum=None):
        counts = diagnostics["rejection_reasons"]
        counts[reason] = counts.get(reason,0)+1
        if stratum in diagnostics["by_stratum"]:
            counts = diagnostics["by_stratum"][stratum]["rejection_reasons"]
            counts[reason] = counts.get(reason,0)+1
    candidate_options = dict(options)
    if representative:
        candidate_options.update(tensor_grid_search=False,random_generation_attempts=options.get("max_generation_attempts",5000000))
    for rectangles in candidates(models,lookup,declaration,canvas,stride,candidate_options,quotas,selected,rng,seed):
        diagnostics["generation_attempts"] += 1
        first, second = rectangles
        union = union_rectangle(first,second)
        shapes = [stride_rounded_shape(r.height,r.width,stride) for r in (first,second,union)]
        if not all(models["linear"].is_in_domain(*hw) for hw in shapes):
            reject("out_of_domain")
            continue
        if strict_lookup and not all(lookup.is_in_domain(*hw) for hw in shapes):
            raise ValueError("Missing lookup entry for accepted shape")
        key = tuple(v for hw in shapes for v in hw)
        if key in keys or key in excluded_keys:
            reject("duplicate_or_prior_computational_key")
            continue
        primary,tags,gains,distance,boundary_scores = classify(shapes,models,options,lookup,declaration)
        if representative:
            primary = "reference"
        if primary in diagnostics["by_stratum"]:
            diagnostics["by_stratum"][primary]["generation_attempts"] += 1
        if primary not in quotas or len(selected[primary]) >= quotas[primary]:
            reject("quota_full_or_unrequested",primary)
            continue
        # Reserve half of each boundary quota for each sign.
        if primary.endswith("_boundary"):
            name = primary.removesuffix("_boundary")
            side = boundary_scores[name]>0
            limit = (quotas[primary]+1)//2 if side else quotas[primary]//2
            if sum((p["boundary_scores_ms"][name]>0) == side for p in selected[primary]) >= limit:
                reject("boundary_side_full",primary)
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
                    predicted_gains_ms=gains, boundary_scores_ms=boundary_scores, linear_signed_distance=distance,
                    boundary_side=("high" if boundary_scores[primary.removesuffix("_boundary")]>0 else "low") if primary.endswith("_boundary") else None,
                    threshold_side="merge" if distance<0 else "separate", calibration_hash=digest,
                    generator_seed=seed, schema_version=1)
        diagnostics["by_stratum"][primary]["generated_pairs"] += 1
        selected[primary].append(pair)
        pairs.append(pair)
        keys.add(key)
        if len(pairs) == count:
            break
    counts = {name:len(items) for name,items in selected.items()}
    if counts != quotas:
        sides = {name:{"high":sum(p["boundary_scores_ms"][name.removesuffix("_boundary")]>0 for p in items),
                       "low":sum(p["boundary_scores_ms"][name.removesuffix("_boundary")]<=0 for p in items)}
                 for name,items in selected.items() if name.endswith("_boundary")}
        raise ValueError(f"Unattainable strata quotas/boundary sides after random and tensor-grid search: obtained={counts}, required={quotas}, sides={sides}. Priority={PRIORITY}. No substitution performed.")
    for name,items in selected.items():
        if name.endswith("_boundary") and items and len({p["boundary_scores_ms"][name.removesuffix("_boundary")]>0 for p in items}) != 2:
            raise ValueError(f"Both sides of {name} are required")
    for pair in pairs:
        pair["bootstrap_stability"] = stability(artifact,pair["effective_areas"],config)
    distribution = dict(name="synthetic_reference_distribution",roi_dimensions="independent_uniform_integer_1_to_canvas",positions="uniform_feasible_top_left",conditioning="all_three_shapes_in_domain_and_unique_computational_key",trace=None)
    payload = dict(schema_version=2, evaluation_design=evaluation_design,
                   aggregate_scope="reference_distribution" if representative else "unweighted_challenge_average",
                   reference_distribution=distribution if representative else None,
                   validation_declaration=validation_declaration(config),used_for_policy_development=False,
                   sampling_diagnostics=diagnostics,excluded_pair_specs_hashes=prior_hashes,
                   generation_method="random_then_tensor_grid" if candidate_options.get("tensor_grid_search",True) else "random",
                   canvas_hw=canvas, stride=stride, calibration_hash=digest,
                   generator_seed=seed, priority=PRIORITY, quotas=quotas, config=options, decision_stability=config.get("decision_stability", {"stable_separate_max_probability":.1,"stable_merge_min_probability":.9}), pairs=pairs)
    write_json(target, dict(payload=payload, sha256=canonical_hash(payload)))
    return payload


def load(path, config, calibration, adapter, image):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = document["payload"]
    if document["sha256"] != canonical_hash(payload):
        raise ValueError("Pair specs hash mismatch")
    if payload["schema_version"] not in {1,2} or payload["stride"] != adapter.stride or payload["canvas_hw"] != config.get("experiment_b", {}).get("canvas_hw", [640,640]):
        raise ValueError("Pair specs schema/stride/canvas mismatch")
    if payload["calibration_hash"] != file_sha256(calibration):
        raise ValueError("Pair specs calibration hash mismatch")
    artifact = load_calibration(calibration)
    models = load_latency_models(artifact)
    strict_lookup = config.get("experiment_b",{}).get("strict_lookup",False) or config.get("publication_run",False)
    lookup_data = artifact.get("latency_models",{}).get("shape_lookup")
    lookup = ShapeLookupLatencyModel(lookup_data,artifact["calibration_envelope"],adapter.stride) if lookup_data else None
    declaration = policy_declaration(config)
    if strict_lookup and lookup is None:
        raise ValueError("Missing lookup table")
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
        if strict_lookup and not all(lookup.is_in_domain(*hw) for hw in shapes):
            raise ValueError("Missing lookup entry for validation shape")
        primary,tags,gains,distance,boundary_scores = classify(shapes,models,payload["config"],lookup,declaration)
        if payload.get("evaluation_design") == "representative":
            primary = "reference"
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
            sides = {p["specification"].get("boundary_scores_ms",p["specification"]["predicted_gains_ms"])[model_name]>0 for p in pairs if p["boundary_bin"] == name}
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
