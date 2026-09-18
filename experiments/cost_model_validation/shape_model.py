"""Ordered shape lookup and predeclared constant-time merge policies."""
from collections import Counter
from dataclasses import dataclass
from math import sqrt, ceil
from statistics import NormalDist

import numpy as np

from .geometry import stride_rounded_shape, Rectangle, union_rectangle
from .models import CalibrationEnvelope, load_latency_models, decide_merge
from .reproducibility import statistics, canonical_hash


def calibration_grid(envelope, stride):
    envelope = CalibrationEnvelope(**envelope) if isinstance(envelope,dict) else envelope
    start_h,start_w = stride_rounded_shape(envelope.min_tensor_h,envelope.min_tensor_w,stride)
    return [(h,w) for h in range(start_h,envelope.max_tensor_h+1,stride)
            for w in range(start_w,envelope.max_tensor_w+1,stride) if envelope.contains(h,w)]


@dataclass(frozen=True)
class ShapeLatencyEstimate:
    point_s: float
    standard_error_s: float | None
    ci_s: tuple | None
    repetitions: int


class ShapeLookupLatencyModel:
    def __init__(self, data, envelope, stride=32):
        self.table = {tuple(map(int,key.split("x"))):ShapeLatencyEstimate(**value) for key,value in data["table"].items()}
        self.envelope = CalibrationEnvelope(**envelope)
        self.stride = stride
        for shape,estimate in self.table.items():
            if (not self.is_in_domain(*shape) or not np.isfinite(estimate.point_s) or estimate.point_s<=0
                    or estimate.repetitions<1 or (estimate.standard_error_s is not None and
                    (not np.isfinite(estimate.standard_error_s) or estimate.standard_error_s<0))):
                raise ValueError(f"Invalid lookup estimate for {shape}")

    def estimate(self, tensor_h, tensor_w):
        if not self.is_in_domain(tensor_h,tensor_w):
            raise ValueError(f"Missing lookup entry or out-of-domain shape: {(tensor_h,tensor_w)}")
        return self.table[(tensor_h,tensor_w)]

    def predict_seconds(self, tensor_h, tensor_w):
        return self.estimate(tensor_h,tensor_w).point_s

    def is_in_domain(self, tensor_h, tensor_w):
        return ((tensor_h,tensor_w) in self.table and tensor_h % self.stride == 0 and tensor_w % self.stride == 0
                and self.envelope.contains(tensor_h,tensor_w))


def build_lookup(groups, statistic, trim, count, seed, confidence=.95):
    rng = np.random.default_rng(seed)
    table = {}
    alpha = (1-confidence)/2
    for (h,w),values in sorted(groups.items()):
        values = np.asarray(values)
        samples = rng.choice(values,(count,len(values)),replace=True)
        if statistic == "trimmed_mean":
            k = int(len(values)*trim)
            draws = np.sort(samples,axis=1)[:,k:len(values)-k].mean(axis=1)
        elif statistic == "median":
            draws = np.median(samples,axis=1)
        elif statistic == "mean":
            draws = samples.mean(axis=1)
        else:
            raise ValueError("Unsupported lookup statistic")
        table[f"{h}x{w}"] = dict(point_s=statistics(values,trim)[statistic+"_s"],
                                 standard_error_s=float(draws.std(ddof=1)) if count>1 else None,
                                 ci_s=np.quantile(draws,[alpha,1-alpha]).tolist(), repetitions=len(values))
    return dict(table=table,statistic=statistic,trim_fraction_each_tail=trim,
                uncertainty_method="within_shape_statistic_bootstrap",confidence_level=confidence,count=count,seed=seed)


def policy_declaration(config):
    options = config.get("policies",{})
    confidence = float(options.get("confidence_level",.95))
    margin = float(options.get("decision_margin_s",0))
    if not .5 < confidence < 1 or margin < 0 or not np.isfinite(margin):
        raise ValueError("Invalid predeclared confidence or decision margin")
    primary = options.get("primary","linear_tau")
    allowed = {"linear_tau","shape_lookup","shape_lookup_conservative","conservative_consensus"}
    if primary not in allowed:
        raise ValueError("Primary policy must be explicitly supported; area nonlinear rules are offline baselines")
    return dict(primary=primary,candidates=["linear_tau","quadratic_direct_cost","piecewise_direct_cost",
                "shape_lookup","shape_lookup_conservative","conservative_consensus"],
                confidence_level=confidence,decision_margin_s=margin,
                uncertainty_assumption="worst_case_correlation_bound",
                exploratory_policies=["conservative_consensus"],fallback=options.get("fallback","error"),
                primary_metric="mean_latency_regret_ms",secondary_metric="p95_latency_regret_ms")


class MergePolicy:
    """No bootstrap fitting or iteration in the runtime path. Counters are auditable."""
    def __init__(self, artifact, declaration=None):
        self.area_models = load_latency_models(artifact)
        self.declaration = declaration or artifact.get("policy_declaration") or policy_declaration({})
        self.stride = artifact.get("shape_policy",{}).get("stride",artifact.get("provenance",{}).get("model_stride",32))
        data = artifact.get("latency_models",{}).get("shape_lookup")
        self.lookup = ShapeLookupLatencyModel(data,artifact["calibration_envelope"],self.stride) if data else None
        self.fallback_counts = Counter()
        self.z = NormalDist().inv_cdf(self.declaration["confidence_level"])

    def predict(self, shapes, name=None, strict=True):
        name = name or self.declaration["primary"]
        shapes = [stride_rounded_shape(*hw,self.stride) for hw in shapes]
        margin = self.declaration["decision_margin_s"]
        if name in {"shape_lookup","shape_lookup_conservative"}:
            if self.lookup is None or not all(self.lookup.is_in_domain(*hw) for hw in shapes):
                if strict or self.declaration.get("fallback","error") == "error":
                    raise ValueError("Missing lookup entry; strict validation forbids fallback")
                fallback = self.declaration.get("fallback")
                self.fallback_counts[fallback] += 1
                if fallback == "conservative_area" and all(m.is_in_domain(*hw) for m in self.area_models.values() for hw in shapes):
                    result = self.predict(shapes,"conservative_consensus",strict=False)
                elif fallback in {"separate","conservative_area"}:
                    result = dict(predicted_merge=False,predicted_gain_s=None)
                else:
                    raise ValueError("Unknown fallback")
                return {**result,"fallback_used":fallback}
            estimates = [self.lookup.estimate(*hw) for hw in shapes]
            separate = estimates[0].point_s + estimates[1].point_s
            merged = estimates[2].point_s
            gain = separate-merged
            # Sum of marginal SEs bounds the SD for arbitrary correlations; it
            # also handles repeated shape keys without an independence claim.
            ses = [e.standard_error_s for e in estimates]
            se = sum(ses) if all(v is not None for v in ses) else None
            lcb = gain-self.z*se if se is not None else None
            decision_value = gain if name == "shape_lookup" else lcb
            return dict(predicted_merge=decision_value is not None and decision_value>margin,
                        predicted_gain_s=gain,predicted_merged_cost_s=merged,predicted_separate_cost_s=separate,
                        gain_se_bound_s=se,gain_lcb_s=lcb,fallback_used=None)
        if name == "conservative_consensus":
            decisions = [decide_merge(model,*shapes) for model in self.area_models.values()]
            return dict(predicted_merge=len(decisions)==3 and all(d["predicted_gain_s"]>margin for d in decisions),
                        predicted_gain_s=min(d["predicted_gain_s"] for d in decisions),exploratory=True)
        names = {"linear_tau":"linear","quadratic_direct_cost":"quadratic","piecewise_direct_cost":"piecewise"}
        model = self.area_models[names[name]]
        if not all(model.is_in_domain(*hw) for hw in shapes):
            raise ValueError("Merge shape outside calibration domain")
        return decide_merge(model,*shapes)

    def rectangles(self, first, second, name=None, strict=True):
        first,second = Rectangle(*first),Rectangle(*second)
        union = union_rectangle(first,second)
        shapes = [stride_rounded_shape(ceil(r.height),ceil(r.width),self.stride) for r in (first,second,union)]
        return self.predict(shapes,name,strict)


def declaration_hash(config):
    return canonical_hash(policy_declaration(config))
