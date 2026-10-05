"""Opt-in calibrated ROI merger; default calibrated policy remains linear_tau."""
import logging
from itertools import combinations

from experiments.cost_model_validation.artifacts import load_calibration
from experiments.cost_model_validation.common import file_sha256
from experiments.cost_model_validation.shape_model import MergePolicy, policy_declaration
from .merger_helper import bbox_union


class CalibratedRoiMerger:
    def __init__(self, calibration, policies=None):
        self.calibration_hash = file_sha256(calibration)
        self.policy = MergePolicy(load_calibration(calibration),policy_declaration({"policies":policies or {}}))
        self.decisions = []
        self.merge_count = 0

    def metadata(self):
        return dict(calibration_sha256=self.calibration_hash,policy_declaration=self.policy.declaration,
                    fallback_counts=dict(self.policy.fallback_counts))

    def __call__(self, rois, image_size=None, effective_shape=None, **_):
        self.decisions = []
        self.merge_count = 0
        clusters = list(rois)
        while len(clusters)>1:
            best = None
            for i,j in combinations(range(len(clusters)),2):
                union = bbox_union(clusters[i], clusters[j])
                if effective_shape is None:
                    result = self.policy.rectangles(clusters[i],clusters[j],strict=False)
                    shapes = None
                else:
                    shapes = [effective_shape(r) for r in (clusters[i], clusters[j], union)]
                    result = self.policy.predict(shapes, strict=False, exact_shapes=True)
                reason = None
                if result.get("fallback_used"):
                    lookup = self.policy.lookup
                    if shapes is not None and not all(self.policy.area_models["linear"].is_in_domain(*hw) for hw in shapes):
                        reason = "outside_envelope"
                    else:
                        reason = "missing_lookup_entry" if lookup is not None else "missing_lookup_table"
                elif self.policy.declaration["primary"] == "shape_lookup_conservative" and result.get("gain_lcb_s") is None:
                    reason = "missing_uncertainty"
                    if self.policy.declaration.get("fallback") != "separate":
                        raise ValueError("Conservative lookup requires uncertainty or separate fallback")
                    result["fallback_used"] = "separate"
                    self.policy.fallback_counts["separate"] += 1
                self.decisions.append(dict(pair=[i,j], shapes=shapes, reason=reason, **result))
                if result.get("fallback_used"):
                    logging.getLogger(__name__).debug("Calibration fallback %s; counts=%s",result["fallback_used"],dict(self.policy.fallback_counts))
                if result["predicted_merge"] and (best is None or result["predicted_gain_s"]>best[0]):
                    best = (result["predicted_gain_s"],i,j)
            if best is None:
                break
            _,i,j = best
            self.merge_count += 1
            union = bbox_union(clusters[i],clusters[j])
            clusters = [r for k,r in enumerate(clusters) if k not in (i,j)]+[union]
        return clusters
