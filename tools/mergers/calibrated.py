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

    def metadata(self):
        return dict(calibration_sha256=self.calibration_hash,policy_declaration=self.policy.declaration,
                    fallback_counts=dict(self.policy.fallback_counts))

    def __call__(self, rois, image_size=None, **_):
        clusters = list(rois)
        while len(clusters)>1:
            best = None
            for i,j in combinations(range(len(clusters)),2):
                result = self.policy.rectangles(clusters[i],clusters[j],strict=False)
                if result.get("fallback_used"):
                    logging.getLogger(__name__).warning("Calibration fallback %s; counts=%s",result["fallback_used"],dict(self.policy.fallback_counts))
                if result["predicted_merge"] and (best is None or result["predicted_gain_s"]>best[0]):
                    best = (result["predicted_gain_s"],i,j)
            if best is None:
                break
            _,i,j = best
            union = bbox_union(clusters[i],clusters[j])
            clusters = [r for k,r in enumerate(clusters) if k not in (i,j)]+[union]
        return clusters
