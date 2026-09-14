"""Inference-time cost model validation experiments."""

from .geometry import Rectangle, effective_area, union_rectangle
from .models import fit_linear_model, merge_decision

__all__ = ["Rectangle", "effective_area", "union_rectangle", "fit_linear_model", "merge_decision"]