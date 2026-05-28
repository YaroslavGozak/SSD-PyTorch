"""
RoiSSDPenalized
~~~~~~~~~~~~~~~
Thin wrapper around a RoiSSD model that subtracts a fixed confidence
penalty from every detection score returned at inference time.

Example
-------
    model = RoiSSD(...)
    wrapped = RoiSSDPenalized(model, penalty=0.05)
    # A raw score of 1.0 becomes 0.95, 0.50 becomes 0.45, etc.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class RoiSSDPenalized(nn.Module):
    """Wraps a RoiSSD model and decreases every detection confidence by *penalty*.

    The penalty is applied after inference so that training behaviour and the
    model's internal thresholding are unaffected.  Scores are clamped to
    ``[0, 1]`` after the subtraction to avoid negative values.

    Parameters
    ----------
    model:
        A RoiSSD (or compatible) model whose ``forward`` returns
        ``(losses, detections)`` where ``detections`` is a list of dicts
        with keys ``"boxes"``, ``"scores"``, ``"labels"``.
    penalty:
        Amount to subtract from each detection score.  Defaults to ``0.05``.
    """

    def __init__(self, model: nn.Module, penalty: float = 0.05) -> None:
        super().__init__()
        if not (0.0 <= penalty <= 1.0):
            raise ValueError(f"penalty must be in [0, 1], got {penalty}")
        self.model = model
        self.penalty = float(penalty)

    # ------------------------------------------------------------------
    # Delegate attribute access so callers can use e.g. wrapped.idx2label
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # nn.Module stores its own attributes via __setattr__; fall back to
        # the wrapped model for anything not found on this object.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    # ------------------------------------------------------------------
    # Pass-through helpers expected by the training/inference pipeline
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "RoiSSDPenalized":
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self) -> "RoiSSDPenalized":
        return self.train(False)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Accept both flat keys (bare RoiSSD checkpoint) and prefixed keys."""
        first_key = next(iter(state_dict), "")
        if first_key and not first_key.startswith("model."):
            state_dict = {"model." + k: v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[Any] = None,
        ignore_regions: Optional[Any] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        losses, detections = self.model(x, targets=targets, ignore_regions=ignore_regions)

        if not self.training:
            penalized: List[Dict[str, torch.Tensor]] = []
            for det in detections:
                scores = det["scores"] - self.penalty
                scores = scores.clamp(min=0.0, max=1.0)
                penalized.append(
                    {
                        "boxes": det["boxes"],
                        "scores": scores,
                        "labels": det["labels"],
                    }
                )
            return losses, penalized

        return losses, detections
