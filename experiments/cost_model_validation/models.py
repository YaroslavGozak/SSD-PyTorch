"""Statistical models for latency as a function of effective tensor area."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence, Tuple

import numpy as np


class LatencyModel(Protocol):
    name: str

    def predict_seconds(self, effective_area: int) -> float:
        ...

    def is_in_domain(self, tensor_h: int, tensor_w: int) -> bool:
        ...

    def metadata(self) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class CalibrationEnvelope:
    min_effective_area: float
    max_effective_area: float
    min_tensor_h: int
    max_tensor_h: int
    min_tensor_w: int
    max_tensor_w: int
    min_aspect_ratio: float
    max_aspect_ratio: float

    def contains(self, tensor_h: int, tensor_w: int) -> bool:
        area = int(tensor_h) * int(tensor_w)
        aspect = float(tensor_w) / float(tensor_h)
        return (self.min_effective_area <= area <= self.max_effective_area and
                self.min_tensor_h <= tensor_h <= self.max_tensor_h and
                self.min_tensor_w <= tensor_w <= self.max_tensor_w and
                self.min_aspect_ratio <= aspect <= self.max_aspect_ratio)


@dataclass(frozen=True)
class PolynomialLatencyModel:
    name: str
    coefficients: Dict[str, float]
    envelope: CalibrationEnvelope
    breakpoint_area: Optional[float] = None

    def predict_seconds(self, effective_area: int) -> float:
        area = float(effective_area)
        if not np.isfinite(area) or area <= 0:
            raise ValueError("effective_area must be a positive finite value")
        b0 = self.coefficients["b0"]
        b1 = self.coefficients["b1"]
        if self.name == "linear":
            value = b0 + b1 * area
        elif self.name == "quadratic":
            value = b0 + b1 * area + self.coefficients["b2"] * area ** 2
        elif self.name == "piecewise":
            value = b0 + b1 * area + self.coefficients["b2"] * max(0.0, area - float(self.breakpoint_area))
        else:
            raise ValueError(f"Unknown latency model: {self.name}")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Model {self.name} produced invalid latency {value}")
        return float(value)

    def is_in_domain(self, tensor_h: int, tensor_w: int) -> bool:
        return self.envelope.contains(int(tensor_h), int(tensor_w))

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "coefficients": dict(self.coefficients), "breakpoint_area": self.breakpoint_area,
                "calibration_envelope": self.envelope.__dict__.copy()}


def decide_merge(model: LatencyModel, shape_1: Tuple[int, int], shape_2: Tuple[int, int], merged_shape: Tuple[int, int]) -> Dict[str, Any]:
    areas = {"a1": int(shape_1[0]) * int(shape_1[1]), "a2": int(shape_2[0]) * int(shape_2[1]), "merged": int(merged_shape[0]) * int(merged_shape[1])}
    merged_cost = model.predict_seconds(areas["merged"])
    separate_cost = model.predict_seconds(areas["a1"]) + model.predict_seconds(areas["a2"])
    return {"predicted_merge": merged_cost < separate_cost,
            "predicted_merged_cost_s": merged_cost, "predicted_separate_cost_s": separate_cost,
            "predicted_gain_s": separate_cost - merged_cost, "effective_areas": areas}


def load_latency_models(artifact: Dict[str, Any]) -> Dict[str, PolynomialLatencyModel]:
    envelope_data = artifact.get("calibration_envelope")
    if not envelope_data:
        raise ValueError("Calibration artifact has no calibration_envelope")
    envelope = CalibrationEnvelope(**envelope_data)
    source = artifact.get("latency_models", {})
    if "linear" not in source and artifact.get("K_t_s") is not None:
        source["linear"] = {"coefficients": {"b0": artifact["K_t_s"], "b1": artifact["c_t_s_per_pixel"]}}
    models = {}
    for name in ("linear", "quadratic", "piecewise"):
        data = source.get(name)
        if not data:
            continue
        coefficients = data.get("coefficients", {})
        if name == "linear" and not coefficients:
            coefficients = {"b0": artifact.get("K_t_s"), "b1": artifact.get("c_t_s_per_pixel")}
        if any(key not in coefficients or not np.isfinite(float(coefficients[key])) for key in ("b0", "b1")):
            continue
        if name in {"quadratic", "piecewise"} and "b2" not in coefficients:
            continue
        breakpoint_area = data.get("breakpoint_area")
        if name == "piecewise" and breakpoint_area is None:
            continue
        models[name] = PolynomialLatencyModel(name, {key: float(value) for key, value in coefficients.items()}, envelope, breakpoint_area)
    if "linear" not in models:
        raise ValueError("Calibration artifact does not contain a valid linear model")
    return models


@dataclass
class FitResult:
    name: str
    coefficients: Dict[str, float]
    predictions: np.ndarray
    r2: float
    adjusted_r2: float
    rmse: float
    mae: float
    aic: float
    bic: float
    breakpoint: Optional[float] = None


def control_predictions(models, effective_area: int, measured_ms: float):
    """Evaluate every available A model against the same measured control input."""
    predictions = {}
    for name in ("linear", "quadratic", "piecewise"):
        if name not in models:
            predictions[name] = {"available": False, "predicted_ms": None,
                                 "relative_error": None, "reason": "Missing calibration coefficients"}
            continue
        predicted_ms = models[name].predict_seconds(effective_area) * 1000.0
        predictions[name] = {"available": True, "predicted_ms": predicted_ms,
                             "relative_error": (measured_ms - predicted_ms) / predicted_ms}
    return predictions


def _metrics(name: str, y: np.ndarray, prediction: np.ndarray, coefficients: Dict[str, float], parameter_count: int, breakpoint: Optional[float] = None) -> FitResult:
    residual = y - prediction
    rss = float(np.dot(residual, residual))
    n = len(y)
    r2 = 1.0 - rss / max(float(np.dot(y - y.mean(), y - y.mean())), np.finfo(float).eps)
    adjusted = 1.0 - (1.0 - r2) * (n - 1) / max(n - parameter_count, 1)
    scale = max(rss / max(n, 1), np.finfo(float).eps)
    return FitResult(name, coefficients, prediction, r2, adjusted,
                     float(np.sqrt(np.mean(residual ** 2))),
                     float(np.mean(np.abs(residual))),
                     float(n * np.log(scale) + 2 * parameter_count),
                     float(n * np.log(scale) + parameter_count * np.log(max(n, 1))),
                     breakpoint)


def fit_linear_model(area: Sequence[float], time_s: Sequence[float]) -> FitResult:
    x, y = _arrays(area, time_s)
    intercept, slope = np.polyfit(x, y, 1)[::-1]
    prediction = intercept + slope * x
    return _metrics("linear", y, prediction, {"b0": float(intercept), "b1": float(slope)}, 2)


def fit_quadratic_model(area: Sequence[float], time_s: Sequence[float]) -> FitResult:
    x, y = _arrays(area, time_s)
    b2, b1, b0 = np.polyfit(x, y, 2)
    prediction = b0 + b1 * x + b2 * x ** 2
    return _metrics("quadratic", y, prediction,
                    {"b0": float(b0), "b1": float(b1), "b2": float(b2)}, 3)


def fit_piecewise_model(area: Sequence[float], time_s: Sequence[float], breakpoint: Optional[float] = None) -> FitResult:
    x, y = _arrays(area, time_s)
    unique = np.unique(x)
    candidates = [breakpoint] if breakpoint is not None else [v for v in unique if np.quantile(x, .1) <= v <= np.quantile(x, .9)]
    if not candidates:
        raise ValueError("At least three unique areas are required for a breakpoint fit")
    best = None
    for candidate in candidates:
        design = np.column_stack((np.ones_like(x), x, np.maximum(0.0, x - candidate)))
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ coefficients
        rss = float(np.dot(y - prediction, y - prediction))
        if best is None or rss < best[0]:
            best = (rss, float(candidate), coefficients, prediction)
    _, chosen, coefficients, prediction = best
    return _metrics("piecewise", y, prediction,
                    {"b0": float(coefficients[0]), "b1": float(coefficients[1]), "b2": float(coefficients[2])}, 4, chosen)


def fit_all(area: Sequence[float], time_s: Sequence[float]) -> Dict[str, FitResult]:
    return {"linear": fit_linear_model(area, time_s),
            "quadratic": fit_quadratic_model(area, time_s),
            "piecewise": fit_piecewise_model(area, time_s)}


def merge_decision(delta_effective_area: float, tau: float) -> bool:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return float(delta_effective_area) < float(tau)


def _arrays(area: Iterable[float], time_s: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(list(area), dtype=np.float64)
    y = np.asarray(list(time_s), dtype=np.float64)
    if x.size != y.size or x.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("area and time_s must be equal-length finite arrays with at least two observations")
    if np.unique(x).size < 2:
        raise ValueError("at least two unique areas are required")
    return x, y
