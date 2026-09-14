"""Statistical models for latency as a function of effective tensor area."""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


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