"""Timing boundaries shared by both collection experiments."""

import time
from dataclasses import dataclass
from typing import Any, Optional

from .adapters import DetectorAdapter, PreparedInput


@dataclass
class TimingResult:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    detector_call_ms: float
    raw_output: Any
    processed_output: Any


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000.0


def measure(adapter: DetectorAdapter, image: Any, requested_hw: tuple[int, int], mode: str) -> tuple[PreparedInput, TimingResult]:
    if mode not in {"inference_only", "detector_call"}:
        raise ValueError(f"Unknown timing mode: {mode!r}")
    prep_start = time.perf_counter_ns()
    prepared = adapter.prepare(image, requested_hw)
    tensor = prepared.tensor
    actual_h, actual_w = int(tensor.shape[-2]), int(tensor.shape[-1])
    if (prepared.tensor_h, prepared.tensor_w) != (actual_h, actual_w):
        raise AssertionError("PreparedInput dimensions do not match the actual model tensor")
    if prepared.effective_area != actual_h * actual_w:
        raise AssertionError("PreparedInput effective area does not match the actual model tensor")
    if adapter.stride > 1 and (actual_h % adapter.stride or actual_w % adapter.stride):
        raise AssertionError("Prepared tensor is not stride aligned")
    prep_end = time.perf_counter_ns()
    infer_start = time.perf_counter_ns()
    raw = adapter.infer(prepared)
    infer_end = time.perf_counter_ns()
    post_start = time.perf_counter_ns()
    processed = adapter.postprocess(raw)
    post_end = time.perf_counter_ns()
    prep_ms = _elapsed_ms(prep_start, prep_end)
    infer_ms = _elapsed_ms(infer_start, infer_end)
    post_ms = _elapsed_ms(post_start, post_end)
    return prepared, TimingResult(
        preprocess_ms=prep_ms,
        inference_ms=infer_ms,
        postprocess_ms=post_ms,
        detector_call_ms=prep_ms + infer_ms + post_ms,
        raw_output=raw,
        processed_output=processed,
    )