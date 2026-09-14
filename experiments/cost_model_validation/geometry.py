"""Geometry and effective tensor-area helpers."""

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class Rectangle:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


def as_rectangle(value: Iterable[int]) -> Rectangle:
    x1, y1, x2, y2 = (int(v) for v in value)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid rectangle coordinates: {value!r}")
    return Rectangle(x1, y1, x2, y2)


def union_rectangle(first: Rectangle, second: Rectangle) -> Rectangle:
    return Rectangle(
        min(first.x1, second.x1), min(first.y1, second.y1),
        max(first.x2, second.x2), max(first.y2, second.y2),
    )


def effective_area(tensor_hw: Tuple[int, int]) -> int:
    height, width = (int(v) for v in tensor_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"Tensor shape must be positive, got {tensor_hw!r}")
    return height * width


def stride_rounded_shape(height: int, width: int, stride: int) -> Tuple[int, int]:
    if min(height, width, stride) <= 0:
        raise ValueError("height, width, and stride must be positive")
    return ((height + stride - 1) // stride * stride,
            (width + stride - 1) // stride * stride)