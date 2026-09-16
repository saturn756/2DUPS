"""I4 MatchResult（边 E12、E13，均为可选）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import Producer, Ref, Status


@dataclass
class FramePair:
    frame_a: str
    frame_b: str
    variant_id: str


@dataclass
class Transform:
    type: str                                  # homography / fundamental / ...
    params: list[float] = field(default_factory=list)
    inlier_ratio: float | None = None
    residual: float | None = None


@dataclass
class MatchResult:
    """status 非 ok 时 transform 不得作为几何证据；结果只能用于 frame_b 及其之后的帧。"""

    INTERFACE = "I4"

    pair: FramePair
    status: Status
    producer: Producer
    keypoints_ref: Ref | None = None
    matches_ref: Ref | None = None
    inlier_mask_ref: Ref | None = None
    transform: Transform | None = None
